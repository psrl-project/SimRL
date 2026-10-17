"""Value types describing one rollout instance to the drain forecast.

An `InstanceState` is everything `InstanceForecast` may know about an engine instance, and it is
built from read-only projections so the forecast cannot mutate the engine, whichever backend
holds it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..component.rollout.engine import blocks_for, traj_view

if TYPE_CHECKING:
    from ..component.rollout.engine import TrajView
    from ..cost.rollout.base import RolloutCostModel


@dataclass(frozen=True, slots=True)
class Resident:
    """One request already placed on the instance."""

    key: int
    group_id: int
    prompt_key: int
    """Identity of the prompt, shared by the siblings that reuse its KV blocks."""
    prompt_len: int
    ctx: int
    """Prompt tokens plus tokens generated so far."""
    prefilled: int
    """Tokens whose KV is computed. Equals `ctx` for a running request."""
    remaining: int
    """Predicted decode tokens still to generate, at least 1."""
    running: bool
    hit: int = 0
    """Tokens the prefix cache would cover if this request were admitted now, 0 once it has started."""

    @classmethod
    def from_traj(cls, traj: TrajView, remaining: float, hit: int = 0) -> Resident:
        """Project a `TrajView`, rounding the predicted remaining length to whole tokens."""
        return cls(
            key=traj.rid,
            group_id=traj.group_id,
            prompt_key=traj.prompt_key,
            prompt_len=traj.prompt_len,
            ctx=traj.ctx,
            prefilled=traj.prefilled,
            remaining=max(1, round(remaining)),
            running=traj.running,
            hit=hit,
        )


@dataclass(frozen=True, slots=True)
class Arrival:
    """A hypothetical request the instance receives at `arrive_time`."""

    key: int
    arrive_time: float
    prompt_len: int
    generated: int
    remaining: int
    """Predicted decode tokens still to generate, at least 1."""
    group_id: int = 0
    prompt_key: int | None = None
    """Prompt identity, set to hit the cached prompt blocks of residents that share it."""

    def __post_init__(self) -> None:
        if self.remaining < 1:
            raise ValueError(f"An arrival must have at least one token left to generate, got {self.remaining!r}.")

    @property
    def ctx(self) -> int:
        return self.prompt_len + self.generated


@dataclass(frozen=True, slots=True)
class Interrupt:
    """A scheduled weight publish: every resident is aborted and the instance idles until resume.

    Mirrors `Simulator._publish_weights` under `partial_rollout`: at `time` every running and
    waiting request, and any arrival already admitted, is aborted in place (tokens kept, KV and the
    prefix cache dropped). The instance then does nothing until `time + resume_delay`, when the
    aborted set is queued again with `prefilled` reset to 0.
    """

    time: float
    resume_delay: float

    def __post_init__(self) -> None:
        if self.resume_delay < 0:
            raise ValueError(f"resume_delay must be non-negative, got {self.resume_delay!r}.")


@dataclass(frozen=True, slots=True)
class InstanceState:
    """An engine instance at one moment: its limits and the requests it holds."""

    cost: RolloutCostModel
    block_size: int
    kv_blocks: int
    token_budget: int
    max_concurrency: int | None
    chunk_cap: int | None
    tp: int
    now: float
    waiting: tuple[Resident, ...]
    """The queue in the order the engine serves it."""
    running: tuple[Resident, ...]

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.kv_blocks <= 0 or self.token_budget <= 0:
            raise ValueError(
                f"block_size, kv_blocks and token_budget must be positive, got "
                f"{self.block_size!r}, {self.kv_blocks!r}, {self.token_budget!r}."
            )
        if len(self.running) > self.token_budget:
            raise ValueError(
                f"{len(self.running)} running requests exceed the token budget {self.token_budget!r}, "
                f"which leaves some of them undecoded and is not modelled."
            )
        if any(not r.running for r in self.running) or any(r.running for r in self.waiting):
            raise ValueError("`running` and `waiting` must hold requests of the matching status.")
        if any(r.remaining < 1 for r in (*self.running, *self.waiting)):
            raise ValueError("Every resident must have at least one token left to generate.")
        if any(r.prefilled >= r.ctx for r in self.waiting):
            raise ValueError("A waiting resident must still have prompt or context tokens to prefill.")

    @property
    def used_blocks(self) -> int:
        """Blocks the residents hold, counting a prompt shared by siblings once per sibling."""
        return sum(blocks_for(r.ctx, self.block_size) for r in self.running) + sum(
            blocks_for(r.prefilled, self.block_size) for r in self.waiting
        )

    @classmethod
    def from_instance(
        cls,
        inst,
        *,
        cost: RolloutCostModel,
        token_budget: int,
        max_concurrency: int | None,
        remaining: Callable[[TrajView], float],
        chunk_cap: int | None = None,
    ) -> InstanceState:
        """
        Read a live `NativeInstance` or `CppInstance`.

        Args:
            inst: The engine instance. Only its public read surface is used.
            cost (RolloutCostModel): The Python model the instance prices steps with. The C++
                instance holds a wrapped copy it cannot hand back.
            token_budget (int): The instance's per-iteration token budget.
            max_concurrency (int | None): The instance's batch cap, None when unlimited.
            remaining (Callable[[TrajView], float]): Predicted decode tokens left for a request.
            chunk_cap (int | None): The instance's prefill chunk cap, None when unset.
        """
        waiting = []
        for req in inst.waiting:
            view = traj_view(req)
            hit = inst.peek_hit_tokens(req) if view.prefilled == 0 else 0
            waiting.append(Resident.from_traj(view, remaining(view), hit=hit))
        running = [Resident.from_traj(v, remaining(v)) for v in map(traj_view, inst.running)]
        return cls(
            cost=cost,
            block_size=inst.block_size,
            kv_blocks=inst.kv_blocks,
            token_budget=token_budget,
            max_concurrency=max_concurrency,
            chunk_cap=chunk_cap,
            tp=inst.tp,
            now=inst.t_local,
            waiting=tuple(waiting),
            running=tuple(running),
        )
