"""Value objects describing engine instance state.

`StepStat` accumulates what happened during engine steps (telemetry), while
`InstanceLoad` and its `InstanceView` extension are what a router sees when it
picks a target instance. `TrajView` is the read-only projection of one request that
strategies and length predictors receive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .request import Request


@dataclass
class StepStat:
    """Aggregated engine steps.

    `dt`, `phase`, and `t` describe wall-clock cost so telemetry (the
    `rollout_step` log stream) can report per-step timing, not just token
    counts. `phase` is `"mixed"` for one engine step, which decodes the running
    batch and prefills whatever the token budget still allows, and `"decode"`
    for a closed-form segment covering many steps of a batch with no prefill in
    it. The names match `RolloutWorkload.phase`, which the cost models key
    their fitted coefficients on. `t` is the simulation time at which the step
    ends (commits), matching `NativeInstance.t_local` after the step.

    `n_prefill` is the work this step charged. `n_prefill_hit` is what the
    prefix cache absorbed and `n_prefill_computed` is what was really computed,
    named to match the real engine's own columns (`prefill_tokens` and
    `computed (cache miss)` in `outputs/simulate/SMG_LOOPBACK_FIX.md`) so
    `srl-compare` puts like against like.
    """

    n_tokens: int
    n_decode: int
    n_prefill: int
    saturated: bool
    steps: int = 1
    saturated_steps: int = 0
    dt: float = 0.0
    phase: str = "decode"
    t: float = 0.0
    n_prefill_hit: int = 0
    n_prefill_computed: int = 0

    def __post_init__(self) -> None:
        if self.saturated and self.saturated_steps == 0:
            self.saturated_steps = self.steps


@dataclass
class InstanceLoad:
    """Snapshot of instance load for routing decisions."""

    instance_id: int
    n_waiting: int
    n_running: int
    kv_utilization: float
    total_ctx: int
    tp: int = 1
    """Tensor-parallel degree of this instance.

    What a TP-aware routing strategy ranks on. Carried on the load snapshot rather
    than looked up from a plan so a strategy stays a pure function of what it is
    shown, and so instance, not group, is the unit of placement.
    """

    @property
    def n_active(self) -> int:
        """Requests the instance holds, queued or decoding."""
        return self.n_waiting + self.n_running


@dataclass
class InstanceView(InstanceLoad):
    """What a routing strategy is allowed to see about one instance.

    Extends `InstanceLoad` with the two things a cache-aware or cost-based
    strategy needs and a pure load-balancer does not. Strategies read this
    rather than `NativeInstance` so that a strategy cannot mutate the engine,
    and so that a strategy can be unit tested without building one.

    Attributes:
        free_blocks: KV blocks currently unallocated.
        committed_blocks: Blocks the resident requests will need for the
            context they already hold, including ones still queued and so
            holding nothing yet. The same quantity `add_request` reserves on,
            so a strategy and the admission gate agree about what full means.
        prefix_hit_tokens: Tokens a prefill of a given request would skip
            because the instance's cache already holds them. Non-mutating, so
            asking does not move the counters compared against the real
            engine's `vllm:prefix_cache_queries`.
        step_time_delta: Predicted increase in step latency from adding a
            given request, in seconds.
        residents: Read-only projections of the waiting and running requests,
            for a strategy that prices what admitting would slow down.
            Projections rather than requests, so the capability boundary above
            still holds and both engine backends answer identically.
        cost_model: The instance's rollout cost model, for pricing a
            hypothetical batch. Read-only: a strategy calls `step_time`.
        kv_blocks: Total KV blocks this instance owns, the denominator a
            strategy sizes a projected footprint against.
        block_size: Tokens per KV block.
    """

    free_blocks: int = 0
    committed_blocks: int = 0
    prefix_hit_tokens: Callable[["Request"], int] = lambda req: 0
    step_time_delta: Callable[["Request"], float] = lambda req: 0.0
    residents: Callable[[], list] = list
    cost_model: object | None = None
    kv_blocks: int = 0
    block_size: int = 16


@dataclass(frozen=True, slots=True)
class TrajView:
    """A read-only projection of one trajectory.

    Strategies and length predictors receive this rather than a `Request` so they
    cannot mutate the engine, and so the same code runs against either backend.
    `target_len` is oracle information that an observable predictor must not read.
    """

    generated: int
    prompt_len: int
    target_len: int
    prompt_key: int
    v_traj: int
    admit_time: float
    rid: int = -1
    group_id: int = 0
    prefilled: int = 0
    """Tokens whose KV the instance has already computed, 0 after an abort or preemption."""
    running: bool = False
    """Whether the request is decoding, as opposed to queued or still prefilling."""

    @property
    def remaining(self) -> int:
        return max(0, self.target_len - self.generated)

    @property
    def ctx(self) -> int:
        """Context this trajectory occupies, the depth a decode step pays for."""
        return self.prompt_len + self.generated


def traj_view(req) -> TrajView:
    """Project a `Request` or its C++ equivalent, which share these field names."""
    return TrajView(
        generated=req.generated,
        prompt_len=req.prompt_len,
        target_len=req.target_len,
        prompt_key=req.source_group_id,
        v_traj=req.v_traj,
        admit_time=req.admit_time,
        rid=req.rid,
        group_id=req.group_id,
        prefilled=req.prefilled,
        running=req.status.name == "RUNNING",
    )
