"""Read-only facade a routing strategy may bind to see the live simulator.

`SchedulingRuntime` holds a reference to the simulator and reads its current state on every call. It
never caches, mutates, or decides anything. A strategy that wants one declares `bind_runtime` and the
simulator hands it in once, right after the router is built, following the same capability-probe
pattern as `observe_completion`/`observe_train_cycle`. No strategy implements it yet.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from psrl.workers.ps.staleness import InventorySnapshot

from ..component.rollout.engine import NativeInstance, Request, TrajView
from ..forecast import InstanceState, TrainState

if TYPE_CHECKING:
    from .simulator import Simulator


class SchedulingRuntime:
    """Thin, read-only view of one `Simulator`, for a strategy that forecasts the pipeline."""

    def __init__(self, sim: Simulator) -> None:
        self._sim = sim

    # --- scalars -----------------------------------------------------

    @property
    def now(self) -> float:
        return self._sim.t

    @property
    def train_version(self) -> int:
        return self._sim.train_version

    @property
    def rollout_version(self) -> int:
        return self._sim.rollout_version

    @property
    def horizon(self) -> int | None:
        return self._sim.cfg.staleness_horizon

    @property
    def ingress_window_groups(self) -> int:
        return self._sim._ingress_group_credits

    @property
    def batch_size(self) -> int:
        return self._sim.cfg.batch_size

    @property
    def group_size(self) -> int:
        return self._sim.cfg.group_size

    @property
    def weight_load(self) -> float:
        return self._sim.cfg.weight_load_time

    @property
    def token_budget(self) -> int:
        return self._sim.cfg.token_budget

    # --- inventory and training ---------------------------------------

    def inventory(self) -> InventorySnapshot:
        """
        Snapshot the staleness inventory.

        Raises:
            TypeError: If the simulator's policy is not psrl, which is the only one that implements
                `inventory_snapshot`.
        """
        policy = self._sim.policy
        get_snapshot = getattr(policy, "inventory_snapshot", None)
        if not callable(get_snapshot):
            raise TypeError(f"SchedulingRuntime.inventory requires policy=psrl, got {policy.name!r}.")
        return get_snapshot(self._sim._state())

    def cycle_time(self, sequence_lengths: Sequence[int]) -> float:
        """Price train plus sync plus recompute for a training step over `sequence_lengths`."""
        estimate, _, sync_time, recompute_time = self._sim._train_cycle(tuple(sequence_lengths))
        return estimate.latency_s + sync_time + recompute_time

    def mean_cycle(self, predicted_group_lengths: Sequence[float]) -> float:
        """
        Price one training cycle for `batch_size * group_size` sequences at the mean predicted length.

        Args:
            predicted_group_lengths (Sequence[float]): One predicted training sequence length per
                upcoming group, already capped the way `Simulator._training_sequence_length` caps a
                real request. The mean replaces every sequence in the priced step.

        Raises:
            ValueError: If `predicted_group_lengths` is empty.
        """
        if not predicted_group_lengths:
            raise ValueError("predicted_group_lengths must not be empty.")
        mean_len = sum(predicted_group_lengths) / len(predicted_group_lengths)
        sequence_length = max(1, round(mean_len))
        n = self._sim.cfg.batch_size * self._sim.cfg.group_size
        return self.cycle_time((sequence_length,) * n)

    def train_state(self, sequence_lengths_hint: Sequence[int]) -> TrainState:
        """Build the `TrainState` for the next buffer, priced on `sequence_lengths_hint`."""
        return TrainState(
            training_end=self._sim._training_end,
            now=self._sim.t,
            cycle=self.cycle_time(sequence_lengths_hint),
            weight_load=self.weight_load,
        )

    # --- fleet and queue views -----------------------------------------

    @property
    def instances(self) -> list[NativeInstance]:
        return list(self._sim.instances)

    @property
    def inflight(self) -> list[Request]:
        return list(self._sim.inflight)

    @property
    def staged_groups(self) -> dict[int, list[Request]]:
        return dict(self._sim._staged_groups)

    @property
    def admitted_group_ids(self) -> frozenset[int]:
        return frozenset(self._sim._admitted_group_ids)

    @property
    def pending(self) -> dict[int, dict[int, float]]:
        """Finished-but-incomplete groups, each member's request id to its completion time."""
        return {
            group_id: {req.rid: req.complete_time for req in members}
            for group_id, members in self._sim.pending.items()
        }

    def queued(self) -> dict[int, list[Request]]:
        """Return every request still waiting for placement, grouped by group id."""
        queue = self._sim.router.queue
        out: dict[int, list[Request]] = {}
        for version in queue.versions():
            for req in queue.candidates_version(version, 1 << 30):
                out.setdefault(req.group_id, []).append(req)
        return out

    def cost_model(self, tp: int):
        """Return the rollout cost model for TP degree `tp`.

        TP invariant until a shape replanner adds a per-TP map. The same object prices every
        instance today, on both the native and C++ backends.
        """
        return self._sim.rollout_cost

    def instance_state(
        self,
        inst: NativeInstance,
        remaining_fn: Callable[[TrajView], float],
    ) -> InstanceState:
        """Project `inst` into the forecast's read-only `InstanceState`."""
        return InstanceState.from_instance(
            inst,
            cost=self._sim.rollout_cost,
            token_budget=self._sim.cfg.token_budget,
            max_concurrency=self._sim.cfg.max_concurrency,
            remaining=remaining_fn,
            # Neither backend configures a chunk cap from `SimConfig` today.
            chunk_cap=None,
        )
