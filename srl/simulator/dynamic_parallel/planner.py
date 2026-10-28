"""Deciding the target awake set from the fleet's current occupancy.

Kept narrow on purpose, mirroring the real psrl `CohortPlanner.plan(table, state) -> awake` seam
(see `memory/psrl-dynamic-parallel-mechanism-and-costs.md`): a planner reads a `SlotTable` and a
`ReshapeState` snapshot and returns either `None` (no change) or a new GPU-disjoint awake set. It
never touches instances, the router, or telemetry -- that is `SlotFleet.reshape`'s job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .plan import Slot, SlotTable


@dataclass(frozen=True)
class ReshapeState:
    """What a planner sees when deciding whether to reshape.

    Carries more than `ResidentThresholdPlanner` uses (`remaining_tokens`, `queue_depth`,
    `train_version`) because the whole point of this seam is that a later deadline-aware planner
    reads the same snapshot without a new plumbing pass -- the real psrl `BoundaryState` had to
    be extended once already for exactly this reason.
    """

    t: float
    awake: frozenset[int]
    """Slot ids currently awake. A drained awake slot (0 residents) is still awake, and is
    exactly the state a merge trigger watches for -- distinct from a slot that was never woken."""
    residents: dict[int, int]
    """Slot id -> resident request count. Only awake slots can be non-zero."""
    remaining_tokens: dict[int, float] = field(default_factory=dict)
    """Slot id -> predicted tokens its residents still have to generate, summed."""
    queue_depth: int = 0
    """Requests waiting in the router, unplaced."""
    train_version: int = 0


class ReshapePlanner(Protocol):
    def plan(self, table: SlotTable, state: ReshapeState) -> frozenset[int] | None:
        """Return the target awake set, or None to leave the current one alone."""
        ...


def _children(table: SlotTable, parent: Slot, child_tp: int) -> list[Slot]:
    """Slots of `child_tp` whose GPUs are a subset of `parent`'s GPUs.

    `SlotTable.from_degrees` lays every degree's windows out as aligned, nested ranges, so a
    higher-degree slot's GPU set is always exactly tiled by some slots of any smaller degree that
    divides it -- this lookup is a filter, not a search for an approximate cover.
    """
    return sorted(
        (slot for slot in table.slots if slot.tp == child_tp and slot.gpus <= parent.gpus),
        key=lambda slot: slot.slot_id,
    )


class ResidentThresholdPlanner:
    """Merge drained same-degree slot pairs into the next TP degree up; split them back when
    the merged slot's own load climbs back past a (higher) threshold.

    This is the "roofline-driven" trigger: a slot's own resident count crossing a threshold,
    not a fleet-wide average and not raw idleness. `merge_below` must be comfortably under the
    decode-cost crossover for the TP degrees in play (see
    `memory/tp-concentration-helps-only-below-the-crossover-total-demand.md`) or every merge
    pays more in lost per-GPU throughput than it gains in latency.
    """

    name = "resident_threshold"

    def __init__(
        self,
        merge_below: float,
        split_above: float,
        min_dwell_s: float = 5.0,
        max_tp: int | None = None,
    ) -> None:
        """
        Args:
            merge_below: Merge two same-degree sibling slots once both have fewer residents
                than this.
            split_above: Split a merged slot back down once its own resident count exceeds this.
                Must be greater than `merge_below`: equal or lower thresholds thrash every
                completion, paying a reconfig and a re-prefill tax each time, since the hook
                fires on every `REQUEST_COMPLETE`.
            min_dwell_s: Minimum simulated seconds between two reshapes. Without this, a fleet
                sitting near the threshold reshapes on single-request arrival/departure noise.
            max_tp: Largest TP degree a merge may produce. None means the table's own largest
                degree is the ceiling.
        """
        if not split_above > merge_below:
            raise ValueError(
                f"split_above ({split_above}) must exceed merge_below ({merge_below}), "
                "or every reshape would immediately reverse itself."
            )
        self.merge_below = float(merge_below)
        self.split_above = float(split_above)
        self.min_dwell_s = float(min_dwell_s)
        self.max_tp = max_tp
        self._last_change_t: float | None = None

    def plan(self, table: SlotTable, state: ReshapeState) -> frozenset[int] | None:
        if self._last_change_t is not None and state.t - self._last_change_t < self.min_dwell_s:
            return None
        awake = state.awake
        degrees = sorted({slot.tp for slot in table.slots})
        ceiling = self.max_tp if self.max_tp is not None else degrees[-1]

        for lower, higher in zip(degrees, degrees[1:]):
            if higher > ceiling:
                break
            for parent in (slot for slot in table.slots if slot.tp == higher):
                children = _children(table, parent, lower)
                if not all(child.slot_id in awake for child in children):
                    continue
                if any(state.residents.get(child.slot_id, 0) >= self.merge_below for child in children):
                    continue
                self._last_change_t = state.t
                return frozenset(awake - {c.slot_id for c in children} | {parent.slot_id})

        for lower, higher in zip(degrees, degrees[1:]):
            for parent in (slot for slot in table.slots if slot.tp == higher):
                if parent.slot_id not in awake:
                    continue
                if state.residents.get(parent.slot_id, 0) <= self.split_above:
                    continue
                children = _children(table, parent, lower)
                self._last_change_t = state.t
                return frozenset(awake - {parent.slot_id} | {c.slot_id for c in children})

        return None


_PLANNERS: dict[str, type] = {
    "resident_threshold": ResidentThresholdPlanner,
}


def build_reshape_planner(name: str, **kwargs) -> ReshapePlanner:
    """Construct the named planner. Mirrors `component/rollout/router/strategy.build_strategy`'s
    registry pattern, kept local since only one planner exists so far."""
    try:
        cls = _PLANNERS[name]
    except KeyError:
        raise ValueError(f"Unknown reshape planner {name!r}. Available: {sorted(_PLANNERS)}.") from None
    return cls(**kwargs)
