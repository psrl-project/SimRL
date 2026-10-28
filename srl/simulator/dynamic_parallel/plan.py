"""What a fleet slot table is: every TP shape a run may ever use, fixed before the run starts.

A `Slot` is a GPU-disjoint window reserved for one TP degree. It exists for the whole run with a
stable id and never changes degree -- only whether it is *awake*. `SlotTable` is the full
catalogue of slots a run may draw on; at any moment the fleet has only a GPU-disjoint subset of
them awake and serving requests.

This is deliberately not "one TP degree per instance, instance count changes on every reshape"
(the earlier `FleetPlan` design): that design had no way to express "the same physical GPUs, now
serving a different shape" without renumbering instances, which corrupts per-instance telemetry
history and breaks `Simulator.instances[event.instance_id]` indexing mid-run. A slot's identity
never moves; only its awake/asleep state does.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Slot:
    """One fixed-shape reservation: a TP degree and the GPUs it occupies when awake."""

    slot_id: int
    tp: int
    gpus: frozenset[int]

    def __post_init__(self) -> None:
        if self.tp <= 0:
            raise ValueError(f"Slot {self.slot_id}: TP degree must be positive, got {self.tp!r}.")
        if len(self.gpus) != self.tp:
            raise ValueError(
                f"Slot {self.slot_id} declares TP={self.tp} but spans {len(self.gpus)} GPUs: "
                f"{sorted(self.gpus)!r}."
            )


@dataclass(frozen=True)
class SlotTable:
    """Every slot a run may wake, plus the GPU budget they all draw from."""

    slots: tuple[Slot, ...]
    n_gpus: int

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError("A slot table needs at least one slot.")
        ids = [slot.slot_id for slot in self.slots]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate slot ids: {ids!r}.")
        for slot in self.slots:
            if any(gpu < 0 or gpu >= self.n_gpus for gpu in slot.gpus):
                raise ValueError(
                    f"Slot {slot.slot_id} uses GPUs {sorted(slot.gpus)!r} outside the "
                    f"n_gpus={self.n_gpus} budget."
                )

    @property
    def by_id(self) -> dict[int, Slot]:
        return {slot.slot_id: slot for slot in self.slots}

    def awake_is_valid(self, awake: frozenset[int]) -> bool:
        """Whether `awake` (a set of slot ids) uses every GPU at most once.

        This is the one invariant a planner cannot be trusted to respect on its own: a planner
        bug that wakes two overlapping slots would otherwise silently simulate more GPUs than the
        operator configured, the same failure mode `FleetPlan.__post_init__` raised on.
        """
        by_id = self.by_id
        if any(sid not in by_id for sid in awake):
            return False
        used: set[int] = set()
        for sid in awake:
            gpus = by_id[sid].gpus
            if used & gpus:
                return False
            used |= gpus
        return True

    def describe(self, awake: frozenset[int]) -> str:
        """Return a compact human-readable shape such as ``1xTP2 + 2xTP1`` for `awake`."""
        by_id = self.by_id
        hist = Counter(by_id[sid].tp for sid in awake if sid in by_id)
        parts = [f"{count}xTP{tp}" for tp, count in sorted(hist.items(), reverse=True)]
        return " + ".join(parts) if parts else "(nothing awake)"

    @classmethod
    def from_degrees(cls, tp_degrees: Sequence[int], n_gpus: int) -> SlotTable:
        """Build every aligned GPU window for each degree in `tp_degrees`.

        ``{1, 2}`` on 8 GPUs gives 8 TP1 slots (one per GPU) plus 4 TP2 slots (the windows
        ``[0,1], [2,3], [4,5], [6,7]``) -- the full replica-slot table a dynamic fleet may wake
        subsets of, mirroring the real psrl `dynamic_parallel` package's own construction.

        Args:
            tp_degrees: Every TP degree the run may use. Each must divide `n_gpus`.
            n_gpus: Total GPU budget every awake subset must spend exactly.

        Returns:
            The full slot table, GPU windows assigned in ascending order per degree.
        """
        degrees = sorted(set(tp_degrees))
        if not degrees:
            raise ValueError("Need at least one TP degree.")
        slots: list[Slot] = []
        slot_id = 0
        for tp in degrees:
            if tp <= 0:
                raise ValueError(f"TP degree must be positive, got {tp!r}.")
            if n_gpus % tp != 0:
                raise ValueError(f"TP={tp} does not divide n_gpus={n_gpus}.")
            for start in range(0, n_gpus, tp):
                slots.append(Slot(slot_id=slot_id, tp=tp, gpus=frozenset(range(start, start + tp))))
                slot_id += 1
        return cls(slots=tuple(slots), n_gpus=n_gpus)

    def uniform_awake(self, tp: int) -> frozenset[int]:
        """Return the all-slots-of-one-degree awake set: the homogeneous baseline, in this
        vocabulary, for comparing against a dynamic run on equal terms."""
        matching = frozenset(slot.slot_id for slot in self.slots if slot.tp == tp)
        if not matching:
            raise ValueError(f"No TP={tp} slots in this table.")
        if not self.awake_is_valid(matching):
            raise ValueError(f"TP={tp} slots do not tile the GPU budget disjointly.")
        return matching


@dataclass(frozen=True)
class DynamicParallelConfig:
    """A dynamic-parallel run's fixed shape: every TP degree in play, the GPU budget, and the
    planner that decides when to reshape. Lives on `SimConfig.dynamic_parallel`.

    The per-degree cost models are not here: they go through `Simulator`'s own `rollout_costs`
    constructor argument, the same way the single-TP path already takes its one `rollout_cost`
    outside `SimConfig` (a config dataclass holds declarative shape, not runtime cost objects).
    """

    tp_degrees: tuple[int, ...]
    n_gpus: int
    kv_blocks_per_tp: dict[int, int]
    """TP degree -> KV capacity in blocks while a slot of that degree is awake. Sized by the
    caller for whatever memory model the run assumes (e.g. sleep mode freeing an asleep slot's
    share back to the pool, so an awake slot gets the full `gpu_memory_utilization` budget for
    its own degree)."""
    reconfig_time_s: float = 5.0
    """Pause charged to a newly woken slot. The real psrl `dynamic_parallel` package measures
    4.9-9.0 s for an awake-set switch; 5.0 is its default, not a guess."""
    planner: str = "resident_threshold"
    planner_args: dict = field(default_factory=dict)
    initial_tp: int | None = None
    """Homogeneous degree every slot starts awake at. None uses the smallest configured degree."""

    def __post_init__(self) -> None:
        if not self.tp_degrees:
            raise ValueError("dynamic_parallel needs at least one TP degree.")
        missing_kv = sorted(set(self.tp_degrees) - set(self.kv_blocks_per_tp))
        if missing_kv:
            raise ValueError(f"tp_degrees {self.tp_degrees} has no kv_blocks_per_tp entry for {missing_kv}.")
