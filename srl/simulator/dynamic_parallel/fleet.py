"""Owning every TP slot a run may use, and reshaping which GPU-disjoint subset is awake.

A slot's TP degree never changes: `SlotFleet` builds one `NativeInstance` per slot at
construction and never discards or renumbers it, so per-instance telemetry history (`step_stats`,
`instance_id`) lines up for the whole run -- `Simulator.instances[event.instance_id]` indexing
depends on this. Only the *awake* subset is handed to the router; a sleeping slot holds no
requests and produces no events (`Simulator._seed_events` already skips `n_active == 0`), so its
telemetry row stays flat at zero until it wakes.

Reshaping costs real work: `abort_all` drops the KV of every request leaving an awake slot,
charging `reprefill_tokens` exactly the way a weight-update rebuild always has (see
`component/rollout/engine/instance.py:abort_all`). The caller must re-enqueue the returned
requests into the router.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...component.rollout.engine import NativeInstance, Request

if TYPE_CHECKING:
    from ...component.rollout.router import Router
    from ...cost.rollout.base import RolloutCostModel
    from .plan import Slot, SlotTable


class SlotFleet:
    """Owns every slot's instance and tracks which GPU-disjoint subset is awake."""

    def __init__(
        self,
        table: "SlotTable",
        cost_models: dict[int, "RolloutCostModel"],
        kv_blocks_per_tp: dict[int, int],
        router: "Router",
        instance_cfg: dict,
        initial_awake: frozenset[int],
        t: float = 0.0,
    ) -> None:
        """
        Build every slot's instance once and wake the starting subset.

        Args:
            table: The fixed catalogue of slots this run may ever use.
            cost_models: TP degree -> cost model, one per degree the table declares.
            kv_blocks_per_tp: TP degree -> KV capacity in blocks while a slot of that degree is
                awake. Sized by the caller for whatever memory model the run assumes (e.g. a
                sleeping slot releasing its memory back to the pool via vLLM sleep mode).
            router: The router to rebind on every reshape. Must be the same object the simulator
                dispatches through.
            instance_cfg: Engine knobs shared by every instance (token budget, block size,
                admission gates, prefix caching, eviction).
            initial_awake: Slot ids awake at `t`. Must be GPU-disjoint.
            t: Simulation time every instance starts at.
        """
        self._table = table
        self._cost_models = cost_models
        self._kv_blocks_per_tp = kv_blocks_per_tp
        self._router = router
        self._instance_cfg = instance_cfg
        unknown = sorted({slot.tp for slot in table.slots} - set(cost_models))
        if unknown:
            raise ValueError(f"Slot table uses TP {unknown} but no cost model was supplied for it.")
        unknown_kv = sorted({slot.tp for slot in table.slots} - set(kv_blocks_per_tp))
        if unknown_kv:
            raise ValueError(f"Slot table uses TP {unknown_kv} but no KV capacity was supplied for it.")
        if not table.awake_is_valid(initial_awake):
            raise ValueError(f"Initial awake set {sorted(initial_awake)} is not GPU-disjoint.")

        self._by_slot_id: dict[int, NativeInstance] = {
            slot.slot_id: self._new_instance(slot) for slot in table.slots
        }
        for inst in self._by_slot_id.values():
            inst.t_local = t
        self.awake: frozenset[int] = frozenset()
        self.n_reshapes = 0
        self.n_heterogeneous_awakes = 0
        self._set_awake(initial_awake)

    @property
    def table(self) -> "SlotTable":
        return self._table

    @property
    def instances(self) -> list[NativeInstance]:
        """Every slot's instance, in slot-id order -- what `Simulator.instances` binds to.

        Fixed length and fixed order for the whole run: position `i` is always slot id `i`'s
        instance, awake or not, which is what keeps `Simulator`'s index-by-instance-id bookkeeping
        valid across a reshape.
        """
        return [self._by_slot_id[slot.slot_id] for slot in self._table.slots]

    @property
    def awake_instances(self) -> list[NativeInstance]:
        """Currently awake instances only -- what the router may place work on."""
        return [self._by_slot_id[sid] for sid in sorted(self.awake)]

    def reshape(self, awake: frozenset[int], t: float, reconfig_time_s: float = 0.0) -> list[Request]:
        """
        Abort everything leaving the awake set, wake the rest, and hand aborted work back.

        Args:
            awake: Target slot ids. Must be GPU-disjoint per `table.awake_is_valid`.
            t: Current simulation time.
            reconfig_time_s: Pause charged only to slots newly woken by this call. An
                already-awake slot keeps its KV and its clock untouched -- only the slots
                actually involved in the transition pause.

        Returns:
            Requests aborted off slots leaving the awake set: generated tokens and version
            segments intact, KV gone. The caller must re-enqueue them into the router.
        """
        if not self._table.awake_is_valid(awake):
            raise ValueError(f"Awake set {sorted(awake)} is not GPU-disjoint.")
        leaving = self.awake - awake
        waking = awake - self.awake
        aborted: list[Request] = []
        for slot_id in leaving:
            aborted.extend(self._by_slot_id[slot_id].abort_all(requeue=False))
        self._set_awake(awake, t=t, waking=waking, reconfig_time_s=reconfig_time_s)
        self.n_reshapes += 1
        if len({self._table.by_id[sid].tp for sid in awake}) > 1:
            self.n_heterogeneous_awakes += 1
        return aborted

    def _set_awake(
        self,
        awake: frozenset[int],
        t: float | None = None,
        waking: frozenset[int] | None = None,
        reconfig_time_s: float = 0.0,
    ) -> None:
        """Commit the new awake set and rebind the router to it."""
        self.awake = frozenset(awake)
        if t is not None:
            for slot_id in self.awake:
                inst = self._by_slot_id[slot_id]
                # Never rewind a clock: an already-awake slot may be ahead of `t`.
                inst.t_local = max(inst.t_local, t)
            for slot_id in waking or ():
                inst = self._by_slot_id[slot_id]
                inst.t_local = max(inst.t_local, t + reconfig_time_s)
        # `router.instances` stays the full, dense, position-equals-instance_id list for the
        # whole run -- `Router._place` indexes it positionally. `awake` is the narrowing filter;
        # see `Router._active_instances`.
        self._router.instances = self.instances
        self._router.awake = self.awake

    def _new_instance(self, slot: "Slot") -> NativeInstance:
        """Construct the one instance this slot will ever have."""
        cfg = self._instance_cfg
        return NativeInstance(
            cost=self._cost_models[slot.tp],
            token_budget=cfg["token_budget"],
            kv_blocks=self._kv_blocks_per_tp[slot.tp],
            block_size=cfg["block_size"],
            max_running=cfg["max_running"],
            max_concurrency=cfg["max_concurrency"],
            instance_id=slot.slot_id,
            tp=slot.tp,
            reject_if_kv_full=cfg["reject_if_kv_full"],
            reject_if_waiting=cfg["reject_if_waiting"],
            reject_if_running_full=cfg["reject_if_running_full"],
            enable_prefix_caching=cfg["enable_prefix_caching"],
            eviction=cfg.get("eviction"),
        )
