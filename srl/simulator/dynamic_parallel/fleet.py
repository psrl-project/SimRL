"""Applying a fleet plan: rebuild the instance list at a weight update.

A weight update invalidates every KV block anyway. `abort_all` frees them and
`reset_hashes` clears the prefix-cache map, so this is the one moment a rebuild
costs nothing beyond scheduling metadata. There is no KV to migrate.

Instances are reused when their TP degree is unchanged, which keeps their
`step_stats` history: telemetry reads it per instance, so discarding an instance
would reset every per-instance row at each weight update and make span and
utilisation comparisons against a measured run wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...component.rollout.engine import NativeInstance, Request

if TYPE_CHECKING:
    from ...component.rollout.router import Router
    from ...cost.rollout.base import RolloutCostModel
    from .plan import FleetPlan


class DynamicParallelFleet:
    """Owns the rollout instances and re-shapes them on demand."""

    def __init__(
        self,
        cost_models: dict[int, "RolloutCostModel"],
        kv_blocks_per_tp: dict[int, int],
        router: "Router",
        instance_cfg: dict,
        initial_plan: "FleetPlan",
        t: float = 0.0,
    ) -> None:
        """
        Build the starting fleet and bind it to the router.

        Args:
            cost_models: TP degree -> cost model, one per degree the plan may use.
            kv_blocks_per_tp: TP degree -> KV capacity in blocks. Genuinely differs
                per degree: sharding weights across more ranks frees memory for
                cache, so a TP=2 rank holds more blocks than a TP=1 rank, not fewer.
            router: The router to rebind on every rebuild. Must be the same object
                the simulator dispatches through.
            instance_cfg: Engine knobs shared by every instance (token budget, block
                size, admission gates, prefix caching).
            initial_plan: The starting layout.
            t: Simulation time the first instances start at.
        """
        self._cost_models = cost_models
        self._kv_blocks_per_tp = kv_blocks_per_tp
        self._router = router
        self._instance_cfg = instance_cfg
        self.instances: list[NativeInstance] = []
        self.plan = initial_plan
        self.n_rebuilds = 0
        self.n_heterogeneous_plans = 0
        self._apply(initial_plan, t)

    def rebuild(self, plan: "FleetPlan", t: float) -> list[Request]:
        """
        Abort everything in flight, re-shape the fleet, and hand the work back.

        Args:
            plan: The new layout.
            t: Current simulation time. Newly built instances start here, since a
                fresh instance would otherwise sit at 0.0 and trip the simulator's
                clock-alignment assertion later, far from the cause.

        Returns:
            The aborted requests, for the caller to re-enqueue into the router.
            They keep their generated tokens and version segments. Only the KV is
            lost, which is what a weight update means.
        """
        aborted: list[Request] = []
        for inst in self.instances:
            aborted.extend(inst.abort_all(requeue=False))
        self._apply(plan, t)
        self.n_rebuilds += 1
        if plan.is_heterogeneous:
            self.n_heterogeneous_plans += 1
        return aborted

    def _apply(self, plan: "FleetPlan", t: float) -> None:
        """Materialise `plan`, reusing instances whose TP degree is unchanged."""
        unknown = sorted(set(plan.tp_per_instance) - set(self._cost_models))
        if unknown:
            raise ValueError(
                f"Plan uses TP {unknown} but no cost model was supplied for it."
            )

        # Reuse pool keyed on TP alone. Reading the degree off `inst.tp` rather than
        # off the cost model is deliberate: only ComposedRollout carries a TP degree,
        # so deriving it would make reuse silently never fire for the fitted models.
        pool: dict[int, list[NativeInstance]] = {}
        for inst in self.instances:
            pool.setdefault(inst.tp, []).append(inst)

        rebuilt: list[NativeInstance] = []
        for instance_id, tp in enumerate(plan.tp_per_instance):
            available = pool.get(tp)
            if available:
                inst = available.pop(0)
                inst.instance_id = instance_id
                # Never rewind a clock: a reused instance may already be ahead of a
                # caller-supplied `t`, and moving it backwards would surface as a
                # failure at some later alignment rather than here.
                inst.t_local = max(inst.t_local, t)
            else:
                inst = self._new_instance(instance_id, tp)
                inst.t_local = t
            rebuilt.append(inst)

        self.instances = rebuilt
        self.plan = plan
        self._router.instances = rebuilt

    def _new_instance(self, instance_id: int, tp: int) -> NativeInstance:
        """Construct one instance at the given TP degree."""
        cfg = self._instance_cfg
        return NativeInstance(
            cost=self._cost_models[tp],
            token_budget=cfg["token_budget"],
            kv_blocks=self._kv_blocks_per_tp[tp],
            block_size=cfg["block_size"],
            max_running=cfg["max_running"],
            max_concurrency=cfg["max_concurrency"],
            instance_id=instance_id,
            tp=tp,
            reject_if_kv_full=cfg["reject_if_kv_full"],
            reject_if_waiting=cfg["reject_if_waiting"],
            reject_if_running_full=cfg["reject_if_running_full"],
            enable_prefix_caching=cfg["enable_prefix_caching"],
        )
