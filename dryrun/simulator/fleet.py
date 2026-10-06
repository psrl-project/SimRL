"""The heterogeneous rollout fleet: owns instances and the router.

`HeterogeneousFleet.rebuild` is called at each weight publish.  It aborts every
in-flight request on old instances, constructs new `NativeInstance` objects per
the solver's `FleetPlan`, and re-binds the router's instance list.  Instances
whose `(cohort_id, tp)` shape is unchanged are reused so their `step_stats`
history survives across rebuilds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..component.rollout.engine import NativeInstance, Request
from ..component.rollout.router.strategies import CohortAffinity

if TYPE_CHECKING:
    from ..component.rollout.router import Router
    from ..cost.rollout.base import RolloutCostModel
    from .water_filling import FleetPlan


class HeterogeneousFleet:
    """Owns instances and router; rebuilds them at weight-update boundaries."""

    def __init__(
        self,
        cost_models: dict[int, "RolloutCostModel"],
        router: "Router",
        initial_plan: "FleetPlan",
        kv_blocks_per_tp: dict[int, int],
        instance_cfg: dict,
        t: float,
    ) -> None:
        if not isinstance(router.strategy, CohortAffinity):
            raise ValueError(
                "HeterogeneousFleet requires the router to use the 'cohort_affinity' "
                "strategy; got a different strategy.  Set routing_strategy=cohort_affinity "
                "in the simulation config when heterogeneous.enabled=True."
            )
        self._cost_models = cost_models
        self._router = router
        self._kv_blocks_per_tp = kv_blocks_per_tp
        self._instance_cfg = instance_cfg
        self._current_plan: FleetPlan | None = None
        self.instances: list[NativeInstance] = []

        self._build_instances(initial_plan, t)
        self._current_plan = initial_plan

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def rebuild(self, plan: "FleetPlan", t: float) -> list[Request]:
        """Abort all in-flight requests, rebuild instances per `plan`, re-bind router.

        Returns the aborted requests so the simulator can re-enqueue them.
        """
        # Abort everything on current instances.
        aborted: list[Request] = []
        for inst in self.instances:
            aborted.extend(inst.abort_all(requeue=False))

        self._build_instances(plan, t)
        self._current_plan = plan
        return aborted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_instances(self, plan: "FleetPlan", t: float) -> None:
        """Construct instances for `plan`, reusing unchanged ones."""
        new_instances: list[NativeInstance] = []
        # Build a quick lookup: (cohort_id, tp) -> list of reusable instances
        reuse_pool: dict[tuple[int, int], list[NativeInstance]] = {}
        for inst in self.instances:
            # `inst.tp` is set at construction below. Reading the TP degree off the
            # cost model instead (e.g. `cost._ops.tensor_parallel`) would reach into
            # a private attribute that only ComposedRollout has -- UnifiedRoofline and
            # PSRLFitted do not, so reuse would silently never fire for them.
            reuse_pool.setdefault((inst.cohort_id, inst.tp), []).append(inst)

        instance_id_counter = 0
        for cohort_id, cohort in enumerate(plan.cohorts):
            tp = cohort.tp
            kv_blocks = self._kv_blocks_per_tp.get(
                tp, self._instance_cfg.get("kv_blocks", 100_000)
            )
            cost = self._cost_models[tp]
            key = (cohort_id, tp)
            for _ in range(cohort.n_instances):
                reusable = reuse_pool.get(key, [])
                if reusable:
                    inst = reusable.pop(0)
                    inst.instance_id = instance_id_counter
                    # t_local must not go backwards; bring it to current time
                    inst.t_local = max(inst.t_local, t)
                else:
                    inst = NativeInstance(
                        cost=cost,
                        token_budget=self._instance_cfg["token_budget"],
                        kv_blocks=kv_blocks,
                        block_size=self._instance_cfg["block_size"],
                        max_running=self._instance_cfg["max_running"],
                        max_concurrency=self._instance_cfg["max_concurrency"],
                        instance_id=instance_id_counter,
                        cohort_id=cohort_id,
                        tp=tp,
                        reject_if_kv_full=self._instance_cfg["reject_if_kv_full"],
                        reject_if_waiting=self._instance_cfg["reject_if_waiting"],
                        reject_if_running_full=self._instance_cfg["reject_if_running_full"],
                        enable_prefix_caching=self._instance_cfg["enable_prefix_caching"],
                    )
                    inst.t_local = t
                inst.cohort_id = cohort_id
                new_instances.append(inst)
                instance_id_counter += 1

        self.instances = new_instances
        self._router.instances = new_instances
        # Tell the routing strategy about the new plan
        self._router.strategy.update_plan(plan)
