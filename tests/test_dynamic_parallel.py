"""The dynamic-parallel package: plan invariants, solver, fleet, and routing.

Placement is a soft router preference here, not a partition, so the tests that
matter most are the ones showing a high-TP instance stays *available* to current
work. A hard reservation would pass most of these; only
`test_current_version_work_can_use_a_high_tp_instance` separates the two.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from dryrun.component.rollout.engine import NativeInstance, Request
from dryrun.component.rollout.engine.state import InstanceView
from dryrun.component.rollout.router.strategy import build_strategy
from dryrun.cost import UnifiedRoofline
from dryrun.simulator.dynamic_parallel import (
    DynamicParallelFleet,
    FleetPlan,
    ParallelismSolver,
    RequestSnapshot,
)


def _cost() -> UnifiedRoofline:
    return UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_d=1e-7)


def _view(instance_id: int, tp: int, n_active: int = 0) -> InstanceView:
    return InstanceView(
        instance_id=instance_id,
        n_waiting=n_active,
        n_running=0,
        kv_utilization=0.0,
        total_ctx=0,
        tp=tp,
    )


def _req(v_traj: int, rid: int = 0) -> Request:
    return Request(rid=rid, prompt_len=50, target_len=100, v_traj=v_traj)


# ----------------------------------------------------------------------
# FleetPlan
# ----------------------------------------------------------------------


def test_plan_describes_its_shape():
    plan = FleetPlan(tp_per_instance=(2, 1, 1), n_gpus=4)
    assert plan.n_instances == 3
    assert plan.is_heterogeneous
    assert plan.tp_histogram == {1: 2, 2: 1}
    assert plan.describe() == "1xTP2 + 2xTP1"


def test_a_uniform_plan_is_not_heterogeneous():
    """Four TP=1 instances is what a static config produces; it must not count."""
    plan = FleetPlan.uniform(tp=1, n_gpus=4)
    assert plan.n_instances == 4
    assert not plan.is_heterogeneous


def test_plan_rejects_a_misspent_budget():
    with pytest.raises(ValueError, match="GPU budget"):
        FleetPlan(tp_per_instance=(2, 2, 2), n_gpus=4)


def test_uniform_rejects_an_indivisible_budget():
    with pytest.raises(ValueError, match="cannot spend"):
        FleetPlan.uniform(tp=2, n_gpus=5)


# ----------------------------------------------------------------------
# Routing: the soft preference
# ----------------------------------------------------------------------


def test_old_version_work_prefers_the_highest_tp_even_when_it_is_busiest():
    strategy = build_strategy("version_aware_tp")
    strategy.set_reference_version(2)
    # The TP=2 instance is by far the busiest; the preference must still win,
    # because concentration is what the shrinking tail benefits from.
    candidates = [_view(0, tp=2, n_active=9), _view(1, tp=1, n_active=1), _view(2, tp=1, n_active=0)]
    chosen = strategy.select(_req(v_traj=0), candidates)
    assert candidates[chosen].tp == 2
    assert strategy.n_old_to_high_tp == 1


def test_current_version_work_balances_and_ignores_tp():
    strategy = build_strategy("version_aware_tp")
    strategy.set_reference_version(2)
    candidates = [_view(0, tp=2, n_active=9), _view(1, tp=1, n_active=5), _view(2, tp=1, n_active=0)]
    chosen = strategy.select(_req(v_traj=2), candidates)
    assert candidates[chosen].instance_id == 2  # least active, TP irrelevant
    assert strategy.n_old_placements == 0


def test_current_version_work_can_use_a_high_tp_instance():
    """THE test that separates a soft preference from a hard reservation.

    Once the old-version tail drains, the high-TP instances must be usable by
    current work. Under the earlier predicate-gated design they were reserved,
    so they idled while the low-TP instances queued -- which is the load-balancing
    failure this redesign exists to remove.
    """
    strategy = build_strategy("version_aware_tp")
    strategy.set_reference_version(5)
    only_high_tp = [_view(0, tp=2, n_active=0)]
    chosen = strategy.select(_req(v_traj=5), only_high_tp)
    assert chosen == 0, "current-version work was refused a high-TP instance"


def test_old_version_work_still_balances_within_the_highest_tp():
    """A stream of tail requests must not pile onto one of several equal instances."""
    strategy = build_strategy("version_aware_tp")
    strategy.set_reference_version(3)
    candidates = [_view(0, tp=2, n_active=7), _view(1, tp=2, n_active=1), _view(2, tp=1, n_active=0)]
    chosen = strategy.select(_req(v_traj=0), candidates)
    assert candidates[chosen].instance_id == 1  # the idler of the two TP=2 instances


def test_without_a_reference_version_it_degrades_to_pure_balance():
    """Before the first weight publish there is no tail, so no preference applies."""
    strategy = build_strategy("version_aware_tp")
    candidates = [_view(0, tp=2, n_active=9), _view(1, tp=1, n_active=0)]
    chosen = strategy.select(_req(v_traj=0), candidates)
    assert candidates[chosen].instance_id == 1
    assert strategy.n_old_placements == 0


def test_the_strategy_never_strands_a_request():
    """Both paths must return a candidate; None would livelock the simulator."""
    strategy = build_strategy("version_aware_tp")
    strategy.set_reference_version(4)
    for v_traj in (0, 4):
        assert strategy.select(_req(v_traj), [_view(0, tp=1, n_active=3)]) == 0


def test_tp_is_a_declared_field_on_the_instance_view():
    """A runtime-injected attribute would let these tests pass with no plumbing."""
    import dataclasses

    from dryrun.component.rollout.engine.state import InstanceLoad

    assert "tp" in {f.name for f in dataclasses.fields(InstanceLoad)}


def test_get_load_propagates_tp():
    inst = NativeInstance(cost=_cost(), token_budget=1000, kv_blocks=500, tp=2)
    assert inst.get_load().tp == 2


# ----------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------


def _mock_models(step_time=lambda n, t2, ctx, reqs, tp: 0.010 / tp) -> dict:
    models = {}
    for tp in (1, 2):
        model = MagicMock()
        model.step_time.side_effect = lambda n, t2, ctx, reqs, _tp=tp: step_time(
            n, t2, ctx, reqs, _tp
        )
        models[tp] = model
    return models


def test_fixed_strategy_returns_its_plan_verbatim():
    solver = ParallelismSolver(
        cost_models=_mock_models(),
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="fixed",
        fixed_tp_per_instance=(2, 1, 1),
    )
    plan = solver.solve([RequestSnapshot(v_traj=0, ctx=5000)], rollout_version=1)
    assert plan.tp_per_instance == (2, 1, 1)


def test_fixed_strategy_rejects_a_misspent_plan_at_construction():
    """Naming the config error beats failing minutes into a simulation."""
    with pytest.raises(ValueError, match="GPU budget"):
        ParallelismSolver(
            cost_models=_mock_models(),
            n_gpus=4,
            tp_candidates=(1, 2),
            strategy="fixed",
            fixed_tp_per_instance=(2, 2, 2),
        )


def test_a_missing_cost_model_is_refused():
    with pytest.raises(ValueError, match="No cost model for TP"):
        ParallelismSolver(
            cost_models={1: MagicMock()}, n_gpus=4, tp_candidates=(1, 2)
        )


def test_every_search_plan_spends_the_budget_exactly():
    solver = ParallelismSolver(
        cost_models=_mock_models(), n_gpus=4, tp_candidates=(1, 2)
    )
    for r_old, r_new in [(0, 0), (8, 0), (0, 300), (8, 300), (200, 200)]:
        snapshot = [RequestSnapshot(v_traj=0, ctx=9000) for _ in range(r_old)] + [
            RequestSnapshot(v_traj=1, ctx=300) for _ in range(r_new)
        ]
        plan = solver.solve(snapshot, rollout_version=1)
        assert sum(plan.tp_per_instance) == 4, (r_old, r_new, plan.tp_per_instance)


def test_the_enumeration_covers_every_legal_multiset():
    """n_gpus=4 over {1,2} admits exactly 1111, 112, 22 -- no more, no fewer."""
    solver = ParallelismSolver(
        cost_models=_mock_models(), n_gpus=4, tp_candidates=(1, 2)
    )
    legal = {tuple(sorted(p)) for p in solver._legal_plans}
    assert legal == {(1, 1, 1, 1), (1, 1, 2), (2, 2)}


def test_a_budget_no_fleet_can_spend_is_refused():
    with pytest.raises(ValueError, match="spends 5 GPUs exactly|No fleet"):
        ParallelismSolver(cost_models=_mock_models(), n_gpus=5, tp_candidates=(2,))


def test_ties_prefer_the_smaller_fleet():
    """A batch-insensitive model makes every plan score alike; simplest must win."""
    solver = ParallelismSolver(
        cost_models=_mock_models(step_time=lambda n, t2, ctx, reqs, tp: 0.010),
        n_gpus=4,
        tp_candidates=(1, 2),
    )
    plan = solver.solve(
        [RequestSnapshot(v_traj=1, ctx=1000) for _ in range(40)], rollout_version=1
    )
    assert plan.n_instances == 2  # 2xTP2, not 4xTP1


def test_the_scorer_is_live():
    """With a batch-superlinear model, spreading must beat concentrating."""
    solver = ParallelismSolver(
        cost_models=_mock_models(
            step_time=lambda n, t2, ctx, reqs, tp: (0.002 * reqs * reqs) / tp
        ),
        n_gpus=4,
        tp_candidates=(1, 2),
    )
    plan = solver.solve(
        [RequestSnapshot(v_traj=1, ctx=2000) for _ in range(400)], rollout_version=1
    )
    assert plan.tp_per_instance == (1, 1, 1, 1)


def test_high_tp_bias_is_range_checked():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="high_tp_bias"):
            ParallelismSolver(
                cost_models=_mock_models(),
                n_gpus=4,
                tp_candidates=(1, 2),
                high_tp_bias=bad,
            )


def test_the_snapshot_cannot_leak_the_future():
    """Reading target_len would make every measurement an oracle bound."""
    snap = RequestSnapshot(v_traj=0, ctx=1000)
    assert not hasattr(snap, "target_len")
    assert not hasattr(snap, "remaining")


def test_expected_placement_models_the_routers_preference():
    """The solver must price what the router will do, not an assumed partition."""
    solver = ParallelismSolver(
        cost_models=_mock_models(), n_gpus=4, tp_candidates=(1, 2)
    )
    # Plan (2, 1, 1): the TP=2 instance is first in rank order, so the tail goes there.
    placement = solver._expected_placement((2, 1, 1), r_old=6, r_new=30)
    old_counts = [old for old, _ in placement]
    assert old_counts[0] >= old_counts[1], placement
    # Current work spreads over every instance, including the high-TP one.
    assert all(new > 0 for _, new in placement), placement


# ----------------------------------------------------------------------
# Fleet
# ----------------------------------------------------------------------


def _fleet(plan: FleetPlan, **kw) -> tuple[DynamicParallelFleet, object]:
    from dryrun.component.rollout.router import Router

    cost = _cost()
    router = Router(strategy=build_strategy("version_aware_tp"), instances=[])
    fleet = DynamicParallelFleet(
        cost_models={1: cost, 2: cost},
        kv_blocks_per_tp=kw.get("kv", {1: 2000, 2: 3000}),
        router=router,
        instance_cfg=dict(
            token_budget=4096,
            block_size=16,
            max_running=None,
            max_concurrency=None,
            reject_if_kv_full=True,
            reject_if_waiting=False,
            reject_if_running_full=False,
            enable_prefix_caching=True,
        ),
        initial_plan=plan,
        t=kw.get("t", 0.0),
    )
    return fleet, router


def test_the_fleet_materialises_its_plan():
    fleet, router = _fleet(FleetPlan(tp_per_instance=(2, 1, 1), n_gpus=4))
    assert [i.tp for i in fleet.instances] == [2, 1, 1]
    assert [i.instance_id for i in fleet.instances] == [0, 1, 2]
    assert [i.kv_blocks for i in fleet.instances] == [3000, 2000, 2000]
    assert router.instances is fleet.instances


def test_rebuild_preserves_history_for_unchanged_degrees():
    """Telemetry reads step_stats per instance; a reset would corrupt every row."""
    fleet, _ = _fleet(FleetPlan.uniform(tp=1, n_gpus=4))
    for i, inst in enumerate(fleet.instances):
        inst.step_stats.append(f"hist-{i}")
    fleet.rebuild(FleetPlan.uniform(tp=1, n_gpus=4), t=5.0)
    assert [inst.step_stats for inst in fleet.instances] == [
        ["hist-0"], ["hist-1"], ["hist-2"], ["hist-3"]
    ]


def test_rebuild_never_rewinds_a_reused_clock():
    """A backwards clock surfaces at some later alignment, far from the cause."""
    fleet, _ = _fleet(FleetPlan.uniform(tp=1, n_gpus=2))
    fleet.instances[0].t_local = 20.0
    fleet.rebuild(FleetPlan.uniform(tp=1, n_gpus=2), t=5.0)
    assert fleet.instances[0].t_local == 20.0


def test_a_newly_built_instance_starts_at_the_current_time():
    """A fresh instance left at 0.0 trips the simulator's alignment assertion."""
    fleet, _ = _fleet(FleetPlan.uniform(tp=1, n_gpus=2))
    fleet.rebuild(FleetPlan(tp_per_instance=(2,), n_gpus=2), t=7.0)
    assert fleet.instances[0].tp == 2
    assert fleet.instances[0].t_local == 7.0


def test_rebuild_returns_work_from_both_queues():
    fleet, _ = _fleet(FleetPlan.uniform(tp=1, n_gpus=2))
    running, waiting = _req(0, rid=1), _req(0, rid=2)
    fleet.instances[0].running.append(running)
    fleet.instances[0].waiting.append(waiting)
    aborted = fleet.rebuild(FleetPlan(tp_per_instance=(2,), n_gpus=2), t=1.0)
    assert {r.rid for r in aborted} == {1, 2}
    # Identity, not equality: the queues hold these exact objects.
    assert any(r is running for r in aborted)


def test_rebuild_counts_heterogeneous_plans():
    """Reported so a run can say whether heterogeneity actually engaged."""
    fleet, _ = _fleet(FleetPlan.uniform(tp=1, n_gpus=4))
    fleet.rebuild(FleetPlan.uniform(tp=1, n_gpus=4), t=1.0)
    fleet.rebuild(FleetPlan(tp_per_instance=(2, 1, 1), n_gpus=4), t=2.0)
    assert fleet.n_rebuilds == 2
    assert fleet.n_heterogeneous_plans == 1


def test_a_plan_needing_an_unsupplied_model_is_refused():
    from dryrun.component.rollout.router import Router

    cost = _cost()
    router = Router(strategy=build_strategy("version_aware_tp"), instances=[])
    with pytest.raises(ValueError, match="no cost model"):
        DynamicParallelFleet(
            cost_models={1: cost},
            kv_blocks_per_tp={1: 2000},
            router=router,
            instance_cfg=dict(
                token_budget=4096, block_size=16, max_running=None,
                max_concurrency=None, reject_if_kv_full=True,
                reject_if_waiting=False, reject_if_running_full=False,
                enable_prefix_caching=True,
            ),
            initial_plan=FleetPlan(tp_per_instance=(2,), n_gpus=2),
        )


def test_rebuild_tells_the_strategy_the_new_plan():
    fleet, router = _fleet(FleetPlan.uniform(tp=1, n_gpus=4))
    seen = []
    router.strategy.update_plan = lambda plan: seen.append(plan)
    new = FleetPlan(tp_per_instance=(2, 1, 1), n_gpus=4)
    fleet.rebuild(new, t=1.0)
    assert seen and seen[-1] is new


# ----------------------------------------------------------------------
# Opt-in
# ----------------------------------------------------------------------


def test_the_package_is_not_imported_unless_used():
    """`enabled=false` must leave this package untouched, which is what opt-in means."""
    for name in list(sys.modules):
        if "dynamic_parallel" in name:
            del sys.modules[name]
    import importlib

    import dryrun.simulator.simulator as sim_module

    importlib.reload(sim_module)
    leaked = [n for n in sys.modules if "dynamic_parallel" in n]
    assert not leaked, f"importing the simulator pulled in {leaked}"
