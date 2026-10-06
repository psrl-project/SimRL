# tests/test_water_filling.py
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from dryrun.simulator.water_filling import (
    CohortPlan,
    FleetPlan,
    RequestSnapshot,
    WaterFillingSolver,
)


def _mock_cost_models(tp_set=(1, 2)):
    """Return mock ComposedRollout objects that report a step_time inversely proportional to TP."""
    models = {}
    for tp in tp_set:
        m = MagicMock()
        # step_time decreases with TP: TP=2 is ~1.5x faster per token at moderate batch
        m.step_time.side_effect = lambda n, t2, ctx, reqs, _tp=tp: 0.010 / _tp
        models[tp] = m
    return models


def _snap(v_traj: int, ctx: int = 5000) -> RequestSnapshot:
    return RequestSnapshot(v_traj=v_traj, ctx=ctx, instance_id=0)


def test_fleet_plan_budget_invariant():
    plan = FleetPlan(cohorts=(CohortPlan(tp=2, n_instances=1, max_version=0),
                               CohortPlan(tp=1, n_instances=2, max_version=None)))
    assert sum(c.tp * c.n_instances for c in plan.cohorts) == 4


def test_fleet_plan_budget_violated():
    with pytest.raises(ValueError, match="GPU budget"):
        FleetPlan(cohorts=(CohortPlan(tp=2, n_instances=2, max_version=None),
                            CohortPlan(tp=1, n_instances=2, max_version=None)))


def test_solver_fixed_strategy():
    models = _mock_cost_models()
    solver = WaterFillingSolver(
        cost_models=models,
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="fixed",
        fixed_cohorts=[{"tp": 1, "n_instances": 4}],
        max_concurrency=None,
    )
    snapshot = [_snap(v_traj=0) for _ in range(20)]
    plan = solver.solve(snapshot, rollout_version=1)
    assert len(plan.cohorts) == 1
    assert plan.cohorts[0].tp == 1
    assert plan.cohorts[0].n_instances == 4


def test_solver_fixed_two_cohorts():
    models = _mock_cost_models()
    solver = WaterFillingSolver(
        cost_models=models,
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="fixed",
        fixed_cohorts=[{"tp": 2, "n_instances": 2}],
        max_concurrency=None,
    )
    plan = solver.solve([], rollout_version=0)
    assert plan.cohorts[0].tp == 2
    assert plan.cohorts[0].n_instances == 2


def test_solver_search_degenerate_no_old():
    """When all requests are new-version, solver returns a single homogeneous cohort."""
    models = _mock_cost_models()
    solver = WaterFillingSolver(
        cost_models=models,
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="search",
        fixed_cohorts=[],
        max_concurrency=None,
    )
    # All requests have v_traj == rollout_version, so R_old == 0
    snapshot = [_snap(v_traj=1) for _ in range(100)]
    plan = solver.solve(snapshot, rollout_version=1)
    assert len(plan.cohorts) == 1
    assert sum(c.tp * c.n_instances for c in plan.cohorts) == 4


def test_solver_search_produces_valid_budget():
    models = _mock_cost_models()
    solver = WaterFillingSolver(
        cost_models=models,
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="search",
        fixed_cohorts=[],
        max_concurrency=None,
    )
    snapshot = (
        [_snap(v_traj=0) for _ in range(8)]    # old-version (small cohort)
        + [_snap(v_traj=1) for _ in range(200)]  # new-version (large cohort)
    )
    plan = solver.solve(snapshot, rollout_version=1)
    assert sum(c.tp * c.n_instances for c in plan.cohorts) == 4


def test_solver_target_len_not_accessible():
    """RequestSnapshot must not have target_len; the firewall is structural."""
    snap = RequestSnapshot(v_traj=0, ctx=1000, instance_id=None)
    assert not hasattr(snap, "target_len")
    assert not hasattr(snap, "remaining")


def test_solver_search_ties_prefer_fewer_cohorts():
    """When two configurations score identically, prefer fewer cohorts (homogeneous)."""
    # Use a mock that scores all fleets equally — tie-break must favour 1 cohort
    models: dict = {}
    for tp in (1, 2):
        m = MagicMock()
        m.step_time.return_value = 0.010  # all fleets score the same
        models[tp] = m
    solver = WaterFillingSolver(
        cost_models=models,
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="search",
        fixed_cohorts=[],
        max_concurrency=None,
    )
    snapshot = [_snap(v_traj=0, ctx=100) for _ in range(50)]
    plan = solver.solve(snapshot, rollout_version=1)
    # Tie → prefer 1 cohort
    assert len(plan.cohorts) == 1
