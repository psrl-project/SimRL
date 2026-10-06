# tests/test_fleet.py
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from dryrun.simulator.water_filling import CohortPlan, FleetPlan


def _mock_cost() -> MagicMock:
    m = MagicMock()
    m.step_time.return_value = 0.01
    m.decode_coeffs.return_value = (0.0, 0.01, 0.0)
    return m


def _simple_fleet():
    from dryrun.simulator.fleet import HeterogeneousFleet
    from dryrun.component.rollout.router import Router, AdmissionGate, SortKey
    from dryrun.component.rollout.router.strategy import build_strategy

    models = {1: _mock_cost(), 2: _mock_cost()}
    strategy = build_strategy("cohort_affinity")
    router = Router(strategy=strategy, instances=[])

    # CohortPlan(tp=2, n_instances=1) consumes 2 GPU-ranks → n_gpus=2.
    plan = FleetPlan(
        cohorts=(CohortPlan(tp=2, n_instances=1, max_version=None),),
        n_gpus=2,
    )
    fleet = HeterogeneousFleet(
        cost_models=models,
        router=router,
        initial_plan=plan,
        kv_blocks_per_tp={1: 31294, 2: 69532},
        instance_cfg=dict(
            token_budget=12800,
            block_size=16,
            max_running=None,
            max_concurrency=None,
            reject_if_kv_full=True,
            reject_if_waiting=False,
            reject_if_running_full=False,
            enable_prefix_caching=True,
        ),
        t=0.0,
    )
    return fleet, router


def test_fleet_initial_instances_match_plan():
    fleet, _ = _simple_fleet()
    # Plan: 1 instance at TP=2 → 1 NativeInstance
    assert len(fleet.instances) == 1
    assert fleet.instances[0].cohort_id == 0


def test_fleet_rebuild_creates_new_instances():
    fleet, _ = _simple_fleet()
    new_plan = FleetPlan(
        cohorts=(
            CohortPlan(tp=2, n_instances=1, max_version=0),
            CohortPlan(tp=1, n_instances=2, max_version=None),
        ),
        n_gpus=4,
    )
    aborted = fleet.rebuild(new_plan, t=10.0)
    # Two cohorts → 1 + 2 = 3 instances
    assert len(fleet.instances) == 3
    assert fleet.instances[0].cohort_id == 0
    assert fleet.instances[1].cohort_id == 1
    assert fleet.instances[2].cohort_id == 1


def test_fleet_rebuild_sets_t_local():
    fleet, _ = _simple_fleet()
    new_plan = FleetPlan(
        cohorts=(CohortPlan(tp=1, n_instances=4, max_version=None),),
        n_gpus=4,
    )
    fleet.rebuild(new_plan, t=42.0)
    for inst in fleet.instances:
        assert inst.t_local == pytest.approx(42.0)


def test_fleet_rebuild_router_instances_updated():
    fleet, router = _simple_fleet()
    new_plan = FleetPlan(
        cohorts=(CohortPlan(tp=1, n_instances=4, max_version=None),),
        n_gpus=4,
    )
    fleet.rebuild(new_plan, t=5.0)
    assert router.instances is fleet.instances


def test_fleet_rebuild_reuses_instance_when_shape_unchanged():
    fleet, _ = _simple_fleet()
    original_instance = fleet.instances[0]
    # Rebuild with the same plan → same instance object reused
    same_plan = FleetPlan(
        cohorts=(CohortPlan(tp=2, n_instances=1, max_version=None),),
        n_gpus=2,
    )
    fleet.rebuild(same_plan, t=5.0)
    assert fleet.instances[0] is original_instance


def test_fleet_rebuild_aborts_inflight_requests():
    from dryrun.component.rollout.engine import Request
    fleet, _ = _simple_fleet()
    req = Request(rid=0, prompt_len=10, target_len=100, v_traj=0)
    fleet.instances[0].waiting.append(req)

    new_plan = FleetPlan(
        cohorts=(CohortPlan(tp=1, n_instances=4, max_version=None),),
        n_gpus=4,
    )
    aborted = fleet.rebuild(new_plan, t=5.0)
    assert req in aborted


def test_fleet_raises_on_non_cohort_affinity_strategy():
    from dryrun.simulator.fleet import HeterogeneousFleet
    from dryrun.component.rollout.router import Router, AdmissionGate, SortKey
    from dryrun.component.rollout.router.strategy import build_strategy

    models = {1: _mock_cost()}
    strategy = build_strategy("request_num_balance")
    router = Router(strategy=strategy, instances=[])
    plan = FleetPlan(cohorts=(CohortPlan(tp=1, n_instances=4, max_version=None),), n_gpus=4)

    with pytest.raises(ValueError, match="cohort_affinity"):
        HeterogeneousFleet(
            cost_models=models,
            router=router,
            initial_plan=plan,
            kv_blocks_per_tp={1: 31294},
            instance_cfg=dict(
                token_budget=12800, block_size=16, max_running=None,
                max_concurrency=None, reject_if_kv_full=True,
                reject_if_waiting=False, reject_if_running_full=False,
                enable_prefix_caching=True,
            ),
            t=0.0,
        )
