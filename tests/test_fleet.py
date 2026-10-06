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

    # Finding 2: t_local on a reused instance must not go backwards.
    # Advance the clock to 20.0, then rebuild at t=5.0 (< 20.0).
    fleet.instances[0].t_local = 20.0
    fleet.rebuild(same_plan, t=5.0)
    assert fleet.instances[0] is original_instance
    assert fleet.instances[0].t_local == pytest.approx(20.0), (
        "Reused instance had its clock rewound: "
        f"expected max(20.0, 5.0) == 20.0, got {fleet.instances[0].t_local}"
    )

    # Other half: a NEWLY constructed instance must receive exactly t, never 0.0.
    new_shape_plan = FleetPlan(
        cohorts=(CohortPlan(tp=1, n_instances=2, max_version=None),),
        n_gpus=2,
    )
    fleet.rebuild(new_shape_plan, t=7.0)
    for inst in fleet.instances:
        assert inst.t_local == pytest.approx(7.0), (
            f"Newly constructed instance has t_local={inst.t_local!r}, expected 7.0"
        )


def test_fleet_rebuild_aborts_inflight_requests():
    from dryrun.component.rollout.engine import Request
    fleet, _ = _simple_fleet()
    req_waiting = Request(rid=0, prompt_len=10, target_len=100, v_traj=0)
    # Finding 3: seed both queues so the running path is covered.
    req_running = Request(rid=1, prompt_len=10, target_len=100, v_traj=0)
    fleet.instances[0].waiting.append(req_waiting)
    fleet.instances[0].running.append(req_running)

    new_plan = FleetPlan(
        cohorts=(CohortPlan(tp=1, n_instances=4, max_version=None),),
        n_gpus=4,
    )
    aborted = fleet.rebuild(new_plan, t=5.0)
    assert req_waiting in aborted
    assert req_running in aborted


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


def test_fleet_rebuild_updates_routing_strategy():
    """Verify that update_plan is called on the strategy after every rebuild.

    The initial plan is a single catch-all cohort (tp=1, n_instances=2).  After
    rebuild to a two-cohort plan — TP=2 for old-version (max_version=0) and TP=1
    for new-version (catch-all) — a new-version request (v_traj=1) must route to
    cohort_id=1 (the TP=1 cohort).

    Discrimination analysis: routing v_traj=0 to cohort 0 does not discriminate
    because both the old (single catch-all at index 0) and new plans resolve
    v_traj=0 to cohort index 0, so the assertion passes even with update_plan
    stubbed out.  Using v_traj=1 is the minimal change that forces a difference:
    the fresh plan maps it to cohort 1, but the stale plan still maps it to
    cohort 0 (the old catch-all), routing the request to the TP=2 instance
    instead of a TP=1 instance.
    """
    from dryrun.simulator.fleet import HeterogeneousFleet
    from dryrun.component.rollout.router import Router
    from dryrun.component.rollout.router.strategy import build_strategy
    from dryrun.component.rollout.engine import Request
    from dryrun.cost import UnifiedRoofline

    cost_models = {1: UnifiedRoofline(F=0.01), 2: UnifiedRoofline(F=0.01)}
    strategy = build_strategy("cohort_affinity")
    router = Router(strategy=strategy, instances=[])

    # Initial plan: single catch-all cohort (tp=1).  The strategy stores this as
    # its only cohort, so every request maps to cohort index 0.
    initial_plan = FleetPlan(
        cohorts=(CohortPlan(tp=1, n_instances=2, max_version=None),),
        n_gpus=2,
    )
    fleet = HeterogeneousFleet(
        cost_models=cost_models,
        router=router,
        initial_plan=initial_plan,
        kv_blocks_per_tp={1: 31294, 2: 69532},
        instance_cfg=dict(
            token_budget=12800,
            block_size=16,
            max_running=None,
            max_concurrency=None,
            reject_if_kv_full=True,
            reject_if_waiting=False,
            reject_if_running_full=False,
            enable_prefix_caching=False,
        ),
        t=0.0,
    )

    # Rebuild to two cohorts: TP=2 for old-version (v_traj <= 0), TP=1 for new.
    # After this rebuild the fleet has:
    #   instance 0: cohort_id=0, tp=2
    #   instance 1: cohort_id=1, tp=1
    #   instance 2: cohort_id=1, tp=1
    two_cohort_plan = FleetPlan(
        cohorts=(
            CohortPlan(tp=2, n_instances=1, max_version=0),
            CohortPlan(tp=1, n_instances=2, max_version=None),
        ),
        n_gpus=4,
    )
    fleet.rebuild(two_cohort_plan, t=0.0)

    # A new-version request: v_traj=1 > max_version=0, so the fresh plan routes
    # it to cohort 1 (TP=1).  Under the stale plan (single cohort, catch-all at
    # index 0) it would route to cohort 0 (TP=2) instead.
    req = Request(rid=0, prompt_len=50, target_len=100, v_traj=1)
    router.enqueue([req], t=0.0)
    placed = router.dispatch(t=0.0)

    assert len(placed) == 1, "Request was not dispatched"
    _placed_req, instance_id = placed[0]
    placed_instance = fleet.instances[instance_id]
    assert placed_instance.cohort_id == 1, (
        f"Expected cohort_id=1 (TP=1 new-version cohort) but got "
        f"{placed_instance.cohort_id}.  This means update_plan was not called "
        "after rebuild and the strategy is using the stale single-cohort plan."
    )
    assert strategy.n_cohort_leaks == 0
