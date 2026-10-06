from __future__ import annotations
import pytest
from dryrun.component.rollout.engine import NativeInstance, Request
from dryrun.component.rollout.engine.state import InstanceView
from dryrun.component.rollout.router import Router
from dryrun.component.rollout.router.strategy import build_strategy
from dryrun.cost import UnifiedRoofline
from dryrun.simulator.water_filling import CohortPlan, FleetPlan


def _view(instance_id: int, cohort_id: int, n_active: int = 0) -> InstanceView:
    """Build an InstanceView with cohort_id set as a real field.

    cohort_id is a declared field on InstanceLoad (which InstanceView extends), so it
    is passed to the constructor. Injecting it with object.__setattr__ instead would
    make these tests pass even if the field were never added, which is the bug the
    tests exist to catch.
    """
    return InstanceView(
        instance_id=instance_id,
        n_waiting=n_active,
        n_running=0,
        kv_utilization=0.0,
        total_ctx=0,
        cohort_id=cohort_id,
    )


def _req(v_traj: int) -> Request:
    r = Request(rid=0, prompt_len=10, target_len=100, v_traj=v_traj)
    return r


def _plan(old_max_version: int = 0) -> FleetPlan:
    return FleetPlan(
        cohorts=(
            CohortPlan(tp=2, n_instances=1, max_version=old_max_version),
            CohortPlan(tp=1, n_instances=2, max_version=None),
        ),
        n_gpus=4,
    )


def test_cohort_id_is_a_declared_field_on_instance_load():
    """cohort_id must be a dataclass field, not an attribute bolted on at runtime."""
    import dataclasses
    from dryrun.component.rollout.engine.state import InstanceLoad
    names = {f.name for f in dataclasses.fields(InstanceLoad)}
    assert "cohort_id" in names


def test_cohort_id_and_tp_default_on_instance():
    from dryrun.cost import UnifiedRoofline
    cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_d=1e-7)
    inst = NativeInstance(cost=cost, token_budget=1000, kv_blocks=500)
    assert inst.cohort_id == 0
    assert inst.tp == 1


def test_get_load_propagates_cohort_id():
    """A router reading get_load() must see the instance's real cohort."""
    from dryrun.cost import UnifiedRoofline
    cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_d=1e-7)
    inst = NativeInstance(cost=cost, token_budget=1000, kv_blocks=500, cohort_id=3)
    assert inst.get_load().cohort_id == 3


def test_cohort_affinity_routes_old_to_high_tp():
    plan = _plan(old_max_version=0)
    strategy = build_strategy("cohort_affinity", plan=plan)

    # Old-version request (v_traj=0) → should prefer cohort 0 (TP=2, max_version=0)
    req = _req(v_traj=0)
    candidates = [
        _view(instance_id=0, cohort_id=0, n_active=5),   # cohort 0 (TP=2 old)
        _view(instance_id=1, cohort_id=1, n_active=3),   # cohort 1 (TP=1 new)
        _view(instance_id=2, cohort_id=1, n_active=4),   # cohort 1 (TP=1 new)
    ]
    chosen = strategy.select(req, candidates)
    # Must select from cohort 0 (index 0 in candidates)
    assert chosen == 0


def test_cohort_affinity_routes_new_to_low_tp():
    plan = _plan(old_max_version=0)
    strategy = build_strategy("cohort_affinity", plan=plan)

    # New-version request (v_traj=1) → should go to cohort 1 (catch-all, TP=1)
    req = _req(v_traj=1)
    candidates = [
        _view(instance_id=0, cohort_id=0, n_active=2),
        _view(instance_id=1, cohort_id=1, n_active=10),
        _view(instance_id=2, cohort_id=1, n_active=3),
    ]
    chosen = strategy.select(req, candidates)
    # Should pick instance_id=2 (cohort 1, least active)
    assert candidates[chosen].cohort_id == 1


def test_cohort_affinity_fallback_when_cohort_full():
    """When the preferred cohort has no candidates, fall back to the full list."""
    plan = _plan(old_max_version=0)
    strategy = build_strategy("cohort_affinity", plan=plan)

    req = _req(v_traj=0)  # old-version, wants cohort 0
    # No cohort-0 candidates — only cohort-1 available
    candidates = [
        _view(instance_id=1, cohort_id=1, n_active=3),
        _view(instance_id=2, cohort_id=1, n_active=7),
    ]
    chosen = strategy.select(req, candidates)
    # Falls back to least-active across all candidates
    assert chosen == 0  # index 0, n_active=3
    assert strategy.n_cohort_leaks == 1


def test_router_view_wires_cohort_id():
    """Router._view() must carry the instance's cohort_id into the InstanceView.

    This is the seam that broke silently: deleting `cohort_id=load.cohort_id`
    from Router._view() still lets all strategy-level tests pass because those
    tests construct InstanceView objects directly.  This test exercises the
    actual wiring path.
    """
    cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_d=1e-7)
    inst = NativeInstance(cost=cost, token_budget=4096, kv_blocks=4000,
                          instance_id=7, cohort_id=3, tp=2)
    plan = FleetPlan(
        cohorts=(CohortPlan(tp=2, n_instances=1, max_version=0),),
        n_gpus=2,
    )
    strategy = build_strategy("cohort_affinity", plan=plan)
    router = Router(strategy=strategy, instances=[inst])
    view = router._view(inst)
    assert view.cohort_id == 3, (
        f"Router._view() returned cohort_id={view.cohort_id}, expected 3. "
        "Check that cohort_id=load.cohort_id is present in router.py:_view()."
    )


def _make_cohort_affinity_fleet() -> tuple[list[NativeInstance], Router]:
    """Build the 3-instance fleet used by the end-to-end wiring test.

    Instance 0: cohort 0, TP=2 (old-version high-TP cohort, max_version=0).
    Instances 1-2: cohort 1, TP=1 (catch-all new-version cohort).
    """
    cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_d=1e-7)
    insts = [
        NativeInstance(cost=cost, token_budget=4096, kv_blocks=4000,
                       instance_id=i, cohort_id=(0 if i == 0 else 1),
                       tp=(2 if i == 0 else 1))
        for i in range(3)
    ]
    plan = FleetPlan(
        cohorts=(CohortPlan(tp=2, n_instances=1, max_version=0),
                 CohortPlan(tp=1, n_instances=2, max_version=None)),
        n_gpus=4,
    )
    strategy = build_strategy("cohort_affinity", plan=plan)
    router = Router(strategy=strategy, instances=insts)
    return insts, router


def test_router_dispatch_enforces_cohort_affinity():
    """End-to-end: 4 old-version requests must all land on cohort 0 (instance 0).

    This is the probe that catches the Router._view() wiring bug: with the line
    `cohort_id=load.cohort_id` deleted, the InstanceView cohort_id defaults to 0
    for every instance, so the strategy sees all instances as cohort 0 and spreads
    requests across all three instances.  The test catches this because instances
    1 and 2 have cohort_id=1, so any placement there is a violation.
    """
    insts, router = _make_cohort_affinity_fleet()
    strategy = router.strategy
    router.enqueue([Request(rid=i, prompt_len=50, target_len=100, v_traj=0)
                    for i in range(4)], 0.0)
    placed = router.dispatch(0.0)

    assert len(placed) == 4, f"Expected 4 placements, got {len(placed)}"
    assert all(insts[iid].cohort_id == 0 for _, iid in placed), (
        f"Some requests leaked to the wrong cohort: "
        f"{[(req.rid, iid, insts[iid].cohort_id) for req, iid in placed]}"
    )
    assert strategy.n_cohort_leaks == 0, (
        f"n_cohort_leaks={strategy.n_cohort_leaks}; a leak would mask the wiring bug"
    )


def test_preferred_cohort_last_fallback_for_no_catchall_plan():
    """_preferred_cohort_id falls back to the last cohort when no catch-all exists.

    A well-formed solver plan always ends in a catch-all (max_version=None), so
    this branch is unreachable in normal use.  When a hand-built plan omits the
    catch-all, a request whose v_traj exceeds every bound must still be served
    rather than raising, and "last cohort" is the documented degraded behaviour.
    """
    plan = FleetPlan(
        cohorts=(
            CohortPlan(tp=2, n_instances=1, max_version=0),
            CohortPlan(tp=1, n_instances=1, max_version=2),
        ),
        n_gpus=3,
    )
    strategy = build_strategy("cohort_affinity", plan=plan)
    # v_traj=99 exceeds both max_version bounds; should land on the last cohort (index 1)
    req = Request(rid=0, prompt_len=10, target_len=100, v_traj=99)
    preferred = strategy._preferred_cohort_id(req)
    assert preferred == 1, (
        f"Expected last-cohort fallback index 1, got {preferred}"
    )
