from __future__ import annotations
import pytest
from dryrun.component.rollout.engine import NativeInstance, Request
from dryrun.component.rollout.engine.state import InstanceView
from dryrun.component.rollout.router.strategy import build_strategy
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
