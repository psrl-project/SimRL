"""The dynamic-parallel package: `FleetPlan` invariants and `DynamicParallelFleet` rebuilds."""

from __future__ import annotations

import dataclasses
import sys

import pytest

from dryrun.component.rollout.engine import NativeInstance, Request
from dryrun.component.rollout.engine.state import InstanceLoad
from dryrun.component.rollout.router import Router
from dryrun.component.rollout.router.strategy import build_strategy
from dryrun.cost import UnifiedRoofline
from dryrun.simulator.dynamic_parallel import DynamicParallelFleet, FleetPlan

_INSTANCE_CFG = dict(
    token_budget=4096,
    block_size=16,
    max_running=None,
    max_concurrency=None,
    reject_if_kv_full=True,
    reject_if_waiting=False,
    reject_if_running_full=False,
    enable_prefix_caching=True,
)


def _cost() -> UnifiedRoofline:
    return UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_d=1e-7)


def _req(v_traj: int, rid: int = 0) -> Request:
    return Request(rid=rid, prompt_len=50, target_len=100, v_traj=v_traj)


# --- FleetPlan ---


def test_plan_describes_its_shape():
    plan = FleetPlan(tp_per_instance=(2, 1, 1), n_gpus=4)
    assert plan.n_instances == 3
    assert plan.is_heterogeneous
    assert plan.tp_histogram == {1: 2, 2: 1}
    assert plan.describe() == "1xTP2 + 2xTP1"


def test_a_uniform_plan_is_not_heterogeneous():
    """Four TP=1 instances is what a static config produces, so it must not count."""
    plan = FleetPlan.uniform(tp=1, n_gpus=4)
    assert plan.n_instances == 4
    assert not plan.is_heterogeneous


def test_plan_rejects_a_misspent_budget():
    with pytest.raises(ValueError, match="GPU budget"):
        FleetPlan(tp_per_instance=(2, 2, 2), n_gpus=4)


def test_uniform_rejects_an_indivisible_budget():
    with pytest.raises(ValueError, match="cannot spend"):
        FleetPlan.uniform(tp=2, n_gpus=5)


# --- TP on the load snapshot ---


def test_tp_is_a_declared_field_on_the_instance_view():
    """A runtime-injected attribute would let routing tests pass with no plumbing."""
    assert "tp" in {f.name for f in dataclasses.fields(InstanceLoad)}


def test_get_load_propagates_tp():
    inst = NativeInstance(cost=_cost(), token_budget=1000, kv_blocks=500, tp=2)
    assert inst.get_load().tp == 2


# --- Fleet ---


def _fleet(plan: FleetPlan, **kw) -> tuple[DynamicParallelFleet, Router]:
    router = Router(strategy=build_strategy("request_num_balance"), instances=[])
    fleet = DynamicParallelFleet(
        cost_models={1: _cost(), 2: _cost()},
        kv_blocks_per_tp=kw.get("kv", {1: 2000, 2: 3000}),
        router=router,
        instance_cfg=dict(_INSTANCE_CFG),
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
    """Telemetry reads step_stats per instance, so a reset would corrupt every row."""
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
    # Identity, not equality, since the queues hold these exact objects.
    assert any(r is running for r in aborted)


def test_rebuild_counts_heterogeneous_plans():
    """Reported so a run can say whether heterogeneity actually engaged."""
    fleet, _ = _fleet(FleetPlan.uniform(tp=1, n_gpus=4))
    fleet.rebuild(FleetPlan.uniform(tp=1, n_gpus=4), t=1.0)
    fleet.rebuild(FleetPlan(tp_per_instance=(2, 1, 1), n_gpus=4), t=2.0)
    assert fleet.n_rebuilds == 2
    assert fleet.n_heterogeneous_plans == 1


def test_a_plan_needing_an_unsupplied_model_is_refused():
    router = Router(strategy=build_strategy("request_num_balance"), instances=[])
    with pytest.raises(ValueError, match="no cost model"):
        DynamicParallelFleet(
            cost_models={1: _cost()},
            kv_blocks_per_tp={1: 2000},
            router=router,
            instance_cfg=dict(_INSTANCE_CFG),
            initial_plan=FleetPlan(tp_per_instance=(2,), n_gpus=2),
        )


# --- Isolation from the simulator ---


def test_the_package_is_not_imported_unless_used():
    """The simulator must not depend on the fleet, which nothing constructs yet."""
    for name in list(sys.modules):
        if "dynamic_parallel" in name:
            del sys.modules[name]
    import importlib

    import dryrun.simulator.simulator as sim_module

    importlib.reload(sim_module)
    leaked = [n for n in sys.modules if "dynamic_parallel" in n]
    assert not leaked, f"importing the simulator pulled in {leaked}"
