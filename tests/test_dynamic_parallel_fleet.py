"""The dynamic-parallel package: `SlotTable` invariants and `SlotFleet` reshapes."""

from __future__ import annotations

import sys

import pytest

from srl.component.rollout.engine import NativeInstance, Request
from srl.component.rollout.engine.state import InstanceLoad
from srl.component.rollout.router import Router
from srl.component.rollout.router.strategy import build_strategy
from srl.cost import UnifiedRoofline
from srl.simulator.dynamic_parallel import Slot, SlotFleet, SlotTable

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


# --- SlotTable ---


def test_from_degrees_tiles_every_degree_disjointly():
    table = SlotTable.from_degrees([1, 2], n_gpus=4)
    tp1 = sorted(s.gpus for s in table.slots if s.tp == 1)
    tp2 = sorted(s.gpus for s in table.slots if s.tp == 2)
    assert tp1 == [frozenset({0}), frozenset({1}), frozenset({2}), frozenset({3})]
    assert tp2 == [frozenset({0, 1}), frozenset({2, 3})]


def test_from_degrees_rejects_a_degree_that_does_not_divide_n_gpus():
    with pytest.raises(ValueError, match="does not divide"):
        SlotTable.from_degrees([3], n_gpus=4)


def test_uniform_awake_is_gpu_disjoint():
    table = SlotTable.from_degrees([1, 2], n_gpus=4)
    awake = table.uniform_awake(1)
    assert len(awake) == 4
    assert table.awake_is_valid(awake)


def test_awake_is_valid_rejects_overlap():
    table = SlotTable.from_degrees([1, 2], n_gpus=4)
    tp1_slot = next(s for s in table.slots if s.tp == 1)
    overlapping_tp2 = next(s for s in table.slots if s.tp == 2 and s.gpus & tp1_slot.gpus)
    assert not table.awake_is_valid(frozenset({tp1_slot.slot_id, overlapping_tp2.slot_id}))


def test_describe_formats_the_awake_shape():
    table = SlotTable.from_degrees([1, 2], n_gpus=4)
    tp2 = next(s.slot_id for s in table.slots if s.tp == 2)
    tp1 = next(s.slot_id for s in table.slots if s.tp == 1 and not (s.gpus & table.by_id[tp2].gpus))
    assert table.describe(frozenset({tp2, tp1})) == "1xTP2 + 1xTP1"


def test_slot_rejects_a_gpu_count_mismatched_with_tp():
    with pytest.raises(ValueError, match="spans 1 GPUs"):
        Slot(slot_id=0, tp=2, gpus=frozenset({0}))


# --- TP on the load snapshot ---


def test_get_load_propagates_tp():
    inst = NativeInstance(cost=_cost(), token_budget=1000, kv_blocks=500, tp=2)
    assert inst.get_load().tp == 2


def test_tp_is_a_declared_field_on_the_instance_view():
    import dataclasses

    assert "tp" in {f.name for f in dataclasses.fields(InstanceLoad)}


# --- SlotFleet ---


def _fleet(table: SlotTable, awake: frozenset[int], **kw) -> tuple[SlotFleet, Router]:
    router = Router(strategy=build_strategy("request_num_balance"), instances=[])
    fleet = SlotFleet(
        table=table,
        cost_models={1: _cost(), 2: _cost()},
        kv_blocks_per_tp=kw.get("kv", {1: 2000, 2: 3000}),
        router=router,
        instance_cfg=dict(_INSTANCE_CFG),
        initial_awake=awake,
        t=kw.get("t", 0.0),
    )
    return fleet, router


def test_the_fleet_builds_one_instance_per_slot_never_just_the_awake_ones():
    table = SlotTable.from_degrees([1, 2], n_gpus=4)
    fleet, _ = _fleet(table, table.uniform_awake(1))
    assert len(fleet.instances) == len(table.slots)
    assert {inst.instance_id for inst in fleet.instances} == set(table.by_id)


def test_instance_positions_never_move_across_a_reshape():
    """Simulator indexes self.instances[event.instance_id] directly; a reshape must not
    renumber or reorder the full list."""
    table = SlotTable.from_degrees([1, 2], n_gpus=4)
    fleet, _ = _fleet(table, table.uniform_awake(1))
    before = [(inst.instance_id, inst.tp) for inst in fleet.instances]
    fleet.reshape(table.uniform_awake(2), t=5.0)
    after = [(inst.instance_id, inst.tp) for inst in fleet.instances]
    assert before == after


def test_router_only_sees_awake_instances():
    table = SlotTable.from_degrees([1, 2], n_gpus=4)
    fleet, router = _fleet(table, table.uniform_awake(1))
    assert router.awake == table.uniform_awake(1)
    assert {inst.instance_id for inst in router.instances} == set(table.by_id)
    fleet.reshape(table.uniform_awake(2), t=5.0)
    assert router.awake == table.uniform_awake(2)
    # `router.instances` stays the full, dense list across every reshape.
    assert {inst.instance_id for inst in router.instances} == set(table.by_id)


def test_sleeping_slots_are_excluded_by_awake_but_present_in_instances():
    table = SlotTable.from_degrees([1, 2], n_gpus=4)
    fleet, router = _fleet(table, table.uniform_awake(1))
    asleep_tp2 = {sid for sid in table.by_id if table.by_id[sid].tp == 2}
    assert asleep_tp2.isdisjoint(router.awake)
    assert all(sid in {i.instance_id for i in router.instances} for sid in asleep_tp2)
    assert all(sid in {i.instance_id for i in fleet.instances} for sid in asleep_tp2)


def test_reshape_rejects_an_overlapping_awake_set():
    table = SlotTable.from_degrees([1, 2], n_gpus=4)
    fleet, _ = _fleet(table, table.uniform_awake(1))
    tp1_ids = {s.slot_id for s in table.slots if s.tp == 1}
    tp2_ids = {s.slot_id for s in table.slots if s.tp == 2}
    with pytest.raises(ValueError, match="not GPU-disjoint"):
        fleet.reshape(frozenset(tp1_ids | {next(iter(tp2_ids))}), t=1.0)


def test_reshape_preserves_history_on_slots_that_stay_awake():
    """Telemetry reads step_stats per instance; a slot that never leaves the awake set must
    keep it across an unrelated reshape elsewhere in the fleet."""
    table = SlotTable.from_degrees([1], n_gpus=4)
    fleet, _ = _fleet(table, table.uniform_awake(1), kv={1: 2000})
    for inst in fleet.instances:
        inst.step_stats.append(f"hist-{inst.instance_id}")
    fleet.reshape(table.uniform_awake(1), t=5.0)
    assert [inst.step_stats for inst in fleet.instances] == [[f"hist-{i}"] for i in range(4)]


def test_reshape_never_rewinds_a_staying_slots_clock():
    table = SlotTable.from_degrees([1], n_gpus=2)
    fleet, _ = _fleet(table, table.uniform_awake(1), kv={1: 2000})
    fleet.instances[0].t_local = 20.0
    fleet.reshape(table.uniform_awake(1), t=5.0)
    assert fleet.instances[0].t_local == 20.0


def test_reshape_charges_reconfig_time_only_to_newly_woken_slots():
    table = SlotTable.from_degrees([1, 2], n_gpus=2)
    fleet, _ = _fleet(table, table.uniform_awake(1))
    fleet.reshape(table.uniform_awake(2), t=10.0, reconfig_time_s=5.0)
    tp2_id = next(s.slot_id for s in table.slots if s.tp == 2)
    assert fleet.instances[tp2_id].t_local == 15.0


def test_reshape_does_not_charge_reconfig_time_to_a_slot_that_stays_awake():
    table = SlotTable.from_degrees([1], n_gpus=2)
    fleet, _ = _fleet(table, table.uniform_awake(1), kv={1: 2000})
    fleet.reshape(table.uniform_awake(1), t=10.0, reconfig_time_s=5.0)
    assert fleet.instances[0].t_local == 10.0


def test_reshape_returns_work_from_leaving_slots_both_queues():
    table = SlotTable.from_degrees([1, 2], n_gpus=2)
    fleet, _ = _fleet(table, table.uniform_awake(1))
    tp1_id = next(s.slot_id for s in table.slots if s.tp == 1)
    other_tp1_id = next(s.slot_id for s in table.slots if s.tp == 1 and s.slot_id != tp1_id)
    running, waiting = _req(0, rid=1), _req(0, rid=2)
    fleet.instances[tp1_id].running.append(running)
    fleet.instances[other_tp1_id].waiting.append(waiting)
    aborted = fleet.reshape(table.uniform_awake(2), t=1.0)
    assert {r.rid for r in aborted} == {1, 2}
    assert any(r is running for r in aborted)


def test_reshape_counts_heterogeneous_awakes():
    table = SlotTable.from_degrees([1, 2], n_gpus=4)
    fleet, _ = _fleet(table, table.uniform_awake(1))
    fleet.reshape(table.uniform_awake(1), t=1.0)
    tp2_slot = next(s for s in table.slots if s.tp == 2)
    untouched_tp1 = {s.slot_id for s in table.slots if s.tp == 1 and not (s.gpus & tp2_slot.gpus)}
    mixed = frozenset({tp2_slot.slot_id} | untouched_tp1)
    fleet.reshape(mixed, t=2.0)
    assert fleet.n_reshapes == 2
    assert fleet.n_heterogeneous_awakes == 1


def test_a_fleet_needing_an_unsupplied_cost_model_is_refused():
    table = SlotTable.from_degrees([1, 2], n_gpus=2)
    router = Router(strategy=build_strategy("request_num_balance"), instances=[])
    with pytest.raises(ValueError, match="no cost model"):
        SlotFleet(
            table=table,
            cost_models={1: _cost()},
            kv_blocks_per_tp={1: 2000, 2: 3000},
            router=router,
            instance_cfg=dict(_INSTANCE_CFG),
            initial_awake=table.uniform_awake(1),
        )


def test_a_fleet_needing_unsupplied_kv_capacity_is_refused():
    table = SlotTable.from_degrees([1, 2], n_gpus=2)
    router = Router(strategy=build_strategy("request_num_balance"), instances=[])
    with pytest.raises(ValueError, match="no KV capacity"):
        SlotFleet(
            table=table,
            cost_models={1: _cost(), 2: _cost()},
            kv_blocks_per_tp={1: 2000},
            router=router,
            instance_cfg=dict(_INSTANCE_CFG),
            initial_awake=table.uniform_awake(1),
        )


def test_eviction_strategy_reaches_every_slot():
    table = SlotTable.from_degrees([1], n_gpus=2)
    router = Router(strategy=build_strategy("request_num_balance"), instances=[])
    sentinel = object()
    cfg = dict(_INSTANCE_CFG, eviction=sentinel)
    fleet = SlotFleet(
        table=table,
        cost_models={1: _cost()},
        kv_blocks_per_tp={1: 2000},
        router=router,
        instance_cfg=cfg,
        initial_awake=table.uniform_awake(1),
    )
    assert all(inst.eviction is sentinel for inst in fleet.instances)


# --- Isolation from the simulator ---


def test_the_package_is_importable_without_pulling_in_the_simulator_eagerly():
    """Importing `srl.simulator.dynamic_parallel` must not require `srl.simulator.simulator`
    to already be configured for it -- only a run that opts in via `SimConfig.dynamic_parallel`
    should touch this package."""
    for name in list(sys.modules):
        if "dynamic_parallel" in name:
            del sys.modules[name]
    import importlib

    import srl.simulator.dynamic_parallel as dp_module

    importlib.reload(dp_module)
