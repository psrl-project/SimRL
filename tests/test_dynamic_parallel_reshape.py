"""`ResidentThresholdPlanner` deciding target awake sets from occupancy snapshots."""

from __future__ import annotations

import pytest

from srl.simulator.dynamic_parallel import ResidentThresholdPlanner, SlotTable
from srl.simulator.dynamic_parallel.planner import ReshapeState


def _slot(table: SlotTable, tp: int, index: int = 0):
    return sorted((s for s in table.slots if s.tp == tp), key=lambda s: s.slot_id)[index]


def test_rejects_a_split_threshold_not_above_merge_threshold():
    with pytest.raises(ValueError, match="must exceed"):
        ResidentThresholdPlanner(merge_below=4.0, split_above=4.0)


def test_no_change_when_nothing_crosses_a_threshold():
    table = SlotTable.from_degrees([1, 2], n_gpus=4)
    planner = ResidentThresholdPlanner(merge_below=4.0, split_above=20.0, min_dwell_s=0.0)
    awake = table.uniform_awake(1)
    state = ReshapeState(t=0.0, awake=awake, residents={sid: 8 for sid in awake})
    assert planner.plan(table, state) is None


def test_merges_a_drained_sibling_pair_into_the_next_degree():
    table = SlotTable.from_degrees([1, 2], n_gpus=4)
    planner = ResidentThresholdPlanner(merge_below=4.0, split_above=20.0, min_dwell_s=0.0)
    awake = table.uniform_awake(1)
    tp1 = sorted((s for s in table.slots if s.tp == 1), key=lambda s: s.slot_id)
    residents = {s.slot_id: 8 for s in tp1}
    residents[tp1[0].slot_id] = 1
    residents[tp1[1].slot_id] = 2
    state = ReshapeState(t=0.0, awake=awake, residents=residents)
    result = planner.plan(table, state)
    assert result is not None
    merged_parent = next(s for s in table.slots if s.tp == 2 and s.gpus == tp1[0].gpus | tp1[1].gpus)
    assert merged_parent.slot_id in result
    assert tp1[0].slot_id not in result and tp1[1].slot_id not in result
    # every other slot stays exactly as it was
    assert result - {merged_parent.slot_id} == awake - {tp1[0].slot_id, tp1[1].slot_id}


def test_does_not_merge_unless_every_sibling_is_below_threshold():
    table = SlotTable.from_degrees([1, 2], n_gpus=2)
    planner = ResidentThresholdPlanner(merge_below=4.0, split_above=20.0, min_dwell_s=0.0)
    awake = table.uniform_awake(1)
    tp1 = sorted((s for s in table.slots if s.tp == 1), key=lambda s: s.slot_id)
    residents = {tp1[0].slot_id: 1, tp1[1].slot_id: 8}
    state = ReshapeState(t=0.0, awake=awake, residents=residents)
    assert planner.plan(table, state) is None


def test_splits_an_overloaded_merged_slot_back_down():
    table = SlotTable.from_degrees([1, 2], n_gpus=2)
    planner = ResidentThresholdPlanner(merge_below=4.0, split_above=20.0, min_dwell_s=0.0)
    awake = table.uniform_awake(2)
    tp2 = _slot(table, 2)
    state = ReshapeState(t=0.0, awake=awake, residents={tp2.slot_id: 25})
    result = planner.plan(table, state)
    assert result is not None
    tp1_children = {s.slot_id for s in table.slots if s.tp == 1}
    assert result == tp1_children


def test_a_merge_respects_the_max_tp_ceiling():
    table = SlotTable.from_degrees([1, 2, 4], n_gpus=4)
    planner = ResidentThresholdPlanner(merge_below=4.0, split_above=20.0, min_dwell_s=0.0, max_tp=2)
    awake = table.uniform_awake(1)
    residents = {sid: 1 for sid in awake}
    state = ReshapeState(t=0.0, awake=awake, residents=residents)
    result = planner.plan(table, state)
    assert result is not None
    assert all(table.by_id[sid].tp <= 2 for sid in result)


def test_min_dwell_suppresses_a_second_change_too_soon():
    table = SlotTable.from_degrees([1, 2], n_gpus=2)
    planner = ResidentThresholdPlanner(merge_below=4.0, split_above=20.0, min_dwell_s=10.0)
    awake = table.uniform_awake(1)
    tp1 = sorted((s for s in table.slots if s.tp == 1), key=lambda s: s.slot_id)
    residents = {s.slot_id: 1 for s in tp1}
    state = ReshapeState(t=0.0, awake=awake, residents=residents)
    first = planner.plan(table, state)
    assert first is not None

    tp2 = _slot(table, 2)
    overloaded = ReshapeState(t=1.0, awake=frozenset({tp2.slot_id}), residents={tp2.slot_id: 25})
    assert planner.plan(table, overloaded) is None  # too soon after the first change

    later = ReshapeState(t=11.0, awake=frozenset({tp2.slot_id}), residents={tp2.slot_id: 25})
    assert planner.plan(table, later) is not None


def test_merge_is_tried_before_split_in_the_same_call():
    """A planner given both an overloaded slot and a drained pair elsewhere must not thrash:
    pick one change and let the next call see the other."""
    table = SlotTable.from_degrees([1, 2], n_gpus=4)
    planner = ResidentThresholdPlanner(merge_below=4.0, split_above=20.0, min_dwell_s=0.0)
    tp2 = _slot(table, 2, 0)
    tp1_drained = [s for s in table.slots if s.tp == 1 and not (s.gpus & tp2.gpus)]
    awake = frozenset({tp2.slot_id} | {s.slot_id for s in tp1_drained})
    residents = {tp2.slot_id: 25, **{s.slot_id: 1 for s in tp1_drained}}
    state = ReshapeState(t=0.0, awake=awake, residents=residents)
    result = planner.plan(table, state)
    assert result is not None
    # the drained pair merged; the overloaded TP2 slot is untouched by this call
    assert tp2.slot_id in result
