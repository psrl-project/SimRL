"""The lane planner: assignment, KV-safe concurrency, pricing, and the upgrade search."""

from __future__ import annotations

import dataclasses
import math

import pytest
from psrl.workers.ps.staleness import BufferView, InventorySnapshot

from srl.cost.rollout import UnifiedRoofline
from srl.forecast import TrainState
from srl.forecast.lanes import (
    LaneJob,
    LaneSlot,
    _candidate_moves,
    kv_safe_concurrency,
    lpt_assign,
    plan_lanes,
    price,
    state_for,
)


def job(rid: int, remaining: int, group_id: int = 0, prompt_len: int = 100) -> LaneJob:
    return LaneJob(
        rid=rid,
        group_id=group_id,
        prompt_key=group_id,
        prompt_len=prompt_len,
        generated=0,
        remaining=remaining,
    )


def slot(instance_id: int, kv_blocks: int = 1000) -> LaneSlot:
    return LaneSlot(
        instance_id=instance_id,
        cost=None,
        block_size=16,
        kv_blocks=kv_blocks,
        token_budget=8192,
        max_concurrency=None,
        tp=1,
    )


class TestLaneJob:
    def test_final_ctx_sums_prompt_generated_and_remaining(self):
        assert LaneJob(0, 0, 0, prompt_len=100, generated=50, remaining=200).final_ctx == 350

    def test_remaining_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="at least one token"):
            LaneJob(0, 0, 0, prompt_len=100, generated=0, remaining=0)


class TestLptAssign:
    def test_longest_job_goes_to_the_first_slot(self):
        got = lpt_assign([job(0, 100), job(1, 300), job(2, 200)], [7, 9])
        assert got[1] == 7
        assert got[2] == 9

    def test_balances_total_remaining_across_slots(self):
        jobs = [job(i, length) for i, length in enumerate([500, 400, 300, 200, 100])]
        got = lpt_assign(jobs, [0, 1])
        load = {0: 0, 1: 0}
        for j in jobs:
            load[got[j.rid]] += j.remaining
        assert abs(load[0] - load[1]) <= 100

    def test_pinned_jobs_keep_their_slot_and_count_as_load(self):
        jobs = [job(0, 1000), job(1, 10)]
        got = lpt_assign(jobs, [0, 1], pinned={0: 0})
        assert got[0] == 0
        assert got[1] == 1

    def test_ties_break_on_rid_so_the_result_is_deterministic(self):
        first = lpt_assign([job(0, 100), job(1, 100)], [0, 1])
        second = lpt_assign([job(1, 100), job(0, 100)], [0, 1])
        assert first == second

    def test_no_slots_is_rejected(self):
        with pytest.raises(ValueError, match="at least one slot"):
            lpt_assign([job(0, 100)], [])


class TestKvSafeConcurrency:
    def test_every_job_fits_when_the_pool_is_large(self):
        jobs = [job(i, 1000) for i in range(4)]
        assert kv_safe_concurrency(jobs, slot(0, kv_blocks=1000)) == 4

    def test_counts_only_the_jobs_whose_final_contexts_fit(self):
        # Each job's final ctx is 100 + 1500 = 1600 tokens = 100 blocks at block_size 16.
        jobs = [job(i, 1500) for i in range(10)]
        assert kv_safe_concurrency(jobs, slot(0, kv_blocks=250)) == 2

    def test_at_least_one_job_is_always_allowed(self):
        jobs = [job(0, 100_000)]
        assert kv_safe_concurrency(jobs, slot(0, kv_blocks=1)) == 1

    def test_no_jobs_means_no_concurrency(self):
        assert kv_safe_concurrency([], slot(0)) == 0


ROOFLINE = UnifiedRoofline(F=0.006, W=0.002, G=4e-6, A_p=3e-10, A_d=5e-8, b=16)


def priced_slot(instance_id: int, kv_blocks: int = 100_000) -> LaneSlot:
    return LaneSlot(
        instance_id=instance_id,
        cost=ROOFLINE,
        block_size=16,
        kv_blocks=kv_blocks,
        token_budget=8192,
        max_concurrency=None,
        tp=1,
    )


class TestStateFor:
    def test_jobs_become_waiting_residents_with_no_kv_computed(self):
        state = state_for(priced_slot(0), [job(0, 100), job(1, 200)], now=5.0)
        assert state.now == 5.0
        assert state.running == ()
        assert {r.key for r in state.waiting} == {0, 1}
        assert all(r.prefilled == 0 for r in state.waiting)

    def test_concurrency_is_capped_by_the_pool(self):
        # Final ctx 100 + 1500 = 1600 tokens = 100 blocks each. A 250-block pool holds two.
        jobs = [job(i, 1500) for i in range(10)]
        assert state_for(priced_slot(0, kv_blocks=250), jobs, now=0.0).max_concurrency == 2

    def test_the_slot_cap_wins_when_it_is_tighter_than_the_pool(self):
        jobs = [job(i, 100) for i in range(10)]
        tight = dataclasses.replace(priced_slot(0), max_concurrency=3)
        assert state_for(tight, jobs, now=0.0).max_concurrency == 3


class TestPrice:
    def snapshot_one_buffer(self, group_ids):
        reserved = tuple((group_id, 0) for group_id in group_ids)
        return InventorySnapshot(
            staleness=0,
            num_entries=len(group_ids),
            next_buffer_id=1,
            buffers=(BufferView(buffer_id=0, ready_num_entries=len(group_ids), n_occupied=0, reserved=reserved),),
        )

    def test_the_objective_is_the_training_end_of_the_horizon(self):
        jobs = [job(0, 500, group_id=0), job(1, 500, group_id=1)]
        slots = [priced_slot(0), priced_slot(1)]
        pricing = price(
            {0: 0, 1: 1},
            jobs,
            slots,
            snapshot=self.snapshot_one_buffer([0, 1]),
            train=TrainState(training_end=None, now=0.0, cycle=10.0, weight_load=0.0),
            members={0: [0], 1: [1]},
            finished={},
            now=0.0,
            horizon_buffers=2,
        )
        assert pricing.pipeline.buffers == (0,)
        assert pricing.objective == pytest.approx(pricing.pipeline.train_end[0])
        assert pricing.objective > 10.0

    def test_splitting_two_long_jobs_beats_stacking_them(self):
        # NOTE(claude): Remaining must be long enough that stacked ctxsum clears ROOFLINE's F
        # floor. Below that, both scenarios are floor-bound and tie regardless of concurrency.
        jobs = [job(0, 100_000, group_id=0), job(1, 100_000, group_id=1)]
        slots = [priced_slot(0), priced_slot(1)]
        common = dict(
            snapshot=self.snapshot_one_buffer([0, 1]),
            train=TrainState(training_end=None, now=0.0, cycle=10.0, weight_load=0.0),
            members={0: [0], 1: [1]},
            finished={},
            now=0.0,
            horizon_buffers=2,
        )
        split = price({0: 0, 1: 1}, jobs, slots, **common)
        stacked = price({0: 0, 1: 0}, jobs, slots, **common)
        assert split.objective < stacked.objective

    def test_a_finished_member_does_not_need_a_placement(self):
        jobs = [job(0, 500, group_id=0)]
        pricing = price(
            {0: 0},
            jobs,
            [priced_slot(0)],
            snapshot=self.snapshot_one_buffer([0]),
            train=TrainState(training_end=None, now=0.0, cycle=10.0, weight_load=0.0),
            members={0: [0, 99]},
            finished={99: 3.0},
            now=0.0,
            horizon_buffers=2,
        )
        assert math.isfinite(pricing.group_eta[0])

    def test_an_unplaced_member_makes_the_objective_infinite(self):
        jobs = [job(0, 500, group_id=0)]
        pricing = price(
            {0: 0},
            jobs,
            [priced_slot(0)],
            snapshot=self.snapshot_one_buffer([0]),
            train=TrainState(training_end=None, now=0.0, cycle=10.0, weight_load=0.0),
            members={0: [0, 99]},
            finished={},
            now=0.0,
            horizon_buffers=2,
        )
        assert pricing.objective == math.inf

    def test_an_assignment_naming_an_unknown_slot_is_rejected(self):
        with pytest.raises(ValueError, match="unknown slot"):
            price(
                {0: 5},
                [job(0, 500, group_id=0)],
                [priced_slot(0)],
                snapshot=self.snapshot_one_buffer([0]),
                train=TrainState(training_end=None, now=0.0, cycle=10.0, weight_load=0.0),
                members={0: [0]},
                finished={},
                now=0.0,
                horizon_buffers=2,
            )


class TestPlanLanes:
    def fleet(self, n: int):
        return [priced_slot(i) for i in range(n)]

    def snapshot_two_buffers(self, early_groups, late_groups):
        """Buffer 0 holds `early_groups`, buffer 1 holds `late_groups`, staleness 1."""
        return InventorySnapshot(
            staleness=1,
            num_entries=max(len(early_groups), len(late_groups)),
            next_buffer_id=2,
            buffers=(
                BufferView(
                    buffer_id=0,
                    ready_num_entries=max(len(early_groups), len(late_groups)),
                    n_occupied=0,
                    reserved=tuple((g, 0) for g in early_groups),
                ),
                BufferView(
                    buffer_id=1,
                    ready_num_entries=max(len(early_groups), len(late_groups)),
                    n_occupied=0,
                    reserved=tuple((g, 1) for g in late_groups),
                ),
            ),
        )

    def test_a_balanced_fleet_needs_no_upgrade(self):
        jobs = [job(i, 2000, group_id=i) for i in range(4)]
        plan = plan_lanes(
            jobs,
            self.fleet(4),
            snapshot=self.snapshot_one_buffer_n(4),
            train=TrainState(training_end=None, now=0.0, cycle=1e6, weight_load=0.0),
            members={i: [i] for i in range(4)},
            finished={},
            now=0.0,
        )
        assert plan.moves == 0
        assert plan.fast_instances == frozenset()

    def test_a_straggler_is_moved_onto_a_slot_of_its_own(self):
        # Four jobs on two slots forces LPT to pair two per slot. The straggler's pair is
        # slower than consolidating the other three onto one slot, so the search isolates it.
        jobs = [job(i, L, group_id=i) for i, L in enumerate([550_000, 340_000, 320_000, 290_000])]
        plan = plan_lanes(
            jobs,
            self.fleet(2),
            snapshot=self.snapshot_one_buffer_n(4),
            train=TrainState(training_end=None, now=0.0, cycle=5.0, weight_load=0.0),
            members={i: [i] for i in range(4)},
            finished={},
            now=0.0,
            c_max=1,
        )
        straggler_slot = plan.assignment[0]
        assert [rid for rid, s in plan.assignment.items() if s == straggler_slot] == [0]
        assert straggler_slot in plan.fast_instances

    def test_the_plan_never_loses_to_the_bulk_only_assignment(self):
        jobs = [job(0, 30_000, group_id=0)] + [job(i, 2_000, group_id=i) for i in range(1, 8)]
        slots = self.fleet(4)
        common = dict(
            snapshot=self.snapshot_one_buffer_n(8),
            train=TrainState(training_end=None, now=0.0, cycle=5.0, weight_load=0.0),
            members={i: [i] for i in range(8)},
            finished={},
            now=0.0,
        )
        plan = plan_lanes(jobs, slots, **common)
        bulk = price(lpt_assign(jobs, [s.instance_id for s in slots]), jobs, slots, horizon_buffers=2, **common)
        assert plan.objective <= bulk.objective

    def test_max_moves_bounds_the_search(self):
        jobs = [job(i, 30_000, group_id=i) for i in range(6)]
        plan = plan_lanes(
            jobs,
            self.fleet(6),
            snapshot=self.snapshot_one_buffer_n(6),
            train=TrainState(training_end=None, now=0.0, cycle=5.0, weight_load=0.0),
            members={i: [i] for i in range(6)},
            finished={},
            now=0.0,
            max_moves=1,
        )
        assert plan.moves <= 1

    def test_pinned_jobs_are_never_moved(self):
        jobs = [job(0, 30_000, group_id=0)] + [job(i, 1_000, group_id=i) for i in (1, 2)]
        plan = plan_lanes(
            jobs,
            self.fleet(2),
            snapshot=self.snapshot_one_buffer_n(3),
            train=TrainState(training_end=None, now=0.0, cycle=5.0, weight_load=0.0),
            members={i: [i] for i in range(3)},
            finished={},
            now=0.0,
            pinned={0: 1},
        )
        assert plan.assignment[0] == 1

    def test_a_pinned_coresident_of_the_eviction_host_never_moves(self):
        # Job 1 is pinned to slot 1 alongside free job 2. The mover (job 0) settles on slot
        # 1 as host, and only job 2, not pinned job 1, may be displaced.
        by_rid = {j.rid: j for j in (job(0, 600), job(1, 50), job(2, 50))}
        candidates = _candidate_moves(
            movers=[0],
            assignment={0: 0, 1: 1, 2: 1},
            by_rid=by_rid,
            slot_ids=[0, 1],
            fast=set(),
            c_max=1,
            pinned={1: 1},
        )
        assert len(candidates) == 1
        candidate, _fast = candidates[0]
        assert candidate[1] == 1
        assert set(candidate) == {0, 1, 2}

    def snapshot_one_buffer_n(self, n):
        return InventorySnapshot(
            staleness=0,
            num_entries=n,
            next_buffer_id=1,
            buffers=(
                BufferView(buffer_id=0, ready_num_entries=n, n_occupied=0, reserved=tuple((g, 0) for g in range(n))),
            ),
        )
