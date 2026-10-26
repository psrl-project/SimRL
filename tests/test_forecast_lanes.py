"""The lane planner: assignment, KV-safe concurrency, pricing, and the upgrade search."""

from __future__ import annotations

import pytest

from srl.forecast.lanes import LaneJob, LaneSlot, kv_safe_concurrency, lpt_assign


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
