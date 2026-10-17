"""Simulated counterparts of the four controlled prefix-cache scenarios.

These run the simulator's own block pool through the same four shapes
`srl/workload/prefix_probe_run.py` drives a real vLLM server through, so the
two sides of the comparison in {doc}`../docs/simulation/prefix-cache` are both
committed code rather than one of them being a transcript.

Expectations are arithmetic over `prompt_len` and `block_size`, registered
before measuring. Nothing here is fitted.
"""

from __future__ import annotations

import pytest

from srl.component.rollout.engine import NativeInstance, Request
from srl.cost.rollout import UnifiedRoofline

_BLOCK = 16
_PROMPT = 2048
_GROUP = 8


def _inst(kv_blocks: int = 100_000, **kwargs) -> NativeInstance:
    return NativeInstance(
        cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
        token_budget=65536,
        kv_blocks=kv_blocks,
        block_size=_BLOCK,
        **kwargs,
    )


def _req(rid: int, prompt_len: int, group: int, target_len: int = 16) -> Request:
    return Request(
        rid=rid,
        prompt_len=prompt_len,
        target_len=target_len,
        v_traj=0,
        group_id=group,
        source_group_id=group,
    )


def _drain(inst: NativeInstance, version: int = 0, cap: int = 5000) -> None:
    for _ in range(cap):
        if inst.advance_one_event(version=version) is None:
            return
    raise AssertionError(f"Instance did not drain within {cap} events.")


class TestWithinGroupSharedPrompt:
    """One prompt at n=8, the GRPO shape.

    Exercises a hit on blocks another live request currently holds.
    """

    def test_seven_of_eight_siblings_skip_the_whole_shared_prompt(self):
        inst = _inst()
        for rid in range(_GROUP):
            assert inst.add_request(_req(rid, _PROMPT, group=99), t=0.0)
        _drain(inst)

        # 2048 is a whole number of 16-token blocks, so every block of the
        # prompt hits -- less the one vLLM hands back so the step has work.
        expected = (_GROUP - 1) * (_PROMPT - _BLOCK)
        assert sum(r.prefix_hit_tokens for r in inst.completed) == expected

    def test_the_hit_rate_approaches_seven_eighths(self):
        inst = _inst()
        for rid in range(_GROUP):
            assert inst.add_request(_req(rid, _PROMPT, group=99), t=0.0)
        _drain(inst)

        rate = inst.kv.hits / inst.kv.queries
        # 7/8 = 0.875, less one handed-back block per hitting request.
        assert rate == pytest.approx(0.868, abs=0.01)

    def test_shared_blocks_are_counted_once_in_occupancy(self):
        inst = _inst()
        for rid in range(_GROUP):
            assert inst.add_request(_req(rid, _PROMPT, group=99), t=0.0)
        inst.advance_one_event(version=0)

        # The 128 shared prompt blocks, plus one private tail block for each
        # sibling that hit (the handed-back one it had to recompute).
        assert inst.blocks_used() == _PROMPT // _BLOCK + (_GROUP - 1)


class TestContinuedPrefix:
    """Round two's context is round one's prompt plus its output.

    The shape a resumed partial rollout and an agentic next round both take.
    Exercises a hit on blocks whose `ref_cnt` has already reached zero, which is
    the pool's central invariant.
    """

    def test_a_continued_context_hits_what_the_first_round_left(self):
        inst = _inst()
        first = _req(1, _PROMPT, group=7, target_len=64)
        assert inst.add_request(first, t=0.0)
        _drain(inst)
        context = first.prompt_len + first.generated

        queries_before, hits_before = inst.kv.queries, inst.kv.hits
        second = _req(1, context, group=7, target_len=64)
        assert inst.add_request(second, t=0.0)
        _drain(inst)

        hit_rate = (inst.kv.hits - hits_before) / (inst.kv.queries - queries_before)
        assert hit_rate > 0.9, "round one's context is still cached in full"

    def test_the_blocks_it_hits_are_unheld_at_the_time(self):
        inst = _inst()
        first = _req(1, _PROMPT, group=7, target_len=64)
        assert inst.add_request(first, t=0.0)
        _drain(inst)

        # Nothing holds any block now, yet the hashes are still addressable.
        assert inst.blocks_used() == 0
        assert inst.kv.pool.cached_block(("p", 7, 0)) is not None


class TestHitRateAfterReset:
    """`reset_prefix_cache`, then resend: the weight-sync path.

    The post-reset rate is not zero, which is the mechanism behind the 9%
    measured in `outputs/simulate/SMG_LOOPBACK_FIX.md`.
    """

    def test_a_reset_empties_the_map(self):
        inst = _inst()
        for rid in range(_GROUP):
            assert inst.add_request(_req(rid, _PROMPT, group=99), t=0.0)
        _drain(inst)
        assert inst.kv.pool.cached_block(("p", 99, 0)) is not None

        inst.kv.reset_hashes()
        assert inst.kv.pool.cached_block(("p", 99, 0)) is None

    def test_the_group_rebuilds_the_same_hit_rate_after_a_reset(self):
        inst = _inst()
        for rid in range(_GROUP):
            assert inst.add_request(_req(rid, _PROMPT, group=99), t=0.0)
        _drain(inst)
        inst.kv.reset_hashes()

        queries_before, hits_before = inst.kv.queries, inst.kv.hits
        for rid in range(_GROUP, 2 * _GROUP):
            assert inst.add_request(_req(rid, _PROMPT, group=99), t=0.0)
        _drain(inst, version=1)

        rate = (inst.kv.hits - hits_before) / (inst.kv.queries - queries_before)
        assert rate == pytest.approx(0.868, abs=0.01), (
            "the first request repopulates the prompt hashes and the rest hit them, "
            "so the rate recovers rather than staying at zero"
        )


class TestLruEviction:
    """Capacity squeezed until reuse must evict.

    The only scenario that reaches `BlockPool._maybe_evict`: KV never saturates
    under the project's main workload, so without this the eviction path would
    never be executed at all.
    """

    def test_revisiting_a_working_set_larger_than_the_pool_evicts(self):
        # 32 distinct prompts of 2048 tokens need 4096 blocks; give 2000.
        inst = _inst(kv_blocks=2000)
        for rid in range(32):
            if not inst.add_request(_req(rid, _PROMPT, group=rid), t=0.0):
                break
            _drain(inst)

        assert inst.kv.pool.eviction_count > 0, "the pool had to reuse hashed blocks"

    def test_evicted_prompts_no_longer_hit(self):
        inst = _inst(kv_blocks=2000)
        for rid in range(32):
            if not inst.add_request(_req(rid, _PROMPT, group=rid), t=0.0):
                break
            _drain(inst)

        # The earliest group was evicted to make room for later ones.
        assert inst.kv.pool.cached_block(("p", 0, 0)) is None

    def test_a_pool_with_room_to_spare_evicts_nothing(self):
        """The control: same workload, more capacity, no eviction. Without it,
        the test above could pass for the wrong reason."""
        inst = _inst(kv_blocks=100_000)
        for rid in range(32):
            assert inst.add_request(_req(rid, _PROMPT, group=rid), t=0.0)
            _drain(inst)

        assert inst.kv.pool.eviction_count == 0
        assert inst.kv.pool.cached_block(("p", 0, 0)) is not None
