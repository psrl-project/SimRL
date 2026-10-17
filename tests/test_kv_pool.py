"""Tests for the rollout KV block pool and provenance-derived block hashes."""

from __future__ import annotations

import pytest

from srl.component.rollout.engine import Request
from srl.component.rollout.engine.kv import (
    BlockPool,
    FreeBlockList,
    KvBlock,
    KvCacheManager,
    block_hashes,
)
from srl.component.rollout.engine.undo import NULL_RECORDER, UndoLog


def _req(rid: int, prompt_len: int, source_group_id: int, generated: int = 0) -> Request:
    req = Request(
        rid=rid,
        prompt_len=prompt_len,
        target_len=4096,
        v_traj=0,
        source_group_id=source_group_id,
    )
    req.generated = generated
    return req


class TestKvBlock:
    def test_starts_unheld_and_unhashed(self):
        block = KvBlock(block_id=7)
        assert block.block_id == 7
        assert block.ref_cnt == 0
        assert block.block_hash is None

    def test_refcount_tracks_holders(self):
        block = KvBlock(block_id=0)
        block.incr_ref()
        block.incr_ref()
        assert block.ref_cnt == 2
        block.decr_ref()
        assert block.ref_cnt == 1

    def test_reassigning_a_hash_is_a_bug_not_an_update(self):
        """One physical block holding two logical identities is the hardest bug
        class here, so assigning twice raises instead of silently overwriting."""
        block = KvBlock(block_id=0)
        block.block_hash = ("p", 42, 0)
        with pytest.raises(ValueError, match="already"):
            block.block_hash = ("p", 42, 1)

    def test_reset_hash_allows_reassignment(self):
        block = KvBlock(block_id=0)
        block.block_hash = ("p", 42, 0)
        block.reset_hash()
        assert block.block_hash is None
        block.block_hash = ("r", 9, 3)
        assert block.block_hash == ("r", 9, 3)


class TestFreeBlockList:
    def test_starts_with_every_block_in_order(self):
        blocks = [KvBlock(i) for i in range(4)]
        free = FreeBlockList(blocks)
        assert free.num_free == 4
        assert free.ids() == [0, 1, 2, 3]

    def test_popleft_takes_the_least_recently_freed(self):
        free = FreeBlockList([KvBlock(i) for i in range(3)])
        assert free.popleft(NULL_RECORDER).block_id == 0
        assert free.ids() == [1, 2]
        assert free.num_free == 2

    def test_append_puts_a_block_at_the_tail(self):
        blocks = [KvBlock(i) for i in range(3)]
        free = FreeBlockList(blocks)
        taken = free.popleft(NULL_RECORDER)
        free.append(taken, NULL_RECORDER)
        assert free.ids() == [1, 2, 0], "released blocks go to the back of the LRU queue"

    def test_remove_takes_a_block_out_of_the_middle(self):
        blocks = [KvBlock(i) for i in range(4)]
        free = FreeBlockList(blocks)
        free.remove(blocks[2], NULL_RECORDER)
        assert free.ids() == [0, 1, 3]
        assert free.num_free == 3

    def test_removing_a_block_that_is_not_in_the_list_raises(self):
        """A block outside the list has no neighbours to unlink, so proceeding
        would corrupt the list silently."""
        blocks = [KvBlock(i) for i in range(2)]
        free = FreeBlockList(blocks)
        free.remove(blocks[0], NULL_RECORDER)
        with pytest.raises(RuntimeError, match="not in the free list"):
            free.remove(blocks[0], NULL_RECORDER)

    def test_popleft_on_an_empty_list_raises(self):
        free = FreeBlockList([])
        with pytest.raises(RuntimeError, match="empty"):
            free.popleft(NULL_RECORDER)


class TestFreeListUndo:
    def test_undoing_popleft_restores_the_block_to_the_head(self):
        """Not the tail. Putting it back in the wrong place leaves `num_free`
        correct while silently rewriting LRU order, which `blocks_used()` cannot
        see and which decides every later eviction."""
        blocks = [KvBlock(i) for i in range(4)]
        free = FreeBlockList(blocks)
        log = UndoLog()
        free.popleft(log)
        log.rollback()
        assert free.ids() == [0, 1, 2, 3]
        assert free.num_free == 4

    def test_undoing_remove_restores_the_exact_position(self):
        blocks = [KvBlock(i) for i in range(5)]
        free = FreeBlockList(blocks)
        log = UndoLog()
        free.remove(blocks[2], log)
        log.rollback()
        assert free.ids() == [0, 1, 2, 3, 4]

    def test_undoing_append_removes_the_block_again(self):
        blocks = [KvBlock(i) for i in range(3)]
        free = FreeBlockList(blocks)
        taken = free.popleft(NULL_RECORDER)
        log = UndoLog()
        free.append(taken, log)
        log.rollback()
        assert free.ids() == [1, 2]
        assert free.num_free == 2

    def test_a_mixed_event_restores_the_whole_order(self):
        """One event can pop, remove and append. Rollback unwinds in reverse,
        so the list must come back byte for byte."""
        blocks = [KvBlock(i) for i in range(6)]
        free = FreeBlockList(blocks)
        before = free.ids()
        log = UndoLog()
        free.popleft(log)
        free.remove(blocks[3], log)
        free.append(blocks[0], log)
        assert free.ids() != before
        log.rollback()
        assert free.ids() == before

    def test_null_recorder_leaves_no_way_back(self):
        """Committed paths pass NULL_RECORDER, so the mutation stands."""
        blocks = [KvBlock(i) for i in range(3)]
        free = FreeBlockList(blocks)
        free.popleft(NULL_RECORDER)
        assert free.ids() == [1, 2]


class TestBlockHashes:
    def test_only_full_blocks_get_a_hash(self):
        """vLLM's own rule: a partially filled block is not cacheable. At
        prompt_len=150 with block_size=16 that is 9 blocks over 144 tokens."""
        ids = block_hashes(_req(rid=1, prompt_len=150, source_group_id=42), block_size=16)
        assert len(ids) == 9

    def test_prompt_interior_blocks_key_on_the_shared_prompt(self):
        ids = block_hashes(_req(rid=1, prompt_len=150, source_group_id=42), block_size=16)
        assert ids[0] == ("p", 42, 0)
        assert ids[8] == ("p", 42, 8)

    def test_siblings_share_every_prompt_block(self):
        """Eight responses to one prompt are what GRPO samples, and the whole
        point of this module is that they pay for the prompt once."""
        a = block_hashes(_req(rid=1, prompt_len=150, source_group_id=42), block_size=16)
        b = block_hashes(_req(rid=2, prompt_len=150, source_group_id=42), block_size=16)
        assert a == b

    def test_different_prompts_share_nothing(self):
        a = block_hashes(_req(rid=1, prompt_len=150, source_group_id=42), block_size=16)
        b = block_hashes(_req(rid=2, prompt_len=150, source_group_id=43), block_size=16)
        assert set(a).isdisjoint(b)

    def test_the_block_straddling_the_prompt_end_is_private(self):
        """prompt_len=150 leaves block 9 covering tokens 144-159, of which only
        144-149 are prompt. Its content differs per response, so it cannot be
        shared even though most of its tokens are."""
        ids = block_hashes(
            _req(rid=7, prompt_len=150, source_group_id=42, generated=20), block_size=16
        )
        assert ids[8] == ("p", 42, 8)
        assert ids[9] == ("r", 7, 9)

    def test_generated_blocks_are_private_to_their_request(self):
        a = block_hashes(
            _req(rid=1, prompt_len=160, source_group_id=42, generated=64), block_size=16
        )
        b = block_hashes(
            _req(rid=2, prompt_len=160, source_group_id=42, generated=64), block_size=16
        )
        assert a[:10] == b[:10]
        assert set(a[10:]).isdisjoint(b[10:])

    def test_a_prompt_that_is_a_whole_number_of_blocks_shares_all_of_them(self):
        ids = block_hashes(_req(rid=1, prompt_len=160, source_group_id=42), block_size=16)
        assert len(ids) == 10
        assert all(h[0] == "p" for h in ids)

    def test_continued_context_reproduces_its_own_earlier_hashes(self):
        """A request resumed after a weight update re-prefills prompt+generated.
        Its generated blocks must hash exactly as they did when first filled,
        or resumption would never hit."""
        first = block_hashes(
            _req(rid=5, prompt_len=160, source_group_id=42, generated=64), block_size=16
        )
        later = block_hashes(
            _req(rid=5, prompt_len=160, source_group_id=42, generated=128), block_size=16
        )
        assert later[: len(first)] == first

    def test_a_context_shorter_than_one_block_has_no_hashes(self):
        assert block_hashes(_req(rid=1, prompt_len=10, source_group_id=1), block_size=16) == []


class TestBlockPoolAllocation:
    def test_starts_entirely_free(self):
        pool = BlockPool(num_blocks=10)
        assert pool.num_free_blocks == 10
        assert pool.num_used_blocks == 0
        assert pool.usage == 0.0

    def test_new_blocks_are_held_and_removed_from_free(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(3, NULL_RECORDER)
        assert len(blocks) == 3
        assert all(b.ref_cnt == 1 for b in blocks)
        assert pool.num_free_blocks == 7
        assert pool.usage == 0.3

    def test_overallocation_raises_rather_than_overcommitting(self):
        pool = BlockPool(num_blocks=2)
        with pytest.raises(ValueError, match="free"):
            pool.new_blocks(3, NULL_RECORDER)

    def test_freeing_returns_blocks_only_when_the_last_holder_leaves(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(2, NULL_RECORDER)
        pool.touch(blocks, NULL_RECORDER)  # a second holder
        pool.free_blocks(blocks, NULL_RECORDER)
        assert pool.num_free_blocks == 8, "still held by the second holder"
        pool.free_blocks(blocks, NULL_RECORDER)
        assert pool.num_free_blocks == 10


class TestBlockPoolHits:
    def test_a_cached_hash_is_findable(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(2, NULL_RECORDER)
        pool.cache_full_blocks(
            blocks, [("p", 1, 0), ("p", 1, 1)], n_cached=0, n_full=2, recorder=NULL_RECORDER
        )
        assert pool.cached_block(("p", 1, 0)) is blocks[0]
        assert pool.cached_block(("p", 1, 1)) is blocks[1]

    def test_an_unknown_hash_misses(self):
        assert BlockPool(num_blocks=10).cached_block(("p", 1, 0)) is None

    def test_a_freed_block_keeps_its_hash_and_still_hits(self):
        """The pool's central invariant. Reuse after a weight update and across
        an agentic round boundary both happen on blocks whose ref_cnt is zero
        but whose hash is still mapped."""
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(
            blocks, [("p", 1, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER
        )
        pool.free_blocks(blocks, NULL_RECORDER)
        assert pool.num_free_blocks == 10
        assert pool.cached_block(("p", 1, 0)) is blocks[0]

    def test_touch_revives_a_free_block_without_double_counting(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(
            blocks, [("p", 1, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER
        )
        pool.free_blocks(blocks, NULL_RECORDER)
        hit = pool.cached_block(("p", 1, 0))
        pool.touch([hit], NULL_RECORDER)
        assert hit.ref_cnt == 1
        assert pool.num_free_blocks == 9

    def test_cache_full_blocks_skips_already_cached_prefix(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(3, NULL_RECORDER)
        pool.cache_full_blocks(
            blocks,
            [("p", 1, 0), ("p", 1, 1), ("p", 1, 2)],
            n_cached=2,
            n_full=3,
            recorder=NULL_RECORDER,
        )
        assert pool.cached_block(("p", 1, 0)) is None
        assert pool.cached_block(("p", 1, 2)) is blocks[2]

    def test_caching_is_a_no_op_when_disabled(self):
        pool = BlockPool(num_blocks=10, enable_caching=False)
        blocks = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(
            blocks, [("p", 1, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER
        )
        assert pool.cached_block(("p", 1, 0)) is None

    def test_reset_hashes_clears_the_map_but_not_occupancy(self):
        """A weight update calls reset_prefix_cache, which empties the block
        hash map. Blocks a live request still holds stay held."""
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(2, NULL_RECORDER)
        pool.cache_full_blocks(
            blocks, [("p", 1, 0), ("p", 1, 1)], n_cached=0, n_full=2, recorder=NULL_RECORDER
        )
        pool.reset_hashes(NULL_RECORDER)
        assert pool.cached_block(("p", 1, 0)) is None
        assert pool.num_free_blocks == 8
        assert blocks[0].block_hash is None


class TestBlockPoolEviction:
    def test_reuse_evicts_the_oldest_freed_block_first(self):
        pool = BlockPool(num_blocks=2)
        first = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(
            first, [("p", 1, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER
        )
        second = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(
            second, [("p", 2, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER
        )
        pool.free_blocks(first, NULL_RECORDER)
        pool.free_blocks(second, NULL_RECORDER)
        # Both are free and both still hit. Allocating one must take the
        # least recently freed, which is `first`.
        pool.new_blocks(1, NULL_RECORDER)
        assert pool.cached_block(("p", 1, 0)) is None, "oldest freed was evicted"
        assert pool.cached_block(("p", 2, 0)) is second[0], "newer one survived"
        assert pool.eviction_count == 1

    def test_eviction_frees_the_hash_for_reassignment(self):
        pool = BlockPool(num_blocks=1)
        blocks = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(
            blocks, [("p", 1, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER
        )
        pool.free_blocks(blocks, NULL_RECORDER)
        reused = pool.new_blocks(1, NULL_RECORDER)
        assert reused[0].block_hash is None
        pool.cache_full_blocks(
            reused, [("p", 9, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER
        )
        assert pool.cached_block(("p", 9, 0)) is reused[0]


def _pool_state(pool: BlockPool) -> tuple:
    """Everything an undo has to restore.

    Occupancy alone is not enough: a block put back in the wrong free-list slot
    leaves `num_used_blocks` right while rewriting LRU order, so the list order
    and every block's refcount and hash are all part of the state.
    """
    return (
        pool.num_used_blocks,
        pool.free_list_ids(),
        pool.eviction_count,
        dict(pool._hash_to_block),
        [(b.ref_cnt, b.block_hash) for b in pool.blocks],
    )


class TestBlockPoolUndo:
    def test_undoing_allocation_restores_occupancy_and_order(self):
        pool = BlockPool(num_blocks=10)
        before = _pool_state(pool)
        log = UndoLog()
        pool.new_blocks(3, log)
        log.rollback()
        assert _pool_state(pool) == before

    def test_undoing_a_hit_restores_the_free_list_position(self):
        """A hit removes a block from the middle of the LRU queue. Putting it
        back at the tail would keep occupancy correct and the eviction order
        wrong, which is why the list is linked rather than an OrderedDict."""
        pool = BlockPool(num_blocks=6)
        blocks = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(
            blocks, [("p", 1, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER
        )
        pool.free_blocks(blocks, NULL_RECORDER)
        before = _pool_state(pool)

        log = UndoLog()
        pool.touch([pool.cached_block(("p", 1, 0))], log)
        log.rollback()
        assert _pool_state(pool) == before

    def test_undoing_eviction_restores_the_hash_and_the_counter(self):
        """eviction_count feeds telemetry as eviction pressure, so an
        uncommitted event must not inflate it permanently."""
        pool = BlockPool(num_blocks=1)
        blocks = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(
            blocks, [("p", 1, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER
        )
        pool.free_blocks(blocks, NULL_RECORDER)
        before = _pool_state(pool)

        log = UndoLog()
        pool.new_blocks(1, log)
        assert pool.cached_block(("p", 1, 0)) is None and pool.eviction_count == 1
        log.rollback()
        assert _pool_state(pool) == before

    def test_undoing_caching_removes_the_hashes_it_added(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(2, NULL_RECORDER)
        before = _pool_state(pool)
        log = UndoLog()
        pool.cache_full_blocks(
            blocks, [("p", 1, 0), ("p", 1, 1)], n_cached=0, n_full=2, recorder=log
        )
        log.rollback()
        assert _pool_state(pool) == before

    def test_undoing_reset_hashes_restores_the_whole_map(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(2, NULL_RECORDER)
        pool.cache_full_blocks(
            blocks, [("p", 1, 0), ("p", 1, 1)], n_cached=0, n_full=2, recorder=NULL_RECORDER
        )
        before = _pool_state(pool)

        log = UndoLog()
        pool.reset_hashes(log)
        assert pool.cached_block(("p", 1, 0)) is None
        log.rollback()
        assert _pool_state(pool) == before

    def test_undoing_a_mixed_event_restores_every_field(self):
        """One event can hit, allocate, cache and free. This is the test that
        would catch a mutation someone forgot to record."""
        pool = BlockPool(num_blocks=8)
        blocks = pool.new_blocks(2, NULL_RECORDER)
        pool.cache_full_blocks(
            blocks, [("p", 1, 0), ("p", 1, 1)], n_cached=0, n_full=2, recorder=NULL_RECORDER
        )
        pool.free_blocks(blocks, NULL_RECORDER)
        before = _pool_state(pool)

        log = UndoLog()
        pool.touch([pool.cached_block(("p", 1, 0))], log)
        fresh = pool.new_blocks(3, log)
        pool.cache_full_blocks(
            fresh, [("r", 9, 0), ("r", 9, 1), ("r", 9, 2)], n_cached=0, n_full=3, recorder=log
        )
        pool.free_blocks(fresh, log)
        assert _pool_state(pool) != before
        log.rollback()
        assert _pool_state(pool) == before

    def test_null_recorder_leaves_the_mutation_standing(self):
        pool = BlockPool(num_blocks=10)
        pool.new_blocks(3, NULL_RECORDER)
        assert pool.num_used_blocks == 3


class TestComputedBlocks:
    def test_a_cold_manager_hits_nothing(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        blocks, tokens = mgr.computed_blocks(_req(rid=1, prompt_len=150, source_group_id=42))
        assert blocks == []
        assert tokens == 0

    def test_a_sibling_hits_every_shared_prompt_block(self):
        """prompt_len=150 shares 9 full blocks, so 144 of 150 prompt tokens are
        free for the second response onward. This is the limitation in
        docs/reference/limitations.md that reads 'Prompt prefill charged once
        per response'."""
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        first = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(first, n_tokens=150, hit_blocks=[])

        sibling = _req(rid=2, prompt_len=150, source_group_id=42)
        blocks, tokens = mgr.computed_blocks(sibling)
        assert len(blocks) == 9
        assert tokens == 144

    def test_hits_stop_at_the_first_miss(self):
        """Hashes carry chained semantics, so a miss at block j means content
        has diverged and later blocks must not be used even if they collide."""
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        # Cache only a 32-token prefix of group 42.
        short = _req(rid=1, prompt_len=32, source_group_id=42)
        mgr.allocate(short, n_tokens=32, hit_blocks=[])

        blocks, tokens = mgr.computed_blocks(
            _req(rid=2, prompt_len=160, source_group_id=42)
        )
        assert len(blocks) == 2
        assert tokens == 32

    def test_a_full_hit_hands_the_last_block_back(self):
        """vLLM's own edge case
        (vllm_v1_engine_replica_scheduler.py:1393-1399): a full hit leaves zero
        new tokens, but the model must forward at least one token to produce
        the next logits. Fires for real whenever prompt_len is a multiple of
        block_size."""
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        first = _req(rid=1, prompt_len=160, source_group_id=42)
        mgr.allocate(first, n_tokens=160, hit_blocks=[])

        sibling = _req(rid=2, prompt_len=160, source_group_id=42)
        blocks, tokens = mgr.computed_blocks(sibling)
        assert len(blocks) == 9, "ten blocks match, but the last is handed back"
        assert tokens == 144
        assert sibling.prompt_len - tokens == 16

    def test_counters_track_queries_and_hits_in_blocks(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        first = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(first, n_tokens=150, hit_blocks=[])
        mgr.computed_blocks(_req(rid=2, prompt_len=150, source_group_id=42))
        assert mgr.queries == 9, "only the sibling queried; allocate does not"
        assert mgr.hits == 9

    def test_disabled_caching_never_hits(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16, enable_caching=False)
        first = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(first, n_tokens=150, hit_blocks=[])
        blocks, tokens = mgr.computed_blocks(_req(rid=2, prompt_len=150, source_group_id=42))
        assert blocks == []
        assert tokens == 0

    def test_peek_does_not_move_the_counters(self):
        """`_step_time_estimate` predicts a step without performing it, so it
        must not perturb the counters compared against vllm:prefix_cache_*."""
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        first = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(first, n_tokens=150, hit_blocks=[])
        before = (mgr.queries, mgr.hits)
        tokens = mgr.peek_hit_tokens(_req(rid=2, prompt_len=150, source_group_id=42))
        assert tokens == 144
        assert (mgr.queries, mgr.hits) == before


class TestAllocation:
    def test_allocation_accounts_occupancy(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        req = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(req, n_tokens=150, hit_blocks=[])
        assert mgr.num_used_blocks == 10, "ceil(150/16)"
        assert mgr.blocks_held(req) == 10

    def test_a_hit_sibling_only_allocates_its_own_tail(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        first = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(first, n_tokens=150, hit_blocks=[])
        used_after_first = mgr.num_used_blocks

        sibling = _req(rid=2, prompt_len=150, source_group_id=42)
        hit_blocks, hit_tokens = mgr.computed_blocks(sibling)
        sibling.prefilled = hit_tokens
        mgr.allocate(sibling, n_tokens=sibling.prompt_len - hit_tokens, hit_blocks=hit_blocks)
        assert mgr.num_used_blocks == used_after_first + 1, "only the private tail block"

    def test_can_allocate_refuses_what_does_not_fit(self):
        mgr = KvCacheManager(num_blocks=4, block_size=16)
        req = _req(rid=1, prompt_len=150, source_group_id=42)
        assert not mgr.can_allocate(req, n_tokens=150, hit_blocks=[])
        assert mgr.allocate(req, n_tokens=150, hit_blocks=[]) is None

    def test_free_releases_everything_the_request_held(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        req = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(req, n_tokens=150, hit_blocks=[])
        mgr.free(req)
        assert mgr.num_used_blocks == 0
        assert mgr.blocks_held(req) == 0

    def test_a_resumed_request_hits_its_own_generated_blocks(self):
        """Partial rollout: an abort keeps `generated` and zeroes `prefilled`,
        so the context must be re-prefilled. Its generated blocks hash exactly
        as they did when first filled, so they hit."""
        mgr = KvCacheManager(num_blocks=1000, block_size=16)
        req = _req(rid=5, prompt_len=160, source_group_id=42, generated=320)
        mgr.allocate(req, n_tokens=480, hit_blocks=[])
        mgr.free(req)

        req.prefilled = 0
        blocks, tokens = mgr.computed_blocks(req)
        assert tokens == 464, "30 full blocks hit, less the handed-back last one"
