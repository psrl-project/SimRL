"""Undo must restore the pool exactly, including what telemetry cannot see.

`instance/*.jsonl` records occupancy but not free-list order, eviction count or
the prefix-hit counters, so a run whose telemetry is byte-identical can still
have a corrupted pool. These tests compare the *whole* internal state across an
undo instead: LRU order, per-block refcounts and hashes, the hash map, the held
map, and the counters compared against `vllm:prefix_cache_*`.

The scenarios are driven to where the interesting cases live -- an
oversubscribed pool, so events preempt; siblings sharing a prompt, so refcounts
exceed one; and weight-update aborts, which reset the hash map. Each is run with
prefix caching both on and off.

These were written after a performance rework replaced several state snapshots
with inverse-operation journals. They are the check that rework is held to:
occupancy alone would have passed either way.
"""

from __future__ import annotations

import random

import pytest

from dryrun.component.rollout.engine import NativeInstance, Request
from dryrun.cost.rollout import UnifiedRoofline


def full_state(inst: NativeInstance) -> dict:
    """Every piece of state an event can touch, pool internals included."""
    pool = inst.kv.pool
    return {
        "t_local": inst.t_local,
        "waiting": [r.rid for r in inst.waiting],
        "running": [r.rid for r in inst.running],
        "completed": [r.rid for r in inst.completed],
        "n_preemptions": inst.n_preemptions,
        "last_decode_blocked": inst._last_decode_blocked,
        # The pool internals telemetry never records.
        "free_list_order": pool.free_list_ids(),
        "num_free": pool.num_free_blocks,
        "num_used": pool.num_used_blocks,
        "eviction_count": pool.eviction_count,
        "queries": inst.kv.queries,
        "hits": inst.kv.hits,
        "blocks": [
            (b.block_id, b.ref_cnt, b.block_hash)
            for b in pool.blocks
            if b.ref_cnt != 0 or b.block_hash is not None
        ],
        "hash_map": sorted(
            (str(k), v.block_id) for k, v in pool._hash_to_block.items()
        ),
        "held": sorted(
            (rid, [b.block_id for b in blocks])
            for rid, blocks in inst.kv._held.items()
        ),
        "n_cached": sorted(inst.kv._n_cached.items()),
        "n_step_stats": len(inst.step_stats),
        "reqs": sorted(
            (
                r.rid,
                r.prefilled,
                r.generated,
                str(r.status),
                r.num_preemptions,
                r.num_aborts,
                r.reprefill_tokens,
                r.prefix_hit_tokens,
                tuple((s.version, s.tokens) for s in r.segments),
            )
            for r in (*inst.waiting, *inst.running, *inst.completed)
        ),
    }


def _loaded_instance(seed: int, kv_blocks: int, caching: bool) -> NativeInstance:
    """An instance whose batch oversubscribes its pool, so events preempt."""
    rng = random.Random(seed)
    inst = NativeInstance(
        cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
        token_budget=rng.choice([2048, 4096, 12_800]),
        kv_blocks=kv_blocks,
        block_size=16,
        max_concurrency=rng.choice([8, 64, 1024]),
        enable_prefix_caching=caching,
    )
    budget = kv_blocks * 16 // 2
    for rid in range(24):
        prompt = min(rng.choice([140, 320, 1024]), budget // 2)
        target = max(64, min(rng.choice([200, 600, 1500]), budget - prompt))
        inst.add_request(
            Request(
                rid=rid,
                prompt_len=prompt,
                target_len=target,
                v_traj=0,
                # Siblings share `source_group_id`, which is what makes prompt
                # blocks shared and so drives refcounts above one.
                group_id=rid // 4,
                source_group_id=rid // 4,
            ),
            t=float(rid),
        )
    return inst


@pytest.mark.parametrize("kv_blocks", [200, 600, 2000])
@pytest.mark.parametrize("caching", [True, False])
def test_every_undo_restores_the_whole_pool(kv_blocks, caching):
    """Swept rather than spot-checked, and asserted after *every* event.

    An undo that is subtly wrong -- a block returned to the wrong free-list
    slot, a refcount restored once when it was taken twice -- shows up only in
    some pool states, so the sweep is what makes the check meaningful.

    `hash_map` is included. It was excluded while `MappingKeysAdded` popped
    overwritten keys instead of restoring them, and while `_HashMapEntry`
    resurrected pairs the map never held; with both fixed, the whole state
    round-trips and there is nothing left to carve out.
    """
    inst = _loaded_instance(seed=kv_blocks, kv_blocks=kv_blocks, caching=caching)
    rng = random.Random(kv_blocks)
    undone = 0
    for step in range(400):
        before = full_state(inst)
        if inst.advance_one_event(version=step % 3) is None:
            break
        inst.undo_last_event()
        assert full_state(inst) == before, f"undo diverged at step {step}"
        undone += 1

        if inst.advance_one_event(version=step % 3) is None:
            break
        inst.commit_last_event()

        if rng.random() < 0.05:
            # A weight update: abort everything and reset the hash map, which
            # is the largest single-event mutation the pool sees.
            before_abort = full_state(inst)
            inst._save_snapshot()
            inst.abort_all(requeue=True)
            inst.undo_last_event()
            assert full_state(inst) == before_abort, f"abort undo diverged at {step}"
            inst.abort_all(requeue=True)

        if not inst.running and not inst.waiting:
            break
    assert undone >= 8, f"only {undone} undos were exercised"


def test_undo_never_makes_a_hash_addressable_that_was_not():
    """An undone event must not add a `hash -> block` pair the map never held.

    `_maybe_evict` records a restorable pair only when the map actually named
    the block it is evicting. It is not always so: two siblings racing to cache
    the same prompt block leave the loser holding its hash while the map names
    the winner. Restoring the loser's pair unconditionally resurrected a key
    that had been absent, which over-reports prefix reuse on every later lookup.
    """
    for kv_blocks in (200, 600):
        inst = _loaded_instance(seed=kv_blocks, kv_blocks=kv_blocks, caching=True)
        for step in range(200):
            before = dict(inst.kv.pool._hash_to_block)
            if inst.advance_one_event(version=step % 3) is None:
                break
            inst.undo_last_event()
            gained = {
                key: block
                for key, block in inst.kv.pool._hash_to_block.items()
                if key not in before
            }
            assert not gained, (
                f"undo at step {step} (kv_blocks={kv_blocks}) made "
                f"{sorted(map(str, gained))} addressable; the map did not hold "
                f"them before the event"
            )
            if inst.advance_one_event(version=step % 3) is None:
                break
            inst.commit_last_event()
            if not inst.running and not inst.waiting:
                break


def test_undo_must_not_drop_a_hash_held_by_a_live_request():
    """The defect `MappingKeysChanged` was introduced to fix.

    When an event re-points an existing hash from block A to block B,
    `cache_full_blocks` writes over an entry that already had a value. The old
    `MappingKeysAdded` recorded only the key, so rollback popped it outright --
    even though block A still held that hash and was still held by a live
    request. The content stopped being addressable while the blocks backing it
    were still resident, so the simulated engine under-reported prefix reuse.

    Reproduced at seed 600 / 600 blocks, step 1: `('p', 4, 19)` moved from block
    243 to 297, and the undo left it absent while block 243 (ref_cnt 1, held by
    request 18) still carried it. `MappingKeysChanged.note` now captures the
    displaced value and restores it.
    """
    for kv_blocks in (200, 600):
        inst = _loaded_instance(seed=kv_blocks, kv_blocks=kv_blocks, caching=True)
        for step in range(200):
            before = dict(inst.kv.pool._hash_to_block)
            if inst.advance_one_event(version=step % 3) is None:
                break
            inst.undo_last_event()
            after = inst.kv.pool._hash_to_block
            for key, block in before.items():
                assert key in after, (
                    f"undo at step {step} (kv_blocks={kv_blocks}) dropped "
                    f"{key!r}, which block {block.block_id} "
                    f"(ref_cnt={block.ref_cnt}) still holds"
                )
            if inst.advance_one_event(version=step % 3) is None:
                break
            inst.commit_last_event()
            if not inst.running and not inst.waiting:
                break
            break
        inst.commit_last_event()
        if not inst.running and not inst.waiting:
            break



@pytest.mark.parametrize("caching", [True, False])
def test_repeated_undo_of_the_same_event_is_stable(caching):
    """Advance and roll back the same state repeatedly.

    A journal that restored a *captured* value would be stable here by
    construction; one that replays inverse operations is stable only if each
    entry is applied exactly once, so this is where a double-apply shows up.
    """
    inst = _loaded_instance(seed=7, kv_blocks=400, caching=caching)
    for _ in range(6):
        if inst.advance_one_event(version=0) is None:
            break
        inst.commit_last_event()

    before = full_state(inst)
    for attempt in range(8):
        if inst.advance_one_event(version=1) is None:
            pytest.skip("instance went idle before the repeated-undo check")
        inst.undo_last_event()
        assert full_state(inst) == before, f"state drifted on attempt {attempt}"
