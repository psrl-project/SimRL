"""The KV block pool: allocation, LRU reuse, and eviction.

Ported from vLLM v1 (`vllm/v1/core/block_pool.py`) rather than approximated,
because what this has to reproduce is an engine's exact admission and reuse
behaviour under pressure, and that behaviour lives in the mechanism.

The central invariant: **a block may sit in the free list and in the hash map
at the same time.** That is the "nobody holds it, but it is still addressable"
state. Reuse across a weight update and across an agentic round boundary both
land on exactly those blocks, so a pool that dropped hashes on release would
be wrong in the two cases that matter most here.

A hash is destroyed only when its block is handed to a new owner
(`new_blocks` reaching `_maybe_evict`), never on release.

Every mutating method takes a `Recorder`. Occupancy, hashes and list order are
all registered through it, so `undo_last_event` can unwind an uncommitted event
without the pool ever being copied. `NULL_RECORDER` is what committed paths pass.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Hashable, Optional

from ..undo import MappingKeysChanged, Recorder
from .block import KvBlock
from .free_list import FreeBlockList


class BlockPool:
    """A fixed set of KV blocks, with prefix-cache bookkeeping.

    Args:
        num_blocks: Pool capacity, from `srl/cost/rollout/kv_cache.py`.
        enable_caching: When False, blocks are still allocated and freed but no
            hash is ever recorded, so nothing can hit. This is a data state,
            not a second code path.

    Raises:
        ValueError: If `num_blocks` is negative.
    """

    def __init__(self, num_blocks: int, enable_caching: bool = True) -> None:
        if num_blocks < 0:
            raise ValueError(f"Expected num_blocks >= 0, got {num_blocks!r}.")
        self.num_blocks = int(num_blocks)
        self.enable_caching = bool(enable_caching)
        self.blocks = [KvBlock(block_id=i) for i in range(self.num_blocks)]
        self._free = FreeBlockList(self.blocks)
        self._hash_to_block: dict[Hashable, KvBlock] = {}
        self.eviction_count = 0

    @property
    def num_free_blocks(self) -> int:
        return self._free.num_free

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self._free.num_free

    @property
    def usage(self) -> float:
        """Fraction of the pool currently held, 0.0 for an empty pool."""
        if self.num_blocks <= 0:
            return 0.0
        return self.num_used_blocks / self.num_blocks

    def free_list_ids(self) -> list[int]:
        """Free block ids, least recently freed first. Diagnostics and tests.

        O(num_free), so not for the hot path. Undo tests need it because
        occupancy counts cannot detect a reordering of the LRU queue.
        """
        return self._free.ids()

    def cached_block(self, block_hash: Hashable) -> Optional[KvBlock]:
        """The block holding `block_hash`, or None on a miss."""
        return self._hash_to_block.get(block_hash)

    def new_blocks(self, n: int, recorder: Recorder) -> list[KvBlock]:
        """Take `n` unheld blocks, evicting their hashes as they change hands.

        Args:
            n: How many blocks to allocate.
            recorder: Where to register old values for rollback.

        Returns:
            The allocated blocks, each with `ref_cnt == 1`.

        Raises:
            ValueError: If fewer than `n` blocks are free. The caller is
                expected to have checked capacity; overcommitting the pool
                would silently invent memory the engine does not have.
        """
        if n > self.num_free_blocks:
            raise ValueError(
                f"Cannot allocate {n} KV blocks: only {self.num_free_blocks} free."
            )
        taken: list[KvBlock] = []
        for _ in range(n):
            block = self._free.popleft(recorder)
            # Evict before taking the reference, and let `_maybe_evict` record
            # the hash it drops. Rollback unwinds newest-first, so the reference
            # is released before the hash comes back -- the exact reverse of the
            # order here.
            self._maybe_evict(block, recorder)
            block.incr_ref()
            recorder.record(_RefIncremented(block))
            taken.append(block)
        return taken

    def touch(self, blocks: list[KvBlock], recorder: Recorder) -> None:
        """Take a reference to blocks found by hash.

        A hit block may currently be free, in which case it has to leave the
        free list before anyone can grow into it. That removal is from the middle
        of the list, which is the case the linked list exists to make reversible.
        """
        for block in blocks:
            if block.ref_cnt == 0:
                self._free.remove(block, recorder)
            block.incr_ref()
            recorder.record(_RefIncremented(block))

    def free_blocks(self, blocks: Iterable[KvBlock], recorder: Recorder) -> None:
        """Release a reference, returning blocks whose last holder left.

        The hash is deliberately kept: a released block stays addressable so a
        later request with the same prefix can hit it.
        """
        for block in blocks:
            block.decr_ref()
            recorder.record(_RefDecremented(block))
            if block.ref_cnt == 0:
                self._free.append(block, recorder)

    def cache_full_blocks(
        self,
        blocks: list[KvBlock],
        hashes: list[Hashable],
        n_cached: int,
        n_full: int,
        recorder: Recorder,
        hash_offset: int = 0,
    ) -> None:
        """Record hashes for blocks that have become full.

        Args:
            blocks: This request's blocks, in block order.
            hashes: Block hashes for this request, covering block indices
                `hash_offset` onwards.
            n_cached: How many leading blocks are already recorded.
            n_full: How many leading blocks are now full.
            recorder: Where to register old values for rollback.
            hash_offset: Block index `hashes[0]` stands for. Non-zero lets a
                caller build only the hashes of the range about to be cached,
                which is O(new blocks) rather than O(context).
        """
        if not self.enable_caching or n_cached >= n_full:
            return
        changed = MappingKeysChanged.capture(self._hash_to_block)
        recorder.record(changed)
        last = min(n_full, hash_offset + len(hashes), len(blocks))
        for index in range(max(n_cached, hash_offset), last):
            block = blocks[index]
            if block.block_hash is None:
                block_hash = hashes[index - hash_offset]
                # Noted *before* the write, so the entry records what this key
                # displaced. A hash already mapped to another block is the case
                # that matters: the previous block keeps holding it, so rollback
                # has to put the old pair back rather than delete the key.
                changed.note(block_hash)
                block.block_hash = block_hash
                self._hash_to_block[block_hash] = block
                # The block held no hash a line ago, so the inverse is to clear
                # it. `changed` unwinds the map; this unwinds the block itself.
                recorder.record(_HashAssigned(block))

    def reset_hashes(self, recorder: Recorder) -> None:
        """Empty the hash map, leaving occupancy untouched.

        This is vLLM's `reset_prefix_cache`, which a weight update triggers via
        PSRL's `pull_model`. Blocks a live request still holds stay held; they
        simply stop being addressable by content.
        """
        if not self._hash_to_block:
            return
        # One entry for the whole map: it already pairs every hash with the
        # block that held it, which is exactly what restoring both sides needs.
        recorder.record(_HashMapRestore(self, dict(self._hash_to_block)))
        for block in self._hash_to_block.values():
            block.reset_hash()
        self._hash_to_block.clear()

    def _maybe_evict(self, block: KvBlock, recorder: Recorder) -> bool:
        """Drop a block's hash as it is handed to a new owner."""
        if block.block_hash is None:
            return False
        evicted_hash = block.block_hash
        # Restore the *pair* only when this block is the one the map names.
        # A hash can be held by a block the map no longer points at: two
        # siblings racing to cache the same prompt block leave the loser holding
        # its hash while the map names the winner. Re-inserting unconditionally
        # resurrected the loser's pair on rollback, making a key addressable
        # that had been absent before the event. The block's own hash is
        # restored either way -- it is cleared either way.
        if self._hash_to_block.get(evicted_hash) is block:
            recorder.record(_HashMapEntry(self._hash_to_block, evicted_hash, block))
            self._hash_to_block.pop(evicted_hash, None)
        else:
            recorder.record(_HashCleared(block, evicted_hash))
        recorder.record_attrs_once(("evictions", id(self)), self, "eviction_count")
        block.reset_hash()
        self.eviction_count += 1
        return True


class _RefIncremented:
    """Undoes one `incr_ref`.

    The refcount is reversed by decrementing rather than by restoring a captured
    value, which is what lets several references taken within one event unwind
    independently. Capturing instead meant one keyed `(ref_cnt, _block_hash)`
    snapshot per touched block per event -- 12.4 million of them in a
    16-instance run.
    """

    __slots__ = ("_block",)

    def __init__(self, block: KvBlock) -> None:
        self._block = block

    def restore(self) -> None:
        self._block.decr_ref()


class _RefDecremented:
    """Undoes one `decr_ref`."""

    __slots__ = ("_block",)

    def __init__(self, block: KvBlock) -> None:
        self._block = block

    def restore(self) -> None:
        self._block.incr_ref()


class _HashAssigned:
    """Undoes assigning a hash to a block that held none."""

    __slots__ = ("_block",)

    def __init__(self, block: KvBlock) -> None:
        self._block = block

    def restore(self) -> None:
        self._block.reset_hash()


class _HashCleared:
    """Restores a block's own hash, without touching the map.

    The eviction case where the map named some *other* block for this hash, so
    only the block side was changed. Restoring the map too would resurrect a
    pair that did not exist.
    """

    __slots__ = ("_block", "_key")

    def __init__(self, block: KvBlock, key: Hashable) -> None:
        self._block, self._key = block, key

    def restore(self) -> None:
        self._block.reset_hash()
        self._block.block_hash = self._key


class _HashMapEntry:
    """Restores one evicted `hash -> block` pair, map side and block side.

    Both sides, because eviction clears both: `_hash_to_block.pop` and
    `block.reset_hash()`. `block_hash`'s setter refuses to overwrite, so the
    block is cleared first -- a later entry in the same event may already have
    assigned a new hash to this very block, and unwinding has to pass through
    that state rather than collide with it.

    Recorded only when the map actually named this block; see `_maybe_evict`.
    """

    __slots__ = ("_mapping", "_key", "_block")

    def __init__(self, mapping: dict, key: Hashable, block: KvBlock) -> None:
        self._mapping, self._key, self._block = mapping, key, block

    def restore(self) -> None:
        self._mapping[self._key] = self._block
        self._block.reset_hash()
        self._block.block_hash = self._key


class _HashMapRestore:
    """Restores the whole hash map after a `reset_hashes`.

    This is the one entry whose size is proportional to the map rather than to
    the event, and it is correct for it to be: `reset_hashes` *is* an
    event-sized operation. It fires once per weight update, not per step.

    Each block's own `_block_hash` is restored here too, from the same mapping:
    `reset_hashes` clears both sides, and the map already pairs every hash with
    the block that held it.
    """

    __slots__ = ("_pool", "_entries")

    def __init__(self, pool: "BlockPool", entries: dict) -> None:
        self._pool, self._entries = pool, entries

    def restore(self) -> None:
        self._pool._hash_to_block.update(self._entries)
        for block_hash, block in self._entries.items():
            block.reset_hash()
            block.block_hash = block_hash
