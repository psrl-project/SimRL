"""The LRU queue of unheld blocks, as a reversible doubly linked list.

Ported from vLLM's `FreeKVCacheBlockQueue`. Head is least recently freed, so
head is the eviction victim; releases append at the tail.

Why a linked list rather than an `OrderedDict`: a prefix-cache hit on a block
that is currently free removes it from the *middle* of the queue, and undoing
that means putting it back exactly where it was. The node's own `prev_free` /
`next_free` are that position, so the information an undo needs already exists.
`OrderedDict.pop(key)` discards it, and re-inserting at the tail would leave
`num_free` correct while silently rewriting LRU order -- invisible to
`blocks_used()` and decisive for every later eviction.

Sentinel head and tail nodes mean a real block's links are never `None` while it
is in the list, which removes the branching vLLM's own comment calls out.

**Rollback is a journal of inverse operations, not a snapshot of state.** Each
of the two mutations has an exact inverse -- unlinking a block is undone by
re-linking it between the neighbours it had, appending one is undone by
unlinking it -- and `UndoLog` already unwinds newest-first, which is precisely
the order those inverses compose in. So one entry per mutation suffices.

Snapshotting instead meant registering three nodes' link pairs plus the free
count per operation, deduplicated by key: four keyed captures per mutation,
41 million of them in a 16-instance run, and they were the single largest cost
in the engine. The journal is also *more* exactly correct, not less: a node
relinked several times within one event unwinds through each step rather than
relying on the first capture having been the right one.
"""

from __future__ import annotations

from ..undo import Recorder
from .block import KvBlock

_SENTINEL_ID = -1


class FreeBlockList:
    """Blocks with `ref_cnt == 0`, ordered least-recently-freed first.

    Args:
        blocks: Every block in the pool, in id order. All start free.
    """

    __slots__ = ("_head", "_tail", "_num_free")

    def __init__(self, blocks: list[KvBlock]) -> None:
        self._head = KvBlock(block_id=_SENTINEL_ID)
        self._tail = KvBlock(block_id=_SENTINEL_ID)
        self._num_free = len(blocks)

        previous = self._head
        for block in blocks:
            previous.next_free = block
            block.prev_free = previous
            previous = block
        previous.next_free = self._tail
        self._tail.prev_free = previous

    @property
    def num_free(self) -> int:
        return self._num_free

    def ids(self) -> list[int]:
        """Block ids from head to tail. For tests and diagnostics.

        Walks the list, so this is O(num_free) and must not be called from the
        engine's hot path. An undo test comparing LRU order is exactly the place
        it belongs: occupancy counts cannot detect a reordering.
        """
        out: list[int] = []
        node = self._head.next_free
        while node is not None and node is not self._tail:
            out.append(node.block_id)
            node = node.next_free
        return out

    def popleft(self, recorder: Recorder) -> KvBlock:
        """Take the least recently freed block.

        Args:
            recorder: Where to register the old links, so the block can be put
                back at the head rather than the tail.

        Returns:
            The block that was at the head.

        Raises:
            RuntimeError: If the list is empty. The caller is expected to have
                checked `num_free`; handing out a sentinel would corrupt the
                pool.
        """
        block = self._head.next_free
        if block is None or block is self._tail:
            raise RuntimeError("Cannot pop from an empty free list.")
        self._unlink(block, recorder)
        return block

    def append(self, block: KvBlock, recorder: Recorder) -> None:
        """Put a block at the tail, as the most recently freed."""
        last = self._tail.prev_free
        assert last is not None, "Sentinel tail always has a predecessor."
        last.next_free = block
        block.prev_free = last
        block.next_free = self._tail
        self._tail.prev_free = block
        self._num_free += 1
        recorder.record(_Appended(self, block))

    def remove(self, block: KvBlock, recorder: Recorder) -> None:
        """Take a block out of the list, wherever it sits.

        Raises:
            RuntimeError: If the block is not currently in the list. It would
                have no neighbours to unlink, and proceeding would corrupt the
                list silently.
        """
        if block.prev_free is None or block.next_free is None:
            raise RuntimeError(
                f"Block {block.block_id} is not in the free list, so it cannot "
                f"be removed from it."
            )
        self._unlink(block, recorder)

    def _unlink(self, block: KvBlock, recorder: Recorder) -> None:
        """Detach `block`, recording enough to re-link it in place."""
        previous, following = block.prev_free, block.next_free
        assert previous is not None and following is not None
        previous.next_free = following
        following.prev_free = previous
        block.prev_free = block.next_free = None
        self._num_free -= 1
        recorder.record(_Unlinked(self, block, previous, following))

    def _relink(self, block: KvBlock, previous: KvBlock, following: KvBlock) -> None:
        """Put `block` back between the neighbours it was unlinked from."""
        previous.next_free = block
        following.prev_free = block
        block.prev_free = previous
        block.next_free = following
        self._num_free += 1

    def _detach(self, block: KvBlock) -> None:
        """Remove `block` again, undoing an `append`."""
        previous, following = block.prev_free, block.next_free
        assert previous is not None and following is not None, (
            f"Block {block.block_id} left the free list between its append and "
            f"the rollback of that append, which the undo journal forbids."
        )
        previous.next_free = following
        following.prev_free = previous
        block.prev_free = block.next_free = None
        self._num_free -= 1


class _Unlinked:
    """Undoes one `_unlink` by re-linking the block between its old neighbours.

    A slotted class rather than a `(callable, args)` pair: both were measured in
    the full simulator, and the class is 1.6x faster (53 s against 84 s of CPU
    over three interleaved rounds), because binding a method allocates where
    constructing this does not. A microbenchmark of the two in isolation says
    the opposite, which is why the decision was made end to end.
    """

    __slots__ = ("_list", "_block", "_prev", "_next")

    def __init__(
        self,
        free_list: FreeBlockList,
        block: KvBlock,
        previous: KvBlock,
        following: KvBlock,
    ) -> None:
        self._list, self._block = free_list, block
        self._prev, self._next = previous, following

    def restore(self) -> None:
        self._list._relink(self._block, self._prev, self._next)


class _Appended:
    """Undoes one `append` by taking the block back out of the list."""

    __slots__ = ("_list", "_block")

    def __init__(self, free_list: FreeBlockList, block: KvBlock) -> None:
        self._list, self._block = free_list, block

    def restore(self) -> None:
        self._list._detach(self._block)
