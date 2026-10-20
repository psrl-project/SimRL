"""The LRU queue of unheld blocks, as a reversible doubly linked list.

Ported from vLLM's `FreeKVCacheBlockQueue`. Head is least recently freed, so
head is the eviction victim. Releases append at the tail.

A linked list, not an `OrderedDict`, because a prefix-cache hit can remove a
free block from the middle of the queue, and undoing that means re-linking it
at its old position. The node's own `prev_free` / `next_free` hold that
position. `OrderedDict.pop(key)` discards it.

Sentinel head and tail nodes mean a real block's links are never `None` while
it is in the list, avoiding a branch on every link update.

Rollback is a journal of inverse operations, not a snapshot of state:
unlinking a block is undone by re-linking it between its old neighbours, and
appending one is undone by unlinking it. `UndoLog` unwinds newest-first,
which is exactly the order those inverses compose in, so one entry per
mutation suffices.
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
                checked `num_free`. Handing out a sentinel would corrupt the
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

    A slotted class rather than a `(callable, args)` pair, since binding a
    method allocates per call where constructing this does not.
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
