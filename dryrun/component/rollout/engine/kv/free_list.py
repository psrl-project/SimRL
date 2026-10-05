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
"""

from __future__ import annotations

from ..undo import AttrSnapshot, Recorder
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
        self._record_links(block, recorder)
        self._record_links(last, recorder)
        self._record_links(self._tail, recorder)
        recorder.record(AttrSnapshot(self, ("_num_free",), (self._num_free,)))
        last.next_free = block
        block.prev_free = last
        block.next_free = self._tail
        self._tail.prev_free = block
        self._num_free += 1

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
        self._record_links(block, recorder)
        self._record_links(previous, recorder)
        self._record_links(following, recorder)
        recorder.record(AttrSnapshot(self, ("_num_free",), (self._num_free,)))
        previous.next_free = following
        following.prev_free = previous
        block.prev_free = block.next_free = None
        self._num_free -= 1

    @staticmethod
    def _record_links(block: KvBlock, recorder: Recorder) -> None:
        """Capture one node's links, once per event.

        Keyed on `id(block)` rather than `block_id` because the two sentinels
        share id -1, and they do get relinked.
        """
        recorder.record_once(
            ("links", id(block)),
            lambda: AttrSnapshot.capture(block, "prev_free", "next_free"),
        )
