"""One slot in the KV block pool.

A block is identified by `block_id` and tracks three things: how many live
requests hold it (`ref_cnt`), which content it holds (`block_hash`), and where
it sits in the free list (`prev_free` / `next_free`).

The first two are deliberately independent. A block whose `ref_cnt` has fallen
to zero keeps its hash, so a later request with the same prefix can still find
and reuse it. That state -- free but still addressable -- is what makes reuse
across a weight update or an agentic round boundary possible at all.

The link fields live here rather than inside the list because they are what
makes a removal reversible: re-linking a block needs the neighbours it had, and
storing them on the node means the "old value" an undo wants is already
materialised. This is also why vLLM keeps them, and why swapping the list for an
`OrderedDict` would not work: `pop(key)` discards position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional


@dataclass(eq=False)
class KvBlock:
    """A pool slot, with a refcount, an optional content hash, and list links.

    `eq=False` keeps identity-based `__eq__` / `__hash__`, as `Request` does.
    A generated `__eq__` would compare `prev_free` and `next_free`, recursing
    through the whole free list and overflowing the stack on the first
    comparison.

    Args:
        block_id: Index of this block in the pool. The sentinel head and tail of
            a `FreeBlockList` use -1, and are never handed out.
    """

    block_id: int
    ref_cnt: int = 0
    _block_hash: Optional[Hashable] = None
    prev_free: Optional["KvBlock"] = None
    next_free: Optional["KvBlock"] = None

    def incr_ref(self) -> None:
        self.ref_cnt += 1

    def decr_ref(self) -> None:
        self.ref_cnt -= 1

    @property
    def block_hash(self) -> Optional[Hashable]:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, value: Hashable) -> None:
        """Assign this block's content hash.

        Raises:
            ValueError: If a hash is already assigned. Overwriting would mean
                one physical block standing for two different prefixes, which
                corrupts every later hit silently rather than loudly.
        """
        if self._block_hash is not None:
            raise ValueError(
                f"Block {self.block_id} already holds hash {self._block_hash!r}; "
                f"call reset_hash() before assigning {value!r}."
            )
        self._block_hash = value

    def reset_hash(self) -> None:
        """Drop this block's content hash, making it assignable again."""
        self._block_hash = None

    def __repr__(self) -> str:
        # The default dataclass repr would walk `prev_free` / `next_free` and
        # print the whole list from any one block.
        return (
            f"KvBlock(block_id={self.block_id}, ref_cnt={self.ref_cnt}, "
            f"block_hash={self._block_hash!r})"
        )
