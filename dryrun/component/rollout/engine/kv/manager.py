"""Per-instance KV cache manager: the engine's view of the block pool.

Mirrors vLLM v1's `KVCacheManager`. `NativeInstance` talks to this and never to
`BlockPool` directly, so occupancy accounting and hit lookup stay in one place.

Every mutator takes a `recorder`, defaulting to `NULL_RECORDER` so that tests
and committed paths need not pass one. `BlockPool`'s equivalents require it:
the pool is internal and should never be called without a decision about
rollback, while this is the engine-facing surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..undo import NULL_RECORDER, Recorder
from .block import KvBlock
from .lineage import block_hashes
from .pool import BlockPool

if TYPE_CHECKING:
    from ..request import Request


def _blocks_for(n_tokens: int, block_size: int) -> int:
    """Blocks needed to hold `n_tokens`, rounding up."""
    return (n_tokens + block_size - 1) // block_size


class KvCacheManager:
    """Allocation and prefix lookup for one engine instance.

    Args:
        num_blocks: Pool capacity.
        block_size: Tokens per block.
        enable_caching: When False no hash is recorded, so nothing hits.

    Attributes:
        queries: Blocks looked up since construction, counted per block to
            match vLLM's `vllm:prefix_cache_queries`.
        hits: Blocks found, matching `vllm:prefix_cache_hits`.

    Raises:
        ValueError: If `block_size` is not positive.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        enable_caching: bool = True,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"Expected block_size > 0, got {block_size!r}.")
        self.block_size = int(block_size)
        self.enable_caching = bool(enable_caching)
        self.pool = BlockPool(num_blocks, enable_caching=enable_caching)
        self._held: dict[int, list[KvBlock]] = {}
        self._n_cached: dict[int, int] = {}
        self.queries = 0
        self.hits = 0

    @property
    def num_used_blocks(self) -> int:
        return self.pool.num_used_blocks

    @property
    def num_free_blocks(self) -> int:
        return self.pool.num_free_blocks

    @property
    def usage(self) -> float:
        return self.pool.usage

    def blocks_held(self, req: "Request") -> int:
        """How many blocks this request currently holds."""
        return len(self._held.get(req.rid, ()))

    def computed_blocks(
        self, req: "Request", recorder: Recorder = NULL_RECORDER
    ) -> tuple[list[KvBlock], int]:
        """The longest cached prefix of this request's context.

        Walks the request's block hashes and stops at the first miss, because a
        hash carries chained semantics: a miss at block `j` means content has
        diverged there, so later blocks must not be reused even if their hashes
        happen to match.

        One edge case is ported verbatim from vLLM
        (`vllm_v1_engine_replica_scheduler.py:1393-1399`): on a *full* hit the
        last block is handed back. A full hit would leave zero new tokens, but
        the model must forward at least one token to produce the next logits.
        This fires for real whenever a group's `prompt_len` is a whole number
        of blocks.

        Args:
            req: The request being admitted or re-prefilled.
            recorder: Where to register the counters for rollback.

        Returns:
            The hit blocks and how many tokens they cover.
        """
        if not self.enable_caching:
            return [], 0

        hashes = block_hashes(req, self.block_size)
        hit: list[KvBlock] = []
        for block_hash in hashes:
            block = self.pool.cached_block(block_hash)
            if block is None:
                break
            hit.append(block)

        recorder.record_attrs_once(("counters", id(self)), self, "queries", "hits")
        self.queries += len(hashes)
        self.hits += len(hit)

        tokens = len(hit) * self.block_size
        # A full hit must still leave one block of work to do.
        if hit and tokens >= req.prompt_len + req.generated:
            hit = hit[:-1]
            tokens -= self.block_size
            self.hits -= 1
        return hit, tokens

    def peek_hit_tokens(self, req: "Request") -> int:
        """Tokens a hit would cover, without recording a query.

        `_step_time_estimate` predicts a step without performing it, so it must
        not move the counters that `vllm:prefix_cache_queries` is compared
        against.
        """
        if not self.enable_caching:
            return 0
        n_hit = 0
        for block_hash in block_hashes(req, self.block_size):
            if self.pool.cached_block(block_hash) is None:
                break
            n_hit += 1
        tokens = n_hit * self.block_size
        if n_hit and tokens >= req.prompt_len + req.generated:
            tokens -= self.block_size
        return tokens

    def _new_blocks_needed(
        self,
        req: "Request",
        n_tokens: int,
        hit_blocks: list[KvBlock],
    ) -> int:
        """Blocks to allocate on top of what is held and what hit."""
        computed = req.prefilled + len(hit_blocks) * self.block_size
        required = _blocks_for(computed + n_tokens, self.block_size)
        return required - self.blocks_held(req) - len(hit_blocks)

    def can_allocate(
        self,
        req: "Request",
        n_tokens: int,
        hit_blocks: list[KvBlock],
    ) -> bool:
        """Whether `n_tokens` more of this request fit.

        Hit blocks that are currently free do not consume new capacity, so they
        are discounted from the free count rather than charged twice.
        """
        needed = self._new_blocks_needed(req, n_tokens, hit_blocks)
        revivable = sum(1 for b in hit_blocks if b.ref_cnt == 0)
        return needed <= self.pool.num_free_blocks - revivable

    def allocate(
        self,
        req: "Request",
        n_tokens: int,
        hit_blocks: list[KvBlock],
        recorder: Recorder = NULL_RECORDER,
    ) -> list[KvBlock] | None:
        """Reserve room for `n_tokens` more of this request.

        Args:
            req: The request. `req.prefilled` must already reflect any hit.
            n_tokens: Tokens about to be computed.
            hit_blocks: Blocks from `computed_blocks`, adopted by this request.
            recorder: Where to register old values for rollback.

        Returns:
            The newly allocated blocks, or None when capacity refuses.

        Raises:
            ValueError: If hit blocks are passed while caching is disabled,
                which would mean a caller invented a hit.
        """
        if hit_blocks and not self.enable_caching:
            raise ValueError("Hit blocks passed while prefix caching is disabled.")
        if not self.can_allocate(req, n_tokens, hit_blocks):
            return None

        recorder.record_once(
            ("held", id(self), req.rid),
            lambda: _HeldBlocks.capture(self, req.rid),
        )
        self.pool.touch(hit_blocks, recorder)
        held = self._held.setdefault(req.rid, [])
        held.extend(hit_blocks)

        needed = self._new_blocks_needed(req, n_tokens, hit_blocks)
        fresh = self.pool.new_blocks(needed, recorder) if needed > 0 else []
        held.extend(fresh)

        if self.enable_caching:
            computed = req.prefilled + len(hit_blocks) * self.block_size
            n_full = (computed + n_tokens) // self.block_size
            n_cached = self._n_cached.get(req.rid, len(hit_blocks))
            # Only blocks `[n_cached, n_full)` can gain a hash here, so only
            # their hashes are built. Asking for the whole context rebuilt every
            # id of a 30k-token request on each decode step -- O(context) work
            # per token, and the single largest cost in the engine loop.
            if n_cached < n_full:
                self.pool.cache_full_blocks(
                    held,
                    block_hashes(req, self.block_size, start=n_cached, stop=n_full),
                    n_cached=n_cached,
                    n_full=n_full,
                    hash_offset=n_cached,
                    recorder=recorder,
                )
            self._n_cached[req.rid] = n_full
        return fresh

    def free(self, req: "Request", recorder: Recorder = NULL_RECORDER) -> None:
        """Release every block this request holds, keeping their hashes.

        The hashes survive deliberately: a released block stays addressable, so
        a sibling or a later round can still hit it.
        """
        recorder.record_once(
            ("held", id(self), req.rid),
            lambda: _HeldBlocks.capture(self, req.rid),
        )
        held = self._held.pop(req.rid, [])
        if held:
            self.pool.free_blocks(reversed(held), recorder)
        self._n_cached.pop(req.rid, None)

    def reset_hashes(self, recorder: Recorder = NULL_RECORDER) -> None:
        """Empty the hash map. This is vLLM's `reset_prefix_cache`."""
        recorder.record(_CachedCounts(self, dict(self._n_cached)))
        self.pool.reset_hashes(recorder)
        self._n_cached.clear()


class _HeldBlocks:
    """Restores one request's held-block list and cached-block count.

    Both are per-request dict entries rather than attributes, so neither
    `AttrSnapshot` nor `MappingKeysAdded` fits: the entry may be created,
    extended, or removed within one event.
    """

    __slots__ = ("_manager", "_rid", "_blocks", "_n_cached", "_existed")

    def __init__(self, manager: "KvCacheManager", rid: int) -> None:
        self._manager = manager
        self._rid = rid
        self._existed = rid in manager._held
        self._blocks = list(manager._held.get(rid, ()))
        self._n_cached = manager._n_cached.get(rid)

    @classmethod
    def capture(cls, manager: "KvCacheManager", rid: int) -> "_HeldBlocks":
        return cls(manager, rid)

    def restore(self) -> None:
        if self._existed:
            self._manager._held[self._rid] = self._blocks
        else:
            self._manager._held.pop(self._rid, None)
        if self._n_cached is None:
            self._manager._n_cached.pop(self._rid, None)
        else:
            self._manager._n_cached[self._rid] = self._n_cached


class _CachedCounts:
    """Restores the per-request cached-block counts after a `reset_hashes`."""

    __slots__ = ("_manager", "_counts")

    def __init__(self, manager: "KvCacheManager", counts: dict[int, int]) -> None:
        self._manager, self._counts = manager, counts

    def restore(self) -> None:
        self._manager._n_cached.update(self._counts)
