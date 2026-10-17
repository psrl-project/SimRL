"""Evict whoever holds the most KV, so one eviction frees the most blocks."""

from __future__ import annotations

from ..request import Request
from .strategy import register_eviction


@register_eviction("longest_ctx")
class LongestCtx:
    """Evicts the deepest context.

    The greedy answer to "free the most memory per eviction", and the useful
    middle arm in an ablation: it is deadline-blind like `vllm_lru` but not
    arbitrary, so a value-driven strategy has to beat a policy that is already
    optimising something real.

    The cost is that the deepest context is also the most expensive to
    recompute, which is exactly the trade a value-based strategy prices.
    """

    name = "longest_ctx"

    def select(self, candidates: list[Request], now: float) -> Request | None:
        return max(candidates, key=lambda r: (r.ctx, r.rid))
