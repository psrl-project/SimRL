"""Evict whoever holds the least KV, so the deepest (most invested) contexts stay resident."""

from __future__ import annotations

from ..request import Request
from .strategy import register_eviction


@register_eviction("shortest_ctx")
class ShortestCtx:
    """Evicts the shallowest context.

    The mirror of `longest_ctx`: instead of freeing the most memory per
    eviction, it protects the residents that have accumulated the most
    context (and are therefore the most expensive to re-prefill) by paying
    with whichever request has invested the least so far.
    """

    name = "shortest_ctx"

    def select(self, candidates: list[Request], now: float) -> Request | None:
        return min(candidates, key=lambda r: (r.ctx, r.rid))
