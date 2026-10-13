"""vLLM's own eviction order: the most recently dispatched request pays."""

from __future__ import annotations

from ..request import Request
from .strategy import register_eviction


@register_eviction("vllm_lru")
class VllmLru:
    """Evicts in reverse arrival order, which is what the engine does natively.

    The behaviour-preserving default, so a run that does not name a strategy
    matches the measured baseline exactly. `dispatch_time` is the arrival key
    and `rid` only breaks ties, which keeps the choice deterministic across
    runs.
    """

    name = "vllm_lru"

    def select(self, candidates: list[Request], now: float) -> Request | None:
        return max(candidates, key=lambda r: (r.dispatch_time, r.rid))
