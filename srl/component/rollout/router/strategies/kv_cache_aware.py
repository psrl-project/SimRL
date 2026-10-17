"""KV-cache-aware routing."""

from __future__ import annotations

from ...engine import InstanceView, Request
from ..strategy import register_strategy


@register_strategy("kv_cache_aware")
class KvCacheAware:
    """Choose the instance with the most uncommitted free KV blocks."""

    name = "kv_cache_aware"

    def _headroom(self, view: InstanceView) -> int:
        return view.free_blocks - view.committed_blocks

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        return max(range(len(candidates)), key=lambda i: self._headroom(candidates[i]))

