"""Evict whoever has the latest (least urgent) deadline buffer, so the most urgent requests
keep their KV and make forward progress first."""

from __future__ import annotations

from ..request import Request
from .strategy import register_eviction


@register_eviction("latest_deadline")
class LatestDeadline:
    """Evicts the resident whose training deadline is furthest in the future.

    When KV pressure forces an eviction, the request whose group has the most
    slack (latest deadline buffer) pays. Urgent groups -- those closest to
    missing their training window -- keep their KV and generate uninterrupted,
    which lets the critical buffer fill sooner.

    Requires `deadline_buffer` to be stamped on requests at admission time.
    Requests with `deadline_buffer == 0` (not stamped, or from a strategy that
    does not set it) are treated as maximally urgent and are never chosen over a
    stamped request. Among equal deadlines the longest context pays, matching
    `longest_ctx` behaviour and freeing the most memory per eviction.
    """

    name = "latest_deadline"
    python_engine_only = True

    def select(self, candidates: list[Request], now: float) -> Request | None:
        stamped = [r for r in candidates if r.deadline_buffer > 0]
        pool = stamped if stamped else candidates
        return max(pool, key=lambda r: (r.deadline_buffer, r.ctx, r.rid))
