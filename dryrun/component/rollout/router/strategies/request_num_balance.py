"""Route requests to the instance with the fewest active requests."""

from __future__ import annotations

from ...engine import InstanceView, Request
from ..strategy import register_strategy


@register_strategy("request_num_balance")
class RequestNumBalance:
    """Fewest in-flight requests wins.

    This is the behaviour-preserving default: ties resolve to the lowest
    candidate index, which is instance-id order.
    """

    name = "request_num_balance"

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        return min(range(len(candidates)), key=lambda i: candidates[i].n_active)

