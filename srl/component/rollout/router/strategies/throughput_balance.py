"""Cost-model-based throughput routing."""

from __future__ import annotations

from ...engine import InstanceView, Request
from ..strategy import register_strategy


@register_strategy("throughput_balance")
class ThroughputBalance:
    """Choose the instance with the smallest predicted step-time increase."""

    name = "throughput_balance"

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        return min(
            range(len(candidates)),
            key=lambda i: candidates[i].step_time_delta(req),
        )

