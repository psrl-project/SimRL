"""Round-robin routing."""

from __future__ import annotations

from ...engine import InstanceView, Request
from ..strategy import register_strategy


@register_strategy("round_robin")
class RoundRobin:
    """Cycle through candidates across placements."""

    name = "round_robin"

    def __init__(self) -> None:
        self._counter = 0

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        index = self._counter % len(candidates)
        self._counter += 1
        return index

