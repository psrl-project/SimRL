"""Seeded random routing."""

from __future__ import annotations

import random

from ...engine import InstanceView, Request
from ..strategy import register_strategy


@register_strategy("random")
class RandomChoice:
    """Uniformly choose a candidate using an owned random generator."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        return self._rng.randrange(len(candidates))

