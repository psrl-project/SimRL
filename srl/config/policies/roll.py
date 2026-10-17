"""ROLL preset."""

from __future__ import annotations

import math
from dataclasses import dataclass

from srl.policy.roll import RollPolicy


@dataclass
class RollPolicyConfig:
    """Asynchronous generation ratio for the admission budget and GC window."""

    name: str = "roll"
    async_generation_ratio: float = 1.0

    def build(self, max_inflight: int | None = None) -> RollPolicy:
        policy = RollPolicy(max_inflight=max_inflight)
        policy.name = self.name
        return policy

    @property
    def horizon(self) -> int:
        return math.ceil(self.async_generation_ratio)

    def ingress_window_groups(self, batch_size: int) -> int:
        return (math.ceil(self.async_generation_ratio) + 2) * batch_size
