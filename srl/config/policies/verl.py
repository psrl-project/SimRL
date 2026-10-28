"""verl legacy preset."""

from __future__ import annotations

import math
from dataclasses import dataclass

from srl.policy.verl import VerlPolicy


@dataclass
class VerlPolicyConfig:
    """Sample-overage ratio and batch hand-off knobs."""

    name: str = "verl"
    staleness_threshold: float = 2.0
    require_batches: int = 1

    def build(self, max_inflight: int | None = None) -> VerlPolicy:
        policy = VerlPolicy(
            staleness_threshold=self.staleness_threshold,
            require_batches=self.require_batches,
            max_inflight=max_inflight,
        )
        policy.name = self.name
        return policy

    @property
    def horizon(self) -> int:
        return math.ceil(self.staleness_threshold)

    def ingress_window_groups(
        self,
        batch_size: int,
        group_size: int = 1,
        n_instances: int = 1,
    ) -> int:
        return (math.ceil(self.staleness_threshold) + 1) * batch_size
