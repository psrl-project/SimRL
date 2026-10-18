"""verl v1 preset."""

from __future__ import annotations

import math
from dataclasses import dataclass

from srl.policy.verl_v1 import VerlV1Policy


@dataclass
class VerlV1PolicyConfig:
    """Step-clocked admission and admission-span window knobs."""

    name: str = "verl_v1"
    num_warmup_batches: float = 1.0
    max_off_policy_threshold: int | None = 8
    max_off_policy_strategy: str = "drop"
    pause_mode: str = "abort"

    def build(self, max_inflight: int | None = None) -> VerlV1Policy:
        policy = VerlV1Policy(
            num_warmup_batches=self.num_warmup_batches,
            max_off_policy_threshold=self.max_off_policy_threshold,
            max_off_policy_strategy=self.max_off_policy_strategy,
            pause_mode=self.pause_mode,
        )
        policy.name = self.name
        return policy

    @property
    def horizon(self) -> int | None:
        return self.max_off_policy_threshold

    def ingress_window_groups(self, batch_size: int) -> int:
        h = self.max_off_policy_threshold or 1
        burst = math.ceil(self.num_warmup_batches)
        return (max(h, burst) + 2) * batch_size
