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

    def ingress_window_groups(
        self,
        batch_size: int,
        group_size: int = 1,
        n_instances: int = 1,
    ) -> int:
        # Mirrors VerlV1Policy.admit_quota's own initial credit exactly.
        # _credit never shrinks except via eviction, so staging more is a
        # permanent, unused router surplus, not a transient one.
        return math.floor(self.num_warmup_batches * batch_size) + batch_size
