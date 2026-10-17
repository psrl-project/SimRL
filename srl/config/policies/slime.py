"""slime preset."""

from __future__ import annotations

import math
from dataclasses import dataclass

from srl.policy.slime import SlimePolicy


@dataclass
class SlimePolicyConfig:
    """Loop barrier and over-sampling knobs."""

    name: str = "slime"
    update_weights_interval: int = 1
    over_sampling_ratio: float = 1.0
    # A Literal annotation is rejected by Hydra's structured configs.
    pause_mode: str = "abort"

    def build(self, max_inflight: int | None = None) -> SlimePolicy:
        policy = SlimePolicy(
            update_weights_interval=self.update_weights_interval,
            over_sampling_ratio=self.over_sampling_ratio,
            pause_mode=self.pause_mode,
            max_inflight=max_inflight,
        )
        policy.name = self.name
        return policy

    @property
    def horizon(self) -> None:
        return None

    def ingress_window_groups(self, batch_size: int) -> int:
        return (math.ceil(self.over_sampling_ratio) + 2) * batch_size
