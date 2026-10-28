"""SkyRL preset."""

from __future__ import annotations

from dataclasses import dataclass

from srl.policy.skyrl import SkyRLPolicy


@dataclass
class SkyRLPolicyConfig:
    """Capacity bound and warn-only staleness knobs."""

    name: str = "skyrl"
    max_staleness_steps: int = 4
    pause_mode: str = "in_place"

    def build(self, max_inflight: int | None = None) -> SkyRLPolicy:
        policy = SkyRLPolicy(
            max_staleness_steps=self.max_staleness_steps,
            pause_mode=self.pause_mode,
            max_inflight=max_inflight,
        )
        policy.name = self.name
        return policy

    @property
    def horizon(self) -> int:
        return self.max_staleness_steps

    def ingress_window_groups(
        self,
        batch_size: int,
        group_size: int = 1,
        n_instances: int = 1,
    ) -> int:
        # Matches ProgressQuota's own initial ceiling exactly. `accepted`
        # only ever grows, recovering room via train_version advancing,
        # not via an ordinary take_batch, so staging more is a permanent
        # router surplus.
        return (self.max_staleness_steps + 1) * batch_size
