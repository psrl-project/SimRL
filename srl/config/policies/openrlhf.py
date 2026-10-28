"""OpenRLHF preset."""

from __future__ import annotations

from dataclasses import dataclass

from srl.policy.openrlhf import OpenRLHFPolicy


@dataclass
class OpenRLHFPolicyConfig:
    """Generation permit pool knobs.

    `async_queue_size > 1` requires `partial_rollout_enable: true`.
    """

    name: str = "openrlhf"
    async_queue_size: int = 1
    partial_rollout_enable: bool = False

    def build(self, max_inflight: int | None = None) -> OpenRLHFPolicy:
        policy = OpenRLHFPolicy(
            async_queue_size=self.async_queue_size,
            partial_rollout_enable=self.partial_rollout_enable,
            max_inflight=max_inflight,
        )
        policy.name = self.name
        return policy

    @property
    def horizon(self) -> None:
        return None

    def ingress_window_groups(
        self,
        batch_size: int,
        group_size: int = 1,
        n_instances: int = 1,
    ) -> int:
        # Matches OpenRLHFPolicy.admit_quota's own hard ceiling exactly:
        # it blocks once `_lumps_in_flight >= async_queue_size`. A lump
        # frees only on take_batch, not on an individual completion, so
        # staging more is a permanent router surplus.
        return self.async_queue_size * batch_size
