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

    def ingress_window_groups(self, batch_size: int) -> int:
        return (self.async_queue_size + 1) * batch_size
