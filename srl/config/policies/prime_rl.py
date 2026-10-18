"""prime-rl preset."""

from __future__ import annotations

from dataclasses import dataclass

from srl.policy.prime_rl import PrimeRLPolicy


@dataclass
class PrimeRLPolicyConfig:
    """Dispatch-lag admission and a publish-time staleness sweep."""

    name: str = "prime_rl"
    max_off_policy_steps: int = 8
    max_inflight: int = 1024
    target_lag: int = 1
    action: str = "drop"

    def build(self, max_inflight: int | None = None) -> PrimeRLPolicy:
        policy = PrimeRLPolicy(
            max_off_policy_steps=self.max_off_policy_steps,
            max_inflight=self.max_inflight,
            target_lag=self.target_lag,
            action=self.action,
        )
        policy.name = self.name
        return policy

    @property
    def horizon(self) -> int:
        return self.max_off_policy_steps

    def ingress_window_groups(self, batch_size: int) -> int:
        return min(self.max_inflight, (self.max_off_policy_steps + 2) * batch_size)
