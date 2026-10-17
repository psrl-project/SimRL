"""AReaL preset."""

from __future__ import annotations

from dataclasses import dataclass

from srl.policy.areal import ArealPolicy


@dataclass
class ArealPolicyConfig:
    """Cumulative token bucket knobs."""

    name: str = "areal"
    max_head_offpolicyness: int = 0

    def build(self, max_inflight: int | None = None) -> ArealPolicy:
        policy = ArealPolicy(max_inflight=max_inflight)
        policy.name = self.name
        return policy

    @property
    def horizon(self) -> int:
        return self.max_head_offpolicyness

    def ingress_window_groups(self, batch_size: int) -> int:
        # `max_head_offpolicyness` is both the progress offset and the version span.
        return (2 * self.max_head_offpolicyness + 1) * batch_size
