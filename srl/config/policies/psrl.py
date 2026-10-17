"""PSRL preset."""

from __future__ import annotations

from dataclasses import dataclass

from srl.policy.psrl import PSRLPolicy


@dataclass
class PsrlPolicyConfig:
    """Reserve/Occupy/Consume inventory knobs."""

    name: str = "psrl"
    staleness: int = 2
    proactive_filter: bool = False
    proactive_threshold: int = 0

    def build(self, max_inflight: int | None = None) -> PSRLPolicy:
        policy = PSRLPolicy(
            proactive_filter=self.proactive_filter,
            proactive_threshold=self.proactive_threshold,
        )
        policy.name = self.name
        return policy

    @property
    def horizon(self) -> int:
        return self.staleness

    def ingress_window_groups(self, batch_size: int) -> int:
        return (self.staleness + 1) * batch_size
