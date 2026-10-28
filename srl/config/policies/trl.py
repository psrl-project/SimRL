"""TRL preset."""

from __future__ import annotations

from dataclasses import dataclass

from srl.policy.trl import TRLPolicy


@dataclass
class TRLPolicyConfig:
    """A bounded task queue and a version window checked at dequeue."""

    name: str = "trl"
    max_staleness: int = 4
    max_inflight_tasks: int = -1
    pause_mode: str = "in_place"

    def build(self, max_inflight: int | None = None) -> TRLPolicy:
        policy = TRLPolicy(
            max_staleness=self.max_staleness,
            max_inflight_tasks=self.max_inflight_tasks,
            pause_mode=self.pause_mode,
        )
        policy.name = self.name
        return policy

    @property
    def horizon(self) -> int:
        return self.max_staleness

    def ingress_window_groups(
        self,
        batch_size: int,
        group_size: int = 1,
        n_instances: int = 1,
    ) -> int:
        cap = self.max_inflight_tasks
        if cap < 0:
            cap = max(self.max_staleness, 1) * batch_size
        return cap + batch_size
