"""Dressage preset."""

from __future__ import annotations

from dataclasses import dataclass

from srl.policy.dressage import DressagePolicy


@dataclass
class DressagePolicyConfig:
    """Active/queue watermark, span cap, and distinct-version window knobs."""

    name: str = "dressage"
    keep_versions: int = 0
    max_active_groups: int | None = None
    output_queue_size: int = 1000
    max_partial_rollout_preempts: int | None = None
    pause_mode: str = "in_place"

    def build(self, max_inflight: int | None = None) -> DressagePolicy:
        policy = DressagePolicy(
            keep_versions=self.keep_versions,
            max_active_groups=self.max_active_groups,
            output_queue_size=self.output_queue_size,
            max_partial_rollout_preempts=self.max_partial_rollout_preempts,
            pause_mode=self.pause_mode,
            max_inflight=max_inflight,
        )
        policy.name = self.name
        return policy

    @property
    def horizon(self) -> int | None:
        return self.keep_versions or None

    def ingress_window_groups(
        self,
        batch_size: int,
        group_size: int = 1,
        n_instances: int = 1,
    ) -> int:
        cap = self.max_active_groups if self.max_active_groups is not None else batch_size
        return cap + 2 * batch_size
