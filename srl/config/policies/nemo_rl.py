"""NeMo-RL preset."""

from __future__ import annotations

from dataclasses import dataclass

from srl.policy.nemo_rl import NeMoRLPolicy


@dataclass
class NeMoRLPolicyConfig:
    """Sampler choice and the shared admission gate's knobs."""

    name: str = "nemo_rl"
    sampler: str = "in_order"
    max_staleness_versions: int = 1
    max_lookahead_versions: int = 1
    sample_freshest_first: bool = False
    max_inflight_prompts: int = 32
    max_buffered_rollouts: int = 64
    pause_mode: str = "abort"

    def build(self, max_inflight: int | None = None) -> NeMoRLPolicy:
        policy = NeMoRLPolicy(
            sampler=self.sampler,
            max_staleness_versions=self.max_staleness_versions,
            max_lookahead_versions=self.max_lookahead_versions,
            sample_freshest_first=self.sample_freshest_first,
            max_inflight_prompts=self.max_inflight_prompts,
            max_buffered_rollouts=self.max_buffered_rollouts,
            pause_mode=self.pause_mode,
        )
        policy.name = self.name
        return policy

    @property
    def horizon(self) -> int | None:
        if self.sampler in ("windowed", "weight_fifo"):
            return self.max_staleness_versions
        if self.sampler == "in_order":
            return self.max_lookahead_versions
        return None

    def ingress_window_groups(self, batch_size: int) -> int:
        return self.max_buffered_rollouts + batch_size
