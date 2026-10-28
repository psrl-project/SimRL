"""slime preset."""

from __future__ import annotations

from dataclasses import dataclass

from srl.policy.slime import SlimePolicy


@dataclass
class SlimePolicyConfig:
    """Fully-async rollout worker knobs."""

    name: str = "slime"
    # `--sglang-server-concurrency`, per rollout engine.
    server_concurrency: int = 64
    group_prefetch_factor: int = 1
    flush_cache_interval: int = 1

    def build(self, max_inflight: int | None = None) -> SlimePolicy:
        policy = SlimePolicy(
            server_concurrency=self.server_concurrency,
            group_prefetch_factor=self.group_prefetch_factor,
            flush_cache_interval=self.flush_cache_interval,
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
        # The worker gates two separate counts against `threshold`, so a live
        # group budget of `2 * threshold` is what lets both fill. One more batch
        # keeps staging headroom above consumption.
        threshold = self.build().threshold_groups(group_size, n_instances)
        return 2 * max(threshold, batch_size) + batch_size
