"""slime's fully-async staleness policy."""

from __future__ import annotations

from ..component.rollout.engine import Request
from .base import PauseMode, SimState, StalenessPolicy


def should_flush_cache(flush_cache_interval: int, weight_version: int) -> bool:
    """Whether this weight publication aborts in-flight generation, as slime decides it.

    The first publication always flushes, including after a resume. Later ones
    flush when `(version - 1)` is a multiple of `flush_cache_interval`, and a
    nonpositive interval never flushes again.
    """
    return weight_version == 1 or (
        flush_cache_interval > 0 and (weight_version - 1) % flush_cache_interval == 0
    )


class SlimePolicy(StalenessPolicy):
    """
    slime's staleness control in fully-async rollout.

    Weights advance every training step. What a publication does to in-flight
    generation is decided by `flush_cache_interval`, not by any staleness
    comparison. Admission counts prompt groups the way the rollout worker does:
    at most `threshold` still generating and `threshold` already complete.

    Args:
        server_concurrency (int): `--sglang-server-concurrency`, per rollout engine.
        group_prefetch_factor (int): `--fully-async-group-prefetch-factor`.
        flush_cache_interval (int): `--flush-cache-interval`. One aborts every
            publication, a larger value aborts every N-th, and a nonpositive
            value aborts only at the first.
        max_inflight (int | None): Hard cap on the same count, from `rollout.max_inflight`.
    """

    name = "slime"

    def __init__(
        self,
        server_concurrency: int = 64,
        group_prefetch_factor: int = 1,
        flush_cache_interval: int = 1,
        max_inflight: int | None = None,
    ) -> None:
        assert server_concurrency >= 1, (
            f"server_concurrency must be >= 1, got {server_concurrency!r}."
        )
        assert group_prefetch_factor >= 1, (
            f"group_prefetch_factor must be >= 1, got {group_prefetch_factor!r}."
        )
        self.server_concurrency = server_concurrency
        self.group_prefetch_factor = group_prefetch_factor
        self.flush_cache_interval = flush_cache_interval
        self.max_inflight = max_inflight

    def threshold_groups(
        self,
        group_size: int,
        n_instances: int,
    ) -> int:
        """The prompt-group count slime's worker tops up against."""
        per_engine = max(1, self.server_concurrency * n_instances // group_size)
        threshold = per_engine * self.group_prefetch_factor
        if self.max_inflight is not None:
            threshold = min(threshold, self.max_inflight)
        return threshold

    def validate(self, batch_size: int, group_size: int, n_instances: int) -> None:
        threshold = self.threshold_groups(group_size, n_instances)
        if threshold < batch_size:
            raise ValueError(
                f"slime admits at most {threshold} prompt groups but a training batch "
                f"needs {batch_size}. Raise server_concurrency or group_prefetch_factor."
            )

    def admit_quota(self, st: SimState) -> int:
        threshold = self.threshold_groups(st.group_size, st.n_instances)
        active = len(st.groups) - len(st.ready_groups)
        if active >= threshold or len(st.ready_groups) >= threshold:
            return 0
        return st.group_size

    def peek_batch(self, st: SimState) -> list[Request] | None:
        if len(st.ready_groups) < st.batch_size:
            return None
        return [req for group in st.ready_groups[: st.batch_size] for req in group]

    def pause_mode(self, st: SimState) -> PauseMode:
        if should_flush_cache(self.flush_cache_interval, st.train_version):
            return "abort"
        return "in_place"

    def check_invariants(self, st: SimState) -> None:
        assert st.rollout_version == st.train_version, (
            f"slime publishes every step, so rollout version {st.rollout_version!r} "
            f"must equal train version {st.train_version!r}."
        )
