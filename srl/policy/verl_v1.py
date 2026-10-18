"""verl v1 staleness policy: step-clocked admission, span-based L3."""

from __future__ import annotations

import math

from ..component.rollout.engine import Request
from .base import Cancel, PauseMode, PolicyContext, SimState, StalenessPolicy
from .rules import OffPolicySpan


class VerlV1Policy(StalenessPolicy):
    """
    verl v1's `trainer.v1.sampler` policy.

    L1 is a pure step-clocked feed, not staleness-gated: `num_warmup_batches`
    admits once, then one batch of groups per publish. L3 is the only
    staleness-aware control, via `OffPolicySpan` keyed on `admit_version`.
    """

    name = "verl_v1"

    def __init__(
        self,
        num_warmup_batches: float = 1.0,
        max_off_policy_threshold: int | None = 8,
        max_off_policy_strategy: str = "drop",
        pause_mode: PauseMode = "abort",
    ):
        assert max_off_policy_strategy in ("drop", "wait"), (
            f"Unsupported max_off_policy_strategy {max_off_policy_strategy!r}."
        )
        assert num_warmup_batches > 0, "num_warmup_batches must be > 0, or nothing is ever admitted."
        assert max_off_policy_threshold is None or max_off_policy_threshold > 0, (
            "max_off_policy_threshold must be > 0, or None to disable it."
        )
        self.warmup = num_warmup_batches
        self._span = OffPolicySpan(threshold=max_off_policy_threshold, strategy=max_off_policy_strategy)
        self.pause = pause_mode

        self._credit = 0
        self._admitted = 0
        self._configured = False

    def admit_quota(self, st: SimState) -> int:
        if not self._configured:
            self._credit = math.floor(self.warmup * st.batch_size)
            self._configured = True
        return max(0, self._credit - self._admitted) * st.group_size

    def on_admit(self, group: list[Request], st: SimState) -> None:
        self._admitted += 1

    def on_train_version_advance(self, train_version: int, st: SimState) -> None:
        self._credit += st.batch_size

    def can_sample(self, st: SimState) -> bool:
        return self._span.can_sample(PolicyContext(state=st))

    def on_publish(self, new_version: int, st: SimState) -> list[Cancel]:
        return self._span.on_publish(PolicyContext(state=st), new_version)

    def peek_batch(self, st: SimState) -> list[Request] | None:
        return self._span.select(PolicyContext(state=st), st.ready_groups, st.batch_size)

    def pause_mode(self, st: SimState) -> PauseMode:
        return self.pause
