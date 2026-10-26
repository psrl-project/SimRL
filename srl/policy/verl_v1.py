"""verl v1 staleness policy: step-clocked admission, span-based L3 checked live."""

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

    Real verl's `drop` strategy re-checks a group's span live, inside the
    same loop that assembles a training batch ("选出的 batch 还会再次接受
    检查") -- not a sweep decoupled from consumption. `on_batch_attempt`
    mirrors this: called immediately before every batch-assembly attempt,
    not on a publish timer. `wait` never filters a completed group at all --
    the real system's `wait` only stalls sampling while something is still
    generating past the window (`can_sample`), and an already-finished
    straggler is never re-examined once done; `OffPolicySpan.on_publish`
    already no-ops for `strategy != "drop"`, so no branching is needed here.
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
            # Startup grants the warmup prefetch *and* round 1's own normal
            # per-round admission before round 1's training ever completes
            # ("启动时先额外准入 num_warmup_batches 批，首轮再正常准入
            # train_batch_size 个，合计两者之和"). `on_train_version_advance`
            # grants every later round's `+batch_size`, but it only fires on
            # a version advance -- after a round finishes -- so it cannot
            # cover round 1's own share; that one credit has to start here.
            self._credit = math.floor(self.warmup * st.batch_size) + st.batch_size
            self._configured = True
        return max(0, self._credit - self._admitted) * st.group_size

    def on_admit(self, group: list[Request], st: SimState) -> None:
        self._admitted += 1

    def on_train_version_advance(self, train_version: int, st: SimState) -> None:
        self._credit += st.batch_size

    def can_sample(self, st: SimState) -> bool:
        return self._span.can_sample(PolicyContext(state=st))

    def on_batch_attempt(self, st: SimState) -> list[Cancel]:
        cancels = self._span.on_publish(PolicyContext(state=st), st.train_version)
        # Real verl's drop refill (ReplayBufferAsync.wait_for_sampleable's
        # refill_fn call) submits a replacement prompt straight to
        # generate_sequences, with no quota check at all. Without this,
        # a replacement is only staged via the simulator's own
        # _ingress_group_credits (fed through _apply_cancels) -- it still
        # has to wait for _credit, which only grows on
        # on_train_version_advance, once per version. In steady state
        # (no spare credit left over from the warmup head start) that
        # stalls the replacement until the next version anyway, same as
        # the regular next batch, collapsing real drop's recurring
        # per-step eviction-and-refill into a one-time transient.
        self._credit += len(cancels)
        return cancels

    def peek_batch(self, st: SimState) -> list[Request] | None:
        return self._span.select(PolicyContext(state=st), st.ready_groups, st.batch_size)

    def pause_mode(self, st: SimState) -> PauseMode:
        return self.pause
