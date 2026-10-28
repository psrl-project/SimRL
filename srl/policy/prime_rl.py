"""prime-rl staleness policy: dispatch-lag admission, L2 cancel on publish, L3 checked live at batch assembly."""

from __future__ import annotations

from ..component.rollout.engine import Request
from .base import Cancel, CancelAction, CompleteAction, PauseMode, PolicyContext, SimState, StalenessPolicy
from .rules import FIFO, CancelBelowVersion, VersionWindow


class PrimeRLPolicy(StalenessPolicy):
    """
    prime-rl's continuous rollout loop: a flat inflight cap and a
    dispatch-lag gate bound admission, independent of the staleness window
    itself. Generation is never interrupted by a weight publish except for
    groups the staleness window actually cancels, matching prime-rl's design
    goal of decoupling generation from training syncs.

    `max_inflight` bounds concurrent generation, not buffer residency: a
    slot frees the moment a task finishes generating, whatever its eventual
    staleness verdict ("有空位才继续准入，生成完成后再释放"), so new work is
    admitted without waiting on training to consume this one.

    prime-rl has two genuinely separate staleness mechanisms, not one:
    `on_version_pending` cancels still-generating groups too far behind a
    newly published version (L2, kept here in `on_publish`). `_drop_stale`
    re-checks the ready pool immediately before it assembles a training
    batch (L3, `TrainSink.take_batch()` calls it right before selecting) --
    not on a publish timer decoupled from consumption. `on_batch_attempt`
    mirrors that: called live, right before every batch-assembly attempt.

    `action` defaults to `"replace"`, not `"drop"`: the ready-pool sweep
    (`VersionWindow(refill=True)`) already backfills every group it expires,
    and the runtime cancel matches it, so a wider `target_lag` that produces
    more cancelled stragglers does not permanently shrink the dataset pool
    relative to the sweep's own replacement behavior.
    """

    name = "prime_rl"

    def __init__(
        self,
        max_off_policy_steps: int = 8,
        max_inflight: int = 1024,
        target_lag: int = 1,
        action: str = "replace",
    ):
        assert max_off_policy_steps >= 0, "max_off_policy_steps must be >= 0."
        assert max_inflight > 0, "max_inflight must be > 0."
        assert target_lag >= 0, "target_lag must be >= 0."
        self.max_off_policy_steps = max_off_policy_steps
        self.max_inflight = max_inflight
        self.target_lag = target_lag
        try:
            cancel_action = CancelAction[action.upper()]
        except KeyError:
            raise ValueError(
                f"Unknown action {action!r}, expected one of {[a.name.lower() for a in CancelAction]}."
            ) from None
        self._runtime_cancel = CancelBelowVersion(horizon=max_off_policy_steps, action=cancel_action)
        self._window = VersionWindow(threshold=max_off_policy_steps, refill=True)

        self._dispatch_index = 0
        self._admitted_in_cohort = 0
        self._inflight_groups = 0
        self.gc_dropped = 0

    def admit_quota(self, st: SimState) -> int:
        if self._inflight_groups >= self.max_inflight:
            return 0
        if self._dispatch_index >= st.train_version + self.target_lag:
            return 0
        room = self.max_inflight - self._inflight_groups
        return max(0, room) * st.group_size

    def on_admit(self, group: list[Request], st: SimState) -> None:
        self._inflight_groups += 1
        self._admitted_in_cohort += 1
        if self._admitted_in_cohort >= st.batch_size:
            self._dispatch_index += 1
            self._admitted_in_cohort = 0

    def on_group_complete(self, group: list[Request], st: SimState) -> CompleteAction:
        # Generation concurrency frees the instant a task finishes, whatever
        # its eventual staleness verdict -- same timing fix as TRL and
        # verl_v1's admission bookkeeping. Staleness itself is checked later,
        # live, by `on_batch_attempt`.
        self._inflight_groups -= 1
        return CompleteAction.KEEP

    def peek_batch(self, st: SimState) -> list[Request] | None:
        return FIFO().select(st.ready_groups, st.batch_size)

    def take_batch(self, st: SimState) -> list[Request] | None:
        # No `_inflight_groups` decrement here: every batch member already
        # freed its slot at its own `on_group_complete`, well before this
        # consumption.
        return self.peek_batch(st)

    def on_publish(self, new_version: int, st: SimState) -> list[Cancel]:
        # L2 only: cancel still-generating groups too far behind the newly
        # published version. L3 (the ready-pool sweep) now runs live, in
        # `on_batch_attempt`, not here.
        ctx = PolicyContext(state=st)
        cancels = self._runtime_cancel.on_publish(ctx, new_version)
        # A done group already freed its slot at its own `on_group_complete`;
        # only a still-generating group cancelled here needs it freed now.
        # `CancelBelowVersion` has no status filter of its own, so this guard
        # still matters even though L3's own done-group sweep moved out.
        status_by_id = {g.group_id: g.status for g in st.groups}
        for cancel in cancels:
            if status_by_id.get(cancel.group_id) != "done":
                self._inflight_groups -= 1
        self.gc_dropped += len(cancels)
        return cancels

    def on_batch_attempt(self, st: SimState) -> list[Cancel]:
        cancels = self._window.on_publish(PolicyContext(state=st), st.train_version)
        self.gc_dropped += len(cancels)
        return cancels

    def pause_mode(self, st: SimState) -> PauseMode:
        return "in_place"
