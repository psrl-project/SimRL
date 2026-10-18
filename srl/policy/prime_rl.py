"""prime-rl staleness policy: dispatch-lag admission plus a publish-time sweep."""

from __future__ import annotations

from ..component.rollout.engine import Request
from .base import Cancel, CancelAction, PauseMode, PolicyContext, SimState, StalenessPolicy
from .rules import FIFO, CancelBelowVersion, VersionWindow


class PrimeRLPolicy(StalenessPolicy):
    """
    prime-rl's continuous rollout loop: a flat inflight cap and a
    dispatch-lag gate bound admission, independent of the staleness window
    itself. Generation is never interrupted by a weight publish except for
    groups the staleness window actually cancels, matching prime-rl's design
    goal of decoupling generation from training syncs.

    `on_version_pending` is prime-rl's own name for the publish-time cancel
    over live groups. `_drop_stale` is its name for the ready-pool sweep.
    Both key on the same `max_off_policy_steps` horizon, so a finished group
    past the horizon is found by both and must be deduplicated by group id
    before it is charged against `_inflight_groups` once, not twice.
    """

    name = "prime_rl"

    def __init__(
        self,
        max_off_policy_steps: int = 8,
        max_inflight: int = 1024,
        target_lag: int = 1,
        action: str = "drop",
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

    def peek_batch(self, st: SimState) -> list[Request] | None:
        return FIFO().select(st.ready_groups, st.batch_size)

    def take_batch(self, st: SimState) -> list[Request] | None:
        batch = self.peek_batch(st)
        if batch is None:
            return None
        self._inflight_groups -= len({req.group_id for req in batch})
        return batch

    def on_publish(self, new_version: int, st: SimState) -> list[Cancel]:
        ctx = PolicyContext(state=st)
        running_cancels = self._runtime_cancel.on_publish(ctx, new_version)
        ready_cancels = self._window.on_publish(ctx, new_version)
        # A finished group can match both: the ready-pool sweep always refills,
        # which takes priority over the runtime cancel's configured action.
        by_group_id = {c.group_id: c for c in running_cancels}
        by_group_id.update({c.group_id: c for c in ready_cancels})
        cancels = list(by_group_id.values())
        self._inflight_groups -= len(cancels)
        self.gc_dropped += len(cancels)
        return cancels

    def pause_mode(self, st: SimState) -> PauseMode:
        return "in_place"
