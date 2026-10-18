"""SkyRL capacity-bound staleness policy."""

from __future__ import annotations

from ..component.rollout.engine import Request
from .base import CompleteAction, PauseMode, PolicyContext, SimState, StalenessPolicy
from .rules import FIFO, ProgressQuota, WarnOnly


class SkyRLPolicy(StalenessPolicy):
    """
    SkyRL's fully-async admission with a warn-only L3 check.

    The admission bucket bounds aggregate unconsumed work, same formula as
    AReaL once SkyRL's 1-indexed step is folded in. Nothing is ever dropped
    for staleness: a group past the threshold is still trained, only counted.
    """

    name = "skyrl"

    def __init__(
        self,
        max_staleness_steps: int = 4,
        pause_mode: PauseMode = "in_place",
        max_inflight: int | None = None,
    ):
        assert max_staleness_steps > 0, "max_staleness_steps must be > 0."
        self._quota = ProgressQuota(ofp=max_staleness_steps)
        self._warn = WarnOnly(threshold=max_staleness_steps)
        self.pause = pause_mode
        self.max_inflight = max_inflight

        self.accepted = 0
        self.running = 0
        self.staleness_violations = 0

    def admit_quota(self, st: SimState) -> int:
        ctx = PolicyContext(state=st, accepted=self.accepted, running=self.running)
        capacity = self._quota.quota(ctx)
        if self.max_inflight is not None:
            capacity = min(capacity, self.max_inflight - self.running)
        return max(0, capacity) * st.group_size

    def on_admit(self, group: list[Request], st: SimState) -> None:
        self.running += 1

    def on_group_complete(self, group: list[Request], st: SimState) -> CompleteAction:
        self.running -= 1
        self.accepted += 1
        return CompleteAction.KEEP

    def peek_batch(self, st: SimState) -> list[Request] | None:
        return FIFO().select(st.ready_groups, st.batch_size)

    def take_batch(self, st: SimState) -> list[Request] | None:
        batch = self.peek_batch(st)
        if batch is None:
            return None
        taken_group_ids = {req.group_id for req in batch}
        self.staleness_violations += self._warn.count_violations(PolicyContext(state=st), taken_group_ids)
        return batch

    def pause_mode(self, st: SimState) -> PauseMode:
        return self.pause
