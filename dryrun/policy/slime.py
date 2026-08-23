"""slime loop barrier staleness policy."""

from __future__ import annotations

from ..component.rollout.engine import Request
from .base import CompleteAction, SimState, StalenessPolicy


class SlimePolicy(StalenessPolicy):
    """
    slime's staleness control via training loop barrier structure.

    No staleness parameter. The bound comes purely from the loop structure:
    generation dispatched BEFORE weight push, consumed AFTER. Max staleness
    equals update_weights_interval exactly.
    """

    name = "slime"

    def __init__(
        self,
        update_weights_interval: int = 1,
        over_sampling_ratio: float = 1.0,
        partial_rollout: bool = False,
        max_inflight: int | None = None,
    ):
        assert update_weights_interval >= 1, "update_weights_interval must be >= 1."
        assert over_sampling_ratio >= 1.0, "over_sampling_ratio must be >= 1.0."
        self.interval = update_weights_interval
        self.over_sampling_ratio = over_sampling_ratio
        self.partial_rollout = partial_rollout
        self.max_inflight = max_inflight

        self.remaining = 0
        self._rollout_version = 0
        self._steps_since_push = 0
        self.stall_since: float | None = None
        self.stall_time = 0.0

    def admit_quota(self, st: SimState) -> int:
        if self.remaining >= st.batch_size:
            return 0
        room = max(1, int(round(self.over_sampling_ratio * st.batch_size)))
        if self.max_inflight is not None:
            room = min(room, self.max_inflight - len(st.inflight))
        return max(0, room)

    def on_admit(self, req: Request, st: SimState) -> None:
        self.remaining += 1

    def on_complete(self, req: Request, st: SimState) -> CompleteAction:
        if len(st.ready) < st.batch_size:
            return CompleteAction.KEEP
        return CompleteAction.DROP

    def peek_batch(self, st: SimState) -> list[Request] | None:
        if len(st.ready) < st.batch_size:
            return None
        return list(st.ready[:st.batch_size])

    def take_batch(self, st: SimState) -> list[Request] | None:
        chosen = self.peek_batch(st)
        if chosen is None:
            return None
        if self.stall_since is not None:
            self.stall_time += st.now - self.stall_since
            self.stall_since = None
        self.remaining = 0
        return chosen

    def on_batch_unavailable(self, st: SimState) -> None:
        if st.inflight and self.stall_since is None:
            self.stall_since = st.now

    def rollout_version_after_train(self, st: SimState) -> int:
        self._steps_since_push += 1
        if self._steps_since_push >= self.interval:
            self._steps_since_push = 0
            self._rollout_version = st.train_version
        return self._rollout_version

    def will_sync_after_train(self, st: SimState) -> bool:
        return self._steps_since_push + 1 >= self.interval

    def expire(self, st: SimState, expected_train_version: int) -> list[Request]:
        if self.partial_rollout:
            return []
        return list(st.inflight)

    def check_invariants(self, st: SimState) -> None:
        if not self.partial_rollout:
            assert st.train_version - self._rollout_version <= self.interval, (
                f"Version gap {st.train_version - self._rollout_version} exceeds "
                f"interval {self.interval}."
            )
