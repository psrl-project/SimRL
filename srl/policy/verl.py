"""verl fully async staleness policy."""

from __future__ import annotations

from ..component.rollout.engine import Request
from .base import CompleteAction, SimState, StalenessPolicy
from .rules import LegacyOverage


class VerlPolicy(StalenessPolicy):
    """
    verl's fully async policy with window-based admission and bounded queue.

    staleness_threshold is a float sample-overage ratio (not a version count).
    The window resets at every weight sync. The hand-off queue drops oldest
    on overflow. Delegates its queue/counter bookkeeping to `LegacyOverage`.
    """

    name = "verl"

    def __init__(
        self,
        staleness_threshold: float = 0.1,
        require_batches: int = 1,
        max_inflight: int | None = None,
    ):
        assert staleness_threshold >= 0, "staleness_threshold must be >= 0."
        assert require_batches >= 1, "require_batches must be >= 1."
        self._overage = LegacyOverage(threshold=staleness_threshold, require_batches=require_batches)
        self.max_inflight = max_inflight

    @property
    def threshold(self) -> float:
        return self._overage.threshold

    @property
    def require_batches(self) -> int:
        return self._overage.require_batches

    @property
    def staleness_samples(self) -> int:
        return self._overage.staleness_samples

    @property
    def queue(self):
        return self._overage.queue

    @property
    def max_queue_size(self) -> int:
        return self._overage.max_queue_size

    @property
    def max_required(self) -> int:
        return self._overage.max_required

    @property
    def dropped(self) -> int:
        return self._overage.dropped

    def admit_quota(self, st: SimState) -> int:
        room = self._overage.admit_quota(st.batch_size)
        if self.max_inflight is not None:
            room = min(room, self.max_inflight - len(st.inflight) // st.group_size)
        return max(0, room) * st.group_size

    def on_admit(self, group: list[Request], st: SimState) -> None:
        self._overage.on_admit()

    def on_group_complete(self, group: list[Request], st: SimState) -> CompleteAction:
        keep = self._overage.on_group_complete(group, st.batch_size)
        return CompleteAction.KEEP if keep else CompleteAction.DROP

    def peek_batch(self, st: SimState) -> list[Request] | None:
        return self._overage.peek_batch(st.batch_size)

    def take_batch(self, st: SimState) -> list[Request] | None:
        return self._overage.take_batch(st.batch_size)

    def on_train_version_advance(self, train_version: int, st: SimState) -> None:
        self._overage.on_train_version_advance(len(st.inflight) // st.group_size)

    def check_invariants(self, st: SimState) -> None:
        assert len(self.queue) <= self.max_queue_size, (
            f"Queue size {len(self.queue)} exceeds max {self.max_queue_size}."
        )
