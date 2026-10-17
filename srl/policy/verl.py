"""verl fully async staleness policy."""

from __future__ import annotations

from collections import deque

from ..component.rollout.engine import Request
from .base import CompleteAction, SimState, StalenessPolicy


class VerlPolicy(StalenessPolicy):
    """
    verl's fully async policy with window-based admission and bounded queue.

    staleness_threshold is a float sample-overage ratio (not a version count).
    The window resets at every weight sync. The hand-off queue drops oldest
    on overflow.
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
        self.threshold = staleness_threshold
        self.require_batches = require_batches
        self.max_inflight = max_inflight

        self.staleness_samples = 0
        self.queue: deque = deque()
        self.max_queue_size = 1
        self.max_required = 1
        self.dropped = 0
        self._configured = False

    def _configure(self, st: SimState) -> None:
        # Every count here is in prompt groups, so the queue evicts a whole
        # group rather than orphaning one response of it.
        required = st.batch_size * self.require_batches
        self.max_required = int(required * (self.threshold + 1))
        self.max_queue_size = self.max_required
        self.queue = deque(self.queue, maxlen=self.max_queue_size)
        self._configured = True

    def admit_quota(self, st: SimState) -> int:
        if not self._configured:
            self._configure(st)
        if len(self.queue) >= self.max_queue_size:
            return 0
        if self.staleness_samples >= self.max_required:
            return 0
        room = self.max_required - self.staleness_samples
        if self.max_inflight is not None:
            room = min(room, self.max_inflight - len(st.inflight) // st.group_size)
        return max(0, room) * st.group_size

    def on_admit(self, group: list[Request], st: SimState) -> None:
        self.staleness_samples += 1

    def on_group_complete(self, group: list[Request], st: SimState) -> CompleteAction:
        if not self._configured:
            self._configure(st)
        if len(self.queue) >= self.max_queue_size:
            self.dropped += 1
            return CompleteAction.DROP
        self.queue.append(group)
        return CompleteAction.KEEP

    def peek_batch(self, st: SimState) -> list[Request] | None:
        required = st.batch_size * self.require_batches
        if len(self.queue) < required:
            return None
        return [req for group in list(self.queue)[:required] for req in group]

    def take_batch(self, st: SimState) -> list[Request] | None:
        batch = self.peek_batch(st)
        if batch is None:
            return None
        required = st.batch_size * self.require_batches
        popped = [self.queue.popleft() for _ in range(required)]
        return [req for group in popped for req in group]

    def on_train_version_advance(self, train_version: int, st: SimState) -> None:
        self.staleness_samples = len(st.inflight) // st.group_size + len(self.queue)

    def check_invariants(self, st: SimState) -> None:
        assert len(self.queue) <= self.max_queue_size, (
            f"Queue size {len(self.queue)} exceeds max {self.max_queue_size}."
        )
