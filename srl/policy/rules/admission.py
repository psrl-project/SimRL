"""L1 admission rules: how many new responses may be admitted right now."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ...component.rollout.engine import Request
from ..base import PolicyContext


@dataclass
class ProgressQuota:
    """AReaL's cumulative token bucket: `(ofp + V + 1) * B - (accepted + running)`."""

    ofp: int

    def quota(self, ctx: PolicyContext) -> int:
        st = ctx.state
        return (self.ofp + st.train_version + 1) * st.batch_size - (ctx.accepted + ctx.running)


@dataclass
class WatermarkGate:
    """slime's loop barrier: deny while `remaining >= batch_size`, else allow over-sampled room."""

    over_sampling_ratio: float

    def quota(self, remaining: int, batch_size: int) -> int:
        if remaining >= batch_size:
            return 0
        return max(1, round(self.over_sampling_ratio * batch_size))


class LegacyOverage:
    """verl's sample-overage window: a bounded hand-off queue plus a staleness counter.

    `staleness_threshold` is a float sample-overage ratio, not a version count.
    The window resets at every weight sync; a group completing into a full queue is dropped.
    """

    def __init__(self, threshold: float, require_batches: int = 1):
        self.threshold = threshold
        self.require_batches = require_batches

        self.staleness_samples = 0
        self.queue: deque = deque()
        self.max_queue_size = 1
        self.max_required = 1
        self.dropped = 0
        self._configured = False

    def _configure(self, batch_size: int) -> None:
        required = batch_size * self.require_batches
        self.max_required = int(required * (self.threshold + 1))
        self.max_queue_size = self.max_required
        self.queue = deque(self.queue, maxlen=self.max_queue_size)
        self._configured = True

    def admit_quota(self, batch_size: int) -> int:
        if not self._configured:
            self._configure(batch_size)
        if len(self.queue) >= self.max_queue_size:
            return 0
        if self.staleness_samples >= self.max_required:
            return 0
        return self.max_required - self.staleness_samples

    def on_admit(self) -> None:
        self.staleness_samples += 1

    def on_group_complete(self, group: list[Request], batch_size: int) -> bool:
        """Return True to keep the group, False to drop it."""
        if not self._configured:
            self._configure(batch_size)
        if len(self.queue) >= self.max_queue_size:
            self.dropped += 1
            return False
        self.queue.append(group)
        return True

    def peek_batch(self, batch_size: int) -> list[Request] | None:
        required = batch_size * self.require_batches
        if len(self.queue) < required:
            return None
        return [req for group in list(self.queue)[:required] for req in group]

    def take_batch(self, batch_size: int) -> list[Request] | None:
        batch = self.peek_batch(batch_size)
        if batch is None:
            return None
        required = batch_size * self.require_batches
        popped = [self.queue.popleft() for _ in range(required)]
        return [req for group in popped for req in group]

    def on_train_version_advance(self, inflight_count: int) -> None:
        self.staleness_samples = inflight_count + len(self.queue)
