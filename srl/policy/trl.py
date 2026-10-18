"""TRL staleness policy: one bounded task queue and a version window checked at dequeue."""

from __future__ import annotations

from ..component.rollout.engine import Request
from .base import Cancel, PauseMode, PolicyContext, SimState, StalenessPolicy
from .rules import FIFO, VersionWindow


class TRLPolicy(StalenessPolicy):
    """
    TRL's online trainer: a single admitted-but-unconsumed task queue, drained
    FIFO, bounded by `max_inflight_tasks`. The version window sweeps the ready
    pool at every publish, which is equivalent to checking at dequeue since
    this simulator publishes once per training step.

    TRL actually drops individual stale samples once a group has been
    disaggregated into single completions. This simulator keeps prompt groups
    atomic, so the window drops the whole group instead.
    TODO: model per-sample (non-group-atomic) dropping once the simulator
    supports partial-group training batches.
    """

    name = "trl"

    def __init__(
        self,
        max_staleness: int = 4,
        max_inflight_tasks: int = -1,
        pause_mode: PauseMode = "in_place",
    ):
        assert max_staleness >= 0, "max_staleness must be >= 0."
        self.max_staleness = max_staleness
        self.max_inflight_tasks: int | None = None if max_inflight_tasks < 0 else max_inflight_tasks
        self._window = VersionWindow(threshold=max_staleness, refill=True)
        self.pause = pause_mode

        self._buffered = 0
        self.gc_dropped = 0

    def admit_quota(self, st: SimState) -> int:
        if self.max_inflight_tasks is None:
            self.max_inflight_tasks = max(self.max_staleness, 1) * st.batch_size
        room = self.max_inflight_tasks - self._buffered
        return max(0, room) * st.group_size

    def on_admit(self, group: list[Request], st: SimState) -> None:
        self._buffered += 1

    def peek_batch(self, st: SimState) -> list[Request] | None:
        return FIFO().select(st.ready_groups, st.batch_size)

    def take_batch(self, st: SimState) -> list[Request] | None:
        batch = self.peek_batch(st)
        if batch is None:
            return None
        self._buffered -= len({req.group_id for req in batch})
        return batch

    def on_publish(self, new_version: int, st: SimState) -> list[Cancel]:
        cancels = self._window.on_publish(PolicyContext(state=st), new_version)
        self._buffered -= len(cancels)
        self.gc_dropped += len(cancels)
        return cancels

    def pause_mode(self, st: SimState) -> PauseMode:
        return self.pause
