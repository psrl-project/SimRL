"""TRL staleness policy: one bounded task queue, version-checked live at batch assembly."""

from __future__ import annotations

from ..component.rollout.engine import Request
from .base import Cancel, CompleteAction, PauseMode, PolicyContext, SimState, StalenessPolicy
from .rules import FIFO, VersionWindow


class TRLPolicy(StalenessPolicy):
    """
    TRL's online trainer: `max_inflight_tasks` bounds concurrent generation,
    not buffer residency. A slot frees the moment a task finishes generating
    -- whatever its eventual staleness verdict -- so new work is admitted
    without waiting on the training side, matching how TRL releases a
    generation slot on completion rather than on consumption. Buffered,
    completed groups then drain FIFO, filtered live by `on_batch_attempt`.

    `on_batch_attempt` re-applies the version check immediately before every
    batch-assembly attempt, matching the real system: `RolloutQueueDataset`
    checks the live model version the instant the trainer reads a sample,
    not on a periodic sweep decoupled from consumption. There is a single
    check, not two -- real TRL has no completion-time check of its own, and
    checking live at every attempt (the simulator calls this hook before
    every `peek_batch`/`take_batch`) needs none either.

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

    def on_group_complete(self, group: list[Request], st: SimState) -> CompleteAction:
        # `max_inflight_tasks` bounds generation concurrency, not buffer
        # residency: a slot frees the instant a task finishes generating,
        # whatever its eventual staleness verdict, so new work can be admitted
        # without waiting on the training side to consume this one. Staleness
        # itself is checked later, live, by `on_batch_attempt`.
        self._buffered -= 1
        return CompleteAction.KEEP

    def on_batch_attempt(self, st: SimState) -> list[Cancel]:
        cancels = self._window.on_publish(PolicyContext(state=st), st.train_version)
        self.gc_dropped += len(cancels)
        return cancels

    def peek_batch(self, st: SimState) -> list[Request] | None:
        return FIFO().select(st.ready_groups, st.batch_size)

    def take_batch(self, st: SimState) -> list[Request] | None:
        # No `_buffered` decrement here: every batch member already freed its
        # slot at its own `on_group_complete`, well before this consumption.
        return self.peek_batch(st)

    def pause_mode(self, st: SimState) -> PauseMode:
        return self.pause
