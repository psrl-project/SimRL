"""Dressage staleness policy: active/queue watermarks, span cap, version window."""

from __future__ import annotations

from ..component.rollout.engine import Request
from .base import Cancel, CancelAction, CompleteAction, PauseMode, PolicyContext, SimState, StalenessPolicy
from .rules import FIFO, DistinctVersionWindow, VersionSpanCap


class DressagePolicy(StalenessPolicy):
    """
    Dressage's three-layer control.

    L1 gates on two watermarks: concurrently-active groups and the output
    queue depth. L2 caps how many distinct versions a partial-rollout
    trajectory may span, retrying from scratch past the cap. L3 keeps the
    last `keep_versions` distinct observed versions, keyed on each group's
    *last* trainable token version, not its earliest.

    Dressage is the one system in this study with a genuinely two-check L3:
    `keep_group` at each group's own completion (`on_group_complete` here),
    *and* `generate_rollout_async`'s retroactive `filter_pending`, which
    re-sweeps the whole pending pool and actively evicts (with backfill)
    whenever a newer version arrives -- not a periodic sweep, but inline
    with every batch-assembly attempt (`on_batch_attempt` here). Both checks
    stay; `peek_batch` itself is a plain FIFO selection, since eligibility is
    enforced by eviction, not by filtering what can be selected.
    """

    name = "dressage"

    def __init__(
        self,
        keep_versions: int = 0,
        max_active_groups: int | None = None,
        output_queue_size: int = 1000,
        max_partial_rollout_preempts: int | None = None,
        pause_mode: PauseMode = "in_place",
        max_inflight: int | None = None,
    ):
        self.max_active_groups = max_active_groups
        self.high_watermark = max(1, int(0.8 * output_queue_size))
        self._window = DistinctVersionWindow(keep=keep_versions)
        self._span_cap = VersionSpanCap(
            max_preempts=max_partial_rollout_preempts, action=CancelAction.RETRY_FROM_SCRATCH
        )
        self.pause = pause_mode
        self.max_inflight = max_inflight

        self._active = 0
        self._versions: list[int] = []
        self.gc_dropped = 0

    def _active_cap(self, st: SimState) -> int:
        return self.max_active_groups if self.max_active_groups is not None else st.batch_size

    def admit_quota(self, st: SimState) -> int:
        cap = self._active_cap(st)
        if self._active >= cap:
            return 0
        if len(st.ready_groups) >= self.high_watermark:
            return 0
        room = cap - self._active
        if self.max_inflight is not None:
            room = min(room, self.max_inflight - len(st.inflight) // st.group_size)
        return max(0, room) * st.group_size

    def on_admit(self, group: list[Request], st: SimState) -> None:
        self._active += 1

    def on_group_complete(self, group: list[Request], st: SimState) -> CompleteAction:
        self._active -= 1
        last_version = max(req.v_max for req in group)
        if last_version not in self._versions:
            self._versions.append(last_version)
        if self._window.is_stale(last_version, self._versions):
            self.gc_dropped += 1
            return CompleteAction.DROP
        return CompleteAction.KEEP

    def replacement_groups_on_drop(self, groups: list[list[Request]], st: SimState) -> int:
        return len(groups)

    def peek_batch(self, st: SimState) -> list[Request] | None:
        return FIFO().select(st.ready_groups, st.batch_size)

    def on_batch_attempt(self, st: SimState) -> list[Cancel]:
        # A group's own completion-time check only compares it against
        # versions already observed before it finished. Re-derived here,
        # live, against the current (possibly grown) `self._versions`, so a
        # group that was fine at completion but has since fallen outside the
        # window while waiting in the ready pool is actively evicted (with
        # dataset backfill) rather than merely skipped -- the real system's
        # `filter_pending`, inline with every batch-assembly attempt.
        cancels = [
            Cancel(group[0].group_id, CancelAction.REPLACE)
            for group in st.ready_groups
            if self._window.is_stale(max(req.v_max for req in group), self._versions)
        ]
        self.gc_dropped += len(cancels)
        return cancels

    def on_publish(self, new_version: int, st: SimState) -> list[Cancel]:
        cancels = self._span_cap.on_publish(PolicyContext(state=st), new_version)
        self._active -= len(cancels)
        return cancels

    def pause_mode(self, st: SimState) -> PauseMode:
        return self.pause
