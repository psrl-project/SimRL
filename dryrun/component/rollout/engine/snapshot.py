"""Rollback state for speculative event advancement.

The simulator advances each instance to its next event and discards the events
it does not commit, so every mutation an event performs must be reversible.
These snapshots hold the minimal state needed to undo one event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .request import ReqStatus, Request

if TYPE_CHECKING:
    from .instance import NativeInstance


@dataclass
class ReqSnapshot:
    """Per-request mutable state captured for undo."""

    prefilled: int
    generated: int
    status: ReqStatus
    complete_time: float | None
    num_preemptions: int
    reprefill_tokens: int
    n_segments: int
    last_seg_tokens: int  # 0 when segments list is empty

    @classmethod
    def capture(cls, req: Request) -> ReqSnapshot:
        return cls(
            prefilled=req.prefilled,
            generated=req.generated,
            status=req.status,
            complete_time=req.complete_time,
            num_preemptions=req.num_preemptions,
            reprefill_tokens=req.reprefill_tokens,
            n_segments=len(req.segments),
            last_seg_tokens=req.segments[-1].tokens if req.segments else 0,
        )

    def restore(self, req: Request) -> None:
        req.prefilled = self.prefilled
        req.generated = self.generated
        req.status = self.status
        req.complete_time = self.complete_time
        req.num_preemptions = self.num_preemptions
        req.reprefill_tokens = self.reprefill_tokens
        # The `segments` list is append only, so truncate and fix the last entry.
        del req.segments[self.n_segments :]
        if self.n_segments > 0 and req.segments:
            req.segments[-1].tokens = self.last_seg_tokens


@dataclass
class InstanceSnapshot:
    """
    Minimal snapshot of `NativeInstance` mutable state, taken before each event.

    Sufficient to fully restore state via `NativeInstance.undo_last_event`.
    """

    t_local: float
    waiting: list[Request]
    running: list[Request]
    completed_len: int
    n_preemptions: int
    stats_len: int
    # Paired with the request object itself, not keyed by id(req): the snapshot
    # already pins every request it captured, so restore needs no lookup.
    req_snapshots: list[tuple[Request, ReqSnapshot]]

    @classmethod
    def capture(cls, inst: NativeInstance) -> InstanceSnapshot:
        # `completed` is deliberately not snapshotted per-request. Events only
        # mutate requests in `waiting`/`running`, and a request retired *by* this
        # event was in `running` at capture time, so it is already covered. Once
        # retired a request is never touched again, and `restore` truncates the
        # list back to `completed_len`. Including it would make snapshot cost grow
        # with the whole run instead of with the live batch.
        return cls(
            t_local=inst.t_local,
            waiting=list(inst.waiting),
            running=list(inst.running),
            completed_len=len(inst.completed),
            n_preemptions=inst.n_preemptions,
            stats_len=len(inst.step_stats),
            req_snapshots=[
                (r, ReqSnapshot.capture(r)) for r in (*inst.waiting, *inst.running)
            ],
        )

    def restore(self, inst: NativeInstance) -> None:
        inst.t_local = self.t_local
        inst.waiting = list(self.waiting)
        inst.running = list(self.running)
        del inst.completed[self.completed_len :]
        inst.n_preemptions = self.n_preemptions
        del inst.step_stats[self.stats_len :]
        for req, rs in self.req_snapshots:
            rs.restore(req)
