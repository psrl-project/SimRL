"""Rollback state for speculative event advancement.

The simulator advances each instance to its next event and discards the events
it does not commit, so every mutation an event performs must be reversible.

This is a thin adapter over `..undo`: capturing an instance means opening an
`UndoLog`, registering the old values of what the instance itself owns, and
handing that log to the engine so the pool can register its own. Restoring means
rolling the log back. There is no per-subsystem field list here, which is what
lets a new subsystem become undoable without editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .request import Request
from .undo import AttrSnapshot, Recorder, TruncateList, UndoLog

if TYPE_CHECKING:
    from .instance import NativeInstance

_REQUEST_FIELDS = (
    "prefilled",
    "generated",
    "status",
    "complete_time",
    "num_preemptions",
    "num_aborts",
    "reprefill_tokens",
    "prefix_hit_tokens",
)
"""What an *event* mutates, and therefore what must roll back.

Router-owned fields (`instance_id`, `num_reroutes`, `router_enqueue_time`) are
deliberately absent: the router only ever runs after an event commits, so no
speculative window can touch them. If that ever changes, the assertion in
`Router.dispatch` fires before this list becomes wrong.
"""


def record_request(req: Request, recorder: Recorder) -> None:
    """Register one request's mutable state, once per event.

    `segments` is append-only except for its last entry, which is incremented in
    place, so the length alone is not enough -- hence the extra entry. This is
    the same correction the original hand-written snapshot made.
    """
    recorder.record_attrs_once(("req", id(req)), req, *_REQUEST_FIELDS)
    recorder.record_once(("segs", id(req)), lambda: TruncateList.capture(req.segments))
    if req.segments:
        recorder.record_attrs_once(("seg_tail", id(req)), req.segments[-1], "tokens")


@dataclass
class InstanceSnapshot:
    """One event's worth of captured old values.

    Args:
        log: The undo log the event recorded into.
    """

    log: UndoLog

    @classmethod
    def capture(cls, inst: "NativeInstance") -> "InstanceSnapshot":
        """Open a log and register what the instance itself owns.

        `completed` is deliberately captured by length rather than per request.
        Events only mutate requests in `waiting` / `running`, and a request
        retired *by* this event was in `running` at capture time, so it is
        already covered. Including it would make capture cost grow with the
        whole run instead of with the live batch.

        `waiting` and `running` are captured as *copies* of the lists, because
        the engine reorders them in place. The copy is a shallow one: it holds
        the same `Request` objects, whose own fields are captured separately, so
        the cost is proportional to the live batch rather than to the run.
        """
        log = UndoLog()
        log.record(
            AttrSnapshot.capture(
                inst, "t_local", "n_preemptions", "_last_decode_blocked"
            )
        )
        log.record(
            AttrSnapshot(
                target=inst,
                names=("waiting", "running"),
                values=(list(inst.waiting), list(inst.running)),
            )
        )
        log.record(TruncateList.capture(inst.completed))
        log.record(TruncateList.capture(inst.step_stats))
        for req in (*inst.waiting, *inst.running):
            record_request(req, log)
        return cls(log=log)

    def restore(self, inst: "NativeInstance") -> None:
        """Roll every captured value back, newest first."""
        del inst  # Every entry holds its own target.
        self.log.rollback()
