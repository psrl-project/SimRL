"""The router's dwell queue: requests admitted but not yet placed on an instance.

A request waits here when no instance can take it, which is the state the real
gateway spends its backlog in. The engine's own `waiting` list keeps its
original meaning, work an instance has accepted and not yet started.

Ordering mirrors SMG's `MultiPriorityRequestQueue`: a min-heap on
`(version_tag, sort_key, seq)`, where `seq` is a monotonic counter so equal
keys pop in arrival order. Lower weight versions go first, because a request
generated under an older version is closer to expiring.
"""

from __future__ import annotations

import heapq
from enum import Enum

from ..engine import Request


class SortKey(Enum):
    """How the queue breaks ties within one weight version."""

    SHORT_LENGTH = "short_length"
    LONG_LENGTH = "long_length"
    SMALL_ID = "small_id"


class RouterQueue:
    """A priority-ordered queue of requests awaiting placement.

    Usage:
        queue = RouterQueue(SortKey.SHORT_LENGTH)
        queue.push(req, t=now)
        for req in queue.drain():       # highest priority first
            ...
    """

    def __init__(self, sort_key: SortKey = SortKey.SHORT_LENGTH):
        self.sort_key = sort_key
        self._heap: list[tuple[int, int, int, Request]] = []
        self._seq = 0

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def _tie_break(self, req: Request) -> int:
        """The within-version ordering key.

        `prompt_len` rather than `ctx` is deliberate, and it is the same choice
        SMG makes by restoring a preparation snapshot before re-queueing. A
        looped-back request has accumulated output, so keying on `ctx` would
        push it further back every time it returns and starve exactly the
        requests that have already paid the most.
        """
        if self.sort_key is SortKey.SHORT_LENGTH:
            return req.prompt_len
        if self.sort_key is SortKey.LONG_LENGTH:
            return -req.prompt_len
        return req.rid

    def push(self, req: Request, t: float) -> None:
        """Enqueue `req`, recording when it began waiting."""
        req.router_enqueue_time = t
        self._seq += 1
        heapq.heappush(
            self._heap,
            (req.v_traj, self._tie_break(req), self._seq, req),
        )

    def peek(self) -> Request | None:
        return self._heap[0][3] if self._heap else None

    def pop(self) -> Request:
        """Remove and return the highest-priority request."""
        return heapq.heappop(self._heap)[3]

    def drain(self) -> list[Request]:
        """Return every queued request in priority order, emptying the queue."""
        out = []
        while self._heap:
            out.append(self.pop())
        return out

    def remove(self, reqs: list[Request]) -> list[Request]:
        """Drop `reqs` from the queue, returning those it actually held.

        A dropped request must leave the queue, or dispatch would later place a
        request the simulator has already discarded. Rebuilding is O(n) in the
        queue rather than O(1), which is the right trade here: discards are rare
        and a lazy tombstone would leak into `__len__`, which telemetry reports.
        """
        targets = {id(req) for req in reqs}
        kept = [entry for entry in self._heap if id(entry[3]) not in targets]
        removed = [entry[3] for entry in self._heap if id(entry[3]) in targets]
        if removed:
            self._heap = kept
            heapq.heapify(self._heap)
        return removed
