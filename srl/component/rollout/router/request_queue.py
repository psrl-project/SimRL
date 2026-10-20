"""The router's dwell queue: requests staged but not yet placed on an instance.

A request waits here when no instance can take it, which is the state the real
gateway spends its backlog in. The engine's own `waiting` list keeps its
original meaning, work an instance has accepted and not yet started.

Each weight version owns one heap ordered by `(sort_key, seq)`. Keeping the
versions separate lets a strategy decide whether a blocked version prevents
dispatch from trying the next one.
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

    Requests are first partitioned by `v_traj`, oldest version first. `sort_key`
    only breaks ties inside one version layer.

    Usage:
        queue = RouterQueue(SortKey.SMALL_ID)
        queue.push(req, t=now)
        for req in queue.drain():       # highest priority first
            ...
    """

    def __init__(self, sort_key: SortKey = SortKey.SMALL_ID):
        self.sort_key = sort_key
        self._layers: dict[int, list[tuple[int, int, Request]]] = {}
        self._seq = 0

    def __len__(self) -> int:
        return sum(len(layer) for layer in self._layers.values())

    def __bool__(self) -> bool:
        return bool(self._layers)

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
        layer = self._layers.setdefault(req.v_traj, [])
        heapq.heappush(
            layer,
            (self._tie_break(req), self._seq, req),
        )

    def peek(self) -> Request | None:
        version = self.oldest_version()
        return self.peek_version(version) if version is not None else None

    def peek_version(self, version: int) -> Request | None:
        """Return the first request in `version` without removing it."""
        layer = self._layers.get(version)
        return layer[0][2] if layer else None

    def candidates_version(self, version: int, width: int) -> list[Request]:
        """Return up to `width` requests from `version`, best heap order first.

        The heap key is fixed when a request is pushed, so a caller ranking on
        anything that moves with the clock cannot express its order here. This
        offers a window instead and lets the caller choose inside it, which keeps
        the heap's meaning intact while bounding the work per placement.
        """
        layer = self._layers.get(version)
        if not layer:
            return []
        if width <= 1:
            return [layer[0][2]]
        return [entry[2] for entry in heapq.nsmallest(width, layer)]

    def remove_one(self, version: int, req: Request) -> None:
        """Remove `req` from `version`, wherever it sits in the heap.

        Raises:
            KeyError: If that version does not hold the request, which would
                mean a caller is placing work the queue never had.
        """
        layer = self._layers.get(version)
        if not layer:
            raise KeyError(f"Router queue version {version!r} is empty.")
        kept = [entry for entry in layer if entry[2] is not req]
        if len(kept) == len(layer):
            raise KeyError(f"Request {req.rid} is not queued at version {version!r}.")
        if kept:
            heapq.heapify(kept)
            self._layers[version] = kept
        else:
            del self._layers[version]

    def versions(self) -> tuple[int, ...]:
        """Return nonempty versions from oldest to newest."""
        return tuple(sorted(self._layers))

    def depths(self) -> dict[int, int]:
        """Return the queue depth of every nonempty version."""
        return {version: len(self._layers[version]) for version in self.versions()}

    def oldest_version(self) -> int | None:
        """Return the oldest nonempty version."""
        return min(self._layers, default=None)

    def pop(self) -> Request:
        """Remove and return the highest-priority request."""
        version = self.oldest_version()
        if version is None:
            raise IndexError("Cannot pop from an empty router queue.")
        return self.pop_version(version)

    def pop_version(self, version: int) -> Request:
        """Remove and return the first request in `version`."""
        layer = self._layers.get(version)
        if not layer:
            raise IndexError(f"Cannot pop from empty router queue version {version!r}.")
        req = heapq.heappop(layer)[2]
        if not layer:
            del self._layers[version]
        return req

    def drain(self) -> list[Request]:
        """Return every queued request in priority order, emptying the queue."""
        out = []
        while self:
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
        removed: list[Request] = []
        for version, layer in list(self._layers.items()):
            kept = [entry for entry in layer if id(entry[2]) not in targets]
            removed.extend(entry[2] for entry in layer if id(entry[2]) in targets)
            if kept:
                heapq.heapify(kept)
                self._layers[version] = kept
            else:
                del self._layers[version]
        return removed

    def rekey(self, reqs: list[Request]) -> None:
        """Re-layer queued `reqs` after their trajectory version changes.

        Existing enqueue timestamps are preserved because rebasing a staged
        request is not a new arrival. Reinserted requests receive fresh heap
        sequence numbers, while the configured priority remains authoritative.
        """
        enqueue_times = {id(req): req.router_enqueue_time for req in reqs}
        queued = self.remove(reqs)
        for req in queued:
            self.push(req, enqueue_times[id(req)])
