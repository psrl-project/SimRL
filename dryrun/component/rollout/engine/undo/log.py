"""An ordered collection of captured old values for one speculative event."""

from __future__ import annotations

from typing import Callable, Hashable

from .protocol import Undoable


class UndoLog:
    """Captured old values for one event, restorable in reverse order.

    Reverse order matters: a later entry may have been captured after an earlier
    one already changed something, so rollback has to unwind rather than replay.

    There is deliberately no "capture this whole object" entry point. Callers
    register the specific things they touch, which is what keeps cost
    proportional to the work an event did rather than to the size of the
    subsystem it did the work in. A whole-pool `copy.deepcopy` measured 179 ms
    per event, or 1771 s across a run whose real duration is 3349 s.
    """

    __slots__ = ("_entries", "_keys")

    def __init__(self) -> None:
        self._entries: list[Undoable] = []
        self._keys: set[Hashable] = set()

    def __len__(self) -> int:
        return len(self._entries)

    def record(self, entry: Undoable) -> None:
        """Register an already-captured old value."""
        self._entries.append(entry)

    def record_once(self, key: Hashable, capture: Callable[[], Undoable]) -> None:
        """Capture under `key` the first time it is seen, and only then.

        The first capture is the only correct one: an event that touches one
        block repeatedly must restore the value from before the event, not from
        between two of its own mutations. Passing a callable also means a repeat
        key costs nothing to build.
        """
        if key in self._keys:
            return
        self._keys.add(key)
        self._entries.append(capture())

    def rollback(self) -> None:
        """Restore every captured value, newest first, and empty the log.

        Raises:
            RuntimeError: If the log is empty. A second rollback means the
                caller lost track of event boundaries, and restoring stale
                values would corrupt state silently.
        """
        if not self._entries:
            raise RuntimeError(
                "Cannot roll back an empty undo log: either no event was "
                "recorded, or this event was already rolled back."
            )
        for entry in reversed(self._entries):
            entry.restore()
        self.clear()

    def clear(self) -> None:
        """Discard the log without restoring. This is what committing means."""
        self._entries.clear()
        self._keys.clear()
