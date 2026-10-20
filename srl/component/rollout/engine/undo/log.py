"""An ordered collection of captured old values for one speculative event."""

from __future__ import annotations

from typing import Any, Callable, Hashable

from .protocol import Undoable


class UndoLog:
    """Captured old values for one event, restorable in reverse order.

    Reverse order matters: a later entry may have been captured after an earlier
    one already changed something, so rollback has to unwind rather than replay.
    It is load-bearing even under `record_once`, because two different keys can
    name overlapping state: `("req", id(req))` and `("hit", id(req))` both
    capture `prefilled`, and the earliest capture is the only one holding the
    true pre-event value. Unwinding newest-first lets it win.

    There is deliberately no "capture this whole object" entry point. Callers
    register the specific things they touch, which is what keeps cost
    proportional to the work an event did rather than to the size of the
    subsystem it did the work in.

    An entry is either an `Undoable` or, for the attribute captures that are
    almost all of the traffic, an inline `(target, names, values)` tuple. The
    tuple holds the same information an `AttrSnapshot` would, without building
    one, since most captures are never rolled back and the object would be
    discarded unused. `AttrSnapshot` remains the form callers outside this
    module pass to `record`.
    """

    __slots__ = ("_entries", "_keys")

    def __init__(self) -> None:
        self._entries: list[Undoable | tuple[Any, tuple[str, ...], tuple[Any, ...]]] = []
        self._keys: set[Hashable] = set()

    def __len__(self) -> int:
        return len(self._entries)

    def record(self, entry: Undoable) -> None:
        """Register an already-captured old value, or an inverse operation.

        The engine's invertible mutations (a refcount, a linked-list insertion)
        register a small slotted object whose `restore` performs the inverse
        rather than a captured value, since binding a method into a
        `(callable, args)` pair allocates per call. See `_Unlinked`.
        """
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

    def record_attrs_once(self, key: Hashable, target: Any, *names: str) -> None:
        """`record_once` for the one case that is almost all of the traffic.

        Every hot-path `record_once` captures attributes of one object, and
        passing a callable meant building a closure per call, including for
        the repeat keys the callable exists to avoid paying for. Taking the
        target and names directly means a repeat key costs a set lookup and
        nothing else, which is what the callable was supposed to buy.

        The capture is stored as a plain tuple rather than an `AttrSnapshot`,
        for the reason given in the class docstring. `rollback` recognises it by
        shape.
        """
        keys = self._keys
        if key in keys:
            return
        keys.add(key)
        # Unrolled for one and two names: every call site in the engine passes
        # one or the other, and at this call volume the generator's frame setup
        # outweighed the attribute reads.
        if len(names) == 2:
            self._entries.append(
                (target, names, (getattr(target, names[0]), getattr(target, names[1])))
            )
        elif len(names) == 1:
            self._entries.append((target, names, (getattr(target, names[0]),)))
        else:
            self._entries.append(
                (target, names, tuple(getattr(target, n) for n in names))
            )

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
        # Two entry shapes, told apart without an isinstance call: a tuple is an
        # attribute capture from `record_attrs_once`, anything else is an
        # `Undoable`. The tuple exists so that the hottest capture path does not
        # allocate an object per call. See the class docstring.
        for entry in reversed(self._entries):
            if type(entry) is tuple:
                target, names, values = entry
                for i, name in enumerate(names):
                    setattr(target, name, values[i])
            else:
                entry.restore()
        self.clear()

    def clear(self) -> None:
        """Discard the log without restoring. This is what committing means."""
        self._entries.clear()
        self._keys.clear()
