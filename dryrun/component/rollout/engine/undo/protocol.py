"""What it means for a mutation to be reversible.

`advance_one_event` is speculative: the driver advances every instance by one
event, commits the earliest, and discards the rest (`simulator.py:388-396` and
`:356-360`). That is how instances share one global clock, so reversibility is a
structural requirement of the engine loop rather than a safety net.
"""

from __future__ import annotations

from typing import Callable, Hashable, Protocol, runtime_checkable


@runtime_checkable
class Undoable(Protocol):
    """A captured old value that knows how to put itself back."""

    def restore(self) -> None:
        """Return the captured target to the state it held at capture time."""
        ...


class Recorder(Protocol):
    """Where a mutating method registers what it is about to overwrite.

    Methods take a `Recorder` rather than an optional log, so a committed path
    and a speculative one run the same code. `NULL_RECORDER` is the committed
    case.
    """

    def record(self, entry: Undoable) -> None:
        """Register an already-captured old value."""
        ...

    def record_once(self, key: Hashable, capture: Callable[[], Undoable]) -> None:
        """Register an old value the first time `key` is seen, and only then.

        `capture` is a callable rather than a value so that a repeat key costs
        nothing: a decode segment can touch the same block many times, and the
        first capture is the only correct one anyway.
        """
        ...


class NullRecorder:
    """A `Recorder` that keeps nothing, for paths that cannot be rolled back.

    `advance_to` commits as it goes, so recording there would allocate an entry
    per touched block and then drop it. `record_once` deliberately does not
    invoke its callable.
    """

    __slots__ = ()

    def record(self, entry: Undoable) -> None:
        pass

    def record_once(self, key: Hashable, capture: Callable[[], Undoable]) -> None:
        pass


NULL_RECORDER = NullRecorder()
"""Shared no-op recorder. Stateless, so one instance serves every caller."""
