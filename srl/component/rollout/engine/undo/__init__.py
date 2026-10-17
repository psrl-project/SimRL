"""Reusable rollback for speculative event advancement.

The engine advances each instance by one event, commits the earliest, and
discards the rest, so every mutation must reverse. This package is the one
mechanism for arranging that: a mutating method takes a `Recorder` and registers
what it is about to overwrite, and the caller decides whether that recorder is a
real `UndoLog` or `NULL_RECORDER`.

Adding rollback to a new subsystem therefore means implementing `Undoable` for
whatever it owns, not extending a snapshot dataclass that knows about every
subsystem at once.
"""

from .entries import MISSING, AttrSnapshot, MappingKeysChanged, TruncateList
from .log import UndoLog
from .protocol import NULL_RECORDER, NullRecorder, Recorder, Undoable

__all__ = [
    "MISSING",
    "NULL_RECORDER",
    "AttrSnapshot",
    "MappingKeysChanged",
    "NullRecorder",
    "Recorder",
    "TruncateList",
    "UndoLog",
    "Undoable",
]
