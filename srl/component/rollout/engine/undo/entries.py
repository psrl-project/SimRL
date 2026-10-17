"""The three shapes of captured old value this engine actually needs.

Each mirrors a technique the original `InstanceSnapshot` arrived at by hand:
scalar fields stored and assigned back, append-only lists restored by length
rather than by copy, and writes to a mapping restored to the value they
displaced. Naming them makes the techniques reusable instead of re-derived per
subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, MutableMapping

MISSING = object()
"""Stands for "this key was absent", which `None` cannot: a mapping is allowed
to hold `None` as a value, and conflating the two would turn a restored `None`
into a deletion."""


@dataclass(slots=True)
class AttrSnapshot:
    """Old values of named attributes on one object.

    Args:
        target: The object whose attributes were captured.
        names: Attribute names, in capture order.
        values: The captured values, parallel to `names`.
    """

    target: Any
    names: tuple[str, ...]
    values: tuple[Any, ...]

    @classmethod
    def capture(cls, target: Any, *names: str) -> "AttrSnapshot":
        """Capture `names` from `target`.

        The one- and two-name cases are spelled out rather than left to the
        generator: every capture in the engine's hot path has one or two names,
        and at tens of millions of calls per run the generator's frame setup
        dominated the attribute reads it was wrapping.

        Raises:
            AttributeError: If any name is absent, so a typo fails where it was
                written rather than silently at rollback.
        """
        if len(names) == 2:
            return cls(
                target=target,
                names=names,
                values=(getattr(target, names[0]), getattr(target, names[1])),
            )
        if len(names) == 1:
            return cls(target=target, names=names, values=(getattr(target, names[0]),))
        return cls(
            target=target,
            names=names,
            values=tuple(getattr(target, name) for name in names),
        )

    def restore(self) -> None:
        for name, value in zip(self.names, self.values):
            setattr(self.target, name, value)


@dataclass(slots=True)
class TruncateList:
    """An append-only list, restored by cutting it back to its old length.

    This is the technique `InstanceSnapshot` already applies to `step_stats` and
    `completed`: given the append-only invariant, a length is enough, so capture
    is O(1) rather than O(len).
    """

    target: list
    length: int

    @classmethod
    def capture(cls, target: list) -> "TruncateList":
        return cls(target=target, length=len(target))

    def restore(self) -> None:
        del self.target[self.length :]


@dataclass(slots=True)
class MappingKeysChanged:
    """Writes to a mapping during one event, restored to what they displaced.

    `note` is called with the key *before* the write, so each entry records the
    value that was there -- or `MISSING` when the key was absent. Rollback then
    deletes what was added and puts back what was overwritten.

    Recording the displaced value is the whole point, and replaces an earlier
    `MappingKeysAdded` that stored keys alone and popped them on rollback. That
    was wrong whenever one event *re-pointed* an existing key: rollback deleted
    the entry instead of restoring its previous value. In the KV pool that meant
    a prefix hash moving from block A to block B, then the undo dropping the
    hash entirely while block A still held it and was still held by a live
    request -- content resident in the cache but no longer addressable, so
    prefix reuse was under-reported. See `tests/test_undo_full_state.py`.

    This does not diff the mapping: a key written without a `note` is a caller
    bug, and quietly reverting it would hide the bug instead of surfacing it.
    Only the *first* note for a key is kept, because the first is the only one
    holding the pre-event value.
    """

    target: MutableMapping
    changes: list[tuple[Hashable, Any]] = field(default_factory=list)
    seen: set[Hashable] = field(default_factory=set)

    @classmethod
    def capture(cls, target: MutableMapping) -> "MappingKeysChanged":
        return cls(target=target)

    def note(self, key: Hashable) -> None:
        """Record `key`'s current value, before the caller overwrites it.

        Must be called *before* the write. Calling it after would capture the
        new value as the old one, making rollback a no-op.
        """
        if key in self.seen:
            return
        self.seen.add(key)
        self.changes.append((key, self.target.get(key, MISSING)))

    def restore(self) -> None:
        # Newest first, mirroring `UndoLog.rollback`, so that two notes for one
        # key unwind to the earliest captured value.
        for key, old in reversed(self.changes):
            if old is MISSING:
                # Eviction can remove a key the same event added, so absence is
                # expected rather than exceptional.
                self.target.pop(key, None)
            else:
                self.target[key] = old
