"""The three shapes of captured old value this engine actually needs.

Each mirrors a technique the original `InstanceSnapshot` arrived at by hand:
scalar fields stored and assigned back, append-only lists restored by length
rather than by copy, and additions to a mapping removed by key. Naming them
makes the techniques reusable instead of re-derived per subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, MutableMapping


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

        Raises:
            AttributeError: If any name is absent, so a typo fails where it was
                written rather than silently at rollback.
        """
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
class MappingKeysAdded:
    """Keys added to a mapping during one event, removed on rollback.

    Only noted keys are removed. This does not diff the mapping, because a key
    added without a `note` is a caller bug and quietly removing it would hide
    the bug instead of surfacing it.
    """

    target: MutableMapping
    keys: list[Hashable] = field(default_factory=list)

    @classmethod
    def capture(cls, target: MutableMapping) -> "MappingKeysAdded":
        return cls(target=target)

    def note(self, key: Hashable) -> None:
        """Record that `key` was just added."""
        self.keys.append(key)

    def restore(self) -> None:
        for key in self.keys:
            # Eviction can remove a key the same event added, so absence is
            # expected rather than exceptional.
            self.target.pop(key, None)
