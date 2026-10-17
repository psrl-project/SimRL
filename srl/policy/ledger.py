"""Simulator-owned group state, so policies never scan request lists."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..component.rollout.engine import Request


@dataclass(frozen=True)
class GroupView:
    """Immutable snapshot of one prompt group at the moment it is read."""

    group_id: int
    admit_version: int
    start_version: int
    """Earliest weight version that produced any token, or `admit_version` before generation."""
    status: str
    """`running | partial | done`."""
    admit_time: float
    complete_time: float | None
    members: tuple[int, ...]
    """Request ids of the group, in admission order."""


@dataclass
class _Entry:
    """Mutable ledger state for one live group."""

    members: list[Request]
    admit_version: int
    admit_time: float
    status: str = "running"
    complete_time: float | None = None
    completed: set[int] = field(default_factory=set)


class GroupLedger:
    """Tracks group membership and lifecycle. The simulator calls the mutators."""

    def __init__(self) -> None:
        self._groups: dict[int, _Entry] = {}
        self._rid_to_group: dict[int, int] = {}

    def begin(self, group: list[Request], version: int, now: float) -> None:
        group_id = group[0].group_id
        self._groups[group_id] = _Entry(
            members=list(group),
            admit_version=version,
            admit_time=now,
        )
        for req in group:
            self._rid_to_group[req.rid] = group_id

    def complete(self, rid: int, now: float) -> None:
        """Record that request `rid` finished, completing its group once every member has."""
        group_id = self._rid_to_group.get(rid)
        if group_id is None:
            return
        entry = self._groups[group_id]
        if entry.status == "done":
            return
        entry.status = "partial"
        entry.completed.add(rid)
        if all(req.rid in entry.completed for req in entry.members):
            entry.status = "done"
            entry.complete_time = max(
                (req.complete_time for req in entry.members if req.complete_time is not None),
                default=now,
            )

    def drop(self, group_id: int) -> None:
        entry = self._groups.pop(group_id, None)
        if entry is None:
            return
        for req in entry.members:
            self._rid_to_group.pop(req.rid, None)

    def peek(self, group_id: int) -> GroupView | None:
        entry = self._groups.get(group_id)
        if entry is None:
            return None
        return GroupView(
            group_id=group_id,
            admit_version=entry.admit_version,
            start_version=min(
                (req.v_min for req in entry.members),
                default=entry.admit_version,
            ),
            status=entry.status,
            admit_time=entry.admit_time,
            complete_time=entry.complete_time,
            members=tuple(req.rid for req in entry.members),
        )

    def views(self) -> tuple[GroupView, ...]:
        return tuple(view for gid in self._groups if (view := self.peek(gid)) is not None)
