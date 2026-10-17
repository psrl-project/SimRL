"""Simulator-owned group state, so policies never scan request lists."""

from __future__ import annotations

from dataclasses import dataclass

from ..component.rollout.engine import Request


@dataclass(frozen=True)
class GroupView:
    """Immutable snapshot of one prompt group at the moment it is read."""

    group_id: int
    admit_version: int
    start_version: int
    """Weight version the group was admitted under."""
    status: str
    """`running | partial | done | dropped`."""
    admit_time: float
    complete_time: float | None


class GroupLedger:
    """Tracks group membership and lifecycle. The simulator calls the mutators."""

    def __init__(self) -> None:
        self._members: dict[int, list[Request]] = {}
        self._rid_to_group: dict[int, int] = {}
        self._admit_version: dict[int, int] = {}
        self._admit_time: dict[int, float] = {}
        self._status: dict[int, str] = {}
        self._complete_time: dict[int, float | None] = {}
        self._completed: dict[int, set[int]] = {}

    def begin(self, group: list[Request], version: int, now: float) -> None:
        group_id = group[0].group_id
        self._members[group_id] = list(group)
        for req in group:
            self._rid_to_group[req.rid] = group_id
        self._admit_version[group_id] = version
        self._admit_time[group_id] = now
        self._status[group_id] = "running"
        self._complete_time[group_id] = None
        self._completed[group_id] = set()

    def complete(self, rid: int, now: float) -> None:
        """Record that request `rid` finished, completing its group once every member has."""
        group_id = self._rid_to_group.get(rid)
        if group_id is None:
            return
        self._status[group_id] = "partial"
        self._completed[group_id].add(rid)
        if all(req.rid in self._completed[group_id] for req in self._members[group_id]):
            self._status[group_id] = "done"
            self._complete_time[group_id] = now

    def drop(self, group_id: int) -> None:
        for req in self._members.pop(group_id, ()):
            self._rid_to_group.pop(req.rid, None)
        self._admit_version.pop(group_id, None)
        self._admit_time.pop(group_id, None)
        self._status.pop(group_id, None)
        self._complete_time.pop(group_id, None)
        self._completed.pop(group_id, None)

    def peek(self, group_id: int) -> GroupView | None:
        status = self._status.get(group_id)
        if status is None:
            return None
        return GroupView(
            group_id=group_id,
            admit_version=self._admit_version[group_id],
            start_version=self._admit_version[group_id],
            status=status,
            admit_time=self._admit_time[group_id],
            complete_time=self._complete_time[group_id],
        )

    def views(self) -> tuple[GroupView, ...]:
        return tuple(view for gid in self._status if (view := self.peek(gid)) is not None)
