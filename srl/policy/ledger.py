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
    """Earliest weight version among completed members, or `admit_version` before any completes."""
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
    start_version: int | None = None
    """`None` until a member completes, then the minimum `v_min` over completed members."""
    status: str = "running"
    complete_time: float | None = None
    completed: set[int] = field(default_factory=set)


class GroupLedger:
    """Tracks group membership and lifecycle. The simulator calls the mutators."""

    def __init__(self) -> None:
        self._groups: dict[int, _Entry] = {}

    def begin(self, group: list[Request], version: int, now: float) -> None:
        group_id = group[0].group_id
        self._groups[group_id] = _Entry(
            members=list(group),
            admit_version=version,
            admit_time=now,
        )

    def complete(self, req: Request) -> None:
        """Record that `req` finished, completing its group once every member has.

        Only committed completions reach here, so `start_version` updates from
        committed tokens rather than from segments a speculative event may undo.
        """
        entry = self._groups.get(req.group_id)
        if entry is None or entry.status == "done":
            return
        entry.status = "partial"
        entry.completed.add(req.rid)
        if entry.start_version is None:
            entry.start_version = req.v_min
        else:
            entry.start_version = min(entry.start_version, req.v_min)
        if len(entry.completed) == len(entry.members):
            missing = [member.rid for member in entry.members if member.complete_time is None]
            assert not missing, f"Group {req.group_id} completed without times for {missing!r}."
            entry.status = "done"
            entry.complete_time = max(member.complete_time for member in entry.members)

    def drop(self, group_id: int) -> None:
        self._groups.pop(group_id, None)

    def peek(self, group_id: int) -> GroupView | None:
        entry = self._groups.get(group_id)
        if entry is None:
            return None
        return GroupView(
            group_id=group_id,
            admit_version=entry.admit_version,
            start_version=(entry.admit_version if entry.start_version is None else entry.start_version),
            status=entry.status,
            admit_time=entry.admit_time,
            complete_time=entry.complete_time,
            members=tuple(req.rid for req in entry.members),
        )

    def views(self) -> tuple[GroupView, ...]:
        return tuple(view for gid in self._groups if (view := self.peek(gid)) is not None)
