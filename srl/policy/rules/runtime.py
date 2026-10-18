"""L2 runtime rules: which live groups a weight publication cancels."""

from __future__ import annotations

from dataclasses import dataclass

from ..base import Cancel, CancelAction, PolicyContext


@dataclass
class CancelBelowVersion:
    """Cancel every live group dispatched before the staleness horizon.

    Keys on `admit_version`, the version stamped once at placement and never
    rewritten, matching how ROLL, prime-rl and NeMo-RL key their own cancel
    window: on dispatch version, not on when generation actually started.
    """

    horizon: int
    action: CancelAction

    def on_publish(self, ctx: PolicyContext, new_version: int) -> list[Cancel]:
        floor = new_version - self.horizon
        stale = sorted(g.group_id for g in ctx.state.groups if g.admit_version < floor)
        return [Cancel(group_id, self.action) for group_id in stale]


@dataclass
class VersionSpanCap:
    """Dressage's L2: cap how many distinct versions one trajectory may span.

    Counts the union of every member's segment versions plus the version
    about to be applied by this publish, since that is what a resumed
    partial rollout would add next.
    """

    max_preempts: int | None
    action: CancelAction

    def on_publish(self, ctx: PolicyContext, new_version: int) -> list[Cancel]:
        if self.max_preempts is None:
            return []
        by_group: dict[int, set[int]] = {}
        for req in ctx.state.inflight:
            versions = by_group.setdefault(req.group_id, {new_version})
            versions.update(segment.version for segment in req.segments)
        cancels = [
            Cancel(group_id, self.action)
            for group_id, versions in sorted(by_group.items())
            if len(versions) - 1 > self.max_preempts
        ]
        return cancels
