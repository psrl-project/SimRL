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
