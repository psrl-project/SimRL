"""L3 sampling rules: which finished groups make the next training batch."""

from __future__ import annotations

from dataclasses import dataclass

from ...component.rollout.engine import Request
from ..base import Cancel, CancelAction, PolicyContext


@dataclass
class FIFO:
    """Oldest-first selection by `(dispatch_time, group_id)`, used by AReaL."""

    def select(self, ready_groups: list[list[Request]], batch_size: int) -> list[Request] | None:
        if len(ready_groups) < batch_size:
            return None
        ordered = sorted(ready_groups, key=lambda g: (g[0].dispatch_time, g[0].group_id))
        return [req for group in ordered[:batch_size] for req in group]


@dataclass
class OffPolicySpan:
    """verl v1's admission-span window: `span(g) = v_train - admit_version + 1`.

    `drop` evicts finished groups past the threshold at publish time, since
    span only changes when `train_version` does. `wait` blocks sampling while
    any still-generating group has reached the threshold, and never evicts.
    """

    threshold: int | None
    strategy: str

    def can_sample(self, ctx: PolicyContext) -> bool:
        if self.threshold is None or self.strategy != "wait":
            return True
        return not any(
            ctx.state.train_version - g.admit_version + 1 >= self.threshold
            for g in ctx.state.groups
            if g.status != "done"
        )

    def on_publish(self, ctx: PolicyContext, new_version: int) -> list[Cancel]:
        if self.threshold is None or self.strategy != "drop":
            return []
        stale = sorted(
            g.group_id
            for g in ctx.state.groups
            if g.status == "done" and new_version - g.admit_version + 1 > self.threshold
        )
        return [Cancel(group_id, CancelAction.REPLACE) for group_id in stale]

    def select(
        self, ctx: PolicyContext, ready_groups: list[list[Request]], batch_size: int
    ) -> list[Request] | None:
        if len(ready_groups) < batch_size:
            return None
        admit_version = {g.group_id: g.admit_version for g in ctx.state.groups}
        ordered = sorted(ready_groups, key=lambda group: admit_version.get(group[0].group_id, 0))
        return [req for group in ordered[:batch_size] for req in group]
