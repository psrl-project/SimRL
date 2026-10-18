"""L3 sampling rules: which finished groups make the next training batch."""

from __future__ import annotations

from dataclasses import dataclass

from ...component.rollout.engine import Request


@dataclass
class FIFO:
    """Oldest-first selection by `(dispatch_time, group_id)`, used by AReaL."""

    def select(self, ready_groups: list[list[Request]], batch_size: int) -> list[Request] | None:
        if len(ready_groups) < batch_size:
            return None
        ordered = sorted(ready_groups, key=lambda g: (g[0].dispatch_time, g[0].group_id))
        return [req for group in ordered[:batch_size] for req in group]
