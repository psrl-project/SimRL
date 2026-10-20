"""Isolate a burst's long tail by pricing whole-burst placement candidates, not one request at a
time.

`begin_dispatch` additionally plans the whole tentative tail once per burst (`plan_burst`), scored
by the same buffer-pipeline forecast `InventoryDeadline` already uses -- but now across candidate
*partitions* of the burst, where it can actually discriminate (see
`docs/superpowers/specs/2026-10-20-burst-joint-placement-design.md` for why the per-request version
never could). `select_context` reads that plan for an admitted group's queued member, falling back
to `InventoryDeadline`'s own per-request logic when the plan has nothing for this request or its
planned instance is not offered this call. A staged group's admission is unchanged, inherited from
`InventoryDeadline`.
"""

from __future__ import annotations

from .....forecast.burst_plan import plan_burst
from ...engine import InstanceView, Request
from ..strategy import register_strategy
from .inventory_deadline import InventoryDeadline


@register_strategy("burst_isolation")
class BurstIsolation(InventoryDeadline):
    """`InventoryDeadline`'s admission logic, with jointly-planned placement for the queued tail."""

    name = "burst_isolation"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._plan: dict[int, int] | None = None

    def begin_dispatch(self, t: float) -> None:
        super().begin_dispatch(t)
        session = self._session
        self._plan = plan_burst(session, session.tail_order, self._eps(session)) if session else None

    def select_context(self, req: Request, candidates: list[InstanceView], _context) -> int | None:
        session = self._session
        if (
            session is None
            or req.group_id not in session.members
            or not self._plan
            or req.rid not in self._plan
        ):
            return super().select_context(req, candidates, _context)
        target = self._plan[req.rid]
        candidate_ids = {view.instance_id for view in candidates}
        if target not in candidate_ids:
            return super().select_context(req, candidates, _context)
        session.commit_assignment([req], {req.rid: target})
        del self._plan[req.rid]
        return next(i for i, view in enumerate(candidates) if view.instance_id == target)
