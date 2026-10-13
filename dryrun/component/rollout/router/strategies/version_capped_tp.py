"""Route stale work to the highest TP tier under a shared active-request cap."""

from __future__ import annotations

from ...engine import InstanceView, Request
from ..strategy import RoutingContext, register_strategy
from .request_num_balance import RequestNumBalance


@register_strategy("version_capped_tp")
class VersionCappedTp:
    """Enforce version tiers while balancing within each allowed tier."""

    name = "version_capped_tp"
    requires_active_cap = True

    def __init__(self, reference_version: int = 0) -> None:
        self.reference_version = int(reference_version)
        self._balance = RequestNumBalance()
        self.n_stale_placements = 0
        self.n_current_placements = 0

    def set_reference_version(self, version: int) -> None:
        """Set the version that separates stale from current work."""
        self.reference_version = int(version)

    def update_plan(self, plan) -> None:
        """Accept a fleet update. Candidate views carry the current TP shape."""

    def narrow_candidates(
        self,
        req: Request,
        candidates: list[InstanceView],
        context: RoutingContext,
    ) -> list[int]:
        """Return the instance tier this request may enter."""
        if req.v_traj < self.reference_version:
            return [
                index
                for index, view in enumerate(candidates)
                if view.tp == context.fleet_max_tp
            ]
        if context.blocked_versions:
            return [
                index
                for index, view in enumerate(candidates)
                if view.tp == context.fleet_min_tp
                and context.fleet_min_tp < context.fleet_max_tp
            ]
        return list(range(len(candidates)))

    def allow_version_bypass(
        self,
        _req: Request,
        _context: RoutingContext,
    ) -> bool:
        """Allow newer versions to use tiers that a blocked version cannot."""
        return True

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        """Balance a request within the tier selected by the router."""
        if req.v_traj < self.reference_version:
            self.n_stale_placements += 1
        else:
            self.n_current_placements += 1
        return self._balance.select(req, candidates)
