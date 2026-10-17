"""Prefer high-TP instances for old-version requests."""

from __future__ import annotations

from ...engine import InstanceView, Request
from ..strategy import register_strategy
from .request_num_balance import RequestNumBalance


@register_strategy("version_aware_tp")
class VersionAwareTp:
    """Prefer high TP for the old-version tail and balance current work."""

    name = "version_aware_tp"

    def __init__(self, reference_version: int = 0) -> None:
        self._reference_version = int(reference_version)
        self._balance = RequestNumBalance()
        self.n_old_to_high_tp = 0
        self.n_old_placements = 0

    def set_reference_version(self, version: int) -> None:
        self._reference_version = int(version)

    def update_plan(self, plan) -> None:
        """Accept a fleet update; TP is read directly from candidate views."""

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        if req.v_traj >= self._reference_version:
            return self._balance.select(req, candidates)

        self.n_old_placements += 1
        best_tp = max(view.tp for view in candidates)
        narrowed = [i for i, view in enumerate(candidates) if view.tp == best_tp]
        local = self._balance.select(req, [candidates[i] for i in narrowed])
        chosen = narrowed[local if local is not None else 0]
        if candidates[chosen].tp > min(view.tp for view in candidates):
            self.n_old_to_high_tp += 1
        return chosen

