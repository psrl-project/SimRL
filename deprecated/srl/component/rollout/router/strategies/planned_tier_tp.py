"""Route each request to the tier whose length range holds its estimated remaining.

The planner decides the tiers once per weight update. This strategy only follows
them: a request goes to the instances of the tier whose `[lo, hi)` contains its
estimated remaining length, balanced by resident count inside the tier, and waits
when every instance of that tier is at its cap. Queued requests are offered
longest first, and a full tier never blocks the others.

With the `oracle` length source the planner's own per-request assignment is
followed exactly. With `online`, estimates come from prompt history and from how
far a request has run, and a request that outgrows its tier is paused and
re-placed on a higher one.
"""

from __future__ import annotations

from ....._logging import srl_logger
from ...engine import InstanceView, Request
from ..estimator import LengthEstimator
from ..strategy import RoutingContext, register_strategy


@register_strategy("planned_tier_tp")
class PlannedTierTp:
    """Follows the tiers of the latest `TierPlan`."""

    name = "planned_tier_tp"

    def __init__(
        self,
        order: str = "longest_first",
        length_source: str = "oracle",
        rescue: bool = False,
        rescue_min_generated: int = 512,
        rescue_slots: int = 8,
        use_history: bool = True,
        prior_total: float = 4096.0,
        noise_sigma: float = 0.0,
        miss_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        """
        Args:
            order: `longest_first` offers the longest queued request first, `fifo`
                keeps queue order.
            length_source: `oracle`, `online` or the controlled `noisy` predictor.
            rescue: Pause a request that has outgrown its tier and re-place it.
            rescue_min_generated: Tokens a request must have produced before it can
                be rescued, so a guess from no progress never moves it.
            rescue_slots: Slots a tier keeps beyond its cap for rescued requests. A
                tier is sized to its planned load, so without headroom it is full.
            use_history: Whether the online estimator uses prompt history.
            prior_total: Expected total length before any completion is seen.
            noise_sigma: Log-sd of the `noisy` predictor's error.
            miss_rate: Fraction of requests the `noisy` predictor misses.
            seed: Seed of the per-request noise.

        Raises:
            ValueError: If `order` is unknown.
        """
        if order not in ("fifo", "longest_first"):
            raise ValueError(f"Unknown order {order!r}, expected fifo or longest_first.")
        self.order = order
        self.rescue = bool(rescue)
        self.rescue_min_generated = int(rescue_min_generated)
        self.rescue_slots = int(rescue_slots)
        self.estimator = LengthEstimator(
            length_source,
            prior_total=prior_total,
            use_history=use_history,
            noise_sigma=noise_sigma,
            miss_rate=miss_rate,
            seed=seed,
        )
        self._tiers: tuple = ()
        self._tier_of_instance: dict[int, int] = {}
        self._assignment: dict[int, int] = {}
        self._pinned: dict[int, int] = {}
        self._free: dict[int, int] = {}
        self.n_rescued = 0

    # --- learning from the run -------------------------------------------

    def observe_completion(self, req: Request) -> None:
        self.estimator.observe(req)
        self._pinned.pop(req.rid, None)

    def estimate_plan_remaining(self, reqs) -> list[float]:
        """Estimates handed to the planner at a weight update."""
        return list(self.estimator.plan_remaining(reqs))

    @property
    def plans_by_assignment(self) -> bool:
        """Whether the planner's own per-request mapping is exact enough to follow."""
        return self.estimator.mode in ("oracle", "noisy")

    # --- plan --------------------------------------------------------------

    def update_plan(self, plan) -> None:
        """Adopt the tiers of a new plan. A plan without tiers leaves one open tier."""
        self._tiers = tuple(getattr(plan, "tiers", ()))
        self._assignment = dict(getattr(plan, "assignment", {}) or {}) if self.plans_by_assignment else {}
        self._pinned = {}
        self._tier_of_instance = {iid: index for index, tier in enumerate(self._tiers) for iid in tier.instances}

    def estimate(self, req: Request) -> float:
        return self.estimator.remaining(req)

    def _tier_by_estimate(self, req: Request) -> int:
        """Tier whose length range holds the request's current estimate."""
        rho = self.estimate(req)
        for index, tier in enumerate(self._tiers):
            if rho >= tier.lo:
                return index
        return len(self._tiers) - 1

    def _tier_index(self, req: Request) -> int:
        pinned = self._pinned.get(req.rid)
        if pinned is not None:
            return pinned
        planned = self._assignment.get(req.rid)
        if planned is not None:
            return planned
        return self._tier_by_estimate(req)

    def _allowed(self, req: Request, views: list[InstanceView]) -> list[int]:
        if not self._tiers:
            return list(range(len(views)))
        index = self._tier_index(req)
        tier = self._tiers[index]
        cap = tier.cap + (self.rescue_slots if req.rid in self._pinned else 0)
        return [
            i for i, v in enumerate(views) if self._tier_of_instance.get(v.instance_id) == index and v.n_active < cap
        ]

    # --- routing protocol --------------------------------------------------

    def narrow_candidates(self, req: Request, candidates: list[InstanceView], context: RoutingContext) -> list[int]:
        return self._allowed(req, candidates)

    def allow_version_bypass(self, _req: Request, _context: RoutingContext) -> bool:
        return True

    def pick_queued(self, window: list[Request], views: list[InstanceView]) -> Request | None:
        ordered = window if self.order == "fifo" else sorted(window, key=lambda r: -self.estimate(r))
        for req in ordered:
            if self._allowed(req, views):
                return req
        return None

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        return min(range(len(candidates)), key=lambda i: (candidates[i].n_active, candidates[i].total_ctx))

    # --- rescue --------------------------------------------------------------

    def observe_fleet(self, views: list[InstanceView]) -> None:
        """Count free slots per tier before a round of pause decisions."""
        free = {index: 0 for index in range(len(self._tiers))}
        for v in views:
            index = self._tier_of_instance.get(v.instance_id)
            if index is not None:
                free[index] += max(0, self._tiers[index].cap + self.rescue_slots - v.n_active)
        self._free = free

    def choose_pause(self, view: InstanceView, t: float):
        """Pause the request that has outgrown this instance's tier the most."""
        if not self.rescue or not self._tiers:
            return None
        here = self._tier_of_instance.get(view.instance_id)
        if here is None or here == 0:
            return None
        best, best_target, best_rho = None, None, -1.0
        for _, req in view.decoding():
            if req.generated < self.rescue_min_generated:
                continue
            # The plan's assignment is only the initial placement. Whether the request
            # has outgrown its tier is judged on what its progress says now.
            target = self._tier_by_estimate(req)
            if target >= here:
                continue
            rho = self.estimate(req)
            if self._free.get(target, 0) > 0 and rho > best_rho:
                best, best_target, best_rho = req, target, rho
        if best is None:
            return None
        self._free[best_target] -= 1
        self._pinned[best.rid] = best_target
        self.n_rescued += 1
        if self.n_rescued <= 5 or self.n_rescued % 50 == 0:
            srl_logger.info(
                f"Rescue #{self.n_rescued}: rid {best.rid} at {best.generated} tokens, "
                f"estimate {best_rho:.0f} remaining, tier {here} to {best_target}."
            )
        return best
