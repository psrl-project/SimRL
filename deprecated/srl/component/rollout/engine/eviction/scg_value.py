"""Evict whoever loses the least by waiting, priced in expected trajectories."""

from __future__ import annotations

from .....value import TrajView, Valuator, traj_view
from ..request import Request
from .strategy import register_eviction


@register_eviction("scg_value")
class ScgValue:
    """Evicts whoever loses the least hit probability by being evicted.

    An eviction makes its victim recompute its context before generating again,
    so what it costs is `ctx / prefill_throughput` seconds of generating time
    out of that trajectory's own budget. Priced in `V`, the cheapest victim is
    the one whose chance of finishing in time that shrunken budget barely
    changes.

    **This is not the same as "the one with the most slack".** Slack and
    recompute cost move together: a trajectory far along is both comfortably
    ahead of its deadline and expensive to rebuild. Which effect wins is a
    property of the numbers, not something to assume, and that is the point of
    computing it rather than writing a rule. Measured on a batch with 521 tokens
    of budget left, a trajectory 9900 tokens deep lost its whole hit probability
    to an eviction while one 2000 deep lost 0.013, so the shallow one is the
    right victim there despite having less slack.

    **Never the one predicted to miss, though.** That trajectory is what the
    objective is trying to rescue, the prediction may be wrong, and the delay is
    what would make it miss. It is excluded by construction: a trajectory with
    `V` near zero has nothing left to lose, so this rule would pick it. The
    guard below drops candidates that are already beyond saving.

    Returns None until the valuator can answer, which leaves the engine's own
    order in place rather than ranking on noise.
    """

    name = "scg_value"

    def __init__(
        self,
        valuator: Valuator | None = None,
        prefill_throughput: float = 0.0,
        floor: float = 0.01,
    ):
        """
        Args:
            valuator (Valuator | None): Supplies `V`. Injected by whoever owns
                the length model, normally the routing strategy, so admission
                and eviction price trajectories identically.
            prefill_throughput (float): Tokens per second for the recompute.
            floor (float): Hit probability below which a trajectory is treated as
                beyond saving and excluded from the ranking. Such a trajectory
                has nothing left to lose, so it would otherwise be chosen every
                time, which is the one victim the objective must not pick.
        """
        self.valuator = valuator
        self.prefill_throughput = float(prefill_throughput)
        self.floor = float(floor)

    def bind(self, valuator: Valuator, prefill_throughput: float) -> None:
        """Attach the model and throughput once the fleet is built."""
        self.valuator = valuator
        self.prefill_throughput = float(prefill_throughput)

    def select(self, candidates: list[Request], now: float) -> Request | None:
        if self.valuator is None or not self.valuator.ready:
            return None
        rate = self._observed_rate(candidates, now)
        if rate <= 0.0:
            return None
        # A trajectory already past saving has a loss of zero and would win this
        # comparison every time, which is the one outcome the objective forbids:
        # it is what the schedule is trying to rescue, and evicting it is what
        # would finish it off. Rank only among those that can still land.
        savable = [
            req
            for req in candidates
            if self.valuator.hit_prob(traj_view(req), rate, now) > self.floor
        ]
        pool = savable or candidates
        return min(pool, key=lambda req: self._loss(traj_view(req), rate, now))

    def _loss(self, traj: TrajView, rate: float, now: float) -> float:
        """Return the hit probability this trajectory loses to an eviction."""
        return self.valuator.eviction_loss(traj, rate, now, self.prefill_throughput)

    def _observed_rate(self, candidates: list[Request], now: float) -> float:
        """Return the batch's achieved per-request rate.

        Measured rather than re-derived from the cost model: the engine calls
        this from inside a step, where the batch composition it would have to
        price is the one being changed.
        """
        rates = [
            req.generated / (now - req.admit_time)
            for req in candidates
            if now > req.admit_time and req.generated > 0
        ]
        if not rates:
            return 0.0
        rates.sort()
        return rates[len(rates) // 2]
