"""Pricing a scheduling decision in trajectories, which is what SCG counts.

Every decision here is one comparison:

    admit  <=>  V(newcomer at the diluted rate)  >  what the residents lose

Both sides are differences in `V`, the probability a trajectory is consumed
within its staleness budget, so both are counts of trajectories and can be
subtracted directly.

**Why no capacity bound.** An earlier design derived a resident limit from the
KV pool and refused past it. That reduces a scheduler to a rate limiter: the
limit treats the first request and the fiftieth alike, and the start of a
training cycle like its end. It also forbids a trade that is sometimes worth
making. Measured on dapo_32k, holding recompute to 15.1% of generated tokens
scored *worse* on SCG than letting it reach 24.2%, because the extra residents
bought more throughput than the recompute cost. A bound has nowhere to express
that. A price does.

**Where the bound comes back from.** `V` is bounded in [0, 1], so the same rate
dilution costs almost nothing when the residents have time to spare and a great
deal when they do not. The admission rule is therefore permissive early in a
cycle and strict at its end without being told to, and it answers per request
rather than per instance.
"""

from __future__ import annotations

from dataclasses import dataclass

from .deadline import TrainCycle, budget_tokens
from .length import LengthModel, TrajView
from .tool import NoToolCalls, ToolTimeModel


@dataclass(frozen=True, slots=True)
class BatchState:
    """The decode batch a candidate would join, as the cost model sees it."""

    n_decode: int
    ctxsum: float


class Valuator:
    """Prices admissions, evictions and pauses in expected trajectories.

    Usage:
        valuator = Valuator(model, cycle, max_staleness=2)
        net = valuator.admit_net(candidate, residents, r_before, r_after, now)
    """

    def __init__(
        self,
        model: LengthModel,
        cycle: TrainCycle,
        max_staleness: int,
        tools: ToolTimeModel | None = None,
    ) -> None:
        self.model = model
        self.cycle = cycle
        self.max_staleness = int(max_staleness)
        self.tools = tools or NoToolCalls()

    @property
    def ready(self) -> bool:
        """Whether both the length model and the cycle clock can answer."""
        return self.model.ready and self.cycle.known

    # --- the three primitives --------------------------------------------

    def budget(self, traj: TrajView, rate: float, now: float) -> float:
        """Return the tokens `traj` can still generate before its data ages out.

        Only generating time converts into tokens, so time the trajectory will
        spend blocked on a tool comes out of the slack first. That term is zero
        for every workload DryRun can currently express.
        """
        seconds = self.cycle.seconds_left(traj.v_traj, self.max_staleness, now)
        seconds -= self.tools.expected_wait(traj, seconds)
        return budget_tokens(rate, max(0.0, seconds))

    def hit_prob(self, traj: TrajView, rate: float, now: float) -> float:
        """Return `V`, the probability this trajectory is consumed in time."""
        return self.model.hit_prob(traj, self.budget(traj, rate, now))

    def urgency(self, traj: TrajView, rate: float, now: float, horizon: float = 1.0) -> float:
        """Return `U`, the hit probability lost per second of delay.

        What to rank on when the question is who to rescue. `V` itself answers
        "is this one safe", which ranks a trajectory that cannot be saved
        alongside one that needs nothing. `U` is large exactly where a
        trajectory is turning into a miss, and zero at both ends: a safe one has
        time to spare, and an impossible one has nothing left to lose. The
        second case is the safety property and needs no special case.
        """
        if rate <= 0.0:
            return 0.0
        here = self.hit_prob(traj, rate, now)
        later = self.hit_prob(traj, rate, now + horizon)
        return max(0.0, (here - later) / horizon)

    # --- the decisions ----------------------------------------------------

    def dilution_cost(
        self,
        residents: list[TrajView],
        rate_before: float,
        rate_after: float,
        now: float,
    ) -> float:
        """Return the trajectories the residents lose to a slower rate.

        A difference rather than a derivative. The slope at one batch size is
        nearly flat, so a marginal form prices one more request at almost zero
        and never refuses anything. What harms SCG is the accumulated effect:
        measured, going from 49 to 79 residents dropped each from 31.6 to 19.6
        tokens per second while fleet throughput stayed pinned.
        """
        if rate_after >= rate_before or not residents:
            return 0.0
        return sum(
            max(0.0, self.hit_prob(traj, rate_before, now) - self.hit_prob(traj, rate_after, now))
            for traj in residents
        )

    def eviction_loss(
        self,
        victim: TrajView,
        rate: float,
        now: float,
        prefill_throughput: float,
    ) -> float:
        """Return the trajectories lost by evicting `victim`.

        An eviction frees the victim's KV and makes it recompute its context
        when it resumes. The engine keeps its generated tokens, so what it
        actually loses is the time: `ctx / prefill_throughput` seconds come out
        of its own deadline budget before it can generate again.

        Priced, not forbidden. A victim with time to spare loses nothing by
        waiting and is free to evict, while one near its deadline is expensive.
        That is what lets the scheduler accept some thrashing when the admission
        it enables is worth more.
        """
        if prefill_throughput <= 0.0:
            return 0.0
        lost_seconds = victim.ctx / prefill_throughput
        before = self.hit_prob(victim, rate, now)
        # Charged against the budget, not the clock. Advancing `now` would also
        # advance the deadline arithmetic, so a trajectory whose cohort is
        # collected during the recompute reads as a total loss however little
        # work it had left. What an eviction actually takes is generating time,
        # which is `lost_seconds * rate` tokens off the budget.
        shrunk = max(0.0, self.budget(victim, rate, now) - lost_seconds * rate)
        after = self.model.hit_prob(victim, shrunk)
        return max(0.0, before - after)

    def admit_net(
        self,
        candidate: TrajView,
        residents: list[TrajView],
        rate_before: float,
        rate_after: float,
        now: float,
        evicted: list[TrajView] | None = None,
        prefill_throughput: float = 0.0,
    ) -> float:
        """Return the net trajectories gained by admitting `candidate`.

        Args:
            candidate: The trajectory being placed.
            residents: Trajectories already on the target instance.
            rate_before: Per-request token rate now.
            rate_after: Per-request token rate once the candidate joins.
            now: Current simulation time.
            evicted: Residents the admission would displace, if any.
            prefill_throughput: Tokens per second for their recompute.

        Returns:
            Expected trajectories gained minus lost. Positive means admit.
        """
        gain = self.hit_prob(candidate, rate_after, now)
        cost = self.dilution_cost(residents, rate_before, rate_after, now)
        for victim in evicted or ():
            cost += self.eviction_loss(victim, rate_before, now, prefill_throughput)
        return gain - cost

    def rank_evictable(
        self,
        residents: list[TrajView],
        rate: float,
        now: float,
        prefill_throughput: float,
    ) -> list[tuple[float, TrajView]]:
        """Return residents ordered by what evicting each would cost, cheapest first.

        The cheapest victim is one far enough ahead of its deadline that waiting
        out a recompute changes nothing. Deliberately not the one least likely
        to finish: that one is what the objective is trying to save, evicting it
        still costs a recompute, and the prediction that it will miss may be
        wrong.
        """
        scored = [
            (self.eviction_loss(traj, rate, now, prefill_throughput), traj)
            for traj in residents
        ]
        scored.sort(key=lambda item: item[0])
        return scored


def decode_rates(cost_model, batch: BatchState, candidate_ctx: int) -> tuple[float, float]:
    """Return the per-request token rate before and after one more request joins.

    Reads the same cost-model entry point the engine prices its own decode steps
    with, so the dilution a strategy predicts is the dilution the engine will
    deliver. Mirrors `NativeInstance.step_time_delta`.

    Returns:
        `(rate_before, rate_after)` in tokens per second per request. An empty
        batch reports a zero rate before, which makes dilution zero.
    """
    after_step = cost_model.step_time(
        batch.n_decode + 1,
        0,
        batch.ctxsum + candidate_ctx,
        batch.n_decode + 1,
    )
    rate_after = 1.0 / after_step if after_step > 0.0 else 0.0
    if batch.n_decode <= 0:
        return 0.0, rate_after
    before_step = cost_model.step_time(batch.n_decode, 0, batch.ctxsum, batch.n_decode)
    rate_before = 1.0 / before_step if before_step > 0.0 else 0.0
    return rate_before, rate_after
