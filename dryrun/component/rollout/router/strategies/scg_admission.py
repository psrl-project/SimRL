"""Admission as a trade, priced per trajectory rather than capped per instance.

Every placement answers one question: is what this trajectory stands to gain
worth what it costs the ones already running? Both sides are counted in expected
trajectories, which is what SCG measures, so they subtract directly:

    admit  <=>  V(newcomer)  >  dilution of the residents  +  cost of any eviction

**No capacity limit.** An earlier version derived a resident bound from the KV
pool and refused past it. A bound cannot express that a particular trajectory is
worth a little thrashing, and measured on dapo_32k that mattered: holding
recompute to 15.1% of generated tokens scored worse than letting it reach 24.2%,
because the extra residents paid for themselves. So eviction is priced here, not
forbidden.

The bound comes back anyway, as a consequence rather than a setting. `V` is
bounded, so a given slowdown costs the residents nothing when they have time to
spare and a great deal when they do not. The same rule is permissive at the
start of a training cycle and strict at its end, and it decides per request.
"""

from __future__ import annotations

from .....value import (
    BatchState,
    TrainCycle,
    Valuator,
    build_length_model,
    decode_rates,
    traj_view,
)
from ...engine import InstanceView, Request
from ..strategy import RoutingContext, register_strategy
from .request_num_balance import RequestNumBalance


@register_strategy("scg_admission")
class ScgAdmission:
    """Admits, evicts and pauses by comparing expected trajectories gained and lost."""

    name = "scg_admission"
    requires_cost_model = True

    def __init__(
        self,
        max_staleness: int = 1,
        length_model: str = "empirical",
        length_model_args: dict | None = None,
        training_cycle_s: float = 0.0,
        decay: float = 0.9,
        pause: bool = True,
        price_eviction: bool = True,
    ) -> None:
        """
        Args:
            max_staleness (int): Staleness bound the deadline is derived from.
                Must match `job.max_staleness`, since a trajectory consumed later
                than this earns nothing. The one value this needs, and it is the
                independent variable of the metric rather than a tuning knob.
            length_model (str): Name in `LENGTH_MODEL_REGISTRY`.
            length_model_args (dict | None): Constructor arguments for it.
            training_cycle_s (float): Seed for the cycle estimate, used until the
                first interval is observed.
            decay (float): Factor applied to the length model's evidence at each
                weight publish, so it tracks a policy whose outputs lengthen.
            pause (bool): Whether a resident may be paused to free bandwidth.
            price_eviction (bool): Whether admission accounts for the recompute
                an eviction would force. False isolates the dilution term.
        """
        self.valuator = Valuator(
            build_length_model(length_model, **(length_model_args or {})),
            TrainCycle(seed_s=training_cycle_s),
            max_staleness=max_staleness,
        )
        self.decay = float(decay)
        self.pause = bool(pause)
        self.price_eviction = bool(price_eviction)
        self._balance = RequestNumBalance()
        self._solo_rate = 0.0

        self.n_admitted = 0
        self.n_declined = 0
        self.n_free = 0
        """Placements that harmed nobody, so value had nothing to weigh.

        Counted apart from a priced admission: a run that is mostly free
        placements never exercised the comparison, which is the difference
        between a null result and an untested one.
        """
        self.n_paused = 0
        self.n_fallback = 0
        """Placements made by load balancing because the valuator could not answer.

        A large count means the run never exercised the mechanism, which is the
        difference between a null result and an untested one.
        """

    # --- simulator callbacks ---------------------------------------------

    def observe_train_cycle(self, train_version: int, now: float) -> None:
        """Learn the cycle length and discount the length model. Called at publish."""
        self.valuator.cycle.observe(train_version, now)
        self.valuator.model.decay(self.decay)

    def observe_completion(self, req: Request) -> None:
        """Feed a finished trajectory to the length model."""
        self.valuator.model.observe(traj_view(req), req.generated)

    def bind_rates(self, cost_model) -> None:
        """Record the rate one request sees with no competition.

        Used to ask whether a queued trajectory could still land if the fleet
        gave it everything. Deliberately the optimistic bound: a pessimistic one
        would write off work the fleet has not yet failed to deliver.
        """
        seconds = cost_model.step_time(1, 0, 0.0, 1)
        self._solo_rate = 1.0 / seconds if seconds > 0.0 else 0.0

    def bind_eviction(self, eviction, prefill_throughput: float) -> None:
        """Give an eviction strategy the same valuator admission uses.

        Called by the router once the fleet exists. A strategy that does not
        price trajectories ignores this.
        """
        binder = getattr(eviction, "bind", None)
        if callable(binder):
            binder(self.valuator, prefill_throughput)

    # --- routing ----------------------------------------------------------

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        """Fall back to load balancing when the router gives no clock."""
        return self._balance.select(req, candidates)

    def select_context(
        self,
        req: Request,
        candidates: list[InstanceView],
        context: RoutingContext,
    ) -> int | None:
        """Place `req` where it is worth the most, or decline if nowhere is.

        Before the length model can answer, every `V` is the same number, so the
        gain and the dilution cancel and everything is admitted. That is the
        correct no-information behaviour: with no evidence about who would be
        harmed, a refusal has nothing behind it.
        """
        if not self.valuator.ready:
            self.n_fallback += 1
            return self._balance.select(req, candidates)

        traj = traj_view(req)
        free: list[int] = []
        best_index: int | None = None
        best_net: float | None = None
        for index, view in enumerate(candidates):
            gain, cost = self._price(traj, view, context.now)
            if cost <= 0.0:
                free.append(index)
                continue
            net = gain - cost
            if best_net is None or net > best_net:
                best_index, best_net = index, net

        # Two questions, asked in order, because a single net confuses them.
        # "Does anyone get hurt" is not the same as "is this worth anything",
        # and both produce a net of zero: an empty instance harms nobody, and a
        # trajectory that cannot finish has nothing to gain. Comparing the net
        # against zero made the rule refuse everything in the second case and
        # accept everything in the first, which is a livelock either way.
        if free:
            self.n_free += 1
            local = self._balance.select(req, [candidates[i] for i in free])
            return free[local if local is not None else 0]
        if best_index is not None and best_net > 0.0:
            self.n_admitted += 1
            return best_index
        self.n_declined += 1
        return None

    def rank_queued(self, window: list[Request], now: float) -> Request | None:
        """Return the queued request to place next, or None to keep heap order.

        A trajectory whose deadline has already passed cannot earn SCG whatever
        the fleet does for it, so it goes last. Not refused, though: the whole
        queue can consist of such requests, and refusing them all would idle the
        GPUs over work that costs nothing to run. They are simply never
        preferred over a request that can still land.

        The rate used here is the one the request would see with no competition,
        which is deliberately optimistic. The question being asked is "could
        this still be saved", and answering it against a pessimistic rate would
        write off trajectories the fleet has not actually failed yet.
        """
        if not self.valuator.ready or len(window) < 2:
            return None
        savable = [
            req
            for req in window
            if self.valuator.hit_prob(traj_view(req), self._solo_rate, now) > 0.0
        ]
        # All of them are past saving, so the heap's own order is as good as any
        # and the placement still happens.
        if not savable or len(savable) == len(window):
            return None
        return savable[0]

    def choose_pause(self, view: InstanceView, now: float):
        """Return a resident worth pausing to speed the others up, or None.

        Admission only governs new work, so a batch admitted early in a cycle
        stays wide for the rest of it however the deadline moves. Pausing is how
        that stock is corrected, and it is the same comparison: what the others
        gain from the freed bandwidth against what the victim loses by waiting.
        """
        if not (self.pause and self.valuator.ready):
            return None
        pairs = view.decoding()
        if len(pairs) < 2:
            return None
        n_decode, ctxsum = view.decode_batch()
        if n_decode < 2:
            return None
        rate_now = self._rate(view, n_decode, ctxsum)
        rate_freed = self._rate(view, n_decode - 1, ctxsum - ctxsum / n_decode)
        if rate_now <= 0.0 or rate_freed <= rate_now:
            return None

        residents = [projection for projection, _ in pairs]
        best_request = None
        best_net = 0.0
        for projection, request in pairs:
            others = [other for other in residents if other is not projection]
            # What the survivors gain is a dilution cost run backwards.
            gain = self.valuator.dilution_cost(others, rate_freed, rate_now, now)
            loss = self.valuator.eviction_loss(
                projection, rate_now, now, view.prefill_throughput()
            )
            net = gain - loss
            if net > best_net:
                best_request, best_net = request, net
        if best_request is None:
            return None
        self.n_paused += 1
        return best_request

    # --- internals --------------------------------------------------------

    def _price(self, traj, view: InstanceView, now: float) -> tuple[float, float]:
        """Return `(gain, cost)` for placing `traj` on `view`, both in trajectories.

        Kept as a pair rather than a difference so the caller can tell a free
        placement from a worthless one. Both are zero-cost, and only one of them
        is a reason to refuse.
        """
        n_decode, ctxsum = view.decode_batch()
        rate_before, rate_after = decode_rates(
            view.cost_model,
            BatchState(n_decode=n_decode, ctxsum=ctxsum),
            candidate_ctx=traj.ctx,
        )
        residents = view.residents()
        gain = self.valuator.hit_prob(traj, rate_after, now)
        cost = self.valuator.dilution_cost(residents, rate_before, rate_after, now)
        for victim in self._would_evict(traj, residents, view, rate_before, now):
            cost += self.valuator.eviction_loss(
                victim, rate_before, now, view.prefill_throughput()
            )
        return gain, cost

    def _would_evict(self, traj, residents, view: InstanceView, rate: float, now: float):
        """Return the residents this placement would displace, cheapest first.

        Sized on what the batch is predicted to need rather than what it holds:
        at admission a request owns only its prompt, so a comparison against
        current occupancy reports an empty pool for a batch already committed to
        overflowing it.
        """
        if not self.price_eviction or not residents:
            return []
        pool = view.kv_blocks * view.block_size
        if pool <= 0:
            return []
        projected = self._projected_ctx(traj) + sum(
            self._projected_ctx(other) for other in residents
        )
        overrun = projected - pool
        if overrun <= 0:
            return []
        ranked = self.valuator.rank_evictable(
            residents, rate, now, view.prefill_throughput()
        )
        victims = []
        freed = 0.0
        for _, candidate in ranked:
            victims.append(candidate)
            freed += self._projected_ctx(candidate)
            if freed >= overrun:
                break
        return victims

    def _projected_ctx(self, traj) -> float:
        """Return the context depth this trajectory is predicted to reach."""
        final = self.valuator.model.expected_final_length(traj)
        return max(float(traj.ctx), traj.prompt_len + final)

    def _rate(self, view: InstanceView, n_decode: int, ctxsum: float) -> float:
        """Return the per-request token rate of a hypothetical decode batch."""
        if n_decode <= 0:
            return 0.0
        seconds = view.cost_model.step_time(n_decode, 0, max(0.0, ctxsum), n_decode)
        return 1.0 / seconds if seconds > 0.0 else 0.0
