"""ROLL with proactive abandonment of trajectories that cannot meet their deadline.

**The measurement this exists to fix.** Under `roll`, which is the only policy that
actually enforces staleness by discarding aged cohorts, 30-46% of every token the
rollout fleet generates is spent on trajectories that are later GC-dropped, and the
dropped ones are *longer* than the consumed ones (mean generated 3492 against 2136
on the heavy-tail trace). That work earns zero SCG credit: the batch slot, the KV
blocks and the memory bandwidth all went to a trajectory the trainer threw away.

**Why abandoning beats accelerating.** The paper's §9 argues for giving the deep
tail scarce low-latency capacity. Measured here, accelerating stragglers raised the
wasted fraction (29.9% -> 34.1%) because the accelerated trajectories were dropped
anyway, and attempts to isolate them by routing paid a 65% re-prefill tax. A
straggler that cannot finish before its cohort ages out is the paper's own
**Concede-H** case (§6.7): its marginal value is identically zero, so the correct
action is to stop paying for it, not to pay more. This policy acts on that at the
only place the accounting is sound -- the policy's own `expire` hook.

**Where the deadline comes from.** `roll` GC's cohort `v` at train version
`v + max_staleness + 1` (see `RollPolicy.expire`: `max_gc_step = (E-1) - S`). So a
trajectory admitted at `v`, evaluated when the next version will be `E`, has

    cycles_left = v + max_staleness + 1 - E

training cycles of wall clock left. Multiplying by the observed cycle duration
gives the time it actually has, and the trajectory is doomed when its projected
remaining work exceeds that.

**Group granularity.** A prompt group is consumed only once every response in it
finishes, so a group's completion time is the max over its members and a group is
doomed when its slowest member is. Abandoning per response would orphan the
siblings, which `Simulator._discard` would then have to widen anyway.
"""

from __future__ import annotations

from ..component.rollout.engine import Request
from .base import SimState
from .roll import RollPolicy


class RollDoomPolicy(RollPolicy):
    """ROLL plus a feasibility test that abandons doomed groups early.

    Everything about admission, consumption and GC is inherited unchanged, so with
    ``abandon=False`` this *is* `RollPolicy` and serves as the matched baseline for
    its own ablation.
    """

    name = "roll_doom"

    def __init__(
        self,
        max_inflight: int | None = None,
        abandon: bool = True,
        safety_margin: float = 1.0,
        min_generated: int = 256,
        training_cycle_s: float = 0.0,
        require_observed_waste: bool = True,
        aggressive_margin: float = 0.7,
        waste_rate_trusted: float = 0.15,
    ) -> None:
        """
        Args:
            max_inflight: Passed through to `RollPolicy`.
            abandon: Whether to abandon doomed groups at all. False reproduces
                `RollPolicy` exactly, which is what the ablation needs.
            safety_margin: Multiplier on the time a trajectory is judged to have.
                Above 1.0 is lenient (abandon only the clearly hopeless), below 1.0
                is aggressive. The right value trades wasted tokens against
                trajectories that would have just made it, and 1.0 is the neutral
                starting point rather than a tuned one.
            min_generated: Tokens a response must have produced before its observed
                rate is trusted. A request that has just been placed has no rate to
                project from, and abandoning it on noise would discard work for no
                reason.
            training_cycle_s: Seed for the cycle-duration estimate, used only until
                the first interval has been observed. Zero means "wait for a real
                measurement", which disables abandonment for the first cycle.
            require_observed_waste: Whether to wait for GC to actually discard
                something before abandoning anything.

                **Why this defaults to True.** Abandonment can only recover
                capacity that the baseline was already losing. Measured on two
                workloads: the heavy-tail trace GC-drops 528-752 groups and
                abandonment wins (SCG 1.11x at S=1, 1.47x at S=2), while the
                aligned DAPO trace GC-drops *zero* -- its lengths are uniform
                (p99/p50 = 1.01), so no trajectory is ever hopeless -- and there
                abandonment only cut work that would have landed, losing 0.861x.
                Gating on an observed GC drop makes the policy inert on workloads
                it cannot help, which is the difference between a mechanism and a
                gamble. Note the predictor is length *dispersion*, not tail token
                share: DAPO's tail holds 92.9% of tokens and still has nothing to
                abandon.
            aggressive_margin: Margin used when waste is scarce. A binary gate is
                too coarse on its own: at DAPO S=2 the baseline discards 32 of 1568
                groups, so the gate opens, and there only an aggressive margin pays
                (1.058x at 0.7, losses at 1.0 and 1.5). This is the floor the
                margin is scaled toward when evidence of waste is thin.
            waste_rate_trusted: GC-discard fraction at or above which
                `safety_margin` is used unmodified. Below it the margin interpolates
                down to `aggressive_margin`. The measured transition sits between
                DAPO S=2 (2.0% discarded, only 0.7 wins) and heavy-tail S=2 (26.9%
                discarded, every margin wins), so the default sits between them.
        """
        super().__init__(max_inflight=max_inflight)
        if safety_margin <= 0.0:
            raise ValueError(f"safety_margin must be positive, got {safety_margin!r}.")
        if min_generated < 1:
            raise ValueError(f"min_generated must be >= 1, got {min_generated!r}.")
        if aggressive_margin <= 0.0:
            raise ValueError(
                f"aggressive_margin must be positive, got {aggressive_margin!r}."
            )
        if not 0.0 < waste_rate_trusted <= 1.0:
            raise ValueError(
                f"waste_rate_trusted must lie in (0, 1], got {waste_rate_trusted!r}."
            )
        self.abandon = bool(abandon)
        self.safety_margin = float(safety_margin)
        self.min_generated = int(min_generated)
        self.require_observed_waste = bool(require_observed_waste)
        self.aggressive_margin = float(aggressive_margin)
        self.waste_rate_trusted = float(waste_rate_trusted)
        self.consumed_groups = 0
        """Groups kept for training, the denominator of the observed waste rate."""
        self._cycle_s = float(training_cycle_s)
        self._last_advance_t: float | None = None
        self.abandoned = 0
        """Groups abandoned as infeasible, so the mechanism's reach is measurable."""
        self.abandoned_tokens_saved = 0
        """Tokens the abandoned groups would still have had to generate.

        The quantity the policy is trying to recover. Reported rather than inferred,
        because the whole claim is about wasted work and a claim about waste should
        carry its own measurement.
        """

    # ------------------------------------------------------------------
    # Cycle-duration tracking
    # ------------------------------------------------------------------

    def on_train_version_advance(self, train_version: int, st: SimState) -> None:
        """Advance the budget, and learn how long a training cycle takes.

        The deadline is denominated in training cycles, so the policy needs their
        wall-clock length. Observed rather than configured: it depends on the batch
        that was actually taken, and a stale constant would mis-date every deadline
        in the same direction.

        NOTE(claude): `value/deadline.py` computes the same deadline and is not
        shared with this policy. It subtracts the time already elapsed in the
        current cycle, which this does not, so a trajectory judged here late in a
        cycle is granted up to one extra cycle of budget. The two differ in
        behaviour, not only in structure, so unifying them would move this
        policy's measured results rather than refactor them.
        """
        super().on_train_version_advance(train_version, st)
        if self._last_advance_t is not None:
            interval = st.now - self._last_advance_t
            if interval > 0.0:
                # EWMA rather than last-value: a single long step should not
                # suddenly declare half the fleet feasible.
                self._cycle_s = (
                    interval if self._cycle_s <= 0.0 else 0.7 * self._cycle_s + 0.3 * interval
                )
        self._last_advance_t = st.now

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    def _projected_seconds(self, req: Request, now: float) -> float | None:
        """Seconds `req` still needs, or None when there is no basis to judge.

        Uses the response's own achieved rate since admission, which already folds
        in queueing, preemption and batch dilution -- the three things that actually
        make a trajectory late. A model-based rate would have to re-derive all of
        them.

        TODO(lhy): `req.remaining` is exact here because the trace fixes every
        target length. A real scheduler does not know it and would substitute the
        hazard-model mean of the paper's §6.2. The structure of this test does not
        change when that happens -- only the numerator -- so the substitution is
        isolated to this one line.
        """
        if req.generated < self.min_generated:
            return None
        elapsed = now - req.admit_time
        if elapsed <= 0.0:
            return None
        rate = req.generated / elapsed
        if rate <= 0.0:
            return None
        return req.remaining / rate

    def _group_is_doomed(
        self, group: list[Request], cycles_left: int, now: float, margin: float
    ) -> bool:
        """Whether `group` cannot finish before its cohort is collected.

        Args:
            group: Every live response in the prompt group.
            cycles_left: Training cycles before this cohort is GC'd.
            now: Current simulation time.
            margin: Multiplier on the budget, from `_effective_margin`.

        Returns:
            True when the slowest response's projection overruns the deadline. A
            group with no judgeable member is never abandoned, so missing evidence
            keeps work alive rather than killing it.
        """
        budget = cycles_left * self._cycle_s * margin
        if budget <= 0.0:
            return False
        worst: float | None = None
        for req in group:
            projected = self._projected_seconds(req, now)
            if projected is None:
                continue
            worst = projected if worst is None else max(worst, projected)
        return worst is not None and worst > budget

    def _waste_rate(self) -> float:
        """Fraction of retired prompt groups that GC has discarded.

        The quantity that decides whether abandoning can pay, and how hard. A
        workload whose baseline discards nothing has no capacity to recover, so
        every abandonment is pure loss; one that discards a third of its groups has
        a great deal to recover.
        """
        retired = self.gc_dropped + self.consumed_groups
        if retired <= 0:
            return 0.0
        return self.gc_dropped / retired

    def _effective_margin(self) -> float:
        """The safety margin to judge against, adapted to the observed waste rate.

        **Why adapt rather than configure.** The measured optimum moves with how much
        the baseline wastes, and in the same direction every time:

        | baseline GC-drops | best margin | result |
        |---|---|---|
        | 0 of 1536 (DAPO S=1) | none is safe | 0.861x at every margin |
        | 32 of 1568 (DAPO S=2) | 0.7 only | 1.058x at 0.7, losses at >=1.0 |
        | 752 of 2800 (tail S=2) | 0.7 | 1.473x, and 1.0 still wins at 1.149x |

        So a little waste calls for an aggressive margin (cut only the clearly
        hopeless, because a mistake is expensive relative to the prize), and plenty
        of waste makes the configured margin safe. Interpolating between them is
        what stops one fixed number being wrong at both ends -- which is exactly
        what the earlier binary gate did at DAPO S=2, where it activated and then
        lost at two of three margins.

        Returns:
            The margin to use, at or below `safety_margin`.
        """
        rate = self._waste_rate()
        if rate >= self.waste_rate_trusted:
            return self.safety_margin
        # Below the trusted rate, scale the margin down toward `aggressive_margin`
        # in proportion to how little evidence of waste there is.
        share = rate / self.waste_rate_trusted if self.waste_rate_trusted > 0 else 1.0
        return self.aggressive_margin + share * (self.safety_margin - self.aggressive_margin)

    def on_group_complete(self, group: list[Request], st: SimState):
        """Track consumed groups, so the waste rate has a denominator."""
        action = super().on_group_complete(group, st)
        self.consumed_groups += 1
        return action

    def expire(self, st: SimState, expected_train_version: int) -> list[Request]:
        """Return GC victims, plus groups that provably cannot meet their deadline.

        The GC half is `RollPolicy`'s and is unchanged. The additions are accounted
        the same way GC accounts its own victims -- removed from `cohort.running`
        and decremented out of `running_prompts` -- because the admission budget is
        derived from those counters and leaking them would silently throttle
        admission for the rest of the run.
        """
        victims = super().expire(st, expected_train_version)
        if not self.abandon or self._cycle_s <= 0.0:
            return victims
        # Stay inert until GC has actually discarded something. On a workload with
        # no hopeless work there is nothing to recover, and the feasibility test
        # would only cut trajectories that were going to land -- measured at 0.861x
        # on the aligned DAPO trace, whose baseline GC-drops zero groups.
        if self.require_observed_waste and self.gc_dropped == 0:
            return victims

        margin = self._effective_margin()
        already = {id(req) for req in victims}
        live_by_group: dict[int, list[Request]] = {}
        for req in st.inflight:
            if id(req) not in already:
                live_by_group.setdefault(req.group_id, []).append(req)

        for step in sorted(self.cohorts):
            cycles_left = step + st.max_staleness + 1 - expected_train_version
            if cycles_left <= 0:
                # Already collected by super(), or about to be.
                continue
            cohort = self.cohorts[step]
            for group_id in list(cohort.running):
                group = live_by_group.get(group_id)
                if not group:
                    continue
                if not self._group_is_doomed(group, cycles_left, st.now, margin):
                    continue
                cohort.running.discard(group_id)
                self.running_prompts -= 1
                self.abandoned += 1
                self.abandoned_tokens_saved += sum(max(0, r.remaining) for r in group)
                victims.extend(group)
        return victims
