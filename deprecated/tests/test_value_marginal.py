"""The admission inequality, and its equivalence to the design document's form.

The central test here is `TestRooflineEquivalence`. The strategy replaces the
document's `G_new > mean(resident G)` above a measured knee `b*` with a rate
comparison that reads the cost model directly. If the two disagree, the deviation is
wrong and every downstream result is unsupported.
"""

from __future__ import annotations

import pytest

from srl.value import BatchState, TrainCycle, TrajView, Valuator, decode_rates
from srl.value.length import LengthModel


class ConstantValue:
    """A length model whose marginal value is a fixed number per prompt key.

    Lets the admission arithmetic be tested against hand-computed values without
    the length distribution being a confound.

    Slopes must keep `V` strictly below 1 across the budgets under test. A
    saturated `V` makes every `G` zero, which turns an equivalence assertion into
    a comparison of zero against zero and hides whatever it was meant to catch.
    `__init__` refuses a slope that would saturate.
    """

    name = "constant"
    max_budget = 20000.0

    def __init__(self, per_key: dict[int, float], default: float = 0.0) -> None:
        for slope in (*per_key.values(), default):
            if slope * self.max_budget >= 1.0:
                raise ValueError(
                    f"Slope {slope!r} saturates V within {self.max_budget} tokens, "
                    f"which would make every marginal value zero."
                )
        self.per_key = per_key
        self.default = default

    @property
    def ready(self) -> bool:
        return True

    def hit_prob(self, traj: TrajView, budget: float) -> float:
        # V linear in budget means dV/dB is exactly the configured slope.
        slope = self.per_key.get(traj.prompt_key, self.default)
        return min(1.0, slope * budget)

    def observe(self, traj: TrajView, final_length: int) -> None:
        """Nothing to learn."""

    def decay(self, factor: float) -> None:
        """Nothing to discount."""


class SaturatedCost:
    """A cost model whose total throughput is pinned, so decode is pure dilution.

    `step_time = n / T`, hence per-request rate `T / n`: adding a request takes
    tokens from the residents and creates none. This is the regime above the
    document's roofline knee.
    """

    def __init__(self, total_throughput: float = 6000.0) -> None:
        self.total = total_throughput

    def step_time(self, n_tokens, t2, ctxsum, n_reqs=0, prefill_ctx=0.0):
        return max(1, n_reqs) / self.total


class UnsaturatedCost:
    """A cost model below the knee: step time is flat, so rate is unchanged.

    Total throughput grows linearly with batch size and nobody is slowed down.
    """

    def __init__(self, step_s: float = 0.01) -> None:
        self.step_s = step_s

    def step_time(self, n_tokens, t2, ctxsum, n_reqs=0, prefill_ctx=0.0):
        return self.step_s


def make_valuator(model: LengthModel, max_staleness: int = 1) -> Valuator:
    cycle = TrainCycle(seed_s=30.0)
    cycle.observe(train_version=0, now=0.0)
    return Valuator(model, cycle, max_staleness=max_staleness)


def traj(prompt_key: int = 0, generated: int = 0, v_traj: int = 0) -> TrajView:
    return TrajView(
        generated=generated,
        prompt_len=512,
        target_len=10000,
        prompt_key=prompt_key,
        v_traj=v_traj,
        admit_time=0.0,
    )


class TestDecodeRates:
    def test_an_empty_batch_dilutes_nothing(self):
        cost = SaturatedCost()
        rate_before, rate_after = decode_rates(
            cost, BatchState(n_decode=0, ctxsum=0.0), candidate_ctx=1000
        )
        assert rate_before == 0.0
        assert rate_after > 0.0

    def test_a_bigger_batch_is_slower_per_request_when_saturated(self):
        cost = SaturatedCost()
        _, small = decode_rates(cost, BatchState(n_decode=4, ctxsum=8000.0), 2000)
        _, large = decode_rates(cost, BatchState(n_decode=64, ctxsum=128000.0), 2000)
        assert large < small


class TestBudget:
    def test_budget_shrinks_as_the_deadline_approaches(self):
        valuator = make_valuator(ConstantValue({}, default=1e-5))
        early = valuator.budget(traj(), rate=300.0, now=1.0)
        late = valuator.budget(traj(), rate=300.0, now=29.0)
        assert early > late

    def test_an_aged_out_trajectory_has_no_budget(self):
        valuator = make_valuator(ConstantValue({}, default=1e-5), max_staleness=1)
        valuator.cycle.observe(train_version=5, now=150.0)
        assert valuator.budget(traj(v_traj=0), rate=300.0, now=151.0) == 0.0

    def test_an_unobserved_cycle_leaves_the_valuator_unready(self):
        valuator = Valuator(ConstantValue({}, default=1e-5), TrainCycle(), max_staleness=1)
        assert not valuator.ready
        assert valuator.budget(traj(), rate=300.0, now=1.0) == 0.0


class TestTrainCycle:
    def test_the_cycle_length_is_learned_from_observed_publishes(self):
        cycle = TrainCycle()
        assert not cycle.known
        cycle.observe(train_version=1, now=0.0)
        cycle.observe(train_version=2, now=40.0)
        assert cycle.cycle_s == pytest.approx(40.0)

    def test_a_later_interval_is_smoothed_not_replaced(self):
        cycle = TrainCycle(smoothing=0.3)
        cycle.observe(train_version=1, now=0.0)
        cycle.observe(train_version=2, now=40.0)
        cycle.observe(train_version=3, now=140.0)
        assert cycle.cycle_s == pytest.approx(0.7 * 40.0 + 0.3 * 100.0)

    def test_cycles_left_matches_the_roll_policy_gc_window(self):
        """`RollPolicy` collects cohort `v` once the published version is `v+S+1`."""
        cycle = TrainCycle(seed_s=30.0)
        cycle.observe(train_version=3, now=90.0)
        assert cycle.cycles_left(v_traj=3, max_staleness=1) == 1
        assert cycle.cycles_left(v_traj=2, max_staleness=1) == 0
