"""The value-based admission strategy, on synthetic views and in a whole run."""

from __future__ import annotations

import pytest

from dryrun.component.rollout.engine import InstanceView, Request
from dryrun.component.rollout.router import build_strategy
from dryrun.component.rollout.router.strategy import RoutingContext
from dryrun.cost.rollout import UnifiedRoofline
from dryrun.policy.roll import RollPolicy
from dryrun.simulator.config import SimConfig
from dryrun.simulator.simulator import Simulator
from dryrun.value import TrajView
from dryrun.workload.grouped import GroupedWorkload


class SaturatedCost:
    """Pinned total throughput, so one more request is pure dilution.

    The total is sized so a batch of a few dozen gives each request a budget of a
    few thousand tokens over one 30 s cycle, which lands inside the fixture length
    distribution. A faster engine would put every trajectory past the end of that
    distribution, where V is pinned at 1 and every marginal value is zero.
    """

    def __init__(self, total_throughput: float = 1200.0) -> None:
        self.total = total_throughput

    def step_time(self, n_tokens, t2, ctxsum, n_reqs=0, prefill_ctx=0.0):
        return max(1, n_reqs) / self.total


def _view(instance_id: int, residents: list[TrajView], cost=None) -> InstanceView:
    ctxsum = float(sum(r.ctx for r in residents))
    return InstanceView(
        instance_id=instance_id,
        n_waiting=0,
        n_running=len(residents),
        kv_utilization=0.0,
        total_ctx=int(ctxsum),
        decode_batch=lambda n=len(residents), c=ctxsum: (n, c),
        residents=lambda r=residents: list(r),
        cost_model=cost or SaturatedCost(),
    )


def _resident(generated: int = 2000, prompt_key: int = 1) -> TrajView:
    return TrajView(
        generated=generated,
        prompt_len=512,
        target_len=10000,
        prompt_key=prompt_key,
        v_traj=0,
        admit_time=0.0,
    )


def _req(rid: int = 0, generated: int = 0, prompt_len: int = 512) -> Request:
    req = Request(rid=rid, prompt_len=prompt_len, target_len=10000, v_traj=0)
    req.generated = generated
    return req


def _strategy(**kwargs):
    strategy = build_strategy(
        "scg_admission",
        max_staleness=1,
        length_model="empirical",
        length_model_args={"max_length": 16384, "bin_width": 256, "min_samples": 10},
        **kwargs,
    )
    for length in range(100, 10001, 100):
        strategy.valuator.model.observe(_resident(prompt_key=length), final_length=length)
    # Version 0 published, so a trajectory admitted at v_traj=0 with S=1 still
    # has a full cycle before its cohort is collected. Setting train_version=1
    # here would put every fixture trajectory exactly on its deadline, making
    # every budget zero and every assertion below vacuous.
    strategy.valuator.cycle.cycle_s = 30.0
    strategy.valuator.cycle.train_version = 0
    strategy.valuator.cycle.cycle_start = 0.0
    return strategy


def _context(now: float = 1.0) -> RoutingContext:
    return RoutingContext(blocked_versions=(), fleet_min_tp=1, fleet_max_tp=1, now=now)


def _rates(view: InstanceView, req: Request) -> tuple[float, float]:
    """Return the rates the strategy itself would price this placement at."""
    from dryrun.value import BatchState, decode_rates, traj_view

    n_decode, ctxsum = view.decode_batch()
    return decode_rates(
        view.cost_model,
        BatchState(n_decode=n_decode, ctxsum=ctxsum),
        candidate_ctx=traj_view(req).ctx,
    )


class TestFallback:
    def test_an_unready_valuator_balances_instead_of_declining(self):
        """A strategy with no evidence must not refuse the whole fleet."""
        strategy = build_strategy("scg_admission", length_model="empirical")
        assert not strategy.valuator.ready
        busy = _view(0, [_resident() for _ in range(32)])
        quiet = _view(1, [_resident() for _ in range(4)])
        assert strategy.select_context(_req(), [busy, quiet], _context()) == 1
        assert strategy.n_fallback == 1


class TestIdleFloor:
    def test_an_idle_instance_is_always_taken(self):
        """An idle GPU is worth any work, whatever the request is worth."""
        strategy = _strategy()
        busy = _view(0, [_resident() for _ in range(64)])
        idle = _view(1, [])
        assert strategy.select_context(_req(), [busy, idle], _context()) == 1


class TestContextPlumbing:
    def test_select_without_context_still_places(self):
        """`select` is the protocol. A caller that gives no clock must still work."""
        strategy = _strategy()
        busy = _view(0, [_resident() for _ in range(32)])
        quiet = _view(1, [_resident() for _ in range(4)])
        assert strategy.select(_req(), [busy, quiet]) == 1

    def test_the_router_passes_the_dispatch_time_through(self):
        """`now` must reach the strategy, or every deadline is evaluated at zero."""
        seen: list[float] = []

        class Spy:
            name = "spy"

            def select(self, req, candidates):
                return 0

            def select_context(self, req, candidates, context):
                seen.append(context.now)
                return 0

        from dryrun.component.rollout.engine import NativeInstance
        from dryrun.component.rollout.router import Router

        cost = UnifiedRoofline(F=0.01, W=1e-4, G=1e-6, A_p=1e-9, A_d=1e-9, b=1.0)
        inst = NativeInstance(cost, token_budget=8192, kv_blocks=10000, block_size=16)
        router = Router(strategy=Spy(), instances=[inst], cost_model=cost)
        router.enqueue([_req()], 0.0)
        router.dispatch(42.5)
        assert seen == [42.5]


class TestCostModelRequirement:
    def test_a_router_without_a_cost_model_is_refused(self):
        """Silently pricing against None would make every dilution zero."""
        from dryrun.component.rollout.router import Router

        with pytest.raises(ValueError, match="cost model"):
            Router(strategy=_strategy(), instances=[])


class SaturatingCost:
    """Fleet throughput pinned past a knee, the shape a real engine has.

    Below `knee` total throughput grows with the batch and nobody is harmed.
    Above it the total is fixed, so each resident's share falls as `1/b`. The
    aligned trace measures exactly this: 1550 tokens per second from b=49
    onward, with the per-request rate dropping 31.6 -> 19.6 over 49 -> 79.
    """

    def __init__(self, knee: int = 49, total: float = 1550.0) -> None:
        self.knee = knee
        self.total = total

    def step_time(self, n_tokens, t2, ctxsum, n_reqs=0, prefill_ctx=0.0):
        b = max(1, n_reqs)
        per_request = self.total / b if b >= self.knee else self.total / self.knee
        return 1.0 / per_request

    def decode_coeffs(self, n_d, ctxsum):
        return 0.0, self.step_time(0, 0, ctxsum, n_d), 0.0


class TestUrgencyOrdering:
    """`U` must rank a trajectory about to miss above one that cannot be saved."""

    def _valuator(self):
        strategy = _strategy()
        return strategy.valuator

    def test_a_safe_trajectory_has_no_urgency(self):
        valuator = self._valuator()
        safe = _resident(generated=9900)
        assert valuator.hit_prob(safe, 300.0, 1.0) == pytest.approx(1.0, abs=0.05)
        assert valuator.urgency(safe, 300.0, 1.0) == pytest.approx(0.0, abs=1e-6)

    def test_a_contested_trajectory_has_urgency(self):
        valuator = self._valuator()
        contested = _resident(generated=2000)
        value = valuator.hit_prob(contested, 120.0, 1.0)
        assert 0.0 < value < 1.0, "The fixture must put this one in contention."
        assert valuator.urgency(contested, 120.0, 1.0) > 0.0

    def test_urgency_vanishes_with_the_hit_probability(self):
        """The safety property: nothing left to lose means nothing to prioritise.

        This is what stops the objective from chasing work it cannot rescue, and
        it needs no special case because `V` already collapses there. Stated as a
        limit rather than an equality, because a trajectory with a one-in-30000
        chance is not quite impossible and `U` should reflect that rather than
        being clamped.
        """
        valuator = self._valuator()
        doomed = _resident(generated=100)
        contested = _resident(generated=3000)
        assert valuator.hit_prob(doomed, 0.5, 29.0) < 1e-3
        assert valuator.urgency(doomed, 0.5, 29.0) < 1e-4
        # Three orders of magnitude below a trajectory that can still be saved,
        # which is what makes the ordering safe.
        assert valuator.urgency(doomed, 0.5, 29.0) < 0.01 * valuator.urgency(
            contested, 150.0, 20.0
        )

    def test_urgency_does_not_vanish_where_the_marginal_value_does(self):
        """The reason for the redesign, as a direct comparison.

        `G` goes to zero at both ends of `V`, so a batch that has collapsed to low
        `V` shows no difference between a request about to miss and a safe one.
        `U` keeps them apart.
        """
        valuator = self._valuator()
        nearly_late = _resident(generated=3000)
        safe = _resident(generated=9900)
        rate = 150.0
        assert valuator.hit_prob(nearly_late, rate, 20.0) < 0.5
        assert valuator.urgency(nearly_late, rate, 20.0) > valuator.urgency(
            safe, rate, 20.0
        )


class TestPauseVictim:
    """The victim is the resident with the least to lose, not the one predicted to miss."""

    def _fleet_view(self, generated: list[int], kv_blocks: int = 100_000):
        """A decode batch wide enough to be past the cost model's knee.

        Below the knee pausing frees no bandwidth, so the strategy correctly
        declines and a victim-selection test would pass for the wrong reason.
        """
        residents = [_resident(generated=g) for g in generated]
        requests = [object() for _ in generated]
        ctxsum = float(sum(r.ctx for r in residents))
        return InstanceView(
            instance_id=0,
            n_waiting=0,
            n_running=len(residents),
            kv_utilization=0.0,
            total_ctx=int(ctxsum),
            decode_batch=lambda n=len(residents), c=ctxsum: (n, c),
            residents=lambda r=residents: list(r),
            decoding=lambda r=residents, q=requests: list(zip(r, q)),
            cost_model=SaturatingCost(knee=8),
            kv_blocks=kv_blocks,
            block_size=16,
        ), residents, requests

    def test_the_least_urgent_resident_is_the_one_paused(self):
        """A trajectory with time to spare yields to ones that are running out.

        Deliberately not the one predicted to miss: that one is the most urgent
        by construction and is what the objective is trying to rescue.
        """
        strategy = _strategy()
        # 9900 of a ~10000 token response at this rate still has slack, while
        # the 3000s are mid-distribution and contested.
        view, residents, requests = self._fleet_view([3000] * 40 + [9990])
        rate = strategy._rate(view, *view.decode_batch())
        urgencies = [
            strategy.valuator.urgency(p, rate, 20.0) for p, _ in view.decoding()
        ]
        least = min(range(len(urgencies)), key=lambda i: urgencies[i])
        assert urgencies[least] < max(urgencies), "The fixture must spread urgency."

        victim = strategy.choose_pause(view, 20.0)
        assert victim is requests[least]
        assert strategy.n_paused == 1

    def test_nothing_is_paused_when_no_resident_is_in_danger(self):
        """Freed bandwidth buys nothing, and the victim still pays a re-prefill."""
        strategy = _strategy()
        view, _, _ = self._fleet_view([9900] * 20)
        assert strategy.choose_pause(view, 1.0) is None
        assert strategy.n_paused == 0

    def test_pause_off_never_pauses(self):
        strategy = _strategy(pause=False)
        view, _, _ = self._fleet_view([3000] * 40 + [9990])
        assert strategy.choose_pause(view, 20.0) is None


class TestWholeRun:
    def _run(self, strategy: str, strategy_args: dict) -> object:
        workload = GroupedWorkload.from_lengths(
            [512] * 64,
            [600, 2400, 1200, 4800] * 16,
        )
        cfg = SimConfig(
            batch_size=4,
            group_size=1,
            max_staleness=1,
            n_versions=6,
            n_instances=2,
            kv_blocks=4000,
            block_size=16,
            token_budget=8192,
            workload=workload,
            routing_strategy=strategy,
            routing_strategy_args=strategy_args,
            log_telemetry=False,
        )
        cost = UnifiedRoofline(F=0.01, W=1e-4, G=1e-6, A_p=1e-9, A_d=1e-9, b=1.0)
        sim = Simulator(cost, RollPolicy(), cfg, backend="native")
        return sim.run()

    def test_the_run_completes_and_respects_the_staleness_bound(self):
        """Declining must not stall the loop, which is the real risk here."""
        result = self._run(
            "scg_admission",
            {"max_staleness": 1, "length_model": "oracle", "training_cycle_s": 1.0},
        )
        assert not result.livelocked
        assert result.versions_done == 6
        assert result.n_consumed > 0
        assert result.max_staleness_true <= 1

    def test_the_empirical_model_also_completes(self):
        result = self._run(
            "scg_admission",
            {
                "max_staleness": 1,
                "length_model": "empirical",
                "length_model_args": {"min_samples": 4, "bin_width": 128},
                "training_cycle_s": 1.0,
            },
        )
        assert not result.livelocked
        assert result.versions_done == 6
        assert result.n_consumed > 0
