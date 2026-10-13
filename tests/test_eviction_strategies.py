"""Eviction strategies: the registry, each rule, and the engine's use of them."""

from __future__ import annotations

import pytest

from dryrun.component.rollout.engine import NativeInstance, Request
from dryrun.component.rollout.engine.eviction import (
    EVICTION_REGISTRY,
    build_eviction,
    register_eviction,
)
from dryrun.cost.rollout import UnifiedRoofline
from dryrun.value import EmpiricalLength, TrainCycle, TrajView, Valuator, traj_view


def _req(rid: int, generated: int = 0, dispatch_time: float = 0.0, prompt_len: int = 512):
    req = Request(rid=rid, prompt_len=prompt_len, target_len=10000, v_traj=0)
    req.generated = generated
    req.dispatch_time = dispatch_time
    req.admit_time = 0.0
    return req


class TestRegistry:
    def test_every_built_in_strategy_is_registered(self):
        assert set(EVICTION_REGISTRY) == {"vllm_lru", "longest_ctx", "scg_value"}

    def test_an_unknown_name_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="longest_ctx"):
            build_eviction("nope")

    def test_a_duplicate_name_is_refused(self):
        with pytest.raises(ValueError, match="already registered"):

            @register_eviction("vllm_lru")
            class Clash:
                pass


class TestVllmLru:
    def test_the_newest_dispatched_pays(self):
        strategy = build_eviction("vllm_lru")
        candidates = [_req(i, dispatch_time=float(i)) for i in range(4)]
        assert strategy.select(candidates, now=10.0) is candidates[-1]

    def test_the_choice_is_deterministic_on_a_tie(self):
        strategy = build_eviction("vllm_lru")
        candidates = [_req(i, dispatch_time=5.0) for i in range(4)]
        assert strategy.select(candidates, now=10.0) is candidates[-1]


class TestLongestCtx:
    def test_the_deepest_context_pays(self):
        strategy = build_eviction("longest_ctx")
        candidates = [_req(i, generated=g) for i, g in enumerate([100, 9000, 500])]
        assert strategy.select(candidates, now=10.0) is candidates[1]


class TestScgValue:
    def _bound(self, cycle_s: float = 30.0):
        model = EmpiricalLength(
            max_length=16384, bin_width=256, min_samples=10, per_prompt=False
        )
        for length in range(100, 10001, 100):
            model.observe(
                TrajView(0, 512, length, 0, 0, 0.0), final_length=length
            )
        cycle = TrainCycle(seed_s=cycle_s)
        cycle.observe(train_version=0, now=0.0)
        strategy = build_eviction("scg_value")
        strategy.bind(Valuator(model, cycle, max_staleness=1), prefill_throughput=10000.0)
        return strategy

    def test_an_unbound_strategy_defers_to_the_engine(self):
        """Ranking on no evidence is worse than not ranking."""
        strategy = build_eviction("scg_value")
        assert strategy.select([_req(0, generated=100)], now=1.0) is None

    def test_the_resident_with_the_most_slack_pays(self):
        """Not the one predicted to miss: that is what the objective rescues.

        The cycle is short enough that the deadline actually binds. With a
        generous budget every resident is safe, every eviction loss is zero, and
        the strategy correctly expresses no preference, which would make this
        assertion pass for the wrong reason.
        """
        strategy = self._bound(cycle_s=20.0)
        rate = strategy._observed_rate(
            [_req(0, generated=2000), _req(1, generated=9900)], now=19.0
        )
        contested = [_req(i, generated=2000) for i in range(5)]
        slack = _req(99, generated=9900)
        values = {
            req.rid: strategy.valuator.hit_prob(traj_view(req), rate, 19.0)
            for req in (*contested, slack)
        }
        assert values[0] < 1.0, "The fixture must put the contested ones at risk."
        assert values[99] > values[0], "And leave the slack one safe."

        # Slack and recompute cost move together, so which one is cheapest to
        # evict is a property of the numbers. Assert the rule, not a guess about
        # which way it comes out: the victim is whoever loses least.
        losses = {
            req.rid: strategy._loss(traj_view(req), rate, 19.0)
            for req in (*contested, slack)
        }
        chosen = strategy.select([*contested, slack], now=19.0)
        assert losses[chosen.rid] == min(losses.values())

    def test_a_doomed_trajectory_is_never_the_victim(self):
        """It has nothing left to lose, so it would win on cost every time.

        That is the one choice the objective forbids: a trajectory about to miss
        is what the schedule is trying to rescue, and the eviction delay is what
        would finish it off.
        """
        strategy = self._bound(cycle_s=20.0)
        # Barely started with almost no budget left, against a distribution
        # whose shortest observed length is 100 tokens.
        doomed = _req(0, generated=10)
        savable = [_req(i, generated=9950) for i in range(1, 4)]
        rate = strategy._observed_rate([doomed, *savable], now=19.9)
        value = strategy.valuator.hit_prob(traj_view(doomed), rate, 19.9)
        assert value < strategy.floor, f"The fixture must be doomed, V={value}."
        assert strategy.select([doomed, *savable], now=19.9) is not doomed

    def test_a_batch_with_no_progress_defers(self):
        """An achieved rate needs generated tokens, and inventing one misranks."""
        strategy = self._bound()
        assert strategy.select([_req(i) for i in range(3)], now=1.0) is None


class TestEngineUsesTheStrategy:
    def _instance(self, eviction):
        cost = UnifiedRoofline(F=0.01, W=1e-4, G=1e-6, A_p=1e-9, A_d=1e-9, b=1.0)
        return NativeInstance(
            cost,
            token_budget=8192,
            kv_blocks=64,
            block_size=16,
            eviction=eviction,
        )

    def test_the_default_matches_the_engine_s_own_order(self):
        """No strategy configured must reproduce the measured baseline exactly."""
        inst = self._instance(None)
        victims = [_req(i, dispatch_time=float(i)) for i in range(3)]
        assert inst._pick_victim(victims) is victims[-1]

    def test_a_configured_strategy_decides_instead(self):
        inst = self._instance(build_eviction("longest_ctx"))
        victims = [_req(i, generated=g) for i, g in enumerate([10, 5000, 20])]
        assert inst._pick_victim(victims) is victims[1]

    def test_a_strategy_returning_none_falls_back(self):
        """Declining to rank is allowed, and must not break the engine."""

        class Abstains:
            name = "abstains"

            def select(self, candidates, now):
                return None

        inst = self._instance(Abstains())
        victims = [_req(i, dispatch_time=float(i)) for i in range(3)]
        assert inst._pick_victim(victims) is victims[-1]
