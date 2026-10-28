"""Eviction strategies: the registry, each rule, and the engine's use of them."""

from __future__ import annotations

import pytest

from srl.component.rollout.engine import NativeInstance, Request
from srl.component.rollout.engine.eviction import (
    EVICTION_REGISTRY,
    build_eviction,
    register_eviction,
)
from srl.cost.rollout import UnifiedRoofline


def _req(rid: int, generated: int = 0, dispatch_time: float = 0.0, prompt_len: int = 512):
    req = Request(rid=rid, prompt_len=prompt_len, target_len=10000, v_traj=0)
    req.generated = generated
    req.dispatch_time = dispatch_time
    req.admit_time = 0.0
    return req


class TestRegistry:
    def test_every_built_in_strategy_is_registered(self):
        assert set(EVICTION_REGISTRY) == {"vllm_lru", "longest_ctx", "shortest_ctx", "latest_deadline"}

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


class TestShortestCtx:
    def test_the_shallowest_context_pays(self):
        strategy = build_eviction("shortest_ctx")
        candidates = [_req(i, generated=g) for i, g in enumerate([100, 9000, 500])]
        assert strategy.select(candidates, now=10.0) is candidates[0]


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
