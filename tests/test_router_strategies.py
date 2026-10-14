"""Routing strategies and the admission gate, tested on synthetic views.

A strategy is pure selection over `InstanceView`, so none of this needs an
engine. That is the point of the split: a selection rule is checkable without
building a simulation around it.
"""

from __future__ import annotations

import pytest

from dryrun.component.rollout.engine import InstanceView, Request
from dryrun.component.rollout.router import (
    STRATEGY_REGISTRY,
    AdmissionGate,
    build_strategy,
    register_strategy,
)


def _view(
    instance_id: int,
    n_running: int = 0,
    n_waiting: int = 0,
    free_blocks: int = 0,
    committed_blocks: int = 0,
    hit: int = 0,
    delta: float = 0.0,
) -> InstanceView:
    return InstanceView(
        instance_id=instance_id,
        n_waiting=n_waiting,
        n_running=n_running,
        kv_utilization=0.0,
        total_ctx=0,
        free_blocks=free_blocks,
        committed_blocks=committed_blocks,
        prefix_hit_tokens=lambda req, v=hit: v,
        step_time_delta=lambda req, v=delta: v,
    )


def _req(rid: int = 0, prompt_len: int = 100, group_id: int = 0) -> Request:
    return Request(rid=rid, prompt_len=prompt_len, target_len=50, v_traj=0, group_id=group_id)


class TestRegistry:
    def test_every_built_in_strategy_is_registered(self):
        assert set(STRATEGY_REGISTRY) == {
            "request_num_balance",
            "round_robin",
            "random",
            "kv_cache_aware",
            "prefix_aware",
            "throughput_balance",
            "length_aware",
            "inventory_deadline",
        }

    def test_an_unknown_name_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="request_num_balance"):
            build_strategy("nonexistent")

    def test_a_duplicate_name_is_refused(self):
        """Two strategies sharing a name would silently shadow each other."""
        with pytest.raises(ValueError, match="already registered"):

            @register_strategy("request_num_balance")
            class _Clash:
                pass


class TestRequestNumBalance:
    def test_fewest_active_wins(self):
        strategy = build_strategy("request_num_balance")
        views = [_view(0, n_running=5), _view(1, n_running=2), _view(2, n_running=9)]
        assert strategy.select(_req(), views) == 1

    def test_waiting_counts_towards_active(self):
        strategy = build_strategy("request_num_balance")
        views = [_view(0, n_running=2, n_waiting=8), _view(1, n_running=4, n_waiting=0)]
        assert strategy.select(_req(), views) == 1

    def test_a_tie_takes_the_lowest_index(self):
        """Determinism: a run must not depend on dict or set ordering."""
        strategy = build_strategy("request_num_balance")
        views = [_view(0, n_running=3), _view(1, n_running=3), _view(2, n_running=3)]
        assert strategy.select(_req(), views) == 0

    def test_a_single_candidate_is_chosen(self):
        strategy = build_strategy("request_num_balance")
        assert strategy.select(_req(), [_view(0, n_running=99)]) == 0


class TestRoundRobin:
    def test_it_cycles(self):
        strategy = build_strategy("round_robin")
        views = [_view(0), _view(1), _view(2)]
        assert [strategy.select(_req(), views) for _ in range(7)] == [0, 1, 2, 0, 1, 2, 0]

    def test_the_counter_survives_a_changing_candidate_set(self):
        """The cycle shifts rather than restarting, as SMG's does."""
        strategy = build_strategy("round_robin")
        three = [_view(0), _view(1), _view(2)]
        assert strategy.select(_req(), three) == 0
        assert strategy.select(_req(), [_view(0), _view(1)]) == 1
        assert strategy.select(_req(), three) == 2


class TestRandom:
    def test_it_is_reproducible_from_its_seed(self):
        picks = [
            [build_strategy("random", seed=7).select(_req(), [_view(i) for i in range(4)])]
            for _ in range(3)
        ]
        assert picks[0] == picks[1] == picks[2]

    def test_it_stays_in_range(self):
        strategy = build_strategy("random", seed=1)
        views = [_view(i) for i in range(3)]
        assert all(0 <= strategy.select(_req(), views) < 3 for _ in range(50))


class TestKvCacheAware:
    def test_most_free_blocks_wins(self):
        strategy = build_strategy("kv_cache_aware")
        views = [_view(0, free_blocks=100), _view(1, free_blocks=900), _view(2, free_blocks=50)]
        assert strategy.select(_req(), views) == 1

    def test_it_ignores_request_count(self):
        """The resource that binds is blocks, not request count."""
        strategy = build_strategy("kv_cache_aware")
        views = [_view(0, n_running=1, free_blocks=10), _view(1, n_running=50, free_blocks=800)]
        assert strategy.select(_req(), views) == 1

    def test_committed_but_unallocated_context_still_counts(self):
        """Regression: queued requests hold no blocks yet, so two instances
        with very different commitments report identical `free_blocks`. Reading
        that alone made `max` always return index 0 and piled a whole batch
        onto one instance while the other sat empty."""
        strategy = build_strategy("kv_cache_aware")
        views = [
            _view(0, n_waiting=100, free_blocks=100_000, committed_blocks=9_000),
            _view(1, free_blocks=100_000, committed_blocks=0),
        ]
        assert strategy.select(_req(), views) == 1


class TestPrefixAware:
    def test_the_best_hit_wins_when_balanced(self):
        strategy = build_strategy("prefix_aware", cache_threshold=0.3)
        req = _req(prompt_len=100)
        views = [_view(0, hit=0), _view(1, hit=80)]
        assert strategy.select(req, views) == 1

    def test_a_hit_below_the_threshold_falls_back_to_balance(self):
        strategy = build_strategy("prefix_aware", cache_threshold=0.5)
        req = _req(prompt_len=100)
        # 20% hit on a loaded instance is not worth the imbalance.
        views = [_view(0, n_running=1, hit=0), _view(1, n_running=9, hit=20)]
        assert strategy.select(req, views) == 0

    def test_imbalance_overrides_affinity(self):
        """Both limbs must trip: a big absolute gap AND a big ratio."""
        strategy = build_strategy(
            "prefix_aware",
            cache_threshold=0.1,
            balance_abs_threshold=64,
            balance_rel_threshold=1.5,
        )
        req = _req(prompt_len=100)
        views = [_view(0, n_running=10, hit=0), _view(1, n_running=500, hit=100)]
        assert strategy.select(req, views) == 0

    def test_a_large_ratio_with_a_small_gap_is_not_imbalance(self):
        """2 vs 4 requests is a 2x ratio and means nothing."""
        strategy = build_strategy(
            "prefix_aware",
            cache_threshold=0.1,
            balance_abs_threshold=64,
            balance_rel_threshold=1.5,
        )
        req = _req(prompt_len=100)
        views = [_view(0, n_running=2, hit=0), _view(1, n_running=4, hit=100)]
        assert strategy.select(req, views) == 1


class TestThroughputBalance:
    def test_the_cheapest_marginal_step_wins(self):
        strategy = build_strategy("throughput_balance")
        views = [_view(0, delta=0.05), _view(1, delta=0.01), _view(2, delta=0.30)]
        assert strategy.select(_req(), views) == 1


class TestAdmissionGate:
    def test_it_admits_by_default(self):
        assert AdmissionGate().reject_reason(_view(0, n_running=10_000)) is None

    def test_the_count_cap_rejects_a_full_instance(self):
        gate = AdmissionGate(max_concurrent_seqs_per_instance=4)
        assert gate.reject_reason(_view(0, n_running=4)) == "count_cap"
        assert gate.reject_reason(_view(0, n_running=3)) is None

    def test_a_zero_cap_disables_the_limb(self):
        gate = AdmissionGate(max_concurrent_seqs_per_instance=0)
        assert gate.reject_reason(_view(0, n_running=10_000)) is None

    def test_reject_on_waiting_is_opt_in(self):
        busy = _view(0, n_waiting=1)
        assert AdmissionGate().reject_reason(busy) is None
        assert AdmissionGate(reject_on_waiting=True).reject_reason(busy) == "waiting_queue"

    def test_preempt_waiting_gate_does_not_reject_fresh_work(self):
        busy = _view(0, n_waiting=1)
        gate = AdmissionGate(reject_preempted_on_waiting=True)

        assert gate.reject_reason(busy) is None
        assert (
            gate.reject_reason(busy, is_preempt_loopback=True)
            == "waiting_queue"
        )
