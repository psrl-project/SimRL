"""Group-level length ordering, bounded reordering and load-aware placement.

The strategy is pure selection over queued requests and instance views, so none
of this needs an engine.
"""

from __future__ import annotations

import pytest

from dryrun.component.rollout.engine import InstanceView, Request
from dryrun.component.rollout.router import build_strategy
from dryrun.value import NoisyOracleLength, TrajView, traj_view


def _group(group_id: int, lengths: list[int], rid0: int = 0) -> list[Request]:
    return [
        Request(
            rid=rid0 + i,
            prompt_len=100,
            target_len=length,
            v_traj=0,
            group_id=group_id,
            group_index=i,
            source_group_id=group_id,
        )
        for i, length in enumerate(lengths)
    ]


def _strategy(**kwargs):
    return build_strategy("length_aware", **kwargs)


def _queue(strategy, *groups: list[Request]) -> list[Request]:
    window = []
    for group in groups:
        strategy.observe_enqueue(group)
        window.extend(group)
    return window


class TestOrdering:
    def test_longest_group_first_is_keyed_on_its_slowest_member(self):
        """A group of one long and seven short outranks a group of eight medium ones."""
        strategy = _strategy(order="lpt")
        spiky = _group(0, [9000] + [500] * 7, rid0=0)
        even = _group(1, [4000] * 8, rid0=8)
        picked = strategy.rank_queued(_queue(strategy, even, spiky), 0.0)
        assert picked.group_id == 0
        assert picked.target_len == 9000

    def test_shortest_first_is_the_mirror(self):
        strategy = _strategy(order="spt")
        spiky = _group(0, [9000] + [500] * 7, rid0=0)
        even = _group(1, [4000] * 8, rid0=8)
        assert strategy.rank_queued(_queue(strategy, spiky, even), 0.0).group_id == 1

    def test_fifo_leaves_the_queue_alone(self):
        strategy = _strategy(order="fifo")
        assert strategy.rank_queued(_queue(strategy, _group(0, [10, 20])), 0.0) is None

    def test_siblings_of_a_started_group_go_ahead_of_a_longer_new_group(self):
        """Each sibling left queued holds its group back, so none may be overtaken."""
        strategy = _strategy(order="lpt")
        started = _group(0, [3000, 2900, 100], rid0=0)
        newer = _group(1, [9000, 8000, 7000], rid0=3)
        window = _queue(strategy, started, newer)
        first = window[0]
        strategy.select(first, [_instance(0, 0)])
        remaining = [r for r in window if r is not first]
        assert strategy.rank_queued(remaining, 0.0).group_id == 0


class TestBoundedWindow:
    def test_reordering_stays_inside_a_block_of_arrival_order(self):
        """A long group in block 1 may not jump ahead of a short group in block 0."""
        strategy = _strategy(order="lpt", window_groups=2)
        groups = [
            _group(0, [1000] * 2, rid0=0),
            _group(1, [3000] * 2, rid0=2),
            _group(2, [9000] * 2, rid0=4),
            _group(3, [2000] * 2, rid0=6),
        ]
        window = _queue(strategy, *groups)
        assert strategy.rank_queued(window, 0.0).group_id == 1
        assert strategy.rank_queued([r for r in window if r.group_id != 1], 0.0).group_id == 0

    def test_zero_window_reorders_across_everything(self):
        strategy = _strategy(order="lpt", window_groups=0)
        groups = [_group(0, [1000] * 2, rid0=0), _group(1, [9000] * 2, rid0=2)]
        assert strategy.rank_queued(_queue(strategy, *groups), 0.0).group_id == 1

    def test_a_negative_window_is_refused(self):
        with pytest.raises(ValueError, match="window_groups"):
            _strategy(window_groups=-1)


def _instance(instance_id: int, n_active: int, remaining: tuple[int, ...] = ()) -> InstanceView:
    trajs = [
        TrajView(
            generated=0,
            prompt_len=100,
            target_len=length,
            prompt_key=0,
            v_traj=0,
            admit_time=0.0,
        )
        for length in remaining
    ]
    return InstanceView(
        instance_id=instance_id,
        n_waiting=0,
        n_running=n_active,
        kv_utilization=0.0,
        total_ctx=0,
        residents=lambda t=trajs: t,
    )


class TestPlacement:
    def test_count_balance_picks_the_fewest_requests(self):
        strategy = _strategy(balance="count")
        views = [_instance(0, 5, (10,) * 5), _instance(1, 3, (9000,) * 3)]
        assert strategy.select(_group(0, [100])[0], views) == 1

    def test_load_balance_picks_the_least_remaining_work(self):
        """Fewer requests can still be more work, which is the case load tells apart."""
        strategy = _strategy(balance="load")
        views = [_instance(0, 5, (10,) * 5), _instance(1, 3, (9000,) * 3)]
        assert strategy.select(_group(0, [100])[0], views) == 0


def _kv_instance(instance_id: int, committed: int, kv_blocks: int = 1000) -> InstanceView:
    return InstanceView(
        instance_id=instance_id,
        n_waiting=0,
        n_running=1,
        kv_utilization=0.0,
        total_ctx=0,
        committed_blocks=committed,
        kv_blocks=kv_blocks,
        block_size=16,
    )


class TestKvWatermark:
    def test_it_is_off_by_default(self):
        strategy = _strategy(balance="count")
        assert strategy.select(_group(0, [100])[0], [_kv_instance(0, committed=999)]) == 0

    def test_an_instance_over_the_watermark_is_refused(self):
        strategy = _strategy(balance="count", kv_watermark=0.8)
        assert strategy.select(_group(0, [100])[0], [_kv_instance(0, committed=800)]) is None

    def test_the_newcomers_own_context_counts(self):
        """A 100-token prompt needs 7 blocks, so 794 committed is the last that fits at 0.8."""
        strategy = _strategy(balance="count", kv_watermark=0.8)
        req = _group(0, [100])[0]
        assert strategy.select(req, [_kv_instance(0, committed=793)]) == 0
        assert strategy.select(req, [_kv_instance(0, committed=794)]) is None

    def test_a_resumed_request_is_charged_for_what_it_already_generated(self):
        strategy = _strategy(balance="count", kv_watermark=0.8)
        req = _group(0, [5000])[0]
        req.generated = 3000
        assert strategy.select(req, [_kv_instance(0, committed=500)]) is None

    def test_a_refused_instance_does_not_block_one_with_room(self):
        strategy = _strategy(balance="count", kv_watermark=0.8)
        views = [_kv_instance(0, committed=900), _kv_instance(1, committed=100)]
        assert strategy.select(_group(0, [100])[0], views) == 1

    def test_a_watermark_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValueError, match="kv_watermark"):
            _strategy(kv_watermark=1.5)


class TestNoisyOracle:
    def _traj(self, target: int = 8000, key: int = 3) -> TrajView:
        return TrajView(
            generated=0, prompt_len=140, target_len=target, prompt_key=key, v_traj=0, admit_time=0.0
        )

    def test_zero_noise_is_the_oracle(self):
        model = NoisyOracleLength(sigma=0.0)
        assert model.expected_final_length(self._traj()) == 8000.0

    def test_the_same_request_is_mispredicted_the_same_way_every_time(self):
        model = NoisyOracleLength(sigma=0.5, seed=1)
        t = self._traj()
        assert model.expected_final_length(t) == model.expected_final_length(t)

    def test_a_different_seed_gives_a_different_error(self):
        t = self._traj()
        a = NoisyOracleLength(sigma=0.5, seed=1).expected_final_length(t)
        b = NoisyOracleLength(sigma=0.5, seed=2).expected_final_length(t)
        assert a != b

    def test_the_error_has_the_requested_log_spread(self):
        import math
        import statistics

        model = NoisyOracleLength(sigma=0.5, seed=0)
        errors = [
            math.log(model.expected_final_length(self._traj(5000 + 37 * i, i)) / (5000 + 37 * i))
            for i in range(2000)
        ]
        assert statistics.pstdev(errors) == pytest.approx(0.5, abs=0.05)
        assert statistics.mean(errors) == pytest.approx(0.0, abs=0.05)

    def test_a_negative_sigma_is_refused(self):
        with pytest.raises(ValueError, match="sigma"):
            NoisyOracleLength(sigma=-0.1)

    def test_traj_view_feeds_it_from_a_request(self):
        req = _group(0, [8000])[0]
        assert NoisyOracleLength(sigma=0.0).expected_final_length(traj_view(req)) == 8000.0
