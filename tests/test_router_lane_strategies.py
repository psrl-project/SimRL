"""Candidate rules of the two lane strategies, without building a simulator."""

from __future__ import annotations

from srl.component.rollout.engine import InstanceView, ReqStatus
from srl.component.rollout.router.strategy import RoutingContext, build_strategy, narrow_candidates


class FakeRequest:
    """The read surface `traj_view` needs from a request."""

    def __init__(self, rid: int, group_id: int, prompt_len: int, target_len: int, generated: int = 0):
        self.rid = rid
        self.group_id = group_id
        self.prompt_len = prompt_len
        self.target_len = target_len
        self.generated = generated
        self.source_group_id = group_id
        self.v_traj = 0
        self.admit_time = 0.0
        self.prefilled = 0
        self.status = ReqStatus.QUEUED
        self.instance_id = None

    @property
    def ctx(self) -> int:
        return self.prompt_len + self.generated


def view(instance_id: int, residents=()) -> InstanceView:
    return InstanceView(
        instance_id=instance_id,
        n_waiting=0,
        n_running=len(residents),
        kv_utilization=0.0,
        total_ctx=0,
        tp=1,
        free_blocks=10_000,
        committed_blocks=0,
        residents=lambda: list(residents),
        kv_blocks=10_000,
        block_size=16,
    )


CONTEXT = RoutingContext(blocked_versions=(), fleet_min_tp=1, fleet_max_tp=1, now=0.0)


class TestSlackLaneWithoutAPlan:
    def test_it_offers_every_candidate_before_the_first_dispatch(self):
        strategy = build_strategy("slack_lane")
        req = FakeRequest(rid=0, group_id=0, prompt_len=100, target_len=1000)
        candidates = [view(0), view(1)]
        assert narrow_candidates(strategy, req, candidates, CONTEXT) == [0, 1]

    def test_it_falls_back_to_the_least_loaded_instance(self):
        strategy = build_strategy("slack_lane")
        req = FakeRequest(rid=0, group_id=0, prompt_len=100, target_len=1000)
        busy = view(0, residents=[FakeRequest(9, 9, 100, 50_000)])
        assert strategy.select_context(req, [busy, view(1)], CONTEXT) == 1


class TestLengthLane:
    def test_a_long_request_is_confined_to_the_fast_instances(self):
        strategy = build_strategy("length_lane", n_fast=1, threshold=20_000, c_fast=2)
        req = FakeRequest(rid=0, group_id=0, prompt_len=100, target_len=30_000)
        assert narrow_candidates(strategy, req, [view(0), view(1), view(2)], CONTEXT) == [0]

    def test_a_short_request_is_kept_out_of_the_fast_instances(self):
        strategy = build_strategy("length_lane", n_fast=1, threshold=20_000, c_fast=2)
        req = FakeRequest(rid=0, group_id=0, prompt_len=100, target_len=1_000)
        assert narrow_candidates(strategy, req, [view(0), view(1), view(2)], CONTEXT) == [1, 2]

    def test_a_long_request_spills_to_bulk_when_every_fast_lane_is_full(self):
        full = view(0, residents=[FakeRequest(8, 8, 100, 30_000), FakeRequest(9, 9, 100, 30_000)])
        strategy = build_strategy("length_lane", n_fast=1, threshold=20_000, c_fast=2)
        req = FakeRequest(rid=0, group_id=0, prompt_len=100, target_len=30_000)
        assert narrow_candidates(strategy, req, [full, view(1), view(2)], CONTEXT) == [1, 2]

    def test_n_fast_must_leave_a_bulk_instance(self):
        import pytest

        with pytest.raises(ValueError, match="at least one bulk"):
            build_strategy("length_lane", n_fast=0, threshold=20_000, c_fast=2)
