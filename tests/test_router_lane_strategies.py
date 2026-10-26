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


class TestLaneStrategiesChangeARealRun:
    """Both strategies must move a real simulated run off the default router's result."""

    def build(self, strategy: str, strategy_args: dict):
        """Build a real `Simulator` with a straggler-heavy workload and a contested fleet.

        `staleness_horizon=1` lets a second group wave overlap the first in flight, which is
        what gives `slack_lane` an already-admitted-but-not-yet-placed request to isolate:
        under the default horizon of 0 every wave is admitted and fully routed inside one
        atomic dispatch burst, so the planner never sees a request before it is already
        pinned to an instance. `max_concurrency` is tight enough to keep members of the same
        wave queued across more than one dispatch call instead of draining in one pass.

        The roofline's decode term `A_d` is raised well above a token-level cost so that
        sharing an instance with more neighbours measurably slows every resident's decode.
        Under a floor-dominated roofline, batching is free and a fast lane buys nothing for
        `plan_lanes` to find. `horizon_buffers=1` keeps `slack_lane`'s own forecast inside the
        one buffer this horizon can actually resolve, since projecting a second buffer this
        wave hasn't admitted into yet reports an unresolvable (`inf`) objective that no single
        reassignment can fix, which would stall its own upgrade search before it starts.
        """
        import random

        from srl.component.sync.cost import FixedSyncCost
        from srl.cost.rollout import UnifiedRoofline
        from srl.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel
        from srl.policy.psrl import PSRLPolicy
        from srl.simulator.config import SimConfig
        from srl.simulator.simulator import Simulator
        from srl.workload.grouped import GroupedWorkload
        from srl.workload.trace import RequestGroup

        rng = random.Random(7)
        groups = []
        for group_id in range(24):
            # One straggler per four groups, so a fast lane has something to isolate.
            base = 24_000 if group_id % 4 == 0 else rng.randint(800, 2_500)
            lengths = [base if member == 0 else rng.randint(800, 2_500) for member in range(4)]
            groups.append(RequestGroup(group_id=group_id, prompt_len=120, output_lens=tuple(lengths)))
        cfg = SimConfig(
            n_versions=4,
            batch_size=6,
            group_size=4,
            n_instances=4,
            kv_blocks=20_000,
            block_size=16,
            token_budget=8192,
            max_concurrency=8,
            weight_load_time=0.0,
            log_telemetry=False,
            workload=GroupedWorkload(groups),
            staleness_horizon=1,
        )
        cfg.routing_strategy = strategy
        args = dict(strategy_args)
        if strategy == "slack_lane":
            args.setdefault("horizon_buffers", 1)
        cfg.routing_strategy_args = args
        return Simulator(
            cfg=cfg,
            policy=PSRLPolicy(),
            rollout_cost=UnifiedRoofline(F=0.001, W=0.002, G=4e-6, A_p=3e-10, A_d=5e-6, b=16),
            training_cost=TrainingCostModel(FixedTrainingLatency(40.0), FixedTrainingMemory(0)),
            sync_cost=FixedSyncCost(0.0),
        )

    def test_slack_lane_is_not_the_default_router(self):
        base = self.build("request_num_balance", {}).run()
        lane = self.build("slack_lane", {}).run()
        assert lane.sim_time != base.sim_time
        assert lane.versions_done == base.versions_done == 4

    def test_slack_lane_without_upgrades_differs_from_slack_lane_with_them(self):
        nolane = self.build("slack_lane", {"enable_upgrade": False}).run()
        full = self.build("slack_lane", {"enable_upgrade": True}).run()
        assert nolane.sim_time != full.sim_time

    def test_length_lane_is_not_the_default_router(self):
        base = self.build("request_num_balance", {}).run()
        lane = self.build("length_lane", {"n_fast": 1, "threshold": 20_000, "c_fast": 2}).run()
        assert lane.sim_time != base.sim_time
        assert lane.versions_done == base.versions_done == 4
