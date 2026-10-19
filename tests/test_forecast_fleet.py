"""`FleetForecast`, `estimate_queued`, and `forecast` against the engine they mirror."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import pytest
from psrl.workers.ps.staleness import BufferView, InventorySnapshot

from srl.component.rollout.engine import NativeInstance, Request
from srl.cost.rollout import UnifiedRoofline
from srl.forecast import (
    Arrival,
    FleetForecast,
    InstanceForecast,
    InstanceState,
    Interrupt,
    OracleLength,
    PipelineForecast,
    Resident,
    TrainState,
    chain_buffers,
    estimate_queued,
    forecast,
    group_etas,
    members_of,
)

BLOCK = 16
BUDGET = 8192
BIG_POOL = 1_000_000
ORACLE = OracleLength()
PRICING_PATH = 5e-4
"""The replay prices decode-only iterations from samples, so an extra arrival shifts a time by up to this."""


def cost() -> UnifiedRoofline:
    return UnifiedRoofline(F=0.006, W=0.002, G=4e-6, A_p=3e-10, A_d=5e-8, b=16)


class Engine:
    """One `NativeInstance` with the driving helpers this module shares."""

    def __init__(self, cost_model, *, kv_blocks: int = BIG_POOL) -> None:
        self.cost = cost_model
        self.requests: list[Request] = []
        self.inst = NativeInstance(
            cost_model, token_budget=BUDGET, kv_blocks=kv_blocks, block_size=BLOCK, instance_id=0
        )

    def add(self, rid: int, prompt_len: int, target_len: int, t: float) -> None:
        req = Request(
            rid=rid, prompt_len=prompt_len, target_len=target_len, v_traj=0, group_id=rid, source_group_id=rid
        )
        self.requests.append(req)
        self.inst.add_request(req, t)

    def settle(self) -> None:
        while self.inst.waiting:
            self.inst.advance_one_event_committed(0)

    def drain(self) -> None:
        while self.inst.n_active:
            self.inst.advance_one_event_committed(0)

    def cut(self, t: float) -> None:
        while self.inst.t_local < t:
            event = self.inst.advance_one_event(0)
            if event is None:
                self.inst.t_local = t
                return
            if event.t > t:
                self.inst.undo_last_event()
                while self.inst.t_local < t:
                    _, reached = self.inst.advance_to(t, 0)
                    if reached or not self.inst.n_active:
                        return
                return
            self.inst.commit_last_event()

    def apply_interrupt(self, ix: Interrupt) -> None:
        """Mirror `_Drain._apply_interrupt`: abort at `time`, re-add in `rid` order after the gap."""
        self.cut(ix.time)
        aborted = self.inst.abort_all(requeue=False)
        self.inst.t_local = ix.time + ix.resume_delay
        for req in sorted(aborted, key=lambda r: r.rid):
            self.inst.add_request(req, self.inst.t_local)

    def state(self) -> InstanceState:
        return InstanceState.from_instance(
            self.inst, cost=self.cost, token_budget=BUDGET, max_concurrency=None, remaining=ORACLE.remaining
        )


def resident(key: int, ctx: int, remaining: int, group_id: int | None = None) -> Resident:
    return Resident(
        key=key,
        group_id=key if group_id is None else group_id,
        prompt_key=key,
        prompt_len=min(200, ctx),
        ctx=ctx,
        prefilled=ctx,
        remaining=remaining,
        running=True,
    )


def bare_state(cost_model, *, now=0.0, running=()) -> InstanceState:
    return InstanceState(
        cost=cost_model,
        block_size=BLOCK,
        kv_blocks=BIG_POOL,
        token_budget=BUDGET,
        max_concurrency=None,
        chunk_cap=None,
        tp=1,
        now=now,
        waiting=(),
        running=tuple(running),
    )


@dataclass
class FakeInstance:
    instance_id: int


@dataclass
class StubRuntime:
    """The subset of `SchedulingRuntime` that `forecast` reads, over hand-built pieces."""

    instance_states: dict[int, InstanceState]
    snapshot: InventorySnapshot
    train: TrainState
    queued_requests: dict[int, list[Request]] = field(default_factory=dict)
    inflight_requests: list[Request] = field(default_factory=list)
    pending_groups: dict[int, dict[int, float]] = field(default_factory=dict)
    batch_size: int = 1
    group_size: int = 1

    @property
    def now(self) -> float:
        return next(iter(self.instance_states.values())).now

    @property
    def instances(self):
        return [FakeInstance(i) for i in self.instance_states]

    def instance_state(self, inst, remaining_fn):
        return self.instance_states[inst.instance_id]

    def inventory(self):
        return self.snapshot

    def train_state(self, sequence_lengths_hint):
        return self.train

    def queued(self):
        return self.queued_requests

    @property
    def inflight(self):
        return self.inflight_requests

    @property
    def pending(self):
        return self.pending_groups


def snapshot(
    reserved_by_buffer: dict[int, tuple[tuple[int, int], ...]], *, num_entries: int = 4, staleness: int = 1
):
    """Build a snapshot from explicit reservations, opening a couple of empty trailing buffers."""
    top = max(reserved_by_buffer, default=0) + 2
    buffers = tuple(
        BufferView(
            buffer_id=k,
            ready_num_entries=num_entries,
            n_occupied=0,
            reserved=reserved_by_buffer.get(k, ()),
        )
        for k in range(top + 1)
    )
    return InventorySnapshot(staleness=staleness, num_entries=num_entries, next_buffer_id=top + 1, buffers=buffers)


class TestSingleDecisionNoQueue:
    """No queue contention: a `FleetForecast` placement must match the engine exactly."""

    @pytest.mark.parametrize("seed", range(6))
    def test_one_placement_matches_the_engine(self, seed):
        rng = random.Random(100 + seed)
        engine = Engine(cost())
        for rid in range(rng.randint(3, 12)):
            engine.add(rid, rng.randint(100, 4000), rng.randint(300, 3000), 0.0)
        engine.settle()
        state = engine.state()

        rid = 900
        arrival = Arrival(
            key=rid,
            arrive_time=state.now + rng.uniform(0.5, 5.0),
            prompt_len=rng.randint(100, 3000),
            generated=0,
            remaining=rng.randint(200, 2000),
        )
        fleet = FleetForecast({0: InstanceForecast(state)}).with_placement(0, arrival)
        predicted = fleet.finish_times()

        engine.cut(arrival.arrive_time)
        engine.add(rid, arrival.prompt_len, arrival.remaining, arrival.arrive_time)
        engine.drain()
        for req in engine.requests:
            if req.rid in predicted:
                assert predicted[req.rid] == pytest.approx(req.complete_time, rel=1e-6), f"rid {req.rid}."

    def test_with_placement_leaves_the_other_instances_alone(self):
        model = cost()
        a = bare_state(model, running=(resident(0, 2000, 1000),))
        b = bare_state(model, running=(resident(1, 2000, 1000),))
        fleet = FleetForecast({0: InstanceForecast(a), 1: InstanceForecast(b)})
        placed = fleet.with_placement(0, Arrival(9, 0.0, 500, 0, 300))
        assert placed is not fleet
        assert placed.instances[1] is fleet.instances[1]
        assert 9 in placed.finish_times() and 9 not in fleet.finish_times()


class TestChainingProperty:
    def test_a_later_interrupt_never_moves_an_earlier_buffer(self):
        """The second interrupt is scheduled after the first publish, which is after buffer 0 resolved."""
        model = cost()
        rng = random.Random(7)
        residents = tuple(resident(i, rng.randint(1000, 8000), rng.randint(500, 4000)) for i in range(6))
        state = bare_state(model, running=residents)
        members = {r.group_id: [r.key] for r in residents}

        early, late = residents[:3], residents[3:]
        snap = snapshot(
            {0: tuple((r.group_id, 0) for r in early), 1: tuple((r.group_id, 0) for r in late)},
            num_entries=len(early),
        )
        train = TrainState(training_end=None, now=state.now, cycle=10.0, weight_load=2.0)
        proj = PipelineForecast(snap, train)

        base = InstanceForecast(state)
        f0 = proj.evaluate(group_etas(base.finish_times().finish_times, members, {}))
        publish0 = f0.publish_times[0]

        ix1 = Interrupt(time=publish0, resume_delay=2.0)
        with_ix1 = base.with_interrupt(ix1)
        f1 = proj.evaluate(group_etas(with_ix1.finish_times().finish_times, members, {}))
        publish1 = f1.publish_times[1]
        assert publish1 > publish0

        with_both = with_ix1.with_interrupt(Interrupt(time=publish1, resume_delay=2.0))
        f2 = proj.evaluate(group_etas(with_both.finish_times().finish_times, members, {}))

        # Buffer 0 resolved before `publish0`, so neither the first nor the second interrupt touches it.
        assert f1.ready[0] == f0.ready[0] == f2.ready[0]
        assert f1.train_end[0] == f0.train_end[0] == f2.train_end[0]
        # The second interrupt does move the late groups, which only feed buffer 1.
        assert f2.ready[1] > f0.ready[1]

    def test_new_groups_fill_every_buffer_not_only_the_nearest(self):
        """A hypothetical admission that reserves into buffer 1, not buffer 0, must still show there.

        Buffer 0 is already full from a real resident; staleness 0 forces the hypothetical group at
        version 1 to reserve into buffer 1 and nowhere else. If `chain_buffers` dropped `new_groups`
        after its first evaluation, this buffer would report `inf` forever despite the one group it
        needs being fully known up front.
        """
        model = cost()
        filler = resident(0, 2000, 1000, group_id=0)
        state = bare_state(model, now=0.0, running=(filler,))
        members = {0: [0]}
        snap = snapshot({0: ((0, 0),)}, num_entries=1, staleness=0)
        train = TrainState(None, state.now, 10.0, 1.0)
        runtime = StubRuntime({0: state}, snap, train)

        fleet = FleetForecast({0: InstanceForecast(state)})
        eta = group_etas(fleet.finish_times(), members, {})
        new_groups = [(1, state.now + 5.0)]
        result = chain_buffers(fleet, snap, train, eta, members, runtime, horizon=2, new_groups=new_groups)

        assert result.buffers == (0, 1)
        assert math.isfinite(result.ready[0]), "buffer 0 fills from the real resident alone."
        assert result.ready[1] == pytest.approx(state.now + 5.0), (
            "buffer 1 needs the hypothetical group, which must still be offered to it."
        )

    def test_forecast_buffer_zero_is_invariant_to_the_horizon(self):
        """`forecast(horizon=1)` and `forecast(horizon=2)` must agree on their first buffer."""
        model = cost()
        residents = (resident(0, 4000, 6000, group_id=0), resident(1, 4000, 6000, group_id=1))
        state = bare_state(model, now=0.0, running=residents)
        snap = snapshot({0: ((0, 0), (1, 0))}, num_entries=2)
        runtime = StubRuntime({0: state}, snap, TrainState(None, 0.0, 100.0, 10.0))

        one = forecast(runtime, ORACLE, horizon_buffers=1)
        two = forecast(runtime, ORACLE, horizon_buffers=2)
        assert len(one.buffers) == 1 and len(two.buffers) == 2
        assert two.ready[0] == one.ready[0]
        assert two.train_end[0] == one.train_end[0]

    def test_the_sequential_chain_matches_two_real_publishes(self):
        """Anchor the interrupt mechanism against the engine driven through two real publishes."""
        model = cost()
        rng = random.Random(11)
        engine = Engine(model)
        for rid in range(rng.randint(6, 12)):
            engine.add(rid, rng.randint(200, 4000), rng.randint(2000, 8000), 0.0)
        engine.settle()
        state = engine.state()
        ix1 = Interrupt(time=state.now + 20.0, resume_delay=4.0)
        ix2 = Interrupt(time=state.now + 60.0, resume_delay=4.0)

        predicted = InstanceForecast(state, interrupts=(ix1, ix2)).finish_times().finish_times

        engine.apply_interrupt(ix1)
        engine.apply_interrupt(ix2)
        engine.drain()
        for req in engine.requests:
            assert predicted[req.rid] == pytest.approx(req.complete_time, rel=1e-5), f"rid {req.rid}."


class TestPriorityQueuedPlacement:
    @staticmethod
    def _runtime(model, specs):
        reqs = [
            Request(rid=rid, prompt_len=p, target_len=t, v_traj=0, group_id=0, source_group_id=rid)
            for rid, p, t in specs
        ]
        return StubRuntime(
            {0: bare_state(model, now=0.0)},
            snapshot({}, num_entries=4),
            TrainState(None, 0.0, 10.0, 1.0),
            queued_requests={0: list(reqs)},
            inflight_requests=list(reqs),
        )

    def test_the_longer_member_is_placed_first(self):
        model = cost()
        runtime = self._runtime(model, [(0, 2000, 4000), (1, 200, 300)])
        fleet = FleetForecast({0: InstanceForecast(runtime.instance_states[0])})
        _, finish = estimate_queued(fleet, runtime.queued(), lambda gid: 0, ORACLE.remaining, now=runtime.now)
        assert finish[0] > finish[1]

    def test_the_second_member_sees_the_first(self):
        """The chosen instance's forecast carries the first member when the second is evaluated."""
        model = cost()
        runtime = self._runtime(model, [(0, 2000, 4000), (1, 200, 300)])
        fleet = FleetForecast({0: InstanceForecast(runtime.instance_states[0])})
        _, finish = estimate_queued(fleet, runtime.queued(), lambda gid: 0, ORACLE.remaining, now=runtime.now)

        first = Arrival(0, 0.0, 2000, 0, 4000)
        second = Arrival(1, 0.0, 200, 0, 300)
        reference = fleet.with_placement(0, first).with_placement(0, second).finish_times()
        assert finish[0] == pytest.approx(reference[0], rel=PRICING_PATH)
        assert finish[1] == pytest.approx(reference[1], rel=PRICING_PATH)

    def test_folding_raises_the_second_member_above_an_unfolded_evaluation(self):
        """Mutation guard: evaluating both against the empty instance would understate the second."""
        model = cost()
        runtime = self._runtime(model, [(0, 2000, 4000), (1, 200, 300)])
        empty = FleetForecast({0: InstanceForecast(runtime.instance_states[0])})
        alone_second = empty.with_placement(0, Arrival(1, 0.0, 200, 0, 300)).finish_times()
        _, finish = estimate_queued(empty, runtime.queued(), lambda gid: 0, ORACLE.remaining, now=runtime.now)
        assert finish[1] > alone_second[1]

    def test_a_member_no_instance_can_admit_is_left_unplaced(self):
        model = cost()
        reqs = [Request(rid=0, prompt_len=9000, target_len=9000, v_traj=0, group_id=0, source_group_id=0)]
        state = InstanceState(
            cost=model, block_size=BLOCK, kv_blocks=10, token_budget=BUDGET, max_concurrency=None,
            chunk_cap=None, tp=1, now=0.0, waiting=(), running=(),
        )
        runtime = StubRuntime(
            {0: state}, snapshot({}, num_entries=4), TrainState(None, 0.0, 10.0, 1.0),
            queued_requests={0: reqs}, inflight_requests=list(reqs),
        )
        fleet = FleetForecast({0: InstanceForecast(state)})
        _, finish = estimate_queued(fleet, runtime.queued(), lambda gid: 0, ORACLE.remaining, now=0.0)
        assert finish == {}


class TestMembersOfAgainstARealRun:
    def test_membership_covers_inflight_and_pending(self):
        from srl.component.sync.cost import FixedSyncCost
        from srl.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel
        from srl.policy.psrl import PSRLPolicy
        from srl.simulator.config import SimConfig
        from srl.simulator.runtime import SchedulingRuntime
        from srl.simulator.simulator import Simulator
        from srl.workload.grouped import GroupedWorkload
        from srl.workload.trace import RequestGroup

        workload = GroupedWorkload.from_groups(
            [
                RequestGroup(group_id=0, prompt_len=300, output_lens=(800, 1800)),
                RequestGroup(group_id=1, prompt_len=900, output_lens=(400, 1200)),
                RequestGroup(group_id=2, prompt_len=200, output_lens=(2000, 600)),
            ]
        )
        cfg = SimConfig(
            batch_size=2,
            group_size=2,
            staleness_horizon=1,
            n_versions=4,
            n_instances=2,
            weight_load_time=1.0,
            workload=workload,
            log_telemetry=False,
        )
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.004, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(5.0), FixedTrainingMemory(0)),
            sync_cost=FixedSyncCost(0.5),
            policy=PSRLPolicy(),
            cfg=cfg,
        )
        sim.run()
        runtime = SchedulingRuntime(sim)
        members = members_of(runtime)
        live = {req.group_id for req in runtime.inflight} | set(runtime.pending)
        assert set(members) == live
        for group_id, rids in members.items():
            assert len(rids) == len(set(rids)), f"Group {group_id} has duplicate member ids."
