"""`PlanSession` against brute-force rebuilds, the real engine, and psrl's own replay."""

from __future__ import annotations

import dataclasses
import math
import random
import time
from dataclasses import dataclass, field

import pytest
from psrl.workers.ps.inventory_forecast import BufferView, InventorySnapshot, forecast_ready_times

from srl.component.rollout.engine import NativeInstance, Request, traj_view
from srl.cost.rollout import UnifiedRoofline
from srl.forecast import (
    FleetForecast,
    InstanceForecast,
    InstanceState,
    OracleLength,
    PlanSession,
    Resident,
    TrainState,
    arrival_of,
    chain_buffers,
    fold_member,
    forecast,
    group_etas_from_fleet,
    placement_order,
)
from srl.forecast.plan import _without_arrival

BLOCK = 16
BUDGET = 8192
BIG_POOL = 1_000_000
ORACLE = OracleLength()
PRICING_PATH = 5e-4


def cost() -> UnifiedRoofline:
    return UnifiedRoofline(F=0.006, W=0.002, G=4e-6, A_p=3e-10, A_d=5e-8, b=16)


def bare_state(cost_model, *, now=0.0, running=(), waiting=(), kv_blocks=BIG_POOL) -> InstanceState:
    return InstanceState(
        cost=cost_model,
        block_size=BLOCK,
        kv_blocks=kv_blocks,
        token_budget=BUDGET,
        max_concurrency=None,
        chunk_cap=None,
        tp=1,
        now=now,
        waiting=tuple(waiting),
        running=tuple(running),
    )


def resident(key: int, ctx: int, remaining: int) -> Resident:
    return Resident(
        key=key, group_id=key, prompt_key=key, prompt_len=min(200, ctx),
        ctx=ctx, prefilled=ctx, remaining=remaining, running=True,
    )


def member_request(rid: int, prompt_len: int, ctx: int, remaining: int, group_id: int | None = None) -> Request:
    """A queued or running member whose `OracleLength.remaining` equals `remaining` exactly."""
    gid = rid if group_id is None else group_id
    generated = ctx - prompt_len
    return Request(
        rid=rid, prompt_len=prompt_len, target_len=generated + remaining, v_traj=0,
        group_id=gid, source_group_id=gid, generated=generated,
    )


def make_snapshot(reserved_by_buffer: dict[int, tuple[tuple[int, int], ...]], *, num_entries=None, staleness=1):
    top = max(reserved_by_buffer, default=0) + 2
    if num_entries is None:
        num_entries = max((len(v) for v in reserved_by_buffer.values()), default=1)
    buffers = tuple(
        BufferView(
            buffer_id=k, ready_num_entries=num_entries, n_occupied=0,
            reserved=reserved_by_buffer.get(k, ()),
        )
        for k in range(top + 1)
    )
    return InventorySnapshot(staleness=staleness, num_entries=num_entries, next_buffer_id=top + 1, buffers=buffers)


@dataclass
class FakeInstance:
    instance_id: int


@dataclass
class StubRuntime:
    """The subset of `SchedulingRuntime` that `PlanSession` reads, over hand-built pieces."""

    instance_states: dict[int, InstanceState]
    snapshot: InventorySnapshot
    train: TrainState
    queued_requests: dict[int, list[Request]] = field(default_factory=dict)
    inflight_requests: list[Request] = field(default_factory=list)
    pending_groups: dict[int, dict[int, float]] = field(default_factory=dict)
    staged: dict[int, list[Request]] = field(default_factory=dict)
    batch_size: int = 1
    group_size: int = 1
    rollout_version: int = 0

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

    @property
    def staged_groups(self):
        return self.staged


def random_scenario(seed: int, *, n_tail: int | None = None):
    """A random fleet, with a tail of queued members of admitted (single-member) groups."""
    rng = random.Random(seed)
    model = cost()
    n_instances = rng.randint(1, 3)

    states: dict[int, InstanceState] = {}
    inflight: list[Request] = []
    for i in range(n_instances):
        n_res = rng.randint(3, 12)
        residents, requests = [], []
        for _ in range(n_res):
            rid = len(inflight) + len(residents) + 10_000 * i
            ctx = rng.randint(500, 4000)
            remaining = rng.randint(200, 3000)
            residents.append(resident(rid, ctx, remaining))
            requests.append(member_request(rid, min(200, ctx), ctx, remaining))
        states[i] = bare_state(model, running=tuple(residents))
        inflight.extend(requests)

    n_tail = rng.randint(0, 10) if n_tail is None else n_tail
    tail_requests = []
    reserved_by_buffer: dict[int, tuple[tuple[int, int], ...]] = {}
    for k in range(n_tail):
        rid = 90_000 + k
        prompt_len = rng.randint(100, 2000)
        remaining = rng.randint(200, 3000)
        req = member_request(rid, prompt_len, prompt_len, remaining)
        tail_requests.append(req)
        buffer_id = rng.randint(0, 2)
        reserved_by_buffer[buffer_id] = (*reserved_by_buffer.get(buffer_id, ()), (rid, buffer_id))
    for req in inflight:
        buffer_id = rng.randint(0, 2)
        reserved_by_buffer[buffer_id] = (*reserved_by_buffer.get(buffer_id, ()), (req.group_id, buffer_id))

    snapshot = make_snapshot(reserved_by_buffer)
    train = TrainState(training_end=None, now=0.0, cycle=10.0, weight_load=2.0)
    queued_by_group = {req.group_id: [req] for req in tail_requests}
    runtime = StubRuntime(
        instance_states=states,
        snapshot=snapshot,
        train=train,
        queued_requests=queued_by_group,
        inflight_requests=[*inflight, *tail_requests],
    )
    return runtime, tail_requests


# --- a reference built independently of `PlanSession`'s incremental surgery --------------------


def reference_tail(runtime, exclude_rid=None):
    """Fold every queued admitted member but `exclude_rid` into a fresh fleet, from scratch."""
    remaining_of = ORACLE.remaining
    members = {}
    for req in runtime.inflight:
        members.setdefault(req.group_id, []).append(req.rid)
    fleet = FleetForecast({i: InstanceForecast(state) for i, state in runtime.instance_states.items()})
    queued = {gid: reqs for gid, reqs in runtime.queued().items() if gid in members}
    order = placement_order(queued, runtime.snapshot.deadline, remaining_of)
    assignment = {}
    for req in order:
        if req.rid == exclude_rid:
            continue
        fleet, chosen, _ = fold_member(fleet, req, remaining_of, now=runtime.now)
        if chosen is not None:
            assignment[req.rid] = chosen
    return fleet, assignment, members


def reference_pipeline(runtime, req, option, horizon, defer_until):
    """Rebuild, from scratch, the pipeline `PlanSession.evaluate_member` should produce for `option`."""
    remaining_of = ORACLE.remaining
    full_fleet, full_assignment, members = reference_tail(runtime, exclude_rid=None)
    without_fleet, _, _ = reference_tail(runtime, exclude_rid=req.rid)
    src = full_assignment.get(req.rid)
    if option is not None:
        fleet = without_fleet.with_placement(option, arrival_of(req, runtime.now, remaining_of))
    elif src is not None:
        deferred = dataclasses.replace(arrival_of(req, runtime.now, remaining_of), arrive_time=defer_until)
        fleet = without_fleet.with_placement(src, deferred)
    else:
        fleet, chosen, _ = fold_member(without_fleet, req, remaining_of, now=defer_until)
        if chosen is None:
            return None
    touched = fleet.instances[option if option is not None else (src if src is not None else chosen)]
    if touched.finish_times().kv_overflow:
        return None
    eta = group_etas_from_fleet(fleet, members, runtime)
    train = runtime.train_state(())
    return chain_buffers(fleet, runtime.snapshot, train, eta, members, runtime, horizon)


class TestEquivalenceToBruteForce:
    @pytest.mark.parametrize("seed", range(20))
    def test_evaluate_member_matches_a_from_scratch_rebuild(self, seed):
        runtime, tail_requests = random_scenario(seed)
        if not tail_requests:
            pytest.skip("No tail member to evaluate for this seed.")
        session = PlanSession.build(runtime, ORACLE, horizon_buffers=2)
        req = tail_requests[seed % len(tail_requests)]

        result = session.evaluate_member(req)
        for option in (*runtime.instance_states, None):
            expected = reference_pipeline(runtime, req, option, session.horizon, session.defer_until)
            got = result[option]
            if expected is None:
                assert got is None, f"seed {seed} option {option}: expected invalid."
                continue
            assert got is not None, f"seed {seed} option {option}: expected a pipeline."
            assert got.buffers == expected.buffers
            assert got.ready == pytest.approx(expected.ready, rel=PRICING_PATH, nan_ok=True)
            assert got.train_end == pytest.approx(expected.train_end, rel=PRICING_PATH, nan_ok=True)
            assert got.publish_times == pytest.approx(expected.publish_times, rel=PRICING_PATH, nan_ok=True)


class TestEmptyTailMatchesForecast:
    def test_matches_forecast_override_with_nothing_else_queued(self):
        runtime, _ = random_scenario(seed=3, n_tail=0)
        session = PlanSession.build(runtime, ORACLE, horizon_buffers=2)
        rid = 70_000
        req = member_request(rid, 300, 300, 900)

        for instance_id in runtime.instance_states:
            got = _evaluate_as_tail_member(session, req, instance_id)
            arrival = arrival_of(req, runtime.now, ORACLE.remaining)
            expected = forecast(runtime, ORACLE, horizon_buffers=2, override=(instance_id, arrival))
            assert got is not None
            assert got.buffers == expected.buffers
            assert got.train_end == pytest.approx(expected.train_end, rel=PRICING_PATH)
            assert got.ready == pytest.approx(expected.ready, rel=PRICING_PATH)
            assert got.publish_times == pytest.approx(expected.publish_times, rel=PRICING_PATH)


def _evaluate_as_tail_member(session, req, instance_id):
    """Price `req`, which is not part of the session's own tail, landing on `instance_id`.

    Exercises the exact path `evaluate_member` uses for a tail member with no tentative placement.
    """
    base = dict(session.fleet.instances)
    resolved = (instance_id, arrival_of(req, session.now, ORACLE.remaining))
    return session._price_with(base, resolved)



class _Engine:
    """One `NativeInstance`, with the cut/settle/drain helpers `tests/test_forecast_fleet.py` uses."""

    def __init__(self, cost_model, kv_blocks: int = BIG_POOL) -> None:
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

    def state(self) -> InstanceState:
        return InstanceState.from_instance(
            self.inst, cost=self.cost, token_budget=BUDGET, max_concurrency=None, remaining=ORACLE.remaining
        )


class TestDeferAgainstTheRealEngine:
    def test_defer_then_place_matches_the_session(self):
        rng = random.Random(42)
        model = cost()
        engine = _Engine(model)
        for rid in range(rng.randint(4, 8)):
            engine.add(rid, rng.randint(200, 2000), rng.randint(2000, 6000), 0.0)
        engine.settle()
        state = engine.state()

        new_rid = 999
        new_req = member_request(new_rid, 400, 400, 1500)
        snap = make_snapshot(
            {0: tuple((r.rid, 0) for r in engine.requests), 5: ((new_rid, 5),)},
            num_entries=len(engine.requests),
        )
        train = TrainState(training_end=None, now=state.now, cycle=100.0, weight_load=10.0)
        runtime = StubRuntime(
            instance_states={0: state},
            snapshot=snap,
            train=train,
            queued_requests={new_rid: [new_req]},
            inflight_requests=[*engine.requests, new_req],
        )

        session = PlanSession.build(runtime, ORACLE, horizon_buffers=1)
        assert session.defer_valid
        defer_until = session.defer_until
        predicted = session.evaluate_member(new_req)[None]
        assert predicted is not None

        # Drive the real engine: nothing happens until `defer_until`, then the member is placed.
        engine.cut(defer_until)
        engine.add(new_rid, new_req.prompt_len, new_req.target_len, defer_until)
        engine.drain()
        real_finish = next(r.complete_time for r in engine.requests if r.rid == new_rid)

        # Reconstruct the session's own deferred placement, to compare the raw finish time it
        # predicts (not the chained training pipeline) against the real engine.
        src = session.tail_assignment.get(new_rid)
        base = dict(session.fleet.instances)
        if src is not None:
            base[src] = _without_arrival(base[src], new_rid)
        resolved = session._member_arrival(new_req, None, src, base)
        assert resolved is not None
        target, arrival = resolved
        touched = base[target].with_arrival(arrival)
        session_finish = touched.finish_times().finish_times[new_rid]
        assert session_finish == pytest.approx(real_finish, rel=1e-5)


    def test_defer_until_is_the_earliest_future_finish_capped_at_the_first_publish(self):
        runtime, _ = random_scenario(seed=5, n_tail=3)
        session = PlanSession.build(runtime, ORACLE, horizon_buffers=1)
        finishes = session.fleet.finish_times()
        future = [t for t in finishes.values() if math.isfinite(t) and t > session.now]
        expected = min(future)
        baseline = session._baseline_pipeline()
        if baseline.publish_times:
            expected = min(expected, baseline.publish_times[0])
        assert session.defer_until == pytest.approx(expected, rel=1e-9)

    def test_defer_invalid_when_nothing_is_resident_anywhere(self):
        model = cost()
        state = bare_state(model, running=())
        snap = make_snapshot({}, num_entries=4)
        train = TrainState(training_end=None, now=0.0, cycle=10.0, weight_load=1.0)
        runtime = StubRuntime(instance_states={0: state}, snapshot=snap, train=train)
        session = PlanSession.build(runtime, ORACLE, horizon_buffers=1)
        assert not session.defer_valid
        assert session.defer_until is None


class TestGroupAdmission:
    def _runtime_with_staged_group(self, seed: int, group_size: int = 3):
        """A runtime whose only pending buffer needs exactly the staged group to turn READY.

        Staleness 0 keeps the window to buffer 0 alone, and `num_entries` is sized to the existing
        groups plus one, so admitting the staged group is the only way buffer 0 ever fills.
        """
        runtime, _ = random_scenario(seed, n_tail=0)
        existing_groups = sorted({req.group_id for req in runtime.inflight})
        reserved0 = tuple((gid, 0) for gid in existing_groups)
        num_entries = len(existing_groups) + 1
        runtime.snapshot = InventorySnapshot(
            staleness=0,
            num_entries=num_entries,
            next_buffer_id=1,
            buffers=(BufferView(buffer_id=0, ready_num_entries=num_entries, n_occupied=0, reserved=reserved0),),
        )
        rng = random.Random(seed + 500)
        group_id = 55_000
        members = []
        for index in range(group_size):
            rid = group_id * 10 + index
            prompt_len = rng.randint(200, 1500)
            remaining = rng.randint(300, 4000)
            members.append(member_request(rid, prompt_len, prompt_len, remaining, group_id=group_id))
        runtime.staged[group_id] = members
        return runtime, members

    def test_admission_matches_a_greedy_brute_force_construction(self):
        runtime, members = self._runtime_with_staged_group(seed=11)
        session = PlanSession.build(runtime, ORACLE, horizon_buffers=2)
        result = session.evaluate_admission(members)

        remaining_of = ORACLE.remaining
        f, *rest = members
        rest_sorted = sorted(rest, key=lambda r: -remaining_of(r))
        found_a_finite_ready = False
        for instance_id in runtime.instance_states:
            fleet = FleetForecast({i: InstanceForecast(s) for i, s in runtime.instance_states.items()})
            fleet = fleet.with_placement(instance_id, arrival_of(f, runtime.now, remaining_of))
            first = fleet.instances[instance_id].finish_times()
            if first.kv_overflow:
                assert result[instance_id] is None
                continue
            group_eta = first.finish_times[f.rid]
            ok = True
            for member in rest_sorted:
                fleet, chosen, finish = fold_member(fleet, member, remaining_of, now=runtime.now)
                if chosen is None:
                    ok = False
                    break
                group_eta = max(group_eta, finish)
            if not ok:
                assert result[instance_id] is None
                continue
            eta = group_etas_from_fleet(fleet, session.members, runtime)
            train = runtime.train_state(())
            expected = chain_buffers(
                fleet, runtime.snapshot, train, eta, session.members,
                runtime, session.horizon, new_groups=[(runtime.rollout_version, group_eta)],
            )
            got = result[instance_id]
            assert got is not None
            assert math.isfinite(expected.ready[0])
            found_a_finite_ready = True
            assert got.train_end == pytest.approx(expected.train_end, rel=PRICING_PATH)
            assert got.ready == pytest.approx(expected.ready, rel=PRICING_PATH)
        assert found_a_finite_ready, "No instance option exercised the new group filling buffer 0."


    def test_defer_group_equals_the_baseline_with_no_new_group(self):
        runtime, members = self._runtime_with_staged_group(seed=12)
        session = PlanSession.build(runtime, ORACLE, horizon_buffers=2)
        result = session.evaluate_admission(members)
        baseline = session._baseline_pipeline()
        assert result[None].train_end == baseline.train_end
        assert result[None].ready == baseline.ready

    def test_group_that_overflows_kv_is_invalid(self):
        model = cost()
        state = bare_state(model, running=(), kv_blocks=2)
        runtime = StubRuntime(
            instance_states={0: state}, snapshot=make_snapshot({}, num_entries=4),
            train=TrainState(None, 0.0, 10.0, 1.0),
        )
        group_id = 1
        members = [member_request(group_id * 10 + i, 9000, 9000, 9000, group_id=group_id) for i in range(2)]
        runtime.staged[group_id] = members
        session = PlanSession.build(runtime, ORACLE, horizon_buffers=1)
        result = session.evaluate_admission(members)
        assert result[0] is None

    def test_kv_watermark_refuses_admission_below_literal_overflow(self):
        """A group that would fit without overflowing can still be refused by the softer watermark.

        prompt_len=16, remaining=144 peaks at 160 tokens = 10 blocks of 16. A 20-block pool never
        overflows (10 <= 20), so only the watermark -- not `kv_overflow` -- can explain a refusal.
        """
        model = cost()
        state = bare_state(model, running=(), kv_blocks=20)
        runtime = StubRuntime(
            instance_states={0: state}, snapshot=make_snapshot({}, num_entries=4),
            train=TrainState(None, 0.0, 10.0, 1.0),
        )
        group_id = 1
        members = [member_request(group_id * 10, 16, 16, 144, group_id=group_id)]
        runtime.staged[group_id] = members

        unwatermarked = PlanSession.build(runtime, ORACLE, horizon_buffers=1)
        assert unwatermarked.evaluate_admission(members)[0] is not None

        half_full = PlanSession.build(runtime, ORACLE, horizon_buffers=1, kv_watermark=0.3)
        assert half_full.evaluate_admission(members)[0] is None

        loose = PlanSession.build(runtime, ORACLE, horizon_buffers=1, kv_watermark=0.6)
        assert loose.evaluate_admission(members)[0] is not None

    def test_admission_changes_the_pipeline_consistently_with_psrl_forecast_ready_times(self):
        runtime, members = self._runtime_with_staged_group(seed=13)
        session = PlanSession.build(runtime, ORACLE, horizon_buffers=2)
        result = session.evaluate_admission(members)
        instance_id = next(iter(runtime.instance_states))
        pipeline = result[instance_id]
        assert pipeline is not None

        remaining_of = ORACLE.remaining
        f, *rest = members
        fleet = session.fleet.with_placement(instance_id, arrival_of(f, runtime.now, remaining_of))
        group_eta = fleet.instances[instance_id].finish_times().finish_times[f.rid]
        rest_sorted = sorted(rest, key=lambda r: -remaining_of(r))
        for member in rest_sorted:
            fleet, _, finish = fold_member(fleet, member, remaining_of, now=runtime.now)
            group_eta = max(group_eta, finish)

        base_eta = group_etas_from_fleet(session.fleet, session.members, runtime)
        direct = forecast_ready_times(runtime.snapshot, base_eta, new_groups=[(runtime.rollout_version, group_eta)])
        assert math.isfinite(direct[pipeline.buffers[0]])
        assert pipeline.ready == tuple(direct[b] for b in pipeline.buffers)


class TestCommitRebuildConsistency:
    def test_committing_several_members_then_rebuilding_agrees_with_a_fresh_session(self):
        runtime, tail_requests = random_scenario(seed=21, n_tail=6)
        session = PlanSession.build(runtime, ORACLE, horizon_buffers=2)

        decisions = []
        for req in list(tail_requests)[:3]:
            options = session.evaluate_member(req)
            chosen = next((opt for opt, pipe in options.items() if pipe is not None and opt is not None), None)
            if chosen is None:
                continue
            session.commit(req, chosen)
            decisions.append((req, chosen))

        # Rebuild a runtime whose instances already contain the committed decisions, matching what
        # the live engine would hold after `add_request` placed each one.
        placed_by_instance: dict[int, list] = {i: [] for i in runtime.instance_states}
        for req, instance_id in decisions:
            placed_by_instance[instance_id].append(req)
        new_states = {}
        for i, state in runtime.instance_states.items():
            extra = [
                resident(req.rid, req.prompt_len + req.generated, ORACLE.remaining(traj_view(req)))
                for req in placed_by_instance[i]
            ]
            new_states[i] = bare_state(cost(), running=(*state.running, *tuple(extra)))
        remaining_tail = [req for req in tail_requests if req.rid not in {r.rid for r, _ in decisions}]
        rebuilt = StubRuntime(
            instance_states=new_states,
            snapshot=runtime.snapshot,
            train=runtime.train,
            queued_requests={req.group_id: [req] for req in remaining_tail},
            inflight_requests=runtime.inflight_requests,
        )
        fresh = PlanSession.build(rebuilt, ORACLE, horizon_buffers=2)

        if not remaining_tail:
            pytest.skip("Nothing left to compare after three commits.")
        probe = remaining_tail[0]
        committed_options = session.evaluate_member(probe)
        fresh_options = fresh.evaluate_member(probe)
        for option in (*runtime.instance_states, None):
            a, b = committed_options[option], fresh_options[option]
            if a is None or b is None:
                assert a is None and b is None, f"option {option} disagrees on feasibility."
                continue
            assert a.train_end == pytest.approx(b.train_end, rel=1e-3)


def _traj(req: Request):
    return traj_view(req)


class TestPerformanceAtHeadroomScale:
    def test_build_and_evaluate_latency(self, capsys):
        model = cost()
        rng = random.Random(7)
        n_instances = 2
        states = {}
        inflight: list[Request] = []
        for i in range(n_instances):
            residents, requests = [], []
            for _ in range(64):
                rid = len(inflight) + len(residents) + 100_000 * i
                ctx = rng.randint(500, 4000)
                remaining = rng.randint(200, 3000)
                residents.append(resident(rid, ctx, remaining))
                requests.append(member_request(rid, min(200, ctx), ctx, remaining))
            states[i] = bare_state(model, running=tuple(residents))
            inflight.extend(requests)

        tail_requests = []
        reserved: dict[int, tuple] = {}
        n_groups = 50
        for g in range(n_groups):
            buffer_id = g % 3
            for j in range(8):
                rid = 900_000 + g * 8 + j
                prompt_len = rng.randint(100, 2000)
                remaining = rng.randint(200, 3000)
                req = member_request(rid, prompt_len, prompt_len, remaining, group_id=900_000 + g)
                tail_requests.append(req)
            reserved[buffer_id] = (*reserved.get(buffer_id, ()), (900_000 + g, buffer_id))
        for req in inflight:
            reserved[0] = (*reserved.get(0, ()), (req.group_id, 0))

        snapshot = make_snapshot(reserved)
        train = TrainState(training_end=None, now=0.0, cycle=10.0, weight_load=2.0)
        queued_by_group: dict[int, list[Request]] = {}
        for req in tail_requests:
            queued_by_group.setdefault(req.group_id, []).append(req)
        runtime = StubRuntime(
            instance_states=states, snapshot=snapshot, train=train,
            queued_requests=queued_by_group, inflight_requests=[*inflight, *tail_requests],
        )

        t0 = time.perf_counter()
        session = PlanSession.build(runtime, ORACLE, horizon_buffers=2)
        build_s = time.perf_counter() - t0

        probe = tail_requests[0]
        t0 = time.perf_counter()
        session.evaluate_member(probe)
        member_s = time.perf_counter() - t0

        group_id = 999_000
        group_members = []
        for i in range(8):
            prompt_len = rng.randint(100, 2000)
            group_members.append(
                member_request(group_id + i, prompt_len, prompt_len, rng.randint(200, 3000), group_id=group_id)
            )
        runtime.staged[group_id] = group_members
        t0 = time.perf_counter()
        session.evaluate_admission(group_members)
        admission_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        session.commit(probe, next(iter(runtime.instance_states)))
        commit_s = time.perf_counter() - t0

        with capsys.disabled():
            print(
                f"\n[perf] build={build_s * 1e3:.1f}ms member={member_s * 1e3:.1f}ms "
                f"admission={admission_s * 1e3:.1f}ms commit={commit_s * 1e3:.1f}ms "
                f"Q={len(tail_requests)}"
            )
        assert build_s < 3.0
        assert member_s < 0.4
        assert admission_s < 0.8
