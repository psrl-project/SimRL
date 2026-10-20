"""`generate_candidates`/`plan_burst`: whole-burst placement, scored by the pipeline forecast."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from psrl.workers.ps.staleness import BufferView, InventorySnapshot

from srl.component.rollout.engine import Request
from srl.cost.rollout import UnifiedRoofline
from srl.forecast import InstanceState, OracleLength, PlanSession, Resident, TrainState
from srl.forecast.burst_plan import ISOLATION_FRACTIONS, generate_candidates, plan_burst

BLOCK = 16
BUDGET = 8192
BIG_POOL = 1_000_000
ORACLE = OracleLength()


def cost() -> UnifiedRoofline:
    return UnifiedRoofline(F=0.006, W=0.002, G=4e-6, A_p=3e-10, A_d=5e-8, b=16)


def bare_state(cost_model, *, now=0.0, running=(), kv_blocks=BIG_POOL, max_concurrency=None) -> InstanceState:
    return InstanceState(
        cost=cost_model, block_size=BLOCK, kv_blocks=kv_blocks, token_budget=BUDGET,
        max_concurrency=max_concurrency, chunk_cap=None, tp=1, now=now, waiting=(), running=tuple(running),
    )


def resident(key: int, ctx: int, remaining: int) -> Resident:
    return Resident(
        key=key, group_id=key, prompt_key=key, prompt_len=min(200, ctx),
        ctx=ctx, prefilled=ctx, remaining=remaining, running=True,
    )


def member_request(rid: int, prompt_len: int, ctx: int, remaining: int, group_id: int | None = None) -> Request:
    gid = rid if group_id is None else group_id
    return Request(
        rid=rid, prompt_len=prompt_len, target_len=(ctx - prompt_len) + remaining, v_traj=0,
        group_id=gid, source_group_id=gid, generated=ctx - prompt_len,
    )


def make_snapshot(reserved_by_buffer, *, num_entries=None, staleness=1):
    top = max(reserved_by_buffer, default=0) + 2
    if num_entries is None:
        num_entries = max((len(v) for v in reserved_by_buffer.values()), default=1)
    buffers = tuple(
        BufferView(buffer_id=k, ready_num_entries=num_entries, n_occupied=0, reserved=reserved_by_buffer.get(k, ()))
        for k in range(top + 1)
    )
    return InventorySnapshot(staleness=staleness, num_entries=num_entries, next_buffer_id=top + 1, buffers=buffers)


@dataclass
class FakeInstance:
    instance_id: int


@dataclass
class StubRuntime:
    instance_states: dict
    snapshot: object
    train: object
    queued_requests: dict = field(default_factory=dict)
    inflight_requests: list = field(default_factory=list)
    pending_groups: dict = field(default_factory=dict)
    staged: dict = field(default_factory=dict)
    batch_size: int = 1
    group_size: int = 1
    rollout_version: int = 0

    @property
    def now(self):
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


def _session(instance_states, reserved, queued, *, staleness=1, num_entries=None):
    snapshot = make_snapshot(reserved, staleness=staleness, num_entries=num_entries)
    train = TrainState(None, 0.0, 10.0, 1.0)
    inflight = [req for reqs in queued.values() for req in reqs]
    runtime = StubRuntime(
        instance_states=instance_states, snapshot=snapshot, train=train,
        queued_requests=queued, inflight_requests=inflight,
    )
    return PlanSession.build(runtime, ORACLE, horizon_buffers=1)


class TestGenerateCandidates:
    def test_zero_fraction_is_always_present_and_assigns_every_request(self):
        model = cost()
        session = _session(
            {0: bare_state(model), 1: bare_state(model)},
            {0: ((1, 0), (2, 0))},
            {1: [member_request(1, 100, 100, 300, group_id=1)], 2: [member_request(2, 100, 100, 50, group_id=2)]},
            num_entries=4,
        )
        assert 0.0 in ISOLATION_FRACTIONS
        candidates = generate_candidates(session, session.tail_order)
        assert any(set(c) == {1, 2} for c in candidates)

    def test_empty_burst_yields_no_candidates(self):
        model = cost()
        session = _session({0: bare_state(model)}, {}, {})
        assert generate_candidates(session, []) == []

    def test_single_instance_isolation_has_nowhere_else_to_go(self):
        model = cost()
        session = _session(
            {0: bare_state(model)},
            {0: ((1, 0), (2, 0), (3, 0))},
            {
                1: [member_request(1, 100, 100, 9000, group_id=1)],
                2: [member_request(2, 100, 100, 50, group_id=2)],
                3: [member_request(3, 100, 100, 50, group_id=3)],
            },
            num_entries=4,
        )
        candidates = generate_candidates(session, session.tail_order)
        assert candidates, "at least the 0% candidate must survive on one instance"
        for candidate in candidates:
            assert set(candidate.values()) == {0}


class TestPlanBurst:
    def test_isolating_a_rare_long_member_beats_mixing_it_with_a_crowd(self):
        """Two instances; instance 0 already has one resident. A new burst of one very long member
        plus several short ones should price at least as well isolated onto the empty instance as
        folded in with everyone by plain earliest-finish (the 0% candidate), because mixing the
        long member in with four short arrivals inflates every short member's ctxsum and their
        buffer's F_k, where isolating it costs nothing (the empty instance has room for it alone).
        """
        model = cost()
        session = _session(
            {0: bare_state(model, running=(resident(0, 2000, 400),)), 1: bare_state(model)},
            {0: ((10, 0),) + tuple((i, 0) for i in range(11, 15))},
            {
                10: [member_request(10, 100, 100, 9000, group_id=10)],
                **{i: [member_request(i, 100, 100, 60, group_id=i)] for i in range(11, 15)},
            },
            num_entries=5,
        )
        burst = session.tail_order
        plan = plan_burst(session, burst, eps=0.0)
        assert plan is not None
        assert set(plan) == {10, 11, 12, 13, 14}
        zero_pipeline = session.price_assignment(burst, generate_candidates(session, burst)[0])
        chosen_pipeline = session.price_assignment(burst, plan)
        assert chosen_pipeline is not None and zero_pipeline is not None
        assert chosen_pipeline.train_end <= zero_pipeline.train_end

    def test_empty_burst_returns_an_empty_plan(self):
        model = cost()
        session = _session({0: bare_state(model)}, {}, {})
        assert plan_burst(session, [], eps=0.0) == {}

    def test_every_request_in_the_burst_is_in_the_winning_plan(self):
        model = cost()
        session = _session(
            {0: bare_state(model), 1: bare_state(model)},
            {0: ((1, 0), (2, 0))},
            {1: [member_request(1, 100, 100, 300, group_id=1)], 2: [member_request(2, 100, 100, 50, group_id=2)]},
            num_entries=4,
        )
        plan = plan_burst(session, session.tail_order, eps=0.0)
        assert plan is not None
        assert set(plan) == {1, 2}
