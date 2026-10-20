"""`BurstIsolation`: begin_dispatch plans the tail once, select_context reads the plan."""

from __future__ import annotations

from dataclasses import dataclass, field

from psrl.workers.ps.staleness import BufferView, InventorySnapshot

from srl.component.rollout.engine import InstanceView, Request
from srl.component.rollout.router.strategies.burst_isolation import BurstIsolation
from srl.cost.rollout import UnifiedRoofline
from srl.forecast import InstanceState, Resident, TrainState

BLOCK = 16
BUDGET = 8192
BIG_POOL = 1_000_000


def cost() -> UnifiedRoofline:
    return UnifiedRoofline(F=0.006, W=0.002, G=4e-6, A_p=3e-10, A_d=5e-8, b=16)


def bare_state(cost_model, *, now=0.0, running=(), kv_blocks=BIG_POOL) -> InstanceState:
    return InstanceState(
        cost=cost_model, block_size=BLOCK, kv_blocks=kv_blocks, token_budget=BUDGET,
        max_concurrency=None, chunk_cap=None, tp=1, now=now, waiting=(), running=tuple(running),
    )


def resident(key: int, ctx: int, remaining: int) -> Resident:
    return Resident(
        key=key, group_id=key, prompt_key=key, prompt_len=min(200, ctx),
        ctx=ctx, prefilled=ctx, remaining=remaining, running=True,
    )


def member_request(rid: int, prompt_len: int, remaining: int, group_id: int | None = None) -> Request:
    gid = rid if group_id is None else group_id
    return Request(rid=rid, prompt_len=prompt_len, target_len=remaining, v_traj=0, group_id=gid, source_group_id=gid)


def view(instance_id: int, n_running: int = 0) -> InstanceView:
    return InstanceView(
        instance_id=instance_id, n_waiting=0, n_running=n_running, kv_utilization=0.0, total_ctx=0,
        free_blocks=0, committed_blocks=0, prefix_hit_tokens=lambda req: 0, step_time_delta=lambda req: 0.0,
    )


def make_snapshot(reserved_by_buffer, *, num_entries=4, staleness=1):
    top = max(reserved_by_buffer, default=0) + 2
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
    horizon: int | None = None
    ingress_window_groups: int = 1

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


def _runtime(reserved, queued, instance_states=None):
    model = cost()
    instance_states = instance_states or {0: bare_state(model), 1: bare_state(model)}
    inflight = [req for reqs in queued.values() for req in reqs]
    return StubRuntime(
        instance_states=instance_states, snapshot=make_snapshot(reserved), train=TrainState(None, 0.0, 10.0, 1.0),
        queued_requests=queued, inflight_requests=inflight,
    )


class TestBeginDispatch:
    def test_plans_the_tail_once_per_burst(self):
        s = BurstIsolation(length_model="oracle")
        req1, req2 = member_request(1, 100, 300, group_id=1), member_request(2, 100, 50, group_id=2)
        runtime = _runtime({0: ((1, 0), (2, 0))}, {1: [req1], 2: [req2]})
        s.bind_runtime(runtime)
        s.begin_dispatch(0.0)
        assert s._plan is not None
        assert set(s._plan) == {1, 2}

    def test_empty_tail_yields_an_empty_plan(self):
        s = BurstIsolation(length_model="oracle")
        runtime = _runtime({}, {}, instance_states={0: bare_state(cost())})
        s.bind_runtime(runtime)
        s.begin_dispatch(0.0)
        assert s._plan == {}


class TestSelectContext:
    def test_reads_the_plan_for_an_admitted_member(self):
        s = BurstIsolation(length_model="oracle")
        req1, req2 = member_request(1, 100, 300, group_id=1), member_request(2, 100, 50, group_id=2)
        runtime = _runtime({0: ((1, 0), (2, 0))}, {1: [req1], 2: [req2]})
        s.bind_runtime(runtime)
        s.begin_dispatch(0.0)
        req = next(r for r in s._session.tail_order if r.rid == 1)
        target = s._plan[1]
        views = [view(0), view(1)]
        index = s.select_context(req, views, None)
        assert index == next(i for i, v in enumerate(views) if v.instance_id == target)
        assert 1 not in s._plan

    def test_falls_back_when_the_planned_instance_is_not_a_candidate(self):
        s = BurstIsolation(length_model="oracle")
        req1, req2 = member_request(1, 100, 300, group_id=1), member_request(2, 100, 50, group_id=2)
        runtime = _runtime({0: ((1, 0), (2, 0))}, {1: [req1], 2: [req2]})
        s.bind_runtime(runtime)
        s.begin_dispatch(0.0)
        req = next(r for r in s._session.tail_order if r.rid == 1)
        target = s._plan[1]
        other = 1 if target == 0 else 0
        # Only offer the instance the plan did NOT pick; must still place it somewhere, not hang.
        index = s.select_context(req, [view(other)], None)
        assert index == 0

    def test_falls_back_when_the_plan_has_gone_stale_since_begin_dispatch(self):
        """An earlier commit this same burst can consume enough KV that a later member's planned
        instance, priced once at `begin_dispatch`, no longer fits. `select_context` must re-price
        before committing and fall back instead of letting `commit_assignment`'s `ValueError`
        propagate and kill the whole dispatch."""
        s = BurstIsolation(length_model="oracle")
        req1, req2 = member_request(1, 100, 300, group_id=1), member_request(2, 100, 50, group_id=2)
        runtime = _runtime({0: ((1, 0), (2, 0))}, {1: [req1], 2: [req2]})
        s.bind_runtime(runtime)
        s.begin_dispatch(0.0)
        req = next(r for r in s._session.tail_order if r.rid == 1)
        target = s._plan[1]
        # Force the plan's own target to overflow: shrink that instance's declared pool to nothing
        # left over, by replacing its state with a one-block pool and re-pricing it as staged.
        import dataclasses as dc

        stale_state = dc.replace(s._session.fleet.instances[target].state, kv_blocks=1)
        stale_fc = type(s._session.fleet.instances[target])(stale_state)
        s._session.fleet = type(s._session.fleet)({**s._session.fleet.instances, target: stale_fc})
        index = s.select_context(req, [view(0), view(1)], None)
        assert index is not None, "a feasible instance exists even though the planned one overflowed"
        assert 1 not in s._plan


class TestRegistration:
    def test_is_registered_under_its_own_name(self):
        import srl.component.rollout.router.strategies  # noqa: F401  (runs the registrations)
        from srl.component.rollout.router.strategy import STRATEGY_REGISTRY

        assert "burst_isolation" in STRATEGY_REGISTRY
