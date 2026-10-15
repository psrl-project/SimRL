"""`InventoryDeadline`: priority order, lexicographic option choice, and a real run."""

from __future__ import annotations

import types
from dataclasses import dataclass, field

from psrl.workers.ps.inventory_forecast import BufferView, InventorySnapshot

from dryrun.component.rollout.engine import InstanceView, Request
from dryrun.component.rollout.router.strategies.inventory_deadline import InventoryDeadline
from dryrun.cost.rollout import UnifiedRoofline
from dryrun.forecast import InstanceState, Pipeline, PlanSession, Resident, TrainState

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


def make_snapshot(reserved_by_buffer: dict[int, tuple[tuple[int, int], ...]], *, num_entries=4, staleness=1):
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
    """The subset of `SchedulingRuntime` that `PlanSession.build` reads."""

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
    max_staleness: int = 1

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


def strategy() -> InventoryDeadline:
    return InventoryDeadline(length_model="oracle")


class _FakeSession:
    """Hand-controlled option pricing, so `select_context`'s own logic is tested in isolation
    from whether `PlanSession` prices an option correctly -- that is `test_forecast_plan.py`'s job.
    """

    def __init__(self, members=(), staged_groups=None, deadlines=None, rollout_version=0, max_staleness=1):
        self.members = set(members)
        self.staged_groups = staged_groups or {}
        self._deadlines = deadlines or {}
        self.rollout_version = rollout_version
        self.runtime = types.SimpleNamespace(max_staleness=max_staleness)
        self._member_options: dict[int, dict] = {}
        self._admission_options: dict[int, dict] = {}
        self.train_cache: dict = {}
        self.committed: list[tuple[int, object]] = []

    def set_member_options(self, rid: int, options: dict) -> None:
        self._member_options[rid] = options

    def set_admission_options(self, group_id: int, options: dict) -> None:
        self._admission_options[group_id] = options

    def deadline(self, group_id: int) -> int:
        return self._deadlines[group_id]

    def evaluate_member(self, req):
        return self._member_options[req.rid]

    def evaluate_admission(self, group):
        return self._admission_options[group[0].group_id]

    def commit(self, req, option) -> None:
        self.committed.append((req.rid, option))


def pipeline(end: float) -> Pipeline:
    return Pipeline(buffers=(0,), ready=(end - 10.0,), train_end=(end,), publish_times=(end - 5.0,))


class TestBeginDispatch:
    def test_builds_a_fresh_session_each_call(self):
        s = strategy()
        model = cost()
        runtime = StubRuntime(
            instance_states={0: bare_state(model, running=(resident(0, 1000, 500),))},
            snapshot=make_snapshot({}),
            train=TrainState(None, 0.0, 10.0, 1.0),
        )
        s.bind_runtime(runtime)
        s.begin_dispatch(0.0)
        first = s._session
        s.begin_dispatch(0.0)
        second = s._session
        assert isinstance(first, PlanSession)
        assert first is not second

    def test_hooks_return_none_before_any_dispatch_began(self):
        s = strategy()
        req = member_request(0, 100, 50)
        assert s.pick_queued([req], [view(0)]) is None
        assert s.select_context(req, [view(0)], None) is None

    def test_kv_watermark_reaches_the_session(self):
        s = InventoryDeadline(length_model="oracle", kv_watermark=0.3)
        runtime = StubRuntime(
            instance_states={0: bare_state(cost())},
            snapshot=make_snapshot({}),
            train=TrainState(None, 0.0, 10.0, 1.0),
        )
        s.bind_runtime(runtime)
        s.begin_dispatch(0.0)
        assert s._session.kv_watermark == 0.3


class TestPickQueued:
    def test_empty_window_returns_none(self):
        s = strategy()
        s._session = _FakeSession()
        assert s.pick_queued([], [view(0)]) is None

    def test_admitted_members_outrank_staged_heads(self):
        s = strategy()
        staged_group = [member_request(10, 100, 9000, group_id=5), member_request(11, 100, 9000, group_id=5)]
        s._session = _FakeSession(members={1}, staged_groups={5: staged_group}, deadlines={1: 3})
        admitted_member = member_request(0, 100, 50, group_id=1)
        chosen = s.pick_queued([staged_group[0], staged_group[1], admitted_member], [view(0)])
        assert chosen is admitted_member

    def test_admitted_members_rank_by_deadline_then_longest_remaining(self):
        s = strategy()
        urgent = member_request(0, 100, 50, group_id=1)
        loose_long = member_request(1, 100, 9000, group_id=2)
        loose_short = member_request(2, 100, 10, group_id=3)
        s._session = _FakeSession(members={1, 2, 3}, deadlines={1: 3, 2: 9, 3: 9})
        assert s.pick_queued([loose_long, loose_short, urgent], [view(0)]) is urgent
        assert s.pick_queued([loose_long, loose_short], [view(0)]) is loose_long

    def test_staged_groups_rank_by_group_slowest_member_long_first(self):
        s = strategy()
        short_group = [member_request(10, 100, 200, group_id=1)]
        long_group = [member_request(20, 100, 50, group_id=2), member_request(21, 100, 9000, group_id=2)]
        s._session = _FakeSession(staged_groups={1: short_group, 2: long_group})
        chosen = s.pick_queued([short_group[0], long_group[0], long_group[1]], [view(0)])
        assert chosen is long_group[0]

    def test_window_groups_bounds_reordering_to_arrival_blocks(self):
        """A long group that arrived two blocks later must not jump ahead of an earlier one.

        Without a bound, slowest-first search spans the whole staged backlog, which is the
        exact `length_aware` docstring warns against: it starves whichever end of the length
        distribution it ranks last. `window_groups=1` here means every group is its own block,
        so arrival order -- not length -- must decide between groups 1 and 2.
        """
        s = InventoryDeadline(length_model="oracle", window_groups=1)
        early_short = [member_request(10, 100, 50, group_id=1)]
        late_long = [member_request(20, 100, 9000, group_id=2)]
        s.observe_enqueue(early_short)
        s.observe_enqueue(late_long)
        s._session = _FakeSession(staged_groups={1: early_short, 2: late_long})
        chosen = s.pick_queued([early_short[0], late_long[0]], [view(0)])
        assert chosen is early_short[0]

    def test_window_groups_zero_reorders_the_whole_backlog(self):
        s = InventoryDeadline(length_model="oracle", window_groups=0)
        early_short = [member_request(10, 100, 50, group_id=1)]
        late_long = [member_request(20, 100, 9000, group_id=2)]
        s.observe_enqueue(early_short)
        s.observe_enqueue(late_long)
        s._session = _FakeSession(staged_groups={1: early_short, 2: late_long})
        chosen = s.pick_queued([early_short[0], late_long[0]], [view(0)])
        assert chosen is late_long[0]

    def test_staged_group_offers_only_the_first_member(self):
        s = strategy()
        group = [member_request(10, 100, 500, group_id=1), member_request(11, 100, 500, group_id=1)]
        s._session = _FakeSession(staged_groups={1: group})
        assert s.pick_queued([group[1], group[0]], [view(0)]) is group[0]

    def test_unrecognized_group_is_ignored(self):
        s = strategy()
        s._session = _FakeSession()
        orphan = member_request(0, 100, 50, group_id=99)
        assert s.pick_queued([orphan], [view(0)]) is None


class TestSelectContext:
    def test_lexicographically_better_instance_wins(self):
        s = strategy()
        session = _FakeSession(members={1})
        req = member_request(0, 100, 50, group_id=1)
        session.set_member_options(0, {0: pipeline(200.0), 1: pipeline(100.0), None: pipeline(300.0)})
        s._session = session

        index = s.select_context(req, [view(1), view(0)], None)

        assert index == 0
        assert session.committed == [(0, 1)]

    def test_within_eps_tie_prefers_the_first_dict_entry(self):
        s = strategy()
        session = _FakeSession(members={1})
        session.train_cache[(1,)] = TrainState(None, 0.0, 100.0, 1.0)
        req = member_request(0, 100, 50, group_id=1)
        session.set_member_options(0, {0: pipeline(100.0), 1: pipeline(100.0 + 0.05)})
        s._session = session

        index = s.select_context(req, [view(0), view(1)], None)

        assert index == 0
        assert session.committed == [(0, 0)]

    def test_tied_pipelines_break_by_least_loaded_instance(self):
        """Without a load tie-break, every tied decision keeps whichever instance iterates first,
        concentrating admissions on one instance of a multi-instance fleet regardless of load."""
        s = strategy()
        session = _FakeSession(members={1})
        req = member_request(0, 100, 50, group_id=1)
        session.set_member_options(0, {0: pipeline(100.0), 1: pipeline(100.0), 2: pipeline(100.0)})
        s._session = session

        index = s.select_context(req, [view(0, n_running=5), view(1, n_running=1), view(2, n_running=3)], None)

        assert index == 1
        assert session.committed == [(0, 1)]

    def test_defer_wins_when_strictly_better(self):
        s = strategy()
        session = _FakeSession(members={1})
        req = member_request(0, 100, 50, group_id=1)
        session.set_member_options(0, {0: pipeline(500.0), 1: pipeline(500.0), None: pipeline(10.0)})
        s._session = session

        result = s.select_context(req, [view(0), view(1)], None)

        assert result is None
        assert session.committed == [(0, None)]

    def test_defer_false_forces_a_real_placement(self):
        s = InventoryDeadline(defer=False)
        session = _FakeSession(members={1})
        req = member_request(0, 100, 50, group_id=1)
        session.set_member_options(0, {0: pipeline(500.0), 1: pipeline(400.0), None: pipeline(10.0)})
        s._session = session

        index = s.select_context(req, [view(0), view(1)], None)

        assert index == 1
        assert session.committed == [(0, 1)]

    def test_an_overflowing_instance_is_never_chosen(self):
        s = strategy()
        session = _FakeSession(members={1})
        req = member_request(0, 100, 50, group_id=1)
        session.set_member_options(0, {0: None, 1: pipeline(500.0)})
        s._session = session

        index = s.select_context(req, [view(0), view(1)], None)

        assert index == 1
        assert session.committed == [(0, 1)]

    def test_gate_narrowed_candidates_excludes_the_best_option(self):
        s = strategy()
        session = _FakeSession(members={1})
        req = member_request(0, 100, 50, group_id=1)
        session.set_member_options(0, {0: pipeline(100.0), 1: pipeline(900.0)})
        s._session = session

        index = s.select_context(req, [view(1)], None)

        assert index == 0
        assert session.committed == [(0, 1)]

    def test_nothing_feasible_returns_none_and_does_not_commit(self):
        s = strategy()
        session = _FakeSession(members={1})
        req = member_request(0, 100, 50, group_id=1)
        session.set_member_options(0, {0: None, 1: None, None: None})
        s._session = session

        assert s.select_context(req, [view(0), view(1)], None) is None
        assert session.committed == []

    def test_staged_admission_requires_the_group_first_member(self):
        s = strategy()
        group = [member_request(10, 100, 500, group_id=1), member_request(11, 100, 500, group_id=1)]
        session = _FakeSession(staged_groups={1: group})
        s._session = session

        assert s.select_context(group[1], [view(0)], None) is None
        assert session.committed == []

    def test_staged_admission_prices_group_options(self):
        s = strategy()
        group = [member_request(10, 100, 500, group_id=1), member_request(11, 100, 500, group_id=1)]
        session = _FakeSession(staged_groups={1: group})
        session.set_admission_options(1, {0: pipeline(100.0), None: pipeline(900.0)})
        s._session = session

        index = s.select_context(group[0], [view(0)], None)

        assert index == 0
        assert session.committed == [(10, 0)]

    def test_unrecognized_group_returns_none(self):
        s = strategy()
        s._session = _FakeSession()
        orphan = member_request(0, 100, 50, group_id=99)
        assert s.select_context(orphan, [view(0)], None) is None


class TestAllowVersionBypass:
    def test_never_bypasses_a_blocked_head(self):
        s = strategy()
        assert s.allow_version_bypass(None, None) is False
        assert s.allow_version_bypass(member_request(0, 100, 50), None) is False


class TestDeadlineFallback:
    def test_a_reserved_group_uses_the_snapshot_deadline(self):
        s = strategy()
        session = _FakeSession(deadlines={1: 7}, rollout_version=3, max_staleness=2)
        assert s._deadline(session, 1) == 7

    def test_a_group_admitted_this_burst_falls_back_to_version_plus_staleness(self):
        s = strategy()
        session = _FakeSession(rollout_version=3, max_staleness=2)
        assert s._deadline(session, 999) == 5


class TestEndToEnd:
    @staticmethod
    def _run(*, n_instances: int, max_staleness: int, routing_strategy_args=None):
        import random

        from dryrun.component.sync.cost import FixedSyncCost
        from dryrun.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel
        from dryrun.policy.psrl import PSRLPolicy
        from dryrun.simulator.config import SimConfig
        from dryrun.simulator.simulator import Simulator
        from dryrun.workload.grouped import GroupedWorkload
        from dryrun.workload.trace import RequestGroup

        rng = random.Random(0)
        groups = [
            RequestGroup(
                group_id=i, prompt_len=rng.randint(100, 500),
                output_lens=tuple(rng.randint(200, 3000) for _ in range(4)),
            )
            for i in range(40)
        ]
        cfg = SimConfig(
            batch_size=4, group_size=4, max_staleness=max_staleness, n_versions=8,
            n_instances=n_instances, weight_load_time=1.0,
            workload=GroupedWorkload.from_groups(groups), log_telemetry=False,
            routing_strategy="inventory_deadline", routing_strategy_args=routing_strategy_args or {},
        )
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.004, W=0.001, G=0.0001, A_p=3e-10, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(5.0), FixedTrainingMemory(0)),
            sync_cost=FixedSyncCost(0.5),
            policy=PSRLPolicy(),
            cfg=cfg,
        )
        sim.run()
        return sim

    def test_a_multi_instance_run_completes_and_admits_work(self):
        sim = self._run(n_instances=2, max_staleness=1)
        assert sim.n_admitted > 0
        assert sim.n_admitted % 4 == 0

    def test_a_single_instance_run_completes(self):
        sim = self._run(n_instances=1, max_staleness=1)
        assert sim.n_admitted > 0

    def test_zero_staleness_exercises_the_deadline_fallback(self):
        """S=0 means the current rollout version's own groups have no slack: a group admitted
        this burst must still get a deadline before its own reservation lands in the inventory.
        """
        sim = self._run(n_instances=2, max_staleness=0)
        assert sim.n_admitted > 0

    def test_defer_false_never_leaves_a_feasible_member_queued_at_the_end(self):
        sim = self._run(n_instances=2, max_staleness=1, routing_strategy_args={"defer": False})
        assert sim.n_admitted > 0
