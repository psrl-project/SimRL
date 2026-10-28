"""Scripted scenarios for the new per-system staleness presets (Phase C)."""

from __future__ import annotations

import pytest

from srl.component.rollout.engine import Request
from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel
from srl.policy.base import Cancel, CancelAction, CompleteAction, PolicyContext, SimState, StalenessPolicy
from srl.policy.dressage import DressagePolicy
from srl.policy.ledger import GroupView
from srl.policy.nemo_rl import NeMoRLPolicy
from srl.policy.openrlhf import OpenRLHFPolicy
from srl.policy.prime_rl import PrimeRLPolicy
from srl.policy.rules import FIFO, WarnOnly
from srl.policy.skyrl import SkyRLPolicy
from srl.policy.trl import TRLPolicy
from srl.policy.verl_v1 import VerlV1Policy
from srl.simulator import SimConfig, Simulator
from srl.workload.distributions import uniform
from srl.workload.grouped import GroupedWorkload


def _group_view(group_id: int, admit_version: int, status: str = "done") -> GroupView:
    return GroupView(
        group_id=group_id,
        admit_version=admit_version,
        start_version=admit_version,
        status=status,
        admit_time=0.0,
        complete_time=0.0,
        members=(group_id,),
    )


class _EvictOnSightPolicy(StalenessPolicy):
    """Minimal policy proving `on_batch_attempt`'s `Cancel` is applied before
    `peek_batch`/`take_batch` runs: admits freely, never filters on its own
    in `peek_batch`, but evicts `self.evict_group_id` (with REPLACE, so the
    run can still finish) the first time it sees it sitting in `ready_groups`."""

    def __init__(self, evict_group_id: int) -> None:
        self.evict_group_id = evict_group_id
        self.evicted = False

    def admit_quota(self, st: SimState) -> int:
        return 100

    def on_batch_attempt(self, st: SimState) -> list[Cancel]:
        if not self.evicted and any(g[0].group_id == self.evict_group_id for g in st.ready_groups):
            self.evicted = True
            return [Cancel(self.evict_group_id, CancelAction.REPLACE)]
        return []

    def peek_batch(self, st: SimState) -> list[Request] | None:
        return FIFO().select(st.ready_groups, st.batch_size)


class TestSimulatorBatchAttemptHook:
    """`_next_batch` must apply `on_batch_attempt`'s cancels before calling
    `peek_batch`/`take_batch`, so a group it evicts is never consumed."""

    def test_a_group_evicted_on_sight_is_never_consumed(self) -> None:
        workload = GroupedWorkload.from_lengths([150], uniform(16, 50, 150, seed=7))
        cfg = SimConfig(
            batch_size=1,
            n_versions=4,
            workload=workload,
            ingress_window_groups=8,
            ingress_lifetime_cap=True,
        )
        policy = _EvictOnSightPolicy(evict_group_id=0)
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(0.1), FixedTrainingMemory(0)),
            policy=policy,
            cfg=cfg,
        )
        result = sim.run()

        assert not result.livelocked
        assert policy.evicted, "on_batch_attempt must have seen group 0 and evicted it."
        assert 0 not in {req.group_id for req in result.consumed}, (
            "a group on_batch_attempt evicts must never reach consumption."
        )
        assert any(req.group_id == 0 for req in result.dropped), (
            "the evicted group must show up as dropped, not silently vanish."
        )

    def test_on_batch_attempt_only_runs_at_the_consume_point(self) -> None:
        """`_run_engine`'s `stop_on_batch` check calls `_next_batch(consume=False)`
        from inside its own event loop, where another instance can still
        have an uncommitted, speculative event queued. A policy's cancel
        there would reach `_apply_cancels`, which dispatches through the
        router -- and the router asserts no instance has a pending
        speculative event when it does (confirmed live: this exact path
        raised `AssertionError: Router dispatch ran inside a speculative
        engine event` during this refactor's own real-regime sweep, with
        multiple instances and leftover ingress credit). Only the
        `consume=True` point in `run()`'s own loop, always reached at a
        fully-committed point, may run `on_batch_attempt`."""
        workload = GroupedWorkload.from_lengths([150], uniform(16, 50, 50, seed=7))
        cfg = SimConfig(batch_size=1, n_versions=4, workload=workload, ingress_window_groups=8)

        class _CountingPolicy(StalenessPolicy):
            def __init__(self) -> None:
                self.calls = 0

            def admit_quota(self, st: SimState) -> int:
                return 0

            def on_batch_attempt(self, st: SimState) -> list[Cancel]:
                self.calls += 1
                return []

            def peek_batch(self, st: SimState) -> list[Request] | None:
                return None

        policy = _CountingPolicy()
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(0.1), FixedTrainingMemory(0)),
            policy=policy,
            cfg=cfg,
        )

        sim._next_batch(consume=False)
        assert policy.calls == 0, "on_batch_attempt must not run during the speculative peek."

        sim._next_batch(consume=True)
        assert policy.calls == 1, "on_batch_attempt must run at the safe, committed consume point."


def _verl_v1_sim(
    strategy: str,
    threshold: int | None,
    n_versions: int = 8,
    warmup: float = 1.0,
    policy_cls: type[VerlV1Policy] = VerlV1Policy,
) -> Simulator:
    workload = GroupedWorkload.from_lengths([150], uniform(16, 50, 150, seed=3))
    cfg = SimConfig(
        batch_size=4, n_versions=n_versions, workload=workload, ingress_window_groups=16,
        staleness_horizon=threshold,
    )
    policy = policy_cls(
        num_warmup_batches=warmup, max_off_policy_threshold=threshold, max_off_policy_strategy=strategy
    )
    return Simulator(
        rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
        training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
        policy=policy,
        cfg=cfg,
    )


class _CountingPolicy(VerlV1Policy):
    """Counts how many times `can_sample` denies, to prove the gate fires in situ."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocked = 0

    def can_sample(self, st: SimState) -> bool:
        allowed = super().can_sample(st)
        if not allowed:
            self.blocked += 1
        return allowed


class TestVerlV1BatchAttempt:
    """State-level proof of the L3 span check, now live at every
    batch-assembly attempt via `on_batch_attempt` rather than a
    publish-triggered sweep -- isolated from admission timing."""

    def test_drop_evicts_a_finished_group_past_the_threshold(self) -> None:
        policy = VerlV1Policy(max_off_policy_threshold=1, max_off_policy_strategy="drop")
        stale = _group_view(group_id=0, admit_version=0, status="done")
        fresh = _group_view(group_id=1, admit_version=3, status="done")
        running = _group_view(group_id=2, admit_version=0, status="running")
        st = SimState(train_version=3, groups=(stale, fresh, running))

        cancels = policy.on_batch_attempt(st)

        assert cancels == [Cancel(0, CancelAction.REPLACE)], (
            "only the finished group past the span threshold is evicted. A "
            "running group at the same age is immune under drop."
        )

    def test_drop_ignores_running_groups_even_when_stale(self) -> None:
        policy = VerlV1Policy(max_off_policy_threshold=1, max_off_policy_strategy="drop")
        running_but_ancient = _group_view(group_id=5, admit_version=0, status="running")
        st = SimState(train_version=10, groups=(running_but_ancient,))

        assert policy.on_batch_attempt(st) == [], "drop never cancels in-flight generation."

    def test_wait_blocks_while_any_running_group_is_stale(self) -> None:
        policy = VerlV1Policy(max_off_policy_threshold=1, max_off_policy_strategy="wait")
        running_and_stale = _group_view(group_id=0, admit_version=0, status="running")
        st = SimState(train_version=1, groups=(running_and_stale,))

        assert policy.can_sample(st) is False

    def test_wait_ignores_done_groups_and_never_evicts(self) -> None:
        policy = VerlV1Policy(max_off_policy_threshold=1, max_off_policy_strategy="wait")
        done_and_stale = _group_view(group_id=0, admit_version=0, status="done")
        st = SimState(train_version=10, groups=(done_and_stale,))

        assert policy.can_sample(st) is True
        assert policy.on_batch_attempt(st) == []

    def test_a_group_that_finishes_already_stale_is_evicted_before_the_next_attempt(self) -> None:
        """Replaces the old two-trigger-point race test: with a single live
        check, a straggler that finishes already stale is evicted the very
        next time a batch is attempted -- no publish-vs-consumption gap
        remains to script (matches RESULTS.md's old threshold=1-but-maxS=3-4
        finding, now impossible by construction)."""
        policy = VerlV1Policy(max_off_policy_threshold=1, max_off_policy_strategy="drop")
        straggler = _group_view(group_id=1, admit_version=0, status="done")
        st = SimState(train_version=1, groups=(straggler,))  # span = 1-0+1 = 2 > 1

        cancels = policy.on_batch_attempt(st)

        assert cancels == [Cancel(1, CancelAction.REPLACE)], (
            "a group that finished already stale must be evicted the next time "
            "a batch-assembly attempt happens, before it can ever be selected"
        )


class TestVerlV1AdmissionFreesOnDrop:
    """Real verl's drop refill (`ReplayBufferAsync.wait_for_sampleable`'s
    `self.refill_fn` call, which reaches `_add_prompts_to_generate` ->
    `generate_sequences` directly) submits a replacement prompt to the
    rollout engine with no quota check at all. The old `on_batch_attempt`
    returned `OffPolicySpan`'s cancels without touching `_credit`, so a
    replacement group was only staged (via the simulator's own
    `_ingress_group_credits`, fed through `_apply_cancels`) -- it still had
    to wait for `_credit` to grow, which only happens on
    `on_train_version_advance`, once per version. In steady state (no spare
    credit left over from the warmup head start), that meant eviction
    produced no admission until the next version anyway, understating real
    drop's recurring per-step eviction-and-refill (measured via a real
    separate_async run: 32 groups evicted every step, not once) down to a
    one-time transient the simulator's own warmup overshoot happened to
    cover."""

    def test_a_drop_eviction_frees_quota_for_its_own_replacement(self) -> None:
        policy = VerlV1Policy(num_warmup_batches=1.0, max_off_policy_threshold=1, max_off_policy_strategy="drop")
        st = SimState(batch_size=4, group_size=1, train_version=3)
        # Exhaust every group of startup credit (warmup batch + round 1's own share).
        for group_id in range(8):
            policy.on_admit(_req_group(group_id), st)
        assert policy.admit_quota(st) == 0, "startup credit is fully consumed, no spare room left"

        stale = _group_view(group_id=0, admit_version=0, status="done")
        st = SimState(batch_size=4, group_size=1, train_version=3, groups=(stale,))

        cancels = policy.on_batch_attempt(st)

        assert cancels == [Cancel(0, CancelAction.REPLACE)]
        assert policy.admit_quota(st) == 1, (
            "the evicted group's replacement must be admittable immediately, "
            "not stalled until the next on_train_version_advance"
        )


class TestVerlV1Unbounded:
    def test_null_threshold_disables_l3_for_both_strategies(self) -> None:
        for strategy in ("drop", "wait"):
            policy = VerlV1Policy(max_off_policy_threshold=None, max_off_policy_strategy=strategy)
            st = SimState(train_version=5, groups=(_group_view(0, admit_version=0, status="running"),))
            assert policy.can_sample(st) is True
            assert policy.on_batch_attempt(st) == []


class TestVerlV1Warmup:
    """The blog's own worked example: `train_batch_size=4, num_warmup_batches=4`
    -- startup admits `4*4=16` extra prompts, AND round 1 also normally
    admits 4 more, for 20 total before round 1's training even completes
    ("启动时先额外准入 4x4=16 个 prompt，首轮再正常准入 4 个，合计 20 个").
    The old `_credit` initialization only counted the warmup half; the other
    `+batch_size` only arrived via `on_train_version_advance`, which fires on
    a version advance -- i.e. only after round 1 finishes -- so round 1
    itself ran under-provisioned by one batch relative to the real system."""

    def test_round_one_gets_warmup_plus_its_own_normal_admission(self) -> None:
        policy = VerlV1Policy(num_warmup_batches=4.0)
        st = SimState(batch_size=4, group_size=1, train_version=0)

        assert policy.admit_quota(st) == 20, (
            "16 warmup prompts (4 warmup batches x batch_size 4) plus "
            "round 1's own normal admission of 4, matching the real "
            "system's startup example"
        )


class TestVerlV1Integration:
    def test_drop_runs_to_completion(self) -> None:
        result = _verl_v1_sim("drop", threshold=2, n_versions=6).run()
        assert not result.livelocked
        assert result.versions_done == 6

    def test_wait_runs_to_completion_and_never_evicts(self) -> None:
        result = _verl_v1_sim("wait", threshold=2, n_versions=6).run()
        assert not result.livelocked
        assert result.versions_done == 6
        assert result.dropped == []

    def test_drop_evicts_a_cohort_surplus_end_to_end(self) -> None:
        """num_warmup_batches=2.0 admits two batches worth at once.

        The fastest `batch_size` of them form the first publish, leaving the
        slower same-cohort members behind as a done, unconsumed surplus that
        the very next publish finds stale.
        """
        result = _verl_v1_sim("drop", threshold=1, n_versions=12, warmup=2.0).run()
        assert not result.livelocked
        assert result.dropped, "a cohort surplus must be evicted end to end under drop."
        assert result.staleness_report().strict is True, "eviction is what keeps the bound strict here."

    def test_wait_blocks_sampling_in_situ(self) -> None:
        workload = GroupedWorkload.from_lengths([150], [2000, 50, 50, 50])
        cfg = SimConfig(batch_size=4, n_versions=3, workload=workload, ingress_window_groups=8)
        policy = _CountingPolicy(max_off_policy_threshold=1, max_off_policy_strategy="wait")
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
            policy=policy,
            cfg=cfg,
        )
        result = sim.run()
        assert not result.livelocked
        assert policy.blocked > 0, "the straggler must hold can_sample closed for a while."
        assert result.dropped == []


class TestSkyRLWarnOnly:
    """State-level proof that WarnOnly counts without ever producing a Cancel."""

    def test_counts_a_violation_past_the_threshold(self) -> None:
        rule = WarnOnly(threshold=1)
        stale = GroupView(0, admit_version=0, start_version=0, status="done", admit_time=0.0, complete_time=0.0, members=(0,))
        fresh = GroupView(1, admit_version=3, start_version=3, status="done", admit_time=0.0, complete_time=0.0, members=(1,))
        st = SimState(train_version=3, groups=(stale, fresh))

        violations = rule.count_violations(PolicyContext(state=st), {0, 1})

        assert violations == 1, "only the group past the threshold is counted."

    def test_zero_threshold_is_fully_synchronous_not_rejected(self) -> None:
        policy = SkyRLPolicy(max_staleness_steps=0)
        assert policy.admit_quota(SimState(batch_size=4, train_version=0)) == 4 * 1

    def test_skyrl_never_cancels(self) -> None:
        policy = SkyRLPolicy(max_staleness_steps=1)
        ancient = GroupView(0, admit_version=0, start_version=0, status="done", admit_time=0.0, complete_time=0.0, members=(0,))
        st = SimState(train_version=10, groups=(ancient,))

        assert policy.on_publish(10, st) == [], "SkyRL has no L2 action, staleness only warns."


class TestSkyRLIntegration:
    def test_a_straggler_is_consumed_and_counted_as_a_violation(self) -> None:
        workload = GroupedWorkload.from_lengths([150], [2000, 50, 50, 50, 50, 50, 50, 50])
        cfg = SimConfig(batch_size=4, n_versions=10, workload=workload, ingress_window_groups=16, staleness_horizon=1)
        policy = SkyRLPolicy(max_staleness_steps=1)
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
            policy=policy,
            cfg=cfg,
        )
        result = sim.run()

        assert not result.livelocked
        assert result.dropped == [], "SkyRL never drops for staleness."
        assert 0 in {r.group_id for r in result.consumed}, "the straggler is still trained."
        assert policy.staleness_violations > 0, "the straggler must be flagged, not silently ignored."
        assert result.staleness_report().strict is False, "warn-only never enforces the declared horizon."


class _PeakTrackingPolicy(OpenRLHFPolicy):
    """Records the highest `_lumps_in_flight` ever reached, to prove the permit holds."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.peak_lumps_in_flight = 0

    def on_admit(self, group, st: SimState) -> None:
        super().on_admit(group, st)
        self.peak_lumps_in_flight = max(self.peak_lumps_in_flight, self._lumps_in_flight)


def _req_group(group_id: int) -> list[Request]:
    return [Request(rid=group_id, prompt_len=1, target_len=1, v_traj=0, group_id=group_id)]


class _OverlapObservingPolicy(OpenRLHFPolicy):
    """Counts publishes that found genuinely in-flight work, to prove overlap occurred."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.overlapping_publishes = 0

    def on_train_version_advance(self, train_version: int, st: SimState) -> None:
        super().on_train_version_advance(train_version, st)
        if st.inflight:
            self.overlapping_publishes += 1


class TestOpenRLHFPermitPool:
    def test_quota_closes_once_a_lump_is_fully_admitted(self) -> None:
        policy = OpenRLHFPolicy(async_queue_size=1)
        st = SimState(batch_size=4, group_size=1)
        group = _req_group(0)

        assert policy.admit_quota(st) == 4
        for _ in range(4):
            policy.on_admit(group, st)
        assert policy.admit_quota(st) == 0

    def test_block_releases_only_after_the_publish_that_follows(self) -> None:
        policy = OpenRLHFPolicy(async_queue_size=1, partial_rollout_enable=False)
        ready = [_req_group(gid) for gid in range(4)]
        st = SimState(batch_size=4, group_size=1, ready_groups=ready)
        for _ in range(4):
            policy.on_admit(ready[0], st)

        assert policy.take_batch(st) is not None
        assert policy.admit_quota(st) == 0, "block holds the permit through training."

        policy.on_train_version_advance(1, st)
        assert policy.admit_quota(st) == 4, "the permit releases once the publish completes."

    def test_in_place_releases_at_consumption_not_publish(self) -> None:
        policy = OpenRLHFPolicy(async_queue_size=1, partial_rollout_enable=True)
        ready = [_req_group(gid) for gid in range(4)]
        st = SimState(batch_size=4, group_size=1, ready_groups=ready)
        for _ in range(4):
            policy.on_admit(ready[0], st)

        assert policy.take_batch(st) is not None
        assert policy.admit_quota(st) == 4, "partial rollout releases the permit immediately."


class TestOpenRLHFIntegration:
    def test_at_most_one_lump_is_ever_in_flight(self) -> None:
        workload = GroupedWorkload.from_lengths([150], uniform(32, 50, 150, seed=5))
        cfg = SimConfig(batch_size=4, n_versions=8, workload=workload, ingress_window_groups=8)
        policy = _PeakTrackingPolicy(async_queue_size=1)
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
            policy=policy,
            cfg=cfg,
        )
        result = sim.run()

        assert not result.livelocked
        assert result.versions_done == 8
        assert policy.peak_lumps_in_flight <= 1

    def test_partial_rollout_runs_to_completion(self) -> None:
        workload = GroupedWorkload.from_lengths([150], uniform(32, 50, 150, seed=5))
        cfg = SimConfig(batch_size=4, n_versions=8, workload=workload, ingress_window_groups=8)
        policy = OpenRLHFPolicy(async_queue_size=1, partial_rollout_enable=True)
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
            policy=policy,
            cfg=cfg,
        )
        result = sim.run()

        assert not result.livelocked
        assert result.versions_done == 8

    def test_queue_size_above_one_requires_partial_rollout(self) -> None:
        with pytest.raises(AssertionError):
            OpenRLHFPolicy(async_queue_size=2, partial_rollout_enable=False)

    def test_in_place_preserves_a_lump_overlapping_a_publish(self) -> None:
        """queue_size=2 lets a second lump generate while the first one trains.

        in_place must leave that overlapping lump's progress intact rather
        than aborting and re-prefilling it.
        """
        workload = GroupedWorkload.from_lengths([150], uniform(32, 50, 150, seed=5))
        cfg = SimConfig(batch_size=4, n_versions=8, workload=workload, ingress_window_groups=8)
        policy = _OverlapObservingPolicy(async_queue_size=2, partial_rollout_enable=True)
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
            policy=policy,
            cfg=cfg,
        )
        result = sim.run()

        assert not result.livelocked
        assert policy.overlapping_publishes > 0, "queue_size=2 must let a second lump overlap a publish."
        assert all(r.num_aborts == 0 for r in result.consumed), "in_place must never abort overlapping work."

    def test_a_straggler_is_not_bounded_by_the_permit_pool(self) -> None:
        """The permit pool caps batches in flight, not how long one takes to finish:

        a straggler can hold the single permit for as long as it needs, and
        nothing here checks its generation version against the training
        version once it is finally released.
        """
        workload = GroupedWorkload.from_lengths([150], [10000] + [50] * 11)
        # n_versions well past the brief's 8: a 10000-token straggler needs that many
        # training steps to actually finish and register its staleness.
        cfg = SimConfig(batch_size=2, staleness_horizon=0, n_versions=30, workload=workload, ingress_window_groups=24)
        policy = OpenRLHFPolicy(async_queue_size=1, partial_rollout_enable=True)
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
            policy=policy,
            cfg=cfg,
        )
        result = sim.run()

        assert not result.livelocked
        report = result.staleness_report()
        assert report.max > cfg.staleness_horizon
        assert report.strict is False


def _dressage_group(group_id: int, version: int) -> list[Request]:
    return [Request(rid=group_id, prompt_len=1, target_len=1, v_traj=version, group_id=group_id)]


class TestDressageDistinctVersionWindow:
    """State-level proof that the window keys on observation order, not version value."""

    def test_a_stamped_old_version_is_dropped_once_newer_versions_were_observed_first(self) -> None:
        policy = DressagePolicy(keep_versions=1)
        st = SimState(batch_size=4)
        for group_id, version in [(10, 0), (11, 1), (12, 2)]:
            group = _dressage_group(group_id, version)
            policy.on_admit(group, st)
            assert policy.on_group_complete(group, st) == CompleteAction.KEEP

        straggler = _dressage_group(99, 0)
        policy.on_admit(straggler, st)
        action = policy.on_group_complete(straggler, st)

        assert action == CompleteAction.DROP, "v0 was already observed and is no longer the kept version."
        assert policy.gc_dropped == 1

    def test_zero_keep_versions_never_drops(self) -> None:
        policy = DressagePolicy(keep_versions=0)
        st = SimState(batch_size=4)
        for group_id, version in [(0, 0), (1, 5), (2, 0)]:
            group = _dressage_group(group_id, version)
            policy.on_admit(group, st)
            assert policy.on_group_complete(group, st) == CompleteAction.KEEP
        assert policy.gc_dropped == 0


class TestDressageRetroactiveEviction:
    """A group's own completion-time check only compares its position against
    versions already observed before it finished. The real system also
    re-evaluates the whole pending pool inline with batch assembly
    (`filter_pending`), retroactively evicting an already-accepted group
    that falls outside the window in the meantime ("batch 还在攒的时候，新
    版本到达会驱逐已经收的旧版本") -- not a `peek_batch`-only passive filter,
    but an active eviction with dataset backfill, mirrored here by
    `on_batch_attempt`, called live immediately before every batch-assembly
    attempt."""

    def test_a_group_kept_at_completion_is_evicted_once_newer_versions_arrive(self) -> None:
        policy = DressagePolicy(keep_versions=1)
        st = SimState(batch_size=1)
        straggler = _dressage_group(0, 0)
        policy.on_admit(straggler, st)
        assert policy.on_group_complete(straggler, st) == CompleteAction.KEEP, (
            "version 0 is the only, and therefore newest, version observed so far"
        )

        # A newer version now arrives and completes too.
        fresh = _dressage_group(1, 1)
        policy.on_admit(fresh, st)
        assert policy.on_group_complete(fresh, st) == CompleteAction.KEEP

        # straggler (v0) is now the older of 2 observed versions under
        # keep_versions=1: it has fallen outside the window, even though it
        # was fine at its own completion.
        cancels = policy.on_batch_attempt(SimState(batch_size=1, ready_groups=[straggler, fresh]))

        assert cancels == [Cancel(0, CancelAction.REPLACE)], (
            "straggler must be actively evicted (with backfill), not merely skipped"
        )
        assert policy.gc_dropped == 1

    def test_a_group_still_within_the_window_is_not_evicted(self) -> None:
        policy = DressagePolicy(keep_versions=2)
        st = SimState(batch_size=1)
        straggler = _dressage_group(0, 0)
        policy.on_admit(straggler, st)
        assert policy.on_group_complete(straggler, st) == CompleteAction.KEEP

        fresh = _dressage_group(1, 1)
        policy.on_admit(fresh, st)
        assert policy.on_group_complete(fresh, st) == CompleteAction.KEEP

        cancels = policy.on_batch_attempt(SimState(batch_size=1, ready_groups=[straggler, fresh]))

        assert cancels == [], "both versions are still within a keep_versions=2 window."

    def test_peek_batch_is_a_plain_fifo_selection(self) -> None:
        """Eligibility now lives in `on_batch_attempt`, which actively evicts;
        `peek_batch` itself no longer filters -- a passive filter here would
        leave a stale group simply unselected rather than evicted and
        backfilled, reproducing the undershoot `on_batch_attempt` exists to
        fix."""
        policy = DressagePolicy(keep_versions=1)
        straggler = _dressage_group(0, 0)
        fresh = _dressage_group(1, 1)
        st = SimState(batch_size=2, ready_groups=[straggler, fresh])

        batch = policy.peek_batch(st)

        assert batch is not None
        assert {req.group_id for req in batch} == {0, 1}


class TestDressageWatermarks:
    def test_admit_quota_closes_at_the_active_cap(self) -> None:
        policy = DressagePolicy(max_active_groups=2)
        st = SimState(batch_size=4, group_size=1)
        assert policy.admit_quota(st) == 2
        policy.on_admit(_dressage_group(0, 0), st)
        policy.on_admit(_dressage_group(1, 0), st)
        assert policy.admit_quota(st) == 0

    def test_admit_quota_closes_at_the_queue_watermark(self) -> None:
        policy = DressagePolicy(max_active_groups=10, output_queue_size=10)
        st = SimState(
            batch_size=4, group_size=1, ready_groups=[_dressage_group(i, 0) for i in range(8)]
        )
        assert policy.admit_quota(st) == 0, "high_watermark = 0.8 * 10 = 8, already reached."


class TestDressageVersionSpanCap:
    def test_cancels_a_trajectory_spanning_too_many_versions(self) -> None:
        policy = DressagePolicy(max_partial_rollout_preempts=1)
        req = Request(rid=0, prompt_len=1, target_len=10, v_traj=0, group_id=0)
        req.add_tokens(0, 2)
        req.add_tokens(1, 2)
        st = SimState(inflight=[req])

        cancels = policy.on_publish(2, st)

        assert cancels == [Cancel(0, CancelAction.RETRY_FROM_SCRATCH, "span_cap")]

    def test_disabled_cap_never_cancels(self) -> None:
        policy = DressagePolicy(max_partial_rollout_preempts=None)
        req = Request(rid=0, prompt_len=1, target_len=10, v_traj=0, group_id=0)
        req.add_tokens(0, 2)
        req.add_tokens(1, 2)
        st = SimState(inflight=[req])

        assert policy.on_publish(5, st) == []


class TestDressageIntegration:
    def test_runs_to_completion(self) -> None:
        workload = GroupedWorkload.from_lengths([150], uniform(32, 50, 150, seed=5))
        cfg = SimConfig(
            batch_size=4, n_versions=8, workload=workload, ingress_window_groups=16, staleness_horizon=2
        )
        policy = DressagePolicy(keep_versions=2, max_active_groups=4)
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
            policy=policy,
            cfg=cfg,
        )
        result = sim.run()

        assert not result.livelocked
        assert result.versions_done == 8
        report = result.staleness_report()
        assert report.max_reported <= cfg.staleness_horizon, (
            "keep_versions bounds the last-trainable-token (reported) version, not v_min."
        )

    def test_span_cap_cancellations_release_the_active_slot(self) -> None:
        """A group discarded by VersionSpanCap must free its L1 slot, not leak it.

        A tight preempt cap forces repeated cancellation. If the freed slot
        leaked, admission would shrink every cycle until the run livelocks.
        """
        workload = GroupedWorkload.from_lengths([150], uniform(16, 50, 150, seed=5))
        cfg = SimConfig(batch_size=4, n_versions=8, workload=workload, ingress_window_groups=16)
        policy = DressagePolicy(max_active_groups=2, max_partial_rollout_preempts=0)
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
            policy=policy,
            cfg=cfg,
        )
        result = sim.run()

        assert not result.livelocked
        assert result.versions_done == 8
        assert 0 <= policy._active <= policy.max_active_groups


def _nemo_rl_groups() -> tuple[list[list[Request]], tuple[GroupView, ...]]:
    """Three groups at admit_version 0, 2, 4, with group_id matching arrival order."""
    groups = [_req_group(gid) for gid in (0, 1, 2)]
    views = (
        _group_view(0, admit_version=0),
        _group_view(1, admit_version=2),
        _group_view(2, admit_version=4),
    )
    return groups, views


def _nemo_rl_sim(
    sampler: str, n_versions: int = 8, staleness_horizon: int | None = None, **kwargs
) -> Simulator:
    workload = GroupedWorkload.from_lengths([150], uniform(32, 50, 150, seed=5))
    cfg = SimConfig(
        batch_size=4, n_versions=n_versions, workload=workload, ingress_window_groups=16,
        staleness_horizon=staleness_horizon,
    )
    policy = NeMoRLPolicy(sampler=sampler, max_inflight_prompts=16, max_buffered_rollouts=16, **kwargs)
    return Simulator(
        rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
        training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
        policy=policy,
        cfg=cfg,
    )


class TestNeMoRLWindowed:
    def test_selects_oldest_first_within_the_window(self) -> None:
        policy = NeMoRLPolicy(sampler="windowed", max_staleness_versions=2)
        groups, views = _nemo_rl_groups()
        st = SimState(batch_size=2, train_version=4, ready_groups=groups, groups=views)

        batch = policy.peek_batch(st)

        assert {req.group_id for req in batch} == {1, 2}, "only versions 2 and 4 are within a window of 2."

    def test_freshest_first_takes_the_newest_version_first(self) -> None:
        policy = NeMoRLPolicy(sampler="windowed", max_staleness_versions=2, sample_freshest_first=True)
        groups, views = _nemo_rl_groups()
        st = SimState(batch_size=2, train_version=4, ready_groups=groups, groups=views)

        batch = policy.peek_batch(st)

        assert batch[0].group_id == 2, "version 4 (group 2) must be selected first when freshest-first."

    def test_on_publish_cancels_both_running_and_ready_groups_below_the_window(self) -> None:
        policy = NeMoRLPolicy(sampler="windowed", max_staleness_versions=2)
        running = _group_view(5, admit_version=0, status="running")
        done = _group_view(6, admit_version=0, status="done")
        fresh = _group_view(7, admit_version=4, status="done")
        st = SimState(train_version=4, groups=(running, done, fresh))

        cancels = policy.on_publish(4, st)

        assert {c.group_id for c in cancels} == {5, 6}
        assert policy.gc_dropped == 2

    def test_runs_to_completion(self) -> None:
        result = _nemo_rl_sim("windowed", max_staleness_versions=2).run()
        assert not result.livelocked
        assert result.versions_done == 8

    def test_a_straggler_never_trains_past_the_window(self) -> None:
        workload = GroupedWorkload.from_lengths([150], [3000] + [50] * 11)
        # n_versions well past the brief's 8: a 3000-token straggler needs that many
        # training steps to actually finish and go stale.
        cfg = SimConfig(
            batch_size=2, n_versions=20, workload=workload, ingress_window_groups=24, staleness_horizon=2
        )
        policy = NeMoRLPolicy(sampler="windowed", max_staleness_versions=2, max_inflight_prompts=24, max_buffered_rollouts=24)
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(0.1), FixedTrainingMemory(0)),
            policy=policy,
            cfg=cfg,
        )
        result = sim.run()

        # Not asserting full completion: the windowed sampler's CancelAction.DROP
        # never replaces a group it expires (unlike REPLACE-based GC), and total
        # admission is capped at what n_versions*batch_size will ever consume (see
        # `Simulator._grant_ingress_credits`), so repeatedly dropping this
        # straggler without replacement can exhaust the pool before all 20 steps
        # complete. What this test actually verifies -- the straggler gets
        # excluded rather than silently kept and trained on -- holds regardless
        # of how far the run gets.
        assert policy.gc_dropped > 0, "the straggler must go stale under the window."
        assert result.staleness_report().strict is True


class TestNeMoRLWeightFifo:
    def test_filters_to_the_window_but_never_cancels(self) -> None:
        policy = NeMoRLPolicy(sampler="weight_fifo", max_staleness_versions=2)
        groups, views = _nemo_rl_groups()
        st = SimState(batch_size=2, train_version=4, ready_groups=groups, groups=views)

        batch = policy.peek_batch(st)

        assert {req.group_id for req in batch} == {1, 2}
        assert policy.on_publish(4, st) == [], "weight_fifo has no L2, it only filters at selection."

    def test_runs_to_completion(self) -> None:
        result = _nemo_rl_sim("weight_fifo", max_staleness_versions=2, staleness_horizon=2).run()
        assert not result.livelocked
        assert result.versions_done == 8
        assert result.staleness_report().strict is True, "weight_fifo only ever selects inside the window."


class TestNeMoRLReadyFirst:
    def test_ignores_the_window_entirely(self) -> None:
        policy = NeMoRLPolicy(sampler="ready_first")
        groups, views = _nemo_rl_groups()
        st = SimState(batch_size=2, train_version=4, ready_groups=groups, groups=views)

        batch = policy.peek_batch(st)

        assert {req.group_id for req in batch} == {0, 1}, "no lower bound: takes whichever arrived first."
        assert policy.on_publish(4, st) == []

    def test_runs_to_completion(self) -> None:
        result = _nemo_rl_sim("ready_first").run()
        assert not result.livelocked
        assert result.versions_done == 8

    def test_a_straggler_is_not_bounded_by_fifo_selection(self) -> None:
        """No lower bound means a straggler can sit arbitrarily far behind:

        later, faster cohorts are free to train around it, so nothing here
        checks its generation version against the training version.

        max_staleness_versions=100 is load-bearing, not a copy-paste leftover:
        _gate_window() applies it to every sampler's ADMISSION gate except
        in_order, including ready_first, whose own SELECTION ignores it. At
        the default of 1 that gate itself would bound the straggler, hiding
        the thing this test exists to prove.
        """
        workload = GroupedWorkload.from_lengths([150], [10000] + [50] * 11)
        # n_versions well past the brief's 8: a 10000-token straggler needs that many
        # training steps to actually finish and register its staleness.
        cfg = SimConfig(
            batch_size=2, n_versions=40, workload=workload, ingress_window_groups=24, staleness_horizon=2
        )
        policy = NeMoRLPolicy(
            sampler="ready_first", max_staleness_versions=100, max_inflight_prompts=24, max_buffered_rollouts=24
        )
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(0.1), FixedTrainingMemory(0)),
            policy=policy,
            cfg=cfg,
        )
        result = sim.run()

        assert not result.livelocked
        report = result.staleness_report()
        assert report.max > cfg.staleness_horizon, "the straggler must outrun the declared horizon."
        assert report.strict is False


class TestNeMoRLInOrder:
    def test_takes_only_the_cohort_whose_target_step_matches_train_version(self) -> None:
        policy = NeMoRLPolicy(sampler="in_order", max_lookahead_versions=1)
        admit_state = SimState(batch_size=2, group_size=1, train_version=0)
        cohort_a = [_req_group(0), _req_group(1)]
        cohort_b = [_req_group(2), _req_group(3)]
        for group in cohort_a + cohort_b:
            policy.on_admit(group, admit_state)

        ready = [group for group in cohort_a + cohort_b]
        select_state = SimState(batch_size=2, train_version=0, ready_groups=ready)

        batch = policy.peek_batch(select_state)

        assert {req.group_id for req in batch} == {0, 1}, "only the version-0 cohort is selectable now."

    def test_the_next_cohort_becomes_selectable_once_train_version_catches_up(self) -> None:
        policy = NeMoRLPolicy(sampler="in_order", max_lookahead_versions=1)
        admit_state = SimState(batch_size=2, group_size=1, train_version=0)
        cohort_a = [_req_group(0), _req_group(1)]
        cohort_b = [_req_group(2), _req_group(3)]
        for group in cohort_a + cohort_b:
            policy.on_admit(group, admit_state)

        ready = [group for group in cohort_a + cohort_b]
        select_state = SimState(batch_size=2, train_version=1, ready_groups=ready)

        batch = policy.peek_batch(select_state)

        assert {req.group_id for req in batch} == {2, 3}

    def test_runs_to_completion(self) -> None:
        result = _nemo_rl_sim("in_order", max_lookahead_versions=2, staleness_horizon=2).run()
        assert not result.livelocked
        assert result.versions_done == 8
        assert result.staleness_report().strict is True, "in_order binds every cohort to one target step."


class TestNeMoRLAdmissionGate:
    def test_dispatch_lag_gate_closes_once_the_lookahead_is_exhausted(self) -> None:
        policy = NeMoRLPolicy(sampler="in_order", max_lookahead_versions=1, max_inflight_prompts=100, max_buffered_rollouts=100)
        st = SimState(batch_size=2, group_size=1, train_version=0)
        for group_id in range(2):
            policy.on_admit(_req_group(group_id), st)
        # One full cohort admitted. dispatch_index is now 1, equal to train_version(0) + lookahead(1).
        assert policy.admit_quota(st) == 0


class TestTRLBoundedQueue:
    def test_negative_cap_resolves_from_max_staleness_and_batch_size(self) -> None:
        policy = TRLPolicy(max_staleness=4, max_inflight_tasks=-1)
        st = SimState(batch_size=8, group_size=1, train_version=0)
        assert policy.admit_quota(st) == 32
        assert policy.max_inflight_tasks == 32

    def test_quota_closes_once_the_queue_is_full(self) -> None:
        policy = TRLPolicy(max_staleness=1, max_inflight_tasks=2)
        st = SimState(batch_size=2, group_size=1)
        for group_id in range(2):
            policy.on_admit(_req_group(group_id), st)
        assert policy.admit_quota(st) == 0

    def test_pause_mode_is_in_place(self) -> None:
        assert TRLPolicy().pause_mode(SimState()) == "in_place"


class TestTRLBatchAttempt:
    """Reproduces the brief's scenario: groups admitted at [0, 3, 5], max_staleness=4, v_train=5.
    Checked live, immediately before every batch-assembly attempt, via
    `on_batch_attempt` -- not a periodic publish-triggered sweep."""

    def test_only_the_group_more_than_four_versions_stale_is_dropped(self) -> None:
        policy = TRLPolicy(max_staleness=4, max_inflight_tasks=10)
        groups = tuple(_group_view(v, admit_version=v) for v in (0, 3, 5))
        st = SimState(groups=groups, train_version=5)

        cancels = policy.on_batch_attempt(st)

        assert {c.group_id for c in cancels} == {0}
        assert all(c.action == CancelAction.REPLACE for c in cancels), "TRL refills a dropped slot."
        assert policy.gc_dropped == 1

    def test_a_group_within_budget_is_not_evicted(self) -> None:
        policy = TRLPolicy(max_staleness=2, max_inflight_tasks=10)
        groups = (_group_view(1, admit_version=0),)
        st = SimState(groups=groups, train_version=2)

        assert policy.on_batch_attempt(st) == []
        assert policy.gc_dropped == 0

    def test_only_done_groups_are_evicted_never_in_flight_ones(self) -> None:
        policy = TRLPolicy(max_staleness=1, max_inflight_tasks=10)
        running_but_ancient = _group_view(5, admit_version=0, status="running")
        st = SimState(groups=(running_but_ancient,), train_version=10)

        assert policy.on_batch_attempt(st) == [], (
            "on_batch_attempt must never touch in-flight generation -- TRL has no L2."
        )


class TestTRLGuaranteesTheBoundEndToEnd:
    """With a single check left (`on_batch_attempt`, fired right before every
    batch-assembly attempt instead of a decoupled publish sweep), a straggler
    that finishes already stale is evicted the very next time a batch is
    attempted, before it can ever be selected -- no race window between a
    publish sweep and the next consumption remains to script."""

    def test_a_straggler_that_finishes_already_stale_is_evicted_before_the_next_attempt(self) -> None:
        policy = TRLPolicy(max_staleness=2, max_inflight_tasks=10)
        straggler = _req_group(1)  # v_traj=0
        policy.on_admit(straggler, SimState(batch_size=2, group_size=1, train_version=0))

        # The group finishes at train_version=3: staleness 3 > budget 2, already stale.
        straggler_view = _group_view(1, admit_version=0, status="done")
        st = SimState(groups=(straggler_view,), train_version=3)

        cancels = policy.on_batch_attempt(st)

        assert {c.group_id for c in cancels} == {1}, (
            "a group that finished already stale must be evicted the next time "
            "a batch-assembly attempt happens, before it can ever be selected"
        )
        assert policy.gc_dropped == 1


class TestTRLAdmissionFreesOnCompletion:
    """Real TRL frees an inflight slot the instant generation completes,
    independent of whether training has consumed it yet: "生成完成后就释放
    位置，有空位就准入新任务，不必等训练侧取样" (blog, TRL L1). The old
    `_buffered` bookkeeping instead held the slot until `take_batch` or an
    explicit drop, over-restricting admission relative to the real system."""

    def test_a_kept_completion_frees_its_slot_before_consumption(self) -> None:
        policy = TRLPolicy(max_staleness=4, max_inflight_tasks=1)
        st = SimState(batch_size=2, group_size=1, train_version=0)
        group = _req_group(1)  # v_traj=0
        policy.on_admit(group, st)
        assert policy.admit_quota(st) == 0, "the single slot is occupied while generating"

        action = policy.on_group_complete(group, st)

        assert action == CompleteAction.KEEP
        assert policy.admit_quota(st) == 1, (
            "the slot frees on completion, before this group has been consumed"
        )




class TestPrimeRLAdmission:
    def test_dispatch_lag_gate_closes_once_the_lookahead_is_exhausted(self) -> None:
        policy = PrimeRLPolicy(target_lag=1, max_inflight=100)
        st = SimState(batch_size=2, group_size=1, train_version=0)
        for group_id in range(2):
            policy.on_admit(_req_group(group_id), st)
        # One full cohort admitted. dispatch_index is now 1, equal to train_version(0) + target_lag(1).
        assert policy.admit_quota(st) == 0

    def test_inflight_cap_closes_independent_of_dispatch_lag(self) -> None:
        policy = PrimeRLPolicy(target_lag=100, max_inflight=1)
        st = SimState(batch_size=4, group_size=1, train_version=0)
        policy.on_admit(_req_group(0), st)
        assert policy.admit_quota(st) == 0

    def test_pause_mode_is_in_place(self) -> None:
        assert PrimeRLPolicy().pause_mode(SimState()) == "in_place"

    def test_unknown_action_raises_a_readable_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown action 'bogus'"):
            PrimeRLPolicy(action="bogus")


class TestPrimeRLAdmissionFreesOnCompletion:
    """Real prime-rl frees an inflight slot the instant generation completes,
    independent of whether training has consumed it yet ("有空位才继续准入，
    生成完成后再释放"). `PrimeRLPolicy` never overrode `on_group_complete` at
    all, inheriting the base class's no-op -- so `_inflight_groups` held a
    slot occupied until consumption, over-restricting admission relative to
    the real system, the same bug TRL's `max_inflight_tasks` had."""

    def test_a_kept_completion_frees_its_slot_before_consumption(self) -> None:
        policy = PrimeRLPolicy(max_off_policy_steps=4, max_inflight=1, target_lag=100)
        st = SimState(batch_size=2, group_size=1, train_version=0)
        group = _req_group(1)  # v_traj=0
        policy.on_admit(group, st)
        assert policy.admit_quota(st) == 0, "the single slot is occupied while generating"

        action = policy.on_group_complete(group, st)

        assert action == CompleteAction.KEEP
        assert policy.admit_quota(st) == 1, (
            "the slot frees on completion, before this group has been consumed"
        )


class TestPrimeRLBatchAttempt:
    """The ready-pool sweep (`VersionWindow`) now runs live, immediately
    before every batch-assembly attempt, via `on_batch_attempt` -- not a
    periodic publish-triggered sweep. Matches real prime-rl's
    `TrainSink.take_batch()`, which calls `self._drop_stale()` right before
    assembling the batch, every single attempt."""

    def test_a_group_that_completes_already_stale_is_evicted(self) -> None:
        policy = PrimeRLPolicy(max_off_policy_steps=2, max_inflight=100, target_lag=100)
        stale = _group_view(group_id=1, admit_version=0, status="done")
        st = SimState(train_version=3, groups=(stale,))  # 3-0=3 > 2

        cancels = policy.on_batch_attempt(st)

        assert cancels == [Cancel(1, CancelAction.REPLACE)]
        assert policy.gc_dropped == 1

    def test_a_group_within_budget_is_not_evicted(self) -> None:
        policy = PrimeRLPolicy(max_off_policy_steps=2, max_inflight=100, target_lag=100)
        fresh = _group_view(group_id=1, admit_version=0, status="done")
        st = SimState(train_version=2, groups=(fresh,))  # 2-0=2, not > 2

        assert policy.on_batch_attempt(st) == []
        assert policy.gc_dropped == 0

    def test_running_groups_are_not_touched_here_thats_l2s_job_in_on_publish(self) -> None:
        policy = PrimeRLPolicy(max_off_policy_steps=0, max_inflight=100, target_lag=100)
        running = _group_view(group_id=1, admit_version=0, status="running")
        st = SimState(train_version=10, groups=(running,))

        assert policy.on_batch_attempt(st) == [], (
            "on_batch_attempt only sweeps the ready pool (VersionWindow filters "
            "status=='done'); in-flight staleness is CancelBelowVersion's job in on_publish."
        )


class TestPrimeRLPublishDoesNotDoubleFreeACompletedSlot:
    """A group cancelled by `on_publish`'s L2 sweep (`CancelBelowVersion`, which
    has no status filter of its own) already freed its slot at its own
    `on_group_complete`; `on_publish` must not free it again, or
    `_inflight_groups` drifts negative and opens up admission room it
    shouldn't have. Mirrors `nemo_rl`'s `windowed` sampler, which already
    gets this right by checking each cancelled group's actual status."""

    def test_a_done_groups_slot_is_not_freed_twice(self) -> None:
        policy = PrimeRLPolicy(max_off_policy_steps=1, max_inflight=10, target_lag=100)
        st0 = SimState(batch_size=2, group_size=1, train_version=0)
        group = _req_group(1)  # v_traj=0
        policy.on_admit(group, st0)
        assert policy.admit_quota(st0) == 9

        # Completes within budget at train_version=0: kept, slot freed here.
        action = policy.on_group_complete(group, SimState(batch_size=2, group_size=1, train_version=0))
        assert action == CompleteAction.KEEP

        # Publish v=2: the group is status=="done" and 2-0=2 > threshold(1),
        # so VersionWindow's sweep cancels it -- its slot was already freed.
        done_view = _group_view(1, admit_version=0, status="done")
        cancels = policy.on_publish(2, SimState(train_version=2, groups=(done_view,)))
        assert {c.group_id for c in cancels} == {1}

        # _inflight_groups must not have gone negative from a double free.
        assert policy.admit_quota(SimState(batch_size=2, group_size=1, train_version=2)) == 10


class TestPrimeRLIngressWindow:
    def test_scales_with_target_lag_once_it_exceeds_the_staleness_bound(self) -> None:
        from srl.config.policies.prime_rl import PrimeRLPolicyConfig

        tight = PrimeRLPolicyConfig(max_off_policy_steps=8, max_inflight=1024, target_lag=1)
        loose = PrimeRLPolicyConfig(max_off_policy_steps=8, max_inflight=1024, target_lag=100)

        assert tight.ingress_window_groups(batch_size=8) == 80
        assert loose.ingress_window_groups(batch_size=8) == 808, (
            "a wide dispatch-lag gate admits far more than (max_off_policy_steps+2)*batch_size ahead."
        )

    def test_never_exceeds_max_inflight(self) -> None:
        from srl.config.policies.prime_rl import PrimeRLPolicyConfig

        cfg = PrimeRLPolicyConfig(max_off_policy_steps=8, max_inflight=50, target_lag=100)
        assert cfg.ingress_window_groups(batch_size=8) == 50


class TestPrimeRLPublish:
    """`on_publish` is now L2 only (`CancelBelowVersion`): confirms it still
    correctly cancels a genuinely in-flight stale group after the L3 half
    (`VersionWindow`) moved out to `on_batch_attempt`."""

    def test_a_stale_running_group_is_also_cancelled(self) -> None:
        policy = PrimeRLPolicy(max_off_policy_steps=0, max_inflight=10)
        st = SimState(batch_size=1, group_size=1, train_version=0)
        policy.on_admit(_req_group(0), st)

        stale_running = _group_view(group_id=0, admit_version=0, status="running")
        cancels = policy.on_publish(1, SimState(groups=(stale_running,), train_version=1))

        assert cancels == [Cancel(0, CancelAction.REPLACE)], (
            "pause_mode=in_place leaves this as the only interrupt path; real prime-rl "
            "backfills a cancelled straggler the same way its ready-pool sweep already does."
        )


class _FirstCancelObservingPolicy(PrimeRLPolicy):
    """Records the train_version of the first publish that cancelled anything."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.first_cancel_version: int | None = None

    def on_publish(self, new_version: int, st: SimState) -> list[Cancel]:
        cancels = super().on_publish(new_version, st)
        if cancels and self.first_cancel_version is None:
            self.first_cancel_version = new_version
        return cancels


class TestPrimeRLIntegration:
    def test_a_straggler_is_cancelled_at_v1_and_nothing_stale_is_ever_trained(self) -> None:
        workload = GroupedWorkload.from_lengths([150], [3000] + [50] * 15)
        cfg = SimConfig(
            batch_size=2, n_versions=20, workload=workload, ingress_window_groups=20, staleness_horizon=0
        )
        policy = _FirstCancelObservingPolicy(max_off_policy_steps=0, max_inflight=20, target_lag=100)
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(0.1), FixedTrainingMemory(0)),
            policy=policy,
            cfg=cfg,
        )
        result = sim.run()

        # The runtime cancel now defaults to action="replace" (matching the
        # ready-pool sweep's own refill=True), so repeatedly cancelling this
        # straggler backfills the pool instead of draining it: the run
        # reaches all 20 steps rather than exhausting admission early.
        assert not result.livelocked
        assert result.versions_done == 20
        assert policy.first_cancel_version == 1, "a straggler admitted at v0 is stale the moment v1 publishes."
        assert policy.gc_dropped > 0
        assert all(req.staleness_true() <= 0 for req in result.consumed), (
            "S=0 tolerates zero lag: every trained request's earliest token is at the version it was trained under."
        )
        assert result.staleness_report().strict is True
