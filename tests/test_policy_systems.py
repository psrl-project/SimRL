"""Scripted scenarios for the new per-system staleness presets (Phase C)."""

from __future__ import annotations

import pytest

from srl.component.rollout.engine import Request
from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel
from srl.policy.base import Cancel, CancelAction, CompleteAction, PolicyContext, SimState
from srl.policy.dressage import DressagePolicy
from srl.policy.ledger import GroupView
from srl.policy.openrlhf import OpenRLHFPolicy
from srl.policy.rules import WarnOnly
from srl.policy.skyrl import SkyRLPolicy
from srl.policy.verl_v1 import VerlV1Policy
from srl.simulator import SimConfig, Simulator
from srl.workload.grouped import GroupedWorkload
from srl.workload.distributions import uniform


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


def _verl_v1_sim(
    strategy: str,
    threshold: int | None,
    n_versions: int = 8,
    warmup: float = 1.0,
    policy_cls: type[VerlV1Policy] = VerlV1Policy,
) -> Simulator:
    workload = GroupedWorkload.from_lengths([150], uniform(16, 50, 150, seed=3))
    cfg = SimConfig(batch_size=4, n_versions=n_versions, workload=workload, ingress_window_groups=16)
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


class TestVerlV1OnPublish:
    """State-level proof of the L3 span check, isolated from admission timing."""

    def test_drop_evicts_a_finished_group_past_the_threshold(self) -> None:
        policy = VerlV1Policy(max_off_policy_threshold=1, max_off_policy_strategy="drop")
        stale = _group_view(group_id=0, admit_version=0, status="done")
        fresh = _group_view(group_id=1, admit_version=3, status="done")
        running = _group_view(group_id=2, admit_version=0, status="running")
        st = SimState(train_version=3, groups=(stale, fresh, running))

        cancels = policy.on_publish(3, st)

        assert cancels == [Cancel(0, CancelAction.REPLACE)], (
            "only the finished group past the span threshold is evicted. A "
            "running group at the same age is immune under drop."
        )

    def test_drop_ignores_running_groups_even_when_stale(self) -> None:
        policy = VerlV1Policy(max_off_policy_threshold=1, max_off_policy_strategy="drop")
        running_but_ancient = _group_view(group_id=5, admit_version=0, status="running")
        st = SimState(train_version=10, groups=(running_but_ancient,))

        assert policy.on_publish(10, st) == [], "drop never cancels in-flight generation."

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
        assert policy.on_publish(10, st) == []


class TestVerlV1Unbounded:
    def test_null_threshold_disables_l3_for_both_strategies(self) -> None:
        for strategy in ("drop", "wait"):
            policy = VerlV1Policy(max_off_policy_threshold=None, max_off_policy_strategy=strategy)
            st = SimState(train_version=5, groups=(_group_view(0, admit_version=0, status="running"),))
            assert policy.can_sample(st) is True
            assert policy.on_publish(5, st) == []


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
        cfg = SimConfig(batch_size=4, n_versions=10, workload=workload, ingress_window_groups=16)
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

        assert cancels == [Cancel(0, CancelAction.RETRY_FROM_SCRATCH)]

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
        cfg = SimConfig(batch_size=4, n_versions=8, workload=workload, ingress_window_groups=16)
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
        assert policy._active <= policy.max_active_groups
