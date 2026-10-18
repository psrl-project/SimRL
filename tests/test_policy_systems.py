"""Scripted scenarios for the new per-system staleness presets (Phase C)."""

from __future__ import annotations

from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel
from srl.policy.base import Cancel, CancelAction, PolicyContext, SimState
from srl.policy.ledger import GroupView
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
