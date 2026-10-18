"""Scripted scenarios for the new per-system staleness presets (Phase C)."""

from __future__ import annotations

from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel
from srl.policy.base import Cancel, CancelAction, SimState
from srl.policy.ledger import GroupView
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


def _verl_v1_sim(strategy: str, threshold: int | None, n_versions: int = 8) -> Simulator:
    workload = GroupedWorkload.from_lengths([150], uniform(16, 50, 150, seed=3))
    cfg = SimConfig(batch_size=4, n_versions=n_versions, workload=workload, ingress_window_groups=16)
    policy = VerlV1Policy(
        num_warmup_batches=1.0, max_off_policy_threshold=threshold, max_off_policy_strategy=strategy
    )
    return Simulator(
        rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
        training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
        policy=policy,
        cfg=cfg,
    )


class TestVerlV1OnPublish:
    """Direct state-level proof of the L3 span check.

    An end-to-end run cannot exercise this reliably: `OffPolicySpan.select`
    is oldest-admit_version-first, so a group that is globally the oldest
    ready-and-done group is always consumed by the very take that triggers
    the publish which would otherwise evict it -- it never lingers long
    enough to be "done but unconsumed" when `on_publish` runs. Real verl's
    eviction instead runs on every buffer poll, independent of whether a
    batch is currently assemblable, which this simulator has no equivalent
    tick for. State-level construction sidesteps that gap entirely, the same
    way `tests/test_e2e.py`'s Roll `on_publish` test does.
    """

    def test_drop_evicts_a_finished_group_past_the_threshold(self) -> None:
        policy = VerlV1Policy(max_off_policy_threshold=1, max_off_policy_strategy="drop")
        stale = _group_view(group_id=0, admit_version=0, status="done")
        fresh = _group_view(group_id=1, admit_version=3, status="done")
        running = _group_view(group_id=2, admit_version=0, status="running")
        st = SimState(train_version=3, groups=(stale, fresh, running))

        cancels = policy.on_publish(3, st)

        assert cancels == [Cancel(0, CancelAction.REPLACE)], (
            "only the finished group past the span threshold is evicted; a "
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
    """The full pipeline runs correctly under both strategies; drop-timing is proven above."""

    def test_drop_runs_to_completion(self) -> None:
        result = _verl_v1_sim("drop", threshold=2, n_versions=6).run()
        assert not result.livelocked
        assert result.versions_done == 6

    def test_wait_runs_to_completion_and_never_evicts(self) -> None:
        result = _verl_v1_sim("wait", threshold=2, n_versions=6).run()
        assert not result.livelocked
        assert result.versions_done == 6
        assert result.dropped == []
