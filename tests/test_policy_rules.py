"""Unit tests for the shared L1/L2/L3 rule primitives in `srl.policy.rules`."""

from __future__ import annotations

from srl.policy.areal import ArealPolicy
from srl.policy.base import Cancel, CancelAction, PolicyContext, SimState
from srl.policy.ledger import GroupView
from srl.policy.roll import RollPolicy
from srl.policy.rules import CancelBelowVersion, FIFO, LegacyOverage, ProgressQuota, WatermarkGate
from srl.component.rollout.engine import Request


def _group_view(
    group_id: int,
    admit_version: int,
    start_version: int | None = None,
    status: str = "running",
) -> GroupView:
    return GroupView(
        group_id=group_id,
        admit_version=admit_version,
        start_version=admit_version if start_version is None else start_version,
        status=status,
        admit_time=0.0,
        complete_time=None,
        members=(group_id,),
    )


class TestProgressQuota:
    def test_matches_the_areal_formula_by_hand(self) -> None:
        rule = ProgressQuota(ofp=1)
        assert rule.quota(PolicyContext(state=SimState(train_version=0, batch_size=4))) == 8
        ctx = PolicyContext(state=SimState(train_version=0, batch_size=4), accepted=3, running=5)
        assert rule.quota(ctx) == 0
        assert rule.quota(PolicyContext(state=SimState(train_version=2, batch_size=4))) == 16

    def test_agrees_with_areal_policy(self) -> None:
        policy = ArealPolicy()
        st = SimState(train_version=0, batch_size=4, group_size=1, max_staleness=1)
        rule_quota = ProgressQuota(ofp=1).quota(PolicyContext(state=st, accepted=0, running=0))
        assert policy.admit_quota(st) == max(0, rule_quota) * st.group_size


class TestFIFO:
    def _group(self, group_id: int, dispatch_time: float) -> list[Request]:
        req = Request(rid=group_id, prompt_len=1, target_len=1, v_traj=0, group_id=group_id)
        req.dispatch_time = dispatch_time
        return [req]

    def test_returns_none_below_batch_size(self) -> None:
        groups = [self._group(0, 1.0)]
        assert FIFO().select(groups, batch_size=2) is None

    def test_orders_by_dispatch_time_then_group_id(self) -> None:
        groups = [self._group(2, 5.0), self._group(0, 1.0), self._group(1, 1.0)]
        selected = FIFO().select(groups, batch_size=2)
        assert [req.group_id for req in selected] == [0, 1]


class TestWatermarkGate:
    def test_denies_when_remaining_fills_the_batch(self) -> None:
        rule = WatermarkGate(over_sampling_ratio=1.0)
        assert rule.quota(remaining=4, batch_size=4) == 0

    def test_room_is_the_oversampled_batch(self) -> None:
        rule = WatermarkGate(over_sampling_ratio=1.5)
        assert rule.quota(remaining=0, batch_size=4) == 6

    def test_room_is_at_least_one(self) -> None:
        rule = WatermarkGate(over_sampling_ratio=0.1)
        assert rule.quota(remaining=0, batch_size=4) == 1


class TestLegacyOverage:
    def test_matches_verl_window_budget(self) -> None:
        overage = LegacyOverage(threshold=2.0, require_batches=1)
        assert overage.admit_quota(batch_size=4) == 12
        assert overage.max_queue_size == 12

    def test_drops_on_overflow_and_counts_it(self) -> None:
        overage = LegacyOverage(threshold=0.0, require_batches=1)
        overage.admit_quota(batch_size=1)
        assert overage.max_queue_size == 1
        kept = overage.on_group_complete([Request(rid=0, prompt_len=1, target_len=1, v_traj=0)], batch_size=1)
        assert kept is True
        dropped = overage.on_group_complete([Request(rid=1, prompt_len=1, target_len=1, v_traj=0)], batch_size=1)
        assert dropped is False
        assert overage.dropped == 1


class TestCancelBelowVersion:
    def test_cancels_groups_dispatched_before_the_horizon(self) -> None:
        old = _group_view(group_id=0, admit_version=0)
        recent = _group_view(group_id=1, admit_version=3)
        st = SimState(groups=(old, recent))
        rule = CancelBelowVersion(horizon=2, action=CancelAction.DROP)
        cancels = rule.on_publish(PolicyContext(state=st), new_version=4)
        assert cancels == [Cancel(0, CancelAction.DROP)]

    def test_uses_admit_version_not_start_version(self) -> None:
        """A group idle through a version bump starts generating later than it was admitted."""
        late_starter = _group_view(group_id=0, admit_version=0, start_version=4)
        st = SimState(groups=(late_starter,))
        rule = CancelBelowVersion(horizon=2, action=CancelAction.DROP)
        cancels = rule.on_publish(PolicyContext(state=st), new_version=4)
        assert cancels == [Cancel(0, CancelAction.DROP)], (
            "admit_version=0 is below the floor even though start_version=4 is not."
        )


class TestRollEquivalence:
    """`RollPolicy.on_publish` keeps its own cohort-step loop; this proves the generic
    `CancelBelowVersion` rule would select the identical group-id set for the same scenario.
    """

    def test_cancel_below_version_matches_roll_on_publish(self) -> None:
        policy = RollPolicy()
        st = SimState(batch_size=3, group_size=1, max_staleness=1, train_version=0)
        policy.admit_quota(st)
        groups = {
            gid: [Request(rid=gid, prompt_len=1, target_len=1, v_traj=version, group_id=gid)]
            for gid, version in ((0, 0), (1, 0), (2, 2))
        }
        for group in groups.values():
            policy.on_admit(group, st)
        policy.on_group_complete(groups[1], st)

        new_version = 3
        roll_cancels = policy.on_publish(new_version, SimState(batch_size=3, max_staleness=1, train_version=new_version))
        roll_ids = {c.group_id for c in roll_cancels}

        ledger_state = SimState(
            groups=(
                _group_view(0, admit_version=0, status="running"),
                _group_view(1, admit_version=0, status="done"),
                _group_view(2, admit_version=2, status="running"),
            )
        )
        rule_cancels = CancelBelowVersion(horizon=1, action=CancelAction.REPLACE).on_publish(
            PolicyContext(state=ledger_state), new_version
        )
        rule_ids = {c.group_id for c in rule_cancels}

        assert roll_ids == rule_ids == {0, 1}
