import pytest

from srl.component.rollout.engine import Request
from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel
from srl.policy.base import Cancel, CancelAction, CompleteAction, SimState, StalenessPolicy
from srl.policy.ledger import GroupLedger
from srl.policy.psrl import PSRLPolicy
from srl.simulator import SimConfig, Simulator
from srl.workload.distributions import uniform
from srl.workload.grouped import GroupedWorkload


def _cfg(batch_size: int, n_versions: int) -> SimConfig:
    return SimConfig(
        batch_size=batch_size,
        n_versions=n_versions,
        workload=GroupedWorkload.from_lengths([150], uniform(16, 50, 200, seed=5)),
    )


def _sim(policy, batch_size: int = 2, n_versions: int = 4) -> Simulator:
    return Simulator(
        rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
        training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
        policy=policy,
        cfg=_cfg(batch_size, n_versions),
    )


def _req(rid: int, group_id: int = 0, v: int = 3, complete_time: float | None = None) -> Request:
    return Request(
        rid=rid,
        prompt_len=10,
        target_len=20,
        v_traj=v,
        group_id=group_id,
        complete_time=complete_time,
    )


def test_ledger_records_admit_and_start_versions() -> None:
    ledger = GroupLedger()
    group = [_req(0, complete_time=4.0), _req(1)]
    ledger.begin(group, version=3, now=1.5)
    ledger.complete(group[0])

    view = ledger.peek(0)
    assert view is not None
    assert view.admit_version == 3
    assert view.start_version == 3
    assert view.status == "partial"
    assert view.complete_time is None
    assert view.members == (0, 1)


def test_ledger_completes_only_after_the_last_member() -> None:
    ledger = GroupLedger()
    group = [_req(0, complete_time=1.0), _req(1, complete_time=1.75)]
    ledger.begin(group, version=3, now=0.0)
    assert ledger.peek(0).start_version == 3

    group[0].add_tokens(4, 5)
    ledger.complete(group[0])
    assert ledger.peek(0).status == "partial"
    assert ledger.peek(0).start_version == 4

    group[1].add_tokens(4, 5)
    ledger.complete(group[1])
    assert ledger.peek(0).status == "done"
    assert ledger.peek(0).complete_time == 1.75
    assert ledger.peek(0).start_version == 4


def test_ledger_ignores_uncommitted_segments() -> None:
    ledger = GroupLedger()
    group = [_req(0, v=1), _req(1, v=1)]
    ledger.begin(group, version=1, now=0.0)
    assert ledger.peek(0).start_version == 1

    # A speculative event may add segments that a later undo removes.
    group[0].add_tokens(5, 10)
    assert ledger.peek(0).start_version == 1

    group[0].segments.clear()
    group[0].generated = 0
    assert ledger.peek(0).start_version == 1


def test_ledger_does_not_demote_a_completed_group() -> None:
    ledger = GroupLedger()
    req = _req(0, complete_time=2.0)
    ledger.begin([req], version=1, now=0.0)
    ledger.complete(req)
    assert ledger.peek(0).status == "done"
    assert ledger.peek(0).complete_time == 2.0

    ledger.complete(req)
    assert ledger.peek(0).status == "done"
    assert ledger.peek(0).complete_time == 2.0


def test_ledger_drop_removes_the_group() -> None:
    ledger = GroupLedger()
    ledger.begin([_req(0, group_id=1)], version=0, now=0.0)
    ledger.drop(1)
    assert ledger.peek(1) is None


def test_simulator_exposes_group_views() -> None:
    seen: list[tuple] = []

    class Probe(PSRLPolicy):
        def peek_batch(self, st):
            seen.extend(st.groups)
            return super().peek_batch(st)

    sim = _sim(Probe(), batch_size=2, n_versions=3)
    sim.run()
    assert seen, "the simulator must publish group views."
    done = [view for view in seen if view.status == "done"]
    assert done, "at least one completed group must be visible."
    assert all(len(view.members) == sim.cfg.group_size for view in done), (
        "a completed view must list every group member."
    )
    assert all(len(set(view.members)) == len(view.members) for view in seen), "member ids must be unique."
    assert all(view.start_version >= view.admit_version for view in seen), (
        "start_version cannot precede the admit version."
    )


def test_simulator_prunes_consumed_groups() -> None:
    consumed: set[int] = set()
    live_seen: list[tuple[set[int], set[int]]] = []

    class Probe(PSRLPolicy):
        def take_batch(self, st):
            batch = super().take_batch(st)
            if batch is not None:
                consumed.update(req.group_id for req in batch)
            return batch

        def check_invariants(self, st):
            super().check_invariants(st)
            live_seen.append((set(consumed), {view.group_id for view in st.groups}))

    _sim(Probe(), batch_size=2, n_versions=3).run()
    assert consumed, "the run must consume at least one group."
    assert all(not (live & taken) for taken, live in live_seen), "consumed groups must not remain in SimState.groups."


def test_simulator_prunes_dropped_groups() -> None:
    dropped: set[int] = set()

    class DropOnce(PSRLPolicy):
        def on_group_complete(self, group, st):
            if not dropped:
                dropped.add(group[0].group_id)
                return CompleteAction.DROP
            return CompleteAction.KEEP

    sim = _sim(DropOnce(), batch_size=2, n_versions=2)
    sim.run()
    assert dropped, "the probe must drop one group."
    live = {view.group_id for view in sim._state().groups}
    assert not (live & dropped), "dropped groups must not remain in SimState.groups."


class _KeepPolicy(StalenessPolicy):
    """Admit freely and never select a batch."""

    name = "keep"

    def admit_quota(self, st):
        return 10**9

    def peek_batch(self, st):
        return None


def test_discard_of_a_done_group_removes_the_ledger_entry() -> None:
    sim = _sim(_KeepPolicy(), batch_size=1, n_versions=1)
    req = Request(rid=0, prompt_len=10, target_len=20, v_traj=0, complete_time=1.0)
    sim._ingress_group_credits = 0
    sim._ledger.begin([req], version=0, now=0.0)
    sim._ledger.complete(req)
    assert sim._ledger.peek(0).status == "done"

    sim._discard([req])
    assert sim._ledger.peek(0) is None


def test_recollecting_a_committed_completion_keeps_one_entry() -> None:
    sim = _sim(_KeepPolicy(), batch_size=1, n_versions=1)
    req = Request(rid=0, prompt_len=10, target_len=20, v_traj=0, complete_time=1.0)
    sim._ingress_group_credits = 0
    sim._ledger.begin([req], version=0, now=0.0)

    sim._collect([req])
    sim._collect([req])

    views = [view for view in sim._state().groups if view.group_id == 0]
    assert len(views) == 1, "a repeated collection must not duplicate the group."
    assert views[0].status == "done"
    assert views[0].complete_time == 1.0
    assert views[0].members == (0,)


class _CancelAtOne(PSRLPolicy):
    def on_publish(self, new_version, st):
        if new_version == 1:
            return [Cancel(g.group_id, CancelAction.DROP) for g in st.groups]
        return []


def test_on_publish_cancel_drops_inflight_groups() -> None:
    result = _sim(_CancelAtOne(), batch_size=2, n_versions=4).run()
    assert result.dropped, "the publish hook must be able to discard groups."
    assert all(r.status.name == "DROPPED" for r in result.dropped)


def test_default_publish_hook_is_a_no_op() -> None:
    assert PSRLPolicy().on_publish(3, SimState()) == []


def test_cancel_action_values_are_stable() -> None:
    assert {a.value for a in CancelAction} == {"drop", "replace", "retry_from_scratch"}
    assert CompleteAction.KEEP.value == "keep"


class _ReplacingKeepPolicy(_KeepPolicy):
    """Refill every permanently dropped group, the way ROLL does."""

    def replacement_groups_on_drop(self, groups, st):
        return len(groups)


def _live_group(sim: Simulator, group_id: int = 0, version: int = 0) -> list[Request]:
    reqs = [
        Request(rid=group_id * 10 + index, prompt_len=10, target_len=20, v_traj=version, group_id=group_id)
        for index in range(sim.cfg.group_size)
    ]
    sim._staged_groups.pop(group_id, None)
    sim._admitted_group_ids.add(group_id)
    sim.inflight.extend(reqs)
    sim._ledger.begin(reqs, version=version, now=sim.t)
    return reqs


@pytest.mark.parametrize(
    ("action", "credits"),
    [
        (CancelAction.DROP, 0),
        (CancelAction.REPLACE, 1),
        (CancelAction.RETRY_FROM_SCRATCH, 1),
    ],
)
def test_cancel_credit_follows_the_action(action: CancelAction, credits: int) -> None:
    sim = _sim(_ReplacingKeepPolicy(), batch_size=1, n_versions=1)
    reqs = _live_group(sim)
    sim._ingress_group_credits = 0

    sim._apply_cancels([Cancel(0, action)])

    assert all(req.status.name == "DROPPED" for req in reqs)
    assert sim._ledger.peek(0) is None, "a cancelled group must leave no ledger entry."
    assert sim._ingress_group_credits == credits


def test_cancel_of_a_non_live_group_is_a_no_op() -> None:
    sim = _sim(_ReplacingKeepPolicy(), batch_size=1, n_versions=1)
    reqs = _live_group(sim)
    sim._ingress_group_credits = 0
    sim._discard(reqs)
    dropped = len(sim.dropped)

    sim._apply_cancels([Cancel(0, CancelAction.REPLACE), Cancel(99, CancelAction.DROP)])

    assert len(sim.dropped) == dropped, "an already dropped group must not be dropped twice."
    assert sim._ingress_group_credits == 0, "a non-live group must not grant a credit."
