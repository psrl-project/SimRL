from srl.component.rollout.engine import Request
from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel
from srl.policy.ledger import GroupLedger
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


def _req(rid: int, group_id: int = 0, v: int = 3) -> Request:
    return Request(rid=rid, prompt_len=10, target_len=20, v_traj=v, group_id=group_id)


def test_ledger_records_admit_and_start_versions() -> None:
    ledger = GroupLedger()
    group = [_req(0), _req(1)]
    ledger.begin(group, version=3, now=1.5)
    ledger.complete(0, now=4.0)

    view = ledger.peek(0)
    assert view is not None
    assert view.admit_version == 3
    assert view.start_version == 3
    assert view.status == "partial"
    assert view.complete_time is None


def test_ledger_completes_only_after_the_last_member() -> None:
    ledger = GroupLedger()
    group = [_req(0), _req(1)]
    ledger.begin(group, version=3, now=0.0)
    group[0].add_tokens(4, 5)
    ledger.complete(0, now=1.0)
    assert ledger.peek(0).status == "partial"

    group[1].add_tokens(4, 5)
    ledger.complete(1, now=2.0)
    assert ledger.peek(0).status == "done"
    assert ledger.peek(0).complete_time == 2.0
    assert ledger.peek(0).start_version == 3


def test_ledger_drop_removes_the_group() -> None:
    ledger = GroupLedger()
    ledger.begin([_req(0)], version=0, now=0.0)
    ledger.drop(0)
    assert ledger.peek(0) is None


def test_simulator_exposes_group_views() -> None:
    from srl.policy.psrl import PSRLPolicy

    seen: list[tuple] = []

    class Probe(PSRLPolicy):
        def peek_batch(self, st):
            if st.groups:
                seen.append((st.groups[0].group_id, st.groups[0].admit_version))
            return super().peek_batch(st)

    _sim(Probe(), batch_size=2, n_versions=3).run()
    assert seen, "the simulator must publish group views"
