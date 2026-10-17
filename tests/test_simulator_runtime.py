"""`SchedulingRuntime`: a stub strategy binds it, reads it, and changes nothing."""

from __future__ import annotations

import pytest

from srl.component.rollout.router.strategy import STRATEGY_REGISTRY
from srl.component.sync.cost import FixedSyncCost
from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel
from srl.forecast import OracleLength
from srl.policy.psrl import PSRLPolicy
from srl.simulator.config import SimConfig
from srl.simulator.simulator import Simulator
from srl.workload.distributions import uniform
from srl.workload.grouped import GroupedWorkload

TRAIN_C = 3.0
SYNC_C = 0.5
WEIGHT_LOAD = 1.0
SEQUENCE_HINT = (100,)
STUB_NAME = "runtime_probe_stub"


@pytest.fixture(autouse=True, scope="module")
def _register_stub():
    """Register the stub for this module only, so the global registry is not left polluted."""
    STRATEGY_REGISTRY[STUB_NAME] = RuntimeProbeStub
    yield
    del STRATEGY_REGISTRY[STUB_NAME]


class RuntimeProbeStub:
    """Picks the least-loaded instance, exactly like `request_num_balance`.

    Also binds a `SchedulingRuntime` and records what it reads on every placement attempt, so a
    test can check the facade without changing what the strategy actually decides.
    """

    name = STUB_NAME

    def __init__(self) -> None:
        self.runtime = None
        self.records: list[tuple] = []

    def bind_runtime(self, runtime) -> None:
        self.runtime = runtime

    def select(self, req, candidates):
        if self.runtime is not None:
            inventory = self.runtime.inventory()
            train = self.runtime.train_state(SEQUENCE_HINT)
            self.records.append((self.runtime.now, inventory, train))
            self._touch_the_rest_of_the_surface()
        return min(range(len(candidates)), key=lambda i: candidates[i].n_active)

    def _touch_the_rest_of_the_surface(self) -> None:
        """Exercise every other read without asserting on its value, as a smoke check."""
        rt = self.runtime
        rt.train_version
        rt.rollout_version
        rt.max_staleness
        rt.batch_size
        rt.group_size
        rt.weight_load
        rt.instances
        rt.inflight
        rt.staged_groups
        rt.admitted_group_ids
        rt.pending
        rt.queued()
        rt.cost_model(1)
        for inst in rt.instances:
            rt.instance_state(inst, OracleLength().remaining)


def _cfg() -> SimConfig:
    workload = GroupedWorkload.from_lengths([150], uniform(80, 40, 120, seed=5))
    return SimConfig(
        batch_size=2,
        max_staleness=1,
        n_versions=5,
        n_instances=2,
        weight_load_time=WEIGHT_LOAD,
        workload=workload,
    )


def _build(strategy_name: str) -> Simulator:
    cost = UnifiedRoofline(F=0.004, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
    training_cost = TrainingCostModel(FixedTrainingLatency(TRAIN_C), FixedTrainingMemory(0))
    cfg = _cfg()
    cfg.routing_strategy = strategy_name
    return Simulator(
        rollout_cost=cost,
        training_cost=training_cost,
        sync_cost=FixedSyncCost(SYNC_C),
        policy=PSRLPolicy(),
        cfg=cfg,
    )


class TestBindRuntimeDoesNotChangeTheRun:
    def test_the_bound_stub_reproduces_the_unbound_baseline(self):
        baseline = _build("request_num_balance").run()
        bound = _build(STUB_NAME).run()

        assert (baseline.sim_time, baseline.versions_done, baseline.livelocked) == (
            bound.sim_time,
            bound.versions_done,
            bound.livelocked,
        )
        assert baseline.train_timestamps == bound.train_timestamps
        assert [r.complete_time for r in baseline.consumed] == [r.complete_time for r in bound.consumed]
        assert len(baseline.dropped) == len(bound.dropped)


class TestInvariants:
    def _run_and_record(self) -> tuple:
        sim = _build(STUB_NAME)
        result = sim.run()
        strategy = sim.router.strategy
        assert strategy.records, "The stub never saw a dispatch to record."
        return sim, result, strategy.records

    def test_now_is_monotone_across_records(self):
        _, _, records = self._run_and_record()
        times = [now for now, _, _ in records]
        assert times == sorted(times)

    def test_snapshot_deadlines_equal_version_plus_staleness(self):
        _, _, records = self._run_and_record()
        checked = 0
        for _, inventory, _ in records:
            for buffer in inventory.buffers:
                for prompt_id, version in buffer.reserved:
                    assert inventory.deadline(prompt_id) == version + inventory.staleness
                    checked += 1
        assert checked > 0

    def test_training_end_is_only_set_while_training_and_matches_the_real_publish(self):
        sim, result, records = self._run_and_record()
        real_publish_times = {end for _, end in result.train_timestamps}
        seen_training = False
        for _, _, train in records:
            if train.training_end is None:
                continue
            seen_training = True
            # training_end plus weight_load is exactly when that step's publish actually happened.
            assert train.training_end + sim.cfg.weight_load_time in real_publish_times
        assert seen_training, "No record observed the simulator mid-training."

    def test_cycle_is_the_fixed_train_plus_sync_cost(self):
        _, _, records = self._run_and_record()
        for _, _, train in records:
            assert train.cycle == pytest.approx(TRAIN_C + SYNC_C)

    def test_weight_load_matches_the_config(self):
        _, _, records = self._run_and_record()
        for _, _, train in records:
            assert train.weight_load == WEIGHT_LOAD
