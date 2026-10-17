"""`PipelineForecast`, `compare`, and `group_etas`: hand-checked cases and a property test."""

from __future__ import annotations

import math

import pytest
from psrl.workers.ps.inventory_forecast import BufferView, InventorySnapshot
from psrl.workers.ps.staleness_controller import BufferStatus

from srl.component.sync.cost import FixedSyncCost
from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel
from srl.forecast import Pipeline, PipelineForecast, TrainState, compare, group_etas
from srl.policy.psrl import PSRLPolicy
from srl.simulator.config import SimConfig
from srl.simulator.simulator import Simulator
from srl.workload.distributions import uniform
from srl.workload.grouped import GroupedWorkload

EPS = 0.25
"""A power of two so every tolerance boundary below is exact in binary floating point."""


def snapshot(staleness: int, num_entries: int, next_buffer_id: int, buffers: tuple) -> InventorySnapshot:
    return InventorySnapshot(
        staleness=staleness,
        num_entries=num_entries,
        next_buffer_id=next_buffer_id,
        buffers=buffers,
    )


class TestEvaluateIdleTrainer:
    def test_a_buffer_already_ready_starts_from_now(self):
        snap = snapshot(0, 1, 1, (BufferView(buffer_id=0, ready_num_entries=1, n_occupied=1, reserved=()),))
        train = TrainState(training_end=None, now=5.0, cycle=2.0, weight_load=1.0)
        pipeline = PipelineForecast(snap, train).evaluate({})
        assert pipeline.buffers == (0,)
        assert pipeline.ready == (-math.inf,)
        assert pipeline.publish_times == (7.0,)
        assert pipeline.train_end == (8.0,)

    def test_a_pending_buffer_starts_from_its_ready_time(self):
        snap = snapshot(0, 1, 1, (BufferView(buffer_id=0, ready_num_entries=1, n_occupied=0, reserved=((7, 0),)),))
        train = TrainState(training_end=None, now=2.0, cycle=3.0, weight_load=1.0)
        pipeline = PipelineForecast(snap, train).evaluate({7: 9.0})
        assert pipeline.ready == (9.0,)
        assert pipeline.publish_times == (12.0,)
        assert pipeline.train_end == (13.0,)


class TestEvaluateTrainingInProgress:
    def test_training_end_dominates_an_earlier_ready_time(self):
        snap = snapshot(0, 1, 1, (BufferView(buffer_id=0, ready_num_entries=1, n_occupied=0, reserved=((0, 0),)),))
        train = TrainState(training_end=10.0, now=0.0, cycle=2.0, weight_load=1.0)
        pipeline = PipelineForecast(snap, train).evaluate({0: 5.0})
        # e_prev = training_end + weight_load = 11.0, which beats the 5.0 ready time.
        assert pipeline.publish_times == (13.0,)
        assert pipeline.train_end == (14.0,)

    def test_a_later_ready_time_dominates_training_end(self):
        snap = snapshot(0, 1, 1, (BufferView(buffer_id=0, ready_num_entries=1, n_occupied=0, reserved=((0, 0),)),))
        train = TrainState(training_end=10.0, now=0.0, cycle=2.0, weight_load=1.0)
        pipeline = PipelineForecast(snap, train).evaluate({0: 20.0})
        assert pipeline.publish_times == (22.0,)
        assert pipeline.train_end == (23.0,)

    def test_reconfig_adds_to_the_weight_load(self):
        snap = snapshot(0, 1, 1, (BufferView(buffer_id=0, ready_num_entries=1, n_occupied=1, reserved=()),))
        train = TrainState(training_end=None, now=0.0, cycle=1.0, weight_load=1.0, reconfig=0.5)
        pipeline = PipelineForecast(snap, train).evaluate({})
        assert pipeline.publish_times == (1.0,)
        assert pipeline.train_end == (2.5,)


class TestEvaluateChaining:
    def test_two_independent_buffers_chain_through_the_pipeline(self):
        buffers = (
            BufferView(buffer_id=0, ready_num_entries=1, n_occupied=0, reserved=((0, 0),)),
            BufferView(buffer_id=1, ready_num_entries=1, n_occupied=0, reserved=((1, 1),)),
        )
        snap = snapshot(0, 1, 2, buffers)
        train = TrainState(training_end=None, now=0.0, cycle=2.0, weight_load=0.5)
        pipeline = PipelineForecast(snap, train).evaluate({0: 3.0, 1: 4.0})
        assert pipeline.ready == (3.0, 4.0)
        # Buffer 0: start=3.0, publish=5.0, end=5.5.
        assert pipeline.publish_times[0] == pytest.approx(5.0)
        assert pipeline.train_end[0] == pytest.approx(5.5)
        # Buffer 1 is ready at 4.0 but training is still busy with buffer 0 until 5.5.
        assert pipeline.publish_times[1] == pytest.approx(7.5)
        assert pipeline.train_end[1] == pytest.approx(8.0)

    def test_an_unfillable_buffer_makes_every_later_buffer_infinite(self):
        buffers = (
            BufferView(buffer_id=0, ready_num_entries=2, n_occupied=0, reserved=((0, 0),)),
            BufferView(buffer_id=1, ready_num_entries=2, n_occupied=0, reserved=((1, 1), (2, 1))),
        )
        snap = snapshot(0, 2, 2, buffers)
        train = TrainState(training_end=None, now=0.0, cycle=2.0, weight_load=0.5)
        # Buffer 0 needs a second entry that never arrives, so it never fills. Buffer 1 is fully
        # reserved on its own and is reachable in isolation.
        pipeline = PipelineForecast(snap, train).evaluate({0: 1.0, 1: 2.0, 2: 3.0})
        assert math.isinf(pipeline.ready[0])
        assert math.isinf(pipeline.train_end[0])
        # Buffer 1's own ready time is finite, but it inherits the stall from buffer 0.
        assert pipeline.ready[1] == 3.0
        assert math.isinf(pipeline.train_end[1])


class TestCompare:
    def _pipeline(self, *train_end: float) -> Pipeline:
        return Pipeline(
            buffers=tuple(range(len(train_end))),
            ready=tuple(0.0 for _ in train_end),
            train_end=train_end,
            publish_times=tuple(0.0 for _ in train_end),
        )

    def test_within_tolerance_ties(self):
        assert compare(self._pipeline(10.0), self._pipeline(10.0 + EPS * 0.5), EPS) == 0

    def test_at_the_boundary_ties(self):
        assert compare(self._pipeline(10.0), self._pipeline(10.0 + EPS), EPS) == 0

    def test_past_the_boundary_decides(self):
        assert compare(self._pipeline(10.0), self._pipeline(10.0 + 2 * EPS), EPS) == -1
        assert compare(self._pipeline(10.0 + 2 * EPS), self._pipeline(10.0), EPS) == 1

    def test_a_tied_first_component_falls_through_to_the_second(self):
        a = self._pipeline(10.0, 20.0)
        b = self._pipeline(10.0 + EPS * 0.5, 19.0)
        assert compare(a, b, EPS) == 1

    def test_infinite_beats_finite(self):
        assert compare(self._pipeline(math.inf), self._pipeline(50.0), EPS) == 1
        assert compare(self._pipeline(50.0), self._pipeline(math.inf), EPS) == -1

    def test_infinite_ties_infinite_and_falls_through(self):
        a = self._pipeline(math.inf, 5.0)
        b = self._pipeline(math.inf, 5.0)
        assert compare(a, b, EPS) == 0

    def test_a_shorter_vector_compares_only_the_shared_prefix(self):
        a = self._pipeline(10.0, 20.0)
        b = self._pipeline(10.0)
        assert compare(a, b, EPS) == 0

    def test_tolerance_is_not_transitive(self):
        """a ties b and b ties c, but a and c differ once their gap passes eps."""
        a, b, c = self._pipeline(10.0), self._pipeline(10.0 + EPS), self._pipeline(10.0 + 2 * EPS)
        assert compare(a, b, EPS) == 0
        assert compare(b, c, EPS) == 0
        assert compare(a, c, EPS) == -1


class TestGroupEtas:
    def test_the_group_eta_is_the_max_of_its_members(self):
        etas = group_etas({10: 5.0, 11: 8.0}, {1: [10, 11]}, {})
        assert etas == {1: 8.0}

    def test_a_finished_member_contributes_its_known_completion_time(self):
        etas = group_etas({}, {1: [10, 11]}, {10: 2.0, 11: 1.0})
        assert etas == {1: 2.0}

    def test_a_member_with_neither_source_makes_the_group_infinite(self):
        etas = group_etas({10: 5.0}, {1: [10, 11]}, {})
        assert etas == {1: math.inf}

    def test_a_placed_prediction_takes_precedence_over_a_stale_finished_entry(self):
        etas = group_etas({10: 5.0}, {1: [10]}, {10: 99.0})
        assert etas == {1: 5.0}

    def test_multiple_groups_are_independent(self):
        etas = group_etas({10: 5.0, 20: 1.0}, {1: [10], 2: [20]}, {})
        assert etas == {1: 5.0, 2: 1.0}


# --- Property test against a real simulator run ---


def _build_simulator(train_c: float, sync_c: float, weight_load: float) -> tuple[Simulator, PSRLPolicy]:
    cost = UnifiedRoofline(F=0.004, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
    training_cost = TrainingCostModel(FixedTrainingLatency(train_c), FixedTrainingMemory(0))
    workload = GroupedWorkload.from_lengths([150], uniform(80, 40, 120, seed=3))
    cfg = SimConfig(
        batch_size=2,
        max_staleness=1,
        n_versions=6,
        n_instances=1,
        weight_load_time=weight_load,
        workload=workload,
    )
    policy = _ReadyObservingPolicy()
    sim = _SnapshottingSimulator(
        rollout_cost=cost,
        training_cost=training_cost,
        sync_cost=FixedSyncCost(sync_c),
        policy=policy,
        cfg=cfg,
    )
    return sim, policy


class _ReadyObservingPolicy(PSRLPolicy):
    """Record the first simulated time each buffer's status flips to READY."""

    def __init__(self) -> None:
        super().__init__()
        self.true_ready: dict[int, float] = {}

    def on_group_complete(self, group, st):
        action = super().on_group_complete(group, st)
        if self._inventory is not None:
            for buffer_id in self._inventory.buffers:
                if buffer_id in self.true_ready:
                    continue
                status = self._inventory.get_buffer_status(buffer_id)
                if status in (BufferStatus.READY, BufferStatus.READY_WITH_CAPACITY):
                    self.true_ready[buffer_id] = st.now
        return action


class _SnapshottingSimulator(Simulator):
    """Record the inventory snapshot and `TrainState` at every real training deadline."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.snapshots: list[tuple[InventorySnapshot, TrainState]] = []

    def _run_engine(self, deadline=None, stop_on_batch=False) -> None:
        if deadline is not None:
            train = TrainState(
                training_end=self._training_end,
                now=self.t,
                cycle=self._train_cycle((100,))[0].latency_s
                + self._train_cycle((100,))[2]
                + self._train_cycle((100,))[3],
                weight_load=self.cfg.weight_load_time,
            )
            self.snapshots.append((self.policy.inventory_snapshot(self._state()), train))
        super()._run_engine(deadline=deadline, stop_on_batch=stop_on_batch)


class TestPropertyAgainstARealRun:
    def test_the_first_pending_buffer_matches_the_real_run_exactly(self):
        sim, policy = _build_simulator(train_c=4.0, sync_c=1.0, weight_load=1.5)
        result = sim.run()
        assert not result.livelocked
        assert result.versions_done == sim.cfg.n_versions

        finish_true = {req.group_id: req.complete_time for req in result.consumed}

        checked = 0
        for snap, train in sim.snapshots:
            if not snap.buffers:
                continue
            first = snap.buffers[0]
            # Only a buffer whose reserved supply is already complete at snapshot time can be
            # replayed exactly: a later admission would add a reservation the snapshot never saw.
            if first.n_empty != 0 or not first.reserved:
                continue
            prompt_ids = [prompt_id for prompt_id, _ in first.reserved]
            if not all(prompt_id in finish_true for prompt_id in prompt_ids):
                continue
            if first.buffer_id not in policy.true_ready:
                continue

            pipeline = PipelineForecast(snap, train).evaluate(finish_true)

            assert pipeline.buffers[0] == first.buffer_id
            assert pipeline.ready[0] == pytest.approx(policy.true_ready[first.buffer_id], abs=1e-9)
            actual_end = result.train_timestamps[first.buffer_id][1]
            assert pipeline.train_end[0] == pytest.approx(actual_end, abs=1e-9)
            checked += 1

        assert checked >= 3, f"Too few qualifying buffers to call this a property test: {checked}."
