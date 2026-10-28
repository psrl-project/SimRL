"""End-to-end: a `Simulator` actually configured with `cfg.dynamic_parallel` runs, reshapes
mid-round when the planner says to, and is inert (byte-identical to the static path) when the
planner never fires."""

from __future__ import annotations

import dataclasses

import pytest

from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel
from srl.policy.roll import RollPolicy
from srl.simulator import SimConfig, Simulator
from srl.simulator.dynamic_parallel import DynamicParallelConfig
from srl.workload.grouped import GroupedWorkload

_TRAINING = TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0))


def _cost(tp: int) -> UnifiedRoofline:
    # Distinct coefficients per degree so a merge is numerically distinguishable, not a
    # claim about real TP speedup -- this module tests plumbing, not benefit.
    return UnifiedRoofline(F=0.005 / tp, W=0.001, G=0.0001, A_d=1e-7)


def _workload(n_groups: int, lengths: list[int]) -> GroupedWorkload:
    cycled = [lengths[i % len(lengths)] for i in range(n_groups)]
    return GroupedWorkload.from_lengths([64] * n_groups, cycled)


def test_a_never_firing_planner_is_byte_identical_to_the_static_fleet():
    """`merge_below=0` can never see two siblings both below it (residents are never
    negative), so the dynamic path must reproduce the static 2xTP1 fleet exactly."""
    workload = _workload(n_groups=16, lengths=[40])
    base_cfg = SimConfig(
        batch_size=4,
        staleness_horizon=1,
        n_versions=3,
        workload=workload,
        kv_blocks=4000,
        token_budget=4096,
        log_telemetry=False,
    )

    static_cfg = dataclasses.replace(base_cfg, n_instances=2)
    static_sim = Simulator(
        rollout_cost=_cost(1), training_cost=_TRAINING, policy=RollPolicy(), cfg=static_cfg
    )
    static_result = static_sim.run()

    dp = DynamicParallelConfig(
        tp_degrees=(1, 2),
        n_gpus=2,
        kv_blocks_per_tp={1: 4000, 2: 4000},
        planner_args={"merge_below": 0.0, "split_above": 1e9, "min_dwell_s": 0.0},
    )
    dynamic_cfg = dataclasses.replace(base_cfg, dynamic_parallel=dp)
    dynamic_sim = Simulator(
        rollout_cost=_cost(1),
        training_cost=_TRAINING,
        policy=RollPolicy(),
        cfg=dynamic_cfg,
        rollout_costs={1: _cost(1), 2: _cost(2)},
    )
    dynamic_result = dynamic_sim.run()

    assert dynamic_sim._dynamic_fleet.n_reshapes == 0
    assert not dynamic_result.livelocked
    assert dynamic_result.versions_done == static_result.versions_done
    assert [r.rid for r in dynamic_result.consumed] == [r.rid for r in static_result.consumed]
    assert sum(r.ctx for r in dynamic_result.consumed) == sum(r.ctx for r in static_result.consumed)


def test_a_merge_fires_mid_round_and_the_run_still_completes_cleanly():
    # Staggered lengths so residents on a instance drain at different real times instead of
    # in lockstep -- a synchronized drain is undone and re-hidden by `_undo_and_align` before
    # the sibling instance's own identical-timed completion ever commits.
    workload = _workload(n_groups=32, lengths=[20, 20, 60, 60])
    cfg = SimConfig(
        batch_size=4,
        staleness_horizon=0,
        n_versions=4,
        workload=workload,
        kv_blocks=4000,
        token_budget=4096,
        log_telemetry=False,
        dynamic_parallel=DynamicParallelConfig(
            tp_degrees=(1, 2),
            n_gpus=2,
            kv_blocks_per_tp={1: 4000, 2: 4000},
            planner_args={"merge_below": 2.0, "split_above": 1e9, "min_dwell_s": 0.0},
        ),
    )
    sim = Simulator(
        rollout_cost=_cost(1),
        training_cost=_TRAINING,
        policy=RollPolicy(),
        cfg=cfg,
        rollout_costs={1: _cost(1), 2: _cost(2)},
    )
    result = sim.run()

    assert not result.livelocked
    assert result.versions_done == cfg.n_versions
    assert sim._dynamic_fleet.n_reshapes > 0
    # Every slot the fleet ever built is still exactly where it started: a reshape must
    # never renumber `sim.instances`.
    assert [inst.instance_id for inst in sim.instances] == sorted(
        inst.instance_id for inst in sim.instances
    )


def test_dynamic_parallel_on_the_cpp_backend_is_refused():
    workload = _workload(n_groups=8, lengths=[20])
    cfg = SimConfig(
        batch_size=4,
        n_versions=1,
        workload=workload,
        log_telemetry=False,
        dynamic_parallel=DynamicParallelConfig(
            tp_degrees=(1, 2), n_gpus=2, kv_blocks_per_tp={1: 4000, 2: 4000}
        ),
    )
    with pytest.raises(ValueError, match="dynamic_parallel"):
        Simulator(
            rollout_cost=_cost(1),
            training_cost=_TRAINING,
            policy=RollPolicy(),
            cfg=cfg,
            backend="cpp",
            rollout_costs={1: _cost(1), 2: _cost(2)},
        )


def test_dynamic_parallel_without_rollout_costs_is_refused():
    workload = _workload(n_groups=8, lengths=[20])
    cfg = SimConfig(
        batch_size=4,
        n_versions=1,
        workload=workload,
        log_telemetry=False,
        dynamic_parallel=DynamicParallelConfig(
            tp_degrees=(1, 2), n_gpus=2, kv_blocks_per_tp={1: 4000, 2: 4000}
        ),
    )
    with pytest.raises(ValueError, match="rollout_costs"):
        Simulator(rollout_cost=_cost(1), training_cost=_TRAINING, policy=RollPolicy(), cfg=cfg)
