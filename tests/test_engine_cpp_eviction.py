"""
Check that the C++ engine honours the configured eviction strategy.

The same simulation runs on both backends and must produce identical request
telemetry, which is what makes the backends interchangeable.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("dryrun._engine_cpp")

from dryrun.cost.rollout import UnifiedRoofline  # noqa: E402
from dryrun.cost.training import (  # noqa: E402
    FixedTrainingLatency,
    FixedTrainingMemory,
    TrainingCostModel,
)
from dryrun.policy.psrl import PSRLPolicy  # noqa: E402
from dryrun.simulator import SimConfig, Simulator  # noqa: E402
from dryrun.telemetry.store import JsonlStore  # noqa: E402
from dryrun.workload.distributions import bimodal, uniform  # noqa: E402
from dryrun.workload.grouped import GroupedWorkload  # noqa: E402


def _run(backend: str, eviction: str, tmp_path):
    cfg = SimConfig(
        batch_size=4,
        max_staleness=1,
        n_versions=6,
        kv_blocks=120,
        token_budget=512,
        eviction_strategy=eviction,
        workload=GroupedWorkload.from_lengths(
            uniform(64, 16, 300, seed=101), bimodal(64, 40, 700, 0.3, seed=1)
        ),
    )
    out = tmp_path / backend
    store = JsonlStore(out, overwrite=True)
    sim = Simulator(
        rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=1e-6, A_d=1e-7),
        training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
        policy=PSRLPolicy(),
        cfg=cfg,
        store=store,
        backend=backend,
    )
    result = sim.run()
    store.close()
    rows = [json.loads(line) for line in (out / "request.jsonl").open()]
    return result, rows


@pytest.mark.parametrize("eviction", ["vllm_lru", "longest_ctx"])
def test_backends_agree_under_eviction(eviction, tmp_path):
    native, native_rows = _run("native", eviction, tmp_path / "n")
    cpp, cpp_rows = _run("cpp", eviction, tmp_path / "c")
    preempted = sum(r.get("num_preemptions", 0) for r in native_rows if r["event"] == "consume")
    assert preempted > 0, "The pool is not tight enough to exercise eviction at all."
    assert cpp.sim_time == native.sim_time
    assert cpp_rows == native_rows


def test_eviction_strategy_changes_the_cpp_run(tmp_path):
    """Check that the strategy is consulted rather than accepted and ignored."""
    lru, _ = _run("cpp", "vllm_lru", tmp_path / "lru")
    longest, _ = _run("cpp", "longest_ctx", tmp_path / "longest")
    assert lru.sim_time != longest.sim_time
