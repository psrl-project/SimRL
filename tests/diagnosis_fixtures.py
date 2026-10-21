"""Small real simulator runs with telemetry on, shared by the diagnosis tests.

Every fixture runs a real `Simulator` and writes real streams. Nothing here
fabricates a JSONL record, because the thing under test is whether the
reconstruction agrees with what the simulator actually logs.
"""

from __future__ import annotations

from pathlib import Path

from srl.cost.rollout.roofline import UnifiedRoofline
from srl.policy.psrl import PSRLPolicy
from srl.simulator.config import SimConfig
from srl.simulator.simulator import Simulator
from srl.telemetry.store import JsonlStore
from srl.workload.grouped import GroupedWorkload
from srl.workload.trace import RequestGroup

BLOCK_SIZE = 16
TOKEN_BUDGET = 2048


def _cost_model():
    """A cost model whose decode cost grows with context, so interference is visible."""
    return UnifiedRoofline(F=0.01, G=1e-4, A_d=1e-6)


def _store(tmp_path: Path) -> JsonlStore:
    return JsonlStore(tmp_path / "logs", overwrite=True)


def run_single_request(tmp_path: Path) -> JsonlStore:
    """One request, one instance, one version. No contention by construction."""
    store = _store(tmp_path)
    cfg = SimConfig(
        batch_size=1,
        group_size=1,
        staleness_horizon=1,
        n_versions=1,
        n_instances=1,
        kv_blocks=4096,
        block_size=BLOCK_SIZE,
        token_budget=TOKEN_BUDGET,
        workload=GroupedWorkload.from_lengths([400], [1]),
        log_telemetry=True,
    )
    sim = Simulator(
        rollout_cost=_cost_model(),
        policy=PSRLPolicy(),
        cfg=cfg,
        store=store,
        backend="native",
    )
    sim.run()
    store.close()
    return JsonlStore(tmp_path / "logs")


def run_mixed(tmp_path: Path) -> JsonlStore:
    """Two instances, one long group among shorts, publishes that abort, KV that preempts.

    `kv_blocks` is sized so the long group's context plus a few shorts exceeds
    the pool, which is what forces a preemption rather than merely allowing one.
    """
    store = _store(tmp_path)
    groups = []
    for group_id in range(12):
        is_long = group_id == 3
        length = 3000 if is_long else 120
        groups.append(
            RequestGroup(group_id=group_id, prompt_len=64, output_lens=(length, length))
        )
    cfg = SimConfig(
        batch_size=2,
        group_size=2,
        staleness_horizon=1,
        n_versions=3,
        n_instances=2,
        kv_blocks=(3000 + 64) // BLOCK_SIZE + 8,
        block_size=BLOCK_SIZE,
        token_budget=TOKEN_BUDGET,
        workload=GroupedWorkload.from_groups(groups),
        log_telemetry=True,
    )
    sim = Simulator(
        rollout_cost=_cost_model(),
        policy=PSRLPolicy(),
        cfg=cfg,
        store=store,
        backend="native",
    )
    sim.run()
    store.close()
    return JsonlStore(tmp_path / "logs")
