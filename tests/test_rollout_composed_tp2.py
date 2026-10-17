"""A tp=2 step must price its collectives through the real measured tables.

Every other composed-rollout test at `tp>1` covers construction only, because until the tp=2
GEMM and attention sweeps existed no table could answer a sharded query. So the path
`predict -> comm_queries -> query_comm -> table -> SOL/utility`: the entire reason the
`all_reduce` op exists: had never executed. This file closes that gap.

What is asserted here is that the comm term is present, positive, and exactly what
`comm_queries` prices. It is NOT a claim about end-to-end accuracy: no vLLM tp=2 profiling
baseline exists, so `kernels validate` cannot score a tp=2 prediction. See
`docs/composed-rollout.md`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TP2_TABLES = _REPO / "outputs" / "profile" / "qwen3_8b" / "kernel_tp2" / "kernel"
_TP1_TABLES = _REPO / "outputs" / "profile" / "qwen3_8b" / "kernel_tp1" / "kernel"
_CHECKPOINT = "/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B"

pytestmark = pytest.mark.skipif(
    not (_TP2_TABLES / "h20" / "collective").is_dir()
    or not (_TP2_TABLES / "h20" / "gemm").is_dir()
    or not Path(_CHECKPOINT).is_dir(),
    reason="The tp=2 perf tables and the checkpoint are machine-local.",
)


def _workload(batch: int = 16, depth: int = 2048):
    from srl.cost.rollout import RolloutWorkload

    return RolloutWorkload(
        phase="decode",
        n_tokens=batch,
        sum_l2=0.0,
        ctxsum=float(batch * depth),
        request_count=batch,
    )


def _model(tensor_parallel: int, root: Path):
    from srl.cost.kernel.database import KernelDatabase
    from srl.cost.kernel.model_ops import ModelOps
    from srl.cost.rollout.composed import ComposedRollout

    database = KernelDatabase.from_root(root, gpu="h20", utility_surface="pchip")
    ops = ModelOps.from_checkpoint(_CHECKPOINT, tensor_parallel=tensor_parallel)
    return ComposedRollout(database, ops, tensor_parallel=tensor_parallel), database, ops


def test_a_tp2_step_prices_end_to_end_through_the_tables():
    """The whole point: this raised KernelCoverageError until the tp=2 sweeps existed."""
    model, _, _ = _model(2, _TP2_TABLES)
    assert model.predict(_workload()) > 0.0


def test_the_step_is_exactly_its_gemms_plus_attention_plus_comm():
    """Pins predict()'s decomposition, so a wrong comm multiplier cannot hide inside the total.

    Containment alone (`total > comm`) is too weak: dropping the 2-per-layer factor is a 72x
    error that still satisfies it. Reconstructing the whole sum is what catches that.
    """
    model, database, ops = _model(2, _TP2_TABLES)
    workload = _workload()
    tokens, requests = int(workload.n_tokens), int(workload.request_count)

    gemms = sum(
        multiplier * database.query_gemm(query)
        for query, multiplier in ops.gemm_queries(tokens, requests)
    )
    comm = sum(
        multiplier * database.query_comm(query)
        for query, multiplier in ops.comm_queries(tokens)
    )
    attention = ops.num_layers * database.query_attention(
        ops.attention_query(
            "generation", batch_size=requests, step=int(workload.ctxsum // requests)
        )
    )
    assert comm > 0.0
    assert model.predict(workload) == pytest.approx(gemms + comm + attention, rel=1e-12)


def test_the_collective_runs_twice_per_layer():
    """One all-reduce after o_proj and one after down_proj, so the multiplier is 2 x layers."""
    _, _, ops = _model(2, _TP2_TABLES)
    (query, multiplier), = ops.comm_queries(16)
    assert multiplier == 2 * ops.num_layers
    assert query.world_size == 2


def test_a_deeper_batch_pays_more_comm_because_the_message_grows():
    """message_bytes scales with the token count, so a wider batch moves the comm term."""
    _, database, ops = _model(2, _TP2_TABLES)
    narrow = sum(m * database.query_comm(q) for q, m in ops.comm_queries(16))
    wide = sum(m * database.query_comm(q) for q, m in ops.comm_queries(4096))
    assert wide > narrow


def test_tp1_emits_no_collective_at_all():
    """No TP all-reduce runs at world size 1, which is what keeps every tp=1 number unchanged."""
    _, _, ops = _model(1, _TP1_TABLES)
    assert ops.comm_queries(4096) == ()


def test_an_unmeasured_world_size_raises_rather_than_extrapolating():
    """world_size is a slice column: a 2-rank and a 16-rank ring are different schedules.

    This used to assert on world_size 8, which was unmeasured when each degree needed its
    own config file and its own tree. One sweep now carries every intra-node degree in one
    table, so 8 is measured and 16 is the smallest degree that is not (it requires two
    nodes). The behaviour under test is unchanged: a slice column refuses rather than
    blending two topologies.
    """
    from srl.cost.kernel.database import CommQuery, KernelDatabase
    from srl.cost.kernel.interp import KernelCoverageError

    database = KernelDatabase.from_root(_TP2_TABLES, gpu="h20")
    with pytest.raises(KernelCoverageError):
        database.query_comm(
            CommQuery(
                op="all_reduce",
                world_size=16,
                nodes=2,
                message_bytes=1 << 20,
                backend="nccl",
            )
        )


def test_an_unmeasured_node_count_raises_at_a_measured_world_size():
    """The point of the `nodes` slice: 8 ranks on one node and on two are not the same ring.

    The intra-node sweep measures `(8, 1)`, so a query for `(8, 2)` must refuse rather than
    answer from the NVLink rows. Reading those rows would price a NIC hop at NVLink speed,
    a ~9x optimistic error that interpolates perfectly smoothly and so would never look
    like a coverage gap.
    """
    from srl.cost.kernel.database import CommQuery, KernelDatabase
    from srl.cost.kernel.interp import KernelCoverageError

    database = KernelDatabase.from_root(_TP2_TABLES, gpu="h20")
    measured = database.query_comm(
        CommQuery(op="all_reduce", world_size=8, nodes=1, message_bytes=1 << 20, backend="nccl")
    )
    assert measured > 0.0
    with pytest.raises(KernelCoverageError):
        database.query_comm(
            CommQuery(
                op="all_reduce", world_size=8, nodes=2, message_bytes=1 << 20, backend="nccl"
            )
        )
