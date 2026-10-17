"""The database must obtain SOL from the op, not from its own branch on query type."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TABLES = _REPO / "outputs" / "profile" / "qwen3_8b" / "kernel_tp1" / "kernel"

pytestmark = pytest.mark.skipif(
    not _TABLES.is_dir(), reason="Measured perf tables are machine-local."
)


def test_the_database_has_no_isinstance_branch_on_query_type():
    """A type switch here is the duplication that let database and holdout drift apart."""
    import inspect

    from srl.cost.kernel.database import KernelDatabase

    source = inspect.getsource(KernelDatabase._sol)
    assert "isinstance" not in source, "_sol still branches on query type instead of asking the op"
    assert "GemmQuery" not in source, "_sol still names a concrete query class"


def test_a_gemm_query_prices_through_the_gemm_op():
    from srl.cost.kernel.database import GemmQuery, KernelDatabase
    from srl.cost.kernel.ops.gemm import GEMM_OP

    database = KernelDatabase.from_root(_TABLES, gpu="h20")
    query = GemmQuery(m=128, n=24576, k=4096)
    expected_sol = GEMM_OP.sol(
        database.gpu_spec, slice_key=query.slice_key, coords=query.coords
    ).latency_s
    utility = database.utility(query)
    assert database.query_gemm(query) == pytest.approx(expected_sol / utility, rel=1e-12)


def test_an_attention_query_prices_through_its_phase_op():
    from srl.cost.kernel.database import AttentionQuery, KernelDatabase
    from srl.cost.kernel.ops.attention import GENERATION_ATTENTION_OP

    database = KernelDatabase.from_root(_TABLES, gpu="h20")
    query = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=8, head_dim=128, step=2048
    )
    floors = database._floors_for_op(GENERATION_ATTENTION_OP, "vllm")
    expected_sol = GENERATION_ATTENTION_OP.sol(
        database.gpu_spec,
        slice_key=query.slice_key,
        coords=query.coords,
        floor_s=floors["generation_attention"],
    ).latency_s
    utility = database.utility(query)
    assert database.query_attention(query) == pytest.approx(expected_sol / utility, rel=1e-12)


def test_a_measured_shape_still_prices_to_its_measurement():
    """The round trip SOL/utility must return the stored latency at a measured point."""
    from srl.cost.kernel.database import GemmQuery, KernelDatabase
    from srl.cost.kernel.ops.base import op_for
    from srl.cost.kernel.table import KernelTable

    parquet = next((_TABLES / "h20" / "gemm" / "vllm").glob("*/gemm_perf.parquet"))
    table = KernelTable.load(parquet, op_for("gemm"))
    slice_key = ("bfloat16", 24576, 4096)
    coords, latency = table.rows(slice_key)[3]
    database = KernelDatabase.from_root(_TABLES, gpu="h20")
    query = GemmQuery(m=int(coords[0]), n=24576, k=4096)
    assert database.query_gemm(query) == pytest.approx(latency, rel=1e-9)
