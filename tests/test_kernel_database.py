"""KernelDatabase shape queries, backend selection, and coverage errors."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dryrun.cost.kernel.database import (
    AttentionQuery,
    GemmQuery,
    KernelDatabase,
)
from dryrun.cost.kernel.hardware import load_gpu_spec
from dryrun.cost.kernel.interp import KernelCoverageError
from dryrun.cost.kernel.sol import gemm_sol

_VERSION = "0.26.1.dev0"


def _table_path(root: Path, family: str, backend: str, op_name: str) -> Path:
    path = root / "h20" / family / backend / _VERSION / f"{op_name}_perf.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_gemm(root: Path, shapes: list[tuple[int, int, int, float]]) -> None:
    count = len(shapes)
    pq.write_table(
        pa.table(
            {
                "framework": ["vllm"] * count,
                "version": [_VERSION] * count,
                "device": ["NVIDIA H20"] * count,
                "op_name": ["gemm"] * count,
                "kernel_source": ["torch.nn.functional.linear"] * count,
                "gemm_dtype": ["bfloat16"] * count,
                "m": [shape[0] for shape in shapes],
                "n": [shape[1] for shape in shapes],
                "k": [shape[2] for shape in shapes],
                "latency_s": [shape[3] for shape in shapes],
            }
        ),
        _table_path(root, "gemm", "vllm", "gemm"),
    )


def _write_generation(root: Path, backend: str, kernel_source: str, points: list[tuple[int, int, float]]) -> None:
    count = len(points)
    pq.write_table(
        pa.table(
            {
                "framework": [backend] * count,
                "version": [_VERSION] * count,
                "device": ["NVIDIA H20"] * count,
                "op_name": ["generation_attention"] * count,
                "kernel_source": [kernel_source] * count,
                "num_key_value_heads": [8] * count,
                "head_dim": [128] * count,
                "attn_dtype": ["bfloat16"] * count,
                "kv_cache_dtype": ["bfloat16"] * count,
                "block_size": [16] * count,
                "window_size": [0] * count,
                "num_heads": [32] * count,
                "batch_size": [point[0] for point in points],
                "step": [point[1] for point in points],
                "latency_s": [point[2] for point in points],
            }
        ),
        _table_path(root, "attention", backend, "generation_attention"),
    )


@pytest.fixture
def gemm_root(tmp_path):
    # Measurements set to exactly 2x SOL, so utility is 0.5 everywhere.
    spec = load_gpu_spec("h20")
    shapes = []
    for m in (1024, 2048, 4096):
        sol = gemm_sol(spec, m=m, n=4096, k=4096, dtype="bfloat16").latency_s
        shapes.append((m, 4096, 4096, 2.0 * sol))
    _write_gemm(tmp_path, shapes)
    return tmp_path


def test_gemm_query_inverts_the_utility_definition(gemm_root):
    database = KernelDatabase.from_root(gemm_root, gpu="h20")
    spec = database.gpu_spec
    expected = 2.0 * gemm_sol(spec, m=2048, n=4096, k=4096, dtype="bfloat16").latency_s
    assert database.query_gemm(GemmQuery(m=2048, n=4096, k=4096)) == pytest.approx(expected)


def test_gemm_query_interpolates_an_unmeasured_shape(gemm_root):
    database = KernelDatabase.from_root(gemm_root, gpu="h20")
    spec = database.gpu_spec
    expected = 2.0 * gemm_sol(spec, m=1448, n=4096, k=4096, dtype="bfloat16").latency_s
    assert database.query_gemm(GemmQuery(m=1448, n=4096, k=4096)) == pytest.approx(expected, rel=1e-6)


def test_gemm_utility_is_reported_for_diagnostics(gemm_root):
    database = KernelDatabase.from_root(gemm_root, gpu="h20")
    assert database.utility(GemmQuery(m=2048, n=4096, k=4096)) == pytest.approx(0.5)


def test_generation_attention_query_returns_the_measured_latency(tmp_path):
    _write_generation(tmp_path, "vllm", "vllm_flash_attn_fa3", [(16, 1024, 8.0e-5), (16, 2048, 1.4e-4)])
    database = KernelDatabase.from_root(tmp_path, gpu="h20")
    query = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=8, head_dim=128, step=1024
    )
    assert database.query_attention(query) == pytest.approx(8.0e-5)


def test_backend_override_selects_the_flashinfer_table(tmp_path):
    _write_generation(tmp_path, "vllm", "vllm_flash_attn_fa3", [(16, 1024, 8.0e-5)])
    _write_generation(tmp_path, "flashinfer", "flashinfer_0.6.14", [(16, 1024, 9.5e-5)])
    database = KernelDatabase.from_root(tmp_path, gpu="h20")
    base = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=8, head_dim=128, step=1024
    )
    assert database.query_attention(base) == pytest.approx(8.0e-5)
    assert database.query_attention(replace_backend(base, "flashinfer")) == pytest.approx(9.5e-5)
    assert database.backends("generation_attention") == ("flashinfer", "vllm")


def replace_backend(query: AttentionQuery, backend: str) -> AttentionQuery:
    from dataclasses import replace

    return replace(query, backend=backend)


def test_a_different_target_gpu_rescales_through_sol(gemm_root, tmp_path):
    spec_path = tmp_path / "slow.yaml"
    spec_path.write_text(
        "h20:\n"
        "  name: Half Speed H20\n"
        "  sm_count: 78\n"
        "  memory_bytes: 102010650624\n"
        "  memory_bandwidth_bytes_per_s: 2.0e12\n"
        "  link_bandwidth_bytes_per_s: 4.5e11\n"
        "  tc_flops_per_s:\n"
        "    bfloat16: 74.0e12\n",
        encoding="utf-8",
    )
    from dryrun.cost.kernel.hardware import load_gpu_spec as load_spec

    fast = KernelDatabase.from_root(gemm_root, gpu="h20")
    slow = KernelDatabase.from_root(gemm_root, gpu="h20", gpu_spec=load_spec("h20", path=spec_path))
    query = GemmQuery(m=2048, n=4096, k=4096)
    # Utility is a property of the measurement; halving peak throughput doubles SOL
    # and therefore doubles the predicted latency at the same utility.
    assert slow.query_gemm(query) == pytest.approx(2.0 * fast.query_gemm(query), rel=1e-6)
    assert slow.utility(query) == pytest.approx(fast.utility(query))


def test_missing_backend_directory_names_the_measured_backends(tmp_path):
    _write_generation(tmp_path, "flashinfer", "flashinfer_0.6.14", [(16, 1024, 9.5e-5)])
    database = KernelDatabase.from_root(tmp_path, gpu="h20")
    query = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=8, head_dim=128, step=1024
    )
    with pytest.raises(KernelCoverageError, match="flashinfer"):
        database.query_attention(query)


def test_unmeasured_slice_is_rejected(tmp_path):
    _write_generation(tmp_path, "vllm", "vllm_flash_attn_fa3", [(16, 1024, 8.0e-5)])
    database = KernelDatabase.from_root(tmp_path, gpu="h20")
    query = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=4, head_dim=128, step=1024
    )
    with pytest.raises(KernelCoverageError):
        database.query_attention(query)


def test_context_phase_requires_step_zero():
    with pytest.raises(KernelCoverageError):
        AttentionQuery(
            phase="context", batch_size=1, num_heads=32, num_kv_heads=8, head_dim=128,
            seq_len=1024, step=5,
        )


def test_generation_phase_requires_unit_seq_len():
    with pytest.raises(KernelCoverageError):
        AttentionQuery(
            phase="generation", batch_size=1, num_heads=32, num_kv_heads=8, head_dim=128,
            seq_len=1024, step=5,
        )


def test_unknown_phase_is_rejected():
    with pytest.raises(KernelCoverageError):
        AttentionQuery(phase="mixed", batch_size=1, num_heads=32, num_kv_heads=8, head_dim=128)


def test_a_comm_query_without_a_measured_table_names_the_sweep_to_run(gemm_root):
    """query_comm now prices through the table machinery, so a missing table is the failure."""
    from dryrun.cost.kernel.database import CommQuery

    database = KernelDatabase.from_root(gemm_root, gpu="h20")
    with pytest.raises(KernelCoverageError, match="--family collective"):
        database.query_comm(CommQuery(world_size=2, message_bytes=1 << 20))


def test_ambiguous_version_requires_a_pin(tmp_path):
    _write_generation(tmp_path, "vllm", "vllm_flash_attn_fa3", [(16, 1024, 8.0e-5)])
    second = tmp_path / "h20" / "attention" / "vllm" / "0.27.0"
    second.mkdir(parents=True)
    (second / "generation_attention_perf.parquet").write_bytes(b"")
    database = KernelDatabase.from_root(tmp_path, gpu="h20")
    query = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=8, head_dim=128, step=1024
    )
    with pytest.raises(KernelCoverageError, match="versions"):
        database.query_attention(query)
    pinned = KernelDatabase.from_root(tmp_path, gpu="h20", versions={"vllm": _VERSION})
    assert pinned.query_attention(query) == pytest.approx(8.0e-5)


def _write_generation_device(root: Path, backend: str, device: str, points: list[tuple[int, int, float]]) -> None:
    """Write a generation_attention parquet with a custom device string."""
    count = len(points)
    pq.write_table(
        pa.table(
            {
                "framework": [backend] * count,
                "version": [_VERSION] * count,
                "device": [device] * count,
                "op_name": ["generation_attention"] * count,
                "kernel_source": ["vllm_flash_attn_fa3"] * count,
                "num_key_value_heads": [8] * count,
                "head_dim": [128] * count,
                "attn_dtype": ["bfloat16"] * count,
                "kv_cache_dtype": ["bfloat16"] * count,
                "block_size": [16] * count,
                "window_size": [0] * count,
                "num_heads": [32] * count,
                "batch_size": [point[0] for point in points],
                "step": [point[1] for point in points],
                "latency_s": [point[2] for point in points],
            }
        ),
        _table_path(root, "attention", backend, "generation_attention"),
    )


def test_sol_closure_cache_keeps_its_table_alive(tmp_path):
    """
    Pin the observable correctness property of the (table, slice_key) cache key.

    `self._tables` holds every loaded table alive unconditionally, so a weakref test
    cannot distinguish between `id(table)` and `table` as the cache key: the table is
    never freed either way.  The property that actually matters is behavioral: two
    DIFFERENT tables loaded for different backends must each get their OWN closure, and
    therefore correctly different SOL-derived utilities, even when the tables share the
    same slice_key.  An `id(table)`-keyed cache could in principle collide (same CPython
    address for two distinct objects) whereas the object-keyed cache cannot.

    This test verifies the observable outcome: each backend's table produces results
    consistent with its own closure, not the other's.
    """
    # Two backends, identical shapes but different latencies.
    _write_generation(tmp_path, "vllm", "vllm_flash_attn_fa3", [(16, 1024, 8.0e-5)])
    _write_generation(tmp_path, "flashinfer", "flashinfer_0.6.14", [(16, 1024, 9.5e-5)])

    database = KernelDatabase.from_root(tmp_path, gpu="h20")
    base = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=8, head_dim=128, step=1024
    )
    flashinfer_query = replace_backend(base, "flashinfer")

    utility_vllm = database.utility(base)
    utility_fi = database.utility(flashinfer_query)

    # Both tables share the same slice_key.  If the SOL closures were aliased (wrong
    # cache key), one utility would be computed with the other table's rows and the two
    # values would be equal.  They must differ because the latencies differ.
    assert utility_vllm != pytest.approx(utility_fi), (
        "Two tables loaded for different backends with the same slice_key must produce "
        "distinct utilities derived from their own rows, not each other's."
    )

    # Sanity-check direction: higher latency => lower utility.
    assert utility_vllm > utility_fi


def test_matching_device_loads_fine(tmp_path):
    """A table whose device column matches the directory GPU slug loads without error."""
    _write_generation(tmp_path, "vllm", "vllm_flash_attn_fa3", [(16, 1024, 8.0e-5)])
    database = KernelDatabase.from_root(tmp_path, gpu="h20")
    query = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=8, head_dim=128, step=1024
    )
    # Should not raise.
    result = database.query_attention(query)
    assert result == pytest.approx(8.0e-5)


def test_mismatched_device_raises_with_both_slugs(tmp_path):
    """A table filed under h20/ but tagged as H100 raises naming both slugs."""
    _write_generation_device(
        tmp_path, "vllm", "NVIDIA H100 80GB HBM3", [(16, 1024, 8.0e-5)]
    )
    database = KernelDatabase.from_root(tmp_path, gpu="h20")
    query = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=8, head_dim=128, step=1024
    )
    with pytest.raises(KernelCoverageError) as exc_info:
        database.query_attention(query)
    msg = str(exc_info.value)
    assert "h100_80gb_hbm3" in msg, f"Expected offending slug in error message: {msg!r}"
    assert "h20" in msg, f"Expected directory slug in error message: {msg!r}"


# Regression: step=0 rows in the parquet must not cause query_attention to crash.
# Before the fix, KernelTable.grid() would pass the step=0 row to UtilityGrid, which
# rejected it with KernelCoverageError because log2(0) is undefined.


@pytest.fixture
def step_zero_root(tmp_path):
    """Build a generation-attention table that includes a step=0 row for both backends."""
    _write_generation(
        tmp_path,
        "vllm",
        "vllm_flash_attn_fa3",
        [
            (16, 0, 5.0e-5),      # step=0: non-interpolatable, must be silently skipped
            (16, 128, 6.0e-5),
            (16, 512, 8.0e-5),
            (16, 1024, 1.0e-4),
            (16, 2048, 1.4e-4),
        ],
    )
    return tmp_path


def test_query_attention_succeeds_with_step_zero_row_in_table(step_zero_root):
    # Must not raise; step=0 row is silently excluded from the grid.
    database = KernelDatabase.from_root(step_zero_root, gpu="h20")
    query = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=8, head_dim=128, step=512
    )
    latency = database.query_attention(query)
    assert latency > 0.0, f"Expected positive latency, got {latency!r}."


def test_utility_succeeds_with_step_zero_row_in_table(step_zero_root):
    # db.utility() must also not raise when the table contains a step=0 row.
    database = KernelDatabase.from_root(step_zero_root, gpu="h20")
    query = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=8, head_dim=128, step=512
    )
    utility = database.utility(query)
    assert utility > 0.0, f"Expected positive utility, got {utility!r}."


def test_query_at_exact_positive_step_matches_measured_latency(step_zero_root):
    # Querying at an exactly-measured positive step must return that row's latency.
    # This guards against the step=0 row silently corrupting the interpolation surface.
    database = KernelDatabase.from_root(step_zero_root, gpu="h20")
    query = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=8, head_dim=128, step=1024
    )
    latency = database.query_attention(query)
    assert latency == pytest.approx(1.0e-4)
