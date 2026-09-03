"""Perf-table parquet loading, slicing, and utility grid construction."""

from __future__ import annotations

import dataclasses
import gc
import weakref
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dryrun.cost.kernel.interp import KernelCoverageError
from dryrun.cost.kernel.ops.attention import CONTEXT_ATTENTION_OP, GENERATION_ATTENTION_OP
from dryrun.cost.kernel.ops.base import OP_REGISTRY
from dryrun.cost.kernel.ops.gemm import GEMM_OP
from dryrun.cost.kernel.table import KernelTable, required_columns


def _write_gemm(path: Path, rows: list[dict]) -> Path:
    columns = {
        "framework": [row.get("framework", "vllm") for row in rows],
        "version": [row.get("version", "0.26.1.dev0") for row in rows],
        "device": [row.get("device", "NVIDIA H20") for row in rows],
        "op_name": ["gemm" for _ in rows],
        "kernel_source": [row.get("kernel_source", "torch.nn.functional.linear") for row in rows],
        "gemm_dtype": [row.get("gemm_dtype", "bfloat16") for row in rows],
        "m": [row["m"] for row in rows],
        "n": [row["n"] for row in rows],
        "k": [row["k"] for row in rows],
        "latency_s": [row["latency_s"] for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return path


def test_schema_families_are_declared():
    assert GEMM_OP.family == "gemm"
    assert CONTEXT_ATTENTION_OP.family == "attention"
    assert GENERATION_ATTENTION_OP.family == "attention"
    assert GENERATION_ATTENTION_OP.axis_columns == ("num_heads", "batch_size", "step")
    assert CONTEXT_ATTENTION_OP.axis_columns == ("num_heads", "batch_size", "isl")


def test_required_columns_include_provenance_and_latency():
    assert "latency_s" in required_columns(GEMM_OP)
    assert "kernel_source" in required_columns(GEMM_OP)
    for column in GEMM_OP.axis_columns + GEMM_OP.slice_columns:
        assert column in required_columns(GEMM_OP)


def test_load_exposes_rows_and_slices(tmp_path):
    path = _write_gemm(
        tmp_path / "gemm_perf.parquet",
        [
            {"m": 1024, "n": 4096, "k": 4096, "latency_s": 3.0e-4},
            {"m": 2048, "n": 4096, "k": 4096, "latency_s": 6.0e-4},
        ],
    )
    table = KernelTable.load(path, GEMM_OP)
    assert table.row_count == 2
    # With per-(n,k) slicing the slice key is (dtype, n, k).
    assert table.slice_keys() == (("bfloat16", 4096, 4096),)
    assert table.kernel_sources() == ("torch.nn.functional.linear",)
    assert table.rows(("bfloat16", 4096, 4096))[0] == ((1024.0,), 3.0e-4)


def test_devices_returns_sorted_distinct_device_values(tmp_path):
    path = _write_gemm(
        tmp_path / "gemm_perf.parquet",
        [
            {"m": 1024, "n": 4096, "k": 4096, "latency_s": 3.0e-4, "device": "NVIDIA H20"},
            {"m": 2048, "n": 4096, "k": 4096, "latency_s": 6.0e-4, "device": "NVIDIA H20"},
        ],
    )
    table = KernelTable.load(path, GEMM_OP)
    assert table.devices() == ("NVIDIA H20",)


def test_distinct_dtypes_become_distinct_slices(tmp_path):
    path = _write_gemm(
        tmp_path / "gemm_perf.parquet",
        [
            {"m": 1024, "n": 4096, "k": 4096, "latency_s": 3.0e-4, "gemm_dtype": "bfloat16"},
            {"m": 1024, "n": 4096, "k": 4096, "latency_s": 1.5e-4, "gemm_dtype": "fp8"},
        ],
    )
    table = KernelTable.load(path, GEMM_OP)
    # Each (dtype, n, k) becomes a separate slice; two distinct dtypes, same (n, k).
    assert table.slice_keys() == (("bfloat16", 4096, 4096), ("fp8", 4096, 4096))
    assert len(table.rows(("fp8", 4096, 4096))) == 1


def test_grid_derives_utility_from_sol(tmp_path):
    path = _write_gemm(
        tmp_path / "gemm_perf.parquet",
        [
            {"m": 1024, "n": 4096, "k": 4096, "latency_s": 4.0e-4},
            {"m": 2048, "n": 4096, "k": 4096, "latency_s": 8.0e-4},
        ],
    )
    table = KernelTable.load(path, GEMM_OP)
    # A SOL of exactly half the measurement means a utility of 0.5 everywhere.
    # coords is now 1-D: (m,); the lambda scales SOL proportionally to m.
    grid = table.grid(("bfloat16", 4096, 4096), lambda coords: 2.0e-4 * (coords[0] / 1024.0))
    assert grid.sample_count == 2
    assert grid.utility((1024.0,)) == pytest.approx(0.5)
    assert grid.utility((1448.0,)) == pytest.approx(0.5)


def test_grid_is_memoized_per_slice(tmp_path):
    path = _write_gemm(
        tmp_path / "gemm_perf.parquet",
        [{"m": 1024, "n": 4096, "k": 4096, "latency_s": 4.0e-4}],
    )
    table = KernelTable.load(path, GEMM_OP)
    sol_fn = lambda coords: 2.0e-4  # noqa: E731
    assert table.grid(("bfloat16", 4096, 4096), sol_fn) is table.grid(("bfloat16", 4096, 4096), sol_fn)


def test_unknown_slice_key_is_rejected(tmp_path):
    path = _write_gemm(
        tmp_path / "gemm_perf.parquet",
        [{"m": 1024, "n": 4096, "k": 4096, "latency_s": 4.0e-4}],
    )
    table = KernelTable.load(path, GEMM_OP)
    with pytest.raises(KernelCoverageError):
        table.rows(("fp8", 4096, 4096))


def test_missing_column_is_rejected(tmp_path):
    path = tmp_path / "bad_perf.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"m": [1024], "latency_s": [1.0e-4]}), path)
    with pytest.raises(KernelCoverageError):
        KernelTable.load(path, GEMM_OP)


def test_nonpositive_latency_rows_are_rejected(tmp_path):
    path = _write_gemm(
        tmp_path / "gemm_perf.parquet",
        [{"m": 1024, "n": 4096, "k": 4096, "latency_s": 0.0}],
    )
    with pytest.raises(KernelCoverageError):
        KernelTable.load(path, GEMM_OP)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(KernelCoverageError):
        KernelTable.load(tmp_path / "absent.parquet", GEMM_OP)


def test_duplicate_shapes_keep_the_fastest_measurement(tmp_path):
    path = _write_gemm(
        tmp_path / "gemm_perf.parquet",
        [
            {"m": 1024, "n": 4096, "k": 4096, "latency_s": 5.0e-4},
            {"m": 1024, "n": 4096, "k": 4096, "latency_s": 4.0e-4},
        ],
    )
    table = KernelTable.load(path, GEMM_OP)
    assert table.rows(("bfloat16", 4096, 4096)) == (((1024.0,), 4.0e-4),)


@pytest.mark.parametrize("op_name", sorted(OP_REGISTRY))
def test_every_registered_op_is_frozen(op_name):
    """Each op instance is a registry singleton, so a mutation anywhere changes every reader.

    Parametrized off the registry rather than a name list so a sixth op cannot silently drop
    the decorator: adding one to `ops/` adds a case here with no edit.
    """
    op = OP_REGISTRY[op_name]
    with pytest.raises(dataclasses.FrozenInstanceError):
        op.slice_columns = ("oops",)  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        op.fits_floor = not op.fits_floor  # type: ignore[misc]


def test_grid_does_not_alias_distinct_sol_functions(tmp_path):
    path = _write_gemm(
        tmp_path / "gemm_perf.parquet",
        [
            {"m": 1024, "n": 4096, "k": 4096, "latency_s": 1.0e-4},
            {"m": 2048, "n": 4096, "k": 4096, "latency_s": 1.0e-4},
        ],
    )
    table = KernelTable.load(path, GEMM_OP)
    slice_key = ("bfloat16", 4096, 4096)

    expected_row_count = len(table.rows(slice_key))

    # Two distinct callables that record every coordinate they price.
    sol_a_coords: list[tuple[float, ...]] = []
    sol_b_coords: list[tuple[float, ...]] = []

    def sol_a(coords: tuple[float, ...]) -> float:
        sol_a_coords.append(coords)
        return 1.0e-4

    def sol_b(coords: tuple[float, ...]) -> float:
        sol_b_coords.append(coords)
        return 9.0e-4

    g1 = table.grid(slice_key, sol_a)
    g2 = table.grid(slice_key, sol_b)

    assert g1 is not g2, "Distinct callables must produce distinct grids."

    # Verify the grids compute expected utilities.
    assert g1.utility((1024.0,)) == pytest.approx(1.0)
    assert g2.utility((1024.0,)) == pytest.approx(9.0)

    # Each callable must have priced every row of its own grid. An implementation that
    # aliased the two would serve `g2` from the cache without ever invoking `sol_b`,
    # leaving this recorder empty.
    assert len(sol_a_coords) == expected_row_count, (
        f"sol_a should be called once per row ({expected_row_count}), "
        f"got {len(sol_a_coords)} calls."
    )
    assert len(sol_b_coords) == expected_row_count, (
        f"sol_b should be called once per row ({expected_row_count}), "
        f"got {len(sol_b_coords)} calls."
    )


def test_grid_cache_keeps_its_sol_function_alive(tmp_path):
    """
    Pin the invariant that makes address reuse impossible.

    A cache keyed on `id(sol_fn)` does not keep the callable alive, so CPython can
    recycle a collected function's address and hand a later, different callable the
    earlier grid. Keying on the callable itself makes the cache hold a strong reference,
    which is the property this asserts directly rather than trying to provoke the
    allocator into reusing an address.
    """
    path = _write_gemm(
        tmp_path / "gemm_perf.parquet",
        [{"m": 1024, "n": 4096, "k": 4096, "latency_s": 1.0e-4}],
    )
    table = KernelTable.load(path, GEMM_OP)

    def sol(coords: tuple[float, ...]) -> float:
        return 1.0e-4

    reference = weakref.ref(sol)
    table.grid(("bfloat16", 4096, 4096), sol)
    del sol
    gc.collect()

    assert reference() is not None, (
        "The grid cache must hold a strong reference to its SOL function, otherwise the "
        "function's address can be recycled and a different callable can collide with it."
    )


def test_non_finite_sol_raises(tmp_path):
    path = _write_gemm(
        tmp_path / "gemm_perf.parquet",
        [{"m": 1024, "n": 4096, "k": 4096, "latency_s": 1.0e-4}],
    )
    table = KernelTable.load(path, GEMM_OP)
    with pytest.raises(KernelCoverageError):
        table.grid(("bfloat16", 4096, 4096), lambda c: float("nan"))


def _write_generation_attn(path: Path, rows: list[dict]) -> Path:
    """Write a generation_attention parquet for table tests."""
    columns = {
        "framework": [row.get("framework", "vllm") for row in rows],
        "version": [row.get("version", "0.26.1.dev0") for row in rows],
        "device": [row.get("device", "NVIDIA H20") for row in rows],
        "op_name": ["generation_attention" for _ in rows],
        "kernel_source": [row.get("kernel_source", "vllm_flash_attn_fa3") for row in rows],
        "num_key_value_heads": [row.get("num_key_value_heads", 8) for row in rows],
        "head_dim": [row.get("head_dim", 128) for row in rows],
        "attn_dtype": [row.get("attn_dtype", "bfloat16") for row in rows],
        "kv_cache_dtype": [row.get("kv_cache_dtype", "bfloat16") for row in rows],
        "block_size": [row.get("block_size", 16) for row in rows],
        "window_size": [row.get("window_size", 0) for row in rows],
        "num_heads": [row.get("num_heads", 32) for row in rows],
        "batch_size": [row["batch_size"] for row in rows],
        "step": [row["step"] for row in rows],
        "latency_s": [row["latency_s"] for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return path


def test_grid_skips_non_positive_coordinate_rows_and_counts_the_rest(tmp_path):
    # A generation-attention table with a step=0 row must not crash grid(); the step=0
    # row is excluded from the UtilityGrid while all positive-step rows are included.
    path = _write_generation_attn(
        tmp_path / "generation_attention_perf.parquet",
        [
            {"batch_size": 16, "step": 0, "latency_s": 5.0e-5},   # non-positive — skipped
            {"batch_size": 16, "step": 128, "latency_s": 6.0e-5},
            {"batch_size": 16, "step": 512, "latency_s": 8.0e-5},
            {"batch_size": 16, "step": 1024, "latency_s": 1.0e-4},
        ],
    )
    table = KernelTable.load(path, GENERATION_ATTENTION_OP)
    slice_key = (8, 128, "bfloat16", "bfloat16", 16, 0)
    grid = table.grid(slice_key, lambda coords: 3.0e-5)
    # Only the three positive-step rows are placed in the grid; step=0 is excluded.
    assert grid.sample_count == 3
