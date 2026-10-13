"""Parquet perf-table writing, path layout, and round-trip through KernelTable."""

from __future__ import annotations

from pathlib import Path

import pytest

from dryrun.cost.kernel.ops.attention import GENERATION_ATTENTION_OP
from dryrun.cost.kernel.ops.gemm import GEMM_OP
from dryrun.cost.kernel.table import KernelTable
from dryrun.profiler.kernel.measurements import PerfTableWriter


def _writer(tmp_path: Path, **overrides) -> PerfTableWriter:
    defaults = dict(
        output_root=tmp_path,
        device="NVIDIA H20",
        framework="vllm",
        version="0.26.1.dev0",
        kernel_source="vllm_flash_attn_fa3",
    )
    return PerfTableWriter(**{**defaults, **overrides})


def _gemm_row(m: int, latency_s: float) -> dict:
    return {"m": m, "n": 4096, "k": 4096, "gemm_dtype": "bfloat16", "latency_s": latency_s}


def test_table_paths_follow_the_layout(tmp_path):
    writer = _writer(tmp_path)
    assert writer.table_path("gemm") == (
        tmp_path / "h20" / "gemm" / "vllm" / "0.26.1.dev0" / "gemm_perf.parquet"
    )
    assert writer.table_path("generation_attention") == (
        tmp_path / "h20" / "attention" / "vllm" / "0.26.1.dev0" / "generation_attention_perf.parquet"
    )


def test_written_gemm_table_loads_through_kernel_table(tmp_path):
    writer = _writer(tmp_path, kernel_source="torch.nn.functional.linear")
    path = writer.write("gemm", [_gemm_row(1024, 1.5e-4), _gemm_row(2048, 3.0e-4)])
    table = KernelTable.load(path, GEMM_OP)
    assert table.row_count == 2
    # With per-(n,k) slicing the slice key is (dtype, n, k).
    assert table.slice_keys() == (("bfloat16", 4096, 4096),)
    assert table.kernel_sources() == ("torch.nn.functional.linear",)
    assert table.rows(("bfloat16", 4096, 4096))[0] == ((1024.0,), 1.5e-4)


def test_written_attention_table_loads_through_kernel_table(tmp_path):
    writer = _writer(tmp_path)
    path = writer.write(
        "generation_attention",
        [
            {
                "batch_size": 16,
                "num_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "step": 1024,
                "attn_dtype": "bfloat16",
                "kv_cache_dtype": "bfloat16",
                "block_size": 16,
                "window_size": 0,
                "latency_s": 8.0e-5,
            }
        ],
    )
    table = KernelTable.load(path, GENERATION_ATTENTION_OP)
    assert table.row_count == 1
    assert table.slice_keys() == ((8, 128, "bfloat16", "bfloat16", 16, 0),)
    assert table.rows((8, 128, "bfloat16", "bfloat16", 16, 0)) == (((32.0, 16.0, 1024.0), 8.0e-5),)


def test_writing_replaces_an_existing_table_and_leaves_no_temp_file(tmp_path):
    writer = _writer(tmp_path, kernel_source="torch.nn.functional.linear")
    writer.write("gemm", [_gemm_row(1024, 1.0e-4)])
    path = writer.write("gemm", [_gemm_row(1024, 2.0e-4), _gemm_row(2048, 4.0e-4)])
    assert KernelTable.load(path, GEMM_OP).row_count == 2
    assert not list(path.parent.glob("*.tmp"))


def test_writing_no_rows_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _writer(tmp_path).write("gemm", [])


def test_unknown_op_name_is_rejected(tmp_path):
    with pytest.raises(Exception):
        _writer(tmp_path).table_path("moe")


def test_attention_rows_are_rejected_against_the_gemm_op(tmp_path):
    """The columns come from the op, so a phase/op mismatch cannot reach parquet."""
    row = {
        "batch_size": 16,
        "num_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "step": 1024,
        "attn_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "block_size": 16,
        "window_size": 0,
        "latency_s": 8.0e-5,
    }
    with pytest.raises(ValueError, match="gemm_dtype"):
        _writer(tmp_path).write("gemm", [row])


def test_a_context_row_is_rejected_by_the_generation_op(tmp_path):
    """Context carries `isl` and generation carries `step`; one shared row type hid that."""
    row = {
        "batch_size": 4,
        "num_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "isl": 1024,
        "attn_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "block_size": 16,
        "window_size": 0,
        "latency_s": 5.0e-4,
    }
    with pytest.raises(ValueError, match="step"):
        _writer(tmp_path).write("generation_attention", [row])
