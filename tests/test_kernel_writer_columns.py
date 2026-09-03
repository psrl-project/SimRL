"""One write method, validated against the op's own column declaration."""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from dryrun.profiler.kernels.measurements import PerfTableWriter


@pytest.fixture
def writer(tmp_path):
    return PerfTableWriter(
        output_root=tmp_path,
        device="NVIDIA H20",
        framework="vllm",
        version="0.0.0",
        kernel_source="torch.nn.functional.linear",
    )


def _gemm_row(m: int) -> dict:
    return {"m": m, "n": 4096, "k": 4096, "gemm_dtype": "bfloat16", "latency_s": 1.0e-4}


def test_a_gemm_table_is_written_with_provenance_and_payload(writer):
    path = writer.write("gemm", [_gemm_row(1), _gemm_row(16)])
    table = pq.read_table(path)
    assert set(table.column_names) >= {
        "framework",
        "version",
        "device",
        "op_name",
        "kernel_source",
        "gemm_dtype",
        "n",
        "k",
        "m",
        "latency_s",
    }
    assert table.num_rows == 2
    assert set(table.column("op_name").to_pylist()) == {"gemm"}


def test_the_written_table_loads_back_through_its_op(writer):
    """A table the writer produced must satisfy the reader's contract by construction."""
    from dryrun.cost.kernel.ops.base import op_for
    from dryrun.cost.kernel.table import KernelTable

    path = writer.write("gemm", [_gemm_row(1), _gemm_row(16)])
    table = KernelTable.load(path, op_for("gemm"))
    assert table.row_count == 2


def test_a_missing_column_is_rejected_at_write_time(writer):
    """write_attention trusted its caller for eleven columns; a typo surfaced only on read."""
    incomplete = dict(_gemm_row(1))
    del incomplete["k"]
    with pytest.raises(ValueError, match="k"):
        writer.write("gemm", [incomplete])


def test_an_extra_column_is_rejected_at_write_time(writer):
    """An unexpected column means the caller and the op disagree about the schema."""
    extra = dict(_gemm_row(1))
    extra["batch_size"] = 8
    with pytest.raises(ValueError, match="batch_size"):
        writer.write("gemm", [extra])


def test_an_empty_row_list_is_rejected(writer):
    with pytest.raises(ValueError, match="no rows"):
        writer.write("gemm", [])


def test_an_unknown_op_is_rejected(writer):
    from dryrun.cost.kernel.interp import KernelCoverageError

    with pytest.raises(KernelCoverageError, match="wodget"):
        writer.write("wodget", [{"latency_s": 1.0}])


def test_a_ragged_row_set_is_rejected(writer):
    """Every row must carry the same keys, or the parquet columns misalign silently."""
    with pytest.raises(ValueError):
        writer.write("gemm", [_gemm_row(1), {"m": 2, "latency_s": 1.0e-4}])


def test_the_path_uses_the_ops_family_directory(writer, tmp_path):
    path = writer.write("gemm", [_gemm_row(1)])
    assert path == tmp_path / "h20" / "gemm" / "vllm" / "0.0.0" / "gemm_perf.parquet"


def test_an_attention_table_uses_the_attention_family_directory(writer, tmp_path):
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
    path = writer.write("context_attention", [row])
    assert path == tmp_path / "h20" / "attention" / "vllm" / "0.0.0" / "context_attention_perf.parquet"


def test_the_per_family_write_methods_are_gone():
    """Keeping them would leave two ways to write a table, one of them unvalidated."""
    assert not hasattr(PerfTableWriter, "write_gemm")
    assert not hasattr(PerfTableWriter, "write_attention")
