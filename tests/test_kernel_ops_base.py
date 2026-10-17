"""The op registry and its four lookups."""

from __future__ import annotations

from pathlib import Path

import pytest

from srl.cost.kernel.interp import KernelCoverageError
from srl.cost.kernel.ops.base import (
    OP_REGISTRY,
    families,
    op_for,
    ops_for_family,
    ops_in,
    register_op,
)
from srl.cost.kernel.sol import SolComponents


class _FakeOp:
    """A minimal op used to exercise the registry without touching real tables."""

    def __init__(self, name: str, family: str, fits_floor: bool = False) -> None:
        self.name = name
        self.family = family
        self.slice_columns = ("dtype",)
        self.axis_columns = ("n_elements",)
        self.fits_floor = fits_floor

    def sol(self, spec, *, slice_key, coords, floor_s=0.0) -> SolComponents:
        return SolComponents(math_s=0.0, memory_s=float(coords[0]), floor_s=floor_s)

    def parse_sweep(self, value: dict) -> object:
        return value

    def table_columns(self) -> tuple[str, ...]:
        return self.slice_columns + self.axis_columns + ("latency_s",)


@pytest.fixture
def isolated_registry():
    """Restore the real registry after a test mutates it."""
    saved = dict(OP_REGISTRY)
    OP_REGISTRY.clear()
    yield OP_REGISTRY
    OP_REGISTRY.clear()
    OP_REGISTRY.update(saved)


def test_a_registered_op_is_retrievable_by_name(isolated_registry):
    op = _FakeOp("widget", "widgets")
    register_op(op)
    assert op_for("widget") is op


def test_an_unknown_op_name_is_rejected_naming_the_known_ones(isolated_registry):
    register_op(_FakeOp("widget", "widgets"))
    with pytest.raises(KernelCoverageError, match="widget"):
        op_for("wodget")


def test_registering_the_same_name_twice_is_rejected(isolated_registry):
    """A duplicate name would make op_for's answer depend on import order."""
    register_op(_FakeOp("widget", "widgets"))
    with pytest.raises(KernelCoverageError, match="already registered"):
        register_op(_FakeOp("widget", "other"))


def test_ops_are_grouped_by_family_and_sorted(isolated_registry):
    register_op(_FakeOp("beta", "shared"))
    register_op(_FakeOp("alpha", "shared"))
    register_op(_FakeOp("solo", "lonely"))
    assert [op.name for op in ops_for_family("shared")] == ["alpha", "beta"]
    assert [op.name for op in ops_for_family("lonely")] == ["solo"]


def test_an_unmeasured_family_yields_no_ops(isolated_registry):
    register_op(_FakeOp("widget", "widgets"))
    assert ops_for_family("absent") == ()


def test_families_are_distinct_and_sorted(isolated_registry):
    register_op(_FakeOp("beta", "shared"))
    register_op(_FakeOp("alpha", "shared"))
    register_op(_FakeOp("solo", "lonely"))
    assert families() == ("lonely", "shared")


def test_ops_in_finds_only_the_parquet_files_present(isolated_registry, tmp_path):
    register_op(_FakeOp("present", "f"))
    register_op(_FakeOp("absent", "f"))
    (tmp_path / "present_perf.parquet").write_bytes(b"")
    assert [op.name for op in ops_in(tmp_path)] == ["present"]


def test_ops_in_on_a_missing_directory_yields_nothing(isolated_registry, tmp_path):
    register_op(_FakeOp("present", "f"))
    assert ops_in(tmp_path / "nope") == ()


def test_ops_in_ignores_a_parquet_with_no_registered_op(isolated_registry, tmp_path):
    """A stray table must not crash discovery; the coverage test in Task 9 catches real gaps."""
    register_op(_FakeOp("known", "f"))
    (tmp_path / "known_perf.parquet").write_bytes(b"")
    (tmp_path / "mystery_perf.parquet").write_bytes(b"")
    assert [op.name for op in ops_in(tmp_path)] == ["known"]


def test_table_columns_puts_latency_last(isolated_registry):
    """PerfTableWriter builds its payload from this order, so it is part of the contract."""
    op = _FakeOp("widget", "widgets")
    assert op.table_columns() == ("dtype", "n_elements", "latency_s")


def test_the_registry_import_does_not_load_torch():
    """The cost path must not pay a 4.2 s GPU-stack import to price a kernel."""
    import subprocess
    import sys

    probe = (
        "import sys; import srl.cost.kernel.ops.base; "
        "assert 'torch' not in sys.modules, 'ops.base pulled in torch'; "
        "assert 'vllm' not in sys.modules, 'ops.base pulled in vllm'; print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_the_table_module_no_longer_exports_a_schema_type():
    """TableSchema is absorbed into KernelOp; leaving an alias invites two sources of truth."""
    import srl.cost.kernel.table as table_module

    for name in (
        "TableSchema",
        "schema_for",
        "GEMM_SCHEMA",
        "CONTEXT_ATTENTION_SCHEMA",
        "GENERATION_ATTENTION_SCHEMA",
    ):
        assert not hasattr(table_module, name), f"{name} should be gone from table.py"


def test_a_table_loads_against_an_op_and_exposes_it():
    """KernelTable's second parameter is the op, and the op is readable back off the table."""
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    from srl.cost.kernel.ops.gemm import GEMM_OP
    from srl.cost.kernel.table import KernelTable

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gemm_perf.parquet"
        pq.write_table(
            pa.table(
                {
                    "framework": ["vllm"],
                    "version": ["0.0.0"],
                    "device": ["NVIDIA H20"],
                    "op_name": ["gemm"],
                    "kernel_source": ["torch.nn.functional.linear"],
                    "gemm_dtype": ["bfloat16"],
                    "m": [64],
                    "n": [4096],
                    "k": [4096],
                    "latency_s": [1.0e-4],
                }
            ),
            path,
        )
        table = KernelTable.load(path, GEMM_OP)
        assert table.op is GEMM_OP
        assert table.row_count == 1


def test_the_config_serves_a_sweep_per_op_name():
    """Adding an op must not require adding a named field to KernelProfileConfig."""
    import os
    from pathlib import Path as _Path

    from srl.profiler.kernel.config import KernelProfileConfig

    config_path = _Path(__file__).resolve().parents[1] / "examples" / "profile" / "kernel" / "qwen3_8b_h20_tp1.yaml"
    if not config_path.is_file():
        pytest.skip("the example kernel config is machine-local")
    # `from_dict` reads the checkpoint's architecture, and the example config takes that
    # checkpoint from `MODEL_PATH`. The assertions below are about the sweep rather than
    # the model, but the construction that reaches them is not.
    model_path = os.environ.get("MODEL_PATH", "")
    if not model_path or not (_Path(model_path) / "config.json").is_file():
        pytest.skip("MODEL_PATH is not set to a local checkpoint")

    # Load through the CLI's loader, not `yaml.safe_load`: the example configs read
    # their checkpoint from `${oc.env:MODEL_PATH,...}`, which only a resolving loader
    # turns into a path.
    from srl.cli.profile import _load_config

    payload, base_dir = _load_config(config_path)
    config = KernelProfileConfig.from_dict(payload, base_dir=base_dir)
    assert config.sweep("gemm").token_counts[0] == 1
    assert config.sweep("context_attention").batch_sizes[0] == 1
    # Both attention phases read one `attention:` subtree.
    assert config.sweep("context_attention") == config.sweep("generation_attention")
    assert not hasattr(config, "gemm")
    assert not hasattr(config, "attention")
