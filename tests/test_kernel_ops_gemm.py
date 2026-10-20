"""GemmKernelOp: its schema, its roofline, and its lack of a floor."""

from __future__ import annotations

import pytest

from srl.cost.kernel.hardware import load_gpu_spec
from srl.cost.kernel.ops.base import op_for
from srl.cost.kernel.ops.gemm import GEMM_OP
from srl.cost.kernel.sol import gemm_sol


@pytest.fixture(scope="module")
def spec():
    return load_gpu_spec("h20")


def test_the_op_is_registered_under_its_table_name():
    assert op_for("gemm") is GEMM_OP


def test_the_schema_matches_the_measured_parquet_columns():
    """These names are what PerfTableWriter writes and KernelTable reads."""
    assert GEMM_OP.slice_columns == ("gemm_dtype", "n", "k")
    assert GEMM_OP.axis_columns == ("m",)
    assert GEMM_OP.family == "gemm"


def test_latency_is_the_last_table_column():
    assert GEMM_OP.table_columns() == ("gemm_dtype", "n", "k", "m", "latency_s")


def test_gemm_takes_no_fitted_floor():
    """GEMM is compute- or bandwidth-limited, not dispatch-limited. Its MRE is 3.91%."""
    assert GEMM_OP.fits_floor is False


def test_sol_reproduces_the_free_function_it_delegates_to(spec):
    """The op must not re-derive the roofline. It binds the existing formula to a slice."""
    slice_key = ("bfloat16", 24576, 4096)
    coords = (128.0,)
    expected = gemm_sol(spec, m=128, n=24576, k=4096, dtype="bfloat16")
    got = GEMM_OP.sol(spec, slice_key=slice_key, coords=coords)
    assert got.math_s == pytest.approx(expected.math_s, rel=1e-12)
    assert got.memory_s == pytest.approx(expected.memory_s, rel=1e-12)
    assert got.latency_s == pytest.approx(expected.latency_s, rel=1e-12)


def test_a_floor_argument_is_ignored_rather_than_applied(spec):
    """fits_floor is False, so passing one must not change the answer."""
    slice_key = ("bfloat16", 4096, 4096)
    coords = (1.0,)
    without = GEMM_OP.sol(spec, slice_key=slice_key, coords=coords).latency_s
    with_floor = GEMM_OP.sol(spec, slice_key=slice_key, coords=coords, floor_s=1.0).latency_s
    assert with_floor == pytest.approx(without, rel=1e-12)


def test_sol_reads_n_and_k_from_the_slice_not_the_coords(spec):
    """m is the only axis. n and k are categorical, so they must come from slice_key."""
    coords = (64.0,)
    small = GEMM_OP.sol(spec, slice_key=("bfloat16", 4096, 4096), coords=coords).latency_s
    large = GEMM_OP.sol(spec, slice_key=("bfloat16", 24576, 4096), coords=coords).latency_s
    assert large > small


def test_a_non_positive_m_is_rejected(spec):
    with pytest.raises(ValueError, match="m"):
        GEMM_OP.sol(spec, slice_key=("bfloat16", 4096, 4096), coords=(0.0,))


def test_parse_sweep_builds_the_gemm_grid():
    sweep = GEMM_OP.parse_sweep({"token_counts": [1, 16, 4096]})
    assert sweep.token_counts == (1, 16, 4096)
    assert sweep.logits_request_counts == (1, 2, 4, 8, 16, 32, 48, 64)


def test_parse_sweep_rejects_an_empty_grid():
    from srl.profiler.schema import ProfileValidationError

    with pytest.raises(ProfileValidationError, match="token_counts"):
        GEMM_OP.parse_sweep({"token_counts": []})
