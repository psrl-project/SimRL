"""Routed mixture-of-experts: the grouped-GEMM kernel op `fused_experts()` measures directly."""

from __future__ import annotations

import pytest

from srl.cost.kernel.database import MoeQuery
from srl.cost.kernel.hardware import load_gpu_spec
from srl.cost.kernel.ops.base import families, op_for, ops_for_family
from srl.cost.kernel.ops.moe import MOE_OP
from srl.cost.kernel.sol import FALLBACK_FLOORS_S, moe_sol


def test_moe_is_registered_as_its_own_family():
    """Not a slice of `gemm`: a real measured kernel, not a dense-GEMM lookup."""
    assert "moe" in families()
    assert ops_for_family("moe") == (MOE_OP,)
    assert op_for("moe") is MOE_OP
    assert MOE_OP.family == "moe"


def test_moe_has_one_live_axis_unlike_gdns_two_tables():
    """No context/decode split: `fused_experts_impl` is the same kernel at every token count."""
    assert MOE_OP.axis_columns == ("num_tokens",)
    assert MOE_OP.slice_columns == (
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
        "moe_dtype",
    )
    assert MOE_OP.table_columns() == (
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
        "moe_dtype",
        "num_tokens",
        "latency_s",
    )


def test_moe_has_a_fallback_floor():
    """A missing `floors.json` must not price MoE at a zero floor."""
    assert MOE_OP.fits_floor is True
    assert FALLBACK_FLOORS_S["vllm"]["moe"] > 0


def test_moe_query_slice_key_matches_the_ops_declared_column_order():
    """A transposition here is silent and catastrophic: `table.py` zips positionally."""
    query = MoeQuery(
        num_tokens=64, hidden_size=111, intermediate_size=222, num_experts=333, top_k=4
    )
    assert query.slice_key == (111, 222, 333, 4, "bfloat16")
    assert query.coords == (64.0,)


def test_moe_query_refuses_a_non_positive_field():
    with pytest.raises(Exception, match="positive"):
        MoeQuery(num_tokens=0, hidden_size=1, intermediate_size=1, num_experts=1, top_k=1)


def test_moe_compute_is_linear_in_tokens_and_top_k():
    """Doubling the token count doubles the row count the grouped GEMM processes."""
    spec = load_gpu_spec("h20")
    kwargs = dict(hidden_size=4096, intermediate_size=1024, num_experts=8, top_k=2, dtype="bfloat16")
    small = moe_sol(spec, num_tokens=1024, **kwargs)
    big = moe_sol(spec, num_tokens=2048, **kwargs)
    assert big.math_s == pytest.approx(2 * small.math_s, rel=1e-9)


def test_moe_weight_read_saturates_well_before_num_tokens_does():
    """The activated-expert count saturates at `num_experts`; a naive per-token weight read
    that scaled linearly with `num_tokens` instead would be a 1024x memory-traffic error here.
    """
    spec = load_gpu_spec("h20")
    kwargs = dict(hidden_size=2048, intermediate_size=512, num_experts=256, top_k=8, dtype="bfloat16")
    few = moe_sol(spec, num_tokens=4, **kwargs)
    many = moe_sol(spec, num_tokens=4096, **kwargs)
    assert many.memory_s < 50 * few.memory_s


def test_moe_refuses_a_non_positive_dimension():
    spec = load_gpu_spec("h20")
    with pytest.raises(ValueError, match="positive"):
        moe_sol(
            spec, num_tokens=0, hidden_size=1, intermediate_size=1, num_experts=1, top_k=1,
            dtype="bfloat16",
        )
