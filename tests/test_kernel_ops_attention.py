"""The two attention ops: shared slicing, distinct third axis, fitted floor."""

from __future__ import annotations

import pytest

from srl.cost.kernel.hardware import load_gpu_spec
from srl.cost.kernel.ops.attention import CONTEXT_ATTENTION_OP, GENERATION_ATTENTION_OP
from srl.cost.kernel.ops.base import op_for
from srl.cost.kernel.sol import context_attention_sol, generation_attention_sol

# (num_kv_heads, head_dim, attn_dtype, kv_cache_dtype, block_size, window_size)
_SLICE = (8, 128, "bfloat16", "bfloat16", 16, 0)


@pytest.fixture(scope="module")
def spec():
    return load_gpu_spec("h20")


def test_both_ops_are_registered_under_their_table_names():
    assert op_for("context_attention") is CONTEXT_ATTENTION_OP
    assert op_for("generation_attention") is GENERATION_ATTENTION_OP


def test_both_phases_share_one_family_and_one_slice_schema():
    """They live in one directory and one floors.json, so their slicing must agree."""
    assert CONTEXT_ATTENTION_OP.family == GENERATION_ATTENTION_OP.family == "attention"
    assert CONTEXT_ATTENTION_OP.slice_columns == GENERATION_ATTENTION_OP.slice_columns
    assert CONTEXT_ATTENTION_OP.slice_columns == (
        "num_key_value_heads",
        "head_dim",
        "attn_dtype",
        "kv_cache_dtype",
        "block_size",
        "window_size",
    )


def test_the_phases_differ_only_in_their_third_axis():
    assert CONTEXT_ATTENTION_OP.axis_columns == ("num_heads", "batch_size", "isl")
    assert GENERATION_ATTENTION_OP.axis_columns == ("num_heads", "batch_size", "step")


def test_both_phases_take_a_fitted_floor():
    """Decode latency is flat across a 64x KV increase at batch 1, so dispatch dominates."""
    assert CONTEXT_ATTENTION_OP.fits_floor is True
    assert GENERATION_ATTENTION_OP.fits_floor is True


def test_context_sol_reproduces_the_free_function(spec):
    coords = (32.0, 4.0, 1024.0)
    expected = context_attention_sol(
        spec,
        batch_size=4,
        seq_len=1024,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        window_size=0,
        floor_s=32e-6,
    )
    got = CONTEXT_ATTENTION_OP.sol(spec, slice_key=_SLICE, coords=coords, floor_s=32e-6)
    assert got.latency_s == pytest.approx(expected.latency_s, rel=1e-12)
    assert got.floor_s == pytest.approx(32e-6, rel=1e-12)


def test_generation_sol_reproduces_the_free_function(spec):
    coords = (32.0, 16.0, 2048.0)
    expected = generation_attention_sol(
        spec,
        batch_size=16,
        step=2048,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        window_size=0,
        floor_s=22e-6,
    )
    got = GENERATION_ATTENTION_OP.sol(spec, slice_key=_SLICE, coords=coords, floor_s=22e-6)
    assert got.latency_s == pytest.approx(expected.latency_s, rel=1e-12)


def test_the_floor_binds_when_the_work_is_small(spec):
    """A floor inside max() is a lower bound; at batch 1 step 1 it must be what binds."""
    tiny = GENERATION_ATTENTION_OP.sol(spec, slice_key=_SLICE, coords=(32.0, 1.0, 1.0), floor_s=22e-6)
    assert tiny.latency_s == pytest.approx(22e-6, rel=1e-12)
    assert tiny.math_s < 22e-6
    assert tiny.memory_s < 22e-6


def test_the_floor_does_not_bind_when_the_work_is_large(spec):
    """It is max(floor, work), not floor + work; deep KV must not pay the floor twice."""
    big = GENERATION_ATTENTION_OP.sol(spec, slice_key=_SLICE, coords=(32.0, 64.0, 32768.0), floor_s=22e-6)
    assert big.latency_s > 22e-6
    assert big.latency_s == pytest.approx(max(big.math_s, big.memory_s), rel=1e-12)


def test_both_ops_put_latency_last_in_their_columns():
    for op, axis in ((CONTEXT_ATTENTION_OP, "isl"), (GENERATION_ATTENTION_OP, "step")):
        assert op.table_columns() == (
            "num_key_value_heads",
            "head_dim",
            "attn_dtype",
            "kv_cache_dtype",
            "block_size",
            "window_size",
            "num_heads",
            "batch_size",
            axis,
            "latency_s",
        )


def test_parse_sweep_builds_the_attention_grid():
    sweep = CONTEXT_ATTENTION_OP.parse_sweep(
        {"batch_sizes": [1, 16], "context_seq_lens": [128, 1024], "generation_steps": [0, 2048]}
    )
    assert sweep.batch_sizes == (1, 16)
    assert sweep.context_seq_lens == (128, 1024)
    assert sweep.generation_steps == (0, 2048)


def test_both_phases_parse_the_same_sweep_subtree():
    """One `attention:` block configures both phases, so both must accept it."""
    payload = {"batch_sizes": [1], "context_seq_lens": [128], "generation_steps": [0]}
    assert CONTEXT_ATTENTION_OP.parse_sweep(payload) == GENERATION_ATTENTION_OP.parse_sweep(payload)


def test_a_negative_generation_step_is_rejected(spec):
    with pytest.raises(ValueError, match="step"):
        GENERATION_ATTENTION_OP.sol(spec, slice_key=_SLICE, coords=(32.0, 1.0, -1.0), floor_s=0.0)
