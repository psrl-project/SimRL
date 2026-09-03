"""Speed-of-light formulas checked against hand-computed H20 values."""

from __future__ import annotations

import pytest

from dryrun.cost.kernel.hardware import load_gpu_spec
from dryrun.cost.kernel.sol import (
    FALLBACK_FLOORS_S,
    SolComponents,
    context_attention_sol,
    gemm_sol,
    generation_attention_sol,
)

_H20_FLOPS = 148e12
_H20_BW = 4.0e12


@pytest.fixture
def spec():
    return load_gpu_spec("h20")


def test_gemm_sol_is_compute_bound_for_a_square_shape(spec):
    result = gemm_sol(spec, m=1024, n=4096, k=4096, dtype="bfloat16")
    assert result.math_s == pytest.approx(2 * 1024 * 4096 * 4096 / _H20_FLOPS)
    assert result.memory_s == pytest.approx(
        2 * (1024 * 4096 + 4096 * 4096 + 1024 * 4096) / _H20_BW
    )
    assert result.latency_s == pytest.approx(result.math_s)
    assert result.latency_s == pytest.approx(2.321604e-04, rel=1e-4)


def test_gemm_sol_is_memory_bound_for_a_skinny_shape(spec):
    result = gemm_sol(spec, m=1, n=4096, k=4096, dtype="bfloat16")
    assert result.latency_s == pytest.approx(result.memory_s)
    assert result.memory_s > result.math_s


def test_gemm_sol_scales_linearly_in_m(spec):
    small = gemm_sol(spec, m=1024, n=4096, k=4096, dtype="bfloat16").math_s
    large = gemm_sol(spec, m=2048, n=4096, k=4096, dtype="bfloat16").math_s
    assert large == pytest.approx(2 * small)


def test_context_attention_sol_matches_hand_computation(spec):
    result = context_attention_sol(
        spec,
        batch_size=1,
        seq_len=1024,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
    )
    expected_math = 2 * 2 * 1 * 32 * 1024 * 1024 * 128 * 0.5 / _H20_FLOPS
    expected_memory = (
        2 * (1 * 1024 * 32 * 128) + 2 * (2 * 1 * 1024 * 8 * 128) + 2 * (1 * 1024 * 32 * 128)
    ) / _H20_BW
    assert result.math_s == pytest.approx(expected_math)
    assert result.memory_s == pytest.approx(expected_memory)
    assert result.latency_s == pytest.approx(5.804010e-05, rel=1e-4)


def test_context_attention_sol_is_quadratic_in_seq_len(spec):
    kwargs = dict(
        batch_size=1,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
    )
    small = context_attention_sol(spec, seq_len=1024, **kwargs).math_s
    large = context_attention_sol(spec, seq_len=2048, **kwargs).math_s
    assert large == pytest.approx(4 * small)


def test_sliding_window_caps_the_context_quadratic(spec):
    kwargs = dict(
        batch_size=1,
        seq_len=8192,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
    )
    full = context_attention_sol(spec, window_size=0, **kwargs).math_s
    windowed = context_attention_sol(spec, window_size=1024, **kwargs).math_s
    # With seq_len=8192, window_size=1024, attended is min(8192, 1024) = 1024.
    # With window_size=0, attended is 8192 * 0.5 = 4096.
    # math_s is linear in attended, so ratio is 1024 / 4096 = 0.25.
    expected_ratio = 1024.0 / (8192.0 * 0.5)
    assert windowed == pytest.approx(full * expected_ratio, rel=1e-9)


def test_a_window_wider_than_the_sequence_doubles_the_causal_estimate(spec):
    # Trap: window_size=0 requests full causal attention (2x saving via triangular).
    # Passing window_size >= seq_len silently yields a 100% overestimate.
    kwargs = dict(
        batch_size=1,
        seq_len=512,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
    )
    full_causal = context_attention_sol(spec, window_size=0, **kwargs).math_s
    wide_window = context_attention_sol(spec, window_size=8192, **kwargs).math_s
    assert wide_window == pytest.approx(2.0 * full_causal)


def test_generation_attention_sol_is_bandwidth_bound(spec):
    result = generation_attention_sol(
        spec,
        batch_size=16,
        step=1024,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
    )
    assert result.memory_s > result.math_s
    assert result.math_s == pytest.approx(2 * 2 * 16 * 32 * 1024 * 128 / _H20_FLOPS)
    assert result.latency_s == pytest.approx(result.memory_s)


def test_generation_attention_sol_includes_the_new_kv_write(spec):
    kwargs = dict(
        batch_size=16,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
    )
    at_zero = generation_attention_sol(spec, step=0, **kwargs)
    assert at_zero.memory_s > 0.0, "A step-0 decode still writes one KV entry per request."


def test_generation_attention_sol_grows_linearly_in_step(spec):
    kwargs = dict(
        batch_size=16,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
    )
    small = generation_attention_sol(spec, step=1024, **kwargs).memory_s
    large = generation_attention_sol(spec, step=2048, **kwargs).memory_s
    assert 1.9 < large / small < 2.0


def test_fp8_kv_cache_halves_the_history_read(spec):
    kwargs = dict(
        batch_size=16,
        step=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
    )
    bf16 = generation_attention_sol(spec, kv_cache_dtype="bfloat16", **kwargs).memory_s
    fp8 = generation_attention_sol(spec, kv_cache_dtype="fp8", **kwargs).memory_s
    assert fp8 < bf16
    assert 1.9 < bf16 / fp8 < 2.1


def test_nonpositive_shapes_are_rejected(spec):
    with pytest.raises(ValueError):
        gemm_sol(spec, m=0, n=4096, k=4096, dtype="bfloat16")


# --- Floor term (dispatch floor) ---


def test_sol_components_has_three_fields():
    # SolComponents must carry floor_s so the holdout and query paths can agree.
    c = SolComponents(math_s=1.0, memory_s=2.0, floor_s=3.0)
    assert c.math_s == 1.0
    assert c.memory_s == 2.0
    assert c.floor_s == 3.0


def test_floor_zero_reproduces_original_latency_s(spec):
    # Default floor_s=0.0 must leave latency_s unchanged — the default path is provably unaffected.
    result_no_floor = generation_attention_sol(
        spec,
        batch_size=16,
        step=1024,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
    )
    result_floor_zero = generation_attention_sol(
        spec,
        batch_size=16,
        step=1024,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        floor_s=0.0,
    )
    assert result_no_floor.latency_s == pytest.approx(result_floor_zero.latency_s)


def test_floor_binds_when_work_is_negligible(spec):
    # batch=1, step=1: both math_s and memory_s are far below a 100 us floor.
    # latency_s must equal the floor.
    floor = 100e-6
    result = generation_attention_sol(
        spec,
        batch_size=1,
        step=1,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        floor_s=floor,
    )
    assert result.math_s < floor
    assert result.memory_s < floor
    assert result.latency_s == pytest.approx(floor)


def test_floor_does_not_affect_large_shapes(spec):
    # At large batch/step the roofline work dominates; a tiny floor must be ignored.
    result_no_floor = generation_attention_sol(
        spec,
        batch_size=64,
        step=8192,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
    )
    floor = 22e-6  # vLLM generation floor
    result_with_floor = generation_attention_sol(
        spec,
        batch_size=64,
        step=8192,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        floor_s=floor,
    )
    # The large-shape roofline must dominate; adding the floor changes nothing.
    assert result_with_floor.latency_s == pytest.approx(result_no_floor.latency_s)
    assert result_with_floor.memory_s > floor


def test_gemm_sol_has_no_floor_field_and_default_is_zero(spec):
    # GEMM must not be altered: its floor_s is not accepted, or defaults to 0 without effect.
    # The default SolComponents.floor_s is 0.0, so GEMM SOL must have floor_s=0.0.
    result = gemm_sol(spec, m=1024, n=4096, k=4096, dtype="bfloat16")
    assert result.floor_s == pytest.approx(0.0)
    # And latency_s must still equal the binding roofline half (unchanged).
    assert result.latency_s == pytest.approx(max(result.math_s, result.memory_s))


def test_context_attention_floor_binds_at_tiny_shapes(spec):
    # batch=1, seq_len=1: both halves are tiny; floor must dominate.
    floor = 32e-6  # vLLM context floor
    result = context_attention_sol(
        spec,
        batch_size=1,
        seq_len=1,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        floor_s=floor,
    )
    assert result.latency_s == pytest.approx(floor)


def test_fallback_floors_mapping_covers_vllm_and_flashinfer():
    # The fitted floor values must be present and positive for both backends and both ops.
    for backend in ("vllm", "flashinfer"):
        assert backend in FALLBACK_FLOORS_S, f"Missing backend {backend!r} in FALLBACK_FLOORS_S."
        for op_name in ("context_attention", "generation_attention"):
            floor = FALLBACK_FLOORS_S[backend][op_name]
            assert floor > 0.0, f"Floor for {backend!r}/{op_name!r} must be positive, got {floor!r}."


def test_every_floor_taking_op_has_a_fallback_under_some_backend():
    """A floor-taking op with no fallback degrades to a zero floor when floors.json is absent.

    Measured, that is not a refinement: elementwise goes 5.01% -> 11.79% MRE and all_reduce
    6.07% -> 11.52%, and for all_reduce the underlying SOL error is 220x at 4096 bytes (2.000 us
    fitted against 0.0091 us of ring transfer) which the fitted utility then partly launders.
    Deriving the check off the registry makes the invariant self-maintaining: a sixth
    floor-taking op fails here rather than relying on its author remembering.
    """
    from dryrun.cost.kernel.ops.base import OP_REGISTRY

    covered = {op_name for floors in FALLBACK_FLOORS_S.values() for op_name in floors}
    for name, op in sorted(OP_REGISTRY.items()):
        if not op.fits_floor:
            continue
        assert name in covered, (
            f"Op {name!r} declares fits_floor but appears under no backend in "
            f"FALLBACK_FLOORS_S, so a missing floors.json prices it with a zero floor."
        )


@pytest.mark.parametrize(
    ("backend", "op_name"),
    sorted(
        (backend, op_name)
        for backend, floors in FALLBACK_FLOORS_S.items()
        for op_name in floors
    ),
)
def test_every_fallback_floor_is_a_positive_number_for_a_registered_op(backend, op_name):
    """A stale key or a zero value would make the fallback quietly equivalent to no floor."""
    from dryrun.cost.kernel.ops.base import op_for

    assert op_for(op_name).fits_floor, (
        f"FALLBACK_FLOORS_S has a floor for {backend!r}/{op_name!r} but that op declares "
        f"fits_floor False, so the value is dead."
    )
    floor = FALLBACK_FLOORS_S[backend][op_name]
    assert floor > 0.0, f"Floor for {backend!r}/{op_name!r} must be positive, got {floor!r}."

