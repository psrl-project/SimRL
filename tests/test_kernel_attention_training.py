"""The training attention family, and why it is not the rollout attention family.

This op exists because substituting `context_attention` for a THD training query was wrong
in both directions at once, and the tests record the shape of that error rather than just
the fix. The measured numbers quoted in the docstrings come from
`srl-profile kernels run --family attention_training` on H20 at tp=2.
"""

from __future__ import annotations

import pytest

from srl.cost.kernel.database import TrainingAttentionQuery
from srl.cost.kernel.hardware import load_gpu_spec
from srl.cost.kernel.interp import KernelCoverageError
from srl.cost.kernel.ops.attention_training import (
    TRAINING_ATTENTION_BACKWARD_OP,
    TRAINING_ATTENTION_FORWARD_OP,
    TRAINING_MASK_TYPES,
)
from srl.cost.kernel.ops.base import op_for
from srl.profiler.kernel.sweep import TrainingAttentionSweep
from srl.profiler.schema import ProfileValidationError

_SLICE = (4, 128, "bfloat16", "padding_causal")


def test_both_directions_are_registered_as_their_own_ops():
    """Separate ops rather than a `direction` slice column.

    A slice implies the two interpolate onto one surface, and they do not: the backward
    computes dQ, dK and dV against the forward's single output, so its roofline has a
    different constant.
    """
    assert op_for("training_attention_forward") is TRAINING_ATTENTION_FORWARD_OP
    assert op_for("training_attention_backward") is TRAINING_ATTENTION_BACKWARD_OP
    assert TRAINING_ATTENTION_FORWARD_OP.family == "attention_training"
    assert TRAINING_ATTENTION_BACKWARD_OP.family == "attention_training"


def test_the_family_is_separate_from_the_rollout_attention_family():
    """Not a new phase on the existing `attention` family, and the separation is the point.

    The rollout table measures vLLM's prefill kernel over a BSHD batch whose grid varies a
    batch dimension a THD pack does not have. This one measures TE's FlashAttention over one
    packed row. Sharing a family directory would put two kernels under one `floors.json`.
    """
    assert op_for("context_attention").family == "attention"
    assert TRAINING_ATTENTION_FORWARD_OP.family != op_for("context_attention").family


def test_the_axis_is_the_packed_token_count_with_no_batch_dimension():
    """A THD microbatch is one row, so batch size is always 1 and cannot be an axis."""
    assert TRAINING_ATTENTION_FORWARD_OP.axis_columns == ("num_heads", "total_tokens")
    assert "batch_size" not in TRAINING_ATTENTION_FORWARD_OP.axis_columns


def test_the_backward_roofline_is_a_larger_multiple_than_the_forward():
    """Measured bwd/fwd is 2.24-2.42, so a backward SOL equal to forward would be 2.4x low.

    `ComposedTraining` previously approximated backward at 1.0x of forward on the strength
    of a cuDNN benchmark for a different layout. That approximation was 9.6pp of the
    composition's total error.
    """
    assert TRAINING_ATTENTION_FORWARD_OP.flop_multiplier == 1.0
    assert TRAINING_ATTENTION_BACKWARD_OP.flop_multiplier > 2.0

    spec = load_gpu_spec("h20")
    coords = (16.0, 12288.0)
    forward = TRAINING_ATTENTION_FORWARD_OP.sol(
        spec, slice_key=_SLICE, coords=coords, floor_s=0.0
    ).latency_s
    backward = TRAINING_ATTENTION_BACKWARD_OP.sol(
        spec, slice_key=_SLICE, coords=coords, floor_s=0.0
    ).latency_s
    assert backward > 2.0 * forward


def test_the_causal_roofline_is_half_the_non_causal_one():
    """A causal mask attends to half the pack on average."""
    spec = load_gpu_spec("h20")
    coords = (16.0, 8192.0)
    causal = TRAINING_ATTENTION_FORWARD_OP.sol(
        spec, slice_key=(4, 128, "bfloat16", "padding_causal"), coords=coords, floor_s=0.0
    ).math_s
    full = TRAINING_ATTENTION_FORWARD_OP.sol(
        spec, slice_key=(4, 128, "bfloat16", "padding"), coords=coords, floor_s=0.0
    ).math_s
    assert causal == pytest.approx(0.5 * full)


def test_the_roofline_is_quadratic_in_the_packed_token_count():
    """Attention is quadratic in sequence length, which is why the token grid must be dense.

    A linear SOL would make the fitted utility absorb the curvature, and the utility surface
    interpolates rather than extrapolates, so a query past the measured frontier would clamp
    onto a wrong slope.
    """
    spec = load_gpu_spec("h20")

    def math_s(tokens: float) -> float:
        return TRAINING_ATTENTION_FORWARD_OP.sol(
            spec, slice_key=_SLICE, coords=(16.0, tokens), floor_s=0.0
        ).math_s

    assert math_s(8192.0) == pytest.approx(4.0 * math_s(4096.0))


def test_a_non_positive_dimension_is_refused():
    spec = load_gpu_spec("h20")
    with pytest.raises(ValueError, match="positive dimensions"):
        TRAINING_ATTENTION_FORWARD_OP.sol(
            spec, slice_key=_SLICE, coords=(16.0, 0.0), floor_s=0.0
        )


def test_plain_causal_is_not_a_known_mask_type():
    """Transformer Engine asserts `"padding" in attn_mask_type` for qkv_format='thd'.

    So a table keyed on plain `causal` could not have been measured, and a query naming it
    is a caller mistake rather than a coverage gap. This is also the misstep that produced
    the design doc's "cuDNN on Hopper+" claim: a backend probe built with `causal` reports
    every backend unavailable, which reads as a hardware fact.
    """
    assert "causal" not in TRAINING_MASK_TYPES
    assert "padding_causal" in TRAINING_MASK_TYPES

    spec = load_gpu_spec("h20")
    with pytest.raises(ValueError, match="Unknown training attention mask type"):
        TRAINING_ATTENTION_FORWARD_OP.sol(
            spec, slice_key=(4, 128, "bfloat16", "causal"), coords=(16.0, 4096.0), floor_s=0.0
        )


def test_the_query_defaults_to_the_training_framework():
    """The database default is `vllm`, which names no directory for this family."""
    query = TrainingAttentionQuery(
        direction="forward", num_heads=16, total_tokens=4096, num_kv_heads=4, head_dim=128
    )
    assert query.backend == "megatron"


def test_the_query_selects_its_op_by_direction():
    base = dict(num_heads=16, total_tokens=4096, num_kv_heads=4, head_dim=128)
    assert (
        TrainingAttentionQuery(direction="forward", **base).op_name
        == "training_attention_forward"
    )
    assert (
        TrainingAttentionQuery(direction="backward", **base).op_name
        == "training_attention_backward"
    )


def test_the_query_refuses_an_unknown_direction():
    with pytest.raises(KernelCoverageError, match="direction must be"):
        TrainingAttentionQuery(
            direction="recompute", num_heads=16, total_tokens=4096, num_kv_heads=4,
            head_dim=128,
        )


def test_the_query_refuses_a_mask_type_transformer_engine_would_reject():
    with pytest.raises(KernelCoverageError, match="rejects plain 'causal'"):
        TrainingAttentionQuery(
            direction="forward", num_heads=16, total_tokens=4096, num_kv_heads=4,
            head_dim=128, attn_mask_type="causal",
        )


@pytest.mark.parametrize(
    "field", ["num_heads", "total_tokens", "num_kv_heads", "head_dim"]
)
def test_the_query_refuses_a_non_positive_dimension(field):
    base = dict(
        direction="forward", num_heads=16, total_tokens=4096, num_kv_heads=4, head_dim=128
    )
    base[field] = 0
    with pytest.raises(KernelCoverageError, match=f"{field} must be positive"):
        TrainingAttentionQuery(**base)


def test_the_sweep_refuses_a_mask_type_that_cannot_be_measured():
    """Caught at config parse rather than after a GPU sweep has already run."""
    with pytest.raises(ProfileValidationError, match="attention_training.mask_types"):
        TrainingAttentionSweep.from_dict(
            {"total_tokens": [4096], "num_heads": [16], "mask_types": ["causal"]}
        )


def test_the_sweep_defaults_to_the_mask_type_megatron_passes():
    sweep = TrainingAttentionSweep.from_dict({"total_tokens": [4096], "num_heads": [16]})
    assert sweep.mask_types == ("padding_causal",)


def test_the_sweep_refuses_a_non_positive_grid_value():
    with pytest.raises(ProfileValidationError, match="attention_training.total_tokens"):
        TrainingAttentionSweep.from_dict({"total_tokens": [4096, 0], "num_heads": [16]})


def test_a_table_measured_on_the_unfused_fallback_is_refused():
    """The failure is silent, which is why it needs a gate rather than a warning.

    A `.post1` flash-attn wheel is rejected by TE's upper bound (PEP 440 orders
    `2.8.3.post1` above `2.8.3`), TE falls back to `UnfusedDotProductAttention`, and the
    only signal is a debug-level log line. The first run of this sweep measured that
    fallback, which materializes the full attention matrix and is several times slower than
    the kernel a training step runs.
    """
    from srl.profiler.kernel.runner.attention_training import _require_fused_backend

    with pytest.raises(ProfileValidationError, match="post-release flash-attn wheel"):
        _require_fused_backend({"UnfusedDotProductAttention"})


def test_a_table_spanning_two_backends_is_refused():
    """One table must describe one kernel."""
    from srl.profiler.kernel.runner.attention_training import _require_fused_backend

    with pytest.raises(ProfileValidationError, match="more than one backend"):
        _require_fused_backend({"FlashAttention", "FusedAttention"})


def test_an_empty_sweep_is_refused_rather_than_writing_an_empty_table():
    from srl.profiler.kernel.runner.attention_training import _require_fused_backend

    with pytest.raises(ProfileValidationError, match="No attention call was made"):
        _require_fused_backend(set())


def test_the_dispatch_floor_is_fitted_at_the_top_of_the_candidate_grid():
    """Recorded because it means the true floor is above the grid, not at its edge.

    A TE attention call costs ~195 us on this machine regardless of work below ~3072 tokens:
    at 4 heads the forward measures 200.1, 199.6 and 193.8 us at 512, 1024 and 2048 tokens,
    flat across 16x more work. The candidate grid tops out at 80 us, so the fit picks 80 and
    under-states the floor by roughly 2.4x.

    Widening `candidates_us` is the right fix and is not done, because the grid is shared by
    every floor-taking op and re-fitting would move the collective and elementwise floors
    that `test_collective_baseline_regression.py` pins. This test exists so the known-low
    value cannot be mistaken for a measurement, and fails if someone widens the grid without
    revisiting it.
    """
    from srl.cost.kernel.sol import FALLBACK_FLOORS_S

    megatron = FALLBACK_FLOORS_S["megatron"]
    assert megatron["training_attention_forward"] == 80e-6
    assert megatron["training_attention_backward"] == 80e-6


def test_a_fused_backend_passes():
    """FlashAttention is what TE selects for THD on sm90, measured rather than assumed."""
    from srl.profiler.kernel.runner.attention_training import _require_fused_backend

    _require_fused_backend({"FlashAttention"})
    _require_fused_backend({"FusedAttention"})


def test_training_ops_builds_queries_against_the_training_family():
    """`TrainingOps.attention_query` must not emit a rollout `AttentionQuery`.

    The mis-sharding this guards against is subtle: both queries carry `num_heads` and a
    token count, so a composition reading the wrong family produces a plausible number.
    """
    from srl.cost.training.training_ops import TrainingOps
    from srl.profiler.schema import ModelDescriptor

    model = ModelDescriptor(
        name="test",
        architecture="Qwen3ForCausalLM",
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        extra={"intermediate_size": 12288, "head_dim": 128, "vocab_size": 151936},
    )
    ops = TrainingOps(model, tensor_parallel=2)
    forward = ops.attention_query(token_count=12288, direction="forward")
    assert isinstance(forward, TrainingAttentionQuery)
    assert forward.op_name == "training_attention_forward"
    # tp=2 shards 32 query heads to 16 and 8 KV heads to 4.
    assert (forward.num_heads, forward.num_kv_heads) == (16, 4)
    assert forward.total_tokens == 12288
    # The backend must be the query's own default, not None: None means "database default",
    # which is `vllm` and resolves to a directory this family does not have.
    assert forward.backend == "megatron"
