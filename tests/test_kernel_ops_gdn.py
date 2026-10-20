"""Gated delta net: the linear-attention ops of a hybrid-attention model."""

from __future__ import annotations

import pytest

from srl.cost.kernel.hardware import load_gpu_spec
from srl.cost.kernel.ops.base import families, op_for, ops_for_family
from srl.cost.kernel.sol import FALLBACK_FLOORS_S, gdn_context_sol, gdn_decode_sol
from srl.profiler.kernel.runner.gdn import gdn_head_counts
from srl.profiler.schema import ModelDescriptor, ProfileValidationError

# Qwen3.5-35B-A3B's GDN geometry, read from its config.json.
_KEY_HEADS, _VALUE_HEADS, _HEAD_DIM = 16, 32, 128


def _hybrid_model(**overrides) -> ModelDescriptor:
    extra = {
        "linear_num_key_heads": _KEY_HEADS,
        "linear_num_value_heads": _VALUE_HEADS,
        "linear_key_head_dim": _HEAD_DIM,
        "linear_value_head_dim": _HEAD_DIM,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
    }
    extra.update(overrides)
    return ModelDescriptor(
        name="qwen35-moe-like",
        architecture="Qwen3_5MoeForConditionalGeneration",
        num_layers=40,
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=2,
        extra=extra,
    )


def test_both_gdn_ops_are_registered_as_their_own_family():
    """GDN is a family, not a slice of `attention`: different kernels, different cost."""
    assert "gdn" in families()
    assert {op.name for op in ops_for_family("gdn")} == {"gdn_context", "gdn_decode"}
    # And both really carry the family tag, so a sweep restricted to `--family gdn` finds them.
    assert op_for("gdn_context").family == "gdn"
    assert op_for("gdn_decode").family == "gdn"


def test_gdn_prefill_is_linear_in_tokens_where_attention_is_quadratic():
    """
    Doubling the token count roughly doubles GDN prefill, rather than quadrupling it.

    This is the whole reason the family exists. A gated delta net carries a fixed-size
    recurrent state instead of a growing KV cache, so a token attends to the state rather
    than to every earlier token. Pricing these layers with `context_attention_sol` would
    overstate a long prompt by about `seq_len / chunk_size`.
    """
    spec = load_gpu_spec("h20")
    kwargs = dict(
        num_key_heads=_KEY_HEADS,
        num_value_heads=_VALUE_HEADS,
        head_dim=_HEAD_DIM,
        attn_dtype="bfloat16",
    )
    # Taken above the chunk size so the intra-chunk term has saturated and the state term
    # dominates. Below it the cost is constant, which is the measured plateau.
    short = gdn_context_sol(spec, total_tokens=8192, **kwargs).latency_s
    long = gdn_context_sol(spec, total_tokens=16384, **kwargs).latency_s
    assert 1.9 < long / short < 2.1


def test_gdn_decode_does_not_depend_on_history_depth():
    """
    The decode op has no `step` axis, and that is structural rather than an omission.

    The recurrent state is a fixed `num_value_heads x head_dim x head_dim` per request
    whatever the KV depth, so a decode step costs the same at depth 128 as at 32768.
    A table interpolated over history would be fitting noise. The axis columns are the
    contract that keeps that true.
    """
    assert op_for("gdn_decode").axis_columns == ("batch_size",)
    assert op_for("gdn_context").axis_columns == ("total_tokens",)


def test_gdn_decode_scales_with_batch_and_is_memory_bound():
    """One token against a large fp32 state: bandwidth decides, not FLOPs."""
    spec = load_gpu_spec("h20")
    kwargs = dict(
        num_key_heads=_KEY_HEADS,
        num_value_heads=_VALUE_HEADS,
        head_dim=_HEAD_DIM,
        attn_dtype="bfloat16",
    )
    one = gdn_decode_sol(spec, batch_size=1, **kwargs)
    many = gdn_decode_sol(spec, batch_size=64, **kwargs)
    assert one.memory_s > one.math_s
    assert many.latency_s == pytest.approx(64 * one.latency_s, rel=1e-6)


def test_gdn_state_is_indexed_by_value_heads_not_key_heads():
    """
    The recurrent state's leading dimension is `num_value_heads`.

    `MambaStateShapeCalculator.gated_delta_net_state_shape` returns
    `(num_v_heads, head_v_dim, head_k_dim)`, and the decode kernel reads `HV` off
    `initial_state.shape[-3]` to validate its `a` and `b` arguments. Keying the state on key
    heads instead raises `a/b must have shape [B, HV] with HV=16` at measurement time, and
    would understate the decode roofline by the value/key head ratio, which is 2x on this model.
    """
    spec = load_gpu_spec("h20")
    base = dict(batch_size=8, head_dim=_HEAD_DIM, attn_dtype="bfloat16")
    # Swapping the two head counts must change the answer. If the formula used key heads it
    # would be symmetric in the wrong direction.
    by_value = gdn_decode_sol(
        spec, num_key_heads=_KEY_HEADS, num_value_heads=_VALUE_HEADS, **base
    ).memory_s
    halved = gdn_decode_sol(
        spec, num_key_heads=_KEY_HEADS, num_value_heads=_VALUE_HEADS // 2, **base
    ).memory_s
    assert by_value > halved


def test_both_gdn_ops_have_a_fallback_floor():
    """
    A missing `floors.json` must not price GDN at a zero floor.

    The measured kernel is flat at ~368 us below 4096 tokens and ~25 us below batch 16, so
    the floor is most of the cost in the regime these layers usually run in.
    """
    vllm_floors = FALLBACK_FLOORS_S["vllm"]
    assert vllm_floors["gdn_context"] > 0
    assert vllm_floors["gdn_decode"] > 0


def test_gdn_head_counts_shard_by_tensor_parallel():
    """Both head counts shard, and an uneven split is refused rather than rounded."""
    model = _hybrid_model()
    assert gdn_head_counts(model, tensor_parallel=1) == (16, 32, 128)
    assert gdn_head_counts(model, tensor_parallel=2) == (8, 16, 128)
    assert gdn_head_counts(model, tensor_parallel=4) == (4, 8, 128)
    with pytest.raises(ProfileValidationError, match="divisible"):
        gdn_head_counts(_hybrid_model(linear_num_key_heads=6), tensor_parallel=4)


def test_gdn_refuses_a_model_without_the_linear_attention_fields():
    """A dense checkpoint has no GDN layer, so profiling one is a configuration error."""
    dense = ModelDescriptor(
        name="dense",
        architecture="Qwen3ForCausalLM",
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        extra={},
    )
    with pytest.raises(ProfileValidationError, match="linear_num_key_heads"):
        gdn_head_counts(dense, tensor_parallel=1)


def test_gdn_refuses_differing_key_and_value_head_dimensions():
    """The SOL assumes one head dimension. A model with two needs separate terms."""
    with pytest.raises(ProfileValidationError, match="linear_value_head_dim"):
        gdn_head_counts(_hybrid_model(linear_value_head_dim=256), tensor_parallel=1)
