"""Tests for the Qwen3.5 dense hybrid kernel-shape family.

The block differs from a dense Llama in two ways that change the measured GEMM set: the
softmax layers double their query rows for an output gate, and three of every four layers
run a gated delta net with its own projection set instead of attention. The tests pin the
widths against the checkpoint's safetensors shapes and check that the cost path multiplies
each set by its own layer count.
"""

from __future__ import annotations

import pytest

from srl.cost.kernel.model_ops import ModelOps
from srl.profiler.kernel.grid.families import FAMILY_REGISTRY, family_for
from srl.profiler.kernel.grid.families_hybrid import QWEN35_DENSE_FAMILY, Qwen35DenseFamily
from srl.profiler.kernel.grid.gemm import (
    linear_attention_gemm_shapes,
    per_layer_gemm_shapes,
    transformer_gemm_shapes,
)
from srl.profiler.schema import ModelDescriptor, ProfileValidationError


def _qwen35_dense_descriptor(
    *,
    num_layers: int = 64,
    hidden_size: int = 5120,
    num_attention_heads: int = 24,
    num_key_value_heads: int = 4,
    head_dim: int = 256,
    intermediate_size: int = 17408,
    vocab_size: int = 248320,
    full_attention_interval: int = 4,
    attn_output_gate: int | None = 1,
    linear_num_key_heads: int = 16,
    linear_num_value_heads: int = 48,
    linear_key_head_dim: int = 128,
    linear_value_head_dim: int = 128,
    architecture: str = "Qwen3_5ForConditionalGeneration",
) -> ModelDescriptor:
    """Build a minimal descriptor matching Qwen3.8-27B's architecture."""
    extra: dict = {
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "intermediate_size": intermediate_size,
        "full_attention_interval": full_attention_interval,
        "linear_num_key_heads": linear_num_key_heads,
        "linear_num_value_heads": linear_num_value_heads,
        "linear_key_head_dim": linear_key_head_dim,
        "linear_value_head_dim": linear_value_head_dim,
    }
    if attn_output_gate is not None:
        extra["attn_output_gate"] = attn_output_gate
    return ModelDescriptor(
        name="test-qwen35-dense",
        architecture=architecture,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_key_value_heads,
        extra=extra,
    )


class TestRegistration:
    def test_multimodal_arch_is_registered(self):
        assert "Qwen3_5ForConditionalGeneration" in FAMILY_REGISTRY

    def test_text_only_arch_is_registered(self):
        assert "Qwen3_5ForCausalLM" in FAMILY_REGISTRY

    def test_family_for_resolves_correctly(self):
        assert family_for(_qwen35_dense_descriptor()) is QWEN35_DENSE_FAMILY

    def test_moe_arch_still_resolves_to_the_moe_family(self):
        from srl.profiler.kernel.grid.families_moe import QWEN35_MOE_FAMILY

        model = _qwen35_dense_descriptor(architecture="Qwen3_5MoeForConditionalGeneration")
        model = ModelDescriptor(
            name=model.name,
            architecture=model.architecture,
            num_layers=model.num_layers,
            hidden_size=model.hidden_size,
            num_attention_heads=model.num_attention_heads,
            num_query_groups=model.num_query_groups,
            extra={**model.extra, "num_experts": 256, "num_experts_per_tok": 8,
                   "moe_intermediate_size": 512},
        )
        assert family_for(model) is QWEN35_MOE_FAMILY


class TestLayerCounts:
    def test_quarter_of_the_layers_attend_over_context(self):
        assert QWEN35_DENSE_FAMILY.n_full_attention_layers(_qwen35_dense_descriptor()) == 16

    def test_missing_interval_is_refused(self):
        model = _qwen35_dense_descriptor()
        model = ModelDescriptor(
            name=model.name,
            architecture=model.architecture,
            num_layers=model.num_layers,
            hidden_size=model.hidden_size,
            num_attention_heads=model.num_attention_heads,
            num_query_groups=model.num_query_groups,
            extra={key: value for key, value in model.extra.items()
                   if key != "full_attention_interval"},
        )
        with pytest.raises(ProfileValidationError):
            QWEN35_DENSE_FAMILY.n_full_attention_layers(model)


class TestFullAttentionShapes:
    def test_shapes_match_the_checkpoint_at_tp1(self):
        shapes = {shape.name: (shape.n, shape.k) for shape in transformer_gemm_shapes(
            _qwen35_dense_descriptor(), tensor_parallel=1
        )}
        # q_proj is 12288 rows because the output gate doubles it, and k/v are 4 x 256 each.
        assert shapes["qkv_proj"] == (14336, 5120)
        assert shapes["o_proj"] == (5120, 6144)
        assert shapes["gate_up_proj"] == (34816, 5120)
        assert shapes["down_proj"] == (5120, 17408)

    def test_output_gate_is_optional(self):
        shapes = {shape.name: (shape.n, shape.k) for shape in transformer_gemm_shapes(
            _qwen35_dense_descriptor(attn_output_gate=None), tensor_parallel=1
        )}
        assert shapes["qkv_proj"] == (8192, 5120)

    def test_tensor_parallel_shards_every_projection(self):
        shapes = {shape.name: (shape.n, shape.k) for shape in transformer_gemm_shapes(
            _qwen35_dense_descriptor(), tensor_parallel=4
        )}
        assert shapes["qkv_proj"] == (3584, 5120)
        assert shapes["o_proj"] == (5120, 1536)
        assert shapes["gate_up_proj"] == (8704, 5120)

    def test_tensor_parallel_beyond_the_query_group_count_is_refused(self):
        with pytest.raises(ProfileValidationError):
            transformer_gemm_shapes(_qwen35_dense_descriptor(), tensor_parallel=8)


class TestLinearAttentionShapes:
    def test_shapes_match_the_fused_engine_modules(self):
        shapes = {shape.name: (shape.n, shape.k) for shape in linear_attention_gemm_shapes(
            _qwen35_dense_descriptor(), tensor_parallel=1
        )}
        # in_proj_qkv is 2 x 16 x 128 + 48 x 128, and in_proj_z is 48 x 128, fused together.
        assert shapes["in_proj_qkvz"] == (16384, 5120)
        assert shapes["in_proj_ba"] == (96, 5120)
        assert shapes["out_proj"] == (5120, 6144)
        # The MoE-less block still has an MLP, and it is the same one on both layer types.
        assert shapes["gate_up_proj"] == (34816, 5120)

    def test_tensor_parallel_shards_the_linear_projections(self):
        shapes = {shape.name: (shape.n, shape.k) for shape in linear_attention_gemm_shapes(
            _qwen35_dense_descriptor(), tensor_parallel=4
        )}
        assert shapes["in_proj_qkvz"] == (4096, 5120)
        assert shapes["in_proj_ba"] == (24, 5120)
        assert shapes["out_proj"] == (5120, 1536)


class TestPerLayerUnion:
    def test_the_union_holds_both_blocks(self):
        names = {shape.name for shape in per_layer_gemm_shapes(
            _qwen35_dense_descriptor(), tensor_parallel=1
        )}
        assert {"qkv_proj", "in_proj_qkvz", "in_proj_ba", "gate_up_proj"} <= names

    def test_a_pure_softmax_model_has_no_linear_shapes(self):
        dense = ModelDescriptor(
            name="dense",
            architecture="Qwen3ForCausalLM",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            extra={"vocab_size": 151936},
        )
        assert linear_attention_gemm_shapes(dense, tensor_parallel=1) == ()


class TestCostPathMultipliers:
    def test_each_projection_is_charged_on_its_own_layer_type(self):
        ops = ModelOps(_qwen35_dense_descriptor(), tensor_parallel=1)
        assert ops.attention_layers == 16
        assert ops.gdn_layers == 48
        multipliers = {}
        for query, mult in ops.gemm_queries(token_count=1024, request_count=1):
            multipliers.setdefault((query.n, query.k), []).append(mult)
        # A softmax layer's qkv is charged on 16 layers, the linear layer's fused input on 48.
        assert multipliers[(14336, 5120)] == [16]
        assert multipliers[(16384, 5120)] == [48]
        # The MLP exists on every layer, split across the two sets.
        assert sum(multipliers[(34816, 5120)]) == 64
        assert sum(multipliers[(5120, 17408)]) == 64
