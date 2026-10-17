"""Tests for the Qwen3.5-MoE kernel-shape family.

The MoE family is structurally different from the dense Llama family in three ways that
matter for testing: it has a router GEMM, the MLP shapes are per activated expert rather
than per layer, and only `full_attention_interval`-th layers are full-attention. The tests
here verify all three properties plus the shared-expert branch and tensor-parallel sharding.
"""

from __future__ import annotations

import pytest

from srl.profiler.kernel.grid.families import FAMILY_REGISTRY, family_for
from srl.profiler.kernel.grid.families_moe import QWEN35_MOE_FAMILY, Qwen35MoeFamily
from srl.profiler.schema import ModelDescriptor, ProfileValidationError


def _qwen35_descriptor(
    *,
    num_layers: int = 40,
    hidden_size: int = 2048,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 2,
    head_dim: int = 256,
    vocab_size: int = 248320,
    num_experts: int = 256,
    num_experts_per_tok: int = 8,
    moe_intermediate_size: int = 512,
    full_attention_interval: int = 4,
    shared_expert_intermediate_size: int | None = 512,
    architecture: str = "Qwen3_5MoeForConditionalGeneration",
) -> ModelDescriptor:
    """Build a minimal `ModelDescriptor` matching Qwen3.5-35B-A3B's architecture."""
    extra: dict = {
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "moe_intermediate_size": moe_intermediate_size,
        "full_attention_interval": full_attention_interval,
    }
    if shared_expert_intermediate_size is not None:
        extra["shared_expert_intermediate_size"] = shared_expert_intermediate_size
    return ModelDescriptor(
        name="test-qwen35-moe",
        architecture=architecture,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_key_value_heads,
        extra=extra,
    )


class TestRegistration:
    def test_multimodal_arch_is_registered(self):
        assert "Qwen3_5MoeForConditionalGeneration" in FAMILY_REGISTRY

    def test_text_only_arch_is_registered(self):
        assert "Qwen3_5MoeForCausalLM" in FAMILY_REGISTRY

    def test_family_for_resolves_correctly(self):
        model = _qwen35_descriptor()
        assert family_for(model) is QWEN35_MOE_FAMILY

    def test_dense_arch_still_resolves_to_dense(self):
        from srl.profiler.kernel.grid.families_dense import DENSE_LLAMA_FAMILY

        dense = ModelDescriptor(
            name="test",
            architecture="Qwen3ForCausalLM",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            extra={"vocab_size": 151936},
        )
        assert family_for(dense) is DENSE_LLAMA_FAMILY


class TestFullAttentionLayerCount:
    def test_10_full_layers_out_of_40_with_interval_4(self):
        model = _qwen35_descriptor(num_layers=40, full_attention_interval=4)
        assert QWEN35_MOE_FAMILY.n_full_attention_layers(model) == 10

    def test_interval_1_gives_all_layers(self):
        model = _qwen35_descriptor(num_layers=12, full_attention_interval=1)
        assert QWEN35_MOE_FAMILY.n_full_attention_layers(model) == 12

    def test_interval_equals_layers_gives_one(self):
        model = _qwen35_descriptor(num_layers=4, full_attention_interval=4)
        assert QWEN35_MOE_FAMILY.n_full_attention_layers(model) == 1

    def test_missing_full_attention_interval_raises(self):
        extra = {
            "head_dim": 256, "vocab_size": 248320, "num_experts": 256,
            "num_experts_per_tok": 8, "moe_intermediate_size": 512,
        }
        model = ModelDescriptor(
            name="test", architecture="Qwen3_5MoeForCausalLM",
            num_layers=40, hidden_size=2048, num_attention_heads=16, extra=extra,
        )
        with pytest.raises(ProfileValidationError, match="full_attention_interval"):
            QWEN35_MOE_FAMILY.n_full_attention_layers(model)


class TestGemmShapesTp1:
    """At tp=1 the per-rank shapes equal the model shapes."""

    def test_qkv_proj_shape(self):
        model = _qwen35_descriptor()
        shapes = {s.name: s for s in QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        # (16 + 2*2) * 256 = 5120 rows; 2048 cols
        assert shapes["qkv_proj"].n == (16 + 2 * 2) * 256
        assert shapes["qkv_proj"].k == 2048

    def test_o_proj_shape(self):
        model = _qwen35_descriptor()
        shapes = {s.name: s for s in QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        # hidden <- heads * head_dim = 16 * 256 = 4096 cols
        assert shapes["o_proj"].n == 2048
        assert shapes["o_proj"].k == 16 * 256

    def test_router_is_not_sharded_at_tp1(self):
        """The router must remain full-width regardless of TP: every rank scores all experts."""
        model = _qwen35_descriptor()
        shapes = {s.name: s for s in QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        assert shapes["router"].n == 256  # num_experts
        assert shapes["router"].k == 2048

    def test_expert_gate_up_charges_only_selected_experts(self):
        """gate_up_proj width = 2 * moe_inter * experts_per_tok, not 2 * moe_inter * num_experts."""
        model = _qwen35_descriptor()
        shapes = {s.name: s for s in QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        # 2 * 512 * 8 = 8192 rows; 2048 cols
        assert shapes["expert_gate_up"].n == 2 * 512 * 8
        assert shapes["expert_gate_up"].k == 2048

    def test_expert_down_shape(self):
        model = _qwen35_descriptor()
        shapes = {s.name: s for s in QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        # 2048 rows; 512 * 8 = 4096 cols
        assert shapes["expert_down"].n == 2048
        assert shapes["expert_down"].k == 512 * 8

    def test_shared_expert_present_when_configured(self):
        model = _qwen35_descriptor(shared_expert_intermediate_size=512)
        shapes = {s.name: s for s in QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        assert "shared_gate_up" in shapes
        assert "shared_down" in shapes
        assert shapes["shared_gate_up"].n == 2 * 512
        assert shapes["shared_gate_up"].k == 2048
        assert shapes["shared_down"].n == 2048
        assert shapes["shared_down"].k == 512

    def test_shared_expert_absent_when_not_configured(self):
        model = _qwen35_descriptor(shared_expert_intermediate_size=None)
        shapes = {s.name: s for s in QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        assert "shared_gate_up" not in shapes
        assert "shared_down" not in shapes

    def test_seven_shapes_returned_with_shared_expert(self):
        model = _qwen35_descriptor()
        assert len(QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)) == 7

    def test_five_shapes_returned_without_shared_expert(self):
        model = _qwen35_descriptor(shared_expert_intermediate_size=None)
        assert len(QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)) == 5


class TestGemmShapesTensorParallel:
    """TP sharding must halve columns-per-rank for column-parallel and rows for row-parallel."""

    def test_tp2_halves_qkv_rows(self):
        model = _qwen35_descriptor()
        tp1 = {s.name: s for s in QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        tp2 = {s.name: s for s in QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=2)}
        assert tp2["qkv_proj"].n == tp1["qkv_proj"].n // 2

    def test_tp2_halves_expert_gate_up_rows(self):
        model = _qwen35_descriptor()
        tp1 = {s.name: s for s in QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        tp2 = {s.name: s for s in QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=2)}
        assert tp2["expert_gate_up"].n == tp1["expert_gate_up"].n // 2

    def test_router_is_not_sharded_at_tp2(self):
        """Every rank needs the full routing decision; no sharding."""
        model = _qwen35_descriptor()
        tp1 = {s.name: s for s in QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        tp2 = {s.name: s for s in QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=2)}
        assert tp2["router"].n == tp1["router"].n
        assert tp2["router"].k == tp1["router"].k

    def test_tp4_indivisible_kv_heads_raises(self):
        """num_key_value_heads=2 is not divisible by tp=4.

        This is a real constraint on the checkpoint, not a test artifact: Qwen3.5-35B-A3B
        declares 2 KV heads, so tp=4 would have to replicate them rather than shard. vLLM
        does replicate in that case, but the GEMM shape then differs from `kv_heads/tp`,
        so the family refuses rather than reporting a width the engine does not run.
        """
        model = _qwen35_descriptor(num_key_value_heads=2)
        with pytest.raises(ProfileValidationError, match="num_query_groups"):
            QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=4)

    def test_tp2_logits_shape(self):
        model = _qwen35_descriptor()
        shape = QWEN35_MOE_FAMILY.logits_shape(model, tensor_parallel=2)
        assert shape.n == 248320 // 2
        assert shape.k == 2048


class TestHeadCounts:
    def test_tp1_head_counts(self):
        model = _qwen35_descriptor()
        heads, kv_heads = QWEN35_MOE_FAMILY.head_counts(model, tensor_parallel=1)
        assert heads == 16
        assert kv_heads == 2

    def test_tp2_head_counts(self):
        model = _qwen35_descriptor()
        heads, kv_heads = QWEN35_MOE_FAMILY.head_counts(model, tensor_parallel=2)
        assert heads == 8
        assert kv_heads == 1


class TestLogitsShape:
    def test_tp1_logits(self):
        model = _qwen35_descriptor()
        shape = QWEN35_MOE_FAMILY.logits_shape(model, tensor_parallel=1)
        assert shape.name == "lm_head"
        assert shape.n == 248320
        assert shape.k == 2048


class TestValidationErrors:
    def test_zero_tensor_parallel_raises(self):
        model = _qwen35_descriptor()
        with pytest.raises(ProfileValidationError, match="tensor_parallel"):
            QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=0)

    def test_missing_num_experts_raises(self):
        extra = {
            "head_dim": 256, "vocab_size": 248320,
            "num_experts_per_tok": 8, "moe_intermediate_size": 512,
            "full_attention_interval": 4,
        }
        model = ModelDescriptor(
            name="test", architecture="Qwen3_5MoeForCausalLM",
            num_layers=40, hidden_size=2048, num_attention_heads=16,
            num_query_groups=2, extra=extra,
        )
        with pytest.raises(ProfileValidationError, match="num_experts"):
            QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)

    def test_missing_moe_intermediate_size_raises(self):
        extra = {
            "head_dim": 256, "vocab_size": 248320,
            "num_experts": 256, "num_experts_per_tok": 8,
            "full_attention_interval": 4,
        }
        model = ModelDescriptor(
            name="test", architecture="Qwen3_5MoeForCausalLM",
            num_layers=40, hidden_size=2048, num_attention_heads=16,
            num_query_groups=2, extra=extra,
        )
        with pytest.raises(ProfileValidationError, match="moe_intermediate_size"):
            QWEN35_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)
