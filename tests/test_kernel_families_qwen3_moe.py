"""Tests for the plain Qwen3-MoE kernel-shape family.

Structurally simpler than `Qwen35MoeFamily` in two ways that matter for testing: every layer
attends over the full context (no `full_attention_interval` split), and there is no shared
expert. The tests here verify the attention and per-expert shapes, tensor-parallel sharding,
and that this family refuses a checkpoint declaring either of the keys it does not describe.
"""

from __future__ import annotations

import pytest

from srl.profiler.kernel.grid.families import FAMILY_REGISTRY, family_for
from srl.profiler.kernel.grid.families_qwen3_moe import QWEN3_MOE_FAMILY, Qwen3MoeFamily
from srl.profiler.schema import ModelDescriptor, ProfileValidationError


def _qwen3_moe_descriptor(
    *,
    num_layers: int = 48,
    hidden_size: int = 2048,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 4,
    head_dim: int = 128,
    vocab_size: int = 151936,
    num_experts: int = 128,
    num_experts_per_tok: int = 8,
    moe_intermediate_size: int = 768,
    shared_expert_intermediate_size: int | None = None,
    full_attention_interval: int | None = None,
    architecture: str = "Qwen3MoeForCausalLM",
) -> ModelDescriptor:
    """Build a minimal `ModelDescriptor` matching Qwen3-30B-A3B's architecture."""
    extra: dict = {
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "moe_intermediate_size": moe_intermediate_size,
    }
    if shared_expert_intermediate_size is not None:
        extra["shared_expert_intermediate_size"] = shared_expert_intermediate_size
    if full_attention_interval is not None:
        extra["full_attention_interval"] = full_attention_interval
    return ModelDescriptor(
        name="test-qwen3-moe",
        architecture=architecture,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_key_value_heads,
        extra=extra,
    )


class TestRegistration:
    def test_arch_is_registered(self):
        assert "Qwen3MoeForCausalLM" in FAMILY_REGISTRY

    def test_family_for_resolves_correctly(self):
        model = _qwen3_moe_descriptor()
        assert family_for(model) is QWEN3_MOE_FAMILY

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

    def test_hybrid_moe_arch_still_resolves_to_the_hybrid_family(self):
        from srl.profiler.kernel.grid.families_moe import QWEN35_MOE_FAMILY

        model = ModelDescriptor(
            name="test",
            architecture="Qwen3_5MoeForCausalLM",
            num_layers=40,
            hidden_size=2048,
            num_attention_heads=16,
            num_query_groups=2,
            extra={
                "head_dim": 256, "vocab_size": 248320, "num_experts": 256,
                "num_experts_per_tok": 8, "moe_intermediate_size": 512,
                "full_attention_interval": 4,
            },
        )
        assert family_for(model) is QWEN35_MOE_FAMILY


class TestGemmShapesTp1:
    """At tp=1 the per-rank shapes equal the model shapes."""

    def test_qkv_proj_shape(self):
        model = _qwen3_moe_descriptor()
        shapes = {s.name: s for s in QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        # (32 + 2*4) * 128 = 5120 rows, 2048 cols
        assert shapes["qkv_proj"].n == (32 + 2 * 4) * 128
        assert shapes["qkv_proj"].k == 2048

    def test_o_proj_shape(self):
        model = _qwen3_moe_descriptor()
        shapes = {s.name: s for s in QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        # hidden <- heads * head_dim = 32 * 128 = 4096 cols
        assert shapes["o_proj"].n == 2048
        assert shapes["o_proj"].k == 32 * 128

    def test_router_is_not_sharded_at_tp1(self):
        """The router must remain full-width regardless of TP: every rank scores all experts."""
        model = _qwen3_moe_descriptor()
        shapes = {s.name: s for s in QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        assert shapes["router"].n == 128  # num_experts
        assert shapes["router"].k == 2048

    def test_expert_gate_up_charges_only_selected_experts(self):
        """gate_up_proj width = 2 * moe_inter * experts_per_tok, not 2 * moe_inter * num_experts."""
        model = _qwen3_moe_descriptor()
        shapes = {s.name: s for s in QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        # 2 * 768 * 8 = 12288 rows, 2048 cols
        assert shapes["expert_gate_up"].n == 2 * 768 * 8
        assert shapes["expert_gate_up"].k == 2048

    def test_expert_down_shape(self):
        model = _qwen3_moe_descriptor()
        shapes = {s.name: s for s in QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        # 2048 rows, 768 * 8 = 6144 cols
        assert shapes["expert_down"].n == 2048
        assert shapes["expert_down"].k == 768 * 8

    def test_no_shared_expert_shapes(self):
        """This architecture never has a shared expert, unlike Qwen3.5-MoE."""
        model = _qwen3_moe_descriptor()
        shapes = {s.name: s for s in QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        assert "shared_gate_up" not in shapes
        assert "shared_down" not in shapes

    def test_five_shapes_returned(self):
        model = _qwen3_moe_descriptor()
        assert len(QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)) == 5


class TestGemmShapesTensorParallel:
    """TP sharding must halve columns-per-rank for column-parallel and rows for row-parallel."""

    def test_tp2_halves_qkv_rows(self):
        model = _qwen3_moe_descriptor()
        tp1 = {s.name: s for s in QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        tp2 = {s.name: s for s in QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=2)}
        assert tp2["qkv_proj"].n == tp1["qkv_proj"].n // 2

    def test_tp2_halves_expert_gate_up_rows(self):
        model = _qwen3_moe_descriptor()
        tp1 = {s.name: s for s in QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        tp2 = {s.name: s for s in QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=2)}
        assert tp2["expert_gate_up"].n == tp1["expert_gate_up"].n // 2

    def test_router_is_not_sharded_at_tp2(self):
        """Every rank needs the full routing decision. No sharding."""
        model = _qwen3_moe_descriptor()
        tp1 = {s.name: s for s in QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)}
        tp2 = {s.name: s for s in QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=2)}
        assert tp2["router"].n == tp1["router"].n
        assert tp2["router"].k == tp1["router"].k

    def test_tp8_indivisible_kv_heads_raises(self):
        """num_key_value_heads=4 is not divisible by tp=8.

        A real constraint on the checkpoint: Qwen3-30B-A3B declares 4 KV heads, so tp=8
        cannot shard them evenly.
        """
        model = _qwen3_moe_descriptor(num_key_value_heads=4)
        with pytest.raises(ProfileValidationError, match="num_query_groups"):
            QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=8)

    def test_tp2_logits_shape(self):
        model = _qwen3_moe_descriptor()
        shape = QWEN3_MOE_FAMILY.logits_shape(model, tensor_parallel=2)
        assert shape.n == 151936 // 2
        assert shape.k == 2048


class TestHeadCounts:
    def test_tp1_head_counts(self):
        model = _qwen3_moe_descriptor()
        heads, kv_heads = QWEN3_MOE_FAMILY.head_counts(model, tensor_parallel=1)
        assert heads == 32
        assert kv_heads == 4

    def test_tp2_head_counts(self):
        model = _qwen3_moe_descriptor()
        heads, kv_heads = QWEN3_MOE_FAMILY.head_counts(model, tensor_parallel=2)
        assert heads == 16
        assert kv_heads == 2


class TestLogitsShape:
    def test_tp1_logits(self):
        model = _qwen3_moe_descriptor()
        shape = QWEN3_MOE_FAMILY.logits_shape(model, tensor_parallel=1)
        assert shape.name == "lm_head"
        assert shape.n == 151936
        assert shape.k == 2048


class TestLinearAttention:
    def test_no_linear_attention_layers(self):
        """Every layer is plain softmax attention: there is no gated-delta-net block here."""
        model = _qwen3_moe_descriptor()
        assert QWEN3_MOE_FAMILY.linear_attention_gemm_shapes(model, tensor_parallel=1) is None


class TestValidationErrors:
    def test_zero_tensor_parallel_raises(self):
        model = _qwen3_moe_descriptor()
        with pytest.raises(ProfileValidationError, match="tensor_parallel"):
            QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=0)

    def test_missing_num_experts_per_tok_raises(self):
        extra = {
            "head_dim": 128, "vocab_size": 151936,
            "num_experts": 128, "moe_intermediate_size": 768,
        }
        model = ModelDescriptor(
            name="test", architecture="Qwen3MoeForCausalLM",
            num_layers=48, hidden_size=2048, num_attention_heads=32,
            num_query_groups=4, extra=extra,
        )
        with pytest.raises(ProfileValidationError, match="num_experts_per_tok"):
            QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)

    def test_missing_moe_intermediate_size_raises(self):
        extra = {
            "head_dim": 128, "vocab_size": 151936,
            "num_experts": 128, "num_experts_per_tok": 8,
        }
        model = ModelDescriptor(
            name="test", architecture="Qwen3MoeForCausalLM",
            num_layers=48, hidden_size=2048, num_attention_heads=32,
            num_query_groups=4, extra=extra,
        )
        with pytest.raises(ProfileValidationError, match="moe_intermediate_size"):
            QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)

    def test_hybrid_attention_is_refused(self):
        """A variant declaring full_attention_interval needs the hybrid family instead."""
        model = _qwen3_moe_descriptor(full_attention_interval=4)
        with pytest.raises(ProfileValidationError, match="full_attention_interval"):
            QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)

    def test_shared_expert_is_refused(self):
        """A variant declaring a shared expert needs its own GEMM pair this family omits."""
        model = _qwen3_moe_descriptor(shared_expert_intermediate_size=512)
        with pytest.raises(ProfileValidationError, match="shared_expert_intermediate_size"):
            QWEN3_MOE_FAMILY.gemm_shapes(model, tensor_parallel=1)

    def test_hybrid_attention_is_refused_in_logits_shape_too(self):
        """Both entry points must refuse, or a caller could price half the model."""
        model = _qwen3_moe_descriptor(full_attention_interval=4)
        with pytest.raises(ProfileValidationError, match="full_attention_interval"):
            QWEN3_MOE_FAMILY.logits_shape(model, tensor_parallel=1)

    def test_shared_expert_is_refused_in_head_counts_too(self):
        model = _qwen3_moe_descriptor(shared_expert_intermediate_size=512)
        with pytest.raises(ProfileValidationError, match="shared_expert_intermediate_size"):
            QWEN3_MOE_FAMILY.head_counts(model, tensor_parallel=1)
