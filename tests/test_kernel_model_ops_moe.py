"""ModelOps's routed-MoE wiring: `moe_layers`/`moe_queries`, and the retired analytical path.

Synthetic descriptors rather than a real checkpoint, mirroring `tests/test_kernel_ops_gdn.py`
and `tests/test_kernel_families_moe.py`: this exercises `ModelOps`'s own sharding and query
construction, not a profiled table, so no real checkpoint or GPU is needed.
"""

from __future__ import annotations

from srl.cost.kernel.model_ops import ModelOps
from srl.profiler.kernel.grid.gemm import linear_attention_gemm_shapes, transformer_gemm_shapes
from srl.profiler.schema import ModelDescriptor


def _qwen3_moe_descriptor(
    *,
    num_layers: int = 8,
    hidden_size: int = 2048,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 4,
    head_dim: int = 128,
    vocab_size: int = 151936,
    num_experts: int = 128,
    num_experts_per_tok: int = 8,
    moe_intermediate_size: int = 768,
) -> ModelDescriptor:
    """A plain (non-hybrid) routed-MoE model, matching Qwen3-30B-A3B's architecture."""
    return ModelDescriptor(
        name="test-qwen3-moe",
        architecture="Qwen3MoeForCausalLM",
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_key_value_heads,
        extra={
            "head_dim": head_dim,
            "vocab_size": vocab_size,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "moe_intermediate_size": moe_intermediate_size,
        },
    )


def _qwen35_moe_descriptor(
    *,
    num_layers: int = 8,
    hidden_size: int = 2048,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 2,
    head_dim: int = 256,
    vocab_size: int = 248320,
    num_experts: int = 256,
    num_experts_per_tok: int = 8,
    moe_intermediate_size: int = 512,
    full_attention_interval: int = 4,
    shared_expert_intermediate_size: int = 512,
) -> ModelDescriptor:
    """A hybrid full-attention/GDN routed-MoE model, matching Qwen3.5-35B-A3B's architecture."""
    return ModelDescriptor(
        name="test-qwen35-moe",
        architecture="Qwen3_5MoeForConditionalGeneration",
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_key_value_heads,
        extra={
            "head_dim": head_dim,
            "vocab_size": vocab_size,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "moe_intermediate_size": moe_intermediate_size,
            "full_attention_interval": full_attention_interval,
            "shared_expert_intermediate_size": shared_expert_intermediate_size,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        },
    )


def _dense_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        name="test-dense",
        architecture="Qwen3ForCausalLM",
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        extra={"head_dim": 128, "vocab_size": 151936},
    )


def test_moe_layers_is_every_layer_for_a_plain_moe_model():
    model = _qwen3_moe_descriptor(num_layers=8)
    ops = ModelOps(model, tensor_parallel=1)
    assert ops.moe_layers == 8


def test_moe_layers_is_every_layer_for_a_hybrid_moe_plus_gdn_model():
    """The MoE block is shared by full-attention AND GDN layers, so the multiplier is
    `num_layers`, not `attention_layers` alone -- this is the case that actually exercises
    that claim, since a plain MoE model has no GDN layers to confuse it with.
    """
    model = _qwen35_moe_descriptor(num_layers=8, full_attention_interval=4)
    ops = ModelOps(model, tensor_parallel=1)
    assert ops.attention_layers == 2
    assert ops.gdn_layers == 6
    assert ops.moe_layers == 8


def test_moe_layers_is_zero_for_a_dense_model():
    ops = ModelOps(_dense_descriptor(), tensor_parallel=1)
    assert ops.moe_layers == 0
    assert ops.moe_queries(token_count=64) == ()


def test_moe_query_carries_this_ranks_sharded_shape():
    model = _qwen3_moe_descriptor(num_layers=8)
    ops = ModelOps(model, tensor_parallel=2)
    query = ops.moe_query(num_tokens=64)
    assert query.num_tokens == 64
    assert query.hidden_size == 2048
    assert query.intermediate_size == 768 // 2
    assert query.num_experts == 128
    assert query.top_k == 8


def test_moe_queries_carries_the_layer_count_multiplier():
    model = _qwen3_moe_descriptor(num_layers=8)
    ops = ModelOps(model, tensor_parallel=1)
    queries = ops.moe_queries(token_count=64)
    assert len(queries) == 1
    query, multiplier = queries[0]
    assert multiplier == 8
    assert query.num_tokens == 64


def test_gemm_queries_no_longer_prices_the_routed_experts_directly():
    """The two expert shapes are dropped from `gemm_queries`: they are priced by
    `moe_queries` off a real measured table instead. Compared against the family's own
    shape count rather than a hand-written number, so a future shape addition cannot make
    this test pass by accident.
    """
    model = _qwen3_moe_descriptor(num_layers=8)
    ops = ModelOps(model, tensor_parallel=1)
    full_shape_count = len(transformer_gemm_shapes(model, tensor_parallel=1))
    queries = ops.gemm_queries(token_count=64, request_count=8)
    # -2 for the dropped expert_gate_up/expert_down, +1 for the once-per-step logits query.
    assert len(queries) == full_shape_count - 2 + 1


def test_gemm_queries_drops_experts_from_both_layer_kinds_on_a_hybrid_model():
    model = _qwen35_moe_descriptor(num_layers=8, full_attention_interval=4)
    ops = ModelOps(model, tensor_parallel=1)
    full_attention_shapes = len(transformer_gemm_shapes(model, tensor_parallel=1))
    linear_shapes = len(linear_attention_gemm_shapes(model, tensor_parallel=1))
    queries = ops.gemm_queries(token_count=64, request_count=8)
    assert len(queries) == (full_attention_shapes - 2) + (linear_shapes - 2) + 1


def test_the_retired_analytical_expert_correction_is_gone():
    """The old GEMM-table-plus-correction methods must not linger as dead code."""
    ops = ModelOps(_qwen3_moe_descriptor(), tensor_parallel=1)
    for name in (
        "expert_weight_bytes",
        "expert_weight_bytes_counted",
        "tokens_per_expert",
        "_activated_experts",
    ):
        assert not hasattr(ops, name), f"ModelOps.{name} should have been retired"
