"""Per-family op-list derivation, and its refusal to price what it cannot model."""

from __future__ import annotations

import pytest

from dryrun.profiler.kernel.grid.families import FAMILY_REGISTRY, family_for, register_family
from dryrun.profiler.kernel.grid.families_dense import DENSE_LLAMA_FAMILY
from dryrun.profiler.schema import ModelDescriptor, ProfileValidationError


def _qwen3_like(**extra) -> ModelDescriptor:
    """Return a Qwen3-8B-shaped descriptor, the family's motivating case."""
    return ModelDescriptor(
        name="qwen3-like",
        architecture="Qwen3ForCausalLM",
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        extra={"head_dim": 128, "intermediate_size": 12288, "vocab_size": 151936, **extra},
    )


def test_a_registered_architecture_resolves_to_its_family():
    assert family_for(_qwen3_like()) is DENSE_LLAMA_FAMILY


def test_the_dense_family_covers_the_llama_and_qwen_architectures():
    """These four are the dense architectures whose shapes were verified against real modules."""
    for architecture in (
        "Qwen3ForCausalLM",
        "Qwen2ForCausalLM",
        "LlamaForCausalLM",
        "MistralForCausalLM",
    ):
        model = _qwen3_like()
        model = ModelDescriptor(
            name=model.name,
            architecture=architecture,
            num_layers=model.num_layers,
            hidden_size=model.hidden_size,
            num_attention_heads=model.num_attention_heads,
            num_query_groups=model.num_query_groups,
            extra=dict(model.extra),
        )
        assert family_for(model) is DENSE_LLAMA_FAMILY


def test_an_unregistered_architecture_is_refused_naming_where_to_extend():
    """Silently pricing an unknown architecture as dense is the failure this prevents."""
    model = _qwen3_like()
    unknown = ModelDescriptor(
        name=model.name,
        architecture="SomeExoticForCausalLM",
        num_layers=model.num_layers,
        hidden_size=model.hidden_size,
        num_attention_heads=model.num_attention_heads,
        num_query_groups=model.num_query_groups,
        extra=dict(model.extra),
    )
    with pytest.raises(ProfileValidationError, match="SomeExoticForCausalLM"):
        family_for(unknown)


def test_a_missing_architecture_is_refused():
    """A descriptor with no architecture cannot be dispatched, so it must not be guessed."""
    model = ModelDescriptor(
        name="anonymous",
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        extra={"head_dim": 128, "intermediate_size": 12288, "vocab_size": 151936},
    )
    with pytest.raises(ProfileValidationError, match="architecture"):
        family_for(model)


def test_the_dense_family_emits_the_four_fused_projections():
    """These are vLLM's fused shapes, verified against Qwen3-8B's real module graph."""
    shapes = DENSE_LLAMA_FAMILY.gemm_shapes(_qwen3_like(), tensor_parallel=1)
    assert {shape.name: (shape.n, shape.k) for shape in shapes} == {
        "qkv_proj": (6144, 4096),
        "o_proj": (4096, 4096),
        "gate_up_proj": (24576, 4096),
        "down_proj": (4096, 12288),
    }


def test_the_logits_shape_is_the_vocab_projection():
    shape = DENSE_LLAMA_FAMILY.logits_shape(_qwen3_like(), tensor_parallel=1)
    assert (shape.name, shape.n, shape.k) == ("lm_head", 151936, 4096)


def test_head_counts_are_sharded_by_the_parallelism_degree():
    assert DENSE_LLAMA_FAMILY.head_counts(_qwen3_like(), tensor_parallel=1) == (32, 8)
    assert DENSE_LLAMA_FAMILY.head_counts(_qwen3_like(), tensor_parallel=4) == (8, 2)


def test_registering_a_duplicate_architecture_is_refused():
    """Two families claiming one architecture would make dispatch depend on import order."""
    with pytest.raises(ProfileValidationError, match="already"):
        register_family("Qwen3ForCausalLM", DENSE_LLAMA_FAMILY)


def test_the_registry_is_not_empty_and_every_entry_is_a_family():
    assert FAMILY_REGISTRY
    for architecture, family in FAMILY_REGISTRY.items():
        assert architecture
        assert hasattr(family, "gemm_shapes")
        assert hasattr(family, "logits_shape")
        assert hasattr(family, "head_counts")


def test_flashinfer_decode_group_sizes_exclude_this_checkpoints_ratio():
    """
    A 40:8 GQA model has no FlashInfer decode kernel, at any tensor-parallel degree.

    FlashInfer dispatches the GQA ratio as a template parameter and compiles
    specializations only for the set declared here (`DISPATCH_GQA_GROUP_SIZE` in
    `flashinfer/data/include/flashinfer/utils.cuh`); anything else raises
    "Unsupported group_size" from inside the generated kernel. This is asserted rather
    than discovered at measurement time because the runner skips the generation phase on
    the strength of it, and because the invariance across TP is the non-obvious part:
    `rank_head_counts` shards query and KV heads by the same degree, so the ratio cannot
    be changed by choosing a different topology.

    DeepSeek-R1-Distill-Qwen-14B is the motivating case at 40 query heads to 8 KV heads.
    """
    from dryrun.profiler.kernel.grid.attention import rank_head_counts
    from dryrun.profiler.kernel.runner._flashinfer_attn import (
        _FLASHINFER_DECODE_GROUP_SIZES,
    )

    model = ModelDescriptor(
        name="deepseek-r1-distill-qwen-14b-like",
        architecture="Qwen2ForCausalLM",
        num_layers=48,
        hidden_size=5120,
        num_attention_heads=40,
        num_query_groups=8,
        extra={"intermediate_size": 13824, "vocab_size": 152064, "model_type": "qwen2"},
    )

    for tensor_parallel in (1, 2, 4, 8):
        heads, kv_heads = rank_head_counts(model, tensor_parallel=tensor_parallel)
        assert heads // kv_heads == 5
        assert 5 not in _FLASHINFER_DECODE_GROUP_SIZES

    # Qwen3-8B's 32:8 is supported, which is why its sweep writes four tables and this
    # checkpoint's writes three.
    heads, kv_heads = rank_head_counts(_qwen3_like(), tensor_parallel=1)
    assert heads // kv_heads in _FLASHINFER_DECODE_GROUP_SIZES
