"""Architecture keys that disqualify the dense family must reach the descriptor and be refused.

The refusal in `families_dense.py` reads `ModelDescriptor.extra`, so it is only reachable if the
HuggingFace loader forwards the keys that signal a block this family does not describe. Before
this, `load_hf_model_descriptor` forwarded four keys and dropped `num_experts`, which meant a
real mixture-of-experts checkpoint was priced as dense with no error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dryrun.profiler.hf import load_hf_model_descriptor
from dryrun.profiler.kernels.shapes import logits_gemm_shape, transformer_gemm_shapes
from dryrun.profiler.schema import ProfileValidationError

_DENSE = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 12288,
    "vocab_size": 151936,
    "head_dim": 128,
}


def _checkpoint(tmp_path: Path, **overrides) -> Path:
    """Write a config.json and return its directory."""
    directory = tmp_path / "ckpt"
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text(json.dumps({**_DENSE, **overrides}), encoding="utf-8")
    return directory


def test_a_dense_checkpoint_still_prices(tmp_path):
    """The guard must not fire on the family it exists to serve."""
    model = load_hf_model_descriptor(_checkpoint(tmp_path))
    shapes = transformer_gemm_shapes(model, tensor_parallel=1)
    assert {shape.name for shape in shapes} == {"qkv_proj", "o_proj", "gate_up_proj", "down_proj"}


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_experts", 128),
        ("num_local_experts", 8),
        ("n_routed_experts", 256),
        ("num_experts_per_tok", 8),
    ],
)
def test_an_expert_key_reaches_the_descriptor_and_is_refused(tmp_path, key, value):
    """Priced as dense, a MoE block loses its router GEMM and every per-expert GEMM."""
    model = load_hf_model_descriptor(_checkpoint(tmp_path, **{key: value}))
    assert model.extra.get(key) == value, f"{key} must be forwarded into extra"
    with pytest.raises(ProfileValidationError, match=key):
        transformer_gemm_shapes(model, tensor_parallel=1)


@pytest.mark.parametrize("key,value", [("kv_lora_rank", 512), ("q_lora_rank", 1536)])
def test_a_compressed_kv_key_reaches_the_descriptor_and_is_refused(tmp_path, key, value):
    """MLA has no (heads + 2 * kv_heads) * head_dim qkv projection at all."""
    model = load_hf_model_descriptor(_checkpoint(tmp_path, **{key: value}))
    assert model.extra.get(key) == value, f"{key} must be forwarded into extra"
    with pytest.raises(ProfileValidationError, match=key):
        transformer_gemm_shapes(model, tensor_parallel=1)


def test_the_logits_shape_is_refused_too(tmp_path):
    """Both entry points must refuse, or a caller could price half the model."""
    model = load_hf_model_descriptor(_checkpoint(tmp_path, num_experts=128))
    with pytest.raises(ProfileValidationError, match="num_experts"):
        logits_gemm_shape(model, tensor_parallel=1)


def test_the_refusal_names_the_kernel_it_would_omit(tmp_path):
    """A bare 'unsupported' message leaves the reader guessing what is actually missing."""
    model = load_hf_model_descriptor(_checkpoint(tmp_path, num_experts=128))
    with pytest.raises(ProfileValidationError, match="router"):
        transformer_gemm_shapes(model, tensor_parallel=1)


def test_head_counts_are_refused_as_well(tmp_path):
    """ModelOps reads head counts separately, so that path needs the same guard."""
    from dryrun.profiler.kernels.families import family_for

    model = load_hf_model_descriptor(_checkpoint(tmp_path, kv_lora_rank=512))
    with pytest.raises(ProfileValidationError, match="kv_lora_rank"):
        family_for(model).head_counts(model, tensor_parallel=1)


def test_a_moe_architecture_is_also_unregistered(tmp_path):
    """Two independent guards: the architecture is unknown AND the keys disqualify it."""
    model = load_hf_model_descriptor(
        _checkpoint(tmp_path, architectures=["Qwen3MoeForCausalLM"], num_experts=128)
    )
    with pytest.raises(ProfileValidationError, match="Qwen3MoeForCausalLM"):
        transformer_gemm_shapes(model, tensor_parallel=1)
