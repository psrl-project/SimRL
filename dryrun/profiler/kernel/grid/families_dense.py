"""
The dense Llama-style family: four fused projections plus a vocabulary projection.

These four shapes are what an inference engine runs, not what HuggingFace declares. HF keeps
`q_proj`, `k_proj`, and `v_proj` separate and `gate_proj`/`up_proj` separate, giving seven 2-D
weights per layer; vLLM fuses them into `QKVParallelLinear` and `MergedColumnParallelLinear`,
giving four. Verified against both local checkpoints: summing HF's row counts reproduces these
`n` values exactly, so this is the engine's shape rather than an approximation of HF's.

`dryrun/profiler/kernel/introspect.py` is the gate that keeps that claim true: it enumerates
a real module graph on the meta device and compares. This module is the fast path and never
loads a model.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...schema import ModelDescriptor, ProfileValidationError
from ..utils.model_dims import model_head_dim, model_intermediate_size, model_vocab_size
from .attention import rank_head_counts
from .families import register_family
from .shape import GemmShape

# Config keys whose presence means this family would price the model wrongly. Each maps to the
# kernel it would omit or misshape. Refusing is deliberate: a mixture-of-experts block priced
# as dense loses its router GEMM and its per-expert GEMMs entirely, and the resulting latency
# looks reasonable, so nothing downstream would notice.
_DISQUALIFYING_KEYS: dict[str, str] = {
    "num_experts": "a mixture-of-experts block needs a router GEMM and per-expert GEMMs",
    "num_local_experts": "a mixture-of-experts block needs a router GEMM and per-expert GEMMs",
    "n_routed_experts": "a mixture-of-experts block needs a router GEMM and per-expert GEMMs",
    "num_experts_per_tok": "a mixture-of-experts block needs a router GEMM and per-expert GEMMs",
    "kv_lora_rank": "compressed-KV attention does not have a (heads + 2 * kv_heads) qkv projection",
    "q_lora_rank": "compressed-KV attention does not have a (heads + 2 * kv_heads) qkv projection",
}


def _require_divisible(value: int, divisor: int, *, name: str) -> int:
    if value % divisor:
        raise ProfileValidationError(
            f"{name} ({value}) must be divisible by tensor_parallel ({divisor})."
        )
    return value // divisor


@dataclass(frozen=True)
class DenseLlamaFamily:
    """Dense pre-norm transformer blocks with a gated MLP.

    Frozen because the instance below is a module-level singleton reached through the registry,
    so a mutation anywhere would change which shapes every model is priced against.
    """

    name: str = "dense_llama"

    def _reject_unsupported(self, model: ModelDescriptor) -> None:
        """Refuse a model whose block this family does not describe."""
        for key, reason in _DISQUALIFYING_KEYS.items():
            if model.extra.get(key) is None:
                continue
            raise ProfileValidationError(
                f"Model declares {key!r}, so family {self.name!r} would price it wrongly: "
                f"{reason}. Implement a family for this architecture rather than accepting a "
                f"plausible-looking number."
            )

    def gemm_shapes(self, model: ModelDescriptor, *, tensor_parallel: int) -> tuple[GemmShape, ...]:
        """
        Return one layer's per-rank projection shapes, fused as the engine runs them.

        Args:
            model (ModelDescriptor): Architecture read from the checkpoint.
            tensor_parallel (int): Ranks the block is sharded across.

        Returns:
            tuple[GemmShape, ...]: `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`.

        Raises:
            ProfileValidationError: If the architecture is incomplete, a dimension is not
                divisible by `tensor_parallel`, or the model is not a dense gated block.
        """
        if tensor_parallel <= 0:
            raise ProfileValidationError(f"tensor_parallel must be positive, got {tensor_parallel!r}.")
        self._reject_unsupported(model)
        if model.hidden_size is None or model.num_attention_heads is None:
            raise ProfileValidationError("Model is missing hidden_size or num_attention_heads.")
        hidden_size = model.hidden_size
        head_dim = model_head_dim(model)
        intermediate_size = model_intermediate_size(model)
        heads = _require_divisible(model.num_attention_heads, tensor_parallel, name="num_attention_heads")
        kv_heads = _require_divisible(
            model.num_query_groups or model.num_attention_heads,
            tensor_parallel,
            name="num_query_groups",
        )
        inter = _require_divisible(intermediate_size, tensor_parallel, name="intermediate_size")
        return (
            GemmShape("qkv_proj", (heads + 2 * kv_heads) * head_dim, hidden_size),
            GemmShape("o_proj", hidden_size, heads * head_dim),
            GemmShape("gate_up_proj", 2 * inter, hidden_size),
            GemmShape("down_proj", hidden_size, inter),
        )

    def logits_shape(self, model: ModelDescriptor, *, tensor_parallel: int) -> GemmShape:
        """
        Return the per-rank vocabulary projection.

        vLLM shards the LM head over ranks via `ParallelLMHead`, which partitions the vocabulary
        dimension by `tp_size`. It runs once per step rather than once per layer, and its `m` is
        the request count because only each sequence's last token is projected.

        Args:
            model (ModelDescriptor): Architecture read from the checkpoint.
            tensor_parallel (int): Ranks the head is sharded across.

        Returns:
            GemmShape: The `lm_head` projection.

        Raises:
            ProfileValidationError: If `hidden_size` or `vocab_size` is missing, or the
                vocabulary is not divisible by `tensor_parallel`.
        """
        if tensor_parallel <= 0:
            raise ProfileValidationError(f"tensor_parallel must be positive, got {tensor_parallel!r}.")
        self._reject_unsupported(model)
        if model.hidden_size is None:
            raise ProfileValidationError("Model is missing hidden_size.")
        vocab_per_rank = _require_divisible(model_vocab_size(model), tensor_parallel, name="vocab_size")
        return GemmShape("lm_head", vocab_per_rank, model.hidden_size)

    def head_counts(self, model: ModelDescriptor, *, tensor_parallel: int) -> tuple[int, int]:
        """Return this rank's `(num_heads, num_kv_heads)`."""
        self._reject_unsupported(model)
        return rank_head_counts(model, tensor_parallel=tensor_parallel)


DENSE_LLAMA_FAMILY = DenseLlamaFamily()

# Every architecture here was checked against its real module graph, or shares a block shape with
# one that was. Qwen3 and Qwen2 are verified on local checkpoints; Llama and Mistral run the same
# dense gated pre-norm block.
for _architecture in (
    "Qwen3ForCausalLM",
    "Qwen2ForCausalLM",
    "LlamaForCausalLM",
    "MistralForCausalLM",
):
    register_family(_architecture, DENSE_LLAMA_FAMILY)
