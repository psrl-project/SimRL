"""
The Qwen3.5-MoE family: a routed mixture-of-experts block with hybrid attention.

**Why this cannot reuse `dense_llama`.** `families_dense.py` refuses any checkpoint that
declares `num_experts`, and the refusal is correct rather than conservative. Three things
differ, each of which changes the measured GEMM set:

* **The MLP is not one gated pair.** A dense block runs `gate_up_proj` of width
  `2 * intermediate_size` and `down_proj` back. This block runs a router projection
  (`hidden -> num_experts`), then the same gated pair per *selected* expert at
  `moe_intermediate_size`, plus a shared expert that every token traverses. Priced as
  dense, the per-expert GEMMs vanish and the router never appears.
* **Only every `full_attention_interval`-th layer attends over context.** 10 of this
  checkpoint's 40 layers are `full_attention`; the other 30 are `linear_attention`, whose
  cost does not grow with context. A family that reports one attention shape times
  `num_layers` overstates the context-dependent term fourfold.
* **`head_dim` is not `hidden_size / num_heads`.** Here it is 256 against
  `2048 / 16 = 128`, so the qkv and output projections are twice the width the dense
  inference would give.

**What this family reports, and what it deliberately does not.** `gemm_shapes` returns the
shapes of one *full-attention* layer, because that is the layer whose cost the rollout model
scales with context, and the token-count axis of the GEMM table is what prices it. The
per-expert GEMMs are reported at their real `moe_intermediate_size` width with `n` scaled by
`num_experts_per_tok`, which is what one token's MLP actually traverses: the routed experts
are not all visited, and charging all 256 would overstate the block by 32x.

`linear_attention` layers are **not** priced here. Their projections are a different shape
set (`linear_num_key_heads`/`linear_num_value_heads` with a `linear_conv_kernel_dim`
convolution) and the kernel table has no family for an SSM scan. Reporting them as though
they were attention layers would be the exact failure `families.py` exists to prevent, so
`gemm_shapes` covers the attention stack it can price and `n_full_attention_layers` tells
the caller how many layers that is. A caller that needs the whole stack must price the
linear layers separately or refuse.

This module stays free of `torch`: the cost path imports it to price kernels from parquet.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...schema import ModelDescriptor, ProfileValidationError
from ..utils.model_dims import model_head_dim, model_vocab_size
from .families import register_family
from .shape import GemmShape


def _require_divisible(value: int, divisor: int, *, name: str) -> int:
    if value % divisor:
        raise ProfileValidationError(
            f"{name} ({value}) must be divisible by tensor_parallel ({divisor})."
        )
    return value // divisor


def _required(model: ModelDescriptor, key: str) -> int:
    """Read one architecture value, refusing rather than defaulting when absent."""
    value = model.extra.get(key)
    if value is None:
        raise ProfileValidationError(
            f"Model is missing {key!r}, which this family needs to derive its GEMM shapes. "
            f"It is read from config.json (or its nested text_config)."
        )
    return int(value)


@dataclass(frozen=True)
class Qwen35MoeFamily:
    """Routed-MoE blocks with hybrid linear/full attention.

    Frozen for the same reason `DenseLlamaFamily` is: the instance below is a
    module-level singleton reached through the registry, so a mutation would change
    what every model of this architecture is priced against.
    """

    name: str = "qwen35_moe"

    def n_full_attention_layers(self, model: ModelDescriptor) -> int:
        """
        Return how many of the model's layers attend over the full context.

        The remaining layers are `linear_attention` and their cost does not grow with
        context, so a caller scaling a per-layer attention cost by `num_layers` would
        overstate the context term by the interval factor.

        Args:
            model (ModelDescriptor): Architecture read from the checkpoint.

        Returns:
            int: `num_layers // full_attention_interval`.

        Raises:
            ProfileValidationError: If `num_layers` or `full_attention_interval` is absent.
        """
        if model.num_layers is None:
            raise ProfileValidationError("Model is missing num_layers.")
        interval = _required(model, "full_attention_interval")
        if interval <= 0:
            raise ProfileValidationError(
                f"full_attention_interval must be positive, got {interval!r}."
            )
        return model.num_layers // interval

    def gemm_shapes(self, model: ModelDescriptor, *, tensor_parallel: int) -> tuple[GemmShape, ...]:
        """
        Return one full-attention MoE layer's per-rank GEMM shapes.

        The MLP shapes are per *token*, not per expert: `num_experts_per_tok` of the
        `num_experts` are selected, so a token traverses that many gated pairs. Charging
        all of them would overstate the block by `num_experts / num_experts_per_tok`
        (32x on this checkpoint).

        Args:
            model (ModelDescriptor): Architecture read from the checkpoint.
            tensor_parallel (int): Ranks the block is sharded across.

        Returns:
            tuple[GemmShape, ...]: `qkv_proj`, `o_proj`, `router`, `expert_gate_up`,
                `expert_down`, `shared_gate_up`, `shared_down`.

        Raises:
            ProfileValidationError: If the architecture is incomplete or a dimension is
                not divisible by `tensor_parallel`.
        """
        if tensor_parallel <= 0:
            raise ProfileValidationError(
                f"tensor_parallel must be positive, got {tensor_parallel!r}."
            )
        if model.hidden_size is None or model.num_attention_heads is None:
            raise ProfileValidationError("Model is missing hidden_size or num_attention_heads.")
        hidden_size = model.hidden_size
        head_dim = model_head_dim(model)
        heads = _require_divisible(
            model.num_attention_heads, tensor_parallel, name="num_attention_heads"
        )
        kv_heads = _require_divisible(
            model.num_query_groups or model.num_attention_heads,
            tensor_parallel,
            name="num_query_groups",
        )
        experts_per_tok = _required(model, "num_experts_per_tok")
        moe_inter = _require_divisible(
            _required(model, "moe_intermediate_size"),
            tensor_parallel,
            name="moe_intermediate_size",
        )
        num_experts = _required(model, "num_experts")

        shapes = [
            # `attn_output_gate` doubles the query rows: the projection emits a gate
            # alongside the query and the two are split after the GEMM. Verified against
            # Qwen3.5-35B-A3B's module graph, where `q_proj` is (8192, 2048) at 16 heads
            # and head_dim 256 -- twice the 4096 an ungated projection would hold. Without
            # the factor this shape understates the fused QKV GEMM by 1.8x (5120 against
            # the real 9216 rows), which `kernels verify-shapes` reports as a mismatch.
            GemmShape(
                "qkv_proj",
                (heads * (2 if model.extra.get("attn_output_gate") else 1) + 2 * kv_heads)
                * head_dim,
                hidden_size,
            ),
            GemmShape("o_proj", hidden_size, heads * head_dim),
            # The router is not sharded: every rank scores every expert so the
            # dispatch decision is identical across ranks.
            GemmShape("router", num_experts, hidden_size),
            # One token's routed MLP: `experts_per_tok` gated pairs at the MoE width.
            GemmShape("expert_gate_up", 2 * moe_inter * experts_per_tok, hidden_size),
            GemmShape("expert_down", hidden_size, moe_inter * experts_per_tok),
        ]
        # The shared expert is optional in the family: Qwen3.5-MoE has one that every
        # token traverses in addition to its routed experts, but a sibling architecture
        # with `shared_expert_intermediate_size` absent runs only the routed path.
        shared = model.extra.get("shared_expert_intermediate_size")
        if shared is not None:
            shared_inter = _require_divisible(
                int(shared), tensor_parallel, name="shared_expert_intermediate_size"
            )
            shapes.extend(
                (
                    GemmShape("shared_gate_up", 2 * shared_inter, hidden_size),
                    GemmShape("shared_down", hidden_size, shared_inter),
                )
            )
        return tuple(shapes)

    def logits_shape(self, model: ModelDescriptor, *, tensor_parallel: int) -> GemmShape:
        """
        Return the per-rank vocabulary projection.

        Identical in form to the dense family's: vLLM's `ParallelLMHead` partitions the
        vocabulary over ranks, and the head runs once per step rather than per layer.

        Args:
            model (ModelDescriptor): Architecture read from the checkpoint.
            tensor_parallel (int): Ranks the head is sharded across.

        Returns:
            GemmShape: The `lm_head` projection.

        Raises:
            ProfileValidationError: If `hidden_size` is missing or the vocabulary is not
                divisible by `tensor_parallel`.
        """
        if tensor_parallel <= 0:
            raise ProfileValidationError(
                f"tensor_parallel must be positive, got {tensor_parallel!r}."
            )
        if model.hidden_size is None:
            raise ProfileValidationError("Model is missing hidden_size.")
        vocab_per_rank = _require_divisible(
            model_vocab_size(model), tensor_parallel, name="vocab_size"
        )
        return GemmShape("lm_head", vocab_per_rank, model.hidden_size)

    def head_counts(self, model: ModelDescriptor, *, tensor_parallel: int) -> tuple[int, int]:
        """
        Return this rank's `(num_heads, num_kv_heads)` for the full-attention layers.

        Not delegated to `rank_head_counts`: that helper derives `head_dim` from
        `hidden_size / num_heads`, which is wrong here (256 against 128), and a caller
        that reconstructed the projection width from these counts would halve it.

        Args:
            model (ModelDescriptor): Architecture read from the checkpoint.
            tensor_parallel (int): Ranks the block is sharded across.

        Returns:
            tuple[int, int]: Per-rank query and key/value head counts.

        Raises:
            ProfileValidationError: If a count is not divisible by `tensor_parallel`.
        """
        if model.num_attention_heads is None:
            raise ProfileValidationError("Model is missing num_attention_heads.")
        heads = _require_divisible(
            model.num_attention_heads, tensor_parallel, name="num_attention_heads"
        )
        kv_heads = _require_divisible(
            model.num_query_groups or model.num_attention_heads,
            tensor_parallel,
            name="num_query_groups",
        )
        return heads, kv_heads


QWEN35_MOE_FAMILY = Qwen35MoeFamily()

# `Qwen3_5MoeForConditionalGeneration` is the multimodal wrapper the checkpoint declares;
# the vision tower is not priced here because the rollout traces are text-only. The text
# backbone is what `Qwen3_5MoeForCausalLM` names, registered alongside so a text-only
# release of the same architecture resolves to the same shapes.
for _architecture in (
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
):
    register_family(_architecture, QWEN35_MOE_FAMILY)
