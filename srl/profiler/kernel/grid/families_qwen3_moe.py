"""
The Qwen3-MoE family: a routed mixture-of-experts block with plain (non-hybrid) attention.

**Why this cannot reuse `families_moe.py`'s `Qwen35MoeFamily`.** That family is built for
Qwen3.5-MoE specifically, which interleaves full-attention layers among gated-delta-net ones and
carries a shared expert every token traverses. This checkpoint (`Qwen3MoeForCausalLM`, e.g.
Qwen3-30B-A3B) has neither:

* **Every layer attends over the full context.** There is no `full_attention_interval` in this
  architecture's `config.json`, so there is no linear-attention block to price separately and no
  `n // interval` split to get wrong.
* **There is no shared expert.** `shared_expert_intermediate_size` is absent. A token's MLP cost
  is the router plus its `num_experts_per_tok` routed experts and nothing else.
* **`head_dim` is still not `hidden_size / num_attention_heads`** (128 against `2048 / 32 = 64`
  on the 30B-A3B checkpoint), the same override `families_dense.py`'s Qwen3 entry already reads
  through `model_head_dim`.

**Why this cannot reuse `families_dense.py`'s `DenseLlamaFamily` either.** That family refuses
any checkpoint declaring `num_experts`, correctly: a dense block priced over a routed MoE drops
the router GEMM and every per-expert GEMM, which is 93% of this checkpoint's parameters
(30.5B of which only ~3.3B activate per token).

**What this family reports.** `gemm_shapes` returns `qkv_proj`, `o_proj` (plain GQA, no
`attn_output_gate` doubling: this architecture has no such field), `router`, `expert_gate_up`
and `expert_down`. The expert shapes are per *token*, not per expert: `num_experts_per_tok` of
`num_experts` are selected, so charging all of them (128 on the 30B-A3B checkpoint, 16x the
routed amount) would overstate the block by exactly that factor.

Verified against the real module graph on the meta device (`kernels verify-shapes`,
`srl/profiler/kernel/introspect.py`): Qwen3-30B-A3B's experts are a fused 3-D parameter
(`mlp.experts.gate_up_proj` shape `[128, 1536, 2048]`, `mlp.experts.down_proj` shape
`[128, 2048, 768]`), the same layout `introspect.py`'s expert-detection path already handles for
Qwen3.5-MoE, so no changes to that module were needed.

This module stays free of `torch`: the cost path imports it to price kernels from parquet.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...schema import ModelDescriptor, ProfileValidationError
from ..utils.model_dims import model_head_dim, model_vocab_size
from .attention import rank_head_counts
from .families import register_family
from .shape import GemmShape

# Config keys this family does not describe. A checkpoint declaring either would be priced
# wrongly if this family claimed it anyway: hybrid attention needs the linear-attention block
# `families_moe.py` prices, and a shared expert needs its own GEMM pair that this family omits
# entirely. Refusing is the same choice `families_dense.py` makes for `num_experts`.
_DISQUALIFYING_KEYS: dict[str, str] = {
    "full_attention_interval": "hybrid attention needs the linear-attention block this family does not price",
    "shared_expert_intermediate_size": "a shared expert needs its own GEMM pair this family does not price",
}


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
class Qwen3MoeFamily:
    """Routed-MoE blocks with plain (every-layer) softmax attention and no shared expert.

    Frozen for the same reason the other families are: the instance below is a module-level
    singleton reached through the registry, so a mutation would change what every model of this
    architecture is priced against.
    """

    name: str = "qwen3_moe"

    def _reject_unsupported(self, model: ModelDescriptor) -> None:
        """Refuse a model whose block this family does not fully describe."""
        for key, reason in _DISQUALIFYING_KEYS.items():
            if model.extra.get(key) is None:
                continue
            raise ProfileValidationError(
                f"Model declares {key!r}, so family {self.name!r} would price it wrongly: "
                f"{reason}. Implement a family for this variant rather than accepting a "
                f"plausible-looking number."
            )

    def gemm_shapes(self, model: ModelDescriptor, *, tensor_parallel: int) -> tuple[GemmShape, ...]:
        """
        Return one layer's per-rank GEMM shapes.

        The MLP shapes are per *token*, not per expert: `num_experts_per_tok` of the
        `num_experts` are selected, so a token traverses that many gated pairs. Charging all
        of them would overstate the block by `num_experts / num_experts_per_tok` (16x on the
        30B-A3B checkpoint).

        Args:
            model (ModelDescriptor): Architecture read from the checkpoint.
            tensor_parallel (int): Ranks the block is sharded across.

        Returns:
            tuple[GemmShape, ...]: `qkv_proj`, `o_proj`, `router`, `expert_gate_up`,
                `expert_down`.

        Raises:
            ProfileValidationError: If the architecture is incomplete, a dimension is not
                divisible by `tensor_parallel`, or the model needs a family this one is not.
        """
        if tensor_parallel <= 0:
            raise ProfileValidationError(f"tensor_parallel must be positive, got {tensor_parallel!r}.")
        self._reject_unsupported(model)
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
        return (
            GemmShape("qkv_proj", (heads + 2 * kv_heads) * head_dim, hidden_size),
            GemmShape("o_proj", hidden_size, heads * head_dim),
            # Not sharded: every rank scores every expert so the dispatch decision is
            # identical across ranks.
            GemmShape("router", num_experts, hidden_size),
            # One token's routed MLP: `experts_per_tok` gated pairs at the MoE width.
            GemmShape("expert_gate_up", 2 * moe_inter * experts_per_tok, hidden_size),
            GemmShape("expert_down", hidden_size, moe_inter * experts_per_tok),
        )

    def logits_shape(self, model: ModelDescriptor, *, tensor_parallel: int) -> GemmShape:
        """
        Return the per-rank vocabulary projection.

        Args:
            model (ModelDescriptor): Architecture read from the checkpoint.
            tensor_parallel (int): Ranks the head is sharded across.

        Returns:
            GemmShape: The `lm_head` projection.

        Raises:
            ProfileValidationError: If `hidden_size` is missing, the vocabulary is not
                divisible by `tensor_parallel`, or the model needs a family this one is not.
        """
        if tensor_parallel <= 0:
            raise ProfileValidationError(f"tensor_parallel must be positive, got {tensor_parallel!r}.")
        self._reject_unsupported(model)
        if model.hidden_size is None:
            raise ProfileValidationError("Model is missing hidden_size.")
        vocab_per_rank = _require_divisible(
            model_vocab_size(model), tensor_parallel, name="vocab_size"
        )
        return GemmShape("lm_head", vocab_per_rank, model.hidden_size)

    def head_counts(self, model: ModelDescriptor, *, tensor_parallel: int) -> tuple[int, int]:
        """Return this rank's `(num_heads, num_kv_heads)`.

        Delegates to `rank_head_counts`, unlike `Qwen35MoeFamily`: that helper only divides
        head *counts*, never derives `head_dim` from them, so the explicit `head_dim` override
        this architecture also carries does not make it unsafe to reuse here.
        """
        self._reject_unsupported(model)
        return rank_head_counts(model, tensor_parallel=tensor_parallel)

    def linear_attention_gemm_shapes(
        self, model: ModelDescriptor, *, tensor_parallel: int
    ) -> tuple[GemmShape, ...] | None:
        """Return None, because every layer of this block is a softmax attention layer."""
        return None


QWEN3_MOE_FAMILY = Qwen3MoeFamily()

# Verified on a local Qwen3-30B-A3B checkpoint (meta-device module graph, `kernels
# verify-shapes`): standard GQA attention plus fused 3-D expert-stack parameters, no shared
# expert, no hybrid attention. Not registering sibling strings (e.g. a VL wrapper) that have not
# been checked against a real checkpoint.
register_family("Qwen3MoeForCausalLM", QWEN3_MOE_FAMILY)
