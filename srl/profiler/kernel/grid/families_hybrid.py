"""
The Qwen3.5 dense hybrid family: softmax attention every fourth layer, gated delta net elsewhere.

`families_dense.py` cannot price this block. It reports one attention shape for every layer,
while this checkpoint runs two layer types. A softmax layer has a query projection carrying an
output gate, and a linear-attention layer has `in_proj_qkvz` and `in_proj_ba` instead. Charging
one shape across all layers prices the other type with the wrong width.

`gemm_shapes` returns the softmax layer. `linear_attention_gemm_shapes` returns the GDN layer,
and a caller must multiply each by its own layer count. The GDN kernel is priced by the `gdn`
op family, not here, because its cost is a scan rather than a projection.

Kept free of `torch`: the cost path imports this module to price kernels from parquet.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...schema import ModelDescriptor, ProfileValidationError
from ..utils.model_dims import model_head_dim, model_intermediate_size, model_vocab_size
from .families import register_family
from .shape import GemmShape


def _require_divisible(value: int, divisor: int, *, name: str) -> int:
    if value % divisor:
        raise ProfileValidationError(
            f"{name} ({value}) must be divisible by tensor_parallel ({divisor})."
        )
    return value // divisor


def _required(model: ModelDescriptor, key: str) -> int:
    value = model.extra.get(key)
    if value is None:
        raise ProfileValidationError(
            f"Model is missing {key!r}, which this family needs to derive its GEMM shapes. "
            f"It is read from config.json (or its nested text_config)."
        )
    return int(value)


@dataclass(frozen=True)
class Qwen35DenseFamily:
    """Dense gated MLP with hybrid softmax and linear attention.

    Frozen because the instance below is a module-level singleton reached through the
    registry, so a mutation would change what every model of this architecture is priced
    against.
    """

    name: str = "qwen35_dense"

    def n_full_attention_layers(self, model: ModelDescriptor) -> int:
        """
        Return how many layers attend over the full context.

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
        Return one full-attention layer's per-rank projection shapes.

        Args:
            model (ModelDescriptor): Architecture read from the checkpoint.
            tensor_parallel (int): Ranks the block is sharded across.

        Returns:
            tuple[GemmShape, ...]: `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`.

        Raises:
            ProfileValidationError: If the architecture is incomplete or a dimension is not
                divisible by `tensor_parallel`.
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
        inter = _require_divisible(
            model_intermediate_size(model), tensor_parallel, name="intermediate_size"
        )
        gate_rows = heads * (2 if model.extra.get("attn_output_gate") else 1)
        return (
            GemmShape("qkv_proj", (gate_rows + 2 * kv_heads) * head_dim, hidden_size),
            GemmShape("o_proj", hidden_size, heads * head_dim),
            GemmShape("gate_up_proj", 2 * inter, hidden_size),
            GemmShape("down_proj", hidden_size, inter),
        )

    def linear_attention_gemm_shapes(
        self, model: ModelDescriptor, *, tensor_parallel: int
    ) -> tuple[GemmShape, ...]:
        """
        Return one gated-delta-net layer's per-rank projection shapes.

        The engine fuses the four small input projections into `in_proj_qkvz` and
        `in_proj_ba`, so those are the shapes a step actually runs.

        Args:
            model (ModelDescriptor): Architecture read from the checkpoint.
            tensor_parallel (int): Ranks the block is sharded across.

        Returns:
            tuple[GemmShape, ...]: `in_proj_qkvz`, `in_proj_ba`, `out_proj`, `gate_up_proj`,
                `down_proj`.

        Raises:
            ProfileValidationError: If the architecture is incomplete or a dimension is not
                divisible by `tensor_parallel`.
        """
        if tensor_parallel <= 0:
            raise ProfileValidationError(
                f"tensor_parallel must be positive, got {tensor_parallel!r}."
            )
        if model.hidden_size is None:
            raise ProfileValidationError("Model is missing hidden_size.")
        hidden_size = model.hidden_size
        key_heads = _require_divisible(
            _required(model, "linear_num_key_heads"), tensor_parallel, name="linear_num_key_heads"
        )
        value_heads = _require_divisible(
            _required(model, "linear_num_value_heads"),
            tensor_parallel,
            name="linear_num_value_heads",
        )
        key_dim = _required(model, "linear_key_head_dim")
        value_dim = int(model.extra.get("linear_value_head_dim") or key_dim)
        inter = _require_divisible(
            model_intermediate_size(model), tensor_parallel, name="intermediate_size"
        )
        qkvz_rows = 2 * key_heads * key_dim + 2 * value_heads * value_dim
        ba_rows = 2 * value_heads
        return (
            GemmShape("in_proj_qkvz", qkvz_rows, hidden_size),
            GemmShape("in_proj_ba", ba_rows, hidden_size),
            GemmShape("out_proj", hidden_size, value_heads * value_dim),
            GemmShape("gate_up_proj", 2 * inter, hidden_size),
            GemmShape("down_proj", hidden_size, inter),
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

        Not delegated to `rank_head_counts`, which derives `head_dim` from
        `hidden_size / num_heads`. Here head_dim is 256 against 5120 / 24, so a caller that
        rebuilt the projection width from these counts would halve it.

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


QWEN35_DENSE_FAMILY = Qwen35DenseFamily()

# `Qwen3_5ForConditionalGeneration` is the multimodal wrapper this checkpoint declares. The
# vision tower is not priced because the rollout traces are text-only. `Qwen3_5ForCausalLM`
# names the same text backbone so a text-only release resolves to identical shapes.
for _architecture in (
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5ForCausalLM",
):
    register_family(_architecture, QWEN35_DENSE_FAMILY)
