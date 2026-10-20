"""
Enumerate the THD attention shapes to measure, both directions.

Extracted from the measurement loop for the KV head derivation, which is the one piece of
arithmetic here that is not a plain cross product:

    num_kv_heads = max(num_heads * total_kv_heads // total_heads, 1)

The sweep declares query head counts, and a GQA model's KV heads must be sharded in the same
ratio or the measured slice is one no query can reach. `rank_head_counts` shards both by the
same `tensor_parallel` on the cost side, so a sweep that pinned `num_kv_heads` to the full
model value would write rows keyed on a head count that only exists at tp=1. The `max(..., 1)`
floor matters at the small end: Qwen3-8B has 32 query heads to 8 KV heads, so a 2-head sweep
point integer-divides to 0 KV heads, which TE rejects.

Points are ordered mask-major then head-major then token-major, matching the order the runner
builds `DotProductAttention` instances in: one module per `(mask_type, num_heads)` pair, reused
across every token count, because constructing it is the expensive part.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...schema import ModelDescriptor, ProfileValidationError
from ..sweep.attention_training import TrainingAttentionSweep
from ..utils.model_dims import model_head_dim


@dataclass(frozen=True)
class TrainingAttentionPoint:
    """One THD attention shape to measure, forward and backward against the same tensors."""

    mask_type: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    total_tokens: int


def training_attention_points(
    model: ModelDescriptor,
    sweep: TrainingAttentionSweep,
) -> tuple[TrainingAttentionPoint, ...]:
    """
    Enumerate every `(mask_type, num_heads, total_tokens)` point with its derived KV heads.

    Args:
        model (ModelDescriptor): Architecture read for `head_dim` and the full head counts
            the GQA ratio is derived from.
        sweep (TrainingAttentionSweep): Mask types, query head counts, and pack lengths.

    Returns:
        tuple[TrainingAttentionPoint, ...]: Points in mask-major, head-major, token-major
            order, matching how the runner reuses one attention module per head count.

    Raises:
        ProfileValidationError: If the architecture is missing `num_attention_heads`.
    """
    if model.num_attention_heads is None:
        raise ProfileValidationError("Model is missing num_attention_heads.")
    head_dim = model_head_dim(model)
    total_heads = int(model.num_attention_heads)
    total_kv_heads = int(model.num_query_groups or total_heads)

    points: list[TrainingAttentionPoint] = []
    for mask_type in sweep.mask_types:
        for num_heads in sweep.num_heads:
            num_kv_heads = max(num_heads * total_kv_heads // total_heads, 1)
            for total_tokens in sweep.total_tokens:
                points.append(
                    TrainingAttentionPoint(
                        mask_type=mask_type,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        total_tokens=int(total_tokens),
                    )
                )
    return tuple(points)
