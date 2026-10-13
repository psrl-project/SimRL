"""Packed token counts and head counts to measure THD training attention at."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...schema import ProfileValidationError
from ..utils.validators import positive_int_tuple


@dataclass(frozen=True)
class TrainingAttentionSweep:
    """Packed token counts and head counts to measure THD attention at, both directions.

    `total_tokens` rather than `(batch_size, seq_len)`: a THD microbatch is one packed row,
    so batch size is always 1 and the pack length is the axis. That is also why the rollout
    `context_attention` table cannot serve this op even where its shapes overlap, since its
    grid varies the batch dimension that a packed layout does not have.
    """

    total_tokens: tuple[int, ...]
    num_heads: tuple[int, ...]
    mask_types: tuple[str, ...] = ("padding_causal",)

    def __post_init__(self) -> None:
        from ....cost.kernel.ops.attention_training import TRAINING_MASK_TYPES

        unknown = sorted(set(self.mask_types) - set(TRAINING_MASK_TYPES))
        if unknown:
            known = ", ".join(TRAINING_MASK_TYPES)
            raise ProfileValidationError(
                f"attention_training.mask_types holds unknown values {unknown}. "
                f"Known types: {known}. Plain 'causal' is rejected by Transformer Engine "
                f"for qkv_format='thd'."
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingAttentionSweep:
        """
        Parse the `attention_training:` subtree of a kernel profile config.

        Args:
            value (dict[str, Any]): The `attention_training:` block.

        Returns:
            TrainingAttentionSweep: The parsed grid.

        Raises:
            ProfileValidationError: If either integer list is empty or holds a non-positive
                entry, or if `mask_types` names a type THD does not support.
        """
        return cls(
            total_tokens=positive_int_tuple(
                value["total_tokens"], field="attention_training.total_tokens"
            ),
            num_heads=positive_int_tuple(
                value["num_heads"], field="attention_training.num_heads"
            ),
            mask_types=tuple(str(item) for item in value.get("mask_types", ("padding_causal",))),
        )
