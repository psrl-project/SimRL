"""Grid of attention shapes, with the memory budgets that truncate its corners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...schema import ProfileValidationError
from ..utils.validators import non_negative_int_tuple, positive_int_tuple

# Raggedness budgets, sized against the hardware this profiler targets rather than set to
# a round number. Both count KV *elements*, which cost 4 bytes each at bfloat16 because K
# and V are both stored.
#
# A context point writes `batch_size * seq_len` tokens of KV, so 262144 tokens is a 1 GiB
# write. A generation point reads `batch_size * step * num_kv_heads * head_dim` elements,
# so 1 << 34 is a 64 GiB cache. The largest decode point the H20 sweeps configure
# (batch 64, step 32768, 8 KV heads, head_dim 128) is 8 GiB, which leaves ample headroom
# inside the ~75 GiB an H20 has free after Qwen3-8B's bfloat16 weights.
#
# These defaults are deliberately generous: a budget that is too small silently drops
# sweep points, and a sparse grid starves the holdout gate of the interior points it
# needs. Lower them explicitly in a config when profiling a smaller card.
DEFAULT_MAX_CONTEXT_TOKENS = 262144
DEFAULT_MAX_GENERATION_KV_TOKENS = 1 << 34


@dataclass(frozen=True)
class AttentionSweep:
    """Grid of attention shapes, with the memory budgets that truncate its corners."""

    batch_sizes: tuple[int, ...]
    context_seq_lens: tuple[int, ...]
    generation_steps: tuple[int, ...]
    block_size: int = 16
    window_size: int = 0
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    max_generation_kv_tokens: int = DEFAULT_MAX_GENERATION_KV_TOKENS

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ProfileValidationError(f"attention.block_size must be positive, got {self.block_size!r}.")
        if self.window_size < 0:
            raise ProfileValidationError(f"attention.window_size must not be negative, got {self.window_size!r}.")
        if self.max_context_tokens <= 0:
            raise ProfileValidationError("attention.max_context_tokens must be positive.")
        if self.max_generation_kv_tokens <= 0:
            raise ProfileValidationError("attention.max_generation_kv_tokens must be positive.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AttentionSweep:
        """
        Parse the `attention:` subtree of a kernel profile config.

        Both attention phases read this one block: backend selection precedes both, so a
        per-phase block could name two different block sizes for one KV cache.

        Args:
            value (dict[str, Any]): The `attention:` block.

        Returns:
            AttentionSweep: The parsed grid with its memory budgets.

        Raises:
            ProfileValidationError: If a grid list is empty or malformed, or a budget is
                not positive.
        """
        return cls(
            batch_sizes=positive_int_tuple(value["batch_sizes"], field="attention.batch_sizes"),
            context_seq_lens=positive_int_tuple(
                value["context_seq_lens"], field="attention.context_seq_lens"
            ),
            generation_steps=non_negative_int_tuple(
                value["generation_steps"], field="attention.generation_steps"
            ),
            block_size=int(value.get("block_size", 16)),
            window_size=int(value.get("window_size", 0)),
            max_context_tokens=int(value.get("max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)),
            max_generation_kv_tokens=int(
                value.get("max_generation_kv_tokens", DEFAULT_MAX_GENERATION_KV_TOKENS)
            ),
        )
