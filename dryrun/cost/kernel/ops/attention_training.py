"""
Training attention: THD-packed forward and backward, as Transformer Engine runs them.

This family exists because the rollout `context_attention` table prices a different kernel
on a different layout, and substituting it was measurably wrong in both directions at once.
Measured on H20 for Qwen3-8B at tp=2 (16 query heads, 4 KV heads, head_dim 128):

| tokens | rollout table | TE forward | ratio |
|---|---|---|---|
| 4096 | 0.566 ms | 1.040 ms | 1.84 |
| 12288 | 4.556 ms | 7.540 ms | 1.65 |
| 22528 | 15.113 ms | 24.058 ms | 1.59 |

Two differences compound. The rollout table is vLLM's own prefill kernel over a BSHD batch;
training runs `DotProductAttention(qkv_format="thd")` over one packed row with `cu_seqlens`.
And the rollout sweep's grid is `(batch_size, seq_len)` where a packed training microbatch is
always `batch_size=1` with `seq_len` equal to the whole pack, so every training query landed
at the edge of the measured surface.

**Backward is 2.4x forward, not 1.0x.** `ComposedTraining` previously approximated it at
1.0x on the strength of a cuDNN FusedAttention benchmark. Measured for the kernel that
actually runs:

| tokens | forward | backward | bwd/fwd |
|---|---|---|---|
| 4096 | 1.040 ms | 2.334 ms | 2.24 |
| 12288 | 7.540 ms | 17.958 ms | 2.38 |
| 22528 | 24.058 ms | 58.218 ms | 2.42 |

The ratio is stable and above 2 because the backward computes three gradients (dQ, dK, dV)
where the forward computes one output, and FlashAttention's backward re-materializes the
attention probabilities to avoid storing them. A measured table is used rather than a
corrected scalar because the ratio drifts with sequence length.

**The backend is FlashAttention, not cuDNN.** The Spec B design doc asserted cuDNN
FusedAttention on the strength of TE disabling flash on Hopper+ for the BSHD path. That is
not what happens for THD: TE reports `Running with FlashAttention backend (version 2.8.3)`
and `FusedAttention=False` with the reason `no backend supports the provided input`. So the
op is tagged `flash` and `attn_mask_type` is a slice column, since `padding_causal` is
mandatory for THD (plain `causal` raises) and selects a different kernel path.

**A `.post1` wheel silently disables the whole thing**, which is how this was nearly
mismeasured a second time. TE reads flash-attn's version from package metadata and gates on
`required <= version <= max`; PEP 440 orders `2.8.3.post1` *above* `2.8.3`, so a wheel
matching the upper bound exactly is rejected as too new and TE falls back to
`UnfusedDotProductAttention` without raising. The first measurement of this op ran on that
fallback. `adapters/megatron/launch.py:_normalize_flash_attn_version` is the existing fix and
the profiler calls it before importing TE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..hardware import GpuSpec, dtype_bytes
from ..sol import SolComponents
from .base import register_op

# Mask types THD supports. `padding_causal` is what Megatron passes and what TE requires for
# a packed layout; `padding` covers a non-causal pack. Plain `causal` is absent because
# `DotProductAttention.forward` asserts against it when `qkv_format="thd"`.
TRAINING_MASK_TYPES: tuple[str, ...] = ("padding_causal", "padding")

_SLICE_COLUMNS = (
    "num_key_value_heads",
    "head_dim",
    "attn_dtype",
    "attn_mask_type",
)
_AXIS_COLUMNS = ("num_heads", "total_tokens")

# Backward computes dQ, dK and dV against the forward's single output, and FlashAttention's
# backward recomputes the probability matrix rather than storing it. The measured ratio is
# 2.24-2.42, so the FLOP multiplier below is a declared constant rather than derived: the
# exact kernel-internal recomputation is an implementation detail of the backward and the
# fitted utility absorbs the remainder, as it does for every other SOL simplification.
_BACKWARD_FLOP_MULTIPLIER = 2.5


def _unpack(slice_key: tuple[Any, ...]) -> dict[str, Any]:
    """Return the slice fields both directions share."""
    num_kv_heads, head_dim, attn_dtype, attn_mask_type = slice_key
    if attn_mask_type not in TRAINING_MASK_TYPES:
        known = ", ".join(TRAINING_MASK_TYPES)
        raise ValueError(
            f"Unknown training attention mask type {attn_mask_type!r}. Known types: {known}."
        )
    return dict(
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        attn_dtype=str(attn_dtype),
        causal=attn_mask_type == "padding_causal",
    )


def _training_attention_sol(
    spec: GpuSpec,
    *,
    num_heads: int,
    total_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    attn_dtype: str,
    causal: bool,
    flop_multiplier: float,
    floor_s: float,
) -> SolComponents:
    """Return the roofline of one THD-packed attention call.

    `total_tokens` is the whole pack, and the quadratic term uses it directly. That is an
    over-estimate whenever a pack holds more than one sequence, because attention does not
    cross a `cu_seqlens` boundary: a pack of two 11264-token sequences does
    `2 * 11264^2 / 2` work, not `22528^2 / 2`. The over-estimate is 1/count, and it is
    absorbed by the fitted utility only because the measurement sweeps the same one-sequence
    packs the SOL assumes. A sweep over multi-sequence packs would need the per-sequence
    squares, which is what `SequenceStats.sum_squared_tokens` carries for exactly this
    reason.
    """
    if min(num_heads, total_tokens, num_kv_heads, head_dim) <= 0:
        raise ValueError(
            f"Training attention needs positive dimensions, got num_heads={num_heads!r}, "
            f"total_tokens={total_tokens!r}, num_kv_heads={num_kv_heads!r}, "
            f"head_dim={head_dim!r}."
        )
    attended = total_tokens * 0.5 if causal else float(total_tokens)
    # QK^T and PV, each 2 * h * s * attended * d FLOP, times the direction's multiplier.
    math_s = (
        flop_multiplier
        * 2.0
        * (2.0 * num_heads * total_tokens * attended * head_dim)
        / spec.flops_for(attn_dtype)
    )
    element_bytes = dtype_bytes(attn_dtype)
    query_and_output = 2 * element_bytes * total_tokens * num_heads * head_dim
    key_and_value = 2 * element_bytes * total_tokens * num_kv_heads * head_dim
    memory_s = flop_multiplier * (query_and_output + key_and_value) / (
        spec.memory_bandwidth_bytes_per_s
    )
    return SolComponents(math_s=math_s, memory_s=memory_s, floor_s=floor_s)


@dataclass(frozen=True)
class _TrainingAttentionOp:
    """Shared table shape and sweep for the two directions of training attention.

    `sol()` lives on the base so the forward and backward cannot drift in how they bind
    onto the roofline: they differ only in their FLOP multiplier, which is the same reason
    `_CollectiveOp` has one `sol()` for five collectives.
    """

    family = "attention_training"
    slice_columns = _SLICE_COLUMNS
    axis_columns = _AXIS_COLUMNS
    fits_floor = True

    @property
    def flop_multiplier(self) -> float:
        """Return this direction's work relative to one forward pass."""
        raise NotImplementedError

    def sol(
        self,
        spec: GpuSpec,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """
        Return the roofline of one THD attention call in this direction.

        Args:
            spec (GpuSpec): Hardware the roofline is evaluated against.
            slice_key (tuple[Any, ...]): `(num_key_value_heads, head_dim, attn_dtype,
                attn_mask_type)`.
            coords (tuple[float, ...]): `(num_heads, total_tokens)`.
            floor_s (float): Fitted dispatch floor in seconds, applied inside the `max`.

        Returns:
            SolComponents: Compute, memory, and dispatch-floor components in seconds.

        Raises:
            ValueError: If any dimension is not positive or the mask type is unknown.
        """
        return _training_attention_sol(
            spec,
            num_heads=int(coords[0]),
            total_tokens=int(coords[1]),
            flop_multiplier=self.flop_multiplier,
            floor_s=floor_s,
            **_unpack(slice_key),
        )

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the `attention_training:` sweep subtree."""
        from ....profiler.kernels.config import TrainingAttentionSweep

        return TrainingAttentionSweep.from_dict(value)

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Time both directions in one sweep and write both tables.

        One call for the family, as the attention family already does: a packed input is
        built, the forward timed against it, and the backward timed against the same
        tensors. Measuring per direction would build the inputs twice and could select a
        different backend for each, which is precisely the failure the `.post1` wheel
        produced.
        """
        from ....profiler.kernels.attention_training import profile_training_attention

        return profile_training_attention(model, config, writer_factory=writer_factory)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)


@dataclass(frozen=True)
class TrainingAttentionForwardOp(_TrainingAttentionOp):
    """One THD-packed attention forward, as `DotProductAttention(qkv_format="thd")` runs it."""

    name = "training_attention_forward"

    @property
    def flop_multiplier(self) -> float:
        """Return 1.0: this is the reference direction."""
        return 1.0


@dataclass(frozen=True)
class TrainingAttentionBackwardOp(_TrainingAttentionOp):
    """The backward of the same call, computing dQ, dK and dV."""

    name = "training_attention_backward"

    @property
    def flop_multiplier(self) -> float:
        """Return the backward's work relative to the forward; measured ratio 2.24-2.42."""
        return _BACKWARD_FLOP_MULTIPLIER


TRAINING_ATTENTION_FORWARD_OP = TrainingAttentionForwardOp()
TRAINING_ATTENTION_BACKWARD_OP = TrainingAttentionBackwardOp()
register_op(TRAINING_ATTENTION_FORWARD_OP)
register_op(TRAINING_ATTENTION_BACKWARD_OP)
