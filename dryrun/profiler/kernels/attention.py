"""Enumerate the attention shapes to measure, and truncate what will not fit.

The sweep grid is deliberately ragged. A large batch crossed with a long sequence is
a point no real deployment reaches and that would OOM on the way to measuring it, so
the memory budgets in `AttentionSweep` drop those corners. Truncating by an explicit
budget rather than by catching OOM keeps the grid reproducible across machines.

Points are ordered largest-footprint first so that a sweep which is going to run out
of memory does so in its first seconds rather than its last hour.

Timing lives in `_vllm_attn` and `_flashinfer_attn`; this module stays importable
without a GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from ..schema import ModelDescriptor, ProfileValidationError
from .config import AttentionSweep, KernelProfileConfig, model_head_dim
from .measurements import PerfTableWriter


@dataclass(frozen=True)
class AttentionPoint:
    """One attention shape to measure."""

    batch_size: int
    seq_len: int
    step: int


def rank_head_counts(model: ModelDescriptor, *, tensor_parallel: int) -> tuple[int, int]:
    """
    Return the per-rank query and key/value head counts.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        tensor_parallel (int): Ranks attention is sharded across.

    Returns:
        tuple[int, int]: `(num_heads, num_kv_heads)` on one rank.

    Raises:
        ProfileValidationError: If the architecture is incomplete or head counts are
            not divisible by `tensor_parallel`.
    """
    if tensor_parallel <= 0:
        raise ProfileValidationError(f"tensor_parallel must be positive, got {tensor_parallel!r}.")
    if model.num_attention_heads is None:
        raise ProfileValidationError("Model is missing num_attention_heads.")
    heads = model.num_attention_heads
    kv_heads = model.num_query_groups or heads
    for name, value in (("num_attention_heads", heads), ("num_query_groups", kv_heads)):
        if value % tensor_parallel:
            raise ProfileValidationError(
                f"{name} ({value}) must be divisible by tensor_parallel ({tensor_parallel})."
            )
    return heads // tensor_parallel, kv_heads // tensor_parallel


def context_sweep_points(sweep: AttentionSweep) -> tuple[AttentionPoint, ...]:
    """
    Enumerate prefill points within the token budget.

    Args:
        sweep (AttentionSweep): Batch sizes, sequence lengths, and the token budget.

    Returns:
        tuple[AttentionPoint, ...]: Points with `step == 0`, largest footprint first.
    """
    points = [
        AttentionPoint(batch_size=batch_size, seq_len=seq_len, step=0)
        for batch_size, seq_len in product(sweep.batch_sizes, sweep.context_seq_lens)
        if batch_size * seq_len <= sweep.max_context_tokens
    ]
    return tuple(sorted(points, key=lambda point: -(point.batch_size * point.seq_len)))


def generation_sweep_points(
    sweep: AttentionSweep,
    *,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[AttentionPoint, ...]:
    """
    Enumerate decode points within the KV element budget.

    Args:
        sweep (AttentionSweep): Batch sizes, KV history lengths, and the KV budget.
        num_kv_heads (int): Key/value heads on this rank.
        head_dim (int): Per-head dimension.

    Returns:
        tuple[AttentionPoint, ...]: Points with `seq_len == 1`, largest footprint first.
    """
    points = [
        AttentionPoint(batch_size=batch_size, seq_len=1, step=step)
        for batch_size, step in product(sweep.batch_sizes, sweep.generation_steps)
        if batch_size * max(step, 1) * num_kv_heads * head_dim <= sweep.max_generation_kv_tokens
    ]
    return tuple(sorted(points, key=lambda point: -(point.batch_size * max(point.step, 1))))


def attention_head_dim(model: ModelDescriptor) -> int:
    """
    Return the per-head dimension of this model.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.

    Returns:
        int: Head dimension.
    """
    return model_head_dim(model)


def profile_vllm_attention(
    model: ModelDescriptor,
    config: KernelProfileConfig,
    *,
    writer_factory: Callable[[str, str, str], PerfTableWriter],
) -> tuple[Path, Path]:
    """
    Measure both attention phases through the backend vLLM selects.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        config (KernelProfileConfig): Sweep grid, dtypes, and iteration counts.
        writer_factory (Callable[[str, str, str], PerfTableWriter]): Builds a writer
            from `(framework, version, kernel_source)`. The kernel source is only known
            after backend selection, so the writer cannot be passed in directly.

    Returns:
        tuple[Path, Path]: The written context and generation table paths.
    """
    from ._vllm_attn import profile_vllm_attention as _impl

    return _impl(model, config, writer_factory=writer_factory)


def profile_flashinfer_attention(
    model: ModelDescriptor,
    config: KernelProfileConfig,
    *,
    writer_factory: Callable[[str, str, str], PerfTableWriter],
) -> tuple[Path, Path]:
    """
    Measure both attention phases through FlashInfer's paged wrappers.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        config (KernelProfileConfig): Sweep grid, dtypes, and iteration counts.
        writer_factory (Callable[[str, str, str], PerfTableWriter]): Builds a writer
            from `(framework, version, kernel_source)`.

    Returns:
        tuple[Path, Path]: The written context and generation table paths.
    """
    from ._flashinfer_attn import profile_flashinfer_attention as _impl

    return _impl(model, config, writer_factory=writer_factory)
