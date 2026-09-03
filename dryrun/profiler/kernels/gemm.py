"""Measure one transformer block's linear layers as a unified GEMM table.

Shapes are sharded by tensor parallelism here, at construction time, so the table
itself is TP-agnostic: it stores `(m, n, k)` and nothing else. That is what lets a
later phase price backward dgrad and wgrad from these same rows by permuting the
axes, with no extra measurement.

Every row's `kernel_source` is `torch.nn.functional.linear`, which is what vLLM's
unquantized bfloat16 linear layers dispatch to.

Shape arithmetic lives in `shapes.py`; this module measures. The names are re-exported here so
existing importers are unaffected.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..schema import ModelDescriptor
from ._timing import time_cuda_call, torch_dtype
from .config import KernelProfileConfig
from .measurements import PerfTableWriter
from .shapes import (
    GemmShape,
    gemm_all_shapes,
    gemm_sweep_shapes,
    gemm_wgrad_shapes,
    logits_gemm_shape,
    logits_sweep_shapes,
    transformer_gemm_shapes,
)

__all__ = [
    "GemmShape",
    "gemm_all_shapes",
    "gemm_sweep_shapes",
    "gemm_wgrad_shapes",
    "logits_gemm_shape",
    "logits_sweep_shapes",
    "profile_gemm",
    "transformer_gemm_shapes",
]


def profile_gemm(
    model: ModelDescriptor,
    config: KernelProfileConfig,
    *,
    writer: PerfTableWriter,
) -> Path:
    """
    Time every swept GEMM shape and write the unified table.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        config (KernelProfileConfig): Sweep grid, dtype, and iteration counts.
        writer (PerfTableWriter): Destination, already carrying the kernel source.

    Returns:
        Path: The written parquet path.
    """
    dtype = torch_dtype(config.measurement.dtype)
    device = torch.device("cuda")
    measurement = config.measurement
    rows: list[dict[str, object]] = []
    for m, n, k in gemm_all_shapes(model, config.sweep("gemm"), tensor_parallel=config.tensor_parallel):
        activation = torch.randn(m, k, dtype=dtype, device=device)
        weight = torch.randn(n, k, dtype=dtype, device=device)
        latency_s = time_cuda_call(
            lambda a=activation, w=weight: torch.nn.functional.linear(a, w),
            warmup_iterations=measurement.warmup_iterations,
            active_iterations=measurement.active_iterations,
        )
        rows.append({"m": m, "n": n, "k": k, "gemm_dtype": measurement.dtype, "latency_s": latency_s})
        del activation, weight
        torch.cuda.empty_cache()
    return writer.write("gemm", rows)
