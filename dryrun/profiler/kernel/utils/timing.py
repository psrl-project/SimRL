"""Shared CUDA-event timing helper for direct kernel profilers."""

from __future__ import annotations

from typing import Callable

import torch

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def torch_dtype(name: str) -> torch.dtype:
    """
    Resolve a config dtype name to a `torch.dtype`.

    Args:
        name (str): One of `float32`, `fp32`, `float16`, `fp16`, `bfloat16`, `bf16`.

    Returns:
        torch.dtype: The matching torch dtype.
    """
    try:
        return _TORCH_DTYPES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}.") from exc


def time_cuda_call(
    call: Callable[[], None],
    *,
    warmup_iterations: int,
    active_iterations: int,
) -> float:
    """
    Time one no-argument callable on the current CUDA device.

    Args:
        call (Callable[[], None]): Zero-argument callable that launches GPU work.
        warmup_iterations (int): Untimed calls before recording, to settle kernel
            autotuning and allocator caches.
        active_iterations (int): Timed calls averaged into the returned latency.

    Returns:
        float: Mean per-call latency in seconds.
    """
    for _ in range(warmup_iterations):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(active_iterations):
        call()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    return (elapsed_ms / 1000.0) / active_iterations
