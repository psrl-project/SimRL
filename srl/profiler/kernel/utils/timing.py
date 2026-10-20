"""Shared CUDA timing helpers for direct kernel profilers.

The one module under `utils/` that imports `torch` at module scope. It is reached only from
`runner/`, which is already on the GPU path. `grid/` and `sweep/` must not import it.

## Two timing modes, and why the choice is per-family rather than global

`time_cuda_call` brackets N back-to-back calls with CUDA events and divides. That measures
the device only while the device is the slower side. When a kernel's *host* path costs more
than its device work, the same code silently measures the Python interpreter instead, and
the resulting table records an overhead the engine may never pay.

`time_cuda_graph_replay` captures the call into a `torch.cuda.CUDAGraph` once and times
replays, so the host cost is paid at capture and excluded from every timed iteration.

Neither is universally right. The correct mode is the one matching how the stack that will
be simulated runs that kernel, which was read off a live engine rather than assumed. A
Qwen3.5-35B-A3B server resolves `cudagraph_mode=FULL_AND_PIECEWISE` (the v1 default) and
captures two sets of graphs at startup:

- `(decode, FULL)`: the entire forward pass, attention and GDN included, in one graph.
- `(mixed prefill-decode, PIECEWISE)`: graphs around the ops in `splitting_ops`, with
  those ops themselves left outside for flexibility.

So the dispatch cost a kernel pays depends on the PHASE, not only on the family. Decode
replays a captured graph and pays none of it. Prefill and mixed batches run the attention
and GDN ops eagerly, and do pay it. Megatron's training step runs eager throughout, so a
graph-timed row would under-report a launch cost genuinely on its critical path.

GDN is graph-timed because FlashInfer dispatches it through the CUTLASS DSL, which
re-marshals every tensor argument in Python on each call (`generate_execution_args` plus a
`from_dlpack` -> `mark_layout_dynamic` -> `_c_pointers` chain per tensor), and that
interpreter cost dominates the kernel's own device time at small shapes.

**Attention is a known, deliberately unconverted case.** It shows the same host-bound pattern
at decode, and capture was verified to produce identical results, so converting it is
mechanically possible. It is not done because the evidence does not yet say which number is
right: decode replays a FULL graph and would want the replayed value, while prefill and mixed
batches run attention eagerly outside the piecewise graph and would want the event-timed
value. One table cannot serve both, and the composed model reads one table for all phases.
Converting attention wholesale would fix decode and break prefill by the same factor. The
honest fix is a phase-aware table, which is a schema change, not a timing-helper change.

A graph launch is not free, but the cost is below every shape in these grids and would only
matter for a sub-microsecond op.
"""

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

    Measures the device only while the device is the slower side. For a kernel whose host
    dispatch costs more than its device work this returns the host issue rate instead. Use
    `time_cuda_graph_replay` for those. See the module docstring for which families are
    which, and for the measurements behind that split.

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


def time_cuda_graph_replay(
    call: Callable[[], None],
    *,
    warmup_iterations: int,
    active_iterations: int,
) -> float:
    """
    Time one callable as a captured CUDA graph, excluding its host dispatch cost.

    Capture must happen after warmup, not before: the first calls are where a kernel
    autotunes, allocates workspace, and populates the module cache, none of which can be
    recorded into a graph. `active_iterations` replays then measure device work plus the
    graph launch, which is what an engine replaying its own captured graph pays.

    The same tensors are reused across replays, so values drift for a kernel that updates
    state in place. That is deliberate and harmless: the shapes, strides and launch
    configuration are what set the latency, and the engine likewise replays one graph against
    one set of static buffers.

    Args:
        call (Callable[[], None]): Zero-argument callable that launches GPU work and
            performs no host synchronisation, device allocation, or CPU-dependent branching.
        warmup_iterations (int): Untimed calls before capture. At least one is required.
        active_iterations (int): Timed replays averaged into the returned latency.

    Returns:
        float: Mean per-replay latency in seconds.

    Raises:
        ValueError: If `warmup_iterations` is zero, since capturing an un-warmed kernel
            records its autotuning path or fails outright.
        RuntimeError: Propagated from capture when the callable is not capturable. Not
            caught: silently falling back to `time_cuda_call` would write a host-bound row
            into a table whose other rows are device-bound, and nothing downstream could
            tell the two apart.
    """
    if warmup_iterations <= 0:
        raise ValueError(
            "time_cuda_graph_replay needs at least one warmup iteration. Capturing a cold "
            "kernel records its autotuning path instead of its steady state."
        )
    for _ in range(warmup_iterations):
        call()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(active_iterations):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)

    # A graph owns its capture memory pool for as long as it is alive, and the sweeps are
    # ordered largest-footprint first on the assumption that each point's memory is released.
    del graph
    return (elapsed_ms / 1000.0) / active_iterations
