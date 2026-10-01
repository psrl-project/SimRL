"""Tensor-parallel all-reduce profiling; skipped at `tensor_parallel=1`.

A single kernel call cannot exercise multi-rank communication, so this spawns
`tensor_parallel` local processes with `torch.multiprocessing` and times an all-reduce over
per-token hidden activations. This is the same all-reduce vLLM issues after each attention and
MLP block under tensor parallelism.

Two tables come out of one sweep, because **which kernel a real server runs is not NCCL.**
`CudaCommunicator.all_reduce` dispatches through NCCL symmetric memory, then vLLM's custom
one-shot/two-shot P2P kernel, and only reaches `torch.distributed.all_reduce` as a last resort.
On H20 (sm_90) at two ranks the custom path covers everything under 8 MiB, while a Qwen3-8B
decode step's message is 8 KiB to 512 KiB, so every collective a `tp=2` server issues is the
custom kernel and none of them is NCCL. Symmetric memory needs four ranks, so it cannot fire at
two. A table measured with `torch.distributed.all_reduce` therefore describes a kernel the
server never invokes.

So the `vllm` table times `get_tp_group().all_reduce`, which is the dispatcher itself and picks
whatever the server would pick. The `nccl` table times `torch.distributed.all_reduce` directly
and is kept as a control: it is what the cost model used to read, and holding both lets the
bias of that choice be stated as a number rather than argued. `CommQuery.backend` selects
between them, defaulting to `vllm`.

Which backend actually fired is recorded per row in `kernel_source`, read back off the live
communicator rather than re-derived from vLLM's size thresholds: a reimplementation of those
thresholds would drift silently the next time vLLM retunes them.

Measurements land in the same parquet perf-table tree as every other op, so an all-reduce is
priced by the same `SOL / utility` split and gated by the same holdout instrument.

Known gap. The custom kernel is timed outside a CUDA graph, where it pays a `cudaMemcpy` that
the server's graph replay skips (`custom_all_reduce.py:276-280` puts it at <=1% of latency).
The same trap in reverse cost the elementwise table 13.2x before it was timed as a raw op.
"""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..schema import ModelDescriptor, ProfileValidationError
from ._timing import time_cuda_call, torch_dtype
from .config import KernelProfileConfig, MeasurementConfig
from .measurements import PerfTableWriter

dryrun_logger = logging.getLogger(__file__)

# Backend labels written into `kernel_source`, in the order `CudaCommunicator.all_reduce`
# tries them. `torch_all_reduce` is the dispatcher's own final fallback, so seeing it on a
# decode-sized row means the custom path was unavailable and the row is not what a server runs.
#
# Measured on H20 at world sizes 2, 4 and 8 (see examples/profile/kernel/DISPATCH.md):
# `torch_all_reduce` never appears, which is exactly why the old NCCL-only table priced a
# kernel no server invokes. `_SYMM_MEM` is unreachable unless VLLM_USE_NCCL_SYMM_MEM is set,
# since it defaults to 0; the sizes that fall through land on `_TORCH_SYMM_MEM` instead.
_SYMM_MEM = "symm_mem"
_CUSTOM_ALL_REDUCE = "custom_all_reduce"
_TORCH_SYMM_MEM = "torch_symm_mem"
_PYNCCL = "pynccl"
_TORCH_ALL_REDUCE = "torch_all_reduce"

# A table whose rows disagree on which kernel produced them cannot be priced as one surface,
# so the writer takes a single `kernel_source`. This label is used when a sweep spans more
# than one backend, and the per-size labels are logged instead. Every measured world size hits
# this: the custom kernel covers the decode range and larger messages fall through to torch
# symmetric memory, so `mixed` is the normal outcome rather than a warning sign.
_MIXED = "mixed"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _dispatched_backend(tensor: Any) -> str:
    """
    Return the label of the backend `get_tp_group().all_reduce` would use for `tensor`.

    Reads the live communicator's own predicates in the dispatcher's order rather than
    recomputing vLLM's size thresholds, so this cannot drift from the code under measurement.

    Args:
        tensor (Any): The tensor about to be all-reduced, sized as the sweep point.

    Returns:
        str: One of the module's backend labels.
    """
    from vllm.distributed.device_communicators.all_reduce_utils import (
        should_nccl_symm_mem_allreduce,
    )
    from vllm.distributed.parallel_state import get_tp_group

    communicator = get_tp_group().device_communicator
    pynccl_comm = communicator.pynccl_comm
    if pynccl_comm is not None and should_nccl_symm_mem_allreduce(pynccl_comm.world_size, tensor):
        return _SYMM_MEM
    ca_comm = communicator.ca_comm
    if ca_comm is not None and not ca_comm.disabled and ca_comm.should_custom_ar(tensor):
        return _CUSTOM_ALL_REDUCE
    symm_mem_comm = communicator.symm_mem_comm
    if symm_mem_comm is not None and symm_mem_comm.should_use_symm_mem(tensor):
        return _TORCH_SYMM_MEM
    if pynccl_comm is not None and not pynccl_comm.disabled:
        return _PYNCCL
    return _TORCH_ALL_REDUCE


def _collective_worker(
    rank: int,
    world_size: int,
    port: int,
    hidden_size: int,
    token_counts: tuple[int, ...],
    measurement: MeasurementConfig,
    nccl_results: Any,
    vllm_results: Any,
    backends: Any,
) -> None:
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dtype = torch_dtype(measurement.dtype)

    # vLLM's parallel state is what builds the CudaCommunicator, and without it there is no
    # dispatcher to time: only the bare NCCL call that measures the wrong kernel.
    # `initialize_model_parallel` reads `parallel_config` off the ambient VllmConfig, so the
    # setup has to run inside `set_current_vllm_config`, the same way the elementwise profiler
    # needs a live CompilationConfig to reach vLLM's fused kernels instead of native fallbacks.
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    vllm_config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port + 1}",
            local_rank=rank,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
    try:
        tp_group = get_tp_group()
        for index, token_count in enumerate(token_counts):
            tensor = torch.randn(token_count, hidden_size, dtype=dtype, device="cuda")
            nccl_latency_s = time_cuda_call(
                lambda tensor=tensor: dist.all_reduce(tensor),
                warmup_iterations=measurement.warmup_iterations,
                active_iterations=measurement.active_iterations,
            )
            backend = _dispatched_backend(tensor)
            vllm_latency_s = time_cuda_call(
                lambda tensor=tensor: tp_group.all_reduce(tensor),
                warmup_iterations=measurement.warmup_iterations,
                active_iterations=measurement.active_iterations,
            )
            if rank == 0:
                nccl_results[index] = nccl_latency_s
                vllm_results[index] = vllm_latency_s
                backends[index] = backend
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _rows(
    token_counts: tuple[int, ...],
    latencies: Any,
    *,
    hidden_size: int,
    element_bytes: int,
    dtype: str,
    world_size: int,
) -> list[dict[str, Any]]:
    return [
        {
            "dtype": dtype,
            "world_size": world_size,
            "message_bytes": int(token_count) * int(hidden_size) * element_bytes,
            "latency_s": float(latency_s),
        }
        for token_count, latency_s in zip(token_counts, latencies, strict=True)
    ]


def profile_collective(
    model: ModelDescriptor,
    config: KernelProfileConfig,
    *,
    writer_factory: Callable[[str, str, str], PerfTableWriter],
) -> tuple[Path, ...]:
    """
    Time an all-reduce of per-token hidden activations at `config.tensor_parallel` ranks.

    Writes two tables from one sweep: `vllm`, timing the dispatcher a real server uses, and
    `nccl`, timing `torch.distributed.all_reduce` as a control.

    Args:
        model (ModelDescriptor): Provides `hidden_size` for the message size.
        config (KernelProfileConfig): `tensor_parallel` sets the world size;
            `collective.token_counts` sets the message sizes.
        writer_factory (Callable[[str, str, str], PerfTableWriter]): Builds a writer from
            `(framework, version, kernel_source)`.

    Returns:
        tuple[Path, ...]: The written parquet paths, or empty at `tensor_parallel <= 1`
            because no TP all-reduce runs at that world size.

    Raises:
        ProfileValidationError: If the architecture is missing `hidden_size`.
    """
    import torch
    import torch.multiprocessing as mp

    if config.tensor_parallel <= 1:
        dryrun_logger.warning(
            f"tensor_parallel is {config.tensor_parallel}; no TP all-reduce runs at that world "
            f"size, so no collective table is written."
        )
        return ()
    if model.hidden_size is None:
        raise ProfileValidationError("Model is missing hidden_size.")
    token_counts = config.sweep("all_reduce").token_counts
    port = _free_port()
    manager = mp.Manager()
    nccl_results = manager.list([None] * len(token_counts))
    vllm_results = manager.list([None] * len(token_counts))
    backends = manager.list([None] * len(token_counts))
    mp.spawn(
        _collective_worker,
        args=(
            config.tensor_parallel,
            port,
            model.hidden_size,
            token_counts,
            config.measurement,
            nccl_results,
            vllm_results,
            backends,
        ),
        nprocs=config.tensor_parallel,
        join=True,
    )
    element_bytes = torch.empty((), dtype=torch_dtype(config.measurement.dtype)).element_size()
    row_kwargs = {
        "hidden_size": model.hidden_size,
        "element_bytes": element_bytes,
        "dtype": config.measurement.dtype,
        "world_size": config.tensor_parallel,
    }

    dispatched = tuple(backends)
    for token_count, backend in zip(token_counts, dispatched, strict=True):
        message_bytes = int(token_count) * int(model.hidden_size) * element_bytes
        dryrun_logger.info(
            f"all_reduce at {message_bytes} bytes dispatched to {backend} at world size "
            f"{config.tensor_parallel}."
        )
    unique = set(dispatched)
    kernel_source = unique.pop() if len(unique) == 1 else _MIXED
    if kernel_source == _MIXED:
        dryrun_logger.warning(
            f"The all_reduce sweep spans more than one backend ({sorted(set(dispatched))}), so "
            f"kernel_source is {_MIXED!r}. The per-size dispatch is logged above."
        )

    import vllm

    vllm_writer = writer_factory("vllm", str(vllm.__version__), kernel_source)
    nccl_writer = writer_factory(
        "nccl", str(torch.version.cuda or "unknown"), "torch.distributed.all_reduce"
    )
    return (
        vllm_writer.write("all_reduce", _rows(token_counts, vllm_results, **row_kwargs)),
        nccl_writer.write("all_reduce", _rows(token_counts, nccl_results, **row_kwargs)),
    )
