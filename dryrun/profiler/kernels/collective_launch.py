"""
The torchrun child that times collectives inside one process group.

Split from `collective.py` for the same reason the training adapters split `launch.py` from
their driver: this module is the only one that may import `torch` at module scope, because
it only ever runs inside a `torchrun` child. The driver stays importable by the CLI.

Why torchrun at all, when `torch.multiprocessing.spawn` already worked: `mp.spawn` binds
`MASTER_ADDR` to `127.0.0.1`, so every world size it can produce is intra-node over NVLink.
No arrangement of local processes measures a NIC. Since a data-parallel gradient reduction
crosses nodes in any real deployment, and NVLink is ~9x the per-rank NIC bandwidth on this
cluster, an intra-node-only table cannot price it even approximately.

All five collectives are timed inside one group. Process-group setup and the NCCL handshake
dominate a small-message measurement and a cross-node rendezvous costs seconds, so
amortizing them once across five ops is what makes the sweep feasible. It also guarantees
every row of every table describes the same communicator.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ._timing import time_cuda_call, torch_dtype

dryrun_logger = logging.getLogger(__file__)

# Backend labels, in the order `CudaCommunicator.all_reduce` tries them. Measured on H20 at
# world sizes 2, 4 and 8 (examples/profile/kernel/DISPATCH.md), `torch_all_reduce` never
# appears, which is why a NCCL-only table prices a kernel no server invokes.
_SYMM_MEM = "symm_mem"
_CUSTOM_ALL_REDUCE = "custom_all_reduce"
_TORCH_SYMM_MEM = "torch_symm_mem"
_PYNCCL = "pynccl"
_TORCH_ALL_REDUCE = "torch_all_reduce"


def dispatched_backend(tensor: Any) -> str:
    """
    Return the label of the backend `get_tp_group().all_reduce` would pick for `tensor`.

    Reads the live communicator's own predicates in the dispatcher's order rather than
    recomputing vLLM's size thresholds, so this cannot drift from the code under
    measurement.

    Args:
        tensor (Any): The tensor about to be all-reduced, sized as the sweep point.

    Returns:
        str: One of this module's backend labels.
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


def _nccl_call(op: str, tensor: Any, *, rank: int, world_size: int) -> Any:
    """
    Return a zero-argument callable issuing one collective through torch.distributed.

    Each op is given a tensor shaped so that `message_bytes` means the same thing across
    ops: the bytes one rank contributes. For `reduce_scatter` and `all_gather` that means
    the collective's own buffer is `world_size` times larger, which is the whole point of
    sharding and must not be papered over by handing every op an identically sized tensor.

    Args:
        op (str): Collective op name.
        tensor (Any): The per-rank payload.
        rank (int): This process's rank, used to orient `p2p` and root `broadcast`.
        world_size (int): Ranks in the group.

    Returns:
        Any: A callable launching the collective.

    Raises:
        ValueError: If the op is not one of the five measured collectives.
    """
    if op == "all_reduce":
        return lambda: dist.all_reduce(tensor)
    if op == "broadcast":
        return lambda: dist.broadcast(tensor, src=0)
    if op == "reduce_scatter":
        # Input is world_size shards; output is this rank's reduced shard.
        shards = [torch.empty_like(tensor) for _ in range(world_size)]
        return lambda: dist.reduce_scatter(tensor, shards)
    if op == "all_gather":
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        return lambda: dist.all_gather(gathered, tensor)
    if op == "p2p":
        # Rank 0 sends to rank 1 and rank 1 receives; every other rank idles. Timed on
        # rank 0, so only the send side is recorded, which is what a pipeline stage
        # boundary costs the upstream stage.
        peer = 1 % world_size
        if rank == 0:
            return lambda: dist.send(tensor, dst=peer)
        if rank == peer:
            return lambda: dist.recv(tensor, src=0)
        return lambda: None
    raise ValueError(f"Unsupported collective op {op!r}.")


def _vllm_call(op: str, tensor: Any) -> Any:
    """
    Return a callable issuing one collective through vLLM's own dispatcher, or None.

    vLLM's `GroupCoordinator` exposes the collectives a server actually issues. Ops it does
    not expose are not measured under the `vllm` framework rather than being silently
    substituted with the torch.distributed call: a row labelled `vllm` that timed
    `torch.distributed` would misattribute the kernel, which is the exact defect the
    two-table arrangement exists to avoid.

    Args:
        op (str): Collective op name.
        tensor (Any): The per-rank payload.

    Returns:
        Any: A callable, or `None` when the dispatcher has no entry point for this op.
    """
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    if op == "all_reduce":
        return lambda: group.all_reduce(tensor)
    if op == "all_gather":
        return lambda: group.all_gather(tensor, dim=0)
    if op == "reduce_scatter":
        if not hasattr(group, "reduce_scatter"):
            return None
        return lambda: group.reduce_scatter(tensor, dim=0)
    if op == "broadcast":
        return lambda: group.broadcast(tensor, src=0)
    # No server-level p2p entry point: vLLM's pipeline send/recv goes through a separate
    # group with its own buffers, so the dispatched path is not comparable here.
    return None


def _measure(spec: dict[str, Any], rank: int, world_size: int) -> list[dict[str, Any]]:
    """Time every (op, dtype, message size) point and return rank 0's rows."""
    rows: list[dict[str, Any]] = []
    warmup = int(spec["warmup_iterations"])
    active = int(spec["active_iterations"])
    nodes = int(spec["nodes"])

    for dtype_name in spec["dtypes"]:
        dtype = torch_dtype(dtype_name)
        element_bytes = torch.empty((), dtype=dtype).element_size()
        for message_bytes in spec["message_bytes"]:
            elements = max(1, int(message_bytes) // element_bytes)
            tensor = torch.randn(elements, dtype=dtype, device="cuda")
            for op in spec["ops"]:
                nccl_call = _nccl_call(op, tensor, rank=rank, world_size=world_size)
                # A barrier before timing, unlike the training measure path which avoids
                # one. The difference is deliberate: a collective's cost genuinely includes
                # synchronization, so rank skew must be paid once outside the timed region
                # rather than once per iteration. A training step's cost does not, and
                # barriering there would charge every rank for the slowest.
                dist.barrier()
                nccl_latency_s = time_cuda_call(
                    nccl_call, warmup_iterations=warmup, active_iterations=active
                )
                vllm_call = _vllm_call(op, tensor)
                vllm_latency_s = None
                backend = None
                if vllm_call is not None:
                    backend = dispatched_backend(tensor) if op == "all_reduce" else "dispatcher"
                    dist.barrier()
                    vllm_latency_s = time_cuda_call(
                        vllm_call, warmup_iterations=warmup, active_iterations=active
                    )
                if rank == 0:
                    rows.append(
                        {
                            "op": op,
                            "dtype": dtype_name,
                            "world_size": world_size,
                            "nodes": nodes,
                            "message_bytes": int(message_bytes),
                            "nccl_latency_s": float(nccl_latency_s),
                            "vllm_latency_s": (
                                None if vllm_latency_s is None else float(vllm_latency_s)
                            ),
                            "dispatched_backend": backend,
                        }
                    )
    return rows


def main() -> None:
    """Join the process group, time every collective, and have rank 0 write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to the JSON measurement spec.")
    parser.add_argument("--output", required=True, help="Path rank 0 writes raw rows to.")
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # vLLM's parallel state is what builds the CudaCommunicator, and without it there is no
    # dispatcher to time: only the bare NCCL call, which measures the wrong kernel.
    # `initialize_model_parallel` reads `parallel_config` off the ambient VllmConfig, so the
    # setup has to run inside `set_current_vllm_config`, the same way the elementwise
    # profiler needs a live CompilationConfig to reach vLLM's fused kernels.
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    vllm_config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=(
                f"tcp://{os.environ['MASTER_ADDR']}:{int(os.environ['MASTER_PORT']) + 1}"
            ),
            local_rank=local_rank,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
    try:
        rows = _measure(spec, rank, world_size)
        if rank == 0:
            Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")
            dryrun_logger.info(f"Wrote {len(rows)} collective measurements to {args.output}.")
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
