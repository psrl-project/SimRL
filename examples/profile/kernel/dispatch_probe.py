"""Print which all-reduce backend vLLM dispatches at each message size, per world size.

Diagnostic only: no table is written. Run under torchrun, e.g.
    torchrun --standalone --nproc-per-node 2 examples/profile/kernel/dispatch_probe.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

HIDDEN_SIZE = 4096
TOKEN_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)

    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))
    with set_current_vllm_config(config):
        init_distributed_environment(backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world_size)

    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from dryrun.profiler.kernels.collective import _dispatched_backend

    if rank == 0:
        print(f"world_size={world_size}", flush=True)
        print(f"{'KiB':>10} {'backend':>20}", flush=True)
    for token_count in TOKEN_COUNTS:
        tensor = torch.randn(token_count, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        backend = _dispatched_backend(tensor)
        if rank == 0:
            print(f"{tensor.nbytes / 1024:>10.1f} {backend:>20}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
