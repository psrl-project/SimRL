"""Measure the TP all-reduce cost at the message sizes a decode step actually sends.

**Why this exists.** The `psrl` rollout model fits whole scheduler iterations, so the
TP communication cost is folded into its constants and cannot be read off. When TP2
turns out to buy far less than the ideal 2x, the natural suspicion is that the
all-reduce is eating the gain. This script answers that directly rather than by
inference: it times `all_reduce` at exactly the sizes Qwen3.5-35B-A3B's decode step
sends, so the comm cost can be compared against the measured per-step latency.

**The arithmetic being checked.** A tensor-parallel transformer all-reduces the
attention output and the MLP output once each per layer, over a `[batch, hidden]`
tensor. For this checkpoint `hidden_size=2048` in bf16, and only the 10
full-attention layers participate in the attention reduction (the 30 linear/SSM
layers are a different block). So one decode step sends roughly
`2 * n_layers * batch * 2048 * 2` bytes, and this script reports both the per-call
microsecond cost and the per-step total so it can be put beside the measured ITL.

Run with torchrun:
    CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \
        examples/profile/kernel/measure_tp_allreduce.py
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

HIDDEN = 2048
# 10 of 40 layers are full_attention; each contributes an attention all-reduce and
# an MLP all-reduce, and the 30 linear-attention layers still reduce their MLP.
FULL_ATTENTION_LAYERS = 10
TOTAL_LAYERS = 40
BATCHES = (1, 4, 8, 16, 32, 64, 128, 256)
WARMUP = 20
ITERS = 200


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    lead = rank == 0

    if lead:
        print(f"world_size={world}  hidden={HIDDEN}  dtype=bfloat16")
        print()
        print(
            f"{'batch':>6} {'bytes/AR':>9} {'AR_us':>8} "
            f"{'attn_10L_ms':>12} {'all_70AR_ms':>12}"
        )

    for batch in BATCHES:
        tensor = torch.randn(batch, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{rank}")
        for _ in range(WARMUP):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier()

        start = time.perf_counter()
        for _ in range(ITERS):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        per_call_us = (time.perf_counter() - start) / ITERS * 1e6

        # Attention reductions only (the 10 full-attention layers).
        attn_ms = per_call_us * FULL_ATTENTION_LAYERS / 1000.0
        # Upper bound: every layer's MLP reduces too, plus the 10 attention ones.
        all_ms = per_call_us * (TOTAL_LAYERS + FULL_ATTENTION_LAYERS) / 1000.0
        if lead:
            print(
                f"{batch:>6} {batch * HIDDEN * 2:>9} {per_call_us:>8.1f} "
                f"{attn_ms:>12.4f} {all_ms:>12.4f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
