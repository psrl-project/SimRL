# TP=2/4/8 all-reduce dispatch, measured on H20 (sm_90), vLLM 0.26.1.dev0+g568afb3a1

Produced by `examples/profile/kernel/dispatch_probe.py`, which reads the live
`CudaCommunicator`'s own predicates rather than re-deriving vLLM's size thresholds.

Message size for Qwen3-8B is `tokens * 4096 * 2` bytes. A decode step at batch 1-64 sits in
**8 KiB - 512 KiB**, and a prefill step reaches a few MiB.

| KiB | world_size=2 | world_size=4 | world_size=8 |
|---:|---|---|---|
| 8 | custom_all_reduce | custom_all_reduce | custom_all_reduce |
| 16 | custom_all_reduce | custom_all_reduce | custom_all_reduce |
| 32 | custom_all_reduce | custom_all_reduce | custom_all_reduce |
| 64 | custom_all_reduce | custom_all_reduce | custom_all_reduce |
| 128 | custom_all_reduce | custom_all_reduce | custom_all_reduce |
| 256 | custom_all_reduce | custom_all_reduce | torch_symm_mem |
| 512 | custom_all_reduce | custom_all_reduce | torch_symm_mem |
| 1024 | custom_all_reduce | custom_all_reduce | torch_symm_mem |
| 2048 | custom_all_reduce | custom_all_reduce | torch_symm_mem |
| 4096 | custom_all_reduce | custom_all_reduce | torch_symm_mem |
| 8192 | torch_symm_mem | torch_symm_mem | torch_symm_mem |
| 16384 | torch_symm_mem | torch_symm_mem | torch_symm_mem |
| 32768 | torch_symm_mem | torch_symm_mem | torch_symm_mem |
| 65536 | torch_symm_mem | pynccl | torch_symm_mem |
| 131072 | pynccl | pynccl | pynccl |

## What this settles

**`torch.distributed.all_reduce` never appears.** The old collective table timed exactly that
call, so it priced a kernel no server invokes at any world size or any message size measured
here. This is the defect the work exists to fix.

**The whole decode range is `custom_all_reduce`** at every world size. This is vLLM's custom
one-shot/two-shot P2P kernel, reached because on sm_90 the custom path's ceiling is 8 MiB at
two ranks (`CUSTOM_ALL_REDUCE_MAX_SIZES["9.0"]`), far above the 8-512 KiB a decode step needs.

**NCCL symmetric memory never fires**, despite `VLLM_ALLREDUCE_USE_SYMM_MEM` defaulting on.
`is_symmetric_memory_enabled()` also requires `VLLM_USE_NCCL_SYMM_MEM`, which defaults to 0.
The `symm_mem` label in `collective.py` is therefore unreachable in this configuration. The
`torch_symm_mem` rows are the *torch* symmetric-memory communicator, a different backend.

## Corrections to the plan's assumptions

The plan predicted a clean single-backend table at tp=2 and a size-split only at tp=8. Both
were wrong in detail:

- **tp=2 is already mixed**, splitting at 8 MiB into `torch_symm_mem` and then `pynccl`. So the
  `kernel_source` column reads `mixed` for every world size, and the per-row dispatch is only
  recoverable from this file and the profiler's log.
- **tp=8 splits at 256 KiB**, not at the 256 KiB custom-AR ceiling the plan cited from
  `CUSTOM_ALL_REDUCE_MAX_SIZES`: the observed boundary agrees with that ceiling, but via
  `should_custom_ar` rather than the table lookup the plan reasoned from.

Neither changes the conclusion for the composed model: a Qwen3-8B rollout step never leaves the
`custom_all_reduce` region at tp=2, which is precisely the region the old NCCL table got wrong.

## Latency, vLLM dispatch vs NCCL control (world_size=2)

| KiB | vllm us | nccl us | nccl/vllm |
|---:|---:|---:|---:|
| 8 | 18.42 | 19.60 | 1.06 |
| 128 | 17.82 | 18.46 | 1.04 |
| 512 | 17.90 | 19.01 | 1.06 |
| 2048 | 18.66 | 28.60 | 1.53 |
| 8192 | 45.83 | 85.92 | 1.87 |
| 131072 | 476.11 | 473.70 | 0.99 |

In the decode range the two agree within ~6%, both pinned to a ~18 us launch floor. The old
table was therefore not badly wrong *in latency* for decode: it was wrong in *identity*, and
the gap opens to 1.5-1.9x in the low-MiB prefill range where the custom kernel wins.
