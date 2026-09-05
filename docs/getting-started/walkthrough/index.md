# Walkthrough

This section walks the whole path on one model and one machine: **Qwen3-8B on 8x H20**.
It starts with a simulation that needs no GPU at all, then measures the real system one
piece at a time, fits a cost model from each measurement, and ends with a simulated
training job driven entirely by fitted numbers.

Every command here was executed, and every number quoted was read off disk rather than
recalled. Where a figure appears, the file it came from is named.

## The stages

| Stage | What it measures | Wall clock | Produces |
|---|---|---|---|
| {doc}`00-first-simulation` | nothing, analytic defaults | seconds | telemetry JSONL and plots |
| {doc}`01-vllm-rollout` | vLLM scheduler iterations | ~6 min | `rollout_roofline.json` |
| {doc}`02-kernels` | individual CUDA kernels | ~1 h | perf tables (parquet) |
| {doc}`03-megatron-training` | training steps under Megatron-Bridge | 2.2 h | `training_hydraulis.json` |
| {doc}`04-fsdp-training` | training steps under FSDP2 | 6.9 h | `training_fsdp.json` |
| {doc}`05-workload-trace` | real prompt and response lengths | ~49 min on 16 GPUs | `joint.jsonl` |
| {doc}`06-simulate` | nothing, reads the above | seconds | telemetry JSONL and plots |

The stages are independent apart from the last one. Stage 6 needs at least one rollout
cost source (stage 1 or 2) and one training cost source (stage 3 or 4). Stage 5 is
optional: a parametric workload works without it, but group-relative advantage does not.

## Every page answers the same four questions

1. **What it needs.** The command, the config file, and any artifact from an earlier stage.
2. **How long it takes.** Measured, with the date and the machine.
3. **What lands on disk.** Paths, and what one real record looks like.
4. **How to tell it worked.** Which number to read, and what a wrong one looks like.

## Before you start

```bash
conda activate dryrun-rl
export MODEL_PATH=/path/to/Qwen3-8B
export DATASET_PATH=/path/to/dapo-math-17k.parquet
```

Every path below is relative to the repository root, and every profiling config reads
the checkpoint from `$MODEL_PATH` through OmegaConf interpolation, so no config file
needs editing. {doc}`../installation` covers the environment itself.

The pages explain *how*. For *why* a pipeline is built the way it is, follow the links
into {doc}`../../profiling/index` and {doc}`../../cost-models/index`.

```{toctree}
:maxdepth: 1
:hidden:

00-first-simulation
01-vllm-rollout
02-kernels
03-megatron-training
04-fsdp-training
05-workload-trace
06-simulate
```
