# Walkthrough

This section walks the whole path on one model and one machine: **Qwen3-8B on 8x H20**.
It starts with a simulation that needs no GPU, then measures the real system one piece at a
time, fits a cost model from each measurement, and ends with a simulated training job driven
entirely by fitted numbers.

Every command on these pages can be previewed without a GPU: `srl --cfg job ...` prints a
resolved simulation config, and `srl-profile <backend> generate` lists what a sweep will
attempt.

## The stages

| Stage | What it measures | Command | Produces |
|---|---|---|---|
| {doc}`/getting-started/walkthrough/00-first-simulation` | nothing, analytic defaults | `srl` | telemetry JSONL and plots |
| {doc}`/getting-started/walkthrough/01-vllm-rollout` | vLLM scheduler iterations | `srl-profile vllm`, `srl-regress rollout` | `rollout_roofline.json` |
| {doc}`/getting-started/walkthrough/02-kernels` | individual CUDA kernels | `srl-profile kernels` | parquet perf tables |
| {doc}`/getting-started/walkthrough/03-megatron-training` | training steps under Megatron-Bridge | `srl-profile megatron`, `srl-regress training` | `training_hydraulis.json` |
| {doc}`/getting-started/walkthrough/04-fsdp-training` | training steps under FSDP2 | `srl-profile fsdp`, `srl-regress training` | `training_fsdp.json` |
| {doc}`/getting-started/walkthrough/05-workload-trace` | real prompt and response lengths | `srl-workload` | `joint.jsonl` |
| {doc}`/getting-started/walkthrough/06-simulate` | nothing, reads the above | `srl` | telemetry JSONL and plots |

The stages are independent apart from the last one. Stage 6 needs one rollout cost source
(stage 1 or 2) and one training cost source (stage 3 or 4). Stage 5 is optional: a parametric
workload runs without it, but it cannot represent GRPO or DAPO groups.

## Every page answers the same four questions

1. **What it needs.** The command, the config file, and any artifact from an earlier stage.
2. **What it costs.** What dominates the runtime, and what is cheap to preview.
3. **What lands on disk.** Paths, and what one real record looks like.
4. **How to tell it worked.** Which number to read, and what a wrong one looks like.

Wall-clock figures are left out because they depend on the node, the checkpoint storage, and
what else is running. {doc}`/pipeline/profiling/index` explains what dominates each sweep.

## Before you start

```bash
conda activate simrl
export MODEL_PATH=/path/to/Qwen3-8B
```

Every path below is relative to the repository root. The profiling configs read the
checkpoint from `$MODEL_PATH` through OmegaConf interpolation, so no config file needs
editing. {doc}`/getting-started/installation` covers the environment itself.

The pages explain *how*. For *why* a pipeline is built the way it is, follow the links into
{doc}`/pipeline/profiling/index` and {doc}`/pipeline/cost-models/index`.

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
