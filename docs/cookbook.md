# Cookbook

A dated, verified command sequence for one real model on one real machine:
**Qwen3-8B on 8x H20**. Every command here was executed on 2026-08-31 and every number was
read off disk rather than recalled.

This page deliberately carries almost no explanation. It is a tape you can replay. For why
each step exists, follow the links — {doc}`profiling/index` for the pipelines,
{doc}`cost-models/index` for the models, {doc}`simulation/index` for the simulator, and
{doc}`profiling/methodology` for the measurement rules.

```bash
conda activate dryrun-rl
export MODEL_PATH=/path/to/Qwen3-8B
export DATASET_PATH=/path/to/dapo-math-17k.parquet
```

All paths below are relative to the repository root.

---

## 0. Simulate with nothing measured

Takes seconds, needs no GPU. Establishes that the install works.

```bash
dryrun policy=areal rollout_cost=roofline training_cost=fixed workload=uniform \
    job.n_versions=5 job.batch_size=32 rollout.n_instances=2
```

Observed: 5 versions, 160 requests consumed, 0 dropped, 1653.7 tokens/s, max true
staleness 3.

See {doc}`getting-started/quickstart`.

---

## 1. vLLM rollout — about 6 minutes

```bash
bash examples/profile/vllm/profile_qwen3_8b.sh all

dryrun-regress rollout --model roofline \
    --input  outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
    --output outputs/cost/qwen3_8b/rollout_roofline.json
```

Yields 49,208 records from 30 sweep points, because one record is one *scheduler iteration*.
Fitted MRE 5.25% (roofline) and 5.16% (psrl), in-sample.

See {doc}`profiling/vllm` and {doc}`cost-models/rollout`.

---

## 2. Kernels — about 1 hour, no regress step

```bash
CFG=examples/profile/kernel/qwen3_8b_h20_tp1.yaml
dryrun-profile kernels run       --config $CFG
dryrun-profile kernels fit-floor --config $CFG
dryrun-profile kernels holdout   --config $CFG --fraction 1.0
dryrun-profile kernels validate  --config $CFG \
    --baseline outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl --min-samples 1
```

Run in that order — `fit-floor` before `holdout`, or the attention families score against a
zero dispatch floor.

`ComposedRollout` reads the resulting parquet tables directly:

```bash
ROLLOUT_MODEL=composed \
KERNEL_ROOT=outputs/profile/qwen3_8b/kernel_tp1/kernel \
GPU=h20 \
bash examples/simulate/simulate.sh
```

See {doc}`profiling/kernels` and {doc}`cost-models/composed-rollout`.

---

## 2b. Kernels at tp=2, with communication

Executed 2026-09-02. A `tp>1` step pays two all-reduces per layer (72 for Qwen3-8B, 13.1% of a
decode step), so the collective table is not optional and the kernel tree is per degree.

```bash
TP=2 bash examples/profile/kernel/profile_qwen3_8b.sh all      # ~1 h, compute + collective
for TP in 4 8; do                                              # seconds each, collective only
    TP=$TP FAMILY=collective bash examples/profile/kernel/profile_qwen3_8b.sh all
done
```

Observed holdout, dispatched table vs the NCCL control: 6.01%/4.96% at `tp=2`, 3.76%/7.56% at
`tp=4`, 7.56%/24.80% at `tp=8`.

Baseline — **serially**, one server at a time:

```bash
CONFIG=examples/profile/vllm/qwen3_8b_tp2_serial.yaml \
    bash examples/profile/vllm/profile_qwen3_8b.sh all
```

Score and attribute:

```bash
CFG=examples/profile/kernel/qwen3_8b_h20_tp2.yaml
BASE=outputs/profile/qwen3_8b/vllm_tp2_serial/profiles/vllm.jsonl
dryrun-profile kernels validate --config $CFG --baseline $BASE \
    --model-path $MODEL_PATH --min-samples 1 --gate 100
python examples/profile/kernel/decompose_tp.py --config $CFG --baseline $BASE \
    --model-path $MODEL_PATH
```

Simulate at that degree:

```bash
ROLLOUT_MODEL=composed ROLLOUT_TP=2 \
KERNEL_ROOT=outputs/profile/qwen3_8b/kernel_tp2/kernel \
GPU=h20 bash examples/simulate/simulate.sh
```

Observed on the serial baseline (2026-09-03): decode **2.96%** at `requests >= 5` (137 buckets),
prefill **3.60%** over the whole grid (29 buckets), mixed 3.64%. Whole-grid decode is 9.51%, the
excess being the pre-existing small-batch over-prediction at `reqs 1-4`. Prefill's six `reqs=1`
points needed re-measuring with `num-prompts: 8` first -- with `num-prompts: 1` they carried a
~78 ms fixed cost and scored 46.98%:

```bash
CONFIG=examples/profile/vllm/qwen3_8b_tp2_reqs1.yaml \
    bash examples/profile/vllm/profile_qwen3_8b.sh all
``` An earlier parallel baseline read 16.39% at decode because one of four
concurrent servers degraded mid-run, which is why the config above is serial.
See {doc}`profiling/tensor-parallel`.

---

## 3. Megatron training — 24 candidates, about 1.1 hours

```bash
bash examples/profile/megatron/profile_qwen3_8b.sh generate
bash examples/profile/megatron/profile_qwen3_8b.sh run       # --resume to continue
bash examples/profile/megatron/profile_qwen3_8b.sh collect

dryrun-regress training --latency-model hydraulis --memory-model hydraulis \
    --input  outputs/profile/qwen3_8b/megatron_bridge/profiles/megatron_bridge.jsonl \
    --output outputs/cost/qwen3_8b/training_hydraulis.json
```

Latency MRE 1.12–2.61% across four cells, 6 samples each, in-sample.

See {doc}`profiling/megatron` and {doc}`cost-models/training-megatron`.

---

## 4. FSDP2 training — 108 candidates, about 6.9 hours

```bash
CFG=examples/profile/fsdp/qwen3_8b.yaml
dryrun-profile fsdp generate --config $CFG
dryrun-profile fsdp run      --config $CFG            # --resume, or --candidate-id ID
dryrun-profile fsdp collect  --config $CFG

dryrun-regress training --latency-model fsdp --memory-model fsdp \
    --input  outputs/profile/qwen3_8b/fsdp/profiles/fsdp.jsonl \
    --output outputs/cost/qwen3_8b/training_fsdp.json
```

Outcome: 73 `success`, 34 `oom`, 1 `failed`. The OOM records are wanted data — they bound
the memory model from below.

Held-out latency MRE 5.62–12.02%, against 8.94–10.65% in-sample. This artifact is the one
model on the project with a genuine validation split.

Smoke-test first on `examples/profile/fsdp/qwen3_8b_smoke.yaml` — two layers, minutes
rather than hours.

See {doc}`profiling/fsdp` and {doc}`cost-models/training-fsdp`.

---

## 5. Collect a workload trace — about 49 minutes on 16 GPUs

```bash
HOSTS_FILE=examples/workload/hosts.txt \
DATASET=$DATASET_PATH MODEL=$MODEL_PATH \
ENV_SCRIPT=/path/to/env-activation.sh \
bash examples/workload/collect_joint_trace.sh
```

`DRY_RUN=1` prints the per-worker commands without launching servers.

Produced 1024 groups of 8 responses: prompt mean 149.7 tokens, output mean 7804, and 40.8%
of responses truncated at the 10240 cap.

See {doc}`workload/trace-collection`.

---

## 6. Simulate against the fitted models

```bash
R=outputs

TRAINING_BACKEND=hydraulis \
TRAINING_ARTIFACT=$R/cost/qwen3_8b/training_hydraulis.json \
bash examples/simulate/simulate.sh

TRAINING_BACKEND=fsdp BATCHING_MODE=pad_to_max \
TRAINING_ARTIFACT=$R/cost/qwen3_8b/training_fsdp.json \
EXTRAPOLATE=[num_microbatches] \
OUTPUT_DIR=$R/simulate/fsdp_pad \
bash examples/simulate/simulate.sh
```

`simulate.sh` cd's to the repository root, so relative paths resolve there no matter where
you invoke it from.

`EXTRAPOLATE=[num_microbatches]` is required whenever the simulated batch needs more
microbatches than the sweep measured. Leaving it unset makes the model refuse rather than
guess.

See {doc}`simulation/index` and {doc}`simulation/domain-gates`.
