# Stage 6: simulate against fitted models

Everything measured so far comes together here. The simulator drives a rollout and training
loop under a staleness policy, asking the fitted cost models for latency and peak memory as it
goes.

This is the stage where the earlier work pays off, and also where the domain gates fire, so
expect the first attempt to be refused.

## What it needs

One rollout cost source and one training cost source, plus a workload:

| | From | Alternative |
|---|---|---|
| rollout | `rollout_roofline.json` ({doc}`/getting-started/walkthrough/01-vllm-rollout`) | kernel tables ({doc}`/getting-started/walkthrough/02-kernels`) |
| training | `training_hydraulis.json` ({doc}`/getting-started/walkthrough/03-megatron-training`) | `training_fsdp.json` ({doc}`/getting-started/walkthrough/04-fsdp-training`) |
| workload | `joint.jsonl` ({doc}`/getting-started/walkthrough/05-workload-trace`) | a parametric distribution |

## The command

```bash
srl policy=psrl policy.staleness=2 \
    rollout_cost=roofline \
    rollout_cost.artifact_path=outputs/cost/qwen3_8b/rollout_roofline.json \
    training_cost=hydraulis \
    training_cost.artifact_path=outputs/cost/qwen3_8b/training_hydraulis.json \
    training_cost.parallelism.tp=1 training_cost.parallelism.dp=8 \
    training_cost.activation_recompute=full \
    workload=dapo_trace \
    workload.trace.path=outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl \
    job.n_versions=3 job.batch_size=4 \
    job.max_prompt_length=1024 job.max_output_length=10240 \
    rollout.n_instances=4 rollout.token_budget=65536
```

Every override picks a config group (`policy`, `rollout_cost`, `training_cost`, `workload`) or
sets a key under one. `srl --cfg job <same overrides>` prints the resolved config without
running anything, which is the cheapest way to check a key name. A key that is not in the
group's YAML fails composition with `Could not override '<key>'` and the hint to use
`+<key>`.

Four things in that command are easy to get wrong:

`job.batch_size`
: Counts prompt **groups**. The trace has `group_size` 8, so 4 groups are 32 responses per
  step. `job.group_size` defaults to the workload's arity and is checked against it.

`training_cost.parallelism.dp`
: Describes the real training mesh, and it divides the batch: `batch_size * group_size`
  responses must be divisible by `dp * micro_batch_size`, otherwise the model raises
  `global_batch_size must be divisible by dp * micro_batch_size`.

`training_cost.activation_recompute` and the other selector keys
: Must match a profiled cell. The Megatron cell lookup is exact on TP (and CP, EP) and on the
  selector. PP is derived.

`policy.staleness`
: The `psrl` preset's version bound. Other presets have their own knobs (`areal` has
  `max_head_offpolicyness`, `verl` has `staleness_threshold`, and so on). A key from one
  preset is rejected by another, so a sweep over policies cannot share a policy knob.

## Expect it to be refused first

Run the command above and it stops:

```text
TrainingOutOfDomainError: Feature 'tokens'=8652.0 is outside the profiled range [512.0, 8192.0].
```

This is the most common first-run failure, and it is the design working. The trace's
sequences are longer than the 8192 tokens the Megatron grid covered, and the model refuses
instead of returning a plausible number for a region it never measured.

Permitting extrapolation is per feature, by name, and is recorded. There is no global
override. Each refusal names one feature, so permitting `tokens` alone is followed by the next:

```text
TrainingOutOfDomainError: Feature 'sum_squared_tokens'=287372169.0 is outside the profiled range [262144.0, 268435456.0].
```

Name the features the data actually exceeds:

```bash
    'training_cost.extrapolate=[tokens,sum_squared_tokens,activation_bytes]'
```

Valid names for the Megatron artifact are `tokens`, `sum_squared_tokens`, `pipeline_slots`,
`micro_batch_size`, `activation_bytes`, and `profiled_dp`. A name that is not one of them raises
`Unknown extrapolation feature(s) [...]` listing the valid ones. An extrapolation is a decision
about how far to trust a linear stretch: the token terms are fitted over a profiled range, and
attention grows with the square of the sequence length, so a long tail is a bounded
approximation. {doc}`/pipeline/simulation/domain-gates` says which features are safe to extend.

Scaling the batch tests the same discipline. With `job.batch_size=32` (256 responses over
`dp=8`, 32 microbatches per rank at `micro_batch_size=1`), the fitted `pipeline_slots` range
of `[1, 7]` is exceeded:

```text
TrainingOutOfDomainError: Feature 'pipeline_slots'=32.0 is outside the profiled range [1.0, 7.0].
```

Extrapolating that axis many-fold is not something to wave through. Describe how the real
trainer batches instead: `training_cost.max_tokens_per_microbatch` packs sequences to a token
budget the way verl does with dynamic batching, which cuts the microbatch count. Pass the
value already multiplied by the sequence-parallel size (verl computes
`ppo_max_token_len_per_gpu * sp_size`, and Megatron sequence parallelism makes `sp_size` equal
TP).

## What it prints

With the three features permitted, the command above prints:

```text
Simulation complete: 3 versions in 223.57s
  Consumed: 96 requests
  Dropped: 0 requests
  Throughput: 3152.4 tokens/s
  Max staleness (true): 2
  Staleness distribution:
    0: 32 requests (33.33%)
    1: 32 requests (33.33%)
    2: 32 requests (33.33%)
  SCG(S) = tokens from trajectories with staleness <= S, per second:
    S=0:     826.8 tok/s ( 26.2% of throughput)
    S=1:    1873.8 tok/s ( 59.4% of throughput)
    S=2:    3152.4 tok/s (100.0% of throughput)  <- policy.horizon
  Staleness report: mean=1.14 p50=1 p95=2 p99=2 max=2 horizon=2 strict=True
```

223.57 s is **simulated** time. The run itself takes seconds. `Consumed: 96` is 3 versions of
32 responses. `Staleness report ... strict=True` says no consumed response exceeded
`horizon`.

## What lands on disk

The same streams and plots as {doc}`/getting-started/walkthrough/00-first-simulation`, but the records now carry predicted
values:

```javascript
{
  "t": 139.7164,
  "start": 114.4909,
  "end": 139.7164,
  "dt": 25.2255,                       // predicted by the fitted latency model
  "version": 1,
  "n_groups": 4,
  "group_size": 8,
  "n_reqs": 32,
  "total_tokens": 187849,
  "train_time": 25.2255,
  "peak_memory_bytes": 70697372972,    // predicted by the fitted memory model
  "training_tp": 1, "training_pp": 1, "training_dp": 8, "training_cp": 1,
  "micro_batch_size": 1,
  "schedule": "1f1b",
  "activation_recompute": "full",
  "sync_time": 0.0,
  "recompute_time": 0.0,
  "n_inflight": 32,
  "n_ready": 64,
  "n_pending": 0,
  "n_dropped_step": 0,
  "extrapolation_notes": [
    "extrapolated 'tokens'=8652.0 outside profiled range [512.0, 8192.0]",
    "extrapolated 'activation_bytes'=5400889262.8992 outside profiled range [319608795.95519996, 5113740735.283199]"
  ]
}
```

Compare that with the stage 0 record, where `dt` was exactly the configured `latency_s` and
`peak_memory_bytes` was 0.

**`extrapolation_notes` is the field to read.** Every permitted extrapolation is written into
the telemetry verbatim, with the feature, the value, and the range it left. A run that carries
notes is a run whose numbers rest partly outside measured ground. If the field is absent,
every query was interpolation.

Set `training_cost.gpu_memory_mib` (97871 for one H20) and the simulator raises as soon as a
predicted peak exceeds it, instead of reporting throughput for a configuration that would run
out of memory:

```text
RuntimeError: Predicted training peak memory 70697372972 exceeds configured capacity 1048576000.
```

(That message is from setting `gpu_memory_mib=1000` to force it.)

## Two more cost knobs

**Log-prob recomputation.** The PPO/GRPO log-prob pass is charged by `recompute_cost`, which
defaults to `fixed` with `time_per_token: 0.0`, i.e. free. `recompute_cost.name=training_forward`
prices it as a FLOP-derived share of the fitted training step, reading the model's hidden size,
vocabulary, and parameter count from the training artifact, so it needs nothing of its own:

```bash
    recompute_cost.name=training_forward
```

**Weight sync.** `sync_cost.sync_time` is added to each training step after the update and
`sync_cost.weight_load_time` is the rollout-side time to load the new weights. Both are seconds
and default to 0.0.

**KV capacity.** `rollout.kv_blocks` is a literal placeholder (`100000`) unless sized. Naming a
checkpoint replaces it:

```bash
    rollout.kv_cache_sizing.model_path=$MODEL_PATH \
    rollout.kv_cache_sizing.gpu_name=H20 rollout.kv_cache_sizing.tp_size=1
```

With `probe: true` (the default) the first run starts a real vLLM engine, reads back the block
count, and caches it in `~/.cache/srl/kv_capacity.json` (override with `SIMRL_KV_CACHE`).
`srl-profile kv-probe --model ... --tp ...` populates that cache ahead of time on a GPU box.
`probe: false` computes the capacity from `config.json` on CPU, which is only reliable for dense
models. {doc}`/pipeline/simulation/prefix-cache` covers capacity and the block pool.

## The FSDP path

```bash
srl rollout_cost=roofline \
    rollout_cost.artifact_path=outputs/cost/qwen3_8b/rollout_roofline.json \
    training_cost=fsdp \
    training_cost.artifact_path=outputs/cost/qwen3_8b/training_fsdp.json \
    training_cost.shard_size=8 training_cost.replicate_size=1 \
    training_cost.batching_mode=pad_to_max \
    workload=dapo_trace \
    workload.trace.path=outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl \
    job.n_versions=3 job.batch_size=4 \
    job.max_prompt_length=1024 job.max_output_length=10240 \
    rollout.n_instances=4 rollout.token_budget=65536
```

Two differences from the Megatron path. `shard_size` and `replicate_size` select the mesh: an
FSDP cell is a measurement of one mesh, so `parallelism.dp` (default 8 in `fsdp.yaml`) must
equal `shard * replicate` or the model refuses. And `batching_mode` exists only here
({doc}`/getting-started/walkthrough/04-fsdp-training` lists the three modes). The FSDP artifact fitted in stage 4 covers
`tokens` up to 16384, and at these settings the run needs no `extrapolate`.

## Rollout from kernel tables instead

```bash
srl rollout_cost=composed \
    rollout_cost.kernel_table_root=outputs/profile/qwen3_8b/kernel_tp1/kernel \
    rollout_cost.gpu=h20 rollout_cost.model_path=$MODEL_PATH ...
```

This substitutes the summed-kernel model for the fitted regression. It prices shapes the vLLM
sweep never ran. For `rollout_cost.parallelism.tp=2` or above, `kernel_table_root` must point
at a tree swept at the same degree ({doc}`/getting-started/walkthrough/02-kernels`).

## Choosing the rollout engine

`srl` runs the compiled C++ rollout engine by default (`SIMRL_BACKEND=cpp`).
`SIMRL_BACKEND=native` forces the Python engine, and `SIMRL_BACKEND=auto` falls back to it only
when the extension cannot be imported or the configuration needs Python-only request state.
{doc}`/pipeline/simulation/backends` covers how the two relate.

## How to tell it worked

```bash
python -c "
import json
rows = [json.loads(l) for l in open('outputs/logs/train.jsonl')]
for r in rows:
    gib = r['peak_memory_bytes'] / 2**30
    print(f\"v{r['version']}  {r['train_time']:7.2f}s  {gib:6.1f} GiB  {len(r.get('extrapolation_notes', []))} notes\")
"
```

Three checks. `train_time` should vary across versions, since a constant value means a
placeholder model is still wired in. `peak_memory_bytes` should be non-zero and below device
capacity. And the note count tells you how much of the run rests on extrapolation. It depends on how
long the batch's longest sequences are, so it can differ between versions of one run.

Then read the plots. `throughput.png` and `staleness_distribution.png` are the two that answer
the question this whole path was built for, and `scg_curve.png` shows throughput as a function
of the staleness you are willing to accept.

## Sweeping configurations

This is the point of having done the measurement. Hydra's multirun crosses any set of
overrides:

```bash
srl --multirun \
    rollout.n_instances=2,4,8 \
    policy=areal policy.max_head_offpolicyness=0,1,2 \
    training_cost=hydraulis \
    training_cost.artifact_path=outputs/cost/qwen3_8b/training_hydraulis.json \
    training_cost.parallelism.dp=8 training_cost.activation_recompute=full \
    rollout_cost=roofline \
    rollout_cost.artifact_path=outputs/cost/qwen3_8b/rollout_roofline.json \
    workload=bimodal job.n_versions=10 job.batch_size=16 \
    log_telemetry=false
```

Nine jobs, each in its own `multirun/<date>/<time>/<n>/` directory. `log_telemetry=false` skips
the JSONL and plots, so a sweep that only reads the printed summary leaves no files behind.
Drop it and set `'output_dir=${hydra:runtime.output_dir}'` to keep each job's telemetry
separate. Every job costs seconds instead of GPU hours, and every one either produces a number
backed by measurement or refuses to.

## Where to go next

{doc}`/pipeline/simulation/policies` compares the staleness policies.
{doc}`/pipeline/simulation/domain-gates` covers extrapolation in full.
{doc}`/reference/validation` collects the accuracy figures with their sample counts, which
is what tells you how far to trust any of this.
