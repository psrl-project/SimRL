# Stage 6: simulate against fitted models

Everything measured so far comes together here. The simulator drives a rollout and training
loop under a staleness policy, asking the fitted cost models for latency and peak memory as
it goes.

This is the stage where the earlier work pays off, and also where the domain gates fire, so
expect the first attempt to be refused.

## What it needs

At least one rollout cost source and one training cost source:

| | From | Alternative |
|---|---|---|
| rollout | `rollout_roofline.json` ({doc}`01-vllm-rollout`) | kernel tables ({doc}`02-kernels`) |
| training | `training_hydraulis.json` ({doc}`03-megatron-training`) | `training_fsdp.json` ({doc}`04-fsdp-training`) |
| workload | `joint.jsonl` ({doc}`05-workload-trace`) | a parametric distribution |

## The command

```bash
TRAINING_BACKEND=hydraulis \
TRAINING_ARTIFACT=outputs/cost/qwen3_8b/training_hydraulis.json \
TRAINING_DP=8 \
N_VERSIONS=3 BATCH_SIZE=32 \
EXTRAPOLATE='[tokens,activation_bytes]' \
bash examples/simulate/simulate.sh
```

`simulate.sh` changes to the repository root itself, so relative paths resolve there no
matter where you invoke it from.

It prints the fitted domain before it starts:

```text
fitted per-microbatch token domain: [512, 8192]
  a step exceeding 8192 tokens in its longest microbatch raises unless
  EXTRAPOLATE=[tokens] is set; the fit is linear in tokens while attention is
  quadratic, so a stretched tail is understated
```

That is deliberate, not noise. It also verifies the requested cell exists before the
simulator does, because the model itself raises a bare `KeyError` naming only the cell it
wanted, which leaves you guessing what was fitted.

## Expect it to be refused first

Run the same thing without `EXTRAPOLATE` and it stops:

```text
TrainingOutOfDomainError: Feature 'tokens'=9470.0 is outside the profiled range [512.0, 8192.0].
```

This is the single most common first-run failure, and it is the design working. The DAPO
trace's median sequence is 8828 tokens, so a real batch immediately exceeds a domain fitted
to 8192. The model refuses rather than returning a number that looks plausible and describes
nothing.

Permitting extrapolation is per feature, by name, and is recorded. There is no global
override.

**`TRAINING_DP` matters more than it looks.** It defaults to 1, which puts the whole batch
on a single rank. At `BATCH_SIZE=64` with `micro_batch_size=1`, that is 64 microbatches
against a profiled range of [1, 4]:

```text
TrainingOutOfDomainError: Feature 'pipeline_slots'=64.0 is outside the profiled range [1.0, 4.0].
```

Extrapolating 16x on that axis is not something to wave through. Setting `TRAINING_DP=8`
splits the batch across eight ranks, giving four microbatches each, which is inside the
domain. The fix is to describe the real topology, not to widen the gate.

## What it prints

For the command above:

```text
Simulation complete: 3 versions in 331.68s
  Consumed: 96 requests
  Dropped: 0 requests
  Throughput: 2373.1 tokens/s
  Max staleness (true): 2
  Staleness distribution:
    0: 32 requests (33.33%)
    1: 32 requests (33.33%)
    2: 32 requests (33.33%)
```

331.68 s is simulated time. The run itself takes well under a second.

## What lands on disk

The same four telemetry streams and five plots as {doc}`00-first-simulation`, but the
records now carry predicted rather than placeholder values:

```javascript
{
  "t": 196.2597,
  "start": 140.3115,
  "end": 196.2597,
  "dt": 55.9482,                       // predicted by the fitted latency model
  "version": 1,
  "batch_size": 32,
  "n_reqs": 32,
  "total_tokens": 275630,
  "train_time": 55.9482,
  "peak_memory_bytes": 74836796652,    // 69.7 GiB, predicted by the memory model
  "training_tp": 1, "training_pp": 1, "training_dp": 8, "training_cp": 1,
  "micro_batch_size": 1,
  "schedule": "1f1b",
  "activation_recompute": "full",
  "sync_time": 0.0,
  "recompute_time": 0.0,
  "n_inflight": 32,
  "n_ready": 64,
  "n_dropped_step": 0,
  "extrapolation_notes": [
    "extrapolated 'tokens'=9235.0 outside profiled range [512.0, 8192.0]",
    "extrapolated 'activation_bytes'=5764818809.856 outside profiled range [319608795.955, 5113740735.283]"
  ]
}
```

Compare that with the stage 0 record, where `dt` was exactly 1.0 and
`peak_memory_bytes` was 0.

**`extrapolation_notes` is the field to read.** Every permitted extrapolation is written
into the telemetry verbatim, with the feature, the value, and the range it left. A run that
carries notes is a run whose numbers rest partly outside measured ground, and the record
says so. If the field is absent, every query was interpolation.

The 69.7 GiB predicted peak against the H20's 97871 MiB is worth sanity-checking on any
run. Set `training_gpu_memory_bytes` and the simulator raises immediately when a predicted
peak exceeds it, rather than reporting throughput for a configuration that would OOM in
practice.

## The FSDP path

```bash
TRAINING_BACKEND=fsdp BATCHING_MODE=pad_to_max \
TRAINING_ARTIFACT=outputs/cost/qwen3_8b/training_fsdp.json \
OUTPUT_DIR=outputs/simulate/fsdp_pad \
bash examples/simulate/simulate.sh
```

Two differences from the Megatron path. `SHARD_SIZE` selects the mesh, and an FSDP cell is
a measurement of one mesh, so `dp` must equal `shard * replicate` or the model refuses.
Left unset, both are read from the artifact.

And `BATCHING_MODE` exists only here. Profiling measured uniform shapes, so how real
trajectory lengths become microbatches is a simulate-time choice: `pad_to_max` charges for
pad tokens, `token_budget` reproduces verl's dynamic batching, and `packed_no_pad` charges
real tokens only. All three can legitimately give identical results, and must at
`micro_batch_size=1`, so agreement is not evidence that the wiring is wrong.

## Rollout from kernel tables instead

```bash
ROLLOUT_MODEL=composed \
KERNEL_ROOT=outputs/profile/qwen3_8b/kernel_tp1/kernel \
MODEL_PATH=$MODEL_PATH GPU=h20 \
bash examples/simulate/simulate.sh
```

This substitutes the summed-kernel model for the fitted regression. It is more accurate on
shapes the vLLM sweep never ran, and it over-predicts small-batch decode by design, so
quote its accuracy at `requests >= 5`. For `ROLLOUT_TP=2` or above, `KERNEL_ROOT` must point
at a tree swept at the same degree.

## How to tell it worked

```bash
python -c "
import json
rows = [json.loads(l) for l in open('outputs/simulate/.../logs/train.jsonl')]
for r in rows:
    gib = r['peak_memory_bytes'] / 2**30
    print(f\"v{r['version']}  {r['train_time']:7.2f}s  {gib:6.1f} GiB  {len(r.get('extrapolation_notes', []))} notes\")
"
```

Three checks. `train_time` should vary across versions, since a constant value means a
placeholder model is still wired in. `peak_memory_bytes` should be non-zero and below device
capacity. And the note count tells you how much of the run rests on extrapolation.

Then read the plots. `throughput.png` and `staleness_distribution.png` are the two that
answer the question this whole path was built for.

## Sweeping configurations

This is the point of having done the measurement. Hydra's multirun crosses any set of
overrides:

```bash
dryrun --multirun \
    rollout.n_instances=2,4,8 job.max_staleness=1,2,4 \
    policy=areal training_cost=hydraulis \
    training_cost.artifact_path=outputs/cost/qwen3_8b/training_hydraulis.json \
    training_cost.parallelism.dp=8 \
    rollout_cost=roofline \
    rollout_cost.artifact_path=outputs/cost/qwen3_8b/rollout_roofline.json \
    workload=bimodal job.n_versions=10 job.batch_size=32
```

Nine runs, each in its own subdirectory. Every one costs seconds instead of GPU hours, and
every one either produces a number backed by measurement or refuses to.

## Where to go next

{doc}`../../simulation/policies` compares the five staleness policies.
{doc}`../../simulation/domain-gates` covers extrapolation in full.
{doc}`../../reference/validation` collects every accuracy figure on the project with its
sample count, which is what tells you how far to trust any of this.
