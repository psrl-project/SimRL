# Stage 0: simulate with nothing measured

The simulator does not need profiling data to run. It ships analytic cost models with
default coefficients, so the first thing worth doing is a run that touches no GPU. It
takes a few seconds and confirms the install works end to end.

## What it needs

Nothing beyond the base install:

```bash
conda activate dryrun-rl
pip install -e ".[dev]"
```

## The command

```bash
dryrun policy=areal rollout_cost=roofline training_cost=fixed workload=uniform \
    job.n_versions=5 job.batch_size=32 rollout.n_instances=2
```

Run it from the repository root.

Each override selects one config group:

`policy=areal`
: The staleness policy. `areal` reproduces AReaL's cumulative token-bucket admission
  control. It needs no external package and has a short admission formula you can read,
  which makes it the right pick for a first run. The default `policy=psrl` delegates
  buffer management to the separate `psrl` package and raises `ImportError` on a machine
  that does not have it. {doc}`../../simulation/policies` compares all five.

`rollout_cost=roofline`
: The rollout cost model, evaluating `max(F, W + G*n_tokens + A_p*sum_l2/b +
  A_d*ctxsum)` with default coefficients. {doc}`01-vllm-rollout` replaces those defaults
  with fitted ones.

`training_cost=fixed`
: A flat 1.0 second per training step and zero memory. The right placeholder until
  stage 3 or 4 has produced a fitted artifact.

`workload=uniform`
: Prompt and response lengths drawn uniformly from the defaults in
  `dryrun/config/defaults/workload/uniform.yaml`. No checkpoint, no dataset.

`job.n_versions=5`
: Five training versions, meaning five gradient updates. Each one waits for a full batch.

`job.batch_size=32`
: 32 **responses** per batch, not 32 prompts. The distinction matters as soon as a
  grouped trace is involved, where one prompt yields eight responses.

`rollout.n_instances=2`
: Two rollout instances. Requests route to whichever has the fewest active requests.

## What it prints

```text
Simulation complete: 5 versions in 11.42s
  Consumed: 160 requests
  Dropped: 0 requests
  Throughput: 1653.7 tokens/s
  Max staleness (true): 3
  Staleness distribution:
    0: 49 requests (30.63%)
    1: 47 requests (29.38%)
    2: 47 requests (29.38%)
    3: 17 requests (10.62%)
```

160 consumed is 32 per version times 5 versions. Nothing was dropped, so the `areal`
bucket never rejected a request: at this batch size rollout keeps pace with training.

The staleness distribution labels every consumed response with the gap between the weight
version it was dispatched under and the version that consumed it. 49 responses were
dispatched and consumed inside the same version. The oldest were three versions behind.
{doc}`../concepts` separates true from reported staleness, which differ when a request's
tokens span several weight versions.

The 11.42 s is **simulated** time, not wall clock. The run itself finishes in well under
a second.

## What lands on disk

```text
outputs/
├── logs/
│   ├── train.jsonl            5 records, one per training step
│   ├── request.jsonl          608 records, one per request lifecycle event
│   ├── rollout_step.jsonl     189 records, one per rollout engine step
│   └── instance/
│       ├── 0.jsonl            108 records, per-step snapshot of instance 0
│       └── 1.jsonl            per-step snapshot of instance 1
├── plots/
│   ├── instance_load.png
│   ├── request_timeline.png
│   ├── rollout_step.png
│   ├── staleness_distribution.png
│   └── throughput.png
├── run.log                    the simulator's own log
└── 2026-09-03/14-39-24/
    ├── main.log
    └── .hydra/
        ├── config.yaml        the fully resolved config
        ├── hydra.yaml
        └── overrides.yaml     the overrides you passed
```

Note where Hydra puts its run directory: under `outputs/<date>/<time>/`, separate from
the telemetry, which goes to `outputs/logs/` because `output_dir` defaults to `outputs`.

One `train.jsonl` record, from the run above:

```javascript
{
  "t": 4.2239819,             // simulated time at the end of this step
  "start": 3.2239819,
  "end": 4.2239819,
  "dt": 1.0,                  // training_cost=fixed, so exactly 1.0 s
  "version": 1,
  "batch_size": 32,
  "n_reqs": 32,
  "total_tokens": 18749,      // summed across the batch
  "train_time": 1.0,
  "peak_memory_bytes": 0,     // FixedTrainingMemory reports zero
  "training_tp": 1, "training_pp": 1, "training_dp": 1, "training_cp": 1,
  "micro_batch_size": 1,
  "schedule": "1f1b",
  "activation_recompute": "none",
  "sync_time": 0.0,
  "recompute_time": 0.0,
  "n_inflight": 32,           // still generating
  "n_ready": 64,              // finished, waiting for a batch
  "n_dropped_step": 0
}
```

`peak_memory_bytes: 0` and `dt: 1.0` are the placeholder cost models showing through.
After stage 3 or 4 both carry predicted values, and the simulator will refuse a
configuration whose predicted peak exceeds `training_gpu_memory_bytes`.

The five PNGs render from the JSONL streams. Regenerate them without re-simulating by
pointing `dryrun-plot` at the run directory:

```bash
dryrun-plot outputs
```

It takes the run directory as a positional argument and looks for `logs/` and `plots/`
beneath it. `--logs-subdir` and `--plots-subdir` override those names.

## How to tell it worked

A progress bar, a summary block, and files under `outputs/`. If the throughput reads
1653.7 tokens/s you have reproduced the run above exactly, since the workload is seeded.

## Varying it

Swap the workload shape:

```bash
dryrun policy=areal rollout_cost=roofline training_cost=fixed workload=bimodal \
    job.n_versions=5 job.batch_size=32 rollout.n_instances=2 \
    workload.output.short_len=1000 workload.output.long_len=8192 \
    workload.output.long_frac=0.4
```

`bimodal` is the right parametric shape for a truncated output distribution, which is
what DAPO produces. {doc}`../../workload/index` explains why, and stage 5 measures the
real thing.

Sweep several configurations in one invocation, using Hydra's multirun:

```bash
dryrun --multirun policy=areal,roll rollout.n_instances=1,2,4 \
    rollout_cost=roofline training_cost=fixed workload=uniform \
    job.n_versions=5 job.batch_size=32
```

Six runs execute in sequence, each writing to its own subdirectory.

Check that an override landed where you meant it to, without running anything:

```bash
dryrun --cfg job policy=areal rollout_cost=roofline training_cost=fixed workload=uniform
```

`--cfg job` prints the resolved config and exits. This is the cheapest way to verify a
key name before committing to a sweep.

## Next

The `roofline` and `fixed` coefficients are not calibrated to any particular hardware.
{doc}`01-vllm-rollout` measures the real inference engine and fits them.
