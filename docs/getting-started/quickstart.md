# Quickstart

This page runs a complete simulation in under a minute. It needs no GPU, no profiling
data, and no external packages beyond the base install.

---

## Prerequisites

Install DryRun-RL following {doc}`installation`. The single required step for this page
is:

```bash
conda activate dryrun-rl
pip install -e ".[dev]"
```

---

## The command

```bash
dryrun policy=areal rollout_cost=roofline training_cost=fixed workload=uniform \
    job.n_versions=5 job.batch_size=32 rollout.n_instances=2
```

Run it from the repository root. It should complete in a few seconds.

### What those arguments mean

`policy=areal`
: The staleness policy. `areal` reproduces AReaL's cumulative token-bucket admission
  control. It requires no external package and has a short, inspectable admission
  formula, which makes it the right choice for exploratory runs.

  **Why not the default `policy=psrl`?** The default policy delegates buffer management
  to `psrl.workers.ps.staleness_controller.StalenessInventory`, which lives in the
  separate `psrl` package. On machines where that package is not installed, constructing
  the default policy raises `ImportError` before the simulation begins. A quickstart
  must not depend on an optional external package. See {doc}`../simulation/policies`
  for the full policy comparison and instructions for installing `psrl`.

`rollout_cost=roofline`
: The rollout cost model. `roofline` uses the formula
  `max(F, W + G*n_tokens + A_p*sum_l2/b + A_d*ctxsum)` with default coefficients.
  No profiling artifact is needed.

`training_cost=fixed`
: A fixed training latency (default 1.0 second per step) and zero memory. This is the
  right placeholder when you have no Megatron or FSDP profiling data yet.

`workload=uniform`
: Prompt and response lengths drawn uniformly from the defaults configured in
  `dryrun/config/defaults/workload/uniform.yaml`. No checkpoint or dataset is required.

`job.n_versions=5`
: Run 5 training versions (gradient update steps). Each version requires one batch of
  responses to complete before training proceeds.

`job.batch_size=32`
: 32 responses per training batch.

`rollout.n_instances=2`
: Two parallel rollout instances. Requests are routed to the instance with the fewest
  active requests.

---

## Expected output

The simulator prints a progress bar as it advances through versions, then summarises
the run. The verified output for this exact command is:

```
5 versions, 160 requests consumed, 0 dropped
throughput: 1653.7 tokens/s
max true staleness: 3
staleness distribution: 0:49 / 1:47 / 2:47 / 3:17
```

**Consumed vs. dropped.** 160 responses were consumed (32 per version × 5 versions).
0 were dropped, meaning the `areal` bucket never rejected a request — at this batch
size the rollout keeps pace with training.

**Staleness distribution.** Each consumed response is labelled with its true staleness:
the gap between the weight version it was dispatched under and the version that consumed
it. Version 0 (dispatched and consumed in the same step) counts 49 responses. The
maximum is 3 — some responses dispatched under version 0 were not consumed until
version 3.

---

## What was written

After the run, three directories appear under `outputs/`:

```
outputs/
  logs/
    train.jsonl          # one record per training step
    request.jsonl        # one record per request lifecycle event
    rollout_step.jsonl   # one record per rollout engine step
    instance/
      0.jsonl            # per-step snapshot of rollout instance 0
      1.jsonl            # per-step snapshot of rollout instance 1
  plots/
    staleness_true.png
    staleness_reported.png
    throughput.png
    training_memory.png
    rollout_timeline.png
  .hydra/
    config.yaml          # the fully resolved Hydra config
    overrides.yaml       # the overrides you passed on the command line
main.log                 # Hydra's run log
```

The five PNGs render automatically from the JSONL streams. Regenerate them at any time
from the logs without re-running the simulation, by pointing `dryrun-plot` at the run
directory:

```bash
dryrun-plot outputs
```

It takes the run directory as a positional argument — the `output_dir` of the prior run,
which is `outputs` here — and finds `logs/` and `plots/` beneath it. `--logs-subdir` and
`--plots-subdir` override those names.

---

## Varying the run

### Swap the workload

```bash
dryrun policy=areal rollout_cost=roofline training_cost=fixed workload=bimodal \
    job.n_versions=5 job.batch_size=32 rollout.n_instances=2 \
    workload.output.short_len=1000 workload.output.long_len=8192 \
    workload.output.long_frac=0.4
```

`bimodal` is the right parametric approximation for a truncated distribution like DAPO's
output lengths — see {doc}`../workload/index` for the reasoning. The six available
families are `fixed`, `uniform`, `normal`, `lognormal`, `bimodal`, and `powerlaw`.

### Swap the policy

```bash
dryrun policy=roll rollout_cost=roofline training_cost=fixed workload=uniform \
    job.n_versions=5 job.batch_size=32 rollout.n_instances=2
```

`roll` uses the same admission formula as `areal` but adds a garbage-collection pass
that evicts requests older than `max_staleness` versions before they can contaminate a
training batch. It is a stricter bound on staleness than `areal` at the cost of
potentially higher drop rates.

### Sweep over policies and instance counts

Hydra's `--multirun` mode accepts comma-separated overrides and runs every combination:

```bash
dryrun --multirun policy=areal,roll rollout.n_instances=1,2,4 \
    rollout_cost=roofline training_cost=fixed workload=uniform \
    job.n_versions=5 job.batch_size=32
```

Six runs execute in sequence, each writing to its own subdirectory under `outputs/`
(Hydra appends the override values to the output path). Compare throughput across runs
by reading the final line of each `train.jsonl`.

### Inspect the resolved config

```bash
dryrun --cfg job policy=areal rollout_cost=roofline training_cost=fixed workload=uniform
```

`--cfg job` prints the fully resolved Hydra config and exits without running the
simulation. This is the fastest way to verify that an override reached the right key
before committing to a sweep.

---

## Where to go next

**Profile a real system.** The `roofline` and `fixed` cost models use default
coefficients that are not calibrated to any specific hardware. To simulate your own
cluster accurately, profile it first and fit cost models from the measurements.
{doc}`../profiling/index` explains the four pipelines and how long each sweep takes.

**Read the vocabulary.** Every other page on this site uses *cell*, *feature range*,
*utility*, *staleness*, and *version* as terms of art. {doc}`concepts` defines each one
precisely with a concrete example — it is worth ten minutes before going further.
