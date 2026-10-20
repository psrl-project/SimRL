# Stage 0: simulate with nothing measured

The simulator does not need profiling data to run. It ships analytic cost models with
default coefficients, so the first thing worth doing is a run that touches no GPU. It takes a
few seconds and confirms the install works end to end.

## What it needs

The base install from {doc}`../installation`, including the `psrl` package that `srl`
imports at startup.

## The command

```bash
srl policy=areal rollout_cost=roofline training_cost=fixed workload=uniform \
    job.n_versions=5 job.batch_size=32 rollout.n_instances=2
```

Run it from the repository root. Each override selects one config group or sets one key:

`policy=areal`
: The staleness policy preset. `areal` reproduces AReaL's cumulative token-bucket admission.
  Its one knob is `policy.max_head_offpolicyness`, which defaults to 0. The other presets are
  `dressage`, `nemo_rl`, `openrlhf`, `prime_rl`, `psrl` (the default), `roll`, `skyrl`,
  `slime`, `trl`, `verl`, and `verl_v1`, and each carries its own system's knobs.
  {doc}`../../simulation/policies` compares them.

`rollout_cost=roofline`
: The rollout cost model, evaluating `max(F, W + G*n_tokens + A_p*sum_l2/b + A_d*ctxsum)`
  with the defaults in `srl/config/defaults/rollout_cost/roofline.yaml`.
  {doc}`01-vllm-rollout` replaces those defaults with fitted ones.

`training_cost=fixed`
: A flat `latency_s` (1.0 s) per training step and `peak_memory_bytes` of zero. The right
  placeholder until stage 3 or 4 has produced a fitted artifact.

`workload=uniform`
: Response lengths drawn uniformly from 50 to 200 tokens and a fixed 512-token prompt, from
  `srl/config/defaults/workload/uniform.yaml`. No checkpoint, no dataset.

`job.n_versions=5`
: Five training versions, meaning five gradient updates.

`job.batch_size=32`
: 32 prompt **groups** per training step. A parametric workload has one response per group,
  so this is also 32 responses. With a grouped trace, a step consumes
  `batch_size * group_size` responses.

`rollout.n_instances=2`
: Two rollout instances. The default router (`rollout.router.strategy=request_num_balance`)
  sends each request to the instance with the fewest in-flight requests.

## What it prints

```text
Simulation complete: 5 versions in 14.03s
  Consumed: 160 requests
  Dropped: 0 requests
  Throughput: 1385.9 tokens/s
  Max staleness (true): 0
  Staleness distribution:
    0: 160 requests (100.00%)
  SCG(S) = tokens from trajectories with staleness <= S, per second:
    S=0:    1385.9 tok/s (100.0% of throughput)  <- policy.horizon
  Staleness report: mean=0.00 p50=0 p95=0 p99=0 max=0 horizon=0 strict=True
```

followed by the log and plot locations. The workload is seeded, so the run is reproducible.

`Consumed: 160` is 32 per version times 5 versions. `areal` with its default
`max_head_offpolicyness=0` admits only what the current version can train on, so every
response is consumed at staleness 0. The 14.03 s is **simulated** time, not wall clock: the
run itself finishes in a few seconds.

The `SCG(S)` rows report tokens per second from trajectories consumed with true staleness at
most `S`. The last row equals `Throughput`, and a lower row is the share of generated work
that arrived fresh enough. It is a rate, so compare it across runs only when `Consumed`
matches.

Allow staleness and the distribution spreads out:

```bash
srl policy=areal policy.max_head_offpolicyness=2 rollout_cost=roofline training_cost=fixed \
    workload=uniform job.n_versions=5 job.batch_size=32 rollout.n_instances=2
```

```text
Simulation complete: 5 versions in 11.48s
  Consumed: 160 requests
  Dropped: 0 requests
  Throughput: 1645.3 tokens/s
  Max staleness (true): 3
  Staleness distribution:
    0: 49 requests (30.63%)
    1: 47 requests (29.38%)
    2: 47 requests (29.38%)
    3: 17 requests (10.62%)
```

True staleness is the gap between the consuming version and the *oldest* weight version
behind the response's tokens, so it can exceed the policy's horizon. {doc}`../concepts`
separates true from reported staleness.

## What lands on disk

`output_dir` defaults to `outputs`, and the run writes telemetry there:

```text
outputs/
├── logs/
│   ├── train.jsonl            one record per training step
│   ├── request.jsonl          one record per request lifecycle event
│   ├── rollout_step.jsonl     one record per batch of engine steps
│   ├── gateway.jsonl          router queue snapshots
│   └── instance/
│       ├── 0.jsonl            per-step snapshot of instance 0
│       └── 1.jsonl            per-step snapshot of instance 1
├── plots/
│   ├── instance_load.png
│   ├── request_timeline.png
│   ├── rollout_step.png
│   ├── scg_curve.png
│   ├── staleness_distribution.png
│   └── throughput.png
├── run.log                    the simulator's own log
└── <date>/<time>/             Hydra's run directory
    ├── main.log
    └── .hydra/
        ├── config.yaml        the fully resolved config
        ├── hydra.yaml
        └── overrides.yaml     the overrides you passed
```

Hydra's run directory is separate from `output_dir`. Re-running with the same `output_dir`
replaces `logs/` instead of appending to it. {doc}`../../simulation/outputs` documents every
stream. The ones used most are below.

One `train.jsonl` record from the first run:

```javascript
{
  "t": 5.6254,                // simulated time at the end of this step
  "start": 4.6254,
  "end": 5.6254,
  "dt": 1.0,                  // training_cost=fixed, so exactly latency_s
  "version": 2,
  "n_groups": 32,
  "group_size": 1,
  "n_reqs": 32,               // n_groups * group_size
  "total_tokens": 20178,      // summed across the batch
  "train_time": 1.0,
  "peak_memory_bytes": 0,     // the fixed model reports zero
  "training_tp": 1, "training_pp": 1, "training_dp": 1, "training_cp": 1,
  "micro_batch_size": 1,
  "schedule": "1f1b",
  "activation_recompute": "none",
  "sync_time": 0.0,
  "recompute_time": 0.0,
  "n_inflight": 32,           // still generating
  "n_ready": 0,               // finished, waiting for a batch
  "n_pending": 0,
  "n_dropped_step": 0
}
```

`peak_memory_bytes: 0` and `dt: 1.0` are the placeholder cost models showing through. After
stages 3 or 4 both carry predicted values, and a fitted training model adds an
`extrapolation_notes` field whenever a query left its fitted range.

`request.jsonl` holds one line per lifecycle event (`enqueue`, `admit`, `complete`,
`consume`, and `drop` when a policy discards a response), each carrying the request's
`prompt_len`, `target_len`, `v_traj`/`v_min`/`v_max`, `staleness_true`, and preemption and
re-prefill counters. `rollout_step.jsonl` has `phase`, `n_prefill`, `n_decode`, `n_tokens`,
and `dt` per run of engine steps, and `instance/<i>.jsonl` has `kv_cache_usage`,
`num_running_reqs`, and `num_waiting_reqs` over time.

The plots render from the JSONL streams. Regenerate them, or one of them, without
re-simulating:

```bash
srl-plot outputs
srl-plot outputs --only throughput --plots-subdir plots_alt
```

`srl-plot` takes the run directory, looks for `logs/` beneath it, and accepts `--logs-subdir`,
`--plots-subdir`, `--only`, and `--time-label`.

## How to tell it worked

A progress bar, the summary block above, and files under `outputs/`. `Consumed` equals
`job.n_versions * job.batch_size * group_size` unless the policy dropped something, in which
case `Dropped` says how many.

## Varying it

Swap the workload shape:

```bash
srl policy=areal rollout_cost=roofline training_cost=fixed workload=bimodal \
    job.n_versions=5 job.batch_size=32 rollout.n_instances=2 \
    workload.output.short_len=1000 workload.output.long_len=8192 \
    workload.output.long_frac=0.4
```

`bimodal` is the parametric shape closest to a truncated output distribution.
{doc}`../../workload/index` lists all six families.

Check that an override landed where you meant it to, without running anything:

```bash
srl --cfg job policy=areal policy.max_head_offpolicyness=2 workload=uniform
```

`--cfg job` prints the resolved config and exits. An unknown key fails here, in Hydra's
composition, with `Could not override '<key>'`.

Sweep several configurations in one invocation with Hydra's multirun. Telemetry goes to
`output_dir`, not to the per-job Hydra directory, so point it at the job directory or every
job overwrites the same `outputs/logs`:

```bash
srl --multirun policy=areal,roll rollout.n_instances=1,2,4 \
    rollout_cost=roofline training_cost=fixed workload=uniform \
    job.n_versions=5 job.batch_size=32 \
    'output_dir=${hydra:runtime.output_dir}'
```

Six jobs run in sequence under `multirun/<date>/<time>/<job number>/`, each with its own
`logs/`, `plots/`, and `run.log`. Add `log_telemetry=false` when a sweep only needs the printed
summary: nothing is written to disk and no plots are made.

## Next

The `roofline` and `fixed` coefficients are not calibrated to any hardware.
{doc}`01-vllm-rollout` measures the real inference engine and fits them.
