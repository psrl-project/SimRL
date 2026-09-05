# Simulation outputs

A completed simulation writes four JSONL streams and five plots under its output
directory. Every file is derived from the streams, so nothing is lost if the plots are
deleted, because `dryrun-plot` can regenerate them from the logs alone without re-simulating.

Setting `log_telemetry: false` in the Hydra config writes nothing to disk at all.
The simulation still runs, but the output directory will contain only `run.log`. This
is useful when running many parameter sweeps where keeping logs for every candidate
would consume significant disk space.

---

## Output directory layout

```
outputs/simulate/<name>/
├── logs/
│   ├── train.jsonl          ← one record per training step
│   ├── request.jsonl        ← one record per request lifecycle event
│   ├── rollout_step.jsonl   ← one record per rollout engine step
│   └── instance/
│       ├── 0.jsonl          ← occupancy samples for instance 0
│       └── 1.jsonl          ← occupancy samples for instance 1, etc.
├── plots/
│   ├── throughput.png
│   ├── rollout_step.png
│   ├── staleness_distribution.png
│   ├── instance_load.png
│   └── request_timeline.png
└── run.log
```

Re-running a simulation with the same `output_dir` replaces the `logs/` directory
rather than appending to it. Each run starts fresh.

---

## Stream reference

### `train.jsonl`: one record per training step

This is the primary output stream. Every line records what the simulator decided about
one training step: when it started and ended, which batch it consumed, the predicted cost,
and how many requests were in flight at the time.

**Worked example** (Qwen3-8B on 8x H20, see
{doc}`../getting-started/walkthrough/06-simulate`):

```javascript
{
  "t": 271.34,             // simulation wall time when the step ended
  "start": 226.86,         // when training began for this step
  "end": 271.34,           // when training ended (same as t)
  "dt": 44.47,             // end - start in seconds
  "version": 1,            // the weight version this step produced
  "batch_size": 64,        // configured batch size
  "n_reqs": 64,            // requests actually consumed (may differ if policy dropped some)
  "total_tokens": 555469,  // sum of all output token counts across the batch
  "train_time": 44.47,     // predicted training latency from the cost model
  "peak_memory_bytes": 80777577749,  // predicted peak GPU memory across all ranks
  "training_tp": 1,
  "training_pp": 1,
  "training_dp": 8,
  "training_cp": 1,
  "micro_batch_size": 1,
  "schedule": "fsdp",
  "activation_recompute": "full",
  "sync_time": 0.0,        // weight-sync cost added after training
  "recompute_time": 0.0,   // partial-rollout recompute cost
  "n_inflight": 64,        // requests still generating when this step began
  "n_ready": 128,          // requests in the ready queue
  "n_dropped_step": 0      // requests discarded by the policy at this step
  // "extrapolation_notes": [...] -- present only when a cost query extrapolated
}
```

`extrapolation_notes` appears only when a cost model answered an out-of-domain query
for a permitted feature (see {doc}`domain-gates`). When present it contains one string
per extrapolated feature, naming the feature, the queried value, and the profiled range.

### `request.jsonl`: request lifecycle events

Each line represents one event in a request's lifecycle. The `event` field is the
discriminator:

| `event` value | When emitted |
|--------------|-------------|
| `admit` | Request accepted by the staleness policy and dispatched to the rollout engine |
| `complete` | Rollout engine finishes generation |
| `consume` | Policy takes the request as part of a training batch |
| `drop` | Policy or engine evicts the request without consuming it |

The most useful event for post-analysis is `consume`, which has all fields populated
including `consume_version`, `v_min`, `v_max`, `staleness_true`, and
`staleness_reported`.

Key fields (all events carry `t`, `event`, `rid`, `group_id`, `instance_id`):

| Field | Meaning |
|-------|---------|
| `admit_time` | Simulation time when the request was dispatched |
| `complete_time` | Time when generation finished (`null` for events before completion) |
| `consume_version` | Training version that consumed this request (`null` until consume) |
| `prompt_len` | Prompt length in tokens |
| `target_len` | Requested output length |
| `generated` | Tokens actually generated |
| `v_traj` | Weight version the rollout engine was serving when the request was admitted |
| `v_min` | Oldest weight version that produced any token for this request |
| `v_max` | Newest weight version producing any token. Differs from `v_min` only if partial rollout spanned a push |
| `staleness_true` | `consume_version - v_min` (null until consume) |
| `staleness_reported` | `consume_version - v_max` (null until consume) |
| `num_preemptions` | Times this request was paused and resumed by the scheduler |
| `num_aborts` | Times this request was aborted by a weight push |

`staleness_true` and `staleness_reported` differ only when a request spans multiple
weight versions due to partial rollout. In the common case they are equal.

### `rollout_step.jsonl`: engine step telemetry

One record per rollout engine iteration (one prefill step or one decode step for one
instance). These records drive the `rollout_step` plot.

| Field | Meaning |
|-------|---------|
| `t` | Simulation time at the end of this step |
| `instance_id` | Which rollout instance executed this step |
| `phase` | `prefill` or `decode` |
| `steps` | Number of token-generation steps in this decode segment |
| `n_prefill` | Requests prefilled |
| `n_decode` | Requests decoded |
| `n_tokens` | Tokens processed |
| `dt` | Predicted wall-clock duration |
| `saturated` | Whether the engine was at capacity during this step |
| `saturated_steps` | Number of steps that hit the token or concurrency cap |

### `instance/N.jsonl`: per-instance occupancy

One record per occupancy sample for instance `N`. Records are de-duplicated: if the
instance state has not changed since the last sample, the writer skips the write.
Genuine state changes (a new request arriving, a request completing, a weight abort)
always produce a record.

| Field | Meaning |
|-------|---------|
| `t` | Simulation time of this snapshot |
| `instance_id` | Instance identifier |
| `num_running_reqs` | Requests in the active batch |
| `num_waiting_reqs` | Requests queued but not yet scheduled |
| `kv_cache_usage` | KV cache occupancy as a fraction of total capacity |
| `kv_blocks_used` | Number of KV blocks in use |
| `total_ctx` | Sum of context lengths across running requests |
| `running_len` | Per-request `{rid: {generated, target_len, ctx, prefilled}}` mapping |
| `waiting_len` | Same structure for the waiting queue |

---

## The five plots

### `throughput.png`

Sliding-window completion rate over simulation time. The window is 5 seconds by default.
This is the most direct measure of training data throughput: a flat line means the loop
is running at a steady cadence. A dip means the policy is starving training of fresh
data.

### `rollout_step.png`

Per-instance scatter of engine step duration over time. Prefill steps are shown in red
and decode segments in blue. Training intervals appear as shaded bands. The plot makes
it easy to see whether the rollout engine is keeping up with training: if decode segments
are visibly longer than prefill steps, the engine is spending more time on continuation
than on new requests, which may indicate a KV cache pressure or scheduling issue.

### `staleness_distribution.png`

Grouped integer bar chart showing the distribution of consumed request staleness across
the whole run. Two bars appear for each staleness value: the *true* staleness
(`consume_version - v_min`, the oldest weight any token came from) and the *reported*
staleness (`consume_version - v_max`, the newest weight any token came from). They
differ only when partial rollout resumed a request under a different weight version. Use
this plot to verify that the staleness policy is holding the bound you configured. If
the bars extend beyond `job.max_staleness`, the policy invariant is not being respected.

### `instance_load.png`

Training interval band at the top, with one subplot per rollout instance below. Each
subplot shows the number of running and waiting requests over time (left axis) and KV
cache utilisation (right axis, shaded). KV utilisation is rendered with linear
interpolation rather than step interpolation, because decode segments involve continuous
KV growth between logged events.

### `request_timeline.png`

Scatter plot with admission time on the x-axis, generation duration on the y-axis, and
output token count as the colour. Long requests are easy to spot as tall, warm-coloured
points. A horizontal band of similar durations is typical of a DAPO-style workload with
a strong bimodal length distribution.

---

## Re-rendering plots with `dryrun-plot`

If you want to iterate on plot styling, or if the plots were not generated (e.g. because
you ran with an older version), you can regenerate them from an existing `logs/`
directory without re-simulating:

```bash
dryrun-plot outputs/simulate/<name>
```

This reads `logs/` under the given directory and writes to `plots/` in the same
directory. Both paths can be overridden:

```bash
dryrun-plot outputs/simulate/<name> \
    --logs-subdir logs \
    --plots-subdir plots_v2
```

`dryrun-plot` never touches the JSONL logs. It is safe to run on a completed simulation
as many times as needed. If `logs/` does not exist, for example because the run used
`log_telemetry: false`, it exits with a clear error rather than silently writing empty
plots.
