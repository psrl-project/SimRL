# Simulation outputs

A completed simulation writes four JSONL streams and five plots under its output
directory. Every file is derived from the streams, so nothing is lost if the plots are
deleted, because `srl-plot` can regenerate them from the logs alone without re-simulating.

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
  // "extrapolation_notes": [...]: present only when a cost query extrapolated
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
| `enqueue` | Request staged in the router; no policy capacity is consumed yet |
| `admit` | First group member is placed, atomically admitting the whole group |
| `complete` | Rollout engine finishes generation |
| `consume` | Policy takes the request as part of a training batch |
| `drop` | Policy or engine evicts the request without consuming it |

The most useful event for post-analysis is `consume`, which has all fields populated
including `consume_version`, `v_min`, `v_max`, `staleness_true`, and
`staleness_reported`.

Key fields (all events carry `t`, `event`, `rid`, `group_id`, `instance_id`):

| Field | Meaning |
|-------|---------|
| `admit_time` | Simulation time when the group's first member was placed |
| `complete_time` | Time when generation finished (`null` for events before completion) |
| `consume_version` | Training version that consumed this request (`null` until consume) |
| `prompt_len` | Prompt length in tokens |
| `target_len` | Requested output length |
| `generated` | Tokens actually generated |
| `v_traj` | Weight version bound at first placement (staged requests can rebase before then) |
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

### `gateway.jsonl`: router queue depth

One record per change in the router's queue, de-duplicated the same way. A simulation and a
converted real run write the **same** schema, which is what lets one `plot_instance_load` draw
both and `srl-compare` put like against like. On the real side the first four fields come
from `smg_routing_status.jsonl`; the reroute counters are simulation-only.

| Field | Meaning |
|-------|---------|
| `t` | Simulation time of this snapshot |
| `queue_len` | Staged requests plus admitted siblings or reroutes awaiting placement |
| `pending_request_num` | Duplicate of `queue_len`, as the real gateway reports it |
| `running_request_num` | Requests the instances hold, running or queued |
| `n_routed` | Cumulative placements, including re-placements |
| `n_reroutes` | Cumulative placements after a request's first |
| `n_cross_instance_reroutes` | Of those, how many moved to a different instance |

The last field is the one to watch after a weight update: the matched real run put it at 58.5%
of loopbacks, and before the router existed it was structurally 0%. See {doc}`routing`.

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

Router queue depth on top, then the training band, then one subplot per rollout instance. Each
subplot shows the number of running and waiting requests over time (left axis) and KV
cache utilisation (right axis, shaded). KV utilisation is rendered with linear
interpolation rather than step interpolation, because decode segments involve continuous
KV growth between logged events.

The band splits each training block into its three phases, drawn lightest to darkest:
`recompute` (the log-prob forward pass), `train` (the optimizer step), and `sync` (the
weight push). A phase with zero duration is not drawn at all, so an unmodelled cost shows
up as an absence rather than being folded silently into its neighbour. With the default
`recompute_cost` of `0.0`, a simulated band therefore shows `train` alone.

The simulator records the three durations but never orders them. It only sums them into
one block. The order drawn here is the one a real PSRL run exhibits.

This is the one figure that renders identically for a simulation and for a real run,
which is what makes the two directly comparable. See {ref}`plotting-a-real-psrl-run`.

### `request_timeline.png`

Scatter plot with admission time on the x-axis, generation duration on the y-axis, and
output token count as the colour. Long requests are easy to spot as tall, warm-coloured
points. A horizontal band of similar durations is typical of a DAPO-style workload with
a strong bimodal length distribution.

---

## Re-rendering plots with `srl-plot`

If you want to iterate on plot styling, or if the plots were not generated (e.g. because
you ran with an older version), you can regenerate them from an existing `logs/`
directory without re-simulating:

```bash
srl-plot outputs/simulate/<name>
```

This reads `logs/` under the given directory and writes to `plots/` in the same
directory. Both paths can be overridden:

```bash
srl-plot outputs/simulate/<name> \
    --logs-subdir logs \
    --plots-subdir plots_v2
```

`srl-plot` never touches the JSONL logs. It is safe to run on a completed simulation
as many times as needed. If `logs/` does not exist, for example because the run used
`log_telemetry: false`, it exits with a clear error rather than silently writing empty
plots.

Pass `--only instance_load` to render just that one figure, and `--time-label` to match
the axis label of a real run you intend to compare against.

---

## Comparing a simulation against a real run

Two scripts under `examples/visualize/` produce the same figure from the two sources and
drop it beside themselves, so the images sit side by side:

```bash
# Real PSRL run -> examples/visualize/instance_load_real.png
PSRL_LOG_DIR=/path/to/megatron_psrl_log/<experiment> bash examples/visualize/plot_psrl_run.sh

# Simulation    -> examples/visualize/instance_load_simulated.png
SIM_RUN_DIR=outputs/simulate/<name> bash examples/visualize/plot_simulate_run.sh
```

Both go through the same `SimPlotter.plot_instance_load`, and both default to the
`Wall-clock Time (s)` axis label, so the only difference between the two images is the
data. Override `FIGURE` to put a figure elsewhere.

For the comparison to mean anything, the simulation has to be configured to match the
real run: same instance count, batch size, staleness, prompt and response caps, and
enough versions to cover a similar span. A simulation with different topology produces a
correct figure of a different experiment.

---

(plotting-a-real-psrl-run)=
## Plotting a real PSRL run with `srl-psrl-plot`

To judge whether a simulation is accurate, draw the real run the same way. The
simulator's accuracy claim is only as good as the comparison, and two figures produced
by two pieces of drawing code will drift apart. So there is only one plotter: a real
PSRL log directory is converted into the same streams a simulation writes, and then
rendered by the same `SimPlotter`.

```bash
bash examples/visualize/plot_psrl_run.sh
```

or directly:

```bash
srl-psrl-plot /path/to/megatron_psrl_log/<experiment> \
    --output-dir outputs/visualize/<experiment>
```

This writes `<output_dir>/logs/` in the standard layout and
`<output_dir>/plots/instance_load.png`. The PSRL directory is read only, never modified.
Only `instance_load` is rendered: the other four plots need per-request and
per-engine-step streams that a PSRL run does not record.

Nothing has to be changed on the PSRL side. Everything needed is already written to the
log directory.

### What is read from where

| Stream | Source in the PSRL log directory |
|---|---|
| `instance/{id}` | `stats_r{id}_dp0.jsonl`, the per-engine vLLM scheduler stats, sampled at 1 Hz |
| `train` | `MainRayTrainer.log` (recompute start) and `TrainWorker_R*.log` (train and sync) |
| `gateway` | `smg_routing_status.jsonl`, the router's queue depth |

The engine stats already use the same field names the simulator writes
(`num_running_reqs`, `num_waiting_reqs`, `kv_cache_usage`), so the conversion is close to
a rename. Three things genuinely differ:

**Rollout replicas versus validation replicas.** Validation replicas write the same stats
files as rollout ones, and the rows carry no role field. The role lives only in the
`DPLBStatCollector_{role}_I{n}.log` filenames, which is what `--role` selects on. Filtering
by "all rows are zero" instead would silently drop a genuinely idle rollout replica, so
that heuristic is deliberately not used. Override the selection with `--instances 0,2`.

**Two clocks.** The JSONL sinks stamp UTC while the Python text logs stamp naive local
time. The offset is recovered by differencing the two clocks at the *end* of the run,
where the last text-log line and the last stats sample are within seconds of each other,
and rounding onto the quarter-hour grid real UTC offsets live on. Anchoring at the start
would not work, because the controller begins logging minutes before the engines come up
and that gap is genuine rather than clock skew. Override with `--utc-offset-hours`.

**Training phases.** The simulator logs the block as one record with three durations, while
PSRL scatters the same three across a controller log and the per-rank train worker logs
as begin/end markers. Ranks are unioned (earliest begin, latest end) rather than sampled,
since the block is claiming the interval during which the trainer was busy.

Every filename and marker above is a field on `PsrlLogLayout` with a matching command-line
flag, so a run that names things differently is retargeted without editing the converter.

### Quantifying the error with `srl-compare`

Two figures side by side show that a simulation is wrong, not by how much or where,
and eyeballing a KV curve cannot separate a cost-model error from a workload
mismatch. `srl-compare` reports the scalars that localise the disagreement:

```bash
srl-compare outputs/simulate/<name> outputs/visualize/<experiment>
```

```
metric                    simulated         real      error
------------------------------------------------------------
train / step (s)             116.61       110.95      +5.1%
recompute / step (s)          31.37        31.57      -0.6%
peak running reqs                64           64      +0.0%
peak KV util                  0.892        1.000     -10.8%
idle fraction                 0.067        0.257     -0.190
```

Occupancy metrics are **time-weighted, never sample-counted**. The two sides log on
different triggers: a real run samples at a fixed 1 Hz while the simulator writes
only on a committed state change, so an idle stretch yields one simulated record
against a hundred real ones. A sample-counted idle fraction reported 21% against 59%
for two runs whose instances both drained to exactly zero for every training block.

`examples/simulate/psrl_aligned.sh` is a worked example: every knob
transcribed from the run being matched, plus the three things no script reveals --
the workload lengths, the KV capacity, and the training cost for that exact mesh.
`outputs/simulate/ALIGNMENT_LOG.md` records what each fix bought.

### Reading the resulting figure

**The orange and green lines coincide in a real run.** Its engine reports
`num_waiting_reqs = 0` throughout, because queuing happens in the router rather than in the
engine. This is real behaviour, not a wiring bug.

The simulator now queues in the router too, so **both sides draw the row** and it is the right
place to compare backlog. The simulated engine's waiting queue is no longer the whole fleet's
backlog, only work an instance has accepted, so its two lines separate much less than they used
to. The router queue is global and is deliberately not attributed to individual instances.
See {doc}`routing`.

**Instance numbers are replica indices, not a dense range.** A run whose rollout replicas
are 0 and 2 produces subplots titled `instance 0` and `instance 2`.

**The simulated band has no `recompute` segment.** The default `recompute_cost` is `0.0`,
whereas a real run spends roughly a quarter of its training block on the log-prob forward
pass. That difference is the point of drawing both: it is left visible rather than
modelled away.

