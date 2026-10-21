# Simulation outputs

A run prints a summary to the console, writes JSONL telemetry streams under its output
directory, and renders plots from them. Every plot is derived from the streams, so nothing
is lost if the plots are deleted: `srl-plot` regenerates them from the logs without
re-simulating.

## The console summary

`run_simulate` (`srl/cli/main.py`) prints the following after the run.

```text
Simulation complete: <versions> versions in <sim_time>s
  Consumed: <n> requests
  Dropped: <n> requests
  Throughput: <tokens/s> tokens/s
  Max staleness (true): <s>
  Staleness distribution:
    <s>: <n> requests (<pct>%)
  SCG(S) = tokens from trajectories with staleness <= S, per second:
    S=<s>: <tok/s> tok/s (<pct>% of throughput)   <- policy.horizon
  Staleness report: mean=.. p50=.. p95=.. p99=.. max=.. horizon=.. strict=..
  Wasted tokens: <total> (<reason>=<count>, ...)
```

`Consumed` counts responses, so it is `batch_size * group_size` per step. `Throughput` is
generated tokens of consumed responses over `sim_time`.

**Staleness.** Each consumed response carries the weight version that produced each segment of its
tokens. `staleness_true` is `consume_version - v_min` (the oldest weight that produced any
token) and `staleness_reported` is `consume_version - v_max` (the newest). They differ
only for a response whose generation spanned a publish without being restarted, which
needs the `in_place` pause mode.

**SCG(S)**, staleness-constrained goodput, is the tokens of consumed responses with
`staleness_true <= S`, per simulated second. It counts tokens rather than trajectories so it
is directly comparable with `Throughput`: the row for the largest S equals it by
construction, and the ratio at a lower S is the fraction of generated work that arrived
fresh enough. It is a **rate**, so it is comparable across runs only if they consumed the
same number of responses. Check `Consumed` first, because a run that ended early divides a
smaller total by a smaller span. `SimResult` also provides `scg_count(s)` and
`scg_w(weight)`.

**Staleness report** (`StalenessReport`, `srl/simulator/report.py`). Percentiles are
token-weighted nearest-rank over consumed responses, so a long response counts in proportion
to its tokens. `strict` is whether every `staleness_true` is within `configured_horizon`, and
is `None` when the policy has no horizon. `wasted_tokens` maps each drop reason to the
generated tokens of dropped requests, and is printed only when something was dropped.

A `WARNING: Simulation livelocked!` line is printed when the run gave up because it stopped
making progress.

## Output directory

```text
<output_dir>/
├── logs/
│   ├── train.jsonl
│   ├── request.jsonl
│   ├── rollout_step.jsonl
│   ├── gateway.jsonl
│   └── instance/
│       ├── 0.jsonl
│       └── 1.jsonl
├── plots/
│   ├── instance_load.png
│   ├── rollout_step.png
│   ├── staleness_distribution.png
│   ├── request_timeline.png
│   ├── throughput.png
│   └── scg_curve.png
└── run.log
```

Re-running into the same `output_dir` replaces `logs/`. `run.log` is always written.

| Setting | Effect |
|---|---|
| `log_telemetry=false` | nothing but `run.log` is written, no plots. Telemetry has priority over any store |
| `+log_exclude='instance,rollout_step'` | suppresses streams by name or prefix (`instance` covers every `instance/N`) at the source, without building their records. For sweeps whose analysis reads only `request.jsonl` |

All logging is **commit-only**: records are written when state is committed, never for an
event that may still be undone. The writer is `SimTelemetry` (`srl/telemetry/writer.py`).
Streams have no enforced schema beyond what it writes.

## Streams

### `train.jsonl`: one record per training step

| Field | Meaning |
|---|---|
| `t`, `start`, `end`, `dt` | `start` is when the step began. `end` is the clock after publish, including `sync_cost.weight_load_time` when the served version changed. `dt = end - start` |
| `version` | the training version this step produced |
| `n_groups`, `group_size`, `n_reqs` | the consumed batch |
| `total_tokens` | trained tokens, using the trainer's per-axis caps, not generated tokens |
| `train_time` | cost-model optimizer-step latency |
| `sync_time`, `recompute_time` | the other two costs added to the step |
| `peak_memory_bytes` | predicted peak GPU memory |
| `training_tp`, `training_pp`, `training_dp`, `training_cp` | the training mesh |
| `micro_batch_size`, `schedule`, `activation_recompute` | the workload descriptor priced |
| `n_inflight`, `n_ready`, `n_pending` | requests generating, finished and ready, finished but waiting for their group, after the step |
| `n_dropped_step` | requests dropped during this step |
| `extrapolation_notes` | present only when a permitted extrapolation happened, see {doc}`domain-gates` |

### `request.jsonl`: request lifecycle

One record per event. `event` is the discriminator.

| `event` | Emitted when |
|---|---|
| `enqueue` | a group is staged in the router. No policy capacity is used |
| `admit` | the group's first member is placed, once for every member of the group |
| `complete` | the engine finishes a response |
| `consume` | a training batch takes it |
| `drop` | removed without being consumed, with `drop_reason` and `retry` |

Every record carries `t`, `event`, `rid`, `group_id`, `group_index`, `source_group_id`,
`instance_id`, `admit_time`, `dispatch_time`, `complete_time`, `consume_version`,
`prompt_len`, `target_len`, `generated`, `ctx`, `v_traj`, `v_min`, `v_max`,
`staleness_true`, `staleness_reported`, `num_preemptions`, `num_aborts`, `num_reroutes`,
`reprefill_tokens`, `prefix_hit_tokens`, `drop_reason` and `retry`. Fields that are not yet
defined are `null`: `consume_version` and both staleness values until `consume`,
`complete_time` before completion. `consume` rows are the most useful for analysis.

| Field | Meaning |
|---|---|
| `v_traj` | rollout version stamped when the group was admitted. Unadmitted staged requests are rebased until then |
| `v_min`, `v_max` | oldest and newest weight version that produced any token |
| `num_preemptions` | times the engine evicted it |
| `num_aborts` | times a publish (abort mode) or a router reclaim released its KV |
| `num_reroutes` | placements after its first |
| `reprefill_tokens` | context re-prefilled because of aborts and preemptions |
| `prefix_hit_tokens` | prompt tokens skipped by the prefix cache, accumulated over resumptions |
| `drop_reason` | `l2_cancel`, `l3_stale`, `span_cap`, `queue_overflow`, see {doc}`policies` |

### `rollout_step.jsonl`: engine steps

One record per committed engine step or decode segment.

| Field | Meaning |
|---|---|
| `t` | simulation time at the end of the step |
| `instance_id` | the instance |
| `phase` | `mixed`: one engine step that decodes the running batch and prefills what the token budget allows. `decode`: a closed-form segment of many steps with no prefill |
| `steps` | engine steps covered |
| `n_prefill` | prefill tokens the step was charged for |
| `n_prefill_hit`, `n_prefill_computed` | of those, what the prefix cache absorbed and what was computed |
| `n_decode`, `n_tokens` | requests decoded and tokens processed |
| `dt` | predicted duration |
| `saturated`, `saturated_steps` | whether, and for how many steps, the step ran above the cost model's low-load floor |

### `instance/N.jsonl`: per-instance occupancy

One record per change in instance `N`'s committed state. A sample identical to the previous
one in `t_local`, running and waiting counts, blocks used and each request's `(rid,
generated, prefilled)` is skipped.

| Field | Meaning |
|---|---|
| `t`, `instance_id`, `tp` | time, instance and its tensor-parallel degree |
| `num_running_reqs`, `num_waiting_reqs` | active batch and engine-side queue |
| `kv_cache_usage`, `kv_blocks_used` | occupancy against `rollout.kv_blocks` |
| `total_ctx` | context summed over running requests |
| `running_len`, `waiting_len` | per request id: `generated`, `target_len`, `ctx`, `prefilled`, `num_aborts` |

### `gateway.jsonl`: router queue

One record per change in the router's queue, deduplicated the same way.

| Field | Meaning |
|---|---|
| `t` | time |
| `queue_len` | requests waiting in the router: staged requests plus admitted siblings and loopbacks |
| `pending_request_num` | copy of `queue_len`, the name the real gateway uses |
| `running_request_num` | requests the instances hold, waiting plus running |
| `queue_by_version` | queue depth per trajectory version |
| `n_routed`, `n_reroutes`, `n_cross_instance_reroutes` | cumulative placements, placements after a first, and those that changed instance |

A converted real run writes the same stream with the first four fields only, which lets one
plot draw both.

## Plots

`SimPlotter` (`srl/visualizer/plots.py`) reads only the streams.

| File | Contents |
|---|---|
| `instance_load.png` | router queue depth on top (when `gateway` exists), a training band, then one subplot per instance with request count (waiting plus running, and running), and KV utilisation shaded on a second axis. The band splits each training block into `recompute`, `train` and `sync` from the recorded durations, lightest to darkest. A phase with zero duration is not drawn. KV is interpolated rather than stepped, because a decode segment grows blocks continuously between events. A heterogeneous fleet is labelled by TP degree |
| `rollout_step.png` | per instance, `dt` of each `mixed` step and `decode` segment against time, with training intervals shaded |
| `staleness_distribution.png` | counts of consumed responses at each integer staleness, true and reported side by side |
| `request_timeline.png` | generation duration against admission time for consumed responses, coloured by output length |
| `throughput.png` | completions per 5 s window of simulated time |
| `scg_curve.png` | SCG(S) as a step curve over `S = 0 .. max staleness` in tokens per second |

A plot whose stream is empty is drawn as a titled placeholder rather than skipped.

### Re-rendering with `srl-plot`

```bash
srl-plot <output_dir> [--logs-subdir logs] [--plots-subdir plots] \
    [--only instance_load|rollout_step|staleness_distribution|request_timeline|throughput|scg_curve] \
    [--time-label "Simulation Time (s)"]
```

It never touches the logs and exits with an error if `logs/` is missing, for example after
a `log_telemetry=false` run. Without `--only` all six plots are written.

(plotting-a-real-psrl-run)=
## Plotting a real PSRL run with `srl-psrl-plot`

To judge a simulation, draw the real run the same way. There is only one plotter: a real
PSRL log directory is converted into the same streams a simulation writes and rendered by
the same `SimPlotter`, so two pieces of drawing code cannot drift apart.

```bash
srl-psrl-plot /path/to/megatron_psrl_log/<experiment> --output-dir outputs/visualize/<experiment>
```

It reads the PSRL directory read-only, writes `<output_dir>/logs/` in the standard layout and
`<output_dir>/plots/instance_load.png`. Only `instance_load` is rendered, because the other
plots need per-request and per-engine-step streams that a PSRL run does not record. The
wrappers `examples/visualize/plot_psrl_run.sh` (set `PSRL_LOG_DIR`) and
`examples/visualize/plot_simulate_run.sh` (set `SIM_RUN_DIR`) draw both sides with the same
label and copy the figure beside themselves.

| Stream | Source in the PSRL log directory (defaults of `PsrlLogLayout`) |
|---|---|
| `instance/{id}` | `stats_r{id}_dp*.jsonl`, the per-engine scheduler stats |
| `train` | `MainRayTrainer.log` (recompute start) and `TrainWorker_R*.log` (train and sync) |
| `gateway` | `smg_routing_status.jsonl`, the router's queue depth |

Every file name and marker is a field of `PsrlLogLayout` and a matching command-line flag
(`srl-psrl-plot --help`), so a run that names things differently is retargeted without
editing the converter. Three conversions need more than a rename:

- **Rollout versus validation replicas.** Both write the same stats files and the rows carry no
  role. The role is in the `DPLBStatCollector_{role}_I{n}.log` file names, which is what
  `--role` selects on. `--instances 0,2` overrides it. Filtering on "all rows are zero"
  would drop a genuinely idle rollout replica, so it is not used.
- **Two clocks.** The JSONL sinks stamp UTC and the text logs stamp naive local time. The
  offset is the gap between the last text-log line and the last stats sample, rounded to the
  quarter-hour grid real offsets live on. The end of the run is used because the controller
  logs for minutes before the engines come up. Override with `--utc-offset-hours`.
- **Training phases.** The simulator logs one record with three durations. PSRL scatters begin
  and end markers across the controller log and per-rank worker logs. Ranks are unioned, the
  earliest begin and the latest end, since the block claims the interval during which the
  trainer was busy.

Instance numbers are replica indices, not a dense range: replicas 0 and 2 give subplots
`instance 0` and `instance 2`.

### Quantifying the gap with `srl-compare`

Two figures show that a simulation is wrong but not by how much or where. `srl-compare`
reduces both to the scalars that localise a disagreement:

```bash
srl-compare <simulated_output_dir> <converted_real_output_dir> [--logs-subdir logs]
```

It prints one row per metric with the simulated value, the real value, and a relative error
(percent for magnitudes, absolute difference for fractions):

| Metric | Implicates |
|---|---|
| train, recompute and sync per step | the training and recompute cost models |
| span, train steps | the rollout cost model and loop structure |
| peak and mean running requests, peak and mean KV utilisation (busy time) | admission control, the KV denominator, the workload |
| idle fraction | the share of time an instance had zero running requests |

Occupancy metrics are time-weighted, never sample-counted. A real run samples at a fixed rate
while the simulator writes only on a committed change, so counting records would measure the
logging cadence: an idle stretch is one simulated record against many real ones.

`examples/simulate/psrl_aligned.sh` is a worked example of matching a run: every knob
transcribed from the real run's launch script, plus the workload lengths, KV capacity and
training cost that no script reveals.

### Reading the instance-load figure

- In a real run the waiting and total-request lines coincide, because the engine reports no
  waiting requests and queueing happens in the router. The simulator queues in the router
  too, so both sides draw the router row, which is the right place to compare backlog. The
  simulated engine-side queue holds only work an instance has accepted.
- The router queue is global and is not attributed to instances.
- With the default `recompute_cost` of `fixed` at `0.0`, a simulated band has no `recompute`
  segment, whereas a real run spends part of its block on the log-prob pass. The absence is
  left visible rather than modelled away.
