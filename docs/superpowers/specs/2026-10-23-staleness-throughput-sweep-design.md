# Staleness-Throughput Sweep Harness

A per-framework knob sweep that decomposes generation throughput by the staleness of
the data it produced, for the third post in the async-RL staleness series
(`/apdcephfs_zwfy/share_303760348/lhy/blog/async-rl-staleness-followups.todo.md`).

## Problem

The first two posts describe staleness *mechanisms* (L1 admission, L2 rollout runtime,
L3 pre-training filter) across eleven real systems. The third post needs measurements:
what each system's own knobs actually do to the staleness composition of its output,
on one workload, one mesh, one budget.

Today SimRL can answer this for exactly one knob of one preset
(`/apdcephfs_zwfy/share_303760348/lhy/SimRL/examples/analysis/verl_staleness_sweep.sh`),
and it plots the *cumulative* form SCG(S). There is no harness that covers the other
eleven presets, no two-dimensional grid, and no figure that shows where generated
tokens go when a knob discards them.

## The figure this exists to produce

One stacked bar per knob setting. The stack is:

- one segment per `staleness_true` level, from the run's `consume` rows, and
- one hatched segment per `drop_reason`, from its `drop` rows.

Bar height is therefore `(all generated tokens) / span` — the full generation
throughput of the fleet. The boundary between the consumed segments and the waste
segments is the *effective training throughput*, and the top of the stack is what the
hardware actually produced. A knob that tightens staleness moves tokens downward
across that boundary instead of making them disappear, which is the trade-off the post
is about.

X axis is the preset's primary knob. One panel per secondary-knob value.

## Measured regime

Pinned identically for every preset:

```
trace        /apdcephfs_zwfy/share_303760348/lhy/SimRL/outputs/workload/qwen3_8b_dapo_38912_tp1/joint.jsonl
             1024 groups x 8, cap 38912, mean 10660 tok, median 8673, p95 25251,
             truncated_fraction 0.0045
model        Qwen3-8B
rollout      2 instances x TP1          (2 GPUs)
training     hydraulis TP2 / PP3 / DP1  (6 GPUs)
batch_size   32 groups      group_size 8      n_versions 12
token_budget 12800          max_tokens_per_microbatch 12800
rollout cost composed, kernel table outputs/profile/qwen3_8b/kernel_tp1/kernel
train cost   outputs/cost/qwen3_8b_tp2_pp3_thd/training_hydraulis.json
```

The split and the mesh are the ones the aligned PSRL run uses
(`/apdcephfs_zwfy/share_303760348/lhy/SimRL/examples/simulate/psrl_aligned.sh`), so the
cost models are being asked for predictions in the configuration they were matched
against. `n_versions` is 12 rather than that script's 5 because staleness cannot exceed
the number of versions a run contains: a probe at 8 versions reached S=7, which is the
run length rather than the mechanism.

### Probe results that fixed this regime

Five runs, `n_versions=8`, otherwise as above. Every run consumed 2048 trajectories, so
the rows are comparable.

| run | span (s) | consumed tok | wasted tok | tok/s | staleness composition |
|---|---|---|---|---|---|
| `verl_v1` warmup=1, thr=8 | 14478 | 22.3M | 0 | 1539 | all S0 |
| `verl_v1` warmup=4, thr=8 | 10701 | 22.3M | 0 | 2082 | S0..S4 |
| `verl_v1` warmup=4, thr=1 | 13080 | 21.4M | 6.2M `l3_stale` | 2106 | S0..S3 |
| `verl_v1` warmup=8, thr=8 | 10627 | 22.3M | 0 | 2097 | S0..S7 |
| `psrl` staleness=2 | 10627 | 22.3M | 0 | 2097 | S0..S2 |

Three facts follow, and the harness is shaped by them.

**The default is synchronous.** At `num_warmup_batches=1` every token is consumed at
staleness 0. Two rollout GPUs against six training GPUs on a 10.7k-mean workload leaves
rollout so far behind that a batch is generated entirely under one weight version. No
L3 threshold can bind, at any value, because nothing is ever stale. This is a property
of the regime, not a bug, and it has to be stated in the post: the async machinery
disengages when rollout is the bottleneck and admission is not opened.

**The L1 admission knob is a throughput lever here.** `num_warmup_batches` 1 -> 4 cuts
span by 1.353x by letting rollout run ahead of training, and saturates past 4
(10701 -> 10627 s). Earlier work recorded admission as a throughput gate rather than a
scheduling option; this is that effect, measured.

**The L3 knob charges twice.** Tightening `max_off_policy_threshold` 8 -> 1 at
warmup=4 holds consumed trajectories at 2048 but costs 22% of span (10701 -> 13080 s)
*and* discards 6.2M of 27.6M generated tokens to `l3_stale`. Both costs land in one
stacked bar: the stack gets shorter in tokens/s and grows a hatched segment.

**psrl bounds staleness without paying either.** At `staleness=2` it caps the
composition at S<=2 with zero waste and the same 10627 s span as the unbounded
warmup=8 run — 1.23x faster than `verl_v1` at thr=1 for a comparable bound. This is the
claim the second post makes mechanically; the harness is what tests it across loads.

## Architecture

Three layers, matching the existing measure/model/simulate seams.

### Reduction: `srl/telemetry/reduce.py` (new)

Pure, torch-free, matplotlib-free, unit-testable.

```python
@dataclass(frozen=True)
class StalenessBreakdown:
    span: float
    n_consumed: int
    consumed_tokens: dict[int, int]   # staleness_true -> tokens
    wasted_tokens: dict[str, int]     # drop_reason -> tokens
    def scg(self, s: int) -> float
    def throughput(self) -> float     # (consumed + wasted) / span

def reduce_request_stream(path_or_store) -> StalenessBreakdown
```

`span` is `max(t)` over all request rows, the definition `SimResult.sim_time` and
every existing sweep figure already use, so new numbers stay comparable with old ones.

`/apdcephfs_zwfy/share_303760348/lhy/SimRL/examples/analysis/plot_policy_cap_sweep.py`
carries an inline copy of this reduction. It is repointed at the library and its copy
deleted, so one definition of staleness-throughput exists in the repo.

### Figure: `srl/visualizer/staleness_sweep.py` (new)

`plot_staleness_throughput(breakdowns, primary_label, out_path, ...)`.

Consumed segments use the blue sequential ramp already declared in
`srl/visualizer/plots.py` (darker = staler); waste segments are hatched and drawn in a
neutral grey, one per `drop_reason`, so they never read as a staleness level. Each bar
is annotated with its span. A solid line marks the consumed/waste boundary.

Exposed as a console script `srl-staleness-plot` alongside the existing seven.

### Harness: `examples/simulate/staleness_throughput/`

```
common.sh            the pinned regime above; sourced by every grid
sweep.sh             shared runner: expands a grid, runs points in parallel,
                     enforces the gates, calls the plotter
run_all.sh           every preset in sequence
README.md            method, gates, the synchronous-degeneracy finding
<preset>/grid.sh     ONLY: PRIMARY_KNOB, PRIMARY_VALUES,
                     SECONDARY_KNOB, SECONDARY_VALUES, NOTES
<preset>/figures/    output PNGs
```

Twelve preset directories: the eleven systems plus `verl_v1`, which is a separate
preset from `verl` and carries the two knobs this study is centred on. A `grid.sh` is
about ten lines and holds no runner logic, so a correction to the method reaches all
twelve at once.

`sweep.sh` drives
`/apdcephfs_zwfy/share_303760348/lhy/SimRL/examples/simulate/psrl_aligned.sh`, which
already transcribes the mesh, the KV sizing and the cost artifacts, and appends to
`EXTRA_OVERRIDES` rather than assigning it.

## Knob grids

Primary knobs are the ones `srl/policy/README.md` names as most directly related to
staleness. `psrl`, `areal` and `roll` take their threshold through
`SimConfig.staleness_horizon` rather than a constructor argument; setting the preset
field drives both that and `ingress_window_groups`, so no extra plumbing is needed.

The "secondary" column is written into each `grid.sh` and commented out, so expanding a
preset to two dimensions later is a one-line edit rather than a fresh design pass.

| preset | primary (x axis), run now | secondary, deferred |
|---|---|---|
| `verl_v1` | `num_warmup_batches` 1,2,3,4,6,8 | `max_off_policy_threshold` 1,2,4,8 — **run now** |
| `verl` | `staleness_threshold` 0.2 .. 3.0 | `require_batches` 1,2 |
| `psrl` | `staleness` 0 .. 6 | `proactive_filter` false/true |
| `areal` | `max_head_offpolicyness` 0 .. 6 | `max_inflight` |
| `roll` | `async_generation_ratio` 0.5 .. 4 | `max_inflight` |
| `slime` | `update_weights_interval` 1 .. 4 | `over_sampling_ratio` 1.0/1.5 |
| `skyrl` | `max_staleness_steps` 0 .. 8 | `pause_mode` in_place/abort |
| `openrlhf` | `async_queue_size` 1 .. 4 | `partial_rollout_enable` false/true |
| `dressage` | `keep_versions` 0 .. 4 | `max_partial_rollout_preempts` |
| `nemo_rl` | `max_staleness_versions` 0 .. 4 | `sampler` in_order/windowed/weight_fifo/ready_first |
| `trl` | `max_staleness` 0 .. 8 | `max_inflight_tasks` |
| `prime_rl` | `max_off_policy_steps` 1 .. 8 | `target_lag` 1/2/4 |

`verl_v1` runs its full two-dimensional grid, 24 points. The other eleven run their
primary knob only, 4 to 9 points each. About 100 runs total; the probe measured four
concurrent runs at 26 s wall, so the sweep is roughly 15 minutes, not a long job.

Two presets are expected to produce flat figures, and that is the measurement:
`skyrl`'s `max_staleness_steps` only drives admission quota and a violation counter and
never discards, and `openrlhf`'s `async_queue_size` bounds in-flight batch lumps rather
than per-sample age. A flat bar there is evidence the knob is not a staleness bound,
which is what the post claims about them.

## Gates

`sweep.sh` refuses rather than warns.

- **Consumption set.** Every point must consume the same trajectory count. Throughput
  is a rate, so a run that ended early scores arbitrarily well. Consumed *tokens* may
  still differ (21.4M vs 22.3M in the probe) because a tight threshold resamples
  different trajectories; that difference is annotated on the figure, not suppressed.
- **Engine identity.** `srl/_engine_cpp*.so` md5 is captured before and after. Another
  session rebuilding it in place kills running sims with a native signal that reads
  like an engine bug, and silently mixes binaries across points when it does not.
- **Disk.** `+log_exclude='instance,rollout_step'` on every run, and the instance
  stream deleted after. The shared filesystem's 2T quota is chronically full and a
  sweep that fills it produces truncated traces that look like failures.
- **Domain.** At cap 38912 a single sequence cannot be split (no CP), so it forms a
  38912-token microbatch against a `tokens` fit domain of [4096, 22528] — 1.7x out —
  and ~200 microbatches against a `pipeline_slots` domain of [3, 66]. `EXTRAPOLATE`
  lists both. The latency form is exactly linear in `pipeline_slots` (one 1F1B stage
  occupancy each), so that extrapolation is principled; the `tokens` exceedance is not,
  and every figure carries the note. Training time in these runs is indicative; the
  rollout numbers and the staleness composition are the claims.

## Out of scope

- No scheduling or policy change. The harness measures existing presets.
- No new cost-model fitting. If the `tokens` extrapolation proves to matter, profiling
  a wider THD cell is separate work.
- No claim that any preset is faster than another as a system. The figures compare one
  preset's knob against itself; cross-preset comparison is the post's job and needs the
  consumption-set argument made explicitly each time.
