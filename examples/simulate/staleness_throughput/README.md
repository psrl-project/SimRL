# Staleness-throughput sweeps

One figure per preset, two for slime: how that system's own staleness knob
redistributes the fleet's generation throughput across staleness levels and into
discarded work.

    bash sweep.sh psrl        # one preset
    bash run_all.sh           # every preset

Design: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/docs/superpowers/specs/2026-10-23-staleness-throughput-sweep-design.md`

## Reading the figure

A bar's height is the fleet's full generation throughput, in tokens per second.
Its blue segments are tokens that reached a training batch, keyed by the
staleness they carried. Its hatched segments are tokens that were generated and
then thrown away, keyed by why. The line across the blue is the effective
training throughput.

For a knob that bounds staleness by discarding (`roll`, `trl`, `verl_v1`'s
`max_off_policy_threshold`), the bar keeps its height while rollout stays
saturated: tightening the knob moves tokens across the line instead of
shortening the bar, since the discard happens after generation. Whether
tightening it further also shortens the bar depends on the policy's own
admission code, not on the fact that it discards: `roll`'s own admission
budget scales with its knob, so its tightest setting is also an admission
cutoff and the bar does shrink there. `trl`'s admission budget clamps to
the same floor at its two tightest settings (`max(max_staleness, 1) *
batch_size`), so nothing extra happens at its own tightest point. `verl_v1`'s
`max_off_policy_threshold` never enters admission at all, only the
post-generation discard. For an admission-type knob (most of the rest),
tightening it can shorten the bar at any point in its range, since the
knob gates how much rollout runs ahead in the first place, and span moves
with it. Read a figure's bar heights from its own policy's mechanism, not
from which category this paragraph puts it in.

The number above each bar is the highest staleness any consumed trajectory in
that run actually reached, not the knob setting itself (already on the x
axis): a knob that overshoots its own ceiling, or a discard-based bound that
holds it exactly, both show up here. Span is not drawn on the figure; every
bar in a figure finished the same number of training steps on the same number
of trajectories (the gate below enforces this), so span differences are a
secondary story told in `RESULTS.md`'s per-point table, not the figure's own
claim.

## The trap

With two rollout GPUs against six training GPUs on a long-tailed trace, a
preset whose admission is not opened runs **fully synchronously**: every token is
consumed at staleness 0, nothing is ever discarded, and no L3 threshold binds at
any value. A flat figure can mean the knob is not a staleness bound, or it can
mean admission never let rollout run ahead in the first place. Check the
staleness spread before concluding anything about a threshold.

Staleness also cannot exceed the number of versions in the run, so a short
`N_VERSIONS` reports the run length rather than the mechanism.

## slime's two figures

`slime_l1/` sweeps the prompt-group threshold at a fixed flush cadence.
`slime_flush/` sweeps the flush cadence at a fixed threshold. They are separate
because the worker gates two independent counts and because the flush cadence
turns out not to be a staleness bound.

Three real slime switches are deliberately not on an axis.
`--mask-offpolicy-in-partial-rollout` changes which tokens contribute to the
loss, not the generated-token staleness this figure measures.
`--partial-rollout` does not change whether an aborted flat sample keeps its
prefix, because the fully-async worker requeues partial flat samples
unconditionally. `--buffer-sort-by-staleness` orders generation resumption, not
the training batch, so it has no L3 counterpart here.

## Gates

`sweep.sh` refuses rather than warns.

- Every point must consume the same trajectory count. Throughput is a rate, so a
  point that ended early scores arbitrarily well. Consumed *tokens* may still
  differ, because a tighter bound resamples a different set of trajectories.
- The engine `.so` md5 must be unchanged across the sweep. Another session
  rebuilding it in place kills running simulations with a native signal that
  reads like an engine bug.
- `log_exclude=instance,rollout_step` on every run, and the instance stream
  deleted after. The shared filesystem quota is chronically full and a sweep that
  fills it produces truncated traces that look like failures.
- At cap 38912 the training cost is extrapolated on `tokens`, `activation_bytes`,
  `pipeline_slots`, and `sum_squared_tokens` (the worst exceedance, 8.27x its
  fitted bound, since it is the quadratic attention term). This caveat lives
  here, not on the figure itself. Treat the training component as indicative.
  The rollout and staleness numbers are the claims.

## Two mandatory pins

Not optional panels. Without them the sweep is illegal or inert.

- `openrlhf` pins `partial_rollout_enable=true`. The policy asserts for any
  `async_queue_size` above 1 otherwise. This also makes the pause mode
  `in_place`, which is the same switch in the real system.
- `nemo_rl` pins `sampler=windowed`. Under the default `in_order`, `horizon`
  reads `max_lookahead_versions` and the L2 cancel never runs, so sweeping
  `max_staleness_versions` would move nothing.

## Axis floors

Axes that start above 0 at 1 each do so because 0 is rejected, inverted, or
livelocks. Validated by running the endpoints. Do not change one without
re-running it.

| preset | knob | why not the floor below it |
|---|---|---|
| `verl_v1` | `num_warmup_batches` | asserts `> 0`, or nothing is ever admitted |
| `verl_v1` | `max_off_policy_threshold` | asserts `> 0`, `None` to disable |
| `slime` | `group_prefetch_factor` | asserts `>= 1`, and 0 would mean no group pool |
| `openrlhf` | `async_queue_size` | asserts `> 0` |
| `dressage` | `keep_versions` | 0 means the filter is **off**, the loose end rather than a tight one, so this axis starts above it on purpose, not because 1 is broken |
| `nemo_rl` | `max_staleness_versions` | `windowed`'s admission gate never opens, livelocks |

`slime`'s other axis, `flush_cache_interval`, starts at 0. A nonpositive
interval is the valid "never flush after the first publication" setting, not a
broken one. `server_concurrency` is pinned rather than swept, because the
worker's threshold is its product with the prefetch factor and only one of the
two needs to move.

`prime_rl_lag/grid.sh` (`target_lag`) and `nemo_rl_in_order/grid.sh`
(`max_lookahead_versions`) also start at 1, by analogy to the same
admission-gate shape as `verl_v1`'s warmup and `nemo_rl`'s own
`max_staleness_versions` floor (dispatch index starts equal to train
version, so a zero window blocks admission from the first tick) -- not
separately validated by running their own 0 point.

`verl.staleness_threshold` shares the 0..5 range but is a sample-overage ratio,
not a version count. Its axis says so, and it is not commensurable with the
version-count axes beside it. slime's two axes are neither: one bounds work and
one bounds KV lifetime.
