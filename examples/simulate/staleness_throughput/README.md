# Staleness-throughput sweeps

One figure per preset: how that system's own staleness knob redistributes the
fleet's generation throughput across staleness levels and into discarded work.

    bash sweep.sh psrl        # one preset
    bash run_all.sh           # all twelve

Design: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/docs/superpowers/specs/2026-10-23-staleness-throughput-sweep-design.md`

## Reading the figure

A bar's height is the fleet's full generation throughput, in tokens per second.
Its blue segments are tokens that reached a training batch, keyed by the
staleness they carried. Its hatched segments are tokens that were generated and
then thrown away, keyed by why. The line across the blue is the effective
training throughput.

A knob that tightens staleness does not shorten the bar. It moves tokens across
that line. That is the trade-off the figures exist to show.

The number above each bar is the simulated span. Every bar in a figure finished
the same number of training steps on the same number of trajectories, so span is
the honest cost axis and the gate below enforces the premise.

## The trap

With two rollout GPUs against six training GPUs on a long-tailed trace, a
preset whose admission is not opened runs **fully synchronously**: every token is
consumed at staleness 0, nothing is ever discarded, and no L3 threshold binds at
any value. A flat figure can mean the knob is not a staleness bound, or it can
mean admission never let rollout run ahead in the first place. Check the
staleness spread before concluding anything about a threshold.

Staleness also cannot exceed the number of versions in the run, so a short
`N_VERSIONS` reports the run length rather than the mechanism.

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
- At cap 38912 the training cost is extrapolated on `tokens` and
  `pipeline_slots`. Every figure carries the footnote. Treat the training
  component as indicative. The rollout and staleness numbers are the claims.

## Two mandatory pins

Not optional panels. Without them the sweep is illegal or inert.

- `openrlhf` pins `partial_rollout_enable=true`. The policy asserts for any
  `async_queue_size` above 1 otherwise. This also makes the pause mode
  `in_place`, which is the same switch in the real system.
- `nemo_rl` pins `sampler=windowed`. Under the default `in_order`, `horizon`
  reads `max_lookahead_versions` and the L2 cancel never runs, so sweeping
  `max_staleness_versions` would move nothing.

## Axis floors

Five axes start at 1, each because 0 is rejected or inverted. Validated by
running the endpoints. Do not change one without re-running it.

| preset | knob | why not 0 |
|---|---|---|
| `verl_v1` | `num_warmup_batches` | asserts `> 0`, or nothing is ever admitted |
| `verl_v1` | `max_off_policy_threshold` | asserts `> 0`, `None` to disable |
| `slime` | `update_weights_interval` | a loop interval, fails at 0 |
| `openrlhf` | `async_queue_size` | asserts `> 0` |
| `dressage` | `keep_versions` | 0 means the filter is **off**, the loose end |

`verl.staleness_threshold` shares the 0..5 range but is a sample-overage ratio,
not a version count. Its axis says so, and it is not commensurable with the
eleven version counts beside it.
