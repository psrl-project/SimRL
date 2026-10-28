# Staleness-throughput sweep results

131 points across 16 presets, one figure per preset (two for `slime`) under
`<preset>/figures/staleness_throughput.png`.
Design: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/docs/superpowers/specs/2026-10-23-staleness-throughput-sweep-design.md`.

This regime moves `BATCH_SIZE` 8 -> 32 and the trace/cap from
`qwen3_8b_dapo_38912_tp1` (cap 38912) to `qwen3_8b_dapo_tp1` (cap 10240),
matching a separate real-run comparison at the same batch/cap. Moving
`BATCH_SIZE` surfaced two classes of grid pin that were implicitly
calibrated against the old value of 8 rather than expressed as a formula:

- `slime_l1`/`slime_flush` pinned `policy.server_concurrency=32` as a bare
  constant. `slime`'s own admission threshold is
  `max(1, server_concurrency * N_INSTANCES // GROUP_SIZE) * group_prefetch_factor`
  groups, so 32 cleared `BATCH_SIZE=8` with room to spare but fails
  `slime.validate()`'s `threshold >= batch_size` assertion outright at
  `BATCH_SIZE=32` for most of `slime_l1`'s own axis and all of
  `slime_flush`'s. Both grids now pin
  `server_concurrency=$((BATCH_SIZE * GROUP_SIZE / N_INSTANCES))`, the same
  formula style `nemo_rl_in_order/grid.sh` already used for its own
  batch-dependent pin, which keeps `per_engine == BATCH_SIZE` at the
  regime's own `N_INSTANCES`/`GROUP_SIZE` regardless of `BATCH_SIZE`.
- `nemo_rl`/`nemo_rl_in_order` left `policy.max_inflight_prompts` at its
  class default (32). `NeMoRLPolicyConfig.ingress_window_groups` shows the
  effective admission window is
  `min(max_inflight_prompts, max_buffered_rollouts, gate_window * BATCH_SIZE)`,
  so at `BATCH_SIZE=32` the unscaled 32 sits at exactly 1x batch size and
  became the binding term at every point on both axes, silently flattening
  both figures to the regime's synchronous floor regardless of the swept
  knob -- no assertion fired, nothing in `sweep.sh`'s own gates caught it,
  the figures were simply uninformative. Both grids now also pin
  `max_inflight_prompts=$((BATCH_SIZE * 6))`, the same margin
  `nemo_rl_in_order/grid.sh` already used for `max_buffered_rollouts`.

The other twelve presets' admission math is expressed directly against
`st.batch_size`/`st.group_size` in the policy's own code (`roll`, `trl`,
`psrl`, `verl`, `verl_v1`, `dressage`, `openrlhf`) or defaults its one
capacity knob to `None` (unbounded, `areal`, `skyrl`, `prime_rl`), so none of
them needed a grid change. `prime_rl`'s own flat figure is a different,
already-documented property (its paired `target_lag` admission knob, not
the swept `max_off_policy_steps`) and is unrelated to this class of bug; see
its own section below.

## Regime

Pinned identically for every preset, from `common.sh`, except the four
exports below `sweep.sh`/`run_all.sh` were called with:

```
backend      cpp (SIMRL_BACKEND), pinned rather than auto so every point comes from one engine
trace        outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl
             1024 groups x 8, cap 10240, mean 7804 tok, median 8690, p95 10240,
             truncated_fraction 0.408
model        Qwen3-8B
rollout      2 instances x TP1          (2 GPUs)
training     hydraulis TP2 / PP3 / DP1  (6 GPUs)
batch_size   32 groups      group_size 8      n_versions 10
rollout cost composed, kernel table outputs/profile/qwen3_8b/kernel_tp1/kernel
train cost   outputs/cost/qwen3_8b_tp2_pp3_thd/training_hydraulis.json
```

41% of this trace's groups are truncated at the 10240 cap (`p95` already sits
on the cap), a much heavier truncation rate than the previous regime's 38912
cap (0.45%). At cap 10240 the longest microbatch, 10240 tokens, sits inside
the training cost model's fitted domain (`[4096, 22528]` per this run's own
log output), so none of the previous regime's extrapolation caveat on
`tokens`/`activation_bytes`/`pipeline_slots`/`sum_squared_tokens` applies
here -- the training-span numbers in this file are not indicative-only this
time, though the rollout numbers and staleness composition remain the
figures' real claim.

`slime_l1` and `slime_flush` additionally override `N_VERSIONS=32` (their
own grids, unrelated to the `BATCH_SIZE` fix above): their knob range tops
out at `group_prefetch_factor`/`flush_cache_interval=8`, and 10 versions
would report the run's own length rather than the mechanism.

## Sweep execution

`BATCH_SIZE=32 N_VERSIONS=10 MAX_OUTPUT_LENGTH=10240
TRACE=outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl REUSE=false bash
examples/simulate/staleness_throughput/run_all.sh` ran all 16 presets.
All 131 points finished and fed their preset's figure; none livelocked, none
were dropped from `PRIMARY_VALUES`, and none left `sweep.sh`'s
consumption-set gate unsatisfied. Every point in every preset consumed
exactly 2560 trajectories (`BATCH_SIZE * GROUP_SIZE * N_VERSIONS = 32*8*10`),
except `slime_l1`/`slime_flush`, whose own `N_VERSIONS=32` override gives
8192 (`32*8*32`) -- both counts uniform across each preset's own points,
satisfying the gate. The engine `.so` md5 was unchanged across the whole
sweep.

## Per-point table

`span` is seconds, `maxS` is the highest `staleness_true` any consumed
trajectory in that run reached, `waste%` is discarded tokens over total
generated tokens, and the spread is consumed tokens by staleness level.
Two-knob presets (`verl_v1`, `verl_v1_wait`, `prime_rl_lag`) name points
`s<secondary>_p<primary>`, one `s` value per secondary-knob panel.

```
preset            point      span  maxS  waste%  spread
psrl                 p0      8992     0     0.0  S0:20.2M
psrl                 p1      6801     1     0.0  S0:1.9M S1:18.3M
psrl                 p2      7104     2     0.0  S0:1.9M S1:1.9M S2:16.4M
psrl                 p3      7257     3     0.0  S0:1.9M S1:1.9M S2:4.0M S3:12.4M
psrl                 p4      7247     4     0.0  S0:1.9M S1:1.9M S2:4.2M S3:4.7M S4:7.4M
psrl                 p5      7282     5     0.0  S0:1.9M S1:1.9M S2:3.3M S3:5.0M S4:5.4M S5:2.8M
verl_v1          s1_p1..5 11791-33885  0  50.7-83.2  S0 only throughout, waste grows with threshold
verl_v1          s2_p1..5  6739-16769  1   4.5-64.0  S0,S1
verl_v1          s3_p1..5  6650-12299  2   0.0-49.6  S0..S2
verl_v1          s4_p1..5  6650-9600   3   0.0-31.4  S0..S3
verl_v1          s5_p1..5  6650-7796   4   0.0-13.1  S0..S4
verl_v1_wait     s1_p1..5  9559-11884  1-5  0.0      S0..S5, grows with warmup not threshold
verl_v1_wait     s2_p1..5  6801-9253   1-5  0.0      S0..S5
verl_v1_wait     s3_p1..5  6650-8757   2-5  0.0      S0..S5
verl_v1_wait     s4_p1..5  6650-8387   2-5  0.0      S0..S5
verl_v1_wait     s5_p1..5  6650-7839   2-5  0.0      S0..S5
verl                 p0      8992     0     0.0  S0:20.2M
verl                 p1      6650     2     0.0  S0:2.4M S1:16.3M S2:1.4M
verl                 p2      6963     3     0.0  S0:1.9M S1:2.8M S2:13.3M S3:2.0M
verl                 p3      7109     4     0.0  S0:1.9M S1:2.2M S2:4.4M S3:10.0M S4:1.5M
verl                 p4      7109     5     0.0  S0:1.9M S1:2.0M S2:4.2M S3:4.8M S4:6.6M S5:0.6M
verl                 p5      7124     6     0.0  S0:1.9M S1:2.0M S2:3.2M S3:4.8M S4:5.3M S5:2.6M S6:0.3M
areal                p0      8992     0     0.0  S0:20.2M
areal                p1      6817     2     0.0  S0:2.3M S1:16.4M S2:1.2M
areal                p2      7103     3     0.0  S0:1.9M S1:2.9M S2:13.0M S3:2.1M
areal                p3      7210     4     0.0  S0:1.9M S1:2.1M S2:4.5M S3:10.0M S4:1.5M
areal                p4      7217     5     0.0  S0:1.9M S1:2.0M S2:4.0M S3:4.6M S4:6.6M S5:1.0M
areal                p5      7217     6     0.0  S0:1.9M S1:2.1M S2:3.7M S3:4.3M S4:5.5M S5:2.2M S6:0.3M
roll                 p0      8992     0     0.0  S0:20.2M
roll                 p1      6778     1     5.0  S0:2.6M S1:17.3M
roll                 p2      7158     2     6.4  S0:1.9M S1:3.7M S2:14.1M
roll                 p3      7155     3     5.0  S0:1.9M S1:2.2M S2:5.3M S3:10.4M
roll                 p4      7174     4     3.8  S0:1.9M S1:2.0M S2:4.5M S3:5.1M S4:6.3M
roll                 p5      7156     5     3.1  S0:1.9M S1:2.0M S2:3.4M S3:4.9M S4:5.4M S5:2.3M
slime_l1             p1     20385     2     0.0  S0:2.2M S1:29.4M S2:32.1M
slime_l1             p2     21398     3     0.0  S0:1.9M S1:5.5M S2:36.2M S3:20.2M
slime_l1             p4     21663     6     0.0  S0:1.9M S1:2.1M S2:5.1M S3:16.7M S4:30.2M S5:7.8M S6:0.0M
slime_l1             p8     21768     9     0.0  S0:1.9M S1:1.9M S2:2.3M S3:4.4M S4:8.9M S5:11.6M S6:12.5M S7:12.2M S8:6.4M S9:1.5M
slime_flush          p0     19715     3     0.0  S0:1.9M S1:4.7M S2:44.0M S3:13.2M
slime_flush          p1     21398     3     0.0  S0:1.9M S1:5.5M S2:36.2M S3:20.2M
slime_flush          p2     20563     3     0.0  S0:1.9M S1:4.7M S2:36.9M S3:20.3M
slime_flush          p4     20084     3     0.0  S0:1.9M S1:4.4M S2:40.8M S3:16.7M
slime_flush          p8     19864     3     0.0  S0:1.9M S1:4.7M S2:42.5M S3:14.6M
skyrl                p0      8992     0     0.0  S0:20.2M
skyrl                p1      6297     2     0.0  S0:2.4M S1:16.2M S2:1.4M
skyrl                p2      6519     3     0.0  S0:1.9M S1:3.0M S2:13.0M S3:2.1M
skyrl                p3      6609     4     0.0  S0:1.9M S1:2.1M S2:5.0M S3:9.4M S4:1.6M
skyrl                p4      6598     5     0.0  S0:1.9M S1:2.0M S2:4.0M S3:4.5M S4:6.5M S5:1.0M
skyrl                p5      6599     6     0.0  S0:1.9M S1:2.1M S2:3.7M S3:4.0M S4:5.1M S5:2.8M S6:0.3M
openrlhf             p1      8992     0     0.0  S0:20.2M
openrlhf             p2      6297     2     0.0  S0:2.4M S1:16.2M S2:1.4M
openrlhf             p3      6519     3     0.0  S0:1.9M S1:3.0M S2:13.0M S3:2.1M
openrlhf             p4      6609     4     0.0  S0:1.9M S1:2.1M S2:5.0M S3:9.4M S4:1.6M
openrlhf             p5      6598     5     0.0  S0:1.9M S1:2.0M S2:4.0M S3:4.5M S4:6.5M S5:1.0M
dressage             p1      8751     1    30.6  S0:3.0M S1:17.1M
dressage             p2      6208     3     0.0  S0:2.1M S1:9.4M S2:8.5M S3:0.1M
dressage             p3      6208     3     0.0  S0:2.1M S1:9.4M S2:8.5M S3:0.1M
dressage             p4      6208     3     0.0  S0:2.1M S1:9.4M S2:8.5M S3:0.1M
dressage             p5      6208     3     0.0  S0:2.1M S1:9.4M S2:8.5M S3:0.1M
nemo_rl              p1      8992     0     0.0  S0:20.2M
nemo_rl              p2      6650     2     0.0  S0:2.4M S1:16.3M S2:1.4M
nemo_rl              p3      6963     3     0.0  S0:1.9M S1:2.8M S2:13.3M S3:2.0M
nemo_rl              p4      7109     4     0.0  S0:1.9M S1:2.2M S2:4.4M S3:10.0M S4:1.5M
nemo_rl              p5      7109     5     0.0  S0:1.9M S1:2.0M S2:4.2M S3:4.8M S4:6.6M S5:0.6M
nemo_rl_in_order     p1      8992     0     0.0  S0:20.2M
nemo_rl_in_order     p2      6748     1     0.0  S0:1.9M S1:18.3M
nemo_rl_in_order     p3      7095     2     0.0  S0:1.9M S1:2.0M S2:16.2M
nemo_rl_in_order     p4      7259     3     0.0  S0:1.9M S1:2.0M S2:5.1M S3:11.2M
nemo_rl_in_order     p5      7237     4     0.0  S0:1.9M S1:2.0M S2:2.3M S3:6.4M S4:7.5M
trl                  p0     11140     0    50.9  S0:18.5M
trl                  p1      6481     1     5.3  S0:2.8M S1:17.0M
trl                  p2      6658     2     4.6  S0:1.9M S1:4.1M S2:13.7M
trl                  p3      6714     3     3.5  S0:1.9M S1:2.0M S2:6.5M S3:9.3M
trl                  p4      6683     4     2.4  S0:1.9M S1:2.1M S2:4.1M S3:6.4M S4:5.4M
trl                  p5      6702     5     2.7  S0:1.9M S1:2.0M S2:4.0M S3:4.5M S4:6.3M S5:1.2M
prime_rl             p0      9101     0     0.0  S0:20.2M
prime_rl             p1      9101     0     0.0  S0:20.2M
prime_rl             p2      9101     0     0.0  S0:20.2M
prime_rl             p3      9101     0     0.0  S0:20.2M
prime_rl             p4      9101     0     0.0  S0:20.2M
prime_rl             p5      9101     0     0.0  S0:20.2M
prime_rl_lag     s1_p1        9101     0     0.0  S0:20.2M
prime_rl_lag     s1_p2        6794     1     1.8  S0:3.1M S1:16.9M
prime_rl_lag     s1_p3        7062     1    11.0  S0:3.1M S1:16.4M
prime_rl_lag     s1_p4        6947     1    12.0  S0:4.4M S1:15.0M
prime_rl_lag     s1_p5        7028     1    13.2  S0:4.9M S1:14.5M
prime_rl_lag     s3_p1        9101     0     0.0  S0:20.2M
prime_rl_lag     s3_p2        6443     2     0.0  S0:2.5M S1:16.0M S2:1.5M
prime_rl_lag     s3_p3        6582     3     0.0  S0:1.9M S1:2.9M S2:13.2M S3:2.1M
prime_rl_lag     s3_p4        6709     3     4.1  S0:1.9M S1:2.1M S2:6.1M S3:9.7M
prime_rl_lag     s3_p5        7292     3    12.1  S0:1.9M S1:3.9M S2:5.3M S3:8.5M
```

A number that recurs across many presets' tightest setting is worth naming
once: span 8992, throughput 2243 tok/s, everything at `S0`, 0% waste. Two
rollout GPUs against six training GPUs on this trace leaves rollout far
enough behind training that the run is fully synchronous, independent of
which preset's own knob happens to be the one pinned tight enough to cause
it: `psrl`, `verl`, `areal`, `roll`, `skyrl` all hit it at knob=0, and
`openrlhf`/`nemo_rl`/`nemo_rl_in_order` hit the identical number at their own
floor (knob=1, since their axes start there). `prime_rl` sits close but not
identical (9101 tok/s=2216, every point, see its own section) and `trl`'s
own knob=0 is a different, higher number (11140) because `trl` discards even
at its tightest setting rather than simply not admitting ahead.

## Per-preset findings

**psrl.** Free bound, `maxS` tracks the knob exactly 0 through 5, 0.0% waste
throughout. Span drops from the 8992s floor to 6801s at `staleness=1` and
settles to 7100-7280s from 2 on.

**verl_v1.** The two-knob grid (`num_warmup_batches` x
`max_off_policy_threshold`, 25 points) remains the richest figure. At
`num_warmup_batches=1`, `maxS` stays 0 at every threshold but span grows
from 11791s to 33885s and waste from 50.7% to 83.2% as the threshold
loosens: admission's warmup credit does not block rollout running ahead
here, it is the discard/replace cycle itself inflating total generated work
while keeping the consumed set clean. From `num_warmup_batches=2` on,
tightening `max_off_policy_threshold` grows a visible `l3_stale` segment
(up to 64.0% waste at `threshold=2, warmup=5`) while `maxS` stays capped at
or below the threshold (no exceedance anywhere in this preset: the drop
strategy holds its own bound exactly). Both axes do real work.

**verl_v1_wait.** `waste%` is 0.0 at every one of its 25 points -- `wait`
never discards, it only stalls sampling while something still-generating
has crossed the window, and an already-finished straggler is never
re-examined. `maxS` tracks `num_warmup_batches`, not
`max_off_policy_threshold`, confirming the real system's documented absence
of a bound for this strategy: the figure's own cap marker fires (red cross)
at 15 of its 25 points, everywhere `threshold <= warmup`, exactly the set
where the wait strategy's own absence of enforcement is visible.

**verl (legacy).** `staleness_threshold` is a sample-overage ratio, not a
version count, and its axis label says so, so there is no cap rule for it
and no exceedance to check. `waste%` is 0.0 at every point. The composition
broadens from `S0`-only at 0 to a spread reaching `S6` at 5, and span drops
from 8992s to about 6650-7124s once the knob is above 0.

**areal.** Confirmed non-bound by the same test `skyrl` fails below: `maxS`
overshoots the knob by exactly +1 at every point once admission opens (knob
1 through 5 all read `knob+1`), with 0.0% waste throughout, so nothing ever
pulls it back. The figure's cap marker fires at all 5 nonzero points. Span
settles to about 6817-7217s once the knob is above 0.

**roll.** Real discard cost: `async_generation_ratio`'s ceiling acts as
both the admission window and an L2 GC, so unlike `psrl`'s free bound,
`roll` pays for its cap -- waste falls from 5.0% at knob=1 to 3.1% at
knob=5 as the window loosens. `maxS` tracks the knob exactly throughout (0
through 5, no overshoot, no cap-marker fire).

**slime.** `slime_l1/` sweeps `group_prefetch_factor` 1, 2, 4, 8 at
`server_concurrency=128` (this run's `BATCH_SIZE`-scaled pin, see "Sweep
execution"), a group threshold of 32, 64, 128, 256 against a batch of 32.
`maxS` runs 2, 3, 6, 9, monotonically, and the composition broadens hugely
by factor=8 (`S0` through `S9`). Throughput falls steadily (3127 to 2926
tok/s) as the KV pool gets more oversubscribed; waste stays 0.0 throughout,
since `slime` never discards. `slime_flush/` sweeps `flush_cache_interval`
0, 1, 2, 4, 8 at a fixed threshold of 64 groups: `maxS` is flat at 3
throughout and the spread barely moves, while span moves 19715s (never
flush) to 21398s (flush every publication, the worst of the five) back down
to 19864s at flush=8 -- confirming the knob changes re-prefill cost, not
staleness, exactly as its own grid comment claims. Neither axis has a cap
rule (`slime` bounds work and KV lifetime, not a trajectory's age), so no
exceedance check applies to either figure.

**skyrl.** Confirmed non-bound: `maxS` overshoots the knob by exactly +1 at
every nonzero point (1->2, 2->3, 3->4, 4->5, 5->6), with 0.0% waste
throughout -- the cap marker fires at all 5. Span moves from the 8992s
floor to 6297-6609s once the knob opens, a real admission effect, but the
composition is never pulled back once admitted; `max_staleness_steps` is an
admission quota and a violation counter, never a filter, exactly as the
grid's own description states.

**openrlhf.** `maxS` tracks the knob with no overshoot in this regime
(1->0, 2->2, 3->3, 4->4, 5->5), unlike `skyrl`/`areal` above -- the cap
marker does not fire anywhere here. This is a property of this particular
batch/trace combination, not evidence `async_queue_size` became an
age-based bound: the knob still counts whole-batch lumps rather than
per-sample age (per its own docstring), and the exact alignment at
`BATCH_SIZE=32` should not be read past this regime. Waste is 0.0%
throughout; span moves from the 8992s floor to 6297-6609s once the
mandatory `partial_rollout_enable=true` pin's queue opens.

**dressage.** `keep_versions=1` is its own point (span 8751s, `maxS=1`,
30.6% waste); 2 through 5 are byte-identical (span 6208s, `maxS=3`, 0.0%
waste, same spread down to the tenth of a megabyte) -- a real,
non-degenerate `S0`-`S3` composition, so admission did open and this is not
a flat-admission trap, just a plateau past `keep_versions=2` in this regime.
The cap marker fires once, at `keep_versions=2` (achieved 3 against a cap of
2); `keep_versions=1`, `3`, `4`, `5` all stay at or under their own cap.

**nemo_rl.** `max_staleness_versions` holds its own bound exactly once
admission has the headroom this run's grid fix gives it (see "Sweep
execution"): knob=1 sits on the 8992s synchronous floor (`maxS=0`, under its
own cap of 1), and knob 2 through 5 all read `maxS == knob`, 0.0% waste
throughout, no cap-marker fire anywhere. Span drops to 6650s at knob=2 and
settles at 7109s for both knob=4 and knob=5 (same wall time, a still-shifting
composition underneath).

**nemo_rl_in_order.** Same fix, and now fully informative across the whole
swept range rather than plateauing: `max_lookahead_versions` 1 through 5
give `maxS` 0, 1, 2, 3, 4 (the knob's own `-1` offset, since dispatch
reserves one step ahead of its own admitted cohort), 0.0% waste, no
overshoot. Span runs 6748-7259s across knob 2-5, after the 8992s floor at
knob=1.

**trl.** The most direct, and the only strictly costly, bound in the set:
`maxS` tracks the knob exactly, 0 through 5, no overshoot anywhere, but
waste is never 0 -- 50.9% at knob=0 (the strict `S0` bound's full cost
against this trace, not a free pass), falling to 2.4-5.3% from knob=1 on,
non-monotonically (5.3, 4.6, 3.5, 2.4, 2.7). Span: 11140s at knob=0 (not the
shared 8992s floor, because `trl` discards even at its tightest setting
rather than simply failing to admit ahead), dropping to 6481s at knob=1 and
settling around 6660-6714s from 2 on.

**prime_rl.** Genuinely flat: every one of its 6 points reads span 9101s,
`maxS=0`, 0.0% waste, tok/s 2216, all 20.2M tokens at `S0`. This is the
regime never letting staleness develop, for a specific, already-documented
reason: `admit_quota` gates on `target_lag` (`dispatch_index >= train_version
+ target_lag`), not on `max_off_policy_steps`, the knob this grid actually
sweeps. `target_lag` defaults to 1 and stays there in this grid, so dispatch
can never build enough of a lead for anything to become stale, at any of the
six `max_off_policy_steps` values tested. Not the same bug class as the
`slime`/`nemo_rl` grid fixes above: nothing here is an unscaled absolute
constant, `target_lag=1` is a deliberate default this preset's own grid
leaves alone on purpose, by design (see `prime_rl_lag` below, which exists
specifically to open this gap).

**prime_rl_lag.** `target_lag` 1 through 5, paneled against
`max_off_policy_steps` at 1 and 3, opens exactly the admission gap
`prime_rl`'s own flat figure identifies. At `max_off_policy_steps=1`, `maxS`
caps at 1 once `target_lag>=2` (waste climbing 1.8% to 13.2% as the window
loosens further, now with `l2_cancel` alongside `l3_stale`); at
`max_off_policy_steps=3`, `maxS` reaches and holds at 3 once `target_lag>=3`
(waste 0.0% through `target_lag=3`, then 4.1% and 12.1%). No exceedance in
either panel: the bound holds once admission actually gives it something to
catch, confirming `max_off_policy_steps` genuinely is a staleness bound,
exactly what this sweep arm was built to show.

## The cap marker, checked against this run

`srl/visualizer/staleness_sweep.py` draws a point whose achieved `maxS`
exceeds its own designed ceiling (per `_CAP_RULES`) as a red `x` on the
staleness line, distinct from the orange circle used for in-bound points --
not merely a printed number next to an unmarked line. Across this run's 16
presets it fires on real data: all 15 of `verl_v1_wait`'s
`threshold <= warmup` points (by design -- the `wait` strategy's documented
absence of a bound), all 5 of `areal`'s and `skyrl`'s nonzero points (both
confirmed non-bounds), and 1 of `dressage`'s 5 points
(`keep_versions=2`). `psrl`, `roll`, `trl`, `verl` (no cap rule, a ratio
axis), `openrlhf`, `nemo_rl`, `nemo_rl_in_order`, `prime_rl`,
`prime_rl_lag`, and `verl_v1`'s own drop-strategy grid show no exceedance
anywhere; `slime`'s two axes carry no cap rule at all.

No preset is compared against another as a system anywhere above: every
span, waste share and composition described is that preset's own knob
compared against itself.
