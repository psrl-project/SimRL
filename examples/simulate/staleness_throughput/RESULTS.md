# Staleness-throughput sweep results

88 points across 12 presets, one figure per preset under `<preset>/figures/staleness_throughput.png`.
Design: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/docs/superpowers/specs/2026-10-23-staleness-throughput-sweep-design.md`.

## Regime

Pinned identically for every preset, from `common.sh`:

```
backend      cpp (SIMRL_BACKEND), pinned rather than auto so every point comes from one engine
trace        outputs/workload/qwen3_8b_dapo_38912_tp1/joint.jsonl
             1024 groups x 8, cap 38912, mean 10660 tok, median 8673, p95 25251,
             truncated_fraction 0.0045
model        Qwen3-8B
rollout      2 instances x TP1          (2 GPUs)
training     hydraulis TP2 / PP3 / DP1  (6 GPUs)
batch_size   32 groups      group_size 8      n_versions 12
rollout cost composed, kernel table outputs/profile/qwen3_8b/kernel_tp1/kernel
train cost   outputs/cost/qwen3_8b_tp2_pp3_thd/training_hydraulis.json
```

At cap 38912 a single sequence cannot be split, so the longest microbatch, the
microbatch count, the activation footprint, and the quadratic attention term
all sit outside the training cost model's fitted domain (`tokens`,
`pipeline_slots`, `activation_bytes`, `sum_squared_tokens`, the last
exceeding its fitted bound by 8.27x, the worst of the four). Every figure
carries that footnote. Training span in these runs is indicative. The
rollout numbers and the staleness composition are the claims the figures
make.

## Sweep execution

`MAX_PARALLEL=16 bash examples/simulate/staleness_throughput/run_all.sh` ran all 88
points. 87 finished and fed their preset's figure. One point, `nemo_rl` at
`max_staleness_versions=0`, livelocked: 0 versions simulated, 0 requests consumed,
the engine's own "Simulation livelocked" warning. Reading `srl/policy/nemo_rl.py`
explains why. The `windowed` sampler's admission gate is

```python
if self._dispatch_index >= st.train_version + self._gate_window():
    return 0
```

and `_gate_window()` returns `staleness_window` for `windowed`. At
`max_staleness_versions=0` this is `0 >= 0 + 0`, true from the very first cohort,
so admission never opens at all, for the entire run. This is a sixth instance of
the axis-floor pattern the README's "Axis floors" table already documents for
five knobs across four other presets (`verl_v1` twice, `slime`, `openrlhf`,
`dressage`), just reached through a livelock rather than an assertion, and only
visible once `nemo_rl` is swept against `sampler=windowed`, which is itself a
pin added for this preset.
`nemo_rl`'s figure is drawn from its remaining five points, `max_staleness_versions`
1 through 5, built from the same `request.jsonl` streams `sweep.sh` already wrote
for those points, through the standalone `srl-staleness-plot` CLI rather than by
editing the harness. The consumption-set gate still holds: all five points consumed
3072 trajectories, same as every other preset's points. `nemo_rl`'s grid, like the
other five floor exceptions, has had its 0 point dropped from `PRIMARY_VALUES`
in the final fix wave of this task, once the harness's own `load_manifest` was
also hardened to refuse a livelocked point cleanly rather than crash on it.

Every other preset's 6 (or 5) points consumed exactly 3072 trajectories each,
satisfying `sweep.sh`'s consumption-set gate. The engine `.so` md5 was unchanged
across the whole sweep. Wall time for the 88 points was a few minutes.

## Per-point table

`span` is seconds, `maxS` is the highest `staleness_true` any consumed trajectory
in that run reached, `waste%` is discarded tokens over total generated tokens, and
the spread is consumed tokens by staleness level. `verl_v1`'s points are named
`s<secondary>_p<primary>`, one `s` value per `max_off_policy_threshold` panel.

```
preset      point     span  maxS  waste%  spread
areal          p0    21656     0     0.0  S0:32.5M
areal          p1    15733     2     0.0  S0:4.1M S1:20.5M S2:8.0M
areal          p2    15842     4     0.0  S0:2.0M S1:6.0M S2:17.3M S3:6.8M S4:0.4M
areal          p3    15889     5     0.0  S0:1.9M S1:3.4M S2:11.5M S3:11.4M S4:4.1M S5:0.2M
areal          p4    15888     5     0.0  S0:1.9M S1:3.0M S2:7.3M S3:10.2M S4:8.4M S5:1.8M
areal          p5    15874     6     0.0  S0:1.9M S1:2.7M S2:5.2M S3:7.8M S4:8.2M S5:5.2M S6:1.6M
dressage       p1    14854     3     0.0  S0:2.4M S1:15.7M S2:13.6M S3:0.8M
dressage       p2    14854     3     0.0  S0:2.4M S1:15.7M S2:13.6M S3:0.8M
dressage       p3    14854     3     0.0  S0:2.4M S1:15.7M S2:13.6M S3:0.8M
dressage       p4    14854     3     0.0  S0:2.4M S1:15.7M S2:13.6M S3:0.8M
dressage       p5    14854     3     0.0  S0:2.4M S1:15.7M S2:13.6M S3:0.8M
nemo_rl        p0  LIVELOCKED: 0 versions, 0 requests consumed (see above)
nemo_rl        p1    21762     0     0.0  S0:32.5M
nemo_rl        p2    15574     2     0.0  S0:4.4M S1:19.7M S2:8.4M
nemo_rl        p3    15559     3     0.0  S0:2.4M S1:15.5M S2:14.1M S3:0.6M
nemo_rl        p4    15559     3     0.0  S0:2.4M S1:15.5M S2:14.1M S3:0.6M
nemo_rl        p5    15559     3     0.0  S0:2.4M S1:15.5M S2:14.1M S3:0.6M
openrlhf       p1    16620     1     0.0  S0:2.4M S1:30.1M
openrlhf       p2    15114     4     0.0  S0:2.1M S1:6.6M S2:17.2M S3:6.5M S4:0.1M
openrlhf       p3    15080     5     0.0  S0:1.9M S1:3.6M S2:14.2M S3:10.3M S4:2.5M S5:0.1M
openrlhf       p4    15164     5     0.0  S0:1.9M S1:3.1M S2:9.4M S3:12.0M S4:5.4M S5:0.8M
openrlhf       p5    15176     6     0.0  S0:1.9M S1:3.1M S2:6.6M S3:7.4M S4:9.4M S5:3.6M S6:0.5M
prime_rl       p0    21762     0     0.0  S0:32.5M
prime_rl       p1    21762     0     0.0  S0:32.5M
prime_rl       p2    21762     0     0.0  S0:32.5M
prime_rl       p3    21762     0     0.0  S0:32.5M
prime_rl       p4    21762     0     0.0  S0:32.5M
prime_rl       p5    21762     0     0.0  S0:32.5M
psrl           p0    21656     0     0.0  S0:32.5M
psrl           p1    16179     1     0.0  S0:1.9M S1:30.6M
psrl           p2    15811     2     0.0  S0:1.9M S1:3.8M S2:26.8M
psrl           p3    15872     3     0.0  S0:1.9M S1:2.6M S2:8.8M S3:19.3M
psrl           p4    15923     4     0.0  S0:1.9M S1:2.5M S2:7.1M S3:11.8M S4:9.2M
psrl           p5    15941     5     0.0  S0:1.9M S1:2.5M S2:5.9M S3:7.4M S4:9.6M S5:5.2M
roll           p0    21762     0     0.0  S0:32.5M
roll           p1    17123     1    18.0  S0:5.6M S1:23.9M
roll           p2    16909     2    14.2  S0:2.3M S1:8.1M S2:19.8M
roll           p3    16557     3    10.0  S0:2.0M S1:3.6M S2:12.7M S3:12.6M
roll           p4    16483     4     9.8  S0:2.0M S1:3.0M S2:8.7M S3:11.1M S4:5.9M
roll           p5    16461     5     8.6  S0:2.0M S1:3.0M S2:7.5M S3:8.2M S4:5.9M S5:4.5M
skyrl          p0    21762     0     0.0  S0:32.5M
skyrl          p1    15014     2     0.0  S0:4.1M S1:20.3M S2:8.1M
skyrl          p2    15228     4     0.0  S0:1.9M S1:5.2M S2:18.9M S3:6.3M S4:0.3M
skyrl          p3    15282     5     0.0  S0:1.9M S1:2.9M S2:8.5M S3:14.2M S4:4.8M S5:0.2M
skyrl          p4    15118     6     0.0  S0:1.9M S1:3.4M S2:9.3M S3:8.6M S4:7.6M S5:1.6M S6:0.1M
skyrl          p5    15134     6     0.0  S0:1.9M S1:3.1M S2:6.4M S3:7.2M S4:7.4M S5:5.1M S6:1.4M
slime          p1    17428     1     0.0  S0:2.4M S1:30.1M
slime          p2    17050     2     0.0  S0:2.4M S1:16.6M S2:13.5M
slime          p3    16927     3     0.0  S0:2.4M S1:10.7M S2:11.0M S3:8.5M
slime          p4    16825     4     0.0  S0:2.4M S1:9.1M S2:8.5M S3:7.6M S4:5.0M
slime          p5    16836     5     0.0  S0:2.4M S1:7.8M S2:5.7M S3:4.8M S4:5.8M S5:6.0M
trl            p0    21220     1    12.8  S0:12.6M S1:25.0M
trl            p1    16620     1     0.0  S0:2.4M S1:30.1M
trl            p2    15806     3     9.1  S0:2.4M S1:8.3M S2:18.7M S3:1.6M
trl            p3    15976     4     8.7  S0:1.9M S1:4.1M S2:16.4M S3:8.4M S4:0.2M
trl            p4    15290     5     5.0  S0:2.0M S1:3.0M S2:9.8M S3:14.2M S4:2.7M S5:0.0M
trl            p5    15526     6     5.7  S0:2.1M S1:3.2M S2:7.2M S3:8.0M S4:8.3M S5:2.8M S6:0.1M
verl           p0    21656     0     0.0  S0:32.5M
verl           p1    15669     3     0.0  S0:4.4M S1:19.9M S2:7.8M S3:0.4M
verl           p2    15797     4     0.0  S0:2.0M S1:5.8M S2:17.3M S3:7.3M S4:0.2M
verl           p3    15784     5     0.0  S0:1.9M S1:3.1M S2:10.2M S3:12.9M S4:4.3M S5:0.2M
verl           p4    15810     6     0.0  S0:1.9M S1:2.9M S2:6.2M S3:9.9M S4:9.6M S5:1.9M S6:0.1M
verl           p5    15792     6     0.0  S0:1.9M S1:2.9M S2:6.8M S3:8.1M S4:6.9M S5:4.6M S6:1.4M
verl_v1     s1_p1    21762     0     0.0  S0:32.5M
verl_v1     s1_p2    21113     1    11.8  S0:27.2M S1:4.0M
verl_v1     s1_p3    20747     3    17.7  S0:17.6M S1:10.1M S2:3.3M S3:0.5M
verl_v1     s1_p4    19783     3    23.5  S0:8.3M S1:11.8M S2:9.1M S3:2.0M
verl_v1     s1_p5    19921     4    22.4  S0:2.7M S1:10.0M S2:13.0M S3:4.9M S4:1.2M
verl_v1     s2_p1    21762     0     0.0  S0:32.5M
verl_v1     s2_p2    16483     2     7.9  S0:9.1M S1:21.2M S2:1.0M
verl_v1     s2_p3    17990     3    15.1  S0:6.7M S1:18.6M S2:5.2M S3:0.7M
verl_v1     s2_p4    18889     3    20.5  S0:4.3M S1:15.1M S2:9.8M S3:1.9M
verl_v1     s2_p5    19685     4    21.3  S0:2.4M S1:10.6M S2:13.3M S3:4.5M S4:1.2M
verl_v1     s3_p1    21762     0     0.0  S0:32.5M
verl_v1     s3_p2    15733     2     0.0  S0:4.1M S1:20.5M S2:8.0M
verl_v1     s3_p3    16590     3     7.3  S0:2.2M S1:9.7M S2:17.9M S3:1.7M
verl_v1     s3_p4    18039     3    14.2  S0:2.1M S1:8.9M S2:17.1M S3:3.4M
verl_v1     s3_p5    18965     4    18.6  S0:2.2M S1:6.6M S2:15.8M S3:5.5M S4:1.6M
verl_v1     s4_p1    21762     0     0.0  S0:32.5M
verl_v1     s4_p2    15733     2     0.0  S0:4.1M S1:20.5M S2:8.0M
verl_v1     s4_p3    15901     3     1.3  S0:2.0M S1:6.3M S2:17.3M S3:6.7M
verl_v1     s4_p4    16398     4     6.6  S0:2.0M S1:3.9M S2:13.8M S3:10.8M S4:0.9M
verl_v1     s4_p5    17986     4    12.3  S0:2.0M S1:4.7M S2:11.2M S3:11.1M S4:2.9M
verl_v1     s5_p1    21762     0     0.0  S0:32.5M
verl_v1     s5_p2    15733     2     0.0  S0:4.1M S1:20.5M S2:8.0M
verl_v1     s5_p3    15842     4     0.0  S0:2.0M S1:6.0M S2:17.3M S3:6.8M S4:0.4M
verl_v1     s5_p4    15906     4     0.5  S0:1.9M S1:3.5M S2:11.5M S3:11.5M S4:4.0M
verl_v1     s5_p5    16576     5     8.0  S0:2.1M S1:3.2M S2:9.5M S3:10.0M S4:6.1M S5:0.4M
```

A number that recurs across many presets' tightest setting is worth naming once:
span 21656 to 21762, throughput 1494 to 1502 tok/s, everything at `S0`, 0% waste.
Two rollout GPUs against six training GPUs on this long-tailed trace leaves
rollout far enough behind training that the run is fully synchronous: no token is
ever stale no matter which preset's admission or L3 knob happens to be the one
pinned tight enough to cause it. This is the regime property the design doc's
probe already recorded for `verl_v1` at `num_warmup_batches=1`. It recurs here
under several different names.

## Per-preset findings

**psrl.** Fully informative, and the cleanest figure in the set. `staleness`
caps the composition at exactly `S <= staleness` at every point (`maxS` tracks
the knob 0 through 5 exactly) with `waste%` at 0.0 throughout. Span drops from
the synchronous floor (21656s) to 16179s at `staleness=1` and settles to about
15810 to 15940s from `staleness=2` on. The bound is free in this regime: it
reshapes which `S` level a token lands in without ever discarding one.

**verl_v1.** The two-knob grid (`num_warmup_batches` x `max_off_policy_threshold`,
25 points) is the richest figure and reproduces the probe that set this regime.
At `num_warmup_batches=1` every `max_off_policy_threshold` panel's first point
sits on the synchronous floor, because warmup=1 alone is tight enough to prevent
rollout from ever running ahead (the same mechanism the design doc already
documents). Once warmup is 2 or higher, tightening `max_off_policy_threshold`
visibly grows a hatched `l3_stale` segment (up to 23.5% waste at
`max_off_policy_threshold=1`, `num_warmup_batches=4`) while the composition
narrows toward low `S`. Both axes are doing
real work here, and the trade the harness exists to show, a tighter bound costing
span and discarded tokens at once, is visible directly in the figure.

**verl (legacy).** `staleness_threshold` is a sample-overage ratio, not a version
count, and its axis label says so. `waste%` is 0.0 at every point: the ratio
gates how many groups are admitted between syncs rather than discarding anything
after the fact. The composition still broadens steadily from `S0`-only at 0 to a
spread reaching `S6` at 5, and span drops from 21656s to about 15670 to 15810s
once the knob is above 0. Informative, with the caveat its own grid comment
carries: the shared 0 to 5 axis is not commensurable with the other eleven
presets' version counts.

**areal.** The cumulative token bucket widens the composition smoothly across
its whole range (`maxS` 0, 2, 4, 5, 5, 6 for knob 0 through 5) with 0.0 waste
throughout, matching its description as an admission-only gate. Span settles to
about 15730 to 15890s once the knob is above 0. Informative.

**roll.** The one preset besides `verl_v1` whose figure shows real discard cost.
`async_generation_ratio`'s ceiling acts as both the admission window and an L2
GC, so unlike `psrl`'s free bound, `roll` pays for its cap: waste falls from
18.0% at knob=1 to 8.6% at knob=5 as the window loosens, visible as a shrinking
hatched segment on top of each bar. `maxS` tracks the knob closely (1 through 5).
Informative, and its own figure is the clearest illustration in the set that a
version-window cap is not automatically free.

**slime.** `update_weights_interval`'s floor is 1, not 0 (it is a loop interval,
not a threshold), so there is no synchronous-floor point on this preset's own
axis. Even at interval=1 the composition already spans `S0` and `S1`. `maxS`
tracks the knob exactly (1 through 5) with 0.0 waste throughout, and span
improves only modestly across the range, 17428s down to about 16825s. Informative,
the mildest of the clean version-cap presets in this regime.

**skyrl.** Confirmed non-bound, but not a flat figure. `waste%` is 0.0 at
every point, and that absence of discarding, not whether `maxS` exceeds the
knob, is what actually separates this preset from a genuine bound: `trl`
also slightly exceeds its own knob (see below) but does discard. Span and
throughput both move substantially (21762s at knob=0 to between 15014s and
15282s from knob=1 on, a real change in bar height), because the knob still
drives an admission quota even though it never caps the composition once
rollout is admitted. This matches the grid's own description. Staleness does
develop here (`maxS` is well above 0 once the knob opens), so this is not the
regime failing to let staleness develop. It is the mechanism itself not being
a staleness bound, exactly as the spec predicted, just not a bound that also
holds span constant.

**openrlhf.** Same shape as `skyrl` and the same conclusion: a confirmed
non-bound, not a flat figure. `waste%` is 0.0 at every point (the actual
distinguishing feature from a genuine bound), consistent with
`async_queue_size` bounding in-flight batch lumps rather than per-sample age.
Span moves too (16620s at knob=1 down to between 15080s and 15176s from
knob=2 on), the same admission effect `skyrl` shows. Staleness clearly
develops (up to `S6`), so this is case (a), not case (b): a confirmed
non-bound. The mandatory `partial_rollout_enable=true` pin is exercised at
every point.

**dressage.** The one preset whose entire swept range is a literal plateau: all
five points (`keep_versions` 1 through 5) are byte-identical, span 14854s,
`maxS=3`, 0.0% waste, same token spread down to the megabyte. This is not case
(b): the composition is a real, non-degenerate distribution (`S0` through `S3`),
so admission did open and staleness did develop. It is also not case (a) by
mechanism: `keep_versions` is described as an actual L3 filter (drop trajectories
outside the `keep` most-recently-novel distinct versions), the same genre of
bound as `psrl` or `trl`. What the data shows is that none of 1 through 5 is
tight enough to trigger that filter even once in this regime: the first distinct
version that would need dropping apparently lies beyond `keep=5`. This is the
plateau the design doc's own caveat anticipated ("the 0..5 ceiling might already
be past unbounded on some axes"), landing on `dressage`'s axis instead of the
ones the spec guessed at. Per that caveat, the right next step is to re-centre
just this preset's axis to a wider range, not to widen all twelve. That is future
work, out of scope here.

**nemo_rl.** Three distinct regimes on one axis. `max_staleness_versions=0`
livelocked outright in the original 88-point sweep (see "Sweep execution"
above), a floor exception like five other presets already carry, just
discovered here. The grid's own `PRIMARY_VALUES` has since been corrected to
drop that value, matching the other five floor exceptions.
`max_staleness_versions=1`
reproduces the regime's synchronous floor exactly (span 21762s, `maxS=0`,
identical numbers to `prime_rl`'s baseline and to every other preset's own
degenerate point): one unit of lookahead is not enough room for staleness to
develop when rollout is this far behind training. From 2 onward the knob is
informative, `maxS` reaches 2 then 3, span drops to 15574s then settles to 15559s, and
3, 4 and 5 are then byte-identical to each other (a smaller, partial plateau:
this regime's natural staleness ceiling for `nemo_rl` is `S3`, and loosening the
window past 3 has nothing further to bind). The axis genuinely is a staleness
bound. Its lowest legal point just happens to coincide with this regime's known
synchronous-floor pattern.

**trl.** The most direct bound in the study, and the busiest figure: a hatched
`l3_stale` segment appears at knob=0 (12.8% waste) and knob=2 through 5
(9.1%, 8.7%, 5.0%, 5.7%), vanishing only at knob=1. `maxS` tracks the knob with
a consistent one-off overshoot (knob=2 reaches `S3`, knob=5 reaches `S6`). Span
is not monotone across the range (15976s at knob=3 down to 15290s at knob=4,
back up to 15526s at knob=5), unlike
the cleaner caps. Informative, and the figure that most directly shows a
staleness bound charging a real, non-trivial, and non-monotonic cost.

**prime_rl.** Flat, and flat for a reason distinct from `skyrl` and `openrlhf`.
`waste%` is 0.0 and `maxS` is 0 at every single point, knob 0 through 5: every
bar is the regime's exact synchronous-floor number (span 21762s, all 32.5M
tokens at `S0`), and the figure's legend never needs an `S1` entry. This is case
(b), the regime never letting staleness develop, but for a specific, traceable
reason rather than bad luck: reading `srl/policy/prime_rl.py`, `admit_quota`
gates admission on `target_lag` (`dispatch_index >= train_version + target_lag`),
not on `max_off_policy_steps`, the knob this preset's grid actually sweeps.
`max_off_policy_steps` only drives the publish-time cancel and the ready-pool
sweep, both of which require something to already be stale before they can act.
`target_lag` defaults to 1 and is left at that default here (the design spec's
knob table lists it as this preset's deferred secondary knob). At `target_lag=1`,
rollout being this far behind training means dispatch can never build up enough of
a lead for anything to become stale, so `max_off_policy_steps`'s cancel horizon
never has anything to catch at any of its six tested values. `max_off_policy_steps`
genuinely is meant to be a staleness bound. This preset's own axis simply cannot
reach it while its paired admission knob sits at the regime's synchronous floor.

## The two predictions checked

The spec predicted `skyrl` and `openrlhf` would be flat. Neither is: both
show the admission quota moving span substantially once the knob opens. What
the spec actually meant, and what holds, is that neither knob ever caps the
staleness composition: zero waste at every point in both, confirming each
knob is an admission quota or an in-flight cap rather than an age-based
ceiling. The spec's own word choice, "flat," conflated these two claims.
`prime_rl` is also flat, but the
spec did not predict it, and the mechanism is different: its own cancel-horizon
knob never gets the chance to bind because the regime's synchronous floor is
reached through a different, un-swept parameter (`target_lag`) left at its
default. `dressage` is the preset whose whole tested range turned out to be one
plateau, exactly the risk the spec's "0..5 ceiling might already be unbounded"
caveat called out, just not on the axis the spec guessed at (`verl_v1` and
`max_off_policy_threshold=5` turned out to still be informative, not flat).
`nemo_rl` carries a smaller version of the same plateau from `max_staleness_versions=3`
onward, alongside a newly found floor at 0.

No preset is compared against another as a system anywhere above: every span,
waste share and composition described is that preset's own knob compared against
itself.
