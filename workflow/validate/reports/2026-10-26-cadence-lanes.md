# Report: cadence-aware lane scheduling (slack_lane / length_lane)

## 0. Correction and new experiment (2026-10-28)

**Every number in Section 3 below was regenerated and is now the corrected figure** -- this
is not a new draft layered on an old one. A real bug was found after this report's first
version shipped: `SimConfig.enable_trajectory_sticky` (whether a request resumed after a
weight-publish abort tries to go back to the exact instance it was on) did not exist in the
simulator at all when the original sweep and its driver were first written. A concurrent
session has since added it, defaulting `True` (documented as the realistic setting -- no real
system this project models migrates a resumed trajectory across instances). The original
sweep's 540 rows were therefore computed under the simulator's *absence* of this mechanism,
letting every weight-publish act as a free full-fleet rebalance for any strategy without its
own explicit placement logic. Confirmed directly: `slack_lane`-based arms (`a_slack`,
`a_nolane`, `a_slack_cap`) are completely insensitive to this flag (their own per-request plan
overrides the router's sticky narrowing before it runs); `request_num_balance` and
`length_lane` arms are not, and got measurably worse under the corrected setting (+0.3% to
+3.5% at sampled points, seed-dependent). The driver now pins `enable_trajectory_sticky=True`
explicitly (comment in `build()`), and the full 27x20 sweep was regenerated.

**A second, entangled change happened in the same regeneration, disclosed by the subagent that
ran it rather than hidden:** the point grid's own `t_sat` calibration was recomputed fresh
rather than held at its stale cached value -- a legitimate thing to do (each point's intended
`rho` should be measured against the fleet's actual achievable rollout rate under the
*corrected* simulator, not a pre-fix rate), but it moves every arm's span too, including
`a_slack`'s (confirmed directly: `a_slack` is unaffected by trajectory-sticky alone at every
sampled seed, but moves +1.5% to +1.9% from the t_sat recalibration alone). A controlled,
three-way decomposition at one representative point across all three seeds (`old` vs.
`sticky-only, old t_sat` vs. `combined, as shipped`) found the two effects' relative sizes are
**not predictable from how much t_sat itself drifted**: seed 0's t_sat moved only +0.22% yet
produced the *largest* span confound on `b0_default` (+2.37pp), while seed 1/2's much bigger
t_sat drifts (-4.22%/-7.43%) produced *smaller* confounds there (+0.49pp/+0.85pp). Both effects
point the same direction at every seed tested (the real gap between `a_slack` and the baseline
family grows), so the numbers below are reported as the corrected, combined, going-forward
truth -- not as a pure measurement of trajectory-sticky alone, which was not separately
re-swept across all 27 points (judged not worth another ~5h run once the direction and rough
magnitude were independently confirmed at representative points).

**A second, independent investigation in the same session tested a different hypothesis the
user raised: is the baseline's own apparent balance an artifact of *not* pinning a prompt
group's siblings to one instance (`enable_group_sticky`)?** This is reported separately below
(new subsection in Section 3) because, unlike `enable_trajectory_sticky`, `enable_group_sticky`
is **not** a bug fix -- its `False` default is a deliberate, already-validated setting matching
a real reference system's measured behavior (`srl/config/schema.py`: "False matches the matched
run, which scatters all 160 groups across both instances"). Testing `True` is a genuine
what-if experiment, not a correction, and the two new arms it produced
(`b0_group_sticky`, `a_slack_group_sticky`) are additive -- they do not replace or change any
of the other 27x20 rows.

## 1. Question and change

Tested whether giving a straggler request a low-concurrency "fast lane" at each psrl weight
publish, planned by `srl.forecast.lanes.plan_lanes`, shortens the wall time of a fixed number of
async-PSRL training steps on a real DAPO workload, compared against the strongest deployable
static baseline. `strategy="slack_lane"` (`strategy_args={}`) is the primary treatment; the
strongest static baseline, `b1_best`, is the best-gated of `request_num_balance` at
`max_concurrent_seqs_per_instance` in `{16, 24, 32, 48}` at that same point. `b0_default`
(`request_num_balance`, no cap) is reported as the secondary baseline. `a_nolane`
(`strategy_args={"enable_upgrade": False}`) isolates the value of `plan_lanes`'s balanced
placement alone from its fast-lane upgrade search. `length_lane` (static, no planning, a fixed
set of fast instances behind a length threshold) is the ablation for whether planning buys
anything over a fixed threshold.

## 2. Session

- Git head: `681f999` as of the correction re-run. `srl/` carries the same long-standing set of
  concurrent-session edits present throughout this entire plan (see `preflight.py`'s
  `dirty_srl_files`); none of them were introduced by this sweep.
- Binary md5 of `srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so`:
  `4c9a6244aeaf872809771252a83e7ad6`, identical before and after both the original ~4h44m sweep
  and the ~5h19m correction re-run.
- `SIMRL_KV_CACHE` unset. Preflight warnings, quoted verbatim: "SIMRL_KV_CACHE is unset. Every
  run re-probes KV capacity." and "Uncommitted changes under srl/. Record them, the baseline is
  this same tree." Both are pre-existing environment conditions, not caused by this sweep.
- Workload: `outputs/workload/qwen3_8b_dapo_38912_tp1` (1024 real DAPO groups x 8, cap 38912),
  permuted 3 ways by seed. Fleet: 16 x TP1 Qwen3-8B on H20, composed cost model, KV from the
  real probe, `weight_load_time=0.0`. All lengths come from the oracle predictor.
- **`ingress_lifetime_cap` note (unchanged from the original sweep).** A concurrent session's
  edit added `SimConfig.ingress_lifetime_cap` (default `False`) mid-sweep, during the *original*
  run; the driver pins `ingress_lifetime_cap=True` explicitly to reproduce the sweep's own
  original semantics. This pin was carried forward unchanged into the correction re-run.
- **`enable_trajectory_sticky` note (Section 0 above).** The driver now also pins
  `enable_trajectory_sticky=True` explicitly, the new, corrected setting -- unlike
  `ingress_lifetime_cap`, this is NOT pinned to the old (pre-fix) value, because the old value
  reflected the literal absence of a mechanism, not a deliberate, still-valid choice.

## 3. Gated results

27 points (`staleness in {1,2,3} x rho in {0.9,1.0,1.1} x seed in {0,1,2}`), 20 arms, 540 rows
(plus 54 additional rows for the 2 new `*_group_sticky` arms, reported separately below --
594 total rows). 513 treatment rows judged for the core 20-arm sweep; **486 gated clean, 27
dropped** (same reason and same exact 27 row_ids as the original sweep, see Section 4 --
confirmed unchanged by the correction). Every number below is independently recomputed from
`summary.json` directly, to 4 decimal places; none are taken on trust from either subagent that
ran a sweep.

### Geometric mean per arm, over gated points (corrected; old pre-fix numbers in parentheses)

| arm | n gated / 27 | geomean vs `b1_best` | geomean vs `b0_default` |
|---|---|---|---|
| `a_slack` (primary treatment) | 27 | **1.0200** (was 1.0121) | 1.0271 |
| `a_nolane` (balance only, no upgrade) | 27 | 1.0135 (was 1.0045) | 1.0206 |
| `a_slack_cap` (slack_lane + best static cap) | 27 | 1.0173 (was 1.0000) | 1.0243 |
| `b1_cap16` | 27 | 0.9950 (was 0.9969) | 1.0019 |
| `b1_cap24` | 27 | 0.9889 (was 0.9922) | 0.9957 |
| `b1_cap32` | 18 | 0.9861 (was 0.9855) | 0.9923 |
| `b1_cap48` | 9 | 0.9889 (was 0.9852) | 0.9966 |

`b1_cap32` and `b1_cap48`'s geomeans are over smaller, staleness-skewed subsets (see Section 4):
do not compare them to the other rows' 27/27 geomeans as if the point sets matched.
`a_slack_cap`'s geomean moved the most in absolute terms (1.0000 -> 1.0173) because its own
per-point cap is resolved from whichever `b1_cap*` arm gates best at that point, and the
corrected `b1_cap*` family's own relative ranking shifted slightly between points.

### Why `a_slack` beats `a_nolane`: the upgrade search is inactive at low staleness

`a_slack` and `a_nolane` are **byte-identical** (same `span`, to full float precision) at all 9
`staleness=1` points, plus 2 `staleness=3` points (`s3_rho1.00_seed1`, `s3_rho1.10_seed0`) --
**11** of 27 points total (was 10 of 27 before the correction; the specific staleness=3 point in
this bucket shifted from `s3_rho1.10_seed1` to `s3_rho1.00_seed1`/`s3_rho1.10_seed0` because the
t_sat recalibration moved which points land in this structurally-zero-improving-move regime --
the mechanism itself, not the correction, decides this set). `a_slack`'s entire edge over
`a_nolane` comes from the 16 of 27 points where the search found a genuine, net-positive move --
still structurally a no-op at `staleness=1` at this 16-instance fleet scale, not a tuning
artifact and not changed in kind by the correction.

### `length_lane` family (static, no planning): honest number vs. in-sample oracle

Reported separately per the plan's own instruction -- do not conflate these:
- **Honest** (the single arm with the best geomean across all 27 gated points): `c_k1_L20000_c2`
  (1 fast instance, length threshold 20000, capacity 2, same winner as before the correction),
  geomean vs `b1_best` = **1.0130** (was 1.0143).
- Full 12-arm range vs `b1_best`: 0.9347 (`c_k4_L30000_c4`) to 1.0130 (the honest pick). Ranking
  order among the 12 arms is unchanged by the correction.
- `length_lane`'s honest number (1.0130) remains comparable to, and now very slightly below,
  `a_slack`'s (1.0200) -- the static ablation is still not dominated by the planned strategy
  here, if anything slightly less competitive than before the correction.

### `headroom_capture` (fraction of the gap between `b1_best` and the contention-free bound
captured), gated rows -- corrected; old numbers in parentheses

| arm | n | median | min | max |
|---|---|---|---|---|
| `a_slack` | 27 | **+0.0755** (was +0.039) | -0.1811 (was -0.238) | +0.3415 (was +0.191) |
| `a_slack_cap` | 27 | +0.0622 (was +0.012) | -0.0631 (was -0.220) | +0.3682 (was +0.179) |
| `a_nolane` | 27 | +0.0463 (was -0.023) | -0.2849 (was -0.274) | +0.2700 (was +0.191) |
| `b1_cap16` | 27 | 0.0000 | -0.1901 (was -0.099) | 0.0000 |
| `b1_cap24` | 27 | -0.0542 (was -0.022) | -0.6322 (was -0.297) | 0.0000 |
| `b1_cap32` | 18 | -0.0726 (was -0.115) | -0.6343 (was -0.370) | 0.0000 |
| `b1_cap48` | 9 | -0.0457 (was -0.112) | -0.4783 (was -0.343) | 0.0000 |

Median `headroom_capture` for `a_slack` roughly doubled (+0.039 -> +0.0755) once the baseline
family no longer benefits from a free full-fleet rebalance at every weight publish -- it is
still below the plan's own 0.5 decision threshold for Algorithm B, so Task 10's **BUILD**
decision for Algorithm B stands unchanged (if anything more clearly warranted: there is still a
large majority of the theoretical headroom uncaptured at the median point).

### Group sticky: testing a deliberately non-validated, alternative baseline assumption

Per Section 0: `enable_group_sticky=True` is not a bug fix, it is a what-if test the user asked
for directly. Two new arms, `b0_group_sticky` (`request_num_balance` + the flag) and
`a_slack_group_sticky` (`slack_lane` + the flag), measured across all 27 points (54 new rows,
appended to `summary.json`, all 54 gated clean, zero dropped). Every pre-existing row was
verified byte-identical to a pre-run backup before and after (0 mismatches across the full
540-row prefix).

**The effect is an order of magnitude larger than the trajectory-sticky correction above, and
the direction confirms the user's intuition precisely.**

| metric | `b0_group_sticky` | `a_slack_group_sticky` |
|---|---|---|
| geomean `speedup` vs `b0_default`, 27/27 | **0.8166** (18.3% worse) | 1.0025 (a wash) |
| geomean `speedup_vs_b1_best`, 27/27 | 0.8110 | 0.9956 |
| per-point spread of `speedup` vs `b0_default` | min 0.7459 (`s1_rho0.90_seed1`, -25.4%), max 0.9056 (`s3_rho1.10_seed2`, -9.4%), median 0.8083 | -- |
| `a_slack_group_sticky` span / plain `a_slack` span, geomean | -- | **1.0245** (forcing group-sticky onto the planned arm costs it ~2.45% by itself) |
| `headroom_capture` median / min / max | -- | **-0.0145** / -0.5901 / +0.3060 (plain `a_slack`: +0.0755 / -0.1811 / +0.3415) |

Three things stand out:
1. **`b0_group_sticky`'s own damage is staleness-band-correlated, not uniform**: every
   `staleness=1` point sits below 0.837 (worst band), `staleness=2` is in the middle
   (0.760-0.852), `staleness=3` is the least damaged (0.806-0.906). The tighter the staleness
   window, the less time the fleet has to let a correlated-length group's siblings naturally
   spread out again after landing together.
2. **`a_slack` absorbs almost all of this cost on its own** -- its geomean `speedup` vs
   `b0_default` stays at 1.0025 (statistically a wash) even while forced to respect
   group-sticky, versus `b0_group_sticky`'s 0.8166. This matches the mechanism found for
   trajectory-sticky: `SlackLane`'s own explicit per-request plan mostly overrides the router's
   sticky narrowing (not completely this time -- `a_slack_group_sticky` does cost ~2.45% versus
   plain `a_slack` in isolation, unlike trajectory-sticky's exact 0.000%, because `SlackLane`'s
   own `narrow_candidates` leaves off-plan arrivals only partially constrained, which is exactly
   where the router's group-pin can still bite).
3. **`headroom_capture` for `a_slack_group_sticky` is not an apples-to-apples number with plain
   `a_slack`'s** -- it is computed against the *same* `b1_best`/`lb` bound as every other arm,
   which itself was never re-measured under `enable_group_sticky=True`. A negative median here
   does not mean `a_slack` lost ground against a `b1_best` that would also get worse under
   group-sticky (it almost certainly would, given `b0_group_sticky`'s own -18.3% damage) --
   it means `a_slack`-under-group-sticky is being compared against a bound that is itself too
   easy relative to that scenario. Flagged in Section 5, not corrected here (would need
   `b1_cap*` re-run under the flag too, out of scope for this already-large correction pass).



### Algorithm B decision and measurement (Task 10)

**Decision: BUILD.** `plan_cadence` (bisecting on a uniform target period between `train.cycle`
and the bulk plan's own mean buffer spacing) was added to `srl/forecast/lanes.py`, gated behind
a new `cadence_target` flag on `SlackLane`, and measured as a new arm, `b_cadence`
(`strategy_args={"cadence_target": True}`), across the same 27 points, reusing each point's
already-computed `b1_best_span`/`lb` rather than re-deriving them. All 27 rows gated clean.

**Measurement: `b_cadence` is byte-identical to `a_slack` at all 27 of 27 points** (same `span`,
`mean_concurrency`, every field, to full float precision) -- not just the 10 points where
`a_slack` is already known to equal `a_nolane` (Section 3's earlier finding), but every single
point, including the 17 where `plan_lanes`'s own search finds a genuine, non-trivial move. Its
geomean `speedup_vs_b1_best` (1.0121) and `headroom_capture` (median +0.0386, min -0.2383, max
+0.1913) are therefore not a new number -- they are `a_slack`'s own, replayed.

**Mechanism, not assumed: `plan_cadence` structurally never ran its own search on this
workload.** Instrumenting `plan_cadence` directly at two representative points
(`s2_rho1.00_seed0`, `s3_rho0.90_seed1`, 51 dispatch bursts each) shows every single call
returning `plan_lanes`'s own plan unchanged, for two distinct, now-isolated reasons:

1. **50 of 51 bursts at the sampled point: the bisection bracket itself is inverted.**
   `plan_cadence`'s own `lo, hi = train.cycle, bulk_mean_period` is followed by
   `feasible_plan_for(hi) if hi > lo else None` -- and `bulk_mean_period`, computed as
   `(bulk.pipeline.ready[n-1] - target_0) / n` under the bulk (no-upgrade) assignment, came out
   *below* `train.cycle` at every one of these 50 bursts (e.g. `hi=108.95` vs `lo=503.14` at one
   burst). The search is never even attempted -- `plan_cadence` falls back to `plan_lanes`
   through the `best_feasible is None` guard before the bisection loop runs once.
2. **The 1 remaining burst, where the bracket was valid (`hi > lo`), still failed to find a
   fully feasible period** within `max_moves=8` -- `_search_for_period` returned `None` there
   too.

**This is a design limitation of the specific bracket construction, not evidence that
uniform-period planning cannot help.** `bulk_mean_period` is a single average over the whole
horizon; a buffer can be individually late relative to `plan_lanes`'s own sequential,
chain-based target (`_target_time`, which depends on the *previous* buffer's actual training
end) while the simple average spacing across the horizon stays below `train.cycle` -- exactly
the raggedness that lets `plan_lanes` find a real move at 17 of 27 points while this bracket
reports "nothing worth tightening" almost everywhere. The two searches are not comparable here
because the uniform-period framing's own upper bound rarely clears its lower bound on this
real, bursty trace -- not because the underlying idea was tested and found wanting.

**Not established:** whether a bracket built from `plan_lanes`'s own achieved (ragged, not
averaged) spacing, or from a different feasibility target entirely, would let `plan_cadence`'s
search engage and find real headroom beyond `a_slack`'s. This was not attempted here -- fixing
the bracket is a fresh design change that needs its own review cycle, not a same-task patch
under this task's own "measure B as one more arm" scope. The data gathered (`b_cadence`'s 27
rows, at the time this section was written) is correct and real; it measures `plan_cadence`
exactly as specified, and that specification's own bracket is what did not engage.

**Post-correction note.** `b_cadence`'s 27 rows do not exist in the current `summary.json` --
they were added via a separate side-script (`_add_b_cadence_arm.py`) against the pre-correction
sweep and were not carried forward or re-run against the corrected baseline (the main 27x20
re-run only regenerates the arms in `cadence_lanes.py`'s own `ARMS` dict, which never included
`b_cadence`). The specific numbers quoted above (geomean 1.0121, headroom_capture +0.0386) are
therefore stale and should not be read against this report's own corrected Section 3 numbers.
**The mechanism finding itself is unaffected** -- it was established by instrumenting
`plan_cadence`'s own code path directly (the bisection bracket's `hi`/`lo` values, confirmed by
reading `_search_for_period`'s logic, not by comparing spans) rather than by anything specific
to the old sweep's measured numbers, and both `plan_cadence` and `plan_lanes` are
`slack_lane`-based, confirmed insensitive to `enable_trajectory_sticky`. Re-measuring
`b_cadence` against the corrected baseline was not done in this pass; it would need re-running
`_add_b_cadence_arm.py` once more, which is cheap but was not judged a priority next step.

### Instance-level placement: checked and closed

After reporting the above, a direct follow-up question was raised: is the small measured
headroom an artifact of the baseline router already colocating a GRPO group's siblings onto
one instance, hiding real load imbalance this plan's lane-reordering work never looked at?
Checked directly against the real router code and the real trace, independently re-verified
rather than taken on a single investigating pass:

1. **No colocation happens, code-confirmed.** `RequestNumBalance.select()`
   (`srl/component/rollout/router/strategies/request_num_balance.py:19-20`) is
   `min(..., key=lambda i: candidates[i].n_active)` -- it never reads `group_id`.
   `Router.enable_group_sticky` would do this, but defaults `False` and this plan's driver
   never sets it (confirmed by direct grep, zero hits).
2. **Instance-busy-seconds imbalance under `b0_default` is real but modest, and shrinks as
   concurrency rises**: 7.2-10.9% spread across 5 sampled points (Task 9's own one point plus 4
   more at varied staleness/arm) -- not the large imbalance a correlated-sibling-colocation
   hypothesis would predict if it were actually happening.
3. **`a_slack`'s own spread is consistently larger** (11.6-19.9%) than the baseline's -- the
   fast-lane mechanism's own deliberate cost of thinning some instances to isolate a straggler,
   not unclaimed baseline imbalance sitting there unnoticed.
4. **Siblings genuinely are length-correlated** (within-group CV 0.2016 vs. across-group CV
   0.5953, independently recomputed directly from
   `outputs/workload/qwen3_8b_dapo_38912_tp1/joint.jsonl`) -- but since point 1 means they are
   never colocated, this correlation is never amplified into a real per-instance load spike.

This converges with two independent prior investigations in this same codebase --
`memory/psrl-headroom-lives-at-s1-and-balanced-training.md` ("No gain from: placement by
predicted load...") and `memory/long-tail-headroom-is-near-zero-fleet-is-saturated-not-starved.md`
(fleet saturated, busy fraction 0.89-1.00, not starved) -- plus this plan's own Task 9
diagnosis (idle-window busy fraction 92.8-99.3%). The fleet here runs consistently near its own
capacity ceiling; reshuffling who-goes-where, at admission time or after, moves contention
around without creating new idle capacity to redirect toward. This closes the loop the user's
own question opened -- it is not left as a dangling "not established" item.

**Update (2026-10-28): the magnitude, directly measured, not just reasoned about.** The
conclusion above ("no colocation happens, so it's never amplified") is still correct as a
statement about the validated baseline -- but it understated how large the amplification would
be if colocation *did* happen. The "group sticky" subsection earlier in this section directly
measures exactly that scenario (`enable_group_sticky=True`): `b0_default` gets **18.3% worse
geomean, up to 25.4% worse at the hardest point** -- the single largest effect size found
anywhere in this entire investigation, bigger than the primary `a_slack` treatment effect
itself. The structural reasoning here (point 4 above: correlated-length siblings forced
together would spike per-instance load) was right in direction; the real magnitude, now
measured rather than inferred, is far larger than the phrase "closed, dead end" implied. The
baseline's real balance remains a deliberate, validated design choice (point 1, unchanged) --
but it is a choice with real stakes, not a free lunch nobody could lose by reversing.

### Step-interval autocorrelation: testing the period-2 oscillation prior

**Not re-run after the trajectory-sticky correction.** The table below predates the Section 0
correction -- `b0_default`'s absolute numbers here are stale by the same +0.3%-to-+3.5%-type
shift documented in Section 0 (and further by the t_sat recalibration), and `a_slack`'s are
stale only via the t_sat effect. Judged low-priority to re-run: this subsection's conclusion is
about the *shape* of step-interval autocorrelation (whether a period-2 oscillation is present),
a qualitative question unlikely to flip sign from a few-percent span shift, and re-running it
would mean another 6 full-telemetry replays for a secondary analysis. Treat the exact
`trainer idle (s)` and `fleet idle frac` values below as illustrative of the shape, not as
current numbers; everything in Section 3 above is current.

Explicit test of the prior that `staleness=S` should produce an oscillation with some period
related to `S`. Measured by re-running 6 representative points (`b0_default` and `a_slack`, one
per staleness level, `rho=1.0, seed=0`) with full telemetry and applying
`srl.diagnosis.cadence.cadence_report` directly (the same function approved in Task 6):

| point | arm | trainer idle (s) | fleet idle frac | acf1 (lag 1) | acf2 (lag 2) |
|---|---|---|---|---|---|
| `s1_rho1.00_seed0` | `b0_default` | 3105.7 | 0.223 | **-0.526** | +0.082 |
| `s1_rho1.00_seed0` | `a_slack` | 2994.6 | 0.211 | **-0.658** | **+0.432** |
| `s2_rho1.00_seed0` | `b0_default` | 1710.8 | 0.200 | -0.274 | -0.314 |
| `s2_rho1.00_seed0` | `a_slack` | 1644.2 | 0.192 | -0.260 | -0.316 |
| `s3_rho1.00_seed0` | `b0_default` | 872.8 | 0.148 | +0.093 | -0.026 |
| `s3_rho1.00_seed0` | `a_slack` | 803.9 | 0.143 | +0.073 | +0.029 |

(Corrected from an earlier draft of this report: the first pass of these 6 re-runs executed in a
fresh process after a concurrent session's edit to `srl/simulator/simulator.py`/`config.py`
landed on disk mid-sweep -- see Section 2's note on `ingress_lifetime_cap` -- and so used a
different ingress-admission default than the frozen-at-import-time process that actually produced
`summary.json`. Every span above now reproduces its point's stored `summary.json` row exactly, to
full float precision; `acf1`/`acf2` barely moved (the shape of the interval sequence is largely
insensitive to the flag), but `fleet_idle_frac` moved substantially -- up to +175% relative at
`staleness=3` -- and this table's prose below is written against the corrected values.)

**The user's period-2 prior does not hold as stated, and this is a negative result worth
reporting as such.** A clean period-2 oscillation requires both a strongly negative lag-1 AND a
strongly positive lag-2 autocorrelation (the signature confirmed in Task 6's own unit test on a
synthetic period-2 series: acf1 < -0.5, acf2 > 0.5). That signature appears at only one of the
six measured cells: `a_slack` at `staleness=1` (acf1=-0.658, acf2=+0.432). Everywhere else the
pattern is weaker or absent:
- `staleness=1`, `b0_default`: a real but weak tendency toward short-long alternation
  (acf1=-0.528) that does not resolve into a clean 2-cycle (acf2 only +0.082).
- `staleness=2` (the case the user explicitly hypothesized about): **both arms show a NEGATIVE
  lag-2 autocorrelation** (-0.309, -0.316), which is the opposite sign a period-2 cycle would
  need. There is a short-long alternation tendency (negative acf1) but it does not repeat with
  period 2.
- `staleness=3`: both autocorrelations are near zero for both arms (|acf1|, |acf2| < 0.1) — no
  periodicity signal of any kind.
- `a_slack` versus `b0_default` at the same point: the lane strategy does not meaningfully
  change the cadence's periodicity signature (near-identical acf1/acf2 at staleness 2 and 3);
  it is only materially different at staleness=1, where it makes the existing weak alternation
  tendency notably *stronger*, not weaker, while still not reaching a clean 2-cycle.

Fleet idle fraction drops monotonically with staleness (~0.22 to ~0.20 to ~0.15, averaged over
the two arms at each level) regardless of arm, consistent with staleness relaxing the admission
window and giving the fleet more continuously-available work — this is the dominant, well-behaved
trend; a repeating oscillation is, at best, a secondary and partial effect confined to low
staleness.

## 4. Failed gates

**27 of 513 treatment rows dropped, all for the same gate: `treatment_differs` (row was
byte-identical to `b0_default`).** No row failed `run_completed`, `consumed_groups`,
`tokens_per_group`, or `max_staleness`. Confirmed directly from `summary.json`: every dropped
row's `span` exactly equals that point's `b0_default` row's `span`.

The pattern is monotonic in staleness, consistent with how `max_concurrent_seqs_per_instance`
interacts with psrl's own admission window — a looser staleness window admits more work in
flight, which is what lets a high cap actually bind:

- `staleness=1`: `b1_cap32` and `b1_cap48` dropped at all 9 points (18 rows) — the cap never
  binds below 32 concurrent requests at this staleness, so capping at 32 or 48 is indistinguishable
  from no cap.
- `staleness=2`: `b1_cap48` dropped at all 9 points (9 rows); `b1_cap32` gated clean at all 9.
- `staleness=3`: no drops; both `b1_cap32` and `b1_cap48` gated clean.

Full list (point, arm, span) -- corrected values, same 27 row_ids as before the correction:

```
s1_rho0.90_seed0/b1_cap32   8526.7084      s1_rho0.90_seed0/b1_cap48   8526.7084
s1_rho1.00_seed0/b1_cap32   8661.4102      s1_rho1.00_seed0/b1_cap48   8661.4102
s1_rho1.10_seed0/b1_cap32   9059.0569      s1_rho1.10_seed0/b1_cap48   9059.0569
s1_rho0.90_seed1/b1_cap32   9061.2164      s1_rho0.90_seed1/b1_cap48   9061.2164
s1_rho1.00_seed1/b1_cap32   9565.7585      s1_rho1.00_seed1/b1_cap48   9565.7585
s1_rho1.10_seed1/b1_cap32   9682.4807      s1_rho1.10_seed1/b1_cap48   9682.4807
s1_rho0.90_seed2/b1_cap32   9219.1717      s1_rho0.90_seed2/b1_cap48   9219.1717
s1_rho1.00_seed2/b1_cap32   9316.7642      s1_rho1.00_seed2/b1_cap48   9316.7642
s1_rho1.10_seed2/b1_cap32   9486.9482      s1_rho1.10_seed2/b1_cap48   9486.9482
s2_rho0.90_seed0/b1_cap48   7036.1264      s2_rho1.00_seed0/b1_cap48   7343.2003
s2_rho1.10_seed0/b1_cap48   7491.2563      s2_rho0.90_seed1/b1_cap48   7457.9730
s2_rho1.00_seed1/b1_cap48   7460.5349      s2_rho1.10_seed1/b1_cap48   7763.1854
s2_rho0.90_seed2/b1_cap48   6894.7688      s2_rho1.00_seed2/b1_cap48   7069.3428
s2_rho1.10_seed2/b1_cap48   7389.0329
```

This is treated as evidence the gate fired correctly on a genuinely non-binding cap, not a
wiring failure: `a_nolane`, `a_slack`, and all 12 `c_*` arms differ from `b0_default` at every
single one of the 27 points, confirming the harness as a whole is sensitive to real changes.

Preemptions and `reprefill_tokens` were recorded on every row without gating. Across all 513
treatment rows (corrected sweep): **2** total preemptions (both in `c_k4_L20000_c2`; was 101
before the correction, 100 in `c_k4_L20000_c2` and 1 in `c_k4_L30000_c2` -- `enable_trajectory_
sticky` noticeably reduced preemption pressure for this arm). **Zero** in `a_slack`, `a_nolane`,
`a_slack_cap`, `b0_default`, or any `b1_cap*` arm, unchanged. This sweep does not meaningfully
exercise KV pressure tight enough to force preemption outside one narrow corner of the
`length_lane` grid, now narrower than before the correction.

## 5. Not established

- **Oracle length predictor only.** Every arm, including the static `length_lane` ablation,
  used `length_model="oracle"`. Whether `slack_lane`'s greedy upgrade search (which prices
  candidate moves using predicted remaining length) degrades gracefully, or falls apart, under
  a realistic noisy predictor was not tested in this sweep.
- **No tight KV pressure.** `preemptions=0` on every row of every arm that matters for the
  headline comparison (`a_slack`, `a_nolane`, `a_slack_cap`, `b0_default`, all `b1_cap*`). This
  sweep's workload and fleet sizing (16 instances, 1024-group real trace, cap 38912) never drove
  the KV pool tight enough to force eviction. Whether the lane planner's assumptions (KV-safe
  concurrency, `InstanceForecast`'s drain replay) hold under genuine preemption pressure is
  untested.
- **One real trace, one task shape.** The DAPO math trace (3 random permutations as the only
  source of seed variance) is the only workload tested. Whether these results generalize to a
  genuinely agentic trace with tool-call-driven length distributions, a different model size,
  or a different TP degree is untested — this plan's own earlier rounds (documented in memory)
  already found that length-distribution shape materially changes heterogeneous-TP's payoff in
  a related but distinct setting, so this is not a safe extrapolation to assume away.
- **Determinism was spot-checked, not swept.** Task 7's smoke test confirmed bit-exact
  determinism at one point (`rel_diff=0.0`); this sweep did not re-run every point twice to
  confirm determinism holds everywhere, nor did it introduce seed variance beyond the 3 trace
  permutations (which change workload order, not the simulator's own randomness, since there
  is none in this deterministic replay engine).
- **`enable_trajectory_sticky`'s effect was not cleanly isolated across the full 27-point grid.**
  The full re-sweep combines the sticky fix with an also-recalibrated `t_sat`; both move the
  real `a_slack`-vs-baseline gap in the same direction at every seed sampled, but the exact
  split between the two causes is only measured at one representative point (Section 0), not
  all 27. A from-scratch re-run holding `t_sat` frozen at its pre-fix cached value would isolate
  the sticky effect alone; not done here, judged not worth another ~5h run once direction and
  rough magnitude were confirmed.
- **`enable_group_sticky`'s `headroom_capture` number is not apples-to-apples.** `b1_cap*` (and
  therefore `b1_best`/`lb`) were never re-measured under `enable_group_sticky=True` -- the
  reported `a_slack_group_sticky` headroom_capture compares against a bound that itself would
  almost certainly also get much worse under that scenario (given `b0_group_sticky`'s own
  -18.3% damage). A clean "headroom under a group-sticky world" number would need the `b1_cap*`
  family re-run under the flag too; not attempted here.
