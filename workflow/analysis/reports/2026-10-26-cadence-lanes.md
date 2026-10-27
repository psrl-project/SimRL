# Diagnosis report

## 1. What was diagnosed

Experiment: `workflow/validate/experiments/cadence_lanes.py` (gitignored driver, read for
`build(spec, output_dir=None)`). Result set: `outputs/simulate/cadence_lanes/summary.json`
(540 rows, 486 gated), the 27-point x 20-arm sweep comparing the `slack_lane` router
strategy (arm `a_slack`) against the strongest gated static baseline per point (`b1_best`, a
`request_num_balance` cap sweep).

**This redoes an earlier diagnosis of the same experiment against a different dataset.**
Between that earlier pass and this one, `SimConfig.enable_trajectory_sticky` (whether a request
resumed after a weight-publish abort returns to the exact instance it was on) was added by a
concurrent session and the driver now pins it explicitly to `True`, the realistic default (no
real system this project models migrates a resumed trajectory across instances). `slack_lane`
arms are confirmed insensitive to this flag; `request_num_balance`/`length_lane` arms are not,
and the full sweep was regenerated under the corrected setting (540 rows, overwritten;
`summary.json.pre-trajectory-sticky-fix.bak` is the old version, not used here). A second,
entangled change -- the point grid's own `t_sat` calibration was recomputed fresh rather than
held at its old cached value -- affects every arm including `a_slack` itself by +1.5% to +1.9%,
unrelated to routing. Both effects are already measured and are not re-investigated here. The
prior report's win/loss points (`s1_rho0.90_seed1`, `s1_rho1.00_seed1`, `s3_rho1.10_seed0`) are
**not** this report's points -- the corrected sweep's gated `speedup_vs_b1_best` extremes land
elsewhere, and no attempt is made below to reconcile this report's numbers with that one's.

Rows selected, as assigned (3 points x 2 rows, already computed from the corrected gated data):

| point | reason | treatment row | control row |
|---|---|---|---|
| `s1_rho1.00_seed1` | largest `a_slack` win vs `b1_best` (speedup_vs_b1_best = 1.0703) | `s1_rho1.00_seed1/a_slack` | `s1_rho1.00_seed1/b1_cap16` |
| `s2_rho0.90_seed1` | 2nd largest win (speedup_vs_b1_best = 1.0631) | `s2_rho0.90_seed1/a_slack` | `s2_rho0.90_seed1/b1_cap32` |
| `s3_rho1.00_seed1` | smallest speedup / biggest loss (speedup_vs_b1_best = 0.9853) | `s3_rho1.00_seed1/a_slack` | `s3_rho1.00_seed1/b1_cap16` |

Each point's own `b1_best_arm` (read from `summary.json["points"]`, not assumed) confirms the
control: `b1_cap16` at `s1_rho1.00_seed1` and `s3_rho1.00_seed1`, `b1_cap32` at
`s2_rho0.90_seed1` -- the winning cap differs by point, as flagged in the task.

Workload: real Qwen3-8B DAPO trace (`outputs/workload/qwen3_8b_dapo_38912_tp1/joint.jsonl`,
permuted per seed), composed TP1 H20 rollout cost model, 16 instances, 24 training steps, PSRL
policy, `backend="native"`. `s1`/`s2`/`s3` are psrl staleness horizons 1/2/3; `rho` is training
speed relative to the fleet's own rollout-bound saturation rate. All 3 points here are
`seed=1`.

## 2. Session

Git head: `681f9998e6aedadc4486087bdfb135946d5b3d02`. Native backend binary
(`srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so`) md5: `4c9a6244aeaf872809771252a83e7ad6`,
mtime 2026-10-17 -- unchanged from the prior diagnosis of this experiment. Preflight: passed, no
blocking or warning entries. Replay wall time per row (single-arm replay through `replay.run`,
24 training steps, telemetry on): `s1_rho1.00_seed1/a_slack` 22.9s, `s1_rho1.00_seed1/b1_cap16`
20.4s, `s2_rho0.90_seed1/a_slack` 27.4s, `s2_rho0.90_seed1/b1_cap32` 26.2s,
`s3_rho1.00_seed1/a_slack` 38.5s, `s3_rho1.00_seed1/b1_cap16` 35.5s.

## 3. Fidelity

**All 6 rows pass.** Per `workflow/analysis/scripts/replay.py`'s `fidelity()` (1e-9 relative for
floats, exact for counts), every one of the 9 gated fields matches bit-for-bit on every row,
under the driver's own pinned `enable_trajectory_sticky=True` / `ingress_lifetime_cap=True`.
Representative table, `s1_rho1.00_seed1/a_slack`:

| field | ok | row | replay |
|---|---|---|---|
| span | pass | 8864.704365090323 | 8864.704365090323 |
| consumed_tokens | pass | 65238857 | 65238857 |
| consumed_groups | pass | 768 | 768 |
| tokens_per_group | pass | 84946.42838541667 | 84946.42838541667 |
| mean_concurrency | pass | 6.146723663096945 | 6.146723663096945 |
| peak_concurrency | pass | 40 | 40 |
| preemptions | pass | 0 | 0 |
| versions_done | pass | 24 | 24 |
| dropped | pass | 0 | 0 |

All 9 fields pass, identically, for the other 5 rows (`s1_rho1.00_seed1/b1_cap16`,
`s2_rho0.90_seed1/a_slack`, `s2_rho0.90_seed1/b1_cap32`, `s3_rho1.00_seed1/a_slack`,
`s3_rho1.00_seed1/b1_cap16`). `consumed_tokens`/`consumed_groups`/`tokens_per_group` are
identical across every arm at a given point (65238857 / 768 / 84946.43 for every row above),
confirmed from the row data itself, not assumed -- the two arms at each point consume exactly
the same set of groups by the end of the run.

### 3.1 The separate cross-source reconstruction check

`diagnose.build()` also requires `srl.diagnosis.timeline.request_phases`'s 4 cross-source
checks to pass (`decode_tokens_match`, `completion_time_match`, `abort_count_match`,
`reprefill_present`) before computing anything. Of the 6 fidelity-passing rows, **5 pass this
second gate too** -- a material change from the prior diagnosis of this experiment, where only
4 of 6 passed. Only one row fails:

- `s3_rho1.00_seed1/b1_cap16`: `decode_tokens_match` -- 10 of 4872 uninterrupted requests
  disagree (e.g. `rid=544`: reconstructed decode tokens 5735 vs recorded `generated` 5796), and
  `reprefill_present` -- 11 requests charged reprefill tokens with no reconstructed reprefill
  phase.

This is the same failure shape, on the same kind of row (a `b1_cap*` control, never `a_slack`,
at the loss point), that the prior diagnosis of this experiment recorded as unresolved -- see
`workflow/analysis/memory/lessons.md`'s 2026-10-27 entry. It recurred here. Per this unit's own
rule, no idle-phase breakdown, bound, or fleet-window table was computed for this row;
`cadence_report` (reads `train.jsonl`/`rollout_step.jsonl` directly, independent of this
reconstruction) is reported for it instead.

## 4. Where the span went

Per row: span, train-busy seconds, idle seconds, and idle as a share of span, from
`srl.diagnosis.cadence.cadence_report` (wall-clock `end - start` per step, `trainer_idle` from
pre-step gaps). The idle-phase breakdown (solo/interference/queue/reprefill/weight_load, as
shares of idle time, from `diagnose.build()`'s `idle_breakdown_by_binding_phase`) is only
available for the 5 rows that passed the cross-source check.

| row | span | train busy | idle | idle share of span | solo | interference | queue | reprefill | weight_load |
|---|---|---|---|---|---|---|---|---|---|
| `s1_rho1.00_seed1/a_slack` | 8864.70 | 5970.26 | 2894.44 | 32.7% | 54.3% | 44.6% | 0.0% | 1.2% | 0.0% |
| `s1_rho1.00_seed1/b1_cap16` | 9487.65 | 5970.26 | 3517.39 | 37.1% | 49.0% | 49.5% | 0.0% | 1.5% | 0.0% |
| `s2_rho0.90_seed1/a_slack` | 6977.35 | 5374.44 | 1602.91 | 23.0% | 38.9% | 58.5% | 0.0% | 2.7% | 0.0% |
| `s2_rho0.90_seed1/b1_cap32` | 7417.48 | 5407.17 | 2010.30 | 27.1% | 38.2% | 58.9% | 0.0% | 2.9% | 0.0% |
| `s3_rho1.00_seed1/a_slack` | 6635.90 | 5970.26 | 665.64 | 10.0% | 21.5% | 75.6% | 0.0% | 2.9% | 0.0% |
| `s3_rho1.00_seed1/b1_cap16` | 6538.45 | 6046.30 | 492.15 | 7.5% | -- | -- | -- | -- | -- |

(`s3_rho1.00_seed1/b1_cap16`'s span/train-busy/idle are from `cadence_report`; its phase-
breakdown columns are blank because that computation never ran, per section 3.1.)

Two things stand out, both only visible because this redo's cross-source check passed at points
the prior diagnosis's did not:

- **`cadence_report`'s wall-clock `train_busy` is not always what it looks like it should be.**
  At `s1_rho1.00_seed1` it is bit-identical between arms (5970.26276182098, matching to every
  digit `LinearTrainingCost` would predict from identical `consumed_tokens`). At
  `s2_rho0.90_seed1` and `s3_rho1.00_seed1` it differs (32.74s and 76.04s respectively, both
  times the `b1_cap*` control "trains more"). Reading `diagnose.build()`'s
  `train_seconds_total` (`train_time + sync_time + recompute_time`, the model's own *priced*
  training cost, independent of wall clock) resolves this: it is **bit-identical between arms at
  every point checked** -- `5970.26276182098` at `s1_rho1.00_seed1` (both arms) and
  `5374.436485638882` at `s2_rho0.90_seed1` (both arms, confirmed directly from `diagnosis.json`);
  at `s3_rho1.00_seed1`, `a_slack`'s priced cost is `5970.26276182098` -- the same number again
  -- and reading `b1_cap16`'s own raw `train.jsonl` directly (bypassing the failed cross-source
  gate, which does not touch the `train` stream) confirms its priced cost is identical too. Since
  `recompute_time` is `0.0` on every one of these records (`activation_recompute="none"`) and
  `sync_time` is a fixed `0.5s x 24` steps regardless of arm, this is not a coincidence: the
  model's priced training cost is a pure function of total consumed tokens and step count, both
  fixed across arms by construction, so it **cannot** differ between arms at a point. The entire
  wall-clock `train_busy` gap is therefore not a training-cost difference at all.
- **The gap is instead a small number of individual training steps whose own `dt` (wall-clock
  `end - start`) exceeds `train_time + sync_time + recompute_time`.** Per `reference/metrics.md`'s
  own `weight_load` phase definition (`dt - train_time - sync_time - recompute_time` before
  `end`), this residual is exactly what `weight_load` means, even though `weight_load_time=0.0`
  is configured: `_train`'s `_run_engine(deadline=...)` call can settle later than its nominal
  training deadline, and `_publish_weights()` runs on whatever `self.t` that leaves. Direct
  inspection of the raw `train.jsonl` per step: `s1_rho1.00_seed1` has no step with a residual
  above `1e-6` on either arm; `s2_rho0.90_seed1/b1_cap32` has exactly one (version 23, `+32.74s`,
  explaining the whole gap); `s3_rho1.00_seed1/b1_cap16` has exactly two (versions 1 and 2,
  `+51.34s` and `+24.70s`, summing to the whole `76.04s` gap). None of these residuals ever
  occurs on an `a_slack` row in this sample (0 of 3). Section 9 ranks this as its own hypothesis;
  it reframes section 7's loss-point reconciliation below.

## 5. Bounds

Computed for the 5 rows that passed the cross-source check. All 5 bind at `headroom = 1.0`
(`span_solo_all` with and without charging re-prefill, and `span_tail_only`, all equal to the
observed span):

| row | `span_solo_all` (reprefill charged) | `span_solo_all` (reprefill removed) | `span_tail_only` |
|---|---|---|---|
| `s1_rho1.00_seed1/a_slack` | 8864.70 (headroom 1.0) | 8864.70 (headroom 1.0) | 8864.70 (headroom 1.0) |
| `s1_rho1.00_seed1/b1_cap16` | 9487.65 (headroom 1.0) | 9487.65 (headroom 1.0) | 9487.65 (headroom 1.0) |
| `s2_rho0.90_seed1/a_slack` | 6977.35 (headroom 1.0) | 6977.35 (headroom 1.0) | 6977.35 (headroom 1.0) |
| `s2_rho0.90_seed1/b1_cap32` | 7417.48 (headroom 1.0) | 7417.48 (headroom 1.0) | 7417.48 (headroom 1.0) |
| `s3_rho1.00_seed1/a_slack` | 6635.90 (headroom 1.0) | 6635.90 (headroom 1.0) | 6635.90 (headroom 1.0) |

**Plainly: these bounds show no headroom for any mechanism at any of these 5 rows.** Per
`reference/metrics.md`, both bounds hold the consumed set and every training step duration fixed
and ignore fleet capacity entirely; `headroom == 1.0` means the clamp cannot beat the observed
span even with every request decoding alone and fleet capacity removed. This is consistent with
every row here being bound by the critical chain through training-step duration and the
staleness-window admission rule (section 6's at-limit fractions are 89-100% wherever defined),
not by contention a solo-decode counterfactual could remove. As before, a bound of exactly 1.0
is reported at face value per the `memory/lessons.md` entry on this clamp's own failure mode: it
was not re-derived by hand per step to rule out the clamp hiding a genuine >1 headroom.

## 6. Fleet state

Idle-window and instance-imbalance summaries, for the 5 rows with a full reconstruction.

| row | idle windows | mean busy fraction in idle windows | mean router queue depth in idle windows | windows at staleness limit | instance busy_seconds (min / mean / max) | spread as % of mean |
|---|---|---|---|---|---|---|
| `s1_rho1.00_seed1/a_slack` | 16 | 97.4% | 103.88 | 92.3% (12/13) | 6398 / 6809 / 7641 | 18.3% |
| `s1_rho1.00_seed1/b1_cap16` | 17 | 91.8% | 59.43 | 100.0% (16/16) | 6189 / 6686 / 7174 | 14.7% |
| `s2_rho0.90_seed1/a_slack` | 11 | 99.8% | 159.45 | 88.9% (8/9) | 5601 / 5972 / 6423 | 13.8% |
| `s2_rho0.90_seed1/b1_cap32` | 10 | 89.6% | 133.03 | 90.0% (9/10) | 5595 / 5912 / 6087 | 8.3% |
| `s3_rho1.00_seed1/a_slack` | 4 | 99.99% | 332.83 | 33.3% (1/3) | 5593 / 5708 / 5833 | 4.2% |

("windows at staleness limit" is `n/n_defined`, excluding windows whose binding group is not
reserved in a buffer, where the field is `null` rather than `True`/`False`.)

Reading this alongside section 4: in every idle window at every row here, the fleet is already
busy 90-100% of the time -- these are not windows where the fleet sits starved for work that
does not exist yet (`reference/metrics.md`'s `Fleet` distinction). The straggler the trainer
waits on is genuinely long, not starved, at both win points and the loss point alike. At
`s1_rho1.00_seed1`, `b1_cap16` sits at the staleness limit on every one of its windows (100.0%)
while `a_slack` is at 92.3%; at `s2_rho0.90_seed1` the two arms' at-limit fractions are close
(90.0% vs 88.9%). Instance imbalance is present but moderate at every row (spread 4-18% of the
mean) and does not obviously track the arm or the win/loss direction: it is widest at the
biggest win (`s1_rho1.00_seed1/a_slack`, 18.3%) and narrowest at the loss
(`s3_rho1.00_seed1/a_slack`, 4.2%) -- not the pattern a "win comes from a more even fleet"
story would predict. Instance imbalance is not the lever here.

## 7. Paired diff

### 7.1 `s1_rho1.00_seed1` -- the largest win, both arms fully reconstructed, train busy identical

`a_slack` span 8864.70 vs `b1_cap16` span 9487.65: a **622.94s win**. `train_busy` is bit-
identical between arms (5970.26, section 4), so **all 622.94s of the win is idle-time
reduction** -- `trainer_idle` falls from 3517.39 (`b1_cap16`) to 2894.44 (`a_slack`), a drop of
exactly 622.94s.

Idle-time component diff (`a_slack` minus `b1_cap16`, absolute seconds, each arm's own idle
total x its own phase shares):

| component | `a_slack` (s) | `b1_cap16` (s) | diff (s) | share of the 622.94s win |
|---|---|---|---|---|
| interference | 1289.68 | 1740.87 | -451.19 | 72.4% |
| solo | 1570.43 | 1722.78 | -152.35 | 24.5% |
| reprefill | 34.33 | 53.73 | -19.40 | 3.1% |
| queue | 0.00 | 0.00 | 0.00 | 0.0% |
| weight_load | 0.00 | 0.00 | 0.00 | 0.0% |

(Shares sum to 100% of the win by construction, up to rounding.) **The win is dominated by less
interference** -- less time the binding (straggler) request spent slowed down by other requests
sharing its instance, 72.4% -- with a secondary contribution from less solo (intrinsic,
uncontended) cost on whichever request ends up binding, 24.5%, and a small reprefill
contribution, 3.1%.

This agrees with the fleet-state read in section 6: `b1_cap16` sits at the staleness limit on
every one of its 17 idle windows (100.0%) with a lower mean busy fraction inside them (91.8%)
and a much lower mean queue depth (59.43) than `a_slack` (92.3% at-limit, 97.4% busy, 103.88
queue depth). `a_slack` runs a fuller, busier fleet with more work queued and ready during its
idle windows, and that fuller fleet is what cuts the contention its own straggler experiences --
consistent with `slack_lane`'s own design (upgrading near-deadline requests out of contention),
though this diagnosis measures the outcome (interference seconds), not the lane-assignment
decisions themselves.

### 7.2 `s2_rho0.90_seed1` -- the 2nd-largest win, both arms fully reconstructed, train busy differs by one step

`a_slack` span 6977.35 vs `b1_cap32` span 7417.48: a **440.13s win**. Unlike section 7.1,
`train_busy` is *not* bit-identical here: `cadence_report` shows `a_slack` at 5374.44 vs
`b1_cap32` at 5407.17, a 32.74s gap. Section 4 traced this to a single training step
(`b1_cap32`'s version 23) whose own `dt` exceeds its priced `train_time + sync_time +
recompute_time` by exactly 32.74s, while the *priced* training cost itself is bit-identical
between arms (`5374.436485638882` both, direct from `diagnosis.json`). So this is not
`a_slack` training less -- it is `b1_cap32` paying a one-step, unpriced residual `a_slack` never
pays. Treating that residual as what it structurally is (`reference/metrics.md`'s `weight_load`
definition, section 4), the full win decomposes as:

| component | diff (s) | share of the 440.13s win |
|---|---|---|
| idle (`trainer_idle`, pre-step gaps) | -407.39 | 92.6% |
| `b1_cap32`'s single-step unpriced residual | -32.74 | 7.4% |
| **span (sum)** | **-440.13** | **100%** |

Within the 407.39s idle reduction, the same component split as section 7.1 applies (each arm's
own idle total x its own phase shares):

| component | `a_slack` (s) | `b1_cap32` (s) | diff (s) | share of the 440.13s win |
|---|---|---|---|---|
| interference | 937.36 | 1183.55 | -246.19 | 55.9% |
| solo | 622.82 | 767.89 | -145.07 | 33.0% |
| reprefill | 42.73 | 58.44 | -15.71 | 3.6% |
| queue | 0.00 | 0.41 | -0.41 | 0.1% |
| weight_load | 0.00 | 0.00 | 0.00 | 0.0% |

**The mechanism matches section 7.1**: interference reduction is the largest single component
(55.9% of the win), solo reduction second (33.0%), both close in relative proportion to
section 7.1's 72.4%/24.5% split, with the point's higher contention (`mean_concurrency` 9.99 vs
6.15 at `s1_rho1.00_seed1`) plausibly compressing interference's dominance somewhat as solo cost
rises in absolute terms. Fleet state is consistent too: `b1_cap32`'s mean busy fraction inside
its idle windows (89.6%) is lower than `a_slack`'s (99.8%), with a higher mean queue depth under
`a_slack` (159.45 vs 133.03) -- the same "fuller, busier fleet relieves the straggler's own
contention" pattern as section 7.1.

### 7.3 `s3_rho1.00_seed1` -- the loss, control's phase breakdown unavailable, reconciles once the train-busy residual is set aside

`a_slack` span 6635.90 vs `b1_cap16` span 6538.45: `a_slack` is **97.45s slower**, matching the
assigned direction (`speedup_vs_b1_best = 0.9853`). As at `s2_rho0.90_seed1`, `train_busy` is
not bit-identical (`a_slack` 5970.26 vs `b1_cap16` 6046.30, a 76.04s gap) but section 4 traced
this entirely to two of `b1_cap16`'s own steps (versions 1 and 2) exceeding their priced cost by
51.34s and 24.70s -- not to `a_slack` doing less real training work, since `a_slack`'s own
priced cost (`5970.26276182098`) is the same number it is at `s1_rho1.00_seed1`, where both
arms' priced costs tie out bit-for-bit.

Reading `trainer_idle` directly (the honest, well-defined quantity, available regardless of the
control's failed cross-source check): `a_slack` idles **173.48s more** than `b1_cap16`
(665.64 vs 492.15). **This is the clean reconciliation the task asked for at a loss point: the
idle component rises**, by more than the net span loss, because part of what look like
`b1_cap16`'s "faster training" is actually its own unpriced per-step residual inflating its
measured `train_busy` -- not a genuine advantage `a_slack` is failing to match.

| component | diff (s) |
|---|---|
| idle (`trainer_idle`) | +173.48 |
| train busy (wall clock, cadence_report) | -76.04 |
| **span (sum)** | **+97.45** |

The control's cross-source check failure (section 3.1) blocks a paired solo/interference/
reprefill diff at this point. `a_slack`'s own unpaired breakdown is solo 21.5% / interference
75.6% / reprefill 2.9% -- interference dominates far more heavily here than at either win point
(75.6% vs 44.6%/58.5%), consistent with this point's much higher contention
(`mean_concurrency` 13.02 vs 6.15/9.99 at the win points, from section 1's fidelity data).

`fleet_idle_frac` (from `cadence_report`, paired and independent of the failed cross-source
check) is higher under `a_slack` (13.99%) than under `b1_cap16` (13.00%), a **+0.99 percentage-
point gap** -- the opposite sign from both win points (`s1_rho1.00_seed1`: `a_slack` 23.19% vs
`b1_cap16` 29.53%, a_slack lower by 6.33pp; `s2_rho0.90_seed1`: `a_slack` 14.40% vs `b1_cap32`
20.29%, lower by 5.89pp). The sign flips cleanly at the loss point, as it did in the prior
diagnosis of this experiment, though the magnitude here (+0.99pp) is much smaller than that
earlier dataset's (+2.59pp) -- this report does not attempt to reconcile the two numbers (see
section 1), but notes the direction recurs.

## 8. Stops and failures

One row, `s3_rho1.00_seed1/b1_cap16`, passes fidelity but fails the cross-source reconstruction
check (`decode_tokens_match`: 10 of 4872 requests disagree; `reprefill_present`: 11 requests
charged reprefill tokens with no reconstructed reprefill phase -- section 3.1's exact detail).
No idle-phase breakdown, bound, or fleet-window table was computed for it.

This is the same failure shape (a `b1_cap*` control, never `a_slack`, at the point with the
highest contention of the three examined) that the prior diagnosis of this experiment recorded
as open and unresolved (`workflow/analysis/memory/lessons.md`'s 2026-10-27 entry). It recurred
here, on a differently-corrected dataset, which is itself evidence the cause is not specific to
whatever changed between the two sweeps (`enable_trajectory_sticky`, the `t_sat` recalibration)
-- it is some property of `request_num_balance`'s capped admission at high contention, present
under both. It was not re-investigated further here: per this unit's own stop condition ("fix
the simulator's missing sync point, do not widen the tolerance"), root-causing it is out of
scope for an analysis-only unit, and the prior diagnosis already named candidate causes and
falsifying experiments (its section 8.2) that remain untried. No new candidate is added here.

## 9. Ranked hypotheses

Ranked by share of total idle time (or, where the win/loss includes a non-idle component, share
of the total span diff) at the point where each is measured most directly.

1. **Reduced interference on the binding (straggler) request is the dominant driver of both wins.**
   Share of the win: 72.4% at `s1_rho1.00_seed1` (section 7.1, paired, component-level), 55.9%
   at `s2_rho0.90_seed1` (section 7.2, paired, component-level). Measurement:
   `binding_decompositions[rid].interference_seconds`, aggregated across steps by
   `idle_breakdown_by_binding_phase`. Falsifying experiment: a `validate` sweep point at either
   of these two points that forces `a_slack`'s lane-upgrade logic off (`strategy_args=
   {"enable_upgrade": False}`, the existing `a_nolane` arm in `cadence_lanes.py`'s `ARMS`) -- if
   interference does not rise back toward the `b1_cap*` control's level, the lane-upgrade
   mechanism is not what is cutting it.
2. **Reduced solo (intrinsic, uncontended) cost on the binding request is a secondary driver at
   both win points.** Share of the win: 24.5% at `s1_rho1.00_seed1`, 33.0% at
   `s2_rho0.90_seed1` (same measurements as hypothesis 1). Falsifying experiment: the same
   `a_nolane` points -- if the binding request's identity (which `rid` is binding at each step)
   does not change between `a_slack` and `a_nolane`, this component should not move either,
   since it is driven by which request becomes the tail, not by contention on it.
3. **At the loss point (`s3_rho1.00_seed1`), trainer idle itself rises** (+173.48s, section 7.3,
   paired and directly measured, independent of the failed cross-source check), **and this is
   larger than the net span loss (+97.45s) once a confound is set aside.** The confound: part of
   `b1_cap16`'s apparent "faster training" at this point (and at `s2_rho0.90_seed1`) is a one-
   or-two-step unpriced wall-clock residual never seen on `a_slack` (hypothesis 4 below), not a
   genuine training-cost advantage. Falsifying experiment: a `validate` sweep varying `rho` and
   `staleness` with `a_slack` vs `b1_best`, tracking `trainer_idle` (not raw span) per arm at
   every point -- if `trainer_idle(a_slack) - trainer_idle(b1_best)` does not track the sign of
   `speedup_vs_b1_best` more cleanly than raw span does across the grid, this reframing does not
   generalize past these 3 points.
4. **The `request_num_balance`-capped control pays a one-or-two-step unpriced wall-clock residual
   that `a_slack` never pays, confirmed at 2 of 3 points (`s2_rho0.90_seed1`: +32.74s, one step;
   `s3_rho1.00_seed1`: +76.04s, two steps; `s1_rho1.00_seed1`: none).** Measured directly from
   each row's own raw `train.jsonl` (`dt` vs `train_time + sync_time + recompute_time`), cross-
   checked against `diagnosis.json`'s `train_seconds_total` being bit-identical between arms at
   both points where both arms' values are available (section 4). Per `reference/metrics.md`'s
   own `weight_load` phase definition, this residual is structurally a weight-load stretch
   (`_run_engine(deadline=...)` settling past its nominal deadline before `_publish_weights()`
   runs), even under `weight_load_time=0.0`. Share of idle/span: 7.4% of the `s2_rho0.90_seed1`
   win, and all of the `-76.04s` "train busy" gap at `s3_rho1.00_seed1`'s loss once reframed as
   not training-cost. Falsifying experiment: instrument `Simulator._run_engine`'s deadline-
   exceeded branch to log which instance/request was still in flight when each publish's nominal
   deadline passed, across a `b1_cap*` vs `a_slack` pair at the same point -- if the in-flight
   request is not consistently one `enable_trajectory_sticky` has pinned to an instance that is
   mid-decode at the deadline (plausible given `request_num_balance`/`b1_cap*` arms are the ones
   the task confirmed are sensitive to that flag, while `a_slack` is not), this mechanism does
   not explain why the residual never appears on `a_slack`.
5. **The cross-source check failure on `request_num_balance`-capped controls at high contention
   recurs across two differently-regenerated sweeps of this same experiment, unrelated to either
   correction applied between them.** Not an idle-time hypothesis -- a data-quality hypothesis
   about this diagnosis unit's own reconstruction. Evidence: section 8's recurrence note.
   Falsifying experiment: the prior diagnosis's own section 8.2 table (re-run `b1_cap16` with
   `n_versions` raised past 24, or re-run `b0_default` uncapped at the same point) -- untried in
   both diagnoses of this experiment so far.

## 10. Not established

- **The component-level mechanism at the loss point (`s3_rho1.00_seed1`) is not measured, only
  read qualitatively from the treatment's own unpaired breakdown.** The control's reconstruction
  fails the cross-source check there (section 3.1), so there is no paired solo/interference/
  reprefill diff -- only at the two win points (sections 7.1, 7.2) is the full component-level
  story measured rather than inferred.
- **Hypothesis 4's mechanism (which specific in-flight request causes the unpriced wall-clock
  residual) is a plausible reading of a precisely isolated number, not a traced-through
  simulator mechanism.** The residual's magnitude, step, and arm are measured exactly; which
  instance/request was in flight at that moment, and whether it is specifically tied to
  `enable_trajectory_sticky`'s pinning, is not -- section 9's falsifying experiment for it was
  not run.
- **The cross-source check failure recurs across two differently-regenerated sweeps but is
  still not root-caused**, in either diagnosis of this experiment. Section 8's falsifying
  experiments (carried over from the prior diagnosis) remain untried.
- **Bounds show headroom = 1.0 at face value (section 5), not re-derived by hand.** Per the
  `memory/lessons.md` entry on `span_tail_only`/`span_solo_all` clamping, a bound reporting
  exactly 1.0 can reflect a genuine absence of headroom or a clamp artifact; distinguishing the
  two needs re-running the chain by hand per step, which this diagnosis did not do.
- **Single seed, single workload, 3 of 27 points.** All 3 points here are `seed=1`; no point
  from `seed=0`, `seed=2`, or `staleness=2` combined with either win point's `rho` was examined.
  The `fleet_idle_frac` sign-flip at the loss point (section 7.3) and hypothesis 4's residual
  pattern are both drawn from exactly 3 points and explicitly need the grid-wide falsifying
  experiments named in section 9 before they generalize.
