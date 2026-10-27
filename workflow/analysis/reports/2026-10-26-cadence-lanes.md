# Diagnosis report

## 1. What was diagnosed

Experiment: `workflow/validate/experiments/cadence_lanes.py` (gitignored driver, read for
`build(spec, output_dir=None)`). Result set: `outputs/simulate/cadence_lanes/summary.json`
(540 rows, 486 gated), Task 8's 27-point x 20-arm sweep comparing the `slack_lane` router
strategy (arm `a_slack`) against the strongest gated static baseline per point (`b1_best`, a
`request_num_balance` cap sweep).

Rows selected, as assigned (3 points x 2 rows, already computed from the gated data, not
re-derived here):

| point | reason | treatment row | control row |
|---|---|---|---|
| `s1_rho0.90_seed1` | largest `a_slack` win vs `b1_best` (speedup_vs_b1_best = 1.0639) | `s1_rho0.90_seed1/a_slack` | `s1_rho0.90_seed1/b1_cap16` |
| `s1_rho1.00_seed1` | 2nd largest `a_slack` win (speedup_vs_b1_best = 1.0612) | `s1_rho1.00_seed1/a_slack` | `s1_rho1.00_seed1/b1_cap16` |
| `s3_rho1.10_seed0` | smallest `a_slack` speedup, the biggest loss (speedup_vs_b1_best = 0.9826) | `s3_rho1.10_seed0/a_slack` | `s3_rho1.10_seed0/b1_cap16` |

Workload: real Qwen3-8B DAPO trace (`outputs/workload/qwen3_8b_dapo_38912_tp1/joint.jsonl`,
permuted per seed), composed TP1 H20 rollout cost model, 16 instances, 24 training steps, PSRL
policy, `backend="native"`. `s1`/`s3` are psrl staleness horizons 1 and 3; `rho` is training
speed relative to the fleet's own rollout-bound saturation rate.

**This report has two layers.** The first attempt at this diagnosis (same day, earlier revision)
found that every one of the 6 assigned rows failed `replay.fidelity()`, root-caused that to a
config default that changed on disk mid-sweep, and stopped there per this unit's own
non-negotiable rule. The coordinator found and fixed the actual cause (`SimConfig
.ingress_lifetime_cap`, see section 8.1) and asked for the diagnosis to be redone. This revision
redoes it: all 6 rows now pass `replay.fidelity()`. Of those 6, 4 also pass the separate
cross-source reconstruction check inside `diagnose.build()` and carry a full idle-phase
breakdown, bounds, and fleet-window table; the other 2 (both `b1_cap16` controls, at the
2nd-win and loss points) pass fidelity but fail that reconstruction check, so they carry
`cadence_report`'s trainer-idle/fleet-idle split (which does not depend on the failed
reconstruction) but not the finer breakdown. Section 8 keeps a full record of both failure
episodes, since both are real, reportable parts of this diagnosis's own story.

## 2. Session

Git head at the start of this revision: `942c66b225c75d51e14aa7bfe7708cee5e228185`; the
coordinator's fix landed as `5582acb` (not authored by this unit, touches
`workflow/validate/experiments/cadence_lanes.py`, which is gitignored, plus
`workflow/validate/memory/lessons.md` and `workflow/validate/reports/2026-10-26-cadence-lanes.md`
-- a different report from this one). Native backend binary
(`srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so`) md5: `4c9a6244aeaf872809771252a83e7ad6`,
mtime 2026-10-17, unchanged across both revisions of this diagnosis. Preflight: passed, no
blocking or warning entries, both times. Replay wall time per row (single-arm replay through
`replay.run`, 24 training steps, telemetry on): `s1_rho0.90_seed1/a_slack` 23s,
`s1_rho0.90_seed1/b1_cap16` 22s, `s1_rho1.00_seed1/a_slack` 24s, `s1_rho1.00_seed1/b1_cap16` 23s,
`s3_rho1.10_seed0/a_slack` 36s, `s3_rho1.10_seed0/b1_cap16` 30s (unchanged by the fix, since it
only changed which simulator config was built, not how long a run takes).

## 3. Fidelity

**All 6 rows pass.** Per `workflow/analysis/scripts/replay.py`'s `fidelity()` (1e-9 relative for
floats, exact for counts), every field matches bit-for-bit on every row once
`build()` pins `ingress_lifetime_cap=True` (the fix committed in `5582acb`). Representative
table, `s1_rho0.90_seed1/a_slack`:

| field | ok | row | replay |
|---|---|---|---|
| span | pass | 8313.823089637575 | 8313.823089637575 |
| consumed_tokens | pass | 65238857 | 65238857 |
| consumed_groups | pass | 768 | 768 |
| tokens_per_group | pass | 84946.42838541667 | 84946.42838541667 |
| mean_concurrency | pass | 6.52806430206081 | 6.52806430206081 |
| peak_concurrency | pass | 42 | 42 |
| preemptions | pass | 0 | 0 |
| versions_done | pass | 24 | 24 |
| dropped | pass | 0 | 0 |

All 9 fields pass, identically, for the other 5 rows (`s1_rho0.90_seed1/b1_cap16`,
`s1_rho1.00_seed1/a_slack`, `s1_rho1.00_seed1/b1_cap16`, `s3_rho1.10_seed0/a_slack`,
`s3_rho1.10_seed0/b1_cap16`) -- full field-by-field values are in each row's
`outputs/analysis/cadence_lanes/<row>/fidelity.json` (for the 2 rows described in section 3.1)
or baked into the committed `diagnosis.json` (for the other 4).

### 3.1 A second, separate gate: the cross-source reconstruction check

Fidelity is not the only gate this unit runs. `diagnose.build()` also requires
`srl.diagnosis.timeline.request_phases`'s 4 cross-source checks to pass before it will compute
anything (`decode_tokens_match`, `completion_time_match`, `abort_count_match`,
`reprefill_present` -- see `reference/metrics.md`'s phase definitions). Of the 6 fidelity-passing
rows, 2 fail this second gate:

- `s1_rho1.00_seed1/b1_cap16`: `decode_tokens_match` -- 1 of 5850 uninterrupted requests
  disagree (`rid=3327`: reconstructed decode tokens 5761 vs recorded `generated` 7471).
- `s3_rho1.10_seed0/b1_cap16`: all of `decode_tokens_match` (12 of 4739 requests),
  `completion_time_match` (4 requests), and `reprefill_present` (31 requests) fail together.

Both failures are on the `b1_cap16` control, never on `a_slack`, at exactly the two points
where the control is the row with the **smaller** span of the pair (the win points) or
comparable contention (the loss point) -- see section 8.2 for what was checked about this
pattern and what remains unestablished. Per this unit's own rule, no idle-phase breakdown,
bound, or fleet-window table was computed for these 2 rows: building one on a reconstruction
that disagrees with the streams would describe phases that did not happen.
`cadence_report` (`srl.diagnosis.cadence`) does **not** depend on this reconstruction -- it
reads `train.jsonl` and `rollout_step.jsonl` directly -- so it is reported for all 6 rows
including these 2.

## 4. Where the span went

Per row: span, train-busy seconds (`train_seconds_total` / `cadence_report.train_busy`, which
agree to the row's own float precision wherever both are available), idle seconds, and idle as a
share of span. The idle-phase breakdown (by the binding request's phases, as shares of idle
time) is only available for the 4 rows that passed the cross-source check.

| row | span | train busy | idle | idle share of span | solo | interference | queue | reprefill | weight_load |
|---|---|---|---|---|---|---|---|---|---|
| `s1_rho0.90_seed1/a_slack` | 8313.82 | 5148.18 | 3165.65 | 38.1% | 53.9% | 44.8% | 0.0% | 1.3% | 0.0% |
| `s1_rho0.90_seed1/b1_cap16` | 8844.82 | 5148.18 | 3696.65 | 41.8% | 50.6% | 48.7% | 0.0% | 0.7% | 0.0% |
| `s1_rho1.00_seed1/a_slack` | 8602.62 | 5718.86 | 2883.75 | 33.5% | 54.3% | 44.5% | 0.0% | 1.1% | 0.0% |
| `s1_rho1.00_seed1/b1_cap16` | 9128.98 | 5718.86 | 3410.12 | 37.3% | -- | -- | -- | -- | -- |
| `s3_rho1.10_seed0/a_slack` | 6618.67 | 5981.82 | 636.85 | 9.6% | 22.9% | 75.7% | 0.0% | 1.4% | 0.0% |
| `s3_rho1.10_seed0/b1_cap16` | 6503.18 | 6079.77 | 423.41 | 6.5% | -- | -- | -- | -- | -- |

(`b1_cap16` rows at `s1_rho1.00_seed1` and `s3_rho1.10_seed0` have span, train busy, and idle
from `cadence_report`; their phase-breakdown columns are blank because that computation never
ran, not because it ran and found nothing -- see section 3.1.)

Two things stand out before the paired diffs in section 7:

- **Train busy is identical between arms at both win points** (`5148.18` at
  `s1_rho0.90_seed1`, `5718.86` at `s1_rho1.00_seed1`, matching to every digit `cadence_report`
  prints). `LinearTrainingCost.predict` (`srl/cost/training/linear.py`) is
  `time_per_token * workload.total_tokens`, purely additive over tokens, and
  `consumed_tokens`/`consumed_groups` already match exactly between arms (section 3's fidelity
  table), so this is expected, not a coincidence: whichever arm wins at these two points, it wins
  entirely through less idle time, never through doing less training work.
- **Train busy differs by ~98s at the loss point** (`a_slack` trains *less*, 5981.82 vs
  6079.77), which the two win points do not show. This sits next to the only point where
  `b1_cap16`'s cross-source check fails on 3 of 4 checks at once, including 31 requests charged
  reprefill tokens with no reconstructed reprefill phase -- `a_slack`'s own reconstruction at
  this point has no such anomalies. Section 9's hypothesis 4 names this as an open, unresolved
  question rather than a mechanism this report claims to have nailed down.

## 5. Bounds

Computed for the 4 rows that passed the cross-source check. All 4 bind at `headroom = 1.0`
(`span_solo_all` with and without charging re-prefill, and `span_tail_only`, all equal to the
observed span):

| row | `span_solo_all` (reprefill charged) | `span_solo_all` (reprefill removed) | `span_tail_only` |
|---|---|---|---|
| `s1_rho0.90_seed1/a_slack` | 8313.82 (headroom 1.0) | 8313.82 (headroom 1.0) | 8313.82 (headroom 1.0) |
| `s1_rho0.90_seed1/b1_cap16` | 8844.82 (headroom 1.0) | 8844.82 (headroom 1.0) | 8844.82 (headroom 1.0) |
| `s1_rho1.00_seed1/a_slack` | 8602.62 (headroom 1.0) | 8602.62 (headroom 1.0) | 8602.62 (headroom 1.0) |
| `s3_rho1.10_seed0/a_slack` | 6618.67 (headroom 1.0) | 6618.67 (headroom 1.0) | 6618.67 (headroom 1.0) |

**Plainly: these bounds show no headroom for any mechanism at any of these 4 rows.** Per
`reference/metrics.md`, both bounds hold the consumed set and every training step duration fixed
and ignore fleet capacity entirely; `headroom == 1.0` means the chain this clamp builds cannot
beat the observed span even with every request decoding alone and fleet capacity removed. That
is consistent with these runs being bound by the critical chain through the training step's own
duration and the staleness-window admission rule (`at_staleness_limit_fraction` in section 6 is
0.94-1.0 for every row with an idle-window table), not by contention a solo-decode
counterfactual could remove. A bound of exactly 1.0 is reported at face value here, per the
`memory/lessons.md` entry on `span_tail_only`/`span_solo_all` clamping: it was not re-derived by
hand per step to rule out the clamp hiding a genuine >1 headroom, which would need its own pass.

## 6. Fleet state

Idle-window and instance-imbalance summaries, for the 4 rows with a full reconstruction. (The
2 cross-source-check-failed rows have no idle-window table; `cadence_report`'s `fleet_idle_frac`
for all 6 rows is in section 4/7.)

| row | idle windows | mean busy fraction in idle windows | mean router queue depth in idle windows | windows at staleness limit | instance busy_seconds (min / mean / max) |
|---|---|---|---|---|---|
| `s1_rho0.90_seed1/a_slack` | 19 | 95.4% | 100.91 | 94.1% | 6472 / 6810 / 7399 |
| `s1_rho0.90_seed1/b1_cap16` | 18 | 92.8% | 59.77 | 100.0% | 6225 / 6588 / 7099 |
| `s1_rho1.00_seed1/a_slack` | 17 | 97.7% | 101.43 | 94.1% | 6539 / 6900 / 7643 |
| `s3_rho1.10_seed0/a_slack` | 4 | 99.3% | 329.17 | 75.0% | 5223 / 5627 / 5854 |

Reading this alongside section 4: during an idle window, the fleet is already busy 93-99% of
the time at every row here -- these are not windows where the fleet sits idle waiting for work
that does not exist yet (`reference/metrics.md`'s distinction in the `Fleet` section). The
straggler the trainer is waiting on is genuinely long, not starved. `b1_cap16` at the win point
has every one of its idle windows (100.0%) at the staleness limit (its binding group's
reservation cannot move to a later buffer); `a_slack` relieves that slightly, to 94.1%, at both
win points it has a full reconstruction for.

Instance imbalance is present but moderate at every row (busy-seconds spread 9-14% of the mean),
and does not obviously track the arm: `a_slack`'s spread is similar to or slightly wider than
`b1_cap16`'s at the one point with both (`s1_rho0.90_seed1`: `a_slack` spread 926s / 13.6% of
mean vs `b1_cap16` spread 874s / 13.3% of mean). Instance imbalance is not the lever here.

## 7. Paired diff

### 7.1 `s1_rho0.90_seed1` -- the largest win, both arms fully reconstructed

`a_slack` span 8313.82 vs `b1_cap16` span 8844.82: a 530.999s win. `train_busy` is identical
between arms (5148.18, section 4), so **all 531s of the win is idle-time reduction** --
`trainer_idle` falls from 3696.65 (`b1_cap16`) to 3165.65 (`a_slack`), a drop of exactly 530.999s.

Idle-time component diff (`a_slack` minus `b1_cap16`, in absolute seconds, computed from each
arm's own idle total x its own phase shares):

| component | `a_slack` (s) | `b1_cap16` (s) | diff (s) | share of the 531s win |
|---|---|---|---|---|
| interference | 1419.4 | 1801.7 | -382.4 | 72.0% |
| solo | 1705.3 | 1868.6 | -163.4 | 30.7% |
| reprefill | 41.0 | 26.6 | +14.4 | -2.7% |
| queue | 0.0 | 0.0 | 0.0 | 0.0% |
| weight_load | 0.0 | 0.0 | 0.0 | 0.0% |

(Shares sum to 100% of the 531s diff by construction, up to float rounding.) **The win is
dominated by less interference** -- less time the binding (straggler) request spent slowed down
by other requests sharing its instance, 72% of the total -- with a secondary contribution from
less solo (intrinsic, uncontended) decode/prefill cost on whichever request ends up binding,
31%. A small, opposite-signed reprefill increase (-2.7%, i.e. `a_slack` pays slightly *more*
reprefill cost on its binding request) is the only component working against the win, and is an
order of magnitude smaller than either of the other two.

This agrees with the fleet-state read in section 6: `b1_cap16` is pinned at the staleness limit
on every one of its 18 idle windows (100.0%), with a lower mean busy fraction inside those
windows (92.8%) and a much lower mean queue depth (59.77) than `a_slack` (94.1% at-limit, 97.7%
busy... wait, 95.4%, and 100.91 queue depth). Read together: `a_slack` is running a fuller,
busier fleet with more work queued and ready during its (shorter, less frequent per-window but
similar in count) idle windows, and that fuller fleet is what cuts the contention
(`interference`) its own straggler experiences -- the opposite of the naive expectation that a
busier fleet would mean *more* contention. `slack_lane`'s own design (the arm, not something
measured end-to-end here) is to upgrade near-deadline requests out of contention; this is
consistent with that mechanism working, though this diagnosis measures the outcome
(interference seconds), not the lane-assignment decisions themselves.

### 7.2 `s1_rho1.00_seed1` -- the 2nd-largest win, control's phase breakdown unavailable

`a_slack` span 8602.62 vs `b1_cap16` span 9128.98: a 526.365s win. `train_busy` is again
identical between arms (5718.86), so again **all of the win is idle-time reduction**:
`trainer_idle` falls from 3410.12 (`b1_cap16`) to 2883.75 (`a_slack`), a drop of exactly
526.365s -- matching the span diff to the same precision as section 7.1's point.

The win's *direction and size* reconcile cleanly with Task 8 (trainer idle falls, by exactly the
amount the span falls, with training cost unchanged). What cannot be shown at this point is the
*component* breakdown done in section 7.1, because `b1_cap16`'s cross-source check failed here
(section 3.1) -- there is no `b1_cap16` idle-phase decomposition to diff against. `a_slack`'s
own (unpaired) breakdown at this point is solo 54.3% / interference 44.5% / reprefill 1.1% /
queue 0.0%, closely matching its own breakdown at the other win point (53.9% / 44.8% / 1.3% /
0.0%, section 7.1) -- consistent with, but not proof of, the same interference-dominated
mechanism recurring here. This is reported as a qualitative match, not a measured one.

### 7.3 `s3_rho1.10_seed0` -- the loss, control's phase breakdown unavailable

`a_slack` span 6618.67 vs `b1_cap16` span 6503.18: `a_slack` is **115.49s slower**, matching
Task 8's direction (`a_slack` lost, `speedup_vs_b1_best = 0.9826`). Unlike both win points,
`train_busy` is *not* identical here: `a_slack` trains for 97.95s *less* than `b1_cap16`
(5981.82 vs 6079.77). The loss decomposes as:

| component | `a_slack` minus `b1_cap16` (s) |
|---|---|
| idle (`trainer_idle`) | +213.44 |
| train busy | -97.95 |
| **span (sum)** | **+115.49** |

So **the idle component rose, as Task 9's reconciliation asked this diagnosis to show for a
loss point** -- `a_slack` idles 213.44s more than `b1_cap16` at this point -- but roughly 46% of
that idle increase is clawed back by `a_slack` training for less, netting the measured 115.49s
loss. As in section 7.2, the control's cross-source check failed here (and more severely: 3 of
4 checks, section 3.1), so the idle *component* breakdown (solo/interference/reprefill) cannot
be diffed paired; only `a_slack`'s own unpaired breakdown is available (solo 22.9% /
interference 75.7% / reprefill 1.4%), which shows interference dominating far more heavily here
than at either win point (75.7% vs ~45%) -- expected, since `mean_concurrency` at this point
(12.6/9.9, from the fidelity table) is roughly double what it is at the `s1` points (6.5/6.1-6.2).

One paired, directly-measured (not reconstruction-dependent) fact *is* available at this point:
`fleet_idle_frac` (from `cadence_report`, reading `rollout_step.jsonl` directly, independent of
the failed cross-source check) is higher under `a_slack` (14.98%) than under `b1_cap16`
(12.39%), a +2.59 percentage-point gap. At the two win points this same quantity moves the other
way or is comparable (`s1_rho0.90_seed1`: `a_slack` 18.09% vs `b1_cap16` 25.52%, `a_slack`
*lower*). Read together with the idle-component story in section 7.1 (where a busier fleet under
`a_slack` is what buys the win), the pattern at this highest-contention point is the mirror
image: `a_slack` leaves *more* aggregate fleet capacity idle here, not less, and that is where it
loses. This is consistent with a lane-reservation mechanism that costs aggregate throughput when
reserved capacity could otherwise be doing useful work everywhere (the regime this experiment's
own `s3`/high-`rho` corner puts the fleet in), mirroring the general shape of the
`zero-marginal-value-is-not-a-reason-to-refuse-work` and `lpt-admission-loses-on-real-vllm`
lessons already in `workflow/analysis/memory/lessons.md` for other mechanisms -- named as a
hypothesis in section 9, not asserted as proven, since the finer decomposition that would nail
it down is exactly what the cross-source check failure blocks.

## 8. Stops and failures

### 8.1 Episode 1 (earlier revision of this report): fidelity gate failure, root-caused and fixed

The first attempt at this diagnosis found all 6 rows failing `replay.fidelity()` (4 of 6 also on
`span`, all 6 on `mean_concurrency`). Four candidate causes were tested directly and ruled out:
telemetry on/off (identical mismatched numbers either way), `PYTHONHASHSEED` pinned vs. unset
(4 fresh-process launches agreed with each other to the last digit), the native `.so` being
rebuilt since the row was captured (same mtime/md5 as 10 days before the sweep ran), and
same-process arm-sequence depth (the mismatch appeared on the very first arm run in a fresh
process, not growing with how many arms preceded it). The actual cause, found and fixed by the
coordinator (commit `5582acb`): a concurrent session's uncommitted edit to
`srl/simulator/config.py` added `SimConfig.ingress_lifetime_cap` (default `False`) with a
file mtime inside the sweep's own execution window. The sweep ran as one long-lived process that
had already imported `srl` before that edit landed, so its 540 rows are internally self-
consistent under the *old*, unconditional ingress cap; this unit's replay was a fresh process
that picked up the *new* on-disk default, a materially different simulator. Pinning
`ingress_lifetime_cap=True` explicitly in `cadence_lanes.py`'s `build()` reproduces all 6 rows
bit-for-bit (section 3). This episode is kept here in full, per the coordinator's own
instruction, rather than erased now that it is resolved.

### 8.2 Episode 2 (this revision): cross-source reconstruction check failure, open

`s1_rho1.00_seed1/b1_cap16` and `s3_rho1.10_seed0/b1_cap16` pass fidelity but fail the separate
cross-source check (section 3.1's exact failures). Three observations, reported rather than
explained (consistent with `reference/metrics.md`'s "an unexplained wait is reported, never
attributed"):

- Both failures are on a `b1_cap16` control, never on `a_slack`, across the 2 points where it
  happens (of the 4 `b1_cap16` control rows checked in total, counting `s1_rho0.90_seed1` where
  it passed cleanly -- 2 of 3 `b1_cap16` rows fail, 0 of 3 `a_slack` rows do, not a large enough
  sample to call a pattern).
- Severity differs sharply between the 2 failing rows: 1 check type / 1 request at
  `s1_rho1.00_seed1`, vs. 3 check types / up to 31 requests at `s3_rho1.10_seed0` -- the same
  point where `train_busy` differs between arms (section 4), and the point with by far the
  highest contention of the 3 (`mean_concurrency` 9.9-12.6 vs 5.9-6.5 at the `s1` points).
  `reprefill_present`'s failure mode -- a request charged reprefill tokens with no reconstructed
  reprefill phase -- is consistent with (but not proven to be caused by) an abort/retry event
  near the run's own stop boundary (`n_versions=24` cutting off mid-attempt) whose telemetry
  record is incomplete, which would plausibly show up more at a higher-contention point with more
  retries in flight, but this was not traced down to a specific missing sync point in the
  simulator.
- This episode was not root-caused as thoroughly as episode 1: the ingress-cap fix was proven by
  reproducing all 6 rows bit-for-bit, a strong, direct test. This failure was observed, and one
  plausible class of cause was named above, but not tested the way episode 1's candidates were
  (e.g. by inspecting the specific requests' raw `instance/N` and `request` stream records around
  their abort/reprefill events to find the missing field or sync point). Per this unit's own
  stop condition ("fix the simulator's missing sync point, do not widen the tolerance"), that
  fix is out of scope for this analysis-only unit; it is named here so a future pass does not
  have to re-discover that it exists.

**Falsifying / follow-up experiments for episode 2:**

| Candidate | Falsifying experiment |
|---|---|
| Telemetry record is incomplete for a request whose final attempt is still in flight when `n_versions` training steps complete | Re-run `s3_rho1.10_seed0/b1_cap16` with `n_versions` raised well past 24 (or inspect whichever of the 12 flagged `rid`s is latest-admitted) and check whether the same `rid`s still disagree once they are not near the run's own stop boundary |
| A genuine missing sync point around abort/reprefill telemetry, present regardless of run length | Pull the raw `instance/N` and `request` stream records for one flagged `rid` (e.g. 619) directly and manually reconstruct its phases by hand; compare against what `request_phases` produced |
| The failure is specific to `request_num_balance`'s capped admission interacting with `ingress_lifetime_cap=True` | Re-run `b0_default` (uncapped) at the same two points through the now-fixed `build()` and check whether its cross-source checks also fail |

## 9. Ranked hypotheses

Ranked by share of total idle time where that share is measured (section 7.1's paired diff);
where only an unpaired or coarser measurement is available, that is stated rather than a false
share reported.

1. **Reduced interference on the binding (straggler) request is the dominant driver of the win
   at `s1_rho0.90_seed1`.** Share of the win: 72.0% (section 7.1, a paired, component-level
   measurement). Measurement: `binding_decompositions[rid].interference_seconds`, aggregated
   across steps by `idle_breakdown_by_binding_phase`. Falsifying experiment: a `validate` sweep
   point that forces `a_slack`'s lane-upgrade logic off (`strategy_args={"enable_upgrade":
   False}`, the existing `a_nolane` arm already in `cadence_lanes.py`'s `ARMS`) at this same
   point -- if interference does not rise back toward `b1_cap16`'s level, the lane-upgrade
   mechanism is not what is cutting it.
2. **Reduced solo (intrinsic, uncontended) cost on the binding request is a secondary driver at
   the same point.** Share of the win: 30.7% (same measurement and point as above). Falsifying
   experiment: same `a_nolane` point: if the binding request's identity (which `rid` is binding
   at each step) does not change between `a_slack` and `a_nolane`, this component should not
   move either, since it is driven by which request becomes the tail, not by contention on it.
3. **The same interference-dominated mechanism recurs at `s1_rho1.00_seed1`.** Not a measured
   share -- `a_slack`'s own unpaired breakdown (44.5% interference, 54.3% solo) is close to its
   breakdown at the other win point (44.8%, 53.9%), but there is no paired control breakdown to
   diff (section 3.1, section 7.2). Falsifying experiment: fix episode 2's cross-source check
   failure (section 8.2's table) for `s1_rho1.00_seed1/b1_cap16` and re-run this diagnosis; if the
   component shares do not land close to section 7.1's, this hypothesis is wrong.
4. **At the loss point (`s3_rho1.10_seed0`), `a_slack` leaves more aggregate fleet capacity idle
   than `b1_cap16`, and that is where it loses.** Measured, paired, and independent of the
   cross-source check: `fleet_idle_frac` 14.98% (`a_slack`) vs. 12.39% (`b1_cap16`), a +2.59pp
   gap, the opposite sign from the win points. This is named as the leading hypothesis for *why*
   idle time rose here (not fully decomposed into solo/interference/reprefill, since that would
   need episode 2's cross-source check fixed first). Falsifying experiment: a `validate` sweep
   varying `rho` and `staleness` with `a_slack` vs `b1_best`, tracking `fleet_idle_frac` per arm
   at every point -- if the sign of `fleet_idle_frac(a_slack) - fleet_idle_frac(b1_best)` does
   not flip from negative (win points, lower `rho`/looser `staleness`) to positive (this loss
   point, higher `rho`/tighter `staleness`) across the grid, this hypothesis does not generalize
   past this one point.
5. **The ~98s train-busy gap at the loss point is driven by the same contention/abort activity
   that fails `b1_cap16`'s cross-source check there.** Not measured directly -- both facts
   (train-busy gap, cross-source check severity) co-occur only at this one point, which is
   suggestive, not proof. Falsifying experiment: episode 2's first falsifying experiment
   (section 8.2's table) plus a direct trace of `workload.total_tokens` per training step for
   both arms at this point, checking whether the gap tracks the number of requests with
   `reprefill_tokens > 0` per step.

## 10. Not established

- **The component-level mechanism at 2 of the 3 points (`s1_rho1.00_seed1`, `s3_rho1.10_seed0`)
  is not measured, only read qualitatively from the treatment's own unpaired breakdown.** The
  control's reconstruction fails the cross-source check at both points (section 3.1), so there is
  no paired solo/interference/reprefill diff at either -- only at `s1_rho0.90_seed1` (section
  7.1) is the full component-level story measured rather than inferred by similarity.
- **Episode 2's cross-source check failure is observed and partially characterized, not
  root-caused.** One plausible class of cause (telemetry incomplete at the run's own stop
  boundary) is named with 3 falsifying experiments (section 8.2); none of the three was run. This
  is a materially weaker root-cause than episode 1's (which was proven by reproducing all 6 rows
  bit-for-bit).
- **Bounds show headroom = 1.0 at face value (section 5), not re-derived by hand.** Per the
  `memory/lessons.md` entry on `span_tail_only`/`span_solo_all` clamping, a bound reporting
  exactly 1.0 can reflect a genuine absence of headroom or a clamp artifact; distinguishing the
  two needs re-running the chain by hand per step, which this diagnosis did not do.
- **Single seed, single workload, 3 of 27 points.** `seed=0` supplies the loss point, `seed=1`
  supplies both win points; no point from `seed=2` or from `staleness=2` was examined. The
  `fleet_idle_frac` sign-flip hypothesis (section 9, hypothesis 4) is drawn from exactly 3 points
  and explicitly needs the grid-wide falsifying experiment named there before it generalizes.
- **The `fleet_idle_frac` sign-flip (hypothesis 4) is a plausible reading of a real, measured
  number, not a proven causal mechanism.** It was not tied back to specific lane-assignment
  decisions inside `slack_lane`'s own logic (which requests it holds back, and why); it is an
  outcome-level observation.
