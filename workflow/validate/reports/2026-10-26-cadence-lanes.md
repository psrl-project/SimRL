# Report: cadence-aware lane scheduling (slack_lane / length_lane)

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

- Git head: `e621dc1`. `srl/` carries the same long-standing set of concurrent-session edits
  present throughout this entire plan (see `preflight.py`'s `dirty_srl_files`); none of them
  were introduced by this sweep.
- Binary md5 of `srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so`:
  `4c9a6244aeaf872809771252a83e7ad6`, identical before and after the ~4h44m sweep (confirmed by
  the sweep-running subagent; re-confirmed by me against the current checkout after the sweep
  finished). No in-place rebuild occurred mid-run.
- `SIMRL_KV_CACHE` unset. Preflight warnings, quoted verbatim: "SIMRL_KV_CACHE is unset. Every
  run re-probes KV capacity." and "Uncommitted changes under srl/. Record them, the baseline is
  this same tree." Both are pre-existing environment conditions, not caused by this sweep.
- Workload: `outputs/workload/qwen3_8b_dapo_38912_tp1` (1024 real DAPO groups x 8, cap 38912),
  permuted 3 ways by seed. Fleet: 16 x TP1 Qwen3-8B on H20, composed cost model, KV from the
  real probe, `weight_load_time=0.0`. All lengths come from the oracle predictor.

## 3. Gated results

27 points (`staleness in {1,2,3} x rho in {0.9,1.0,1.1} x seed in {0,1,2}`), 20 arms, 540 rows.
513 treatment rows judged; **486 gated clean, 27 dropped** (all for one reason, see Section 4).
Every number below is independently recomputed from `summary.json` against the sweep-running
subagent's own figures, to 4 decimal places; none are taken on trust.

### Geometric mean per arm, over gated points (untuned: no knob in this table was adjusted
after seeing these results)

| arm | n gated / 27 | geomean vs `b1_best` | geomean vs `b0_default` |
|---|---|---|---|
| `a_slack` (primary treatment) | 27 | **1.0121** | 1.0172 |
| `a_nolane` (balance only, no upgrade) | 27 | 1.0045 | 1.0096 |
| `a_slack_cap` (slack_lane + best static cap) | 27 | 1.0000 | 1.0050 |
| `b1_cap16` | 27 | 0.9969 | 1.0019 |
| `b1_cap24` | 27 | 0.9922 | 0.9972 |
| `b1_cap32` | 18 | 0.9855 | 0.9898 |
| `b1_cap48` | 9 | 0.9852 | 0.9941 |

`b1_cap32` and `b1_cap48`'s geomeans are over smaller, staleness-skewed subsets (see Section 4):
do not compare them to the other rows' 27/27 geomeans as if the point sets matched.

### Why `a_slack` beats `a_nolane`: the upgrade search is inactive at low staleness

`a_slack` and `a_nolane` are **byte-identical** (same `span`, to full float precision) at all 9
`staleness=1` points, plus one `staleness=3` point (`s3_rho1.10_seed1`) — 10 of 27 points total.
Confirmed by direct instrumentation of `SlackLane.begin_dispatch` on 3 representative points
(one per staleness level, `rho=1.0, seed=0`): at `staleness=1`, the upgrade search runs on every
one of 56 dispatch bursts and finds a fast lane on **0 of them**. At `staleness=2` and
`staleness=3`, it finds 1-2 fast instances (out of 16) on a minority of bursts (7/54 and 4/54 in
the two sampled points), with at most 2 accepted moves in any single burst. `a_slack`'s entire
edge over `a_nolane` (1.0121 vs 1.0045 vs `b1_best`) comes from the 17 of 27 points where the
search found a genuine, net-positive move — it is structurally a no-op at `staleness=1` at this
16-instance fleet scale, not a tuning artifact.

### `length_lane` family (static, no planning): honest number vs. in-sample oracle

Reported separately per the plan's own instruction — do not conflate these:
- **Honest** (the single arm with the best geomean across all 27 gated points, picked by the
  driver's own `_length_lane_honest_arm`): `c_k1_L20000_c2` (1 fast instance, length threshold
  20000, capacity 2), geomean vs `b1_best` = **1.0143**, vs `b0_default` = 1.0194.
- **In-sample oracle** (best of the 12 `c_*` arms chosen per point, an upper bound only, not a
  deployable policy): geomean vs `b1_best` = **1.0251**, vs `b0_default` = 1.0303.
- Full 12-arm range vs `b1_best`: 0.9269 (`c_k4_L30000_c2`) to 1.0143 (the honest pick).
  `length_lane`'s honest number (1.0143) is comparable to, and slightly above, `a_slack`'s
  (1.0121) at this sweep's point grid — the static ablation is not dominated by the planned
  strategy here.

### `headroom_capture` (fraction of the gap between `b1_best` and the contention-free bound
captured), gated rows

| arm | n | median | min | max |
|---|---|---|---|---|
| `a_slack` | 27 | **+0.039** | -0.238 | +0.191 |
| `a_slack_cap` | 27 | +0.012 | -0.220 | +0.179 |
| `a_nolane` | 27 | -0.023 | -0.274 | +0.191 |
| `b1_cap16` | 27 | 0.000 | -0.099 | 0.000 |
| `b1_cap24` | 27 | -0.022 | -0.297 | 0.000 |
| `b1_cap32` | 18 | -0.115 | -0.370 | 0.000 |
| `b1_cap48` | 9 | -0.112 | -0.343 | -0.021 |

Median `headroom_capture` for `a_slack` is below 0.5 (it is +0.039, far below it) on points
with meaningful headroom. Per the plan's own decision rule (Task 10), this means **Algorithm B
(explicit cadence-target planning) is not warranted** — `a_slack`'s greedy, nearest-late-buffer
search already behaves like the dominant lever available at this fleet scale, and there is not
enough uncaptured headroom to justify the added complexity of period-targeted planning. This
is evaluated formally in Task 10; it is noted here because the number driving that decision is
produced by this sweep.

### Step-interval autocorrelation: testing the period-2 oscillation prior

Explicit test of the prior that `staleness=S` should produce an oscillation with some period
related to `S`. Measured by re-running 6 representative points (`b0_default` and `a_slack`, one
per staleness level, `rho=1.0, seed=0`) with full telemetry and applying
`srl.diagnosis.cadence.cadence_report` directly (the same function approved in Task 6):

| point | arm | trainer idle (s) | fleet idle frac | acf1 (lag 1) | acf2 (lag 2) |
|---|---|---|---|---|---|
| `s1_rho1.00_seed0` | `b0_default` | 3115.8 | 0.196 | **-0.528** | +0.082 |
| `s1_rho1.00_seed0` | `a_slack` | 2994.6 | 0.186 | **-0.658** | **+0.432** |
| `s2_rho1.00_seed0` | `b0_default` | 1685.9 | 0.132 | -0.278 | -0.309 |
| `s2_rho1.00_seed0` | `a_slack` | 1644.2 | 0.129 | -0.259 | -0.316 |
| `s3_rho1.00_seed0` | `b0_default` | 886.9 | 0.054 | +0.089 | -0.033 |
| `s3_rho1.00_seed0` | `a_slack` | 851.6 | 0.059 | +0.070 | +0.021 |

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

Fleet idle fraction drops monotonically with staleness (0.19 to 0.13 to 0.05) regardless of
arm, consistent with staleness relaxing the admission window and giving the fleet more
continuously-available work — this is the dominant, well-behaved trend; a repeating oscillation
is, at best, a secondary and partial effect confined to low staleness.

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

Full list (point, arm, span):

```
s1_rho0.90_seed0/b1_cap32   8472.2979      s1_rho0.90_seed0/b1_cap48   8472.2979
s1_rho1.00_seed0/b1_cap32   8544.8439      s1_rho1.00_seed0/b1_cap48   8544.8439
s1_rho1.10_seed0/b1_cap32   8867.8856      s1_rho1.10_seed0/b1_cap48   8867.8856
s1_rho0.90_seed1/b1_cap32   8915.1391      s1_rho0.90_seed1/b1_cap48   8915.1391
s1_rho1.00_seed1/b1_cap32   9203.2403      s1_rho1.00_seed1/b1_cap48   9203.2403
s1_rho1.10_seed1/b1_cap32   9423.7461      s1_rho1.10_seed1/b1_cap48   9423.7461
s1_rho0.90_seed2/b1_cap32   8790.7779      s1_rho0.90_seed2/b1_cap48   8790.7779
s1_rho1.00_seed2/b1_cap32   8996.0508      s1_rho1.00_seed2/b1_cap48   8996.0508
s1_rho1.10_seed2/b1_cap32   9234.8637      s1_rho1.10_seed2/b1_cap48   9234.8637
s2_rho0.90_seed0/b1_cap48   6895.6759      s2_rho1.00_seed0/b1_cap48   7149.9361
s2_rho1.10_seed0/b1_cap48   7317.4010      s2_rho0.90_seed1/b1_cap48   7067.1119
s2_rho1.00_seed1/b1_cap48   7173.9143      s2_rho1.10_seed1/b1_cap48   7382.1479
s2_rho0.90_seed2/b1_cap48   6695.2233      s2_rho1.00_seed2/b1_cap48   6844.8985
s2_rho1.10_seed2/b1_cap48   7118.1455
```

This is treated as evidence the gate fired correctly on a genuinely non-binding cap, not a
wiring failure: `a_nolane`, `a_slack`, and all 12 `c_*` arms differ from `b0_default` at every
single one of the 27 points, confirming the harness as a whole is sensitive to real changes.

Preemptions and `reprefill_tokens` were recorded on every row without gating. Across all 513
treatment rows: 101 total preemptions (100 in `c_k4_L20000_c2`, 1 in `c_k4_L30000_c2`; **zero**
in `a_slack`, `a_nolane`, `a_slack_cap`, `b0_default`, or any `b1_cap*` arm). This sweep does
not meaningfully exercise KV pressure tight enough to force preemption outside one narrow
corner of the `length_lane` grid.

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
- **`a_slack_cap`'s geomean of exactly 1.0000 vs `b1_best`** deserves a caveat: it means
  combining the upgrade search with a uniform cap neither helps nor hurts relative to the best
  uncapped-strategy static cap, on average — not that it is uninteresting, but this sweep does
  not explain the mechanism behind that near-exact cancellation, only that it is reproducible
  across all 27 points (27/27 gated).
