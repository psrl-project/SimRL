# Diagnosis report: headroom on the extreme long-tail PSRL point

## 1. What was diagnosed

Experiment: `long_tail_baseline` (driver
`/apdcephfs_zwfy/share_303760348/lhy/SimRL/workflow/analysis/experiments/long_tail_baseline.py`,
gitignored), workload point `p200k_n6`.

No `summary.json` was supplied and none exists in the repository for this point: the prior cap-grid
sweep that established `cap=0` as the best cap for both arms left no persisted summary (searched
`outputs/2026-10-20`, `outputs/2026-10-21`, `outputs/simulate/long_tail_*`, which hold other points
only). Each row's gate was therefore built here, see section 3.

Rows, all four named by the caller. The reason for all four is the same: these are the two baseline
arms at the best cap and at the widest cap, which is the pair the headroom question needs.

| row_id | arm | cap |
|---|---|---|
| `p200k_n6_cap0/psrl_default` | `request_num_balance` | 0 |
| `p200k_n6_cap96/psrl_default` | `request_num_balance` | 96 |
| `p200k_n6_cap0/length_aware_lpt_load` | `length_aware`, `lpt`/`load` | 0 |
| `p200k_n6_cap96/length_aware_lpt_load` | `length_aware`, `lpt`/`load` | 96 |

Workload shape in one line: 960 groups of 8, 32 of them (3.33%) near 200k tokens carrying 67.8% of
generated tokens, the rest a lognormal short mode at mean 3000, on 6 instances with
`batch=96` groups per step, `staleness=1`, `n_versions=6`, training pinned at 345 s/step.

The consumed set is identical in all four rows: 576 groups, 4608 requests, 40,215,819 generated
tokens, 0 dropped. Of those, 128 requests (2.78%) exceed 100k generated tokens, they form 16 whole
groups with no mixed group (every member of a long group is long), and they carry 63.7% of the
consumed generated tokens.

## 2. Session

- Git head `70bfb0d`. The working tree is **dirty**: 40 modified or untracked paths, including
  `srl/cost/kernel/model_ops.py` (43 lines) and `srl/cost/training/*`, which the rollout cost model
  can reach. All four rows ran against the same tree, so the comparison between them is clean, but
  the absolute numbers are not reproducible from `70bfb0d` alone.
- Binary: not involved. The driver rebuilds the simulator with `backend="native"`, so
  `srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so` (md5 `4c9a6244aeaf872809771252a83e7ad6`) does
  not participate in any number here.
- Preflight: exit 0, no blocking items, no warnings. `SIMRL_WORKSPACE=/apdcephfs_zwfy/share_303760348/lhy`,
  matplotlib `Agg`, scratch root `/tmp/simrl_analysis` writable.
- Telemetry traces, all on local disk per this unit's checklist:
  `/tmp/simrl_analysis/long_tail_baseline/<row_id with / replaced by _>/`, 1.5-1.7 GB each.
- Wall time per row (gate run / telemetry replay / diagnosis, four rows in parallel):

| row | row run | telemetry replay | diagnosis |
|---|---|---|---|
| `cap0/psrl_default` | 76 s | 161 s | numbers written, figures see section 8 |
| `cap0/length_aware_lpt_load` | 79 s | 146 s | same |
| `cap96/psrl_default` | 97 s | 169 s | same |
| `cap96/length_aware_lpt_load` | 109 s | 189 s | same |

The stream reconstruction plus all bounds, idle windows and imbalance took 65-85 s per row.

## 3. Fidelity

No prior summary row exists for this point, so the gate was constructed the only way that keeps it
a real test rather than a formality: for each spec, one run with telemetry **off** (which is
exactly what a `validate` sweep row is, same `build`, same `undoable_events`, same `backend`) was
measured with `workflow/validate/scripts/collect.py`'s `measure` and used as the row, then the
telemetry replay was gated against it on all nine `GATED_KEYS`. This gates the one thing that could
differ between a sweep row and this unit's replay, namely telemetry logging and the store.

All four rows passed 9/9, floats bit-identical, counts exact, `livelocked=False`:

| row | span | consumed_tokens | consumed_groups | mean_conc | peak_conc | preemptions | versions | dropped |
|---|---|---|---|---|---|---|---|---|
| `cap0/psrl_default` | 58832.278535799276 | 40215819 | 576 | 12.622734773609414 | 258 | 85 | 6 | 0 |
| `cap0/length_aware_lpt_load` | 57578.21609747056 | 40215819 | 576 | 12.526466509135883 | 282 | 85 | 6 | 0 |
| `cap96/psrl_default` | 60062.17184319941 | 40215819 | 576 | 11.678719220477968 | 103 | 83 | 6 | 0 |
| `cap96/length_aware_lpt_load` | 60067.29559960279 | 40215819 | 576 | 11.983474320933606 | 105 | 95 | 6 | 0 |

Cross-source checks, every row: `decode_tokens_match`, `completion_time_match`, `abort_count_match`,
`reprefill_present` all `ok=True`. The phase reconstruction never read the `request` stream it is
checked against.

An independent determinism confirmation fell out for free: the pre-existing trace for
`cap0/psrl_default` (written before this session) carries version 1 end `630.5222328302257` and
version 6 end `58832.278535799276`, which the fresh runs reproduced to the last digit.

## 4. Where the span went

| row | span (s) | train compute (s) | idle (s) | idle share of span |
|---|---|---|---|---|
| `cap0/psrl_default` | 58832.3 | 2073.0 | 56759.3 | 96.5% |
| `cap0/length_aware_lpt_load` | 57578.2 | 2073.0 | 55505.2 | 96.4% |
| `cap96/psrl_default` | 60062.2 | 2073.0 | 57989.2 | 96.5% |
| `cap96/length_aware_lpt_load` | 60067.3 | 2073.0 | 57994.3 | 96.5% |

Training is 3.5% of span in every row. The run is entirely rollout-bound.

Idle broken down by the binding request's phases (shares of total idle, weighted by each step's
idle). Each row sums to 1.0000 with no residual:

| row | interference | reprefill | solo | queue | weight_load |
|---|---|---|---|---|---|
| `cap0/psrl_default` | 0.5639 | 0.2819 | 0.1542 | 0.0000 | 0.0000 |
| `cap0/length_aware_lpt_load` | 0.5575 | 0.2849 | 0.1576 | 0.0000 | 0.0000 |
| `cap96/psrl_default` | 0.5756 | 0.2645 | 0.1599 | 0.0000 | 0.0000 |
| `cap96/length_aware_lpt_load` | 0.5330 | 0.3204 | 0.1465 | 0.0000 | 0.0000 |

The same thing in seconds of idle:

| row | interference | reprefill | solo | queue |
|---|---|---|---|---|
| `cap0/psrl_default` | 32007.3 | 15998.8 | 8753.2 | 0.0 |
| `cap0/length_aware_lpt_load` | 30945.7 | 15812.1 | 8747.4 | 0.0 |
| `cap96/psrl_default` | 33378.8 | 15335.9 | 9273.5 | 1.0 |
| `cap96/length_aware_lpt_load` | 30912.2 | 18582.9 | 8496.9 | 2.3 |

Every one of the 24 steps has `wait_cause="completion"`: the step started at the exact moment its
binding request completed. **Unexplained waits: 0 steps, 0.0 s.** Queue and weight-load are
negligible everywhere (0.0-2.3 s of 55-58 ks).

Supporting the reprefill row: re-prefill tokens are 42.6%-46.7% of all generated tokens
(`cap0/psrl` 18,036,680 of 40,215,819; `cap0/lpt` 17,451,176; `cap96/psrl` 17,123,564;
`cap96/lpt` 18,797,446), against 83-95 preemptions and 125-128 aborts.

## 5. Bounds

### 5.1 The three bounds this unit implements

| row | `span_solo_all` reprefill charged | headroom | `span_solo_all` reprefill removed | headroom | `span_tail_only` | headroom |
|---|---|---|---|---|---|---|
| `cap0/psrl_default` | 53488.3 | 1.0999 | 18987.4 | 3.0985 | 58832.3 | 1.000000 |
| `cap0/length_aware_lpt_load` | 52107.9 | 1.1050 | 19013.7 | 3.0283 | 57578.2 | 1.000000 |
| `cap96/psrl_default` | 52231.2 | 1.1499 | 19002.2 | 3.1608 | 60062.2 | 1.000000 |
| `cap96/length_aware_lpt_load` | 56599.6 | 1.0613 | 19022.3 | 3.1577 | 60067.3 | 1.000000 |

Assumptions, verbatim from the bound definitions.

`span_solo_all`, both variants:
- every request decodes at a batch of one, so fleet capacity is ignored
- the consumed set and every training step duration are unchanged
- re-prefill is charged at its observed cost / re-prefill is removed entirely

`span_tail_only`:
- only each step's slowest member runs alone, every other member keeps its observed time
- the next-slowest member becomes binding once the slowest is accelerated
- the consumed set and every training step duration are unchanged

Two further assumptions are implicit in the chain both bounds build, and they matter on this
workload: a request is charged from the **previous step's end**, not from its own admission, and a
charged lifetime includes the request's **observed** (contended) re-prefill duration. Under
`staleness=1` the real run starts a group's generation up to one publish earlier than that, and its
re-prefill seconds carry co-residency inside them, so these chains can price a counterfactual
*slower* than the run they came from. Both chains are clamped at the observed span, so this shows up
as a headroom that cannot fall below 1.

### 5.2 `span_tail_only = 1.000000` is a clamp, not a measurement of zero headroom

Re-running the chain by hand against the binding request's own decomposition and the non-binding
members' observed completions (`cap0/psrl_default`, seconds):

| step | clock | charged binding lifetime | clock+charged | max other completion | binds | observed start | overshoot |
|---|---|---|---|---|---|---|---|
| v1 | 0.0 | 32.4 | 32.4 | 284.9 | others | 285.0 | -0.2 |
| v2 | 630.4 | 15497.7 | 16128.0 | 27790.0 | others | 28021.9 | -231.9 |
| v3 | 28135.5 | 23167.2 | 51302.8 | 33135.8 | binding charge | 33784.8 | +17517.9 |
| v4 | 51648.3 | 3617.7 | 55266.0 | 43821.9 | binding charge | 43928.9 | +11337.1 |
| v5 | 55611.5 | 4347.2 | 59958.7 | 47551.3 | binding charge | 47979.9 | +11978.7 |
| v6 | 60304.2 | 3278.5 | 63582.7 | 58470.9 | binding charge | 58486.8 | +5095.9 |

Chain end 63928.2, clamped to the observed 58832.3. The other three rows behave the same way:
informative at v1-v2 (overshoot -0.2 to -935.2 s), clamped from v3 on (overshoot +2451 to +16000 s).
The cause is the re-prefill charge: the v3 binding request is charged 20153.3 s of observed
re-prefill on top of a clock that has already advanced past 28 ks, against an observed idle window
of 5417.4 s. So `span_tail_only`'s `1.000000` on this point says the bound ran out of meaning, not
that tail isolation is worth nothing. Section 5.3 answers that question with a chain that cannot
clamp this way.

### 5.3 Supplementary bounds computed here, strictly more optimistic

Not part of `srl.diagnosis`. Computed from the `train` and `request` streams only
(`/tmp/simrl_work/free_bounds.py`). A chosen subset of requests is made **infinitely fast and free**:
charged nothing at all, not even their observed re-prefill, so they can never gate a step. Stated
assumptions:

- every non-accelerated request keeps its observed completion time,
- the consumed set and every training step duration are unchanged,
- fleet capacity is ignored: the accelerated requests' work simply vanishes rather than being moved
  somewhere, and the capacity it occupied is **not** charged to anyone,
- a training step still cannot start before the previous step ended.

| row | free the binding request | free the binding group (8) | free all 128 long requests (63.7% of consumed tokens) |
|---|---|---|---|
| `cap0/psrl_default` | 58816.4 (1.0003) | 58775.6 (1.0010) | 45412.6 (1.2955) |
| `cap0/length_aware_lpt_load` | 57451.2 (1.0022) | 57451.2 (1.0022) | 44561.7 (1.2921) |
| `cap96/psrl_default` | 59749.0 (1.0052) | 59749.0 (1.0052) | 46545.6 (1.2904) |
| `cap96/length_aware_lpt_load` | 59905.8 (1.0027) | 59905.8 (1.0027) | 47186.8 (1.2730) |

### 5.4 Is the headroom large enough for a mechanism to be worth building?

Read plainly, for a mechanism in the class the six attempts belong to (protect a rare long group
from contamination by the short population, by routing, admission, isolation or fleet shape):

- **Accelerating the straggler is worth 0.03%-0.52%.** Making each step's binding request, or its
  entire 8-member group, infinitely fast and free leaves the span within half a percent of observed.
  This is an upper bound and it is almost exactly 1.0.
- **Even deleting the whole long population is worth at most 1.27x-1.30x**, under an assumption no
  mechanism can satisfy: the long requests' 63.7% of tokens vanish and nobody pays for the capacity
  they were using, while the short requests keep the completion times they achieved *while being
  contended by those long requests*. A real isolation mechanism does the opposite of the second
  half: it takes capacity away from the 97.2% short population, which is what the six measured
  attempts kept finding.
- **The one genuinely large bound in the set, 3.03x-3.16x, is not a scheduling bound.** It is
  `span_solo_all` with re-prefill removed, and within that same chain re-prefill alone is the
  difference between 1.06x-1.15x and 3.03x-3.16x. That bound's content is "no request is ever
  contended *and* no request ever recomputes a discarded cache". Under fixed capacity the first half
  is unreachable, the second half is an admission and eviction question, not a placement question.

The floor those bounds are approaching is the workload's own solo content: `span_solo_all` with
re-prefill removed lands at 18987-19022 s in every row, which is six steps of (about 3170 s of solo
decode for the slowest member + 345.5 s of training). The observed 57.6-60.1 ks against that 19.0 ks
is the gap, and the measurements in section 6 say the gap is capacity, not placement.

## 6. Fleet state

Idle windows. `busy` is the min and mean over the 6 instances inside the window, `queue` is the mean
router queue depth, `limit` is whether the binding group's reservation sits at the staleness limit
(`buffer_id - v_traj >= 1`), `occ_later` is entries held in buffers at or past this step's version.

`cap0/psrl_default`:

| v | window (s) | width (s) | busy min | busy mean | queue | reserved in | limit | occ_later |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0-285.0 | 285.0 | 1.000 | 1.000 | 768.0 | 1 | yes | 0 |
| 2 | 630.5-28021.9 | 27391.4 | 1.000 | 1.000 | 293.3 | 1 | yes | 94 |
| 3 | 28367.4-33784.8 | 5417.4 | 1.000 | 1.000 | 266.0 | 2 | yes | 93 |
| 4 | 34130.3-43928.9 | 9798.6 | 1.000 | 1.000 | 272.0 | 3 | yes | 95 |
| 5 | 44274.4-47979.9 | 3705.5 | 1.000 | 1.000 | 260.7 | 4 | yes | 93 |
| 6 | 48325.4-58486.8 | 10161.3 | 0.814 | 0.904 | 12.0 | 5 | yes | 0 |

`cap0/length_aware_lpt_load`: v1-v5 every instance at busy 1.000, queue 261.3-768.0, reserved at
limit in all six windows; v6 busy min 0.890, mean 0.940, queue 12.0.

`cap96/psrl_default`: busy min per window 0.999, 0.979, 0.914, 0.978, 0.962, 0.725; busy mean 0.999,
0.995, 0.930, 0.986, 0.983, 0.877; queue 488.3, 133.2, 104.3, 113.7, 112.1, 12.0. Reserved at limit
in v2-v6; **v1 is the single exception in the whole report**: the binding group sat in buffer 0 with
`at_limit=False`, so its reservation could still have moved later.

`cap96/length_aware_lpt_load`: busy min 0.999, 0.987, 0.958, 0.949, 0.958, 0.902; busy mean 0.999,
0.998, 0.967, 0.960, 0.972, 0.949; queue 474.8, 132.3, 104.8, 110.4, 108.5, 12.0; reserved at limit
in all six windows.

Reserve-at-limit totals: **23 of 24 windows**. In every one of those the binding group's reservation
cannot be deferred to a later buffer, so its completion is what releases the step, which is the same
thing the 24/24 `wait_cause="completion"` says from the other side.

No window anywhere in the four rows has an instance idling next to an empty queue. The queue holds
104-768 requests in every window except the last of each run (12), where the workload is simply
running out.

Whole-run instance imbalance:

| row | busy s min-max | busy spread | tokens min-max | token spread |
|---|---|---|---|---|
| `cap0/psrl_default` | 56598.9-58486.5 | 3.23% | 9,267,259-9,801,708 | 5.45% |
| `cap0/length_aware_lpt_load` | 56088.4-57232.6 | 2.00% | 9,215,454-9,520,889 | 3.21% |
| `cap96/psrl_default` | 55596.7-59181.5 | 6.06% | 8,548,356-10,733,452 | 20.36% |
| `cap96/length_aware_lpt_load` | 57816.2-58754.1 | 1.60% | 8,975,184-10,716,455 | 16.25% |

Spread is `(max-min)/max`. At `cap=0` the fleet is close to perfectly balanced in both arms. At
`cap=96` the token spread opens to 16-20% while busy seconds stay within 6%, meaning instances spend
the same time busy on uneven token counts, which is what a per-instance admission cap of 96 does to
a workload whose requests differ by two orders of magnitude in length.

## 7. Paired diff

Both arms exist at both caps, so there are two paired diffs. Control is `psrl_default`
(`request_num_balance`), treatment is `length_aware_lpt_load`. Both arms consume the identical set
(576 groups, 40,215,819 tokens) and pay the identical 2073.0 s of training, so every difference is
idle.

### cap = 0

| component of idle | control (s) | treatment (s) | change (s) |
|---|---|---|---|
| interference | 32007.3 | 30945.7 | -1061.6 |
| reprefill | 15998.8 | 15812.1 | -186.7 |
| solo | 8753.2 | 8747.4 | -5.8 |
| queue | 0.0 | 0.0 | 0.0 |
| weight_load | 0.0 | 0.0 | 0.0 |
| **total idle** | **56759.3** | **55505.2** | **-1254.1** |
| span | 58832.3 | 57578.2 | -1254.1 |

85% of the 1254.1 s difference is interference on the binding request, the rest is re-prefill. Solo
content is unchanged to within 6 s, as it must be: the arms differ in ordering and balance, not in
what has to be generated. Supporting counts: preemptions 85 both arms, re-prefill tokens 18,036,680
to 17,451,176 (-3.2%), aborts 126 to 128.

### cap = 96

| component of idle | control (s) | treatment (s) | change (s) |
|---|---|---|---|
| interference | 33378.8 | 30912.2 | -2466.6 |
| reprefill | 15335.9 | 18582.9 | +3247.0 |
| solo | 9273.5 | 8496.9 | -776.6 |
| queue | 1.0 | 2.3 | +1.3 |
| weight_load | 0.0 | 0.0 | 0.0 |
| **total idle** | **57989.2** | **57994.3** | **+5.1** |
| span | 60062.2 | 60067.3 | +5.1 |

This is the more informative of the two diffs. Total idle is identical to within 5.1 s out of 58 ks
(0.009%), but the mix moved substantially: the treatment removes 2466.6 s of interference and takes
back 3247.0 s of re-prefill. The supporting counts agree: preemptions 83 to 95 (+14%), re-prefill
tokens 17,123,564 to 18,797,446 (+9.8%). Putting long requests first on a load-balanced instance
reduces how much they are contended while they run, and increases how often they are preempted and
have to recompute, and on this point those two effects cancel.

## 8. Stops and failures

- Fidelity: no failures. 36 of 36 gated field comparisons passed across the four rows.
- Cross-source checks: no failures. 16 of 16 passed.
- Timeouts: none. The slowest row took 109 s + 189 s + 85 s of real work.
- One deliverable is incomplete: **`binding_requests.png` is missing for all four rows**, the other
  three figures (`critical_path.png`, `buffers.png`, `idle_windows.png`) are written for all four
  under `/apdcephfs_zwfy/share_303760348/lhy/SimRL/outputs/analysis/long_tail_baseline/<row>/figures/`.
  `srl/visualizer/diagnosis.py`'s `plot_binding_requests` draws one matplotlib patch per phase, and
  on this workload the binding requests hold up to 259,809 phases each (per-row largest-to-smallest:
  cap0/psrl 188912, 188830, 10108, 322, 221, 177; cap96/lpt 259809, 221032, 41586, 8111, 352, 342),
  because a 200k-token resident appears in tens of thousands of consecutive `instance/N` records.
  Attempting it ran 14+ minutes at 24 GB RSS inside `Axes._update_patch_limits` without finishing,
  and was stopped. No number in this report depends on it. Fixing it would mean editing `srl/`,
  which this run was scoped out of.

## 9. Ranked hypotheses

Ranked by share of total idle time. No speedup is claimed for any of them.

**H1. The binding request is contended, and at this capacity it has nowhere else to go.**
53.3%-57.6% of idle (30912-33379 s). Supported by: the interference component of the binding
decomposition; busy fraction 0.89-1.00 in 23 of 24 windows with router queue 104-768; whole-run busy
spread 1.6%-6.1%. Falsified by: the `n_instances` sweep in `validate` with `batch` scaled
proportionally (per `memory/lessons.md`'s own warning that an unscaled sweep tests dilution instead)
, if span per consumed token does **not** improve when capacity is added, interference is not
capacity-bound and this hypothesis is wrong.

**H2. Re-prefill of discarded caches is the second-largest component, and the only one with a large
bound attached.** 26.5%-32.0% of idle (15336-18583 s). Supported by: re-prefill phase totals; 42.6%-
46.7% of all generated tokens being re-prefill tokens; and `span_solo_all` moving from 1.06x-1.15x to
3.03x-3.16x on nothing but zeroing re-prefill. Falsified by: a `validate` arm on this point at cap 0
that cuts preemption-driven re-prefill (admission that reserves for the group's known tokens, or an
eviction rule that will not discard a long resident), if `reprefill_tokens` drops materially and
span does not move, re-prefill is not on the critical path and this hypothesis is wrong.

**H3. There is no straggler to protect: the tail is a bulk.** Accounts for the remaining 14.7%-16.0%
of idle (the binding request's own solo content, 8497-9274 s) and explains why H1 cannot be attacked
by isolation. Supported by: freeing the binding request or its whole group is worth 1.0003x-1.0052x;
16 consumed long groups with **zero** mixed groups, so all 8 members of a long group are long; the
v2 batch holds 56 requests above 100k tokens out of 768; the gap between the slowest and
second-slowest member of a batch is 0.2-1613.5 s inside windows of 256.8-27542.9 s; sibling spread within
the binding group is 113.7-5761.5 s. Falsified by: any `validate` arm at this point whose span beats
the better baseline by more than 0.5% while still consuming the identical 576 groups, that would
contradict the free-the-binding-group bound and mean the bound's chaining assumption, not the
workload, is what is flat.

## 10. Not established

- **Single workload point, single seed.** `p200k_n6` only: 6 instances, batch 96, `staleness=1`,
  6 versions, training fixed at 345 s. Nothing here says anything about other tail fractions, other
  instance counts, other staleness horizons, or a training cost that is not pinned.
- **Two caps only** (0 and 96). The intermediate caps of the earlier sweep were not replayed.
- **The bounds are not achievable targets.** Every one ignores fleet capacity. The supplementary
  bounds additionally delete the accelerated requests' work without charging anyone for the capacity
  it was using, and hold every other request at a completion time that was measured *while* that
  work was running. A mechanism that moves the long work rather than deleting it faces a cost that
  no bound here prices.
- **`span_tail_only` contributed no information on this point**, see section 5.2. The conclusion in
  section 5.4 rests on the supplementary chains and on the fleet measurements, not on that bound.
- **`span_solo_all` with re-prefill charged is partly clamped too**, by the same mechanism: it
  charges each step's slowest member's observed re-prefill from the previous step's end. Its
  1.06x-1.15x is therefore a lower bound on that counterfactual's headroom, not a tight figure. The
  re-prefill-removed variant is the informative member of that pair.
- **H2 is not localised.** Re-prefill is measured, but which admission or eviction decision produces
  it is not. The bound that makes it look large (3.03x-3.16x) also removes all contention, so the
  two are not separated by any measurement in this report.
- **Nothing here is a speedup.** The cap=0 to cap=96 and control to treatment differences in section
  7 are decompositions of two measured runs, not a claim that either arm is better.
- **The tree was dirty** (section 2), including a modified `srl/cost/kernel/model_ops.py` that the
  rollout cost model can reach. Cross-row comparisons are unaffected, absolute seconds may not
  reproduce from `70bfb0d`.
