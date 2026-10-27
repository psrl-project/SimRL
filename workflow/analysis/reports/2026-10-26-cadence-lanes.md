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

**Headline result: every one of the 6 assigned rows failed the fidelity gate.** No idle-time
diagnosis, bound, fleet-state table, or figure was computed for any row, because doing so would
mean diagnosing a run that is not the one the row measured. The remainder of this report
documents that failure, the root-causing done on it, and what cannot be concluded as a result.
Sections 4 through 7 are consequently empty for every row; section 8 and section 9 carry the
substance of this diagnosis.

## 2. Session

Git head: `942c66b225c75d51e14aa7bfe7708cee5e228185`. Native backend binary
(`srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so`) md5: `4c9a6244aeaf872809771252a83e7ad6`,
mtime 2026-10-17 (unchanged since well before this sweep; also identical to
`srl/component/rollout/engine_cpp/build/_engine_cpp.cpython-312-x86_64-linux-gnu.so`, same md5).
Preflight: passed, no blocking or warning entries. Replay wall time per row (single-arm replay
through `replay.run`, 24 training steps, telemetry on): `s1_rho0.90_seed1/a_slack` 23s,
`s1_rho0.90_seed1/b1_cap16` 22s, `s1_rho1.00_seed1/a_slack` 24s, `s1_rho1.00_seed1/b1_cap16` 23s,
`s3_rho1.10_seed0/a_slack` 36s, `s3_rho1.10_seed0/b1_cap16` 30s.

## 3. Fidelity

**No row passed.** Per `workflow/analysis/scripts/replay.py`'s `fidelity()` (1e-9 relative for
floats, exact for counts), every one of the 6 rows failed on `mean_concurrency`, and 4 of the 6
(both arms at both `s1` points) also failed on `span`. The two `s3` rows matched on `span` and
every other field except `mean_concurrency`. Full field-by-field tables are in section 8, since
per this unit's own rule a failed row is reported there, never here.

## 4. Where the span went

Not computed for any row (fidelity gate failed on every row; computing this on a non-gated
replay would describe a run other than the one the row measured).

## 5. Bounds

Not computed for any row, for the same reason.

## 6. Fleet state

Not computed for any row, for the same reason.

## 7. Paired diff

Not computed for any point. Nothing here to diff: neither member of any of the 3 pairs has a
gated diagnosis to compare.

## 8. Stops and failures

### 8.1 Fidelity tables, all 6 rows

`ok` columns use the unit's own tolerance: floats at 1e-9 relative, counts exact.

**`s1_rho0.90_seed1/a_slack`** (treatment, win point)

| field | ok | row | replay |
|---|---|---|---|
| span | **FAIL** | 8313.823089637575 | 8322.49814595014 (+0.104%) |
| consumed_tokens | pass | 65238857 | 65238857 |
| consumed_groups | pass | 768 | 768 |
| tokens_per_group | pass | 84946.42838541667 | 84946.42838541667 |
| mean_concurrency | **FAIL** | 6.52806430206081 | 6.537586220740687 (+0.146%) |
| peak_concurrency | pass | 42 | 42 |
| preemptions | pass | 0 | 0 |
| versions_done | pass | 24 | 24 |
| dropped | pass | 0 | 0 |

**`s1_rho0.90_seed1/b1_cap16`** (control, win point)

| field | ok | row | replay |
|---|---|---|---|
| span | **FAIL** | 8844.822408890013 | 8856.81436656272 (+0.136%) |
| consumed_tokens | pass | 65238857 | 65238857 |
| consumed_groups | pass | 768 | 768 |
| tokens_per_group | pass | 84946.42838541667 | 84946.42838541667 |
| mean_concurrency | **FAIL** | 6.13525968759762 | 6.134922972080777 (-0.0055%) |
| peak_concurrency | pass | 20 | 20 |
| preemptions | pass | 0 | 0 |
| versions_done | pass | 24 | 24 |
| dropped | pass | 0 | 0 |

**`s1_rho1.00_seed1/a_slack`** (treatment, 2nd-win point)

| field | ok | row | replay |
|---|---|---|---|
| span | **FAIL** | 8602.615856081226 | 8780.010382734448 (+2.06%) |
| consumed_tokens | pass | 65238857 | 65238857 |
| consumed_groups | pass | 768 | 768 |
| tokens_per_group | pass | 84946.42838541667 | 84946.42838541667 |
| mean_concurrency | **FAIL** | 6.232535634639387 | 6.2302188184171285 (-0.037%) |
| peak_concurrency | pass | 40 | 40 |
| preemptions | pass | 0 | 0 |
| versions_done | pass | 24 | 24 |
| dropped | pass | 0 | 0 |

**`s1_rho1.00_seed1/b1_cap16`** (control, 2nd-win point)

| field | ok | row | replay |
|---|---|---|---|
| span | **FAIL** | 9128.98095224785 | 9152.141121655468 (+0.254%) |
| consumed_tokens | pass | 65238857 | 65238857 |
| consumed_groups | pass | 768 | 768 |
| tokens_per_group | pass | 84946.42838541667 | 84946.42838541667 |
| mean_concurrency | **FAIL** | 5.88386694752369 | 5.8817157939823 (-0.037%) |
| peak_concurrency | pass | 20 | 20 |
| preemptions | pass | 0 | 0 |
| versions_done | pass | 24 | 24 |
| dropped | pass | 0 | 0 |

**`s3_rho1.10_seed0/a_slack`** (treatment, loss point)

| field | ok | row | replay |
|---|---|---|---|
| span | pass | 6618.6718620468 | 6618.6718620468 |
| consumed_tokens | pass | 64546078 | 64546078 |
| consumed_groups | pass | 768 | 768 |
| tokens_per_group | pass | 84044.37239583333 | 84044.37239583333 |
| mean_concurrency | **FAIL** | 12.577723723533452 | 12.499219594347249 (-0.624%) |
| peak_concurrency | pass | 70 | 70 |
| preemptions | pass | 0 | 0 |
| versions_done | pass | 24 | 24 |
| dropped | pass | 0 | 0 |

**`s3_rho1.10_seed0/b1_cap16`** (control, loss point)

| field | ok | row | replay |
|---|---|---|---|
| span | pass | 6503.177484353644 | 6503.177484353644 |
| consumed_tokens | pass | 64546078 | 64546078 |
| consumed_groups | pass | 768 | 768 |
| tokens_per_group | pass | 84044.37239583333 | 84044.37239583333 |
| mean_concurrency | **FAIL** | 9.92323202548176 | 9.807412454054552 (-1.167%) |
| peak_concurrency | pass | 22 | 22 |
| preemptions | pass | 0 | 0 |
| versions_done | pass | 24 | 24 |
| dropped | pass | 0 | 0 |

### 8.2 Root-causing done on the mismatch, and what it ruled out

Four hypotheses were tested directly, not just reasoned about:

1. **Algorithmic non-determinism re-seeded per call (ruled out).** `cadence_lanes.py --smoke`'s
   own determinism check (`b0_default` built and run twice in one process, same spec) gave
   `rel_diff=0.000000` on span. Re-running that exact check for a spec matching
   `s1_rho1.00_seed1` reproduced `rel_diff=0.000000` here too. The simulator is bit-identical
   given a fixed spec and a fixed process.

2. **Telemetry logging perturbs the schedule (ruled out).** `replay.run` always builds with
   `output_dir` set (`log_telemetry=True`), while the original sweep's `_run_arm` builds with
   `output_dir=None` (`log_telemetry=False`). Re-running `s1_rho1.00_seed1/a_slack` and
   `s3_rho1.10_seed0/b1_cap16` through `driver.build(spec, None)` (telemetry off, matching how
   the sweep itself ran) reproduced the exact same mismatched numbers as the telemetry-on
   replay, to every digit printed. Telemetry logging is not the cause.

3. **Python hash-seed randomization driving set/dict iteration order (ruled out).** Re-ran
   `s3_rho1.10_seed0/b1_cap16` four times across four fresh process invocations — two with
   `PYTHONHASHSEED=0` pinned, two with it unset (the default, randomized per process) — and all
   four gave the identical `span=6503.177484353644`,
   `mean_concurrency=9.807412454054552`, `peak_concurrency=22`. Hash-seed randomization does not
   explain the gap: every fresh-process replay on this machine converges to the same value,
   regardless of hash seed.

4. **Native `.so` rebuilt in place since the row was captured (ruled out).** Both the loaded
   `.so` (`srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so`) and its build-directory copy have
   mtime 2026-10-17 and matching md5 `4c9a6244aeaf872809771252a83e7ad6`; `git log` on
   `srl/component/rollout/engine_cpp/` shows no commits since the `SimRL` rename. The binary
   driving this replay is the one that has been on disk for 10 days, not one changed between the
   sweep's run (today, `summary.json` mtime 2026-10-27 10:24) and this replay.

A fifth check narrowed where the gap shows up rather than ruling out a cause: running every arm
at `s3_rho1.10_seed0` (`b0_default`, `b1_cap16`, `a_nolane`, `a_slack`) sequentially in one fresh
process (mirroring `_run_point`'s own same-process, multi-arm execution order) reproduced the
`mean_concurrency` mismatch on **every** arm, including the very first arm run in that process
(`b0_default`, mismatch -0.96%, not in the assigned row set but checked for this reason). The
mismatch is present from the first simulator call in a fresh process; it does not grow with how
many arms have already run in that process. That weighs against "accumulated heap/allocation
history from the dozens of arms the original sweep ran before this one" as the mechanism, since
a freshly started short process already disagrees with the row by a similar margin on its very
first call.

What is established: this machine, right now, with the current code and the current native
binary, reproduces one specific set of numbers with total internal consistency (across 6+
independent process launches, with and without hash-seed pinning, with and without telemetry,
at 1, 4, and single-arm sequence depths). That consistent set of numbers is not the set recorded
in `summary.json`. `git diff`/`git log` show no edit to `collect.py` or the native engine source
since well before the sweep ran, and `cadence_lanes.py` (gitignored, so invisible to git) was
read directly and matches the `build()` already described in this report — nothing in it reads
calibration state outside the frozen `spec` dict the row carries.

What was not established, for lack of a cheap way to check it from inside this unit: whether the
original sweep ran on a different physical node (no node identity is recorded in
`summary.json`), whether a library version (numpy, the native toolchain, or a BLAS/thread
backend affecting floating-point rounding in the composed cost model) differs between the
process that wrote `summary.json` and this one, or whether some other concurrent session in this
shared, concurrently-edited repository touched a dependency the sweep or this replay reads (e.g.
`$HOME/.cache/srl/kv_capacity.json`, referenced by `cadence_lanes.py`'s `KV_TOKENS_PER_INSTANCE`
comment, or the kernel profile parquet tables under `outputs/profile/qwen3_8b/kernel_tp1/`)
between the sweep's run and this replay. None of these was tested; naming them is as far as this
unit's mandate (diagnose one run, never work around a failed gate) goes.

Two further observations, reported rather than explained:

- The gap appears on `mean_concurrency` at every point tested (6/6 assigned rows, plus the
  untested `b0_default`/`a_nolane` arms at `s3_rho1.10_seed0`), always in a consistent direction
  per point (the replay's own repeated value is always below the row's recorded value at every
  point checked here), with magnitude growing with psrl staleness: ~0.005-0.15% at `staleness=1`
  (the two `s1` points) vs. 0.6-1.2% at `staleness=3` (the `s3` point).
- `span` itself only diverges at the two `staleness=1` points (where it diverges substantially,
  up to 2.06%); at the one `staleness=3` point checked, `span` is bit-identical even though
  `mean_concurrency` is not. A tighter staleness horizon gives admission arbitration less slack,
  so whatever is perturbing tie-break order evidently also reaches the critical path there,
  while at the looser `staleness=3` horizon it stays confined to concurrency bookkeeping away
  from the binding chain. This is a pattern, not an explanation: it narrows where to look, not
  why.

**Falsifying experiments for the remaining candidates**, each cheap enough to run before trusting
any future replay of this result set:

| Candidate | Falsifying experiment |
|---|---|
| Different physical node / CPU microarchitecture between sweep and replay | Record `platform.processor()`/`/proc/cpuinfo` model string and numpy/BLAS build info in `summary.json` going forward; re-run the sweep and this replay back-to-back on the same node and diff |
| A concurrently-edited shared dependency (kernel profile tables, `kv_capacity.json`) changed between the sweep's write and this replay | `md5sum` every file `_composed_cost_model()`/`build()` reads, snapshot it inside the sweep's own row, and compare against this replay's snapshot |
| Genuine floating-point non-associativity in a parallel reduction inside the native engine (thread-count- or scheduling-dependent) | Pin the native engine (if it exposes a thread-count knob) to 1 thread and re-run the sequential 4-arm test in section 8.2 again; if the mismatch disappears, this is confirmed |
| Accumulated process state from the full 27-point sweep (not reproduced by a short 1-4 arm sequence) | Replay the *entire* `seed=0` block of `POINTS()` in original order, in one process, up to and including `s3_rho1.10_seed0`, and check whether that reproduces the row exactly where the short sequence in section 8.2 did not |

## 9. Ranked hypotheses

No idle-time hypothesis about why `a_slack` won or lost at these 3 points can be ranked by share
of total idle time: no row passed the fidelity gate, so no idle-time breakdown exists for any of
the 6 rows, and this unit will not build one on an ungated replay. The hypotheses below are about
the fidelity failure itself, not about the win/loss mechanism Task 9 was asked to explain; they
are ordered by how much evidence rules each one in or out, not by idle-time share (that metric
does not apply here).

1. **Telemetry logging changes the schedule.** Ruled out directly: telemetry-off replay
   reproduces the same (mismatched) numbers as telemetry-on replay, digit for digit, for both
   rows tested this way. Falsifying experiment already run; no further action needed.
2. **Python hash-seed randomization changes tie-break order across processes.** Ruled out
   directly: `PYTHONHASHSEED=0` pinned vs. unset gave identical results across 4 process
   launches. No further action needed.
3. **The native `.so` was rebuilt between the sweep's run and this replay.** Ruled out directly:
   identical mtime and md5 on both copies of the binary, no git history since well before the
   rename. No further action needed.
4. **A difference in the compute environment (node, library version, or a concurrently-edited
   shared dependency) between the sweep's process and this replay's process.** Not ruled out;
   named with 2 falsifying experiments in section 8.2's table. This is the leading open
   candidate given everything else has been eliminated and the mismatch is fully self-consistent
   within this environment.
5. **Thread-scheduling-dependent floating-point non-associativity inside the native engine.**
   Not ruled out; named with 1 falsifying experiment in section 8.2's table. Weighed against,
   weakly, by the fact that 6+ repeated launches on this one machine never disagreed with *each
   other*, which would be a odd property for genuine run-to-run floating-point jitter to have
   (jitter should also appear across repeated launches on the same machine, not just between
   this machine and whatever produced the row).

## 10. Not established

- **The mechanism behind the Task 8 verdict at all 3 assigned points is not established by this
  diagnosis.** The reconciliation this unit was asked to perform (does the idle-time diff agree
  in sign with Task 8's measured win/loss) cannot be attempted: there is no idle-time diff,
  because no row passed the fidelity gate. This is not "the diagnosis disagrees with the sweep,"
  which the brief anticipated as a possible finding — it is "the diagnosis could not be run," a
  different and more basic gap. Task 8's own gated speedup numbers (1.0639, 1.0612, 0.9826 at
  these points) are not disputed by anything in this report; they come from the sweep's own
  gate, not from this unit's replay.
- **Root cause of the fidelity failure is narrowed, not identified.** Four candidates were
  eliminated by direct test (section 8.2); two remain open and unexamined for lack of a cheap
  in-unit way to check them (different compute node/library version; thread-dependent
  floating-point reduction in the native engine). Either could, in principle, also explain small
  differences in other result sets this driver or the native backend's "native" mode produced,
  not just `cadence_lanes`, but that generalization was not tested here and should not be
  assumed.
- **Single point selection.** Only 3 of 27 points were examined (plus the loss point's control
  checked across 2 extra untested arms, `b0_default`/`a_nolane`, purely for root-causing). The
  fidelity failure's pattern (worse at tighter staleness, present in `mean_concurrency` at every
  point checked) is drawn from 4 points total and may not generalize to the other 23.
- **The report template's section 3 says a failed row is reported "in section 7 instead."**
  Section 7 in this template is "Paired diff," which cannot hold a fidelity failure. This report
  follows the task brief's explicit instruction instead and files every failure in section 8.
  Flagging the mismatch here rather than silently picking one, since resolving a contradiction
  in the template's own wording is not this unit's call to make unilaterally.
- **Bounds and fleet-state assumptions.** Not applicable: no bound or fleet-state table was
  computed for any row (section 5, section 6).
