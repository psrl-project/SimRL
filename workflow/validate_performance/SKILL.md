---
name: validate_performance
description: Use when deciding whether a scheduling or system change in this simulator actually improves end-to-end performance, or when asked to validate a knob, a policy, a routing change, or a new scheduler against a baseline.
---

# Validate performance

Decide, from real simulator runs, whether a change improves end-to-end performance. The hard part
is not running a simulation. It is refusing to report a speedup that came from a different workload,
a stale baseline, a different consumed set, or a run that quietly stopped early.

## Non-negotiable foundations

These are the only fixed rules. Everything else, including how you slice the experiment, is yours.

1. **Real simulator runs.** Every performance number comes from a completed run of
   `examples/simulate/simulate.sh` (or the in-process harness that drives the same `Simulator`).
   Never substitute an analytic estimate, a microbenchmark, a single-instance timing, or a partial
   replay.
2. **Baselines are re-run, not remembered.** Every comparison arm is produced in the same session
   with the same workload, the same model, the same TP, and the same binary. A number from an old
   results file is a hint, not a baseline.
3. **Same consumed work.** A speedup counts only if the arm consumed the same work as the
   baseline. Check `consumed_groups == batch_size * n_versions` and
   `tokens_per_group` within 1% of the baseline. If either fails, the run ended early or trained
   on different samples, and its span is meaningless as a speedup.
4. **Byte-identical means not wired.** If the treatment arm is exactly identical to the control,
   the switch did not take effect. That is a wiring bug, not "no effect".
5. **Report what survived.** State the gates each result passed. Drop the ones that failed rather
   than explaining them away.

Analysis is free. Write whatever script, table, or plot answers the question, as long as it reads
real simulator output.

For a long sweep, delegate to the `validate-performance` subagent so run logs stay out of the
caller's context.

## Workflow

0. **Preflight.** `python workflow/validate_performance/scripts/preflight.py`. It needs
   `SIMRL_WORKSPACE` and prints the binary hashes, the git head, and uncommitted `srl/` files. A
   non-zero exit means stop. Quote its warnings in the report.
1. **Stack the change.** Put the mechanism behind a flag with a default that reproduces the
   baseline exactly.
2. **Write the experiment.** Create `experiments/<name>.py` from `experiments/smoke.py`. Declare
   the controls, the treatments, the workload points, and the gates. A matrix is a script, not a
   fixed table in the framework.
3. **Smoke it.** Run one point first. Confirm the control arm reproduces the baseline and the
   treatment arm differs. Read `reference/gates.md` before trusting anything.
4. **Run the sweep.** One process per point, results under `outputs/simulate/<experiment>/`.
5. **Gate it.** Run `scripts/gates.py` on the result set. Fix or drop failing points.
6. **Report.** Fill `reference/report.md`. Lead with the gated table. State the geometric mean
   across points, the concurrency the treatment chose for itself, and every gate that failed.

## Red flags

Stop and re-check when you catch yourself:

- Quoting a speedup whose `consumed_groups` or `tokens_per_group` you did not check.
- Comparing against a baseline from another session, another binary, or another workload.
- Explaining a failing gate instead of dropping the point.
- Treating a treatment arm that matches the control exactly as evidence of "no effect".
- Using a microbenchmark, an analytic model, or a single instance to argue an end-to-end result.
- Tuning a knob on the points you report generalization over.
- Reporting a speedup from the grid the knob was tuned on without also running points that
  played no part in choosing it.
- Reading a pre-existing results file as the baseline because re-running it takes time.

## Failure modes seen in practice

An agent given only "does this knob improve performance" produced a correct tuning-grid number
and still had to be caught on four things. Each is a rule above for this reason.

| What it did | What the rule is |
|---|---|
| Reported +4.6% on the six points the knob had been tuned on as the headline | Headline is the untuned points. Tuned points are an upper bound. |
| Kept a point that lost 8% out of the summary sentence's numbers | Dropped points go in the failure list, all of them, with the value. |
| Accepted a 1.7% `tokens_per_group` drift on two arms as a minor caveat | Above 1% the span is not a speedup. Drop the point. |
| Ran one deterministic workload per point with no seed variation | State that the comparison is deterministic and that no seed variance was measured. |

## Reference

| File | Contents |
|---|---|
| `AGENT.md` | System prompt of the subagent: inputs, output, stop conditions, memory |
| `memory/lessons.md` | Lessons kept across runs, read at start, appended at end |
| `CHECKLIST.md` | Ordered execution checklist, tick each item |
| `reference/simulate.md` | Exact `simulate.sh` invocation, env vars, output layout |
| `reference/gates.md` | Gate definitions, thresholds, and why each exists |
| `reference/report.md` | Report template and the fields every report must carry |
| `scripts/preflight.py` | Session check: workspace, env script, binary hashes, dirty tree |
| `reference/analysis.md` | Result fields, telemetry layout, reusable analysis entry points |
| `experiments/smoke.py` | Smallest experiment that exercises both arms and every gate |
