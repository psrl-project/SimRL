---
name: analysis
description: Diagnose where span goes in one simulator run. Replays a selected validate row with full telemetry and attributes its idle time to the requests that bound each training step.
---

# Analysis

## Non-negotiable

1. **Diagnose, never claim a speedup.** A speedup is the `validate` unit's output. This unit
   explains a run. If a diagnosis suggests a mechanism, say what experiment would test it and
   stop there.
2. **Every number comes from a replayed real run that passed the fidelity gate.** If the replay
   disagrees with the row on any gated field, stop and report the mismatch. Do not diagnose a run
   that is not the one the row measured.
3. **Every cross-source check must pass.** The reconstruction is checked against streams it did not
   read. A failure means the reconstruction or the telemetry is wrong. Fix the simulator's missing
   sync point, do not widen the tolerance.
4. **A bound is reported with its assumptions.** `span_solo_all` ignores fleet capacity entirely.
   Large headroom is necessary for a mechanism to win, never sufficient.
5. **Rank hypotheses by share of total idle time.** Each hypothesis names the measurement that
   supports it and the `validate` experiment that would falsify it.

## Steps

1. Export `SIMRL_WORKSPACE` and run `python workflow/analysis/scripts/preflight.py`. Stop on a
   non-zero exit.
2. Load the `summary.json` you were given. If no row ids were named, run
   `scripts/select.py`'s `pick` to choose at most 6, and record each row's reason.
3. For each row: `scripts/replay.py`'s `run` into `/tmp/simrl_analysis/<experiment>/<row>/`, then
   `fidelity` against the row. Any False means stop.
4. `scripts/diagnose.py`'s `write` into `outputs/analysis/<experiment>/<row>/`.
5. When a point has both a control and a treatment row, diff their diagnoses: which components of
   idle time moved, and by how much. That diff is what explains a win or a loss.
6. Fill `reference/report.md`. Append mechanism lessons to `memory/lessons.md` under its own rules.

## Reference

| File | Holds |
|---|---|
| `reference/metrics.md` | Exact definition of every quantity reported |
| `reference/report.md` | The report template to fill |
| `memory/lessons.md` | Cross-run gotchas, never result numbers |
| `scripts/` | Deterministic work: preflight, select, replay, diagnose |
