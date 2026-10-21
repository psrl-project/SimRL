# Lessons

Things a later run would otherwise rediscover: a check that fired and why, a row that is
unreplayable, a command that needed a workaround, a telemetry cost worth knowing before enabling a
stream. One bullet each, newest last, with the date. **Never record result numbers.** Remove a
bullet when it stops being true.

- 2026-10-22: Running `python workflow/analysis/scripts/preflight.py` directly puts
  `workflow/analysis/scripts/` at the front of `sys.path`, so the sibling `scripts/select.py`
  shadows the standard library `select` module and breaks any subsequent `import subprocess` or
  `import matplotlib`. `preflight.py` strips its own directory from `sys.path` before importing
  anything. Keep that guard if this file is edited.
- 2026-10-22: For the extreme long-tail point (`long_tail_baseline.py`), a full-telemetry replay's
  `instance/*` streams dominate trace size, not `rollout_step`. That makes `rollout_step` a weaker
  size lever than it looks: excluding it via `JsonlStore(exclude=...)` only trades away the
  busy-fraction and instance-imbalance measurement (`diagnosis/fleet.py`'s `_rollout_segments`) for a
  modest size reduction, while the dominant `instance/*` streams cannot be excluded without losing
  phase reconstruction (`diagnosis/timeline.py`'s `_engine_phases`), which is the diagnosis itself.
- 2026-10-22: `span_tail_only` and `span_solo_all(charge_reprefill=True)` can both clamp to the
  observed span and report a headroom of exactly 1, which looks like a measured absence of headroom
  and is not. Their chain charges a request from the previous step's end while also charging its
  **observed** re-prefill duration, which carries co-residency inside it, so when a request's
  re-prefill exceeds the idle window it sat in, the chain prices the counterfactual slower than the
  run it came from. Before reading a 1.0 as a verdict, re-run the chain by hand per step and check
  whether the binding charge or the other members' completions set `ready`. The clamp-free
  alternative is a chain that makes a chosen subset infinitely fast and free, charging it nothing.
- 2026-10-22: `srl/visualizer/diagnosis.py`'s `plot_binding_requests` draws one matplotlib patch per
  phase. A resident that lives for a very long run appears in a huge number of consecutive
  `instance/N` records, so its phase count is enormous and this figure does not terminate in usable
  time (it sits in `Axes._update_patch_limits` while RSS climbs). The other three figures cost under
  two minutes per row. Call `plot_critical_path`, `plot_buffers` and `plot_idle_windows` directly
  instead of `plot_all` on a long-residency workload. `DiagnosisPlotter` needs no `Simulator`, only
  the store, timelines, steps, windows and imbalance, so the figures can be produced in a separate
  process after the diagnosis numbers are written.
- 2026-10-22: A baseline-only diagnosis has no `summary.json` to gate against, and the earlier sweep
  that chose the operating point may have persisted none. Do not skip the gate: run each spec once
  with telemetry **off** through the same `build`, measure it with `workflow/validate/scripts/collect.py`'s
  `measure`, and use that as the row. It still tests the one thing that can differ between a sweep
  row and this unit's replay, namely telemetry logging and the store, and it costs about half of
  what the telemetry replay costs.
