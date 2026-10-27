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
- 2026-10-24: An ad hoc script that calls `replay.run`/`collect.measure` outside `preflight.py`
  hits the same `select.py`-shadows-stdlib bug the first lesson here describes, if it puts
  `workflow/analysis/scripts/` on `sys.path` via `sys.path.insert` + `import replay`. `replay.py`
  and `workflow/validate/scripts/collect.py` have no cross-file imports of their own, so load both
  with `importlib.util.spec_from_file_location` directly and skip `sys.path` entirely.
- 2026-10-24: Forcing KV pressure (lower `kv_scale`) on an LPT-admission + protect-long-eviction
  pair does not necessarily grow its margin over the baseline -- it can shrink it. LPT concentrates
  deep contexts in the resident pool by design, which raises *aggregate* committed KV (and so total
  preemption/reprefill churn across the whole consumed set) faster than a protect-long eviction
  strategy's per-event victim choice claws back. Compare the count of distinct requests ever
  preempted and total reprefill tokens summed over every consumed request between arms, not just
  the step's binding request's own `reprefill_seconds` -- the binding request can look only mildly
  worse while the population-wide churn more than doubles.
- 2026-10-24: For a single-step run (`n_versions=1`), `idle_windows`/`instance_imbalance` collapse
  to one trivial window/one instance (nothing to compare across steps or instances). The signal for
  "why did arm A finish before arm B" lives in `binding_decompositions[rid]` (solo/interference/
  queue seconds for the step's binding request) and `steps[i].sibling_spread` (how tightly the
  binding group's members finished together), not the fleet tables.
- 2026-10-27: A fresh-process replay of a `validate` sweep row can silently pick up a *different*
  `SimConfig` default than the one the sweep itself ran under, with every gated field still
  disagreeing bit-for-bit, if a concurrent session's uncommitted edit adds a new config field
  (with its own default) to a module the sweep process had *already imported* before the edit
  landed on disk. The sweep's own long-lived process freezes its import of that module at start,
  so its rows stay internally self-consistent under the old default throughout; a later,
  freshly-started replay process imports the new on-disk default instead, which is a materially
  different simulator even though every line of code the two processes share is identical. This
  produced a `mean_concurrency`-and-sometimes-`span` mismatch that survived ruling out telemetry
  on/off, `PYTHONHASHSEED`, and the native `.so` being rebuilt -- none of those is the first thing
  to suspect; the first thing to suspect is whether a config field's *default* changed on disk
  inside the sweep's own wall-clock execution window, found by `ls -la --time-style=full-iso` on
  every module the driver's `build()` imports, checked against the sweep's own `summary.json`
  mtime range. The fix is for the replay driver to pin every config field whose default could
  plausibly have moved, explicitly, rather than rely on "the dataclass default happens to match
  what the sweep used."
- 2026-10-27: `diagnose.build()`'s cross-source check (`srl.diagnosis.timeline.request_phases`)
  can fail for a `request_num_balance`-capped control row even after that row's `replay.fidelity()`
  passes bit-for-bit -- these are two separate gates, and passing the first does not imply passing
  the second. Observed failure modes cluster at the highest-contention point tried so far and
  include `reprefill_present` (a request charged reprefill tokens with no reconstructed reprefill
  phase) alongside `decode_tokens_match` and `completion_time_match`, all on the capped control
  arm, never on the lane-strategy arm at the same points. Not yet root-caused to a specific missing
  sync point; `cadence_report` (`srl.diagnosis.cadence`, reads `train.jsonl`/`rollout_step.jsonl`
  directly) does not depend on this reconstruction and stays available even when this check fails,
  which is the fallback to reach for rather than widening the check's own tolerance.
