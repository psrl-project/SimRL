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
- 2026-10-28: Re-ran the above cadence-lanes diagnosis against a regenerated sweep (same driver,
  `enable_trajectory_sticky=True` now pinned, `t_sat` recalibrated) and the cross-source check
  failure recurred -- still only on a `request_num_balance`-capped control, still only at the
  point with the highest contention of the set tried, with the same `decode_tokens_match` /
  `reprefill_present` shape. One point that passed the check in the earlier dataset's sample now
  also passes here (5 of 6 rows this time vs 4 of 6 before), but a point with comparable
  contention still fails. The failure survived both the `ingress_lifetime_cap` fix and the
  `enable_trajectory_sticky` fix, i.e. it is not an artifact of either config-default bug -- it is
  some property of `request_num_balance`'s capped admission at high contention. Still not
  root-caused; do not assume a config fix elsewhere will make it go away without checking.
- 2026-10-28: A training step's wall-clock `dt` (`end - start` in `train.jsonl`) is not always
  equal to its priced cost (`train_time + sync_time + recompute_time`), even though
  `reference/metrics.md` already has a name for the difference (`weight_load`) and
  `weight_load_time=0.0` was configured. `_train`'s `_run_engine(deadline=...)` can settle past
  its nominal deadline before `_publish_weights()` runs, and the telemetry records whatever `self.t`
  that leaves as the step's `end`. This showed up as `cadence_report.train_busy` differing between
  two arms at the same point (`request_num_balance`-capped control higher) while
  `consumed_tokens`/`consumed_groups` were bit-identical between them (ruling out a genuine
  training-cost difference) -- confirmed by checking `diagnose.build()`'s `train_seconds_total`
  (which sums the *priced* components only) was bit-identical between the same two arms, then
  finding the gap lived entirely in 1-2 individual steps' `dt` on the capped-control arm only
  (never on `slack_lane`, 0 of 3 rows checked). Before reporting "arm X trains less/more" from
  `cadence_report`'s `train_busy` alone, diff it against `train_seconds_total` (or sum
  `train_time+sync_time+recompute_time` from the raw `train.jsonl` directly) to rule out this
  per-step residual being mistaken for a training-cost effect.
- 2026-10-28: `examples/simulate/staleness_throughput` sweeps suppress `instance` and
  `rollout_step`, so they have no fleet telemetry. A replay through the same `common.sh` +
  `psrl_aligned.sh` with `EXTRA_OVERRIDES` minus the `+log_exclude` reproduces `request.jsonl`
  and `train.jsonl` byte for byte, which is the fidelity gate for these rows.
- 2026-10-28: A request's last `generated` in `request.jsonl` is stale for anything still in
  flight at run end (rows are written only at enqueue/admit/complete/consume). Fleet work must
  come from `rollout_step` `n_decode` or the per-resident `generated` in `instance/*`, otherwise
  the "generated but never consumed" tail is understated.
- 2026-10-28: If the checkpoint at `rollout_cost.model_path` disappears, `resolve_kv_blocks`
  misses `~/.cache/srl/kv_capacity.json` and tries to launch vLLM. The cache key is
  `checkpoint_fingerprint` (config.json + safetensors index size), not the path, so any copy with
  the same files hits the same entry. Verify with a control run before trusting it.
- 2026-10-28: Fitting `rollout_step` segment `dt` against (steps, n_decode, n_prefill_computed)
  by least squares gives a negative decode coefficient (steps and decode tokens are collinear, and
  decode cost depends on context). To price re-prefill, run env-gated counterfactuals instead
  (`pause_mode=in_place`, scaled `kv_blocks`), injected via a `sitecustomize.py` on `PYTHONPATH`.
  Patch `Simulator.__init__`, not `srl.cli.main`: `python -m srl.cli.main` executes a fresh
  `__main__` copy that ignores patches to the imported module.
- 2026-10-28: Comparing prefetch depths over a short horizon, check that both arms consume the
  same set (tokens per step). With ample KV and no abort, a deep prefetch consumes a shorter-biased
  set within the window (short groups finish first under FIFO-among-ready), so a tokens/s gap can
  appear at an identical step rate.
