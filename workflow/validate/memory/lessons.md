# Lessons

Things a later run would otherwise rediscover: a gate that fired and why, a workload point that is
unreliable, a command that needed a workaround. One bullet each, newest last, with the date.
Never record result numbers. Remove a bullet when it stops being true.

- 2026-10-20: `forecast_accuracy._medium_workload` hardcodes `np.random.default_rng(0)`, and there
  is no seed knob on `MediumRegime`. A multi-seed sweep on that harness needs its own copy of the
  workload builder with an explicit seed, not a patch of the module-level regime dataclass.
- 2026-10-20: Launching two `ProcessPoolExecutor`-based sweep scripts against the same psrl
  editable install at the same time intermittently raises `ModuleNotFoundError:
  psrl.workers.ps.staleness` in one of them (the other finishes clean). Not reproducible when
  rerun alone immediately after. Treat it as sweep-launch contention, not a real import gap, but
  don't run two of these sweeps concurrently on the same host without a retry plan.
- 2026-10-20: To see whether `horizon_buffers` ever reaches a 3rd live buffer in the real
  `InventoryDeadline` router (not just periodic forecast samples), monkeypatch `PlanSession.build`
  or the `compare` name as imported into `inventory_deadline.py` in-process. Patching
  `srl.forecast.pipeline.compare` itself has no effect because the strategy module already bound
  the name at `from ... import compare` time.
- 2026-10-20: To recover which of several candidates a pricing loop picked (e.g. `burst_plan.py`'s
  `plan_burst` choosing among `generate_candidates`'s per-isolation-size dicts) without
  reimplementing the pricing logic, wrap the lower primitive that *builds* each candidate (here
  `_assign_isolated_block`, same module, bare-name call so the patch takes) to record
  `id(candidate_dict)` alongside the label you want (here the isolation size `k`). The dict object
  is mutated in place and returned by reference all the way up, so `id()` on the final winner
  matches one of the recorded calls. No need to duplicate the filtering/selection logic.
- 2026-10-20: At this simulator's `headroom_probe`/`long_tail_isolation` harnesses, scaling `batch`
  up 4x (to hold per-instance pressure constant while sweeping `n_instances` to 8) makes full
  16-arm comparison grids impractical within a session — `py-spy dump` on a live worker showed the
  cost is in the KV-manager's own `allocate`/`can_allocate` event processing, not the strategy
  being tested, and it does not scale linearly with batch. A single-arm instrumentation run (one
  strategy, one point) stays tractable far longer than a full grid; prefer that when the point is a
  mechanism check rather than a speedup number.
- 2026-10-21: `long_tail_isolation.py`'s `long_len=524288` point (`p512_n2`) does not finish inside
  a session-reasonable window even at a single cap: 18 cells in parallel were still at 2-3/10
  versions after ~18 minutes wall clock with per-version ETA still climbing. Doubling `long_len`
  from the 262144 points is not a roughly-2x cost increase for this harness; budget for it
  separately or skip it rather than let it block the points that already answer the question.
- 2026-10-21: A single `sleep N` with `N` above ~115s in the Bash tool gets killed by the tool's
  own 2-minute default foreground timeout (exit 143), independent of whether the background job
  it's waiting on is still healthy. Poll a long-running background sweep with repeated short
  (<120s) `sleep` calls, not one long one.
- 2026-10-21: `examples/analysis/long_tail_isolation.py::build()` hardcodes its own `SimConfig`
  call and does not expose `undoable_events`. To apply the fast-undo knob
  (`undoable_events=frozenset({"request_complete"})`) uniformly across arms, construct `SimConfig`
  directly (copy `build()`'s body into the experiment script) rather than calling `build()`.
  Confirmed correctness-neutral and far faster than the full default
  `{"request_complete","preempt","iteration_done"}` on this harness's workloads; apply it to every
  arm in a comparison, not just a treatment, so it stays a speed knob and not a confound.
- 2026-10-21: Sourcing `env/simrl.sh` and then running python must happen in the *same* Bash tool
  call — the tool resets shell state between calls, so a `conda activate` from one call is gone by
  the next and bare `python`/`psrl` imports silently fall back to the wrong interpreter (surfaces
  as `ModuleNotFoundError` for a submodule, e.g. `psrl.workers.ps.staleness`, not an obvious env
  error) rather than failing to find `python` at all.
- 2026-10-21: `DynamicParallelConfig`/`ShapeReplanner`'s tail/bulk TP split, tested end to end with
  a real TP4 kernel database and `undoable_events` fast-undo (so it is no longer tractability-
  blocked), does reshape the fleet repeatedly across a 6-version run and does not revert to the
  uniform shape by the run's end. A heterogeneous-TP split chosen this way is not automatically a
  win — check mean/peak concurrency alongside span, since a treatment can show higher concurrency
  than every static baseline while still finishing later, which reads as congestion rather than
  throughput.
