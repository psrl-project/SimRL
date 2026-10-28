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
- 2026-10-22: `outputs/profile/qwen3_8b/kernel_tp{4,8}` are incomplete for an end-to-end TP sweep:
  `kernel_tp4` has no `collective/` tables (fails on the all_reduce query) and `kernel_tp8` has no
  `gemm/` tables at all. Only `kernel_tp1`/`kernel_tp2` are complete for this model on this host.
  Re-profile with `srl-profile kernels run --family collective|gemm` before trusting a tp4/tp8 row
  from any experiment that calls `ComposedRollout`/`KernelDatabase.from_root` directly (bypassing
  the `_composed_cost_model()` helpers that hardcode `tensor_parallel=1`).
- 2026-10-22: On the real Qwen3-8B DAPO trace (`outputs/workload/qwen3_8b_dapo_38912_tp1/joint.jsonl`,
  8x shorter tail and real within-group variance vs the earlier synthetic `TailPoint`), the
  near-zero-headroom verdict from the synthetic-point diagnosis reproduces: `length_aware`
  (`order="lpt", balance="load"`) does not clear `request_num_balance` at its own best cap at any
  of n_instances in {2,4,6} with proportionally scaled batch. The earlier verdict was not an
  artifact of the synthetic workload's extreme tail.
- 2026-10-23: `real_dapo_tail.py`'s full `TP_GRID=(1,2,4,8)` sweep (after `kernel_tp4`/`kernel_tp8`
  gained complete gemm/attention/elementwise/collective tables) shows no TP degree where
  `length_aware` reliably clears `request_num_balance`'s best cap across n_instances -- the earlier
  ungated single-point TP=2 hint does not generalize once n_instances is swept against it at the
  same TP; wins and losses of similar magnitude appear at every TP degree with no TP-correlated
  trend. Also: merging this sweep's output into an earlier TP=1-only `results.json` leaves the old
  run's unsuffixed point names (`n2`, `n4`, `n6_base`) sitting alongside the new `_tp1` compound
  names for the same cell -- same values, but double-counts that cell unless filtered out before
  taking an overall geometric mean.
- 2026-10-24: `length_aware`'s `order="lpt"` reorders the *router's* dispatch sequence
  (`rank_queued`), not anything inside `NativeInstance`. At `n_instances=1` that is sufficient to
  make the single instance admit longest-first too, since the router is the only gate before
  `add_request` appends to `waiting` in call order -- no engine-level admission-order code is
  needed for a single-instance LPT test. Verify it actually took effect by comparing
  `target_len` of the requests admitted in the first wave (`req.admit_time == 0.0`, i.e. the ones
  the router placed before any completion freed a `max_concurrent_seqs_per_instance` slot)
  between arms, not by trusting the config was accepted.
- 2026-10-24: At `cap=40` with the real measured TP1 H20 KV budget
  (`MEDIUM_KV_TOKENS_PER_INSTANCE=500_192`, confirmed against
  `$HOME/.cache/srl/kv_capacity.json`'s own TP1 probe entry) and a 256-request/32-group single
  step off `qwen3_8b_dapo_38912_tp1`, a single instance never reaches KV pressure -- zero
  preemptions in either arm. An eviction-strategy comparison (e.g. `shortest_ctx` vs `vllm_lru`)
  is untested at that point; it needs either a smaller KV budget or more concurrent long
  trajectories to ever call `select()`.
- 2026-10-24: `FixedTrainingCost`/`FixedTrainingLatency` raises `ValueError` on `latency_s <= 0`
  (`srl/cost/training/fixed.py`). To isolate a pure-rollout span (no training/sync tax) use a
  negligible positive epsilon (e.g. `1e-6`), not `0.0`. `FixedSyncCost` has no such floor and
  accepts `0.0` directly.
- 2026-10-26: `preflight.py`'s `run_sim.workspace()`/`activate_environment()` expect
  `SIMRL_WORKSPACE` to point at the directory that HOLDS `env/simrl.sh` (one level above this
  repo checkout), not at the repo root itself. Pointing it at the repo root raises
  `Environment script not found` as a blocking error even though the real env script
  (`/.../lhy/env/simrl.sh`, outside the repo) sourced fine in the same shell.
- 2026-10-26: `outputs/profile/<model>/kernel_tp{N}/kernel/h20/attention/{flashinfer,vllm}/<version>/`
  missing `floors.json` (present for other families like `collective`/`elementwise` in the same
  `kernel_tpN` root) is easy to miss because the run still completes and only logs a warning
  per op, not a hard failure. Check with `find <root> -iname floors.json` against a sibling TP
  degree's tree before trusting any attention-heavy prediction at that TP; `srl-profile kernels
  fit-floor --config <the tp's own profile yaml>` regenerates it in place from the existing
  parquet tables (no new profiling run needed).
- 2026-10-26: For a closed-batch makespan search over heterogeneous TP shapes, an exhaustive
  shape enumeration paired with a generic per-shape greedy-LPT + light-SA seed can land on a
  WORSE result than a single well-seeded SA run started from `heddle_presorted_dp`'s own
  construction, even with a comparable total iteration budget spread over more shapes. The
  presorted-contiguous (sort-by-length, cut into tiered contiguous blocks) construction is a
  structurally stronger starting point for this cost model than naive load-balancing greedy;
  don't read a cheap generic-greedy multi-shape sweep scoring worse than the incumbent as
  evidence the incumbent's shape is optimal -- it mainly shows the sweep's own seed was weaker.
  Isolate shape-coverage checks (fixed membership heuristic, vary shape) from membership-depth
  checks (fixed incumbent shape, vary membership search effort) rather than conflating both in
  one pooled search.
- 2026-10-27: `$HOME/.cache/srl/kv_capacity.json` is keyed by (model, tp, `max_model_len`, gpu
  util), not just model/tp. A new `MAX_PROMPT_LENGTH`+`MAX_OUTPUT_LENGTH` combination the cache
  has never seen (e.g. a new workload variant) is a cache miss and triggers a real
  `vllm.LLM(load_format="dummy")` probe. Launching several points that share a brand-new
  `max_model_len` in parallel starts that many real vLLM engines on the same GPU at once (each
  requesting `gpu_memory_utilization=0.9`), which is the same "parallel vLLM servers contaminate
  each other" failure mode as a real-engine benchmark. Run the first point of any new
  length-cap combination alone to populate the cache, then parallelize the rest once the probe
  has written its entry -- cheap simulation-only reruns of an already-cached combination are
  fine to parallelize freely.
- 2026-10-27: A multi-hour sweep driver that writes its `summary.json` only once, at the very
  end, produces long stretches with zero new bytes on disk and no subagent chatter -- that is
  the expected shape of a healthy run, not a stall signal. Distinguish a real hang from this by
  checking the actual OS process (`ps aux` on the python pid, or the background shell's CPU
  time) against wall-clock elapsed time: a healthy run's CPU time tracks wall clock roughly 1:1;
  a hung one's CPU time stops climbing while wall clock keeps going. Don't rely on the
  dispatched subagent's own "still running" self-report as the only signal -- it can be
  correctly waiting on its own spawned child process with nothing new to say.
- 2026-10-27: `JsonlStore(output_dir)` writes `output_dir/*.jsonl` directly -- it does not nest
  under a `logs/` subdirectory. That nesting convention belongs to `simulate.sh`'s own CLI
  wrapper, not to `JsonlStore` itself. Passing a path built on the `logs/`-nested assumption to a
  driver that instantiates `JsonlStore` directly (bypassing `simulate.sh`) raises
  `FileNotFoundError`; pass the directory itself.
- 2026-10-27: A sweep's per-row result schema is not guaranteed to match the field name used
  elsewhere in the same codebase for the same concept -- a gate-pass boolean that reads as
  `gated` in one script's local variable name can be stored on the row as `_gated` (leading
  underscore) by the actual summarizer. Confirm a field name against one real row's own keys
  before writing a verification script around it; a wrong assumed name silently filters to an
  empty set rather than raising, which reads as "zero gated rows" instead of "wrong key."
- 2026-10-27: A driver that loops over many points/arms in ONE long-lived process (`build()`
  called repeatedly, no subprocess-per-point) freezes whatever `srl` module state existed at
  process start -- later on-disk edits to `srl/` from a concurrent session during that process's
  multi-hour run have zero effect on it. A REPLAY attempted afterward in a fresh process is NOT
  protected this way and will silently pick up whatever is on disk at replay time. Confirmed
  exact root cause of a real fidelity-gate failure this way: a concurrent, uncommitted edit to
  `srl/simulator/simulator.py` added a new `SimConfig` field with a default that flips the
  ingress-admission behavior, landing on disk mid-sweep; the sweep's frozen process never saw it,
  a later fresh-process replay did, and `mean_concurrency`/`fleet_idle_frac` (far more than
  `span`) diverged as a result. Before trusting a replay-fidelity mismatch as evidence of
  non-determinism or a different physical node, check `git status --short`/mtimes on the files
  the driver imports against the sweep's own start/end wall-clock window first -- it is a cheap
  check and was the actual cause here, not either of the other two candidates. The fix is to pin
  every config field the driver's `build()` depends on explicitly, rather than relying on
  whatever a dataclass's own default happens to be at call time.
- 2026-10-27: When a new arm comes back byte-identical to an existing one across every point in
  a grid, do not report "no additional benefit" from the aggregate alone -- instrument the new
  code path directly on at least one point with full telemetry and classify every call's
  outcome (search never attempted vs. attempted and failed vs. attempted and tied). A null
  result that looks clean in the summary file can be hiding a guard clause that bails out before
  the new logic ever runs, which is a design-input problem (a badly chosen bracket/threshold),
  not evidence against the mechanism itself. Cheap to check: wrap the new function, recompute its
  own fallback path inline, and compare -- a handful of representative points is enough to see
  the pattern, no need to re-run the full grid under instrumentation.
- 2026-10-28: A driver shaped like `_add_group_sticky_arms.py` (build every arm's rows in one
  loop first, then gate-and-print in a separate second loop) emits zero of its per-row print
  lines until every simulate call across every point/arm has finished -- grepping that print for
  a progress signal reads as a stall for the whole simulate phase, even on a healthy run. Count
  completed `tqdm` "100%" progress-bar terminations in the log instead to track progress (note a
  single run can emit two near-identical 100% substrings, an instantaneous-rate one and a
  final-rate one sharing the same `t=` value, so don't treat the count as an exact per-run
  multiple), and corroborate with `ps`'s CPU-time-vs-wall-clock check, not the print stream alone.
