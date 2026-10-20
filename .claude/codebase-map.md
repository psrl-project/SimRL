# Codebase Map

## System Architecture

SimRL is a discrete-event simulator for async disaggregate RL training. It is organised in three
layers with a versioned on-disk seam between each pair.

Measure: `srl/profiler/` drives vLLM, Megatron-Bridge, an FSDP2 launcher or individual CUDA
kernels and normalises what it observes into `ProfileRecord` JSONL (`schema.py`, `io.py`) or
kernel parquet (`kernel/measurements.py`). Adapters record and never model.

Model: `srl/cost/` turns those records into predictions. Fitted models (`roofline`, `psrl`,
`distserve`, `hydraulis`, `fsdp`) write a `CostArtifact` (`cost/artifact.py`). Composed models
(`ComposedRollout`, `ComposedTraining`) sum measured kernels through `cost/kernel/database.py`.
Models interpolate and never measure.

Simulate: `srl/simulator/` drives the loop admit -> dispatch -> generate -> complete -> select batch
-> recompute + train + sync -> publish, under one of twelve typed policy presets, asking the cost
models for every step. `srl/component/` holds the engine and the per-stage cost models,
`srl/policy/` the staleness presets, `srl/workload/` the request source, `srl/forecast/` what a
scheduler may predict, `srl/telemetry/` and `srl/visualizer/` the outputs, and `srl/cli/` the seven
console scripts.

The rollout engine has two interchangeable backends: `NativeInstance` in Python and
`component/rollout/engine_cpp/` in C++, selected by `SIMRL_BACKEND` (`auto`, `cpp`, `native`).
The C++ module is built to `srl/_engine_cpp*.so`.

Configuration is Hydra: structured dataclasses in `config/schema.py`, typed policy presets in
`config/policies/`, YAML defaults in `config/defaults/`, overrides via `srl key=value`.

`deprecated/` is an archive. It is not packaged, not collected by pytest and not importable.

## Directory Tree

```
srl/
├── __init__.py
├── _logging.py              # srl_logger, SIMRL_LOG_LEVEL
├── _engine_cpp*.so          # Committed build of engine_cpp (CPython 3.12, linux x86_64)
├── component/               # One subpackage per RL loop stage
│   ├── rollout/
│   │   ├── engine/           # The Python engine
│   │   │   ├── request.py       # Request, Segment, ReqStatus (shared vocabulary)
│   │   │   ├── instance.py      # NativeInstance: continuous batching, KV, preemption
│   │   │   ├── event.py         # EventKind, EngineEvent
│   │   │   ├── state.py         # StepStat, InstanceLoad, InstanceView, TrajView, traj_view
│   │   │   ├── snapshot.py      # record_request, InstanceSnapshot (event rollback)
│   │   │   ├── eviction/        # Which resident pays when a decode step cannot fund its token
│   │   │   │   ├── strategy.py      # EvictionStrategy, EVICTION_REGISTRY, register_eviction,
│   │   │   │   │                    #   build_eviction. Importing the package registers all
│   │   │   │   ├── vllm_lru.py      # vllm_lru (default)
│   │   │   │   ├── longest_ctx.py   # longest_ctx
│   │   │   │   └── latest_deadline.py  # latest_deadline
│   │   │   ├── kv/              # KV block pool, a state machine
│   │   │   │   ├── block.py         # KVBlock
│   │   │   │   ├── free_list.py     # LRU queue of unheld blocks
│   │   │   │   ├── lineage.py       # Block hashes from workload provenance
│   │   │   │   ├── pool.py          # BlockPool
│   │   │   │   └── manager.py       # KVCacheManager, what NativeInstance talks to
│   │   │   └── undo/            # Reversible mutations for speculative events
│   │   │       ├── protocol.py      # The reversible-mutation protocol
│   │   │       ├── entries.py       # Captured-old-value shapes
│   │   │       └── log.py           # UndoLog for one event
│   │   ├── engine_cpp/       # C++ engine (pybind11), built to srl/_engine_cpp*.so
│   │   │   ├── CMakeLists.txt, bindings.cc
│   │   │   ├── include/engine/  # block, cost_model, event, free_list, instance, kv_manager,
│   │   │   │                    #   lineage, pool, request, undo, version
│   │   │   └── src/             # cost_model, free_list, instance, kv_manager, pool, request,
│   │   │                        #   smoke, undo
│   │   └── router/           # Cross-instance routing: dwell queue + strategies
│   │       ├── request_queue.py # RouterQueue, SortKey (one heap per version)
│   │       ├── strategy.py      # RoutingStrategy, RoutingContext, STRATEGY_REGISTRY,
│   │       │                    #   register_strategy, build_strategy, optional-hook dispatch
│   │       ├── router.py        # Router (queue + AdmissionGate + dispatch)
│   │       └── strategies/      # Importing the package registers every strategy
│   │           ├── request_num_balance.py  # default
│   │           ├── round_robin.py, random_choice.py
│   │           ├── kv_cache_aware.py, prefix_aware.py, throughput_balance.py
│   │           ├── length_aware.py         # group ordering + predicted-load placement
│   │           └── inventory_deadline.py   # forecasts each option against forecast/plan.py
│   ├── recompute/cost.py    # Fixed, Analytical, TrainingForward recompute models
│   ├── sync/cost.py         # Fixed and Bandwidth sync models
│   └── training/__init__.py # Namespace, the training models live in cost/training/
├── config/
│   ├── __init__.py          # register_configs() for the Hydra ConfigStore
│   ├── schema.py            # SimulateConfig and its blocks
│   ├── policies/            # One typed preset per system: 12 modules, PRESETS, preset_config
│   └── defaults/
│       ├── simulate.yaml
│       ├── policy/          # 12 YAML files, one per preset
│       ├── rollout_cost/    # roofline, psrl, distserve, composed
│       ├── training_cost/   # fixed, analytical, linear, hydraulis, fsdp
│       └── workload/        # uniform, normal, bimodal, lognormal, powerlaw, dapo_trace
├── cost/
│   ├── artifact.py          # CostArtifact, ArtifactEntry, FitMetrics
│   ├── kernel/              # Kernel cost path, torch-free
│   │   ├── ops/             # One class per op. Importing the package fills OP_REGISTRY
│   │   │   ├── base.py        # KernelOp protocol, OP_REGISTRY, op_for/ops_for_family/
│   │   │   │                  #   families/ops_in
│   │   │   ├── gemm.py, attention.py, attention_training.py, elementwise.py, collective.py,
│   │   │   │   gdn.py, optimizer.py
│   │   ├── database.py      # Query dataclasses, KernelDatabase (SOL / interpolated utility)
│   │   ├── floor.py         # fit_floors, fit_floors_report, load_floors, write_floors
│   │   ├── hardware.py      # GpuSpec, gpu_slug, load_gpu_spec, data/gpu_specs.yaml
│   │   ├── holdout.py       # evaluate_holdout, holdout_report, bracketing_indices
│   │   ├── interp.py        # UtilitySample, UtilityGrid, KernelCoverageError
│   │   ├── model_ops.py     # ModelOps: a checkpoint's per-step op list, sharded for one TP
│   │   ├── sol.py           # SOL formulas, SolComponents, FALLBACK_FLOORS_S
│   │   ├── surface.py       # UtilitySurface, PchipSurface, ShepardSurface, SURFACE_REGISTRY
│   │   ├── table.py         # KernelTable (load, slice, dedupe)
│   │   └── validate.py      # load_baseline, score_model (ComposedRollout vs vLLM JSONL)
│   ├── rollout/
│   │   ├── __init__.py      # ROLLOUT_MODEL_REGISTRY
│   │   ├── base.py          # RolloutCostModel ABC, RolloutWorkload
│   │   ├── roofline.py      # UnifiedRoofline, LinearLPS
│   │   ├── psrl.py          # PSRLFitted
│   │   ├── distserve.py     # DistServe
│   │   ├── composed.py      # ComposedRollout
│   │   ├── advance.py       # Closed-form decode advance (crossing_step, elapsed, steps_within)
│   │   ├── segment.py       # LatticeIntegrator
│   │   ├── kv_cache.py      # KvCacheGeometry, kv_blocks_from_memory, parameter_bytes_from_checkpoint
│   │   ├── kv_probe.py      # probe_kv_capacity, KV capacity cache
│   │   ├── _fit.py          # RolloutSample, shared fitting helpers
│   │   └── plot.py          # plot_vllm_rollout_regression
│   └── training/
│       ├── __init__.py      # TRAINING_MODEL_REGISTRY (hydraulis, fsdp)
│       ├── base.py          # TrainingCostModel, latency/memory ABCs, TrainingWorkload
│       ├── analytical.py, fixed.py, linear.py
│       ├── fsdp.py          # FsdpTrainingCost, FsdpLatencyModel, FsdpMemoryModel
│       ├── fsdp_inventory.py
│       ├── hydraulis.py     # HydraulisTrainingCost (Megatron 1F1B)
│       ├── composed.py      # ComposedTraining (not in the registry)
│       ├── training_ops.py  # TrainingOps: kernel inventory, PLUMBING_LAUNCHES_PER_LAYER
│       ├── latency.py       # The shared four-term form and its diagnostics
│       ├── microbatch.py    # BatchingMode, MicrobatchPlanner, verl_workload
│       ├── inventory.py, activation.py, fitting.py, validate.py
├── forecast/
│   ├── drain.py             # InstanceForecast, DrainResult (replays one instance)
│   ├── fleet.py             # FleetForecast, forecast, estimate_queued, evaluate_placement
│   ├── length.py            # LengthPredictor, OracleLength, NoisyOracleLength
│   ├── pipeline.py          # Pipeline, PipelineForecast, TrainState, compare, group_etas
│   ├── plan.py              # PlanSession (one dispatch burst)
│   └── snapshot.py          # Resident, Arrival, Interrupt, InstanceState
├── policy/
│   ├── base.py              # StalenessPolicy ABC, SimState, CompleteAction, Cancel
│   ├── ledger.py            # GroupLedger, GroupView
│   ├── registry.py          # POLICY_REGISTRY (12 policies)
│   ├── rules/               # Shared L1/L2/L3 primitives (admission, sampling, runtime)
│   └── psrl, areal, roll, slime, verl, verl_v1, skyrl, openrlhf, dressage, nemo_rl, trl,
│       prime_rl  # One module per system
├── profiler/
│   ├── schema.py            # ProfileRecord and its descriptors
│   ├── io.py                # read_records, write_records, profile_digest
│   ├── hf.py                # model descriptors from a checkpoint or YAML
│   ├── search_space.py      # ModelConstraints, TrainingCandidate, enumerate_training_candidates
│   ├── fsdp_search_space.py # FsdpCandidate, enumerate_fsdp_candidates
│   ├── adapters/
│   │   ├── __init__.py      # Re-exports the CLI-safe surface of all three adapters
│   │   ├── base/            # run.py (run_candidate, read_status, OOM markers), sidecar.py
│   │   ├── vllm/            # config, prepare, execute, parse, collect
│   │   ├── megatron/        # config, prepare, execute, launch, measure, collect
│   │   └── fsdp/            # config, prepare, execute, launch, measure, collect
│   └── kernel/
│       ├── config.py        # KernelProfileConfig, MeasurementConfig (torch-free)
│       ├── measurements.py  # PerfTableWriter (atomic parquet, columns from the op)
│       ├── introspect.py    # verify_shapes (meta-device module graph, torch inside)
│       ├── collective.py    # Multi-node collective driver (torchrun / ssh)
│       ├── collective_launch.py  # torchrun child, named as a string by collective.py
│       ├── grid/            # Which shapes a sweep enumerates. TORCH-FREE
│       │   ├── families.py      # ModelFamily, FAMILY_REGISTRY, imports the families at its tail
│       │   ├── families_dense.py, families_moe.py
│       │   ├── shape.py         # GemmShape
│       │   └── gemm, attention, attention_training, elementwise, gdn, optimizer
│       ├── sweep/           # One *Sweep parsed from each op's YAML subtree. TORCH-FREE
│       ├── runner/          # The GPU measurement loops, torch imported inside run()
│       │   ├── base.py          # KernelRunner: time()/row(), not the loop
│       │   ├── attention.py, attention_training.py, gemm.py, elementwise.py, gdn.py,
│       │   ├── optimizer.py
│       │   └── _vllm_attn.py, _flashinfer_attn.py  # backend probes
│       └── utils/           # timing.py (the one torch importer), model_dims.py, validators.py
├── simulator/
│   ├── config.py            # SimConfig, SimResult
│   ├── simulator.py         # Simulator: stage, dispatch, engine loop, train, publish
│   ├── runtime.py           # SchedulingRuntime, the read-only facade for strategies
│   ├── report.py            # StalenessReport, summarize
│   └── dynamic_parallel/    # FleetPlan, DynamicParallelFleet. Not constructed by Simulator
├── telemetry/
│   ├── store.py             # JsonlStore
│   ├── writer.py            # SimTelemetry (instance/*, rollout_step, gateway, request, train)
│   └── psrl_log.py          # convert_psrl_logs, extract_workload_trace
├── visualizer/plots.py      # SimPlotter
└── workload/
    ├── distributions.py     # uniform, fixed, bimodal, lognormal, normal, powerlaw
    ├── grouped.py           # GroupedWorkload, the simulator's request source
    ├── trace.py             # RequestGroup trace read/write
    ├── fitting.py           # Dataset prompt deduplication and rendering
    ├── inference.py         # Joint group collection against a served engine
    ├── sharding.py          # Worker topology, shard merge, collection validation
    ├── prefix_probe.py      # vLLM prefix-cache counters
    └── prefix_probe_run.py  # Controlled prefix-cache scenarios
deprecated/                  # Archive. Not packaged, not collected, not importable
workflow/                    # Agent-oriented validation workflows and reference notes
scripts/                     # Standalone analysis scripts
examples/                    # profile/ configs, simulate/ arms, analysis/ scripts, regress/
tests/                       # Flat. Prefixes group them (kernel_, engine_cpp_, forecast_, ...)
```

## Key Interfaces

### Request (`srl/component/rollout/engine/request.py`)
The unit of work every component handles. Import from the engine package:
`from ..component.rollout.engine import Request`. The C++ backend substitutes `CppRequest` with the
same attribute surface.
- `remaining`, `ctx`, `v_min`, `v_max` properties
- `add_tokens(version, n)`, which appends to the request's current `Segment`
- `staleness_true()`, `staleness_reported()`

### RolloutCostModel (`srl/cost/rollout/base.py`)
- `predict(workload) -> float` (abstract)
- `decode_coeffs(n_d, ctxsum) -> (floor, alpha, beta)` (abstract)
- `step_time(n_tokens, t2, ctxsum, n_reqs=0, prefill_ctx=0.0)`
- `decode_segment_time`, `decode_steps_within`, `decode_saturated_steps`, `saturated`

### TrainingCostModel (`srl/cost/training/base.py`)
- `estimate(workload: TrainingWorkload, parallelism: TrainingParallelism) -> TrainingCostEstimate`
- `TrainingCostEstimate.latency_s`, `.peak_memory_bytes`, `.diagnostics`
- Fitted variants add `fit(records, ...)` and `from_artifact(path, **selectors)`. Unknown selectors
  raise.

### NativeInstance (`srl/component/rollout/engine/instance.py`)
- `add_request(req, t) -> bool` (the authority on admission, while the router only gates eligibility)
- `advance_one_event(version) -> EngineEvent | None` with `undo_last_event()` or
  `commit_last_event()`. Exactly one of the two must follow. A snapshot left open outlives its event.
- `advance_to(t_limit, version) -> (completed, reached_limit)`
- `advance_iterations`, `run_until`, `step_time_delta`
- `abort_all(requeue)`, `cancel(reqs, keep_tokens, requeue)`
- `blocks_used()`, `free_blocks()`, `committed_blocks()`. A queued request holds no blocks yet, so a
  router comparing raw free blocks sees no difference between instances.

### Router (`srl/component/rollout/router/router.py`)
- `enqueue(reqs, t)`, `remove(reqs)`, `rekey(reqs)`, `release_group_pins(group_ids)`
- `dispatch(t) -> [(req, instance_id)]`. Runs only after a committed capacity release.
- `has_dispatchable()`. The engine loop must break out for this as well as for admission quota.
- `reclaim_preempted(t)` (vLLM's `preemption_notification_threshold`)
- `queue_len`, `queue_depths`
- Strategies are duck-typed beyond `select`: `narrow_candidates`, `allow_version_bypass`,
  `select_context`, `rank_queued`, `pick_queued`, `observe_enqueue`. `RoutingContext` carries the
  blocked versions, the fleet TP bounds and the clock.

### StalenessPolicy (`srl/policy/base.py`)
- `admit_quota(st) -> int` and `peek_batch(st) / take_batch(st)` are abstract
- `on_admit`, `on_route`, `on_group_complete`, `on_batch_unavailable`, `on_publish`,
  `on_train_version_advance`, `on_interrupt`, `on_discard`
- `can_sample(st)`, `pause_mode(st)`, `will_sync_after_train(st)`,
  `rollout_version_after_train(st)`, `replacement_groups_on_drop`
- Preset knobs live on the dataclasses in `srl/config/policies/` and become a policy through
  `preset_config(cfg).build(max_inflight=...)`. `preset.horizon` and
  `preset.ingress_window_groups(batch_size)` reach `SimConfig`.

### Simulator (`srl/simulator/simulator.py`)
- `__init__(rollout_cost, policy, cfg, training_cost=None, sync_cost=None, recompute_cost=None,
  store=None, backend="native")`
- `run() -> SimResult`
- `SimConfig` requires `group_size == workload.group_size` and `request_complete` in
  `undoable_events`.
- `SimResult`: `throughput`, `scg(s)`, `scg_count(s)`, `scg_curve()`, `staleness_report()`

### CostArtifact (`srl/cost/artifact.py`)
- `load(path)`, `save(path)` (atomic)
- `entry_for(parallelism, *, selector=None, ignore_pp=False)`. Exact match or `KeyError`.
- Entry fields: `coefficients`, `feature_ranges`, `train_metrics`, `validation_metrics`,
  `selector`, `diagnostics`

### SchedulingRuntime (`srl/simulator/runtime.py`)
Read-only facade for a routing strategy that declares `bind_runtime`: `now`, `train_version`,
`rollout_version`, `horizon`, `ingress_window_groups`, `inventory()` (psrl only), `cycle_time`,
`mean_cycle`, `train_state`, `instances`, `inflight`, `staged_groups`, `admitted_group_ids`,
`pending`, `queued`, `cost_model`, `instance_state`.

## Import Dependency Order

1. `_logging` (no internal deps)
2. `profiler.schema`, `profiler.io` (import `_logging`). `cost.kernel.hardware`, `sol`, `interp`,
   `surface`. `workload.distributions`, `workload.trace`, `workload.sharding`
3. `profiler.kernel.utils.{model_dims,validators}`, `profiler.kernel.sweep.*`,
   `profiler.kernel.grid.*` (depend on `profiler.schema`, and `grid.gemm` dispatches through
   `grid.families`, whose tail imports the family modules). All must stay torch-free.
4. `cost.kernel.ops.*` (depend on `cost.kernel.{hardware,sol,interp}`. `ops/__init__` imports every
   op module, the only registration path. `parse_sweep` and `measure` import the profiler lazily).
   MUST stay torch-free.
5. `cost.kernel.{table,floor,holdout,database}` (depend on `ops.base`, and `floor` imports `holdout`
   lazily).
6. `cost.kernel.model_ops` (depends on `profiler.kernel.grid.gemm`, `cost.kernel.database`).
7. `cost.artifact` (imports `profiler.schema`). `cost.rollout.segment`, `cost.rollout.advance`.
8. `cost.training.{base,inventory,fsdp_inventory,latency,microbatch,fitting,activation}`.
   `cost.training.{analytical,fixed,linear,hydraulis,fsdp}`. `cost.training.composed` and
   `cost.training.training_ops`. `cost.training.__init__` (registry).
   `cost.training.validate`.
9. `cost.rollout.{base,roofline,distserve,psrl,composed}` (`composed` depends on
   `cost.kernel.database`, `cost.rollout.base`, `cost.rollout.segment`). `cost.rollout.__init__`.
10. `cost.kernel.validate` (depends on `cost.rollout.base`).
11. `component.rollout.engine.{request,event,state,snapshot,undo,kv,eviction}`. Importing
    `eviction` registers the strategies.
12. `component.rollout.engine.instance` (depends on the engine modules and `cost.rollout.advance`).
13. `forecast.{length,snapshot}`, then `forecast.{drain,pipeline,fleet,plan}`.
14. `component.rollout.router.*` (depends on the engine and `forecast.length`. `strategies/` imports
    `forecast` at module level, and importing the package registers every strategy).
15. `policy.base`, `policy.ledger`, `policy.rules`, `policy.*`, `policy.registry`.
16. `workload.grouped`, `workload.fitting`, `workload.inference`.
17. `buffer.queue`.
18. `telemetry.store`, `telemetry.writer` (depends on the engine), `telemetry.psrl_log`.
19. `visualizer.plots` (depends on `telemetry.store` only).
20. `simulator.config` (depends on the engine and `workload.grouped`). `simulator.report`.
    `simulator.runtime` (depends on `forecast`). `simulator.simulator`.
    `simulator.dynamic_parallel.{plan,fleet}` (nothing in `simulator.simulator` imports them).
21. `config.policies.*` (import the policy classes). `config.schema`. `config.__init__`.
22. `profiler.hf`, `profiler.search_space`, `profiler.fsdp_search_space`.
    `profiler.adapters.base`. `profiler.adapters.{vllm,megatron,fsdp}`.
    `profiler.adapters.__init__`.
23. `cli.*`, with `cli.main` on top of almost everything.

Two cycles are deliberate and confined to torch-free modules:

- `cost` imports `profiler.schema`, `profiler.io`, `profiler.hf`, `profiler.kernel.grid` and
  `profiler.kernel.utils.model_dims`, while `profiler.adapters` calls the cost fitters. The
  modules `cost` imports must stay free of `torch` and vLLM,
  `tests/test_kernel_package_imports.py` enforces that module by module.
- `component.rollout.router.strategies` imports `forecast`, and `forecast.fleet`/`plan` import the
  engine's request views. `forecast.snapshot` reads an instance through `TrajView`, so the
  forecast cannot mutate either backend.

## Naming Notes

Two unrelated concepts use the word "segment":

- `Segment` in `engine/request.py` is a stretch of generated tokens produced under one weight
  version. It is what the staleness metrics read.
- A decode segment in `cost/rollout/advance.py` is a run of decode steps over a fixed batch with
  no prefill mixed in. It is a unit of time integration. `LatticeIntegrator` in
  `cost/rollout/segment.py` integrates it numerically when the step cost has no closed form.

`config` appears in two unrelated places. `srl/config/` is the Hydra configuration. Every
adapter's `config.py` is its workflow YAML type, and `srl/simulator/config.py` holds `SimConfig`,
the simulator's own run input rather than a Hydra node.

`srl/component/training/` is an empty namespace. The training cost models live in
`srl/cost/training/`, and `srl/profiler/adapters/` holds the training profilers.
