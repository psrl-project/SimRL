# Codebase Map

## System Architecture

SimRL is a discrete-event simulator for async disaggregate RL training.
The simulation loop: admit → generate (closed-form) → complete → select batch →
(recompute) → train → weight update → abort/re-prefill.

Code is organised around the RL loop components. Everything that models a stage
of the loop lives under `srl/component/` (rollout, training, sync, recompute),
while `srl/simulator/` drives them and `srl/policy/` decides staleness. `srl/forecast/` holds
what a scheduler may know about a trajectory, and `deprecated/` archives the abandoned SCG scheduling
line (inert: not packaged, not collected, not importable).

Configuration is Hydra-based: structured `@dataclass` configs in `schema.py`,
YAML defaults in `config/defaults/`, CLI overrides via `srl key=value`.

Profiling pipeline: vLLM/Megatron benchmarks → CSV → two-stage regression
fitter → JSON cost model → `PSRLFitted.from_json()`.

## Directory Tree

```
srl/
├── __init__.py
├── _logging.py              # srl_logger
├── component/               # One subpackage per RL loop stage
│   ├── rollout/
│   │   ├── engine/
│   │   │   ├── request.py       # ReqStatus, Segment, Request (shared vocabulary)
│   │   │   ├── instance.py      # NativeInstance (continuous batching, KV, preemption)
│   │   │   ├── event.py         # EventKind, EngineEvent
│   │   │   ├── state.py         # StepStat, InstanceLoad, InstanceView, TrajView/traj_view
│   │   │   ├── snapshot.py      # ReqSnapshot, InstanceSnapshot (event rollback)
│   │   │   ├── eviction/        # Who pays when a decode step cannot fund its next token
│   │   │   │   ├── strategy.py      # EvictionStrategy protocol, EVICTION_REGISTRY, build_eviction
│   │   │   │   ├── vllm_lru.py      # vllm_lru (default): newest dispatched pays
│   │   │   │   └── longest_ctx.py   # longest_ctx: deepest context pays
│   │   │   ├── kv/              # KV block pool, a state machine and not a cost model
│   │   │   │   ├── block.py         # KVBlock (ref count, hash, free-list links)
│   │   │   │   ├── free_list.py     # LRU queue of unheld blocks
│   │   │   │   ├── lineage.py       # Block hashes from workload provenance
│   │   │   │   ├── pool.py          # BlockPool (allocation, LRU reuse, eviction)
│   │   │   │   └── manager.py       # KVCacheManager (what NativeInstance talks to)
│   │   │   └── undo/            # Reversible mutations for speculative events
│   │   │       ├── protocol.py      # What a reversible mutation is
│   │   │       ├── entries.py       # Captured-old-value shapes
│   │   │       └── log.py           # UndoLog for one event
│   │   ├── engine_cpp/          # C++ engine (pybind11), built to srl/_engine_cpp*.so
│   │   │   ├── CMakeLists.txt, bindings.cc
│   │   │   ├── include/engine/  # block, cost_model, event, free_list, instance, kv_manager,
│   │   │   │                    #   lineage, pool, request, undo, version headers
│   │   │   └── src/             # cost_model, free_list, instance, kv_manager, pool, request,
│   │   │                        #   smoke, undo
│   │   └── router/              # Cross-instance routing: dwell queue + strategies
│   │       ├── request_queue.py # RouterQueue, SortKey (one sort-key heap per version)
│   │       ├── strategy.py      # RoutingStrategy protocol, RoutingContext, STRATEGY_REGISTRY,
│   │       │                    #   build_strategy, the optional-hook dispatch helpers
│   │       ├── strategies/      # One module per built-in routing strategy
│   │       │   ├── request_num_balance.py  # default
│   │       │   ├── round_robin.py, random_choice.py
│   │       │   ├── kv_cache_aware.py, prefix_aware.py, throughput_balance.py
│   │       │   ├── length_aware.py         # group ordering + predicted-load placement
│   │       │   └── inventory_deadline.py   # forecasts each option's effect on the nearest
│   │       │                               #   unready psrl buffer via forecast/plan.py's
│   │       │                               #   PlanSession; no configured cap
│   │       └── router.py        # Router (queue + gate + dispatch), AdmissionGate.
│   │                            #   Dispatch runs ONLY after a committed capacity release
│   │                            #   (complete / preempt / weight publish), never mid-segment
│   ├── training/
│   │   └── cost.py          # TrainCostModel, FixedTrainCost, AnalyticalTrainCost
│   ├── sync/
│   │   └── cost.py          # SyncCostModel, FixedSyncCost, BandwidthSyncCost
│   └── recompute/
│       └── cost.py          # RecomputeCostModel, Fixed/Analytical/TrainingForwardRecomputeCost
│                            #   (training_forward prices the log-prob pass as a
│                            #    FLOP-derived fraction of the fitted train step)
├── simulator/
│   ├── config.py            # SimConfig, SimResult (scg/scg_count/scg_curve/scg_w metrics)
│   ├── simulator.py         # Simulator (main driver)
│   └── dynamic_parallel/    # Fleet shape primitives. Nothing in Simulator constructs them yet
│       ├── plan.py          # FleetPlan (one TP degree per instance, GPU budget invariant)
│       └── fleet.py         # DynamicParallelFleet (re-shape the instance list at a weight update)
├── policy/
│   ├── base.py              # StalenessPolicy ABC, SimState, CompleteAction
│   ├── psrl.py              # PSRLPolicy (Reserve/Occupy/Consume)
│   ├── areal.py             # ArealPolicy (token bucket)
│   ├── roll.py              # RollPolicy (admission + GC discard)
│   ├── slime.py             # SlimePolicy (loop barrier)
│   └── verl.py              # VerlPolicy (window + bounded queue)
├── forecast/
│   ├── length.py            # LengthPredictor, OracleLength, NoisyOracleLength, build_length_predictor
│   ├── snapshot.py          # Resident, Arrival, InstanceState (+ from_instance for both backends)
│   └── drain.py             # InstanceForecast, DrainResult: analytic per-instance finish times and KV peak
├── cost/                    # Inference step time models (shared by all instances)
│   ├── base.py              # CostModel ABC
│   ├── analytical.py        # UnifiedRoofline, LinearLPS
│   ├── empirical.py         # PSRLFitted, DistServe
│   ├── artifact.py          # CostArtifact, ArtifactEntry, FitMetrics (shared by training models)
│   ├── kernel/
│   │   ├── ops/             # One KernelOp per measured family; see docs/kernel-ops.md
│   │   │   ├── base.py      # KernelOp protocol, OP_REGISTRY, op_for/ops_for_family/families/ops_in
│   │   │   ├── gemm.py      # GemmKernelOp (no floor; NB model_ops.GemmOp is a different concept)
│   │   │   ├── attention.py # Context/GenerationAttentionKernelOp (shared slice, fitted floor)
│   │   │   ├── elementwise.py  # ElementwiseKernelOp (measured, NOT summed into predict())
│   │   │   └── collective.py   # _CollectiveOp base + all_reduce/reduce_scatter/all_gather/
│   │   │                       #   broadcast/p2p. sol() once on the base; subclasses declare
│   │   │                       #   only traffic_factor. (world_size, nodes) are slice columns
│   │   ├── surface.py       # UtilitySurface protocol, PchipSurface, ShepardSurface
│   │   ├── model_ops.py     # ModelOps (per-step op list, sharded for one TP degree)
│   │   └── validate.py      # load_baseline, score_model (ComposedRollout vs vLLM JSONL)
│   ├── rollout/
│   │   ├── kv_cache.py      # KvCacheGeometry, kv_blocks_from_memory/engine_log (vLLM's own arithmetic)
│   │   ├── base.py          # RolloutCostModel ABC, RolloutWorkload
│   │   ├── advance.py       # Closed-form decode advance (moved from component/rollout/engine/)
│   │   ├── segment.py       # LatticeIntegrator (monotone numeric decode segment)
│   │   └── composed.py      # ComposedRollout (sum kernel latencies, no end-to-end regress)
│   └── training/            # Training cost models
│       ├── __init__.py      # TRAINING_MODEL_REGISTRY (hydraulis, fsdp)
│       ├── base.py          # TrainingCostModel, TrainingLatencyModel, TrainingMemoryModel ABCs
│       ├── analytical.py    # AnalyticalTrainingCost
│       ├── fsdp.py          # FsdpTrainingCost, FsdpLatencyModel, FsdpMemoryModel
│       ├── fsdp_inventory.py # FsdpInventory, fsdp_inventory (static parameter pool arithmetic)
│       ├── hydraulis.py     # HydraulisTrainingCost (Megatron 1F1B)
│       ├── inventory.py     # ParameterInventory, parameter_inventory, dtype_bytes
│       ├── latency.py       # LatencyFeatures, latency_features, robust_step_time
│       ├── linear.py        # LinearTrainingCost
│       └── microbatch.py    # BatchingMode, MicrobatchPlanner, verl_workload
├── config/
│   ├── __init__.py          # register_configs() for Hydra ConfigStore
│   ├── schema.py            # @dataclass structured configs (SimulateConfig, etc.)
│   └── defaults/
│       ├── simulate.yaml    # Top-level default config
│       ├── rollout_cost/    # roofline.yaml, psrl.yaml, distserve.yaml, composed.yaml
│       ├── policy/          # psrl.yaml, areal.yaml, roll.yaml, slime.yaml, verl.yaml
│       ├── workload/        # uniform.yaml, normal.yaml, bimodal.yaml, lognormal.yaml,
│       │                    #   powerlaw.yaml, dapo_trace.yaml
│       └── training_cost/   # fixed.yaml, analytical.yaml, hydraulis.yaml, linear.yaml, fsdp.yaml
├── profiling/
│   ├── __init__.py          # Public exports
│   ├── schemas.py           # SweepPoint, FitResult, FitQuality, profile configs
│   ├── runner.py            # ProfileRunner (subprocess lifecycle)
│   ├── vllm_profiler.py     # run_vllm_profile (server + HTTP sweep)
│   ├── megatron_profiler.py # run_megatron_profile (training step profiling)
│   └── fit.py               # Two-stage regression fitter → JSON
├── profiler/
│   ├── adapters/
│   │   ├── __init__.py      # Exports all adapters: VLLMSweepConfig, MegatronBridgeConfig, FsdpProfileConfig
│   │   ├── base/            # Shared base of megatron+fsdp (run_candidate, OOM helpers, sidecar).
│   │   │                    #   NOT a backend: named `base` like cost/*/base.py, because sitting
│   │   │                    #   beside megatron/fsdp/vllm made `training/` read as a fourth one
│   │   ├── fsdp/            # SimRL-owned FSDP2/HSDP profiling adapter
│   │   │   ├── __init__.py      # Public exports (safe on CLI import path; excludes launch/measure)
│   │   │   ├── collect.py       # collect_fsdp_record, collect_fsdp_records
│   │   │   ├── config.py        # FsdpProfileConfig
│   │   │   ├── execute.py       # run_fsdp_candidate, OOM_EXIT_CODE, oom_marker_paths
│   │   │   ├── launch.py        # torchrun child: model init, forward/backward, sidecar writes
│   │   │   ├── measure.py       # FsdpMeasurement, step timing and memory capture
│   │   │   └── prepare.py       # prepare_fsdp_candidate, read/write_candidate_manifest
│   │   ├── megatron/        # Megatron-Bridge profiling adapter
│   │   └── vllm/            # vLLM timing sweep adapter
│   ├── fsdp_search_space.py # FsdpCandidate, enumerate_fsdp_candidates
│   ├── hf.py                # model_descriptor_from_yaml, AutoConfig helpers
│   ├── io.py                # read_records, write_records, profile_digest
│   └── kernel/              # Singular, matching cost/kernel/ and its own output_dir/"kernel"
│       ├── __init__.py      # Exports KernelProfileConfig only; the 28-name PEP 562 lazy table
│       │                    #   is gone, the layering below replaces what it guarded
│       ├── config.py        # KernelProfileConfig, MeasurementConfig (no per-family fields)
│       ├── measurements.py  # PerfTableWriter (atomic parquet writes, columns from the op)
│       ├── introspect.py    # verify_shapes: declared shapes vs a meta-device module graph
│       ├── collective.py    # Driver: torchrun locally, ssh per node when nodes>1. Forwards
│       │                    #   every NCCL_* var (remote ~/.bashrc disagrees with the env script).
│       │                    #   Outside runner/: it launches rather than times in-process
│       ├── collective_launch.py  # torchrun child, named as a STRING in collective.py so no
│       │                    #   import graph follows a move. Skips vLLM parallel state when
│       │                    #   nodes>1: vLLM disables custom all-reduce across nodes, so it
│       │                    #   costs 60s for a duplicate of the nccl control table
│       ├── utils/           # timing.py (CUDA events; the one torch importer), model_dims.py,
│       │                    #   validators.py
│       ├── sweep/           # One *Sweep per family, parsed from its own YAML subtree.
│       │                    #   Reached by cost/kernel/ops/<op>.py:parse_sweep
│       ├── grid/            # Which shapes each sweep enumerates. TORCH-FREE (gated per module
│       │                    #   by tests/test_kernel_package_imports.py). gemm.py dispatches to
│       │                    #   families.py; elementwise.py's n_elements is re-derived on the
│       │                    #   query side by cost/training/training_ops.py, so it must be
│       │                    #   checkable without a GPU
│       └── runner/          # The measurement loops. base.py KernelRunner carries time()/row()
│                            #   but NOT the loop: attention needs one KV cache before it,
│                            #   optimizer an untimed step inside it, elementwise a vLLM config
│                            #   context around it. Reached by ops/<op>.py:measure
├── workload/
│   ├── distributions.py     # Synthetic bimodal/lognormal/powerlaw/uniform lengths
│   ├── trace.py             # Joint RequestGroup trace read/write
│   ├── fitting.py           # Dataset prompt deduplication and rendering
│   ├── inference.py         # Joint group collection via a served engine
│   ├── sharding.py          # Worker topology, shard merge, collection validation
│   ├── prefix_probe.py      # vLLM prefix-cache counters, for validating the block pool
│   ├── prefix_probe_run.py  # Four controlled prefix-cache scenarios against a server
│   └── grouped.py           # GroupedWorkload (the simulator's request source)
├── buffer/
│   └── queue.py             # CompletionBuffer, RecomputeQueue
├── telemetry/
│   ├── store.py             # JsonlStore (named JSONL stream store)
│   ├── writer.py            # SimTelemetry (owns the instance/train/request/rollout_step schemas)
│   └── psrl_log.py          # convert_psrl_logs, extract_workload_trace, measured_kv_blocks
├── visualizer/
│   └── plots.py             # SimPlotter (matplotlib offline; one impl for simulated and real runs)
└── cli/
    ├── main.py              # Hydra CLI entrypoint (@hydra.main); build_training_cost passes batching_mode
    ├── plot.py              # Re-render plots from a simulation's logs
    ├── psrl_plot.py         # Render instance_load from a real PSRL run's logs
    ├── compare.py           # srl-compare: quantify simulated-vs-real error
    ├── profile.py           # Profiling CLI: vllm, kernels, megatron, fsdp subcommands
    ├── regress.py           # Regression CLI: rollout, training (--latency-model hydraulis|fsdp)
    └── workload.py          # Collect and validate joint grouped workloads
deprecated/                  # Archived SCG scheduling line. NOT packaged (pyproject packages srl*),
│                            #   NOT collected (pytest testpaths=tests), NOT importable. See its README.md
├── README.md                # What moved, why, and where the measurements live
├── srl/                  # value/, abandoned router strategies and estimator, scg_value eviction,
│                            #   dynamic_parallel solver/planner, roll_doom policy and its YAML
├── tests/                   # Tests of the moved modules
├── examples/                # SCG sweep scripts, READMEs, make_heavy_tail_trace.py
├── outputs/                 # Comparison figures
└── docs/                    # SCG design notes and the heterogeneous-TP plan and spec
tests/
├── conftest.py
├── test_advance.py          # Closed-form advance correctness
├── test_e2e.py              # Full simulation tests for all 5 policies
├── test_config.py           # Hydra config + factory function tests
├── test_fit.py              # Two-stage regression fitter tests
└── test_fsdp_cli.py         # FSDP CLI wiring, Hydra config, and sweep-YAML tests
```

## Key Interfaces

### Request (srl/component/rollout/engine/request.py)
The unit of work every component handles. Import it from the engine package:
`from ..component.rollout.engine import Request`.
- `remaining`, `ctx`, `v_min`, `v_max` properties
- `add_tokens(version, n) -> None` appends to the current `Segment`
- `staleness_true()`, `staleness_reported()`

### CostModel (srl/cost/base.py)
- `step_time(n_tokens, t2, ctxsum, n_reqs) -> float`
- `decode_coeffs(n_d, ctxsum) -> (F, alpha, beta)`

### NativeInstance (srl/component/rollout/engine/instance.py)
- `add_request(req, t) -> bool` (the authority on admission; the router only gates eligibility)
- `advance_to(t_limit, version) -> (completed, reached_limit)`
- `advance_one_event(version) -> EngineEvent | None`, paired with `undo_last_event()` or
  `commit_last_event()`. One of the two must be called: a snapshot left open outlives its
  event, so the log keeps growing and an undo would reverse committed work
- `abort_all(requeue) -> list[Request]`, `cancel(reqs, keep_tokens, requeue)`.
  `requeue=False` hands the requests to the caller instead of the instance's own queue,
  which is how the router reclaims work to place elsewhere
- `committed_blocks()` / `free_blocks()`: a queued request holds no blocks yet, so a router
  comparing raw free blocks sees no difference between instances. Use both

### Router (srl/component/rollout/router/router.py)
- `enqueue(reqs, t)`, `remove(reqs)`, `dispatch(t) -> [(req, instance_id)]`
- `has_dispatchable()`: the engine loop must break out for this as well as for admit quota,
  or a saturated policy strands the queue
- `reclaim_preempted(t)`: vLLM's `preemption_notification_threshold`
- `release_group_pins(group_ids)`
- Strategies are duck-typed beyond `select`: `narrow_candidates`, `allow_version_bypass`,
  `select_context`, `rank_queued`, `pick_queued`, `observe_enqueue`. The router reads them through
  `getattr`, and `RoutingContext` carries the blocked versions, fleet TP bounds, and the clock.
  `docs/simulation/routing.md` has the table

### StalenessPolicy (srl/policy/base.py)
- `admit_quota(st: SimState) -> int`
- `on_admit(req, st) -> None`
- `on_complete(req, st) -> CompleteAction`
- `select_batch(st) -> list[Request] | None`
- `on_version_advance(version, st) -> None`
- `expire(st) -> list[Request]`
- `engine_version_after_train(st) -> int`

### Simulator (srl/simulator/simulator.py)
- `__init__(rollout_cost, policy, cfg, training_cost, sync_cost, recompute_cost, store)`
- `run() -> SimResult`

### CostModelFitter (srl/profiling/fit.py)
- `fit_cost_model(data_or_path, key) -> FitResult`
- `save_fit_result(result, path, key) -> Path`

### ProfileRunner (srl/profiling/runner.py)
- `launch() -> Popen`
- `wait_healthy() -> None`
- `shutdown() -> None`

## Import Dependency Order

1. `component.rollout.engine.request` (no internal deps)
2. `cost.rollout.advance` (no internal deps; moved from `component.rollout.engine.advance`)
3. `component.{training,sync,recompute}.cost` (no internal deps)
4. `cost.base` (no internal deps)
5. `cost.analytical`, `cost.empirical` (depend on cost.base)
6. `component.rollout.engine.event`, `.state`, `.snapshot` (depend on engine.request)
6a. `component.rollout.engine.undo` (no internal deps), `.kv` (depends on undo), `.eviction` (depends on
     engine.request; importing the package registers every eviction strategy, the only registration path)
7. `component.rollout.engine.instance` (depends on the other engine modules, `.kv`, `.undo`, cost.rollout.advance)
7a. `forecast.length` (depends on engine.state for the `TrajView` annotation only)
7b. `component.rollout.router.*` (depends on engine, forecast.length; importing the package registers every
     strategy, which is the only registration path)
8. `policy.base` (depends on engine.request)
9. `policy.*` (depend on policy.base, engine.request)
10. `workload.*` (depends on engine.request)
11. `buffer.queue` (depends on engine.request)
12. `simulator.config` (depends on engine.request, workload.grouped)
13. `simulator.simulator` (depends on engine, engine.eviction, router, policy, cost, component.*)
13a. `simulator.dynamic_parallel.{plan,fleet}` (depend on engine, and on the router under `TYPE_CHECKING`
     only. Nothing in `simulator.simulator` imports them.)
14. `config.schema` (no internal deps)
15. `config.__init__` (depends on config.schema)
16. `profiling.schemas`, `profiling.runner` (no internal deps)
17. `profiling.fit` (depends on profiling.schemas, numpy, scipy)
18. `profiling.vllm_profiler`, `profiling.megatron_profiler` (depend on profiling.runner, profiling.schemas)
19. `profiler.kernel.utils.{model_dims,validators}` (depend on profiler.schema)
19a. `profiler.kernel.sweep.*` (depends on profiler.kernel.utils.validators)
19b. `profiler.kernel.grid.*` (depends on profiler.kernel.{sweep,utils.model_dims}; `grid.gemm`
     dispatches through `grid.families`. MUST stay torch-free, gated per module by
     `tests/test_kernel_package_imports.py`, because the cost path imports `grid.gemm` and
     `grid.attention` to build its queries.)
20. `cost.kernel.surface` (depends on cost.kernel.interp)
20a. `cost.kernel.ops.*` (depends on cost.kernel.{hardware,sol,interp}; MUST stay torch-free.
     `ops/__init__` imports every op module, so importing `ops.base` populates OP_REGISTRY --
     that is the only registration path. `parse_sweep` imports `profiler.kernel.sweep` lazily;
     `measure` imports `profiler.kernel.runner.*` lazily.)
21. `cost.kernel.model_ops` (depends on profiler.kernel.grid.gemm, cost.kernel.database)
22. `cost.rollout.segment` (no internal deps)
23. `cost.rollout.composed` (depends on cost.kernel.database, cost.rollout.base, cost.rollout.segment)
24. `cost.kernel.validate` (depends on cost.rollout.base)
25. `cost.artifact` (no internal deps)
26. `cost.training.base` (no internal deps)
27. `cost.training.inventory`, `cost.training.fsdp_inventory` (depend on profiler.schema)
28. `cost.training.latency`, `cost.training.microbatch` (depend on cost.training.base, profiler.schema)
29. `cost.training.{analytical,fixed,hydraulis,linear}` (depend on cost.training.base)
30. `cost.training.fsdp` (depends on cost.training.base, cost.artifact, cost.training.latency, cost.training.microbatch, cost.training.fsdp_inventory)
31. `cost.training.__init__` (re-exports; TRAINING_MODEL_REGISTRY)
32. `profiler.schema`, `profiler.io`, `profiler.hf` (no internal deps)
33. `profiler.fsdp_search_space` (depends on profiler.schema)
34. `profiler.adapters.base` (depends on profiler.schema)
35. `profiler.adapters.fsdp.config` (depends on profiler.schema, profiler.hf)
36. `profiler.adapters.fsdp.prepare` (depends on profiler.fsdp_search_space, profiler.adapters.fsdp.config)
37. `profiler.adapters.fsdp.execute` (depends on profiler.adapters.base, profiler.adapters.fsdp.prepare)
38. `profiler.adapters.fsdp.collect` (depends on profiler.schema, profiler.io)
39. `profiler.adapters.fsdp.__init__` (re-exports safe CLI subset)
40. `profiler.adapters.__init__` (re-exports vllm, megatron, fsdp adapters)
40a. `profiler.kernel.runner.*` (depends on profiler.kernel.{config,grid,measurements,utils.timing};
     the GPU side -- every runner imports torch inside `run()`, so `runner/__init__` itself stays
     cheap. `runner.attention_training` also reaches `profiler.adapters.megatron.launch` for the
     flash-attn version normalization, which must run before transformer_engine is imported.)
40b. `profiler.kernel.collective` (depends on profiler.kernel.{config,measurements,utils.timing},
     cost.kernel.ops.base. Outside `runner/`: it launches torchrun rather than timing in-process,
     and names `profiler.kernel.collective_launch` as a string literal.)
41. `telemetry.store` (no internal deps)
41a. `telemetry.writer` (depends on telemetry.store, component.rollout.engine)
41b. `telemetry.psrl_log` (depends on telemetry.store; parses a real run's logs into the same streams)
41c. `visualizer.plots` (depends on telemetry.store only, never on `SimResult`)
42. `cli.main` (depends on everything except profiling)
42a. `cli.plot` (depends on telemetry.store, visualizer.plots)
42b. `cli.psrl_plot` (depends on telemetry.psrl_log, telemetry.store, visualizer.plots)
43. `cli.profile` (depends on profiling, cost.kernel.*, profiler.adapters.*)
44. `cli.regress` (depends on cost.training, cost.rollout, profiler.io)

## Naming Notes

Two unrelated concepts both use the word "segment":

- `Segment` in `engine/request.py` is a stretch of generated tokens produced
  under one weight version. It is what the staleness metrics read.
- A decode run in `cost/rollout/advance.py` (moved from `component/rollout/engine/`) is a
  stretch of steps over a fixed batch with no prefill mixed in. It is a unit of time
  integration, not data. `LatticeIntegrator` in `cost/rollout/segment.py` integrates it
  numerically when the step cost has no closed form.

The closed-form helpers live in `advance.py` so the word `Segment` belongs to
the request data model alone.
