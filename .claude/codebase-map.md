# Codebase Map

## System Architecture

DryRun-RL is a discrete-event simulator for async disaggregate RL training.
The simulation loop: admit → generate (closed-form) → complete → select batch →
(recompute) → train → weight update → abort/re-prefill.

Code is organised around the RL loop components. Everything that models a stage
of the loop lives under `dryrun/component/` (rollout, training, sync, recompute),
while `dryrun/simulator/` drives them and `dryrun/policy/` decides staleness.

Configuration is Hydra-based: structured `@dataclass` configs in `schema.py`,
YAML defaults in `config/defaults/`, CLI overrides via `dryrun key=value`.

Profiling pipeline: vLLM/Megatron benchmarks → CSV → two-stage regression
fitter → JSON cost model → `PSRLFitted.from_json()`.

## Directory Tree

```
dryrun/
├── __init__.py
├── _logging.py              # dryrun_logger
├── component/               # One subpackage per RL loop stage
│   ├── rollout/
│   │   ├── engine/
│   │   │   ├── request.py       # ReqStatus, Segment, Request (shared vocabulary)
│   │   │   ├── instance.py      # NativeInstance (continuous batching, KV, preemption)
│   │   │   ├── event.py         # EventKind, EngineEvent
│   │   │   ├── state.py         # StepStat, InstanceLoad
│   │   │   └── snapshot.py      # ReqSnapshot, InstanceSnapshot (event rollback)
│   │   └── router/              # (placeholder for routing strategies)
│   ├── training/
│   │   └── cost.py          # TrainCostModel, FixedTrainCost, AnalyticalTrainCost
│   ├── sync/
│   │   └── cost.py          # SyncCostModel, FixedSyncCost, BandwidthSyncCost
│   └── recompute/
│       └── cost.py          # RecomputeCostModel, FixedRecomputeCost, AnalyticalRecomputeCost
├── simulator/
│   ├── config.py            # SimConfig, SimResult
│   └── simulator.py         # Simulator (main driver)
├── policy/
│   ├── base.py              # StalenessPolicy ABC, SimState, CompleteAction
│   ├── psrl.py              # PSRLPolicy (Reserve/Occupy/Consume)
│   ├── areal.py             # ArealPolicy (token bucket)
│   ├── roll.py              # RollPolicy (admission + GC discard)
│   ├── slime.py             # SlimePolicy (loop barrier)
│   └── verl.py              # VerlPolicy (window + bounded queue)
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
│   │   │   └── collective.py   # AllReduceKernelOp (world_size is a slice column)
│   │   ├── surface.py       # UtilitySurface protocol, PchipSurface, ShepardSurface
│   │   ├── model_ops.py     # ModelOps (per-step op list, sharded for one TP degree)
│   │   └── validate.py      # load_baseline, score_model (ComposedRollout vs vLLM JSONL)
│   ├── rollout/
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
│       ├── cost_model/      # roofline.yaml, psrl_fitted.yaml, distserve.yaml
│       ├── policy/          # psrl.yaml, areal.yaml, roll.yaml, slime.yaml, verl.yaml
│       ├── workload/        # uniform.yaml, bimodal.yaml, lognormal.yaml, powerlaw.yaml
│       └── train_cost/      # fixed.yaml, analytical.yaml, hydraulis.yaml, linear.yaml, fsdp.yaml
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
│   │   ├── fsdp/            # DryRun-owned FSDP2/HSDP profiling adapter
│   │   │   ├── __init__.py      # Public exports (safe on CLI import path; excludes launch/measure)
│   │   │   ├── collect.py       # collect_fsdp_record, collect_fsdp_records
│   │   │   ├── config.py        # FsdpProfileConfig
│   │   │   ├── execute.py       # run_fsdp_candidate, OOM_EXIT_CODE, oom_marker_paths
│   │   │   ├── launch.py        # torchrun child: model init, forward/backward, sidecar writes
│   │   │   ├── measure.py       # FsdpMeasurement, step timing and memory capture
│   │   │   └── prepare.py       # prepare_fsdp_candidate, read/write_candidate_manifest
│   │   ├── megatron/        # Megatron-Bridge profiling adapter
│   │   ├── training/        # Shared training-adapter base (run_candidate, OOM helpers)
│   │   └── vllm/            # vLLM timing sweep adapter
│   ├── fsdp_search_space.py # FsdpCandidate, enumerate_fsdp_candidates
│   ├── hf.py                # model_descriptor_from_yaml, AutoConfig helpers
│   ├── io.py                # read_records, write_records, profile_digest
│   └── kernels/
│       ├── shapes.py        # transformer_gemm_shapes, logits_gemm_shape (GEMM shape arithmetic)
│       └── elementwise.py   # Raw fused vLLM op timing (NOT the CustomOp module: 13x slower)
├── workload/
│   ├── distributions.py     # Synthetic bimodal/lognormal/powerlaw/uniform lengths
│   ├── trace.py             # Joint RequestGroup trace read/write
│   ├── fitting.py           # Dataset prompt deduplication and rendering
│   ├── inference.py         # Joint group collection via a served engine
│   ├── sharding.py          # Worker topology, shard merge, collection validation
│   └── grouped.py           # GroupedWorkload (the simulator's request source)
├── buffer/
│   └── queue.py             # CompletionBuffer, RecomputeQueue
├── viz/
│   └── plots.py             # SimPlotter (matplotlib offline)
└── cli/
    ├── main.py              # Hydra CLI entrypoint (@hydra.main); build_training_cost passes batching_mode
    ├── profile.py           # Profiling CLI: vllm, kernels, megatron, fsdp subcommands
    └── regress.py           # Regression CLI: rollout, training (--latency-model hydraulis|fsdp)
tests/
├── conftest.py
├── test_advance.py          # Closed-form advance correctness
├── test_e2e.py              # Full simulation tests for all 5 policies
├── test_config.py           # Hydra config + factory function tests
├── test_fit.py              # Two-stage regression fitter tests
└── test_fsdp_cli.py         # FSDP CLI wiring, Hydra config, and sweep-YAML tests
```

## Key Interfaces

### Request (dryrun/component/rollout/engine/request.py)
The unit of work every component handles. Import it from the engine package:
`from ..component.rollout.engine import Request`.
- `remaining`, `ctx`, `v_min`, `v_max` properties
- `add_tokens(version, n) -> None` appends to the current `Segment`
- `staleness_true()`, `staleness_reported()`

### CostModel (dryrun/cost/base.py)
- `step_time(n_tokens, t2, ctxsum, n_reqs) -> float`
- `decode_coeffs(n_d, ctxsum) -> (F, alpha, beta)`

### NativeInstance (dryrun/component/rollout/engine/instance.py)
- `add_request(req, t) -> bool`
- `advance_to(t_limit, version) -> (completed, reached_limit)`
- `advance_one_event(version) -> EngineEvent | None`, paired with `undo_last_event()`
- `abort_all() -> list[Request]`
- `cancel(reqs, keep_tokens) -> list[Request]`

### StalenessPolicy (dryrun/policy/base.py)
- `admit_quota(st: SimState) -> int`
- `on_admit(req, st) -> None`
- `on_complete(req, st) -> CompleteAction`
- `select_batch(st) -> list[Request] | None`
- `on_version_advance(version, st) -> None`
- `expire(st) -> list[Request]`
- `engine_version_after_train(st) -> int`

### Simulator (dryrun/simulator/simulator.py)
- `__init__(rollout_cost, policy, cfg, training_cost, sync_cost, recompute_cost, store)`
- `run() -> SimResult`

### CostModelFitter (dryrun/profiling/fit.py)
- `fit_cost_model(data_or_path, key) -> FitResult`
- `save_fit_result(result, path, key) -> Path`

### ProfileRunner (dryrun/profiling/runner.py)
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
7. `component.rollout.engine.instance` (depends on the other engine modules, cost.rollout.advance, cost.base)
8. `policy.base` (depends on engine.request)
9. `policy.*` (depend on policy.base, engine.request)
10. `workload.*` (depends on engine.request)
11. `buffer.queue` (depends on engine.request)
12. `simulator.config` (depends on engine.request, workload.grouped)
13. `simulator.simulator` (depends on engine, policy, cost, component.*)
14. `config.schema` (no internal deps)
15. `config.__init__` (depends on config.schema)
16. `profiling.schemas`, `profiling.runner` (no internal deps)
17. `profiling.fit` (depends on profiling.schemas, numpy, scipy)
18. `profiling.vllm_profiler`, `profiling.megatron_profiler` (depend on profiling.runner, profiling.schemas)
19. `profiler.kernels.shapes` (no internal deps)
20. `cost.kernel.surface` (depends on cost.kernel.interp)
20a. `cost.kernel.ops.*` (depends on cost.kernel.{hardware,sol,interp}; MUST stay torch-free.
     `ops/__init__` imports every op module, so importing `ops.base` populates OP_REGISTRY --
     that is the only registration path. `parse_sweep`/`measure` import the profiler lazily.)
21. `cost.kernel.model_ops` (depends on profiler.kernels.shapes, cost.kernel.database)
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
34. `profiler.adapters.training` (depends on profiler.schema)
35. `profiler.adapters.fsdp.config` (depends on profiler.schema, profiler.hf)
36. `profiler.adapters.fsdp.prepare` (depends on profiler.fsdp_search_space, profiler.adapters.fsdp.config)
37. `profiler.adapters.fsdp.execute` (depends on profiler.adapters.training, profiler.adapters.fsdp.prepare)
38. `profiler.adapters.fsdp.collect` (depends on profiler.schema, profiler.io)
39. `profiler.adapters.fsdp.__init__` (re-exports safe CLI subset)
40. `profiler.adapters.__init__` (re-exports vllm, megatron, fsdp adapters)
41. `viz.plots` (depends on simulator)
42. `cli.main` (depends on everything except profiling)
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
