# Architecture

SimRL answers questions about a running asynchronous RL system without running it. The answer is
assembled in three stages, each a separate package group with a documented seam to the next:
measurement, modelling, and simulation.

---

## The three layers

**Profiler adapters** (`srl/profiler/`) drive real systems, a vLLM server, Megatron-Bridge, an
FSDP2 launcher or individual CUDA kernels, and normalise what they observe into versioned
JSONL or parquet. An adapter records and never models.

**Cost models** (`srl/cost/`) turn those records into latency and memory predictions. A fitted
model is a small coefficient set plus the input range it is valid over. A composed model sums
measured kernels. Either way, a cost model interpolates and never measures.

**The simulator** (`srl/simulator/`, `srl/component/`, `srl/policy/`, `srl/workload/`) runs a
rollout and training loop over simulated time, asking the cost models for every engine step and
training step. It never touches a GPU.

The seams are on-disk and versioned: `ProfileRecord` JSONL between adapters and fitters, and
`CostArtifact` JSON or kernel parquet between fitters and the simulator. {doc}`../reference/schemas`
defines all of them.

:::{image} /_static/img/simrl-architecture-plain.png
:alt: How SimRL is built
:width: 100%
:align: center
:::

---

## Package tree

```text
srl/
├── _logging.py                  srl_logger, the one logger for the package
├── buffer/queue.py              CompletionBuffer, RecomputeQueue
├── component/                   One subpackage per stage of the RL loop
│   ├── rollout/
│   │   ├── engine/              Request, NativeInstance, events, KV pool, eviction, undo
│   │   │   ├── request.py       Request, Segment, ReqStatus (the shared vocabulary)
│   │   │   ├── instance.py      NativeInstance: continuous batching, KV, preemption
│   │   │   ├── event.py         EventKind, EngineEvent
│   │   │   ├── state.py         StepStat, InstanceLoad, InstanceView, TrajView
│   │   │   ├── snapshot.py      ReqSnapshot, InstanceSnapshot (rollback)
│   │   │   ├── eviction/        vllm_lru, longest_ctx, latest_deadline
│   │   │   ├── kv/              BlockPool, KVCacheManager, free list, provenance lineage
│   │   │   └── undo/            Reversible mutations and the per-event undo log
│   │   ├── engine_cpp/          C++ engine (pybind11), built to srl/_engine_cpp*.so
│   │   └── router/              RouterQueue, Router, AdmissionGate, eight strategies
│   ├── recompute/cost.py        Log-prob recomputation cost models
│   ├── sync/cost.py             Weight sync cost models
│   └── training/                Namespace for training-loop helpers
├── config/                      Hydra: schema.py, defaults/, policies/ presets
├── cost/
│   ├── artifact.py              CostArtifact, ArtifactEntry, FitMetrics
│   ├── kernel/                  Kernel op registry, perf tables, SOL, utility surfaces, holdout
│   ├── rollout/                 roofline, psrl, distserve, composed, segment advance
│   └── training/                fixed, analytical, linear, hydraulis, fsdp, composed
├── forecast/                    Per-instance drain and training-pipeline forecasts
├── policy/                      StalenessPolicy ABC, twelve presets, shared rule primitives
├── profiler/
│   ├── schema.py, io.py, hf.py  ProfileRecord and its validated I/O
│   ├── adapters/                vllm, megatron, fsdp, and the shared base
│   ├── kernel/                  Kernel sweeps: config, grid, runner, sweep, measurements
│   ├── search_space.py          Training candidate enumeration and constraints
│   └── fsdp_search_space.py     FSDP mesh candidate enumeration
├── simulator/                   SimConfig, SimResult, Simulator, SchedulingRuntime
│   └── dynamic_parallel/        FleetPlan and DynamicParallelFleet (not wired into Simulator)
├── telemetry/                   JsonlStore, SimTelemetry, PSRL log conversion
├── visualizer/plots.py          SimPlotter: one implementation for simulated and real runs
└── cli/                         The seven console scripts
```

Two sibling trees feed it: `workflow/` holds agent-oriented validation workflows, `examples/`
holds profiling configs and analysis scripts, and `third_party/` holds pinned upstream checkouts.

---

## Dependency direction

An import arrow means "imports at module level". The graph is a DAG with two deliberate
exceptions, each harmless because only torch-free leaves are involved.

```text
_logging  <-  everything

workload        depends on: _logging, (trace format only)
profiler        depends on: _logging, cost.kernel (grid/model_dims for shape derivation)
cost            depends on: _logging, profiler.schema/io/hf, profiler.kernel.grid.*, profiler.kernel.utils.model_dims
policy          depends on: component.rollout.engine
component       depends on: cost.rollout (instance prices a step), forecast (length_aware)
forecast        depends on: component.rollout.engine, cost, psrl
simulator       depends on: component, cost, forecast, policy, telemetry, workload
telemetry       depends on: _logging, component.rollout.engine, workload (trace writer)
visualizer      depends on: telemetry.store
cli             depends on: config, cost, policy, profiler, simulator, telemetry, visualizer, workload
config          depends on: policy (typed presets reference the policy classes)
```

Two pairs point both ways:

`cost` and `profiler`
: `cost` reads `profiler.schema`, `profiler.io`, `profiler.hf` and the shape derivation in
  `profiler.kernel.grid` and `profiler.kernel.utils.model_dims`. In the other direction
  `profiler.adapters` writes `ProfileRecord` and calls the cost fitters. The profiler modules that
  `cost` imports must stay importable without `torch`, which
  `tests/test_kernel_package_imports.py` enforces module by module.

`component` and `forecast`
: The `length_aware` and `inventory_deadline` router strategies import `srl.forecast`, and
  `forecast.fleet` and `forecast.plan` import the engine's request views. `forecast.snapshot` reads
  an instance through the read-only `TrajView` projection, so the forecast cannot mutate the
  engine whichever backend holds it.

Two rules hold the layering together:

- **The cost path stays free of the GPU stack.** A simulation prices kernels from parquet and must
  not import `torch` or vLLM. Every module on that path is gated by a fresh-interpreter test.
- **Registration happens as an import side effect, in one place.** `cost/kernel/ops/__init__.py`
  imports every op module, `profiler/kernel/grid/families.py` imports every family module at its
  tail, and `component/rollout/router/__init__.py` imports the strategies package. Each registry
  has exactly one population path, so import order cannot change what is registered.

---

## The event loop

`Simulator.run` repeats one training version at a time:

```text
stage -> dispatch -> generate (event queue) -> complete -> select batch
      -> recompute + train + sync -> publish -> repeat
```

`stage`
: `_stage` admits prompt groups from the dataset into the router, up to the ingress window
  (`ingress_window_groups`, derived from the policy preset). Each weight publish opens one more
  `batch_size` of groups. Ingress is a flow limit, separate from the policy's staleness rule.

`dispatch`
: `Router.dispatch` places queued requests onto instances that can take them. Placement runs only
  after a committed capacity release, meaning a completion, a preemption or a weight publish, and
  never in the middle of a decode segment. The router holds requests that no instance can accept
  yet, so a busy fleet produces queue depth rather than a rejected admission. `Simulator` treats a
  group as admitted when its first response is placed, and calls `policy.on_admit` with the whole
  group.

`generate`
: `_run_engine` drives all instances from one priority queue ordered by event time, always
  advancing the instance with the earliest next event. Each event advances one instance by a
  decode step, a prefill chunk, a preemption or a completion. The loop stops at a deadline (the end
  of the training step, when training runs concurrently) or as soon as the policy reports a batch.

`complete`
: A finished request leaves `inflight`. Its group enters `ready_groups` only when every response
  has finished. `policy.on_group_complete` may keep or drop the group.

`select batch`
: `_next_batch` asks the policy's `can_sample`, then `take_batch`, which returns `batch_size`
  complete groups. Selecting by group and never by response count is what keeps a batch trainable
  under GRPO.

`recompute + train + sync`
: `_train_cycle` prices the batch once: a `TrainingWorkload` built from the batch's sequence
  lengths, the `TrainingCostModel` for latency and peak memory, `RecomputeCostModel` for the
  log-prob pass, and `SyncCostModel` when the policy says the step ends with a weight sync. Peak
  memory above `training.gpu_memory_mib` raises immediately. Rollout then keeps generating for the
  whole duration, with `rollout_version` unchanged until the publish.

`publish`
: `_publish_weights` advances `train_version`, applies the policy's cancels, asks
  `rollout_version_after_train` and `pause_mode`. Under `abort`, in-flight requests are interrupted
  and re-enqueued, unstarted staged groups rebase to the new version, and `weight_load_time` is
  charged before generation resumes. Under `in_place`, requests keep running on their existing KV.

The `SimResult` reports `throughput` (tokens per simulated second over consumed requests) and
`scg(s)` (tokens per second from trajectories whose true staleness is at most `s`).

---

## The closed-form decode advance

Simulating every token would make cost grow with output length, which is expensive at the mean
output lengths real runs reach. A pure-decode segment, a run of steps over a fixed batch with no
prefill and no arrival or departure, is integrated in closed form instead.

Each model exposes `decode_coeffs(n_d, ctxsum) -> (F, alpha, beta)`, and the step cost is modelled
as `tau(k) = max(F, alpha + beta * k)` for KV history `k`. `advance.elapsed`,
`advance.steps_within` and `advance.crossing_step` in `srl/cost/rollout/advance.py` sum that
exactly, and they use `Fraction` arithmetic when given rational inputs. `ComposedRollout` can
instead integrate numerically with `LatticeIntegrator` (`srl/cost/rollout/segment.py`) when its
utility surface is too steep for the closed form. `decode_coeffs` is an abstract method on
`RolloutCostModel`, so a new rollout model cannot omit it.

---

## Speculative events and undo

The engine loop must be able to stop in the middle of an event, because a training deadline or a
ready batch can land between two events. Rather than recompute from a snapshot, an event is
executed against the instance, and its mutations are recorded in an undo log. If the loop stops,
`undo_last_event` reverses it. If the loop continues, `commit_last_event` drops the log.

`SIMRL_UNDOABLE_EVENTS` chooses which event kinds are speculative. The default makes
completions, preemptions and iteration ends reversible, which is exact. Narrowing it to
`request_complete` executes the other two permanently and costs accuracy at a deadline, so it is a
performance knob and not a semantics knob. `request_complete` must always be present, because
completions decide batch order.

---

## The C++ engine

`srl/component/rollout/engine_cpp/` is a pybind11 module built to `srl/_engine_cpp*.so`. It
reimplements `NativeInstance`, its KV pool, the free list, the request model, the undo log and the
closed-form helpers in C++, and mirrors CPython's semantics for the arithmetic its keys depend on:
`float // float` and half-to-even `round` are reimplemented in `bindings.cc` rather than taken from
`std`.

The two engines are selected per run:

- `SIMRL_BACKEND=cpp`, the default, uses `CppInstance` and `CppRequest`.
- `native` uses `NativeInstance` and `Request`.
- `auto` uses C++ when the module imports and no configured component is `python_engine_only`,
  otherwise Python.

The simulator wraps whatever rollout cost model was selected in `PyWrappedCostModel`, so the C++
engine calls the same Python `predict` and `decode_coeffs`. A model that publishes `cpp_term_spec`
is served from C++ side memos and reaches Python only on a miss. The eviction strategy object is the
same one the Python engine uses, so the two backends cannot disagree about eviction policy.

The module is committed for CPython 3.12 on Linux x86_64. Building it needs CMake 3.15, pybind11,
and a C++17 compiler:

```bash
cmake -S srl/component/rollout/engine_cpp -B srl/component/rollout/engine_cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build srl/component/rollout/engine_cpp/build -j
```

The build writes `_engine_cpp<abi>.so` in the build directory. Install it as
`srl/_engine_cpp<abi>.so`, moving it into place rather than copying over a file a running
simulation has mapped.

---

## Forecasting

`srl/forecast/` answers what a scheduling decision can know before it is made:

- `drain.py` replays one instance's engine without running it and predicts when every resident
  finishes, matching `NativeInstance`'s segment and prefill-chunk order.
- `length.py` provides the length predictors: `oracle` reads the trace's target length, and
  `noisy_oracle` perturbs it deterministically to test how much prediction error a policy survives.
- `snapshot.py` holds the read-only instance view the forecast replays.
- `pipeline.py` chains training buffers through the loop to get each one's ready and training-end
  times.
- `fleet.py` and `plan.py` price one placement or one admission against every instance and the
  pending buffers in one session.

`SchedulingRuntime` in `srl/simulator/runtime.py` is the read-only facade a strategy binds to. A
strategy opts in by declaring `bind_runtime`. The `inventory_deadline` router strategy is the
current user, and it requires `policy=psrl` because it forecasts against the PSRL inventory.

---

## Guards

A run that cannot make progress returns instead of hanging. `rollout.livelock_rounds` bounds
consecutive rounds with no state change. `SimResult.livelocked` reports it.

---

## Programmatic API

The simulator is importable without Hydra:

```python
from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import FixedTrainingCost
from srl.policy.areal import ArealPolicy
from srl.simulator import SimConfig, Simulator
from srl.workload.distributions import bimodal
from srl.workload.grouped import GroupedWorkload

lengths = bimodal(200, short_len=50, long_len=500, long_frac=0.25, seed=42)
cfg = SimConfig(
    batch_size=8,
    staleness_horizon=2,
    n_versions=5,
    n_instances=2,
    workload=GroupedWorkload.from_lengths([512], lengths),
    log_telemetry=False,
)
sim = Simulator(
    rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_d=1e-7),
    policy=ArealPolicy(),
    cfg=cfg,
    training_cost=FixedTrainingCost(0.5, 0),
    backend="native",
)
result = sim.run()
print(result.versions_done, result.n_consumed, result.throughput)
```

`SimConfig` validates its own consistency. `group_size` must equal the workload's arity, and
`undoable_events` must contain `request_complete`. `SimResult` exposes `consumed`, `dropped`,
`dropped_reasons`, `sim_time`, `versions_done`, `livelocked`, `train_timestamps`,
`train_peak_memory_bytes`, and the properties `n_consumed`, `staleness_true_values`,
`staleness_reported_values`, `max_staleness_true`, `max_staleness_reported` and `throughput`,
plus `scg(s)`, `scg_count(s)`, `scg_curve()` and `staleness_report()`.

Importing this module needs the `psrl` package, as {doc}`../reference/limitations` explains.
