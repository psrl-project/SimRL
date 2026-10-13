# Architecture

DryRun-RL separates three concerns that are usually tangled together in a real RL
training loop: measurement, modelling, and simulation. This page describes how those
three pieces fit together and the decisions behind the design.

---

## The three-layer separation

**Profiler adapters** drive real systems (vLLM, Megatron-Bridge, an FSDP2 launcher, or
individual CUDA kernels) and normalise what they observe into versioned JSONL. An adapter
never models anything. It records.

**Cost models** are fitted from that JSONL and reduced to a small set of coefficients
plus the input range they are valid over. A cost model never measures anything. It
interpolates.

**The simulator** drives a rollout/training loop under a staleness policy, querying
cost models for latency and peak memory as it goes. The simulator never touches a real
GPU.

The boundary between the first two layers is a documented JSONL schema (`ProfileRecord`
in `dryrun/profiler/schema.py`). Adding a new backend means writing a new adapter, and
adding a new cost model means writing a new fitter. Neither change requires touching the
other side.

---

## The event loop

The simulator's main loop advances one training version at a time. Within each version,
rollout and training interleave around a shared event queue:

```
stage → place/admit → generate → complete → select batch → (recompute) → train → sync → publish
```

`stage`
: Dataset groups enter the router independently of policy capacity. Startup exposes
  `(max_staleness + 1) * batch_size` groups and each publish exposes one more batch.

`place/admit`
: The staleness policy's `admit_quota` gates a staged group's first placement.
  That placement atomically admits every response in the group and calls `on_admit`
  once. Placement is also gated by three configurable flags on each engine
  instance: `reject_if_kv_full` (refuse when the KV cache has no free blocks),
  `reject_if_waiting` (refuse when the instance already has queued requests), and
  `reject_if_running_full` (refuse when the running queue is at capacity).

`generate`
: Admitted requests enter the continuous-batching engine. Each step prefills new requests,
  then decodes all running ones by one token. The engine loop runs on a priority
  queue ordered by event time, always advancing the instance with the earliest next event.

`complete`
: When a request finishes generation (`generated == target_len`), it moves from
  `inflight` to `ready`. The staleness policy's `on_complete` hook can drop it
  (`CompleteAction.DROP`) or retain it for the next training batch.

`select batch`
: The policy's `peek_batch` and `take_batch` methods return the next training batch
  when one is ready. The simulator calls `peek_batch` to check without consuming policy
  state, then `take_batch` to commit.

`recompute`
: If any request in the batch was preempted and interrupted, its prefix tokens must be
  re-prefilled. The `RecomputeCostModel` charges for this. `recompute_time` defaults to
  zero. Set it to a non-zero value when re-prefill is a material cost in your system.

`train`
: The `TrainingCostModel` returns a `TrainingCostEstimate` with `latency_s` and
  `peak_memory_bytes`. If `training_gpu_memory_bytes` is set and the predicted peak
  exceeds it, the simulator raises immediately rather than accepting a configuration
  that would OOM in practice.

`sync`
: The `SyncCostModel` charges for weight synchronisation between the training and
  rollout processes. `sync_time` defaults to zero.

`publish`
: `train_version` increments. The staleness policy's `rollout_version_after_train` hook
  decides whether the rollout engine's visible weight version also advances (it always
  does for most policies, while `slime` defers it by `update_weights_interval` steps).
  In-flight requests whose partial tokens predate the new version are interrupted, and the
  policy's `on_interrupt` hook handles them. Unadmitted, zero-token groups still in the router
  rebase to this new rollout version before their first placement.

---

## The closed-form pure-decode advance

Simulating every decode token individually would make simulation cost proportional to
output length, which is a problem for long DAPO-style outputs at about 7800 mean tokens each. For a
pure-decode segment (no new prefills, same running set), the simulator uses a closed-
form integration instead.

The roofline decode cost per step is modelled as `tau(k) = max(F, alpha + beta*k)`,
where `k` is the KV history length (prompt + previously generated tokens). For a
segment where all requests' contexts grow together, the sum of step times over any
contiguous range has a closed-form expression. The `advance` function in
`dryrun/cost/rollout/advance.py` evaluates this closed form and returns the simulated
time for the entire segment in one call.

The closed form applies whenever a decode segment's running set is unchanged from step
to step, meaning no arrivals, no completions, and no preemptions. As soon as one of those events
occurs, the simulator falls back to a per-step walk for the steps around the event.

---

## Livelock detection

A misconfigured run can deadlock: the policy never releases a training batch because no
request finishes, and no request finishes because none was admitted. The simulator
counts consecutive rounds without progress (`idle_rounds`) against a limit
(`livelock_rounds`, default 50). When the limit is reached, `run()` returns a
`SimResult` with `livelocked=True` rather than hanging. A second guard (`max_engine_iters`,
default 10,000) limits the total engine steps to catch infinite loops.

---

## Package map

| Package | Responsibility |
|---|---|
| `dryrun/simulator/` | Event loop, `SimConfig`, `SimResult` |
| `dryrun/component/rollout/` | Request state, `NativeInstance`, closed-form advance |
| `dryrun/component/recompute/` | Re-prefill cost models |
| `dryrun/component/sync/` | Weight sync cost models |
| `dryrun/component/training/` | Training workload helpers |
| `dryrun/policy/` | PSRL, AReaL, ROLL, Slime, verl staleness policies |
| `dryrun/cost/rollout/` | Roofline, PSRL, DistServe, and ComposedRollout inference models |
| `dryrun/cost/training/` | Hydraulis and FSDP latency and memory models |
| `dryrun/cost/kernel/` | Kernel op registry, SOL formulas, perf-table database, holdout |
| `dryrun/cost/artifact.py` | `CostArtifact` and `ArtifactEntry` envelope |
| `dryrun/profiler/` | `ProfileRecord` schema, constrained candidate generation, adapters |
| `dryrun/profiler/adapters/` | vLLM, Megatron-Bridge, FSDP2, and kernel adapters |
| `dryrun/telemetry/` | JSONL stream writers (`train`, `request`, `rollout_step`, `instance`) |
| `dryrun/visualizer/` | Offline plots from JSONL streams |
| `dryrun/workload/` | Parametric distributions and joint group traces |
| `dryrun/config/` | Hydra config dataclasses and defaults |
| `dryrun/cli/` | Console scripts: `dryrun`, `dryrun-profile`, `dryrun-regress`, `dryrun-plot`, `dryrun-workload` |
| `dryrun/buffer/` | Priority queue used by the staleness buffer |

---

## Two architectural decisions worth stating plainly

### 1. Versioned JSONL is the boundary between execution and modelling

This is the decision that makes a new backend addable without touching the simulator
and a cost model refittable without re-profiling. The `ProfileRecord` schema is the
contract: if an adapter writes valid records, the fitter can consume them, and the
simulator can query the resulting artifact.

The alternative, having adapters return Python objects directly to the fitter, would
couple the adapter interface to the model's input type. Any change to a cost model's
feature set would require an adapter change. The JSONL boundary breaks that coupling:
adapters record everything they observe, and cost models read what they need.

### 2. DryRun-RL owns the training entry point

The Megatron-Bridge adapter (`dryrun/profiler/adapters/megatron/`) does not use the
vendored `scripts/performance` CLI from Megatron-Bridge. That wrapper is designed for
the Slurm and NeMo-Run container flow: it parses a fixed set of known arguments and
silently discards everything else. Profiling settings, activation recompute flags, and
output-path overrides therefore never reach the training config when routed through that
wrapper.

The DryRun-RL adapter writes a launch spec in Python, applies it to a `ConfigContainer`
directly, and records step times and per-rank memory peaks through the official callback
API. The vendored tree is pinned and installed, but the entry point is DryRun-RL's own.

---

## Programmatic API

The simulator is directly importable for use outside the Hydra CLI:

```python
from dryrun.cost.rollout import UnifiedRoofline
from dryrun.cost.training import (
    FixedTrainingLatency,
    FixedTrainingMemory,
    TrainingCostModel,
)
from dryrun.policy.areal import ArealPolicy
from dryrun.simulator import SimConfig, Simulator
from dryrun.workload.distributions import bimodal
from dryrun.workload.grouped import GroupedWorkload

rollout_cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_d=1e-7)
training_cost = TrainingCostModel(
    FixedTrainingLatency(0.5),
    FixedTrainingMemory(0),
)
lengths = bimodal(200, short_len=50, long_len=500, long_frac=0.25, seed=42)
cfg = SimConfig(
    batch_size=8,
    max_staleness=2,
    n_versions=20,
    n_instances=4,
    workload=GroupedWorkload.from_lengths([512], lengths),
)

sim = Simulator(
    rollout_cost=rollout_cost,
    training_cost=training_cost,
    policy=ArealPolicy(),
    cfg=cfg,
)
result = sim.run()
print(result.throughput, "tokens/s")
```

Run verbatim, this prints `1431.977973351689 tokens/s` and completes 20 versions in about
16.3 simulated seconds. Note that `ArealPolicy` requires no arguments, whereas
`PSRLPolicy` requires the separate `psrl` package (see {doc}`../simulation/policies`).

`SimResult` exposes `consumed`, `dropped`, `sim_time`, `versions_done`, `livelocked`,
`train_timestamps`, `train_peak_memory_bytes`, and convenience properties
`throughput`, `staleness_true_values`, `staleness_reported_values`,
`max_staleness_true`, and `max_staleness_reported`.
