# Adapters and cost models

An adapter turns one backend's measurements into `ProfileRecord`. A cost model turns
`ProfileRecord`, or kernel parquet, into predictions. The seam is the versioned record schema in
{doc}`../reference/schemas`.

---

## The adapter contract

`srl/profiler/adapters/` holds three backends:

| Backend | Directory | Drives |
|---|---|---|
| `vllm` | `adapters/vllm/` | A vLLM server through `vllm bench sweep serve`, parsing its scheduler logs |
| `megatron` | `adapters/megatron/` | Megatron-Bridge training recipes through SimRL's own launcher |
| `fsdp` | `adapters/fsdp/` | An FSDP2 or HSDP model through SimRL's own launcher |

`adapters/base/` is shared infrastructure, not a backend. `run.py` executes a prepared candidate in
a subprocess and writes `status.json`. `sidecar.py` reads the `srl_metrics.json` a launcher writes
and applies the warmup policy.

Every training adapter exposes three verbs, wired in `srl/cli/profile.py`:

`generate`
: Enumerate candidates under the search space's constraints and write a manifest. No process is
  started. `search_space.py` derives training candidates from a model's shape and an
  interconnect budget. `fsdp_search_space.py` enumerates FSDP meshes.

`run`
: Execute candidates, one independent process each, a `torchrun` child for the training backends.
  Each child writes `status.json`, and one `oom_rank<N>.json` per rank that caught a
  `torch.OutOfMemoryError`. `run` does not interpret them.

`collect`
: Read what `run` wrote, validate it, and emit `ProfileRecord` JSONL. Collection never executes
  anything.

The `vllm` backend nominally has the same three verbs but works differently. `generate` writes one
job per sweep point and phase, `run` starts servers and replays the benchmark, and `collect` parses
the scheduler logs into iteration payloads. Its input is `vllm bench sweep serve` output, not a
training sidecar.

### Invariants

The `ProfileRecord` constructors enforce the two writer invariants:

- `world_size == tp * pp * dp * cp`.
- For training records, `global_batch_size == micro_batch_size * dp * num_microbatches`.

A record that violates either is rejected before it reaches disk. `write_records` replaces the
destination file atomically, so an interrupted collection cannot leave a truncated JSONL that a
fitter would read as a short but valid file.

### Raw artifacts are preserved

Each adapter keeps the backend's own output under the candidate directory: vLLM timing JSON,
the Megatron sidecar, the FSDP sidecar. The JSONL is derived from it and never written back into it,
so a raw artifact can be re-collected under a new schema version without re-running the job.

### OOM is read from marker files, never from logs

`read_status` in `adapters/base/run.py` classifies a finished run as `success`, `oom` or `failed`.
A non-zero exit code is `oom` only when the launcher wrote at least one `oom_rank<N>.json`. A log
that merely contains the phrase "out of memory" stays `failed`, because an OOM record adds an
inequality constraint to the memory fit that can only raise the predicted floor, and a false
positive would make the model silently more conservative.

A process that dies before writing a marker, for example from a setup error or a NCCL allocation
failure, is `failed` and contributes no bound. See {doc}`../reference/limitations`.

### `--resume`

`run_candidate` returns without executing when the candidate's `status.json` already says
`success`. Everything else, `oom` and `failed` included, is executed again. So a sweep re-run with
more memory headroom retries its OOM points without a manual step.

---

## Import isolation

The training adapters need `torch` and the training framework. The simulator and the regression path
must not pay that import.

Only the `torchrun` child imports `launch.py` and `measure.py`, and neither is exported from the
adapter package's `__init__`. The Megatron package docstring states the rule, and the FSDP package
follows it. Everything else on the adapter import path, configuration, preparation, execution and
collection, is safe to import on a CPU-only machine.

This mirrors the kernel profiler's split: `profiler/kernel/grid` and `profiler/kernel/sweep` are
torch-free, and the GPU work lives in `profiler/kernel/runner`.

---

## The cost model contract

Two registries select a model:

- `ROLLOUT_MODEL_REGISTRY` (`srl/cost/rollout/__init__.py`): `roofline`, `psrl`, `distserve`,
  `composed`.
- `TRAINING_MODEL_REGISTRY` (`srl/cost/training/__init__.py`): `hydraulis`, `fsdp`.

`srl-regress` dispatches on `--model` or `--latency-model`, and each class owns its fitting
formula and the artifact it writes. `srl` dispatches on `rollout_cost.name` and
`training_cost.name`.

### The rollout model interface

`RolloutCostModel` (`srl/cost/rollout/base.py`) has two abstract methods:

- `predict(workload) -> float` returns a strictly positive latency for one engine iteration.
- `decode_coeffs(n_d, ctxsum) -> (floor, alpha, beta)` is the closed form for a pure-decode
  segment. The simulator's decode advance, and any strategy that prices a segment, depend on it.
  Making it abstract is what prevents a new model from omitting it.

`RolloutWorkload` describes one iteration with `phase`, `n_tokens`, `sum_l2`, `ctxsum`,
`request_count` and an optional `prefill_ctx`. A fitted model stores the feature ranges it was
fitted over, and a query outside them is a configuration or workload error even where the code does
not raise.

### The training model interface

`TrainingCostModel` bundles a `TrainingLatencyModel` and a `TrainingMemoryModel` and returns a
`TrainingCostEstimate` with `latency_s`, `peak_memory_bytes` and diagnostics. The two are fitted
separately because their features, their noise and their domains differ, and because the memory fit
uses OOM candidates as inequality constraints that the latency fit cannot use.

The shared four-term latency form lives once, in `srl/cost/training/latency.py`:

| Helper | Responsibility |
|---|---|
| `latency_from_coefficients` | The formula, raising on a non-positive result |
| `latency_coefficients` | Name the fitted parameters in design-row order |
| `check_latency_rank` | Raise unless the design reached `LATENCY_TERM_COUNT` |
| `latency_feature_ranges` | Turn the profiled span into the prediction-time gate |
| `LATENCY_TERM_COUNT` | The one place to change if a term is added |

Both backends end their prediction in the shared call. The backends differ in what they call the
slots axis (`pipeline_slots` for Megatron, `num_microbatches` for FSDP), in whether they divide by
a pipeline degree, and in their diagnostics, so the assembly around the shared call stays
per-backend on purpose.

### Artifacts

`CostArtifact` (`srl/cost/artifact.py`) persists a fit. Two properties matter to a new model:

- `entry_for` returns an exact match on parallelism and selector, or raises `KeyError`. There is no
  nearest cell and no silent fallback. A model whose topology can be derived instead of looked up
  asks for `ignore_pp=True` and gets an exact match on `tp`, `cp` and `ep`.
- `feature_ranges` travels with the coefficients and is what the domain gate reads. An
  out-of-range query raises unless the feature is listed in `training_cost.extrapolate`, and every
  permitted exceedance is recorded.

### Composed models

`ComposedRollout` and `ComposedTraining` sum measured kernels from `KernelDatabase` instead of
fitting end-to-end times. Their inputs are kernel op tables and a per-model op list from
`ModelOps` or `TrainingOps`, both built from a checkpoint's architecture. `ComposedRollout` prices
GEMMs, attention and collectives. It does not read the elementwise table, which belongs to the
training path. Its decode segment is integrated with `LatticeIntegrator` or the closed form
described in {doc}`architecture`.

`ComposedTraining` is only reachable from Python. It is not in `TRAINING_MODEL_REGISTRY` and is
not selectable from the `training_cost` group.

---

## Adding a rollout cost model

1. Subclass `RolloutCostModel` in `srl/cost/rollout/` and implement `predict` and
   `decode_coeffs`. Keep both free of `torch`.
2. Add a YAML file under `srl/config/defaults/rollout_cost/` with the constants it needs, and add a
   field block to `RolloutCostConfig` in `srl/config/schema.py` if the loader must read them.
3. Add a branch to `build_rollout_cost` in `srl/cli/main.py` and an entry to
   `ROLLOUT_MODEL_REGISTRY`.

## Adding a training cost model

1. Build a `TrainingLatencyModel` and a `TrainingMemoryModel`. Use `latency_from_coefficients`,
   `check_latency_rank` and `latency_feature_ranges` rather than re-deriving the form.
2. Implement `fit(records, ...) -> CostArtifact` and `from_artifact(path, ...)`. Pass discrete
   selectors into `entry_for`, and store `feature_ranges`.
3. Register the class in `TRAINING_MODEL_REGISTRY` and add the matching YAML option, the config
   fields and the `build_training_cost` branch.

`tests/test_training_shared.py` and `tests/test_training_cost.py` exercise the shared form and the
per-backend assembly.
