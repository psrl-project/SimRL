# Adapters and cost models

These two extension points are documented together because an adapter without a matching cost model
is useless, and a cost model without profiling data to fit it is untestable. The interface between
them is the `ProfileRecord` JSONL stream, whose schema is documented in {doc}`../reference/schemas`.

---

## The adapter contract

A profiler adapter drives one backend (vLLM, Megatron-Bridge, FSDP) and produces `ProfileRecord`
JSONL. Every adapter exposes three operations:

**`generate`** builds the sweep grid: candidates (vLLM workload points, training topology
candidates) are written to disk as JSON or YAML manifests. No training process is started. The step
can be re-run to add candidates to an existing sweep, though the same topology is never repeated.

**`run`** executes one or more prepared candidates in subprocesses and records status. Each
candidate runs in an independent process. For training backends this means one `torchrun` child per
candidate; each child writes a `status.json` and, if it caught a `torch.OutOfMemoryError`, one
`oom_rank{N}.json` marker per rank. `run` does not read or interpret those files.

**`collect`** reads the files written by `run`, validates them, and emits `ProfileRecord` JSONL.
Collect never re-executes anything. If a candidate directory is missing or its `status.json` is
absent, the record is emitted with `status = "failed"` rather than crashing the collection pass.

### Writer invariants

Every adapter that writes a `ProfileRecord` enforces two invariants at write time:

- `world_size == tp * pp * dp * cp`. Any topology that does not satisfy this is rejected rather
  than normalised.
- For training records: `global_batch_size == micro_batch_size * dp * num_microbatches`. A training
  run that does not satisfy this is not a well-formed training step.

These are checked in `dryrun/profiler/schema.py` before the record is serialised.

### Raw artifacts are preserved untouched

The official benchmark output — vLLM timing JSON, TensorBoard scalars, Megatron sidecar files — is
never modified. The adapter reads it and derives the normalised JSONL from it. If the raw artifact
is later needed for debugging or re-collection under a changed schema, it is still there. The JSONL
is the sole regression input; it is never back-projected into the raw artifact.

### OOM is classified from marker files

OOM status is determined by reading `oom_rank{N}.json` files written by the launcher on a caught
`torch.OutOfMemoryError`, never by scanning a log for the phrase "out of memory". A log can contain
that phrase for reasons unrelated to an actual OOM (a warning, an error message copied from another
rank, a comment in a script). A false-positive OOM classification adds an inequality constraint that
can only raise the fitted memory floor, making the model silently more conservative. The marker-file
approach avoids this: only a caught allocation failure writes the marker.

A candidate that crashes before writing any marker — because of a pre-training setup error, a
network issue, or a signal — is classified as `failed` rather than `oom`, and drops out of the
memory fit rather than contributing a misleading bound. This is acknowledged in
{doc}`../reference/limitations`.

### `--resume` semantics

`--resume` skips candidates whose `status.json` records `"success"`, and runs everything else.
This means that an `oom` candidate is re-run on resume, not skipped. The intent is that a sweep
expanded with more memory headroom (a smaller TP degree, a smaller batch) should naturally revisit
former OOM points without requiring a manual workaround. The implementation is in
`dryrun/profiler/adapters/training/run.py`.

---

## Import isolation for training adapters

Training adapters — Megatron and FSDP — must import `torch`, megatron, and their respective model
libraries. Those imports take several seconds and require a GPU stack to be installed. The simulator
and the regression path should never pay that cost.

The rule is: `launch.py` and `measure.py` import torch and the training framework; they are
deliberately absent from the adapter package's `__init__` namespace. Only the `torchrun` child
process imports them. The `dryrun/profiler/adapters/megatron/__init__.py` docstring states this
explicitly:

> `launch` and `measure` import torch and megatron, so they are deliberately absent from this
> namespace; only the torchrun child imports them.

`dryrun/profiler/adapters/__init__.py` re-exports only collection, preparation, and execution
entry points from each backend. None of the heavy training imports surface at the package level.

### The shared `training/` module

`dryrun/profiler/adapters/training/` holds the backend-agnostic pieces that Megatron and FSDP both
use: subprocess execution (`run.py`) and sidecar file access (`sidecar.py`). The backends differ in
how they build a model and a launch command, but they agree on how a candidate is run, how its
result is classified, and how step-time and memory measurements are extracted. Only that shared
logic lives here. Resolved-config validation stays backend-specific because Megatron reads a
serialised `ConfigContainer` while FSDP inspects a device mesh; a shared signature for those two
would be a parameter bag neither backend actually fills.

---

## The cost model contract

A cost model reads `ProfileRecord` JSONL, fits a predictive formula, and exposes a prediction
interface. The two registries are:

- **`ROLLOUT_MODEL_REGISTRY`** in `dryrun/cost/rollout/__init__.py`: maps `"roofline"`,
  `"psrl"`, `"distserve"`, `"composed"` to their classes.
- **`TRAINING_MODEL_REGISTRY`** in `dryrun/cost/training/__init__.py`: maps `"hydraulis"`,
  `"fsdp"` to their classes.

Each model class owns its fitting formula entirely. `dryrun-regress` dispatches to the right class
based on `--model`; the fitter inside the class decides which features to build, how to regularise,
and what to store in the artifact.

### `decode_coeffs()` is mandatory on every rollout model

Every class in `ROLLOUT_MODEL_REGISTRY` must implement `decode_coeffs(n_d, ctxsum) -> tuple`. This
returns `(floor, alpha, beta)` — the three parameters of a closed-form pure-decode segment
integrator. The simulator's composed rollout advance path uses these coefficients to compute how
long `k` decode steps take without iterating over each one individually. If `decode_coeffs` is
absent or returns wrong values, the closed-form advance path produces incorrect timing and the
simulator silently produces wrong schedules. This is a non-obvious hard requirement: the
`RolloutCostModel` abstract base class in `dryrun/cost/rollout/base.py` declares it `@abstractmethod`
precisely to prevent omitting it.

### The `CostArtifact` and `ArtifactEntry` contract

`CostArtifact` is the JSON envelope that persists a fitted model between the regression step and the
simulation step. Its structure is documented in {doc}`../getting-started/concepts` and the field
schema is in {doc}`../reference/schemas`. The key points for an implementor:

- `entry_for(parallelism, *, selector)` returns an exact match and raises `KeyError` if none is
  found. There is no nearest-neighbour fallback.
- `selector` is the set of discrete keys that distinguish cells at the same parallelism. For FSDP,
  the selector includes `shard_size`, `replicate_size`, `activation_recompute`, `optimizer`,
  `optimizer_offload`, and `reshard_after_forward`. Pass selectors as keyword arguments to
  `FsdpTrainingCost.from_artifact`; unknown keyword arguments raise `ValueError` at construction
  rather than silently being ignored.
- `feature_ranges` stores the `(min, max)` domain each continuous feature was fitted over. Queries
  outside this range raise `TrainingOutOfDomainError` unless the feature is listed in
  `training_cost.extrapolate` in the simulation config. When extrapolation is permitted, the
  exceedance is recorded in `TrainingCostEstimate.diagnostics`.

### Training models separate latency from memory

Every training cost model is composed of two separately fitted sub-models:

- A **latency model** predicts step wall-clock time from batch shape features.
- A **memory model** predicts peak allocated GPU memory and classifies whether the candidate would
  OOM on a given device.

The two are fitted independently because their features, their measurement uncertainty, and their
interpolation domains are different. For the Hydraulis model the latency fitter uses OLS on a
physical-form design matrix; the memory fitter uses inequality-constrained regression to incorporate
OOM bounds. For FSDP the latency form includes `num_microbatches` explicitly because memory does
not depend on it (FSDP allocates by microbatch count, not token count — see
{doc}`../reference/limitations`).

Keeping them separate means that a sweep that produces abundant memory data but sparse latency data
can fit the memory model well while reporting that the latency fit is underdetermined. A combined
model would average over both measurement qualities and obscure the underdetermination.
