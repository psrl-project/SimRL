# Concepts

Every other page on this site uses the terms below as vocabulary without re-defining them.
Several of them look like ordinary words but carry a precise meaning.

---

## ProfileRecord: the measurement atom

A `ProfileRecord` is one validated observation of a real system: a single profiling run on
specific hardware, at a specific topology, producing specific timing and memory numbers. It
is a frozen dataclass in `srl/profiler/schema.py`, serialised as one JSONL line.

Every record carries these groups:

- **record_type**: `rollout` or `training`.
- **run**: a run ID, repetition index, start timestamp, duration, raw-artifact path, and a
  *status* (`success`, `oom`, `failed`, or `skipped`). The status is what separates a measured
  latency from an inequality constraint (see *OOM as an inequality constraint* below).
- **model**: checkpoint name, architecture, parameter count, and layer dimensions.
- **hardware**: GPU name, count, memory, and node topology.
- **software**: backend name and version, source commit, and the command that was run.
- **parallelism**: explicit TP, PP, DP, CP, and EP degrees. `world_size` is validated to equal
  `tp * pp * dp * cp`.
- **payload**: a `RolloutPayload` (one scheduler iteration from an inference server) or a
  `TrainingPayload` (per-microbatch statistics, step times, and per-rank memory peaks).
- **tags**: free-form provenance strings.

`ProfileRecord` is the documented boundary between execution and modelling. An adapter
writes records and a cost model reads them. Nothing in the adapter knows the model it will
feed, and nothing in the cost model knows the system that produced the records.
{doc}`../reference/schemas` lists every field.

---

## CostArtifact and ArtifactEntry: the fitted model envelope

A `CostArtifact` is the JSON file `srl-regress` writes. It stores everything a fitted cost
model needs to predict, without importing any benchmark process. It is defined in
`srl/cost/artifact.py`.

The outer `CostArtifact` carries:

- `kind`: `"rollout"` or `"training"`.
- `model_type`: the formula family, one of `roofline`, `psrl`, `distserve` (rollout) or
  `hydraulis`, `fsdp` (training).
- `model` and `hardware`: JSON snapshots of the descriptors from the profiling run.
- `profile_digest`: a hash of the JSONL input.
- `entries`: one `ArtifactEntry` per cell.
- `artifact_version`, `created_at`, and `diagnostics`.

An `ArtifactEntry` stores:

- `parallelism`: the topology this entry describes.
- `selector`: discrete keys (component, schedule, recompute mode, optimizer, dtypes, FSDP mesh)
  that distinguish this cell from others at the same parallelism.
- `coefficients`: the fitted weights, one float per named term.
- `feature_ranges`: the input domain the fit covers (see below).
- `train_metrics` and `validation_metrics`: `r_squared`, `rmse`, `mre_percent`, `max_error`,
  `n_samples`, `n_outliers`, and `design_rank`.
- `diagnostics`: model-specific notes, including the fitted formula as a string and whether
  `validation_metrics` is in-sample.

The kernel path has no artifact: `ComposedRollout` reads measured parquet tables directly
(see *Slice and axis* below).

---

## Cell and selector: one fit per topology

A *cell* is one distinct (parallelism, selector) combination that was profiled and fitted
independently. `CostArtifact.entry_for` requires exactly one match and raises `KeyError`
otherwise; it never falls back to a neighbour.

This is deliberate. A fit pooled over TP=1 and TP=4 would produce one set of coefficients
that describes neither: per-rank token cost falls with TP while communication cost appears.
The error would be invisible because no reference point lies outside the pooled range.

A *selector* is the set of discrete keys that distinguish cells at the same parallelism. For
the FSDP artifact it includes `shard_size`, `replicate_size`, `optimizer_offload`, and
`reshard_after_forward`, which is why a multi-mesh FSDP artifact needs
`training_cost.shard_size` and `training_cost.replicate_size` to choose one. For the Megatron
(`hydraulis`) artifact, latency cells are looked up by TP, CP, and EP with the pipeline depth
derived from the topology (`ignore_pp`), while memory cells match PP exactly. A rollout
artifact's selector is just `block_size`.

**A fit averaged over cells describes none of them.** If two cells look close enough to pool,
measure both and compare their coefficients first.

---

## feature_ranges: the validity envelope

`feature_ranges` maps a feature name to `(min, max)` bounds and is stored in every
`ArtifactEntry`. The two training cost models (`hydraulis`, `fsdp`) check each continuous
feature of a query against it before predicting.

A query inside the range is *interpolation* and always allowed. A query outside is
*extrapolation* and raises `TrainingOutOfDomainError` unless the feature is named in
`training_cost.extrapolate`. A permitted extrapolation is written verbatim into the step's
`extrapolation_notes` field in `logs/train.jsonl`. A name that is not a feature of the
artifact raises `ValueError` at construction, listing the valid names, so a typo cannot
silently permit nothing.

The rollout fits record `feature_ranges` too, but the rollout cost models do not enforce them.
`ComposedRollout` has its own guard: an unmeasured kernel slice raises
`KernelCoverageError`.

{doc}`../simulation/domain-gates` has the full policy and the feature names each model accepts.

---

## Slice and axis: how kernel tables are indexed

Every kernel op in `srl/cost/kernel/ops/` declares two column tuples:

- **`slice_columns`**: columns that must match a measured value exactly. A query with no
  matching slice raises `KernelCoverageError`.
- **`axis_columns`**: columns interpolated along inside a slice.

The split reflects physics. Interpolating between two token counts of the same GEMM shape is
meaningful; blending a `(4096, 4096)` GEMM with a `(151936, 4096)` GEMM is not, because they
may select different tile sizes.

| Op | `slice_columns` | `axis_columns` |
|---|---|---|
| `gemm` | `gemm_dtype`, `n`, `k` | `m` |
| `context_attention` | `num_key_value_heads`, `head_dim`, dtypes, `block_size`, `window_size` | `num_heads`, `batch_size`, `isl` |
| `generation_attention` | same slice | `num_heads`, `batch_size`, `step` |
| `elementwise` | `dtype`, `op_kind` | `n_elements` |
| `all_reduce` and the other collectives | `dtype`, `world_size`, `nodes` | `message_bytes` |
| `optimizer_step` | `optimizer`, `state_kind` | `param_count` |

`world_size` is a slice column for collectives because a 2-rank and an 8-rank ring run
different schedules over different links. The registered ops are listed by
`srl.cost.kernel.ops.base.families()`; {doc}`../development/kernel-ops` explains how to add one.

---

## SOL, utility, and floor

**Speed of light (SOL)** is the analytic roofline latency, `max(compute_time, memory_time)`,
from the GPU's peak FLOP/s and bandwidth (`srl/cost/kernel/data/gpu_specs.yaml`,
`srl/cost/kernel/sol.py`). It is a lower bound and never a prediction on its own.

**Utility** is the dimensionless ratio `SOL / measured_latency`. It is what the kernel
interpolator learns, not latency, because latency spans orders of magnitude across a sweep
while utility varies slowly. Interpolating utility leaves every order of magnitude to the SOL
formula and asks the surface only how efficient the kernel is. The price is that the SOL
formula must be right: an error in it propagates into every utility and every prediction. The
interpolation is a monotone cubic (`pchip`, the default) or a Shepard blend, selected with
`rollout_cost.utility_surface`. A query past the measured range holds the boundary utility
instead of extrapolating one.

**Floor.** Attention and collective kernels carry a per-call dispatch cost the roofline does
not represent. The floor is a lower bound on latency inside the `max`:

$$\text{latency} = \max(\text{compute\_time},\ \text{memory\_time},\ \text{floor\_s})$$

It sits inside the `max`, not added, so it binds only when the kernel is otherwise too small
to dominate. Floors are fitted per backend and per op by `srl-profile kernels fit-floor` and
stored in `floors.json` beside the perf tables. When that file is missing, the constants in
`FALLBACK_FLOORS_S` (`sol.py`) are used with a warning. GEMM declares no floor.

---

## Version, staleness, and group

A *version* is one published weight generation: the integer `train_version` the simulator
increments after each training step. A request is tagged with the version it was dispatched
under (`v_traj`) and tracks the oldest and newest versions that produced any of its tokens
(`v_min`, `v_max`).

**True staleness** is `consume_version - v_min`: the gap between the version that consumed a
request and the *oldest* weight version behind any of its tokens. **Reported staleness** is
`consume_version - v_max`, the gap to the *newest*. They differ only when a request's tokens
span several versions, for example after an in-place weight update. The summary printed by
`srl` and the `staleness_true` field in `request.jsonl` use true staleness.

A *group* is the set of responses that share a prompt (`group_id`). `job.batch_size` counts
**groups** per training step, and `job.group_size` is the responses per group, so one step
consumes `batch_size * group_size` responses. GRPO and DAPO advantages normalise rewards
within a group, so group boundaries must survive in the workload. A parametric workload has
`group_size` 1; only a joint trace carries real groups. See {doc}`../workload/index`.

A *policy preset* is the typed config for one asynchronous-RL system, selected with
`policy=<name>`: `areal`, `dressage`, `nemo_rl`, `openrlhf`, `prime_rl`, `psrl`, `roll`,
`skyrl`, `slime`, `trl`, `verl`, `verl_v1`. Each preset owns that system's own staleness
knobs. {doc}`../simulation/policies` compares them.

---

## Design rank: why a regression may refuse to fit

A regression's design matrix is the table of input features presented to the least-squares
solver. If one column is a linear function of another, or constant, the matrix is
*rank-deficient*: infinitely many coefficient sets fit equally well and the solver cannot
choose.

The training fitters record `design_rank` in `FitMetrics` and refuse a rank-deficient cell
unless `srl-regress training --allow-rank-deficient` is passed. The latency design row has
four columns, `[1, tokens, slots, sum_squared_tokens]`, for the four coefficients
`per_step_overhead`, `token_cost`, `microbatch_overhead`, and `token_quadratic_cost`. Holding
`num_microbatches` at one value makes the slot column constant, hence collinear with the
intercept, so the rank drops below four. **Check the design before committing GPU time.**
{doc}`../profiling/methodology` gives the minimum sweep for each model.

---

## OOM as an inequality constraint

When a profiling candidate runs out of memory, its record has `status = "oom"` and no timing.
For the cost model that is not missing data. It is a bound: whatever the true peak was, it
exceeded device capacity. The memory fitters keep OOM candidates as inequality constraints
alongside the equalities from successful runs, which is what bounds the valid domain from
above. A sweep that never OOMs has not found the ceiling.

Status comes from marker files (`oom_rank*.json`) the launcher writes on a caught
`torch.OutOfMemoryError`, never from scanning a log for the phrase "out of memory". A log can
mention that phrase for unrelated reasons, and a false positive can only raise the fitted
memory floor.

---

## Compact glossary

```{glossary}
ProfileRecord
    One validated profiling observation. The JSONL boundary between an adapter and a
    cost model. Defined in `srl/profiler/schema.py`.

CostArtifact
    The JSON envelope produced by `srl-regress`. Holds one `ArtifactEntry` per cell.

ArtifactEntry
    Coefficients, feature ranges, and diagnostics for one cell.

cell
    One (parallelism, selector) combination fitted independently. A lookup that finds no
    matching cell raises instead of using a neighbour.

selector
    The discrete keys that identify a cell within a fixed parallelism.

feature_ranges
    The `(min, max)` bounds per feature stored in an `ArtifactEntry`. The training cost
    models refuse queries outside them unless extrapolation is named explicitly.

slice column
    A kernel-table column that must match exactly; never interpolated along.

axis column
    A kernel-table column interpolated along inside a slice.

SOL (speed of light)
    The analytic roofline lower bound, `max(compute_time, memory_time)`.

utility
    `SOL / measured_latency`; the quantity the kernel interpolator learns.

floor
    A per-call dispatch cost placed inside `max(compute, memory, floor)`. Fitted per backend
    and op by `srl-profile kernels fit-floor`.

version
    One published weight generation, incremented after each training step.

true staleness
    `consume_version - v_min`: the gap to the oldest weight version behind a request's tokens.

reported staleness
    `consume_version - v_max`: the gap to the newest.

group
    The responses that share one prompt. `job.batch_size` counts groups.

policy preset
    The typed config and admission logic for one asynchronous-RL system, chosen with
    `policy=<name>`.

design rank
    The column rank of a regression's design matrix. A rank-deficient cell is refused.

OOM constraint
    A profiling result where the system ran out of memory, kept as a lower bound on peak
    memory instead of being discarded.
```
