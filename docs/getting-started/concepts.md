# Concepts

Every other page on this site uses the terms below as vocabulary without re-defining
them. A brief reading here saves confusion later: several of them look like ordinary
words but carry a precise meaning that differs from ordinary usage.

---

## ProfileRecord: the measurement atom

A `ProfileRecord` is one validated observation of a real system: a single profiling run
on specific hardware, at a specific topology, producing specific timing and memory
numbers. It is a frozen dataclass defined in `srl/profiler/schema.py`.

Every record carries five groups of metadata:

- **run**: a unique run ID, repetition index, start timestamp, duration, and *status*
  (`success`, `oom`, `failed`, or `skipped`). The status is what distinguishes a
  measured latency from an inequality constraint (see *OOM as an inequality constraint*
  below).
- **model**: the checkpoint name, architecture string, parameter count, and layer
  dimensions.
- **hardware**: GPU name, count, memory, and node topology.
- **software**: backend name and version, Python and CUDA versions, and the exact
  command that was run.
- **parallelism**: explicit TP, PP, DP, and CP degrees. `world_size` is validated
  to equal their product.

The payload is either a `RolloutPayload` (batch-step statistics from an inference
server) or a `TrainingPayload` (per-microbatch statistics and rank memory peaks from a
training run).

`ProfileRecord` is the documented JSONL boundary between execution and modelling. An
adapter writes records and a cost model reads them. Nothing in the adapter knows about the
model it will feed, and nothing in the cost model knows about the system that produced the
records. Adding a new backend means writing a new adapter and adding nothing else.

---

## CostArtifact and ArtifactEntry: the fitted model envelope

A `CostArtifact` is the JSON file produced by `srl-regress`. It stores everything a
cost model needs to make predictions, without importing any benchmark process.

The outer `CostArtifact` carries:

- `kind`: `"rollout"` or `"training"`.
- `model_type`: the specific formula used (e.g. `"roofline"`, `"hydraulis"`, `"fsdp"`).
- `model` and `hardware`: JSON snapshots of the model and hardware descriptors from
  the profiling run.
- `profile_digest`: a hash of the JSONL input, so re-fitting a changed input produces
  a different artifact rather than silently reusing a cached one.
- `entries`: a tuple of `ArtifactEntry` objects, one per cell.

An `ArtifactEntry` stores:

- `parallelism`: the exact TP/PP/DP/CP topology this entry describes.
- `selector`: additional discrete keys (activation recompute mode, optimizer type,
  etc.) that distinguish this cell from others at the same parallelism.
- `coefficients`: the fitted regression weights, one float per named term.
- `feature_ranges`: the input domain this entry is valid over (see below).
- `train_metrics` and `validation_metrics`: regression diagnostics including
  $r^2$, RMSE, MRE%, max error, sample count, outlier count, and design rank.

---

## Cell and selector: why a fit per topology, not a global one

A *cell* is one distinct (parallelism, configuration) combination that was profiled
and fitted independently. The `entry_for` method on `CostArtifact` requires an exact
match on parallelism and selector and raises `KeyError` if none is found.

This is deliberate. A linear fit that pooled TP=1 and TP=4 data would produce a single
set of coefficients that describe neither: the token cost under TP=4 is lower per rank
(work is split across more ranks), and the communication overhead is absent at TP=1 and
present at TP=4. Averaging across those regimes produces coefficients that are wrong at
both endpoints by a predictable amount, and that wrongness would be invisible because there
would be no reference point outside the pooled range.

A *selector* is the set of discrete keys that distinguish cells at the same parallelism.
For the FSDP artifact, the selector includes `shard_size`, `replicate_size`,
`activation_recompute`, `optimizer`, `optimizer_offload`, and `reshard_after_forward`.
For the rollout artifact, the selector is typically empty because the only relevant
discrete axis is the parallelism itself.

**A fit averaged over cells describes none of them.** If you suspect two cells are close
enough to pool, measure them both, compare their coefficients, and make an informed
decision. Do not pool by default.

---

## feature_ranges: the validity envelope

`feature_ranges` is a dictionary from feature name to `(min, max)` bounds, stored in
every `ArtifactEntry`. When the cost model receives a query, it checks every continuous
feature against its fitted range before computing a prediction.

A query inside the range is *interpolation*: the model is predicting between two
measured points. This is always allowed and produces no warning.

A query outside the range is *extrapolation*: the model is predicting beyond the region
where it was fitted. This raises `TrainingOutOfDomainError` unless the feature name
is listed in `training_cost.extrapolate`. When extrapolation is permitted, the event
is recorded verbatim in the `train.jsonl` telemetry stream.

The distinction matters because the profiled range usually coincides with what the
hardware can execute. In the FSDP sweep for Qwen3-8B on 8x H20, every candidate at
32768 tokens per rank produced an OOM, so there is no measured latency there to interpolate
toward. A prediction at 32768 tokens per rank looks numerically plausible but describes
a configuration the hardware physically cannot run.

See {doc}`../simulation/domain-gates` for the full extrapolation policy and which
features are safe to extrapolate upward.

---

## Slice vs. axis: how kernel tables are indexed

Every kernel op in `srl/cost/kernel/ops/` declares two column lists:

- **`slice_columns`**: columns used for exact lookup. Queries must match a profiled
  value in every slice column exactly, or the lookup raises a coverage error.
- **`axis_columns`**: columns interpolated along. The model fits a utility surface
  over axis values from the measured points and evaluates it at the query.

The distinction reflects a physical reality: it is meaningful to interpolate between
two measured values of the same kernel shape, but not to blend across qualitatively
different shapes.

**Example: GEMM.** `gemm` uses `slice_columns = ("gemm_dtype", "n", "k")` and
`axis_columns = ("m",)`. `m` is the token count, a natural continuous axis. `n`
(output features) and `k` (input features) are categorical: a `(4096, 4096)` GEMM and
a `(151936, 4096)` GEMM may run entirely different tile sizes and register strategies.
Blending their latencies would produce a number describing neither shape.

**Example: all_reduce.** `all_reduce` uses `slice_columns = ("dtype", "world_size")` and
`axis_columns = ("message_bytes",)`. `world_size` is a slice column because a 2-rank
all-reduce and an 8-rank one run different ring schedules over different link topologies.
The ring-all-reduce lower bound `2*(N-1)/N * bytes / bandwidth` varies non-linearly with
`N` even if message size is held constant, so blending across world sizes would
systematically underestimate the cost at intermediate counts that were never profiled.

---

## SOL, utility, and floor

**Speed-of-light (SOL)** is the analytical roofline latency: `max(compute_time,
memory_time)` derived from vendor-published peak FLOP/s and memory bandwidth. It is a
lower bound, since no kernel can run faster than the roofline, but it is never a prediction
on its own.

**Utility** is the dimensionless ratio `SOL / measured_latency`. A utility of 1.0 means
the kernel achieved the theoretical optimum. A utility of 0.5 means it ran at half the
roofline speed.

This is why utility, not latency, is what the kernel interpolator learns. Measured on the
Qwen3-8B GEMM table on H20, latency spans 2268x while utility spans 4.65x, and within the
`(n=4096, k=4096)` slice the figures are 566x against 1.69x. A model interpolating latency
would have to carry the whole dynamic range. Interpolating utility leaves every order of
magnitude to the analytical SOL formula and asks the surface only how efficient the kernel is. The trade-off is
that the SOL formula must be correct: an error in the roofline propagates into every
utility value and therefore into every prediction. On current hardware the GEMM roofline
is accurate enough that the interpolated utility approach reaches 3.91% MRE without
any further correction.

**Floor.** Attention and collective kernels carry a per-call dispatch cost that the
roofline does not represent. At small batch sizes, decode attention latency stays
near-constant across a 64x increase in KV history length, because the kernel is entirely
dispatch-limited rather than compute-bound or memory-bound. The floor is the lower bound on
latency in that regime:

$$\text{latency} = \max(\text{compute\_time},\ \text{memory\_time},\ \text{floor\_s})$$

The floor is placed *inside* the `max`, not added as a constant. Adding it would
double-count in the dispatch-limited regime. Placing it inside the max means it only binds
when the kernel is otherwise too small to dominate. GEMM does not need a floor:
its holdout MRE is 3.91% without one, and it lives in the compute- or bandwidth-limited
regime rather than the dispatch-limited one.

Floors are fitted per-backend and per-op by `srl-profile kernels fit-floor`. The
fitted values for H20 (vllm backend) are: `context_attention` 32 µs,
`generation_attention` 22 µs, `elementwise` 2 µs, `all_reduce` 2 µs. The code in
`srl/cost/kernel/sol.py` documents the fallback values and their origin.

---

## Version, staleness, and group

A *version* is one published weight generation, the integer counter `train_version` that the
simulator increments after each training step. A request dispatched to the
rollout engine is tagged with the version (`v_traj`) that was current at dispatch time.

**True staleness** of a consumed request is `consume_version - v_min`: the gap between
the training version that consumed it and the *earliest* weight version that produced
any of its tokens. A request that was preempted and re-prefilled under a newer weight
version may have tokens from multiple versions, and true staleness counts from the oldest
one.

**Reported staleness** is `consume_version - v_max`: the gap from the *most recent*
weight version. The distinction matters when a request's tokens span multiple weight
versions. Reported staleness is what a real system sees, and true staleness is what the
simulator measures internally.

A *group* is the set of responses that share the same prompt, recorded as `group_id` in the
`Request` dataclass. DAPO-style and GRPO-style training compute group-relative
advantage, which is a normalisation over the rewards within a group. That normalisation
is mathematically undefined if the group boundary is unknown, which is why a parametric
workload (where every response is independent) cannot faithfully represent a GRPO
training job. See {doc}`../workload/index` for the two workload sources and when each
is appropriate.

---

## Design rank: why a regression may refuse to fit

A regression's design matrix is the table of input features presented to the least-
squares solver. If two columns are perfectly correlated (one is a linear function of
another, or one is constant), the matrix is *rank-deficient*: there are infinitely many
coefficient sets that fit the data equally well, and the solver cannot choose between
them.

SimRL's training fitters report `design_rank` in `FitMetrics` and raise rather than
returning degenerate coefficients when the matrix is rank-deficient. The most common
cause is sweeping only one axis while the model needs two. For the FSDP latency model,
the design row is `[1, nmb * tokens, nmb, nmb * sum_squared_tokens]`, which is four columns.
Holding `num_microbatches` constant at one value makes the first and third columns
proportional, capping the rank at 3 while the fit demands 4. An entire six-hour sweep becomes
unfittable.

**Check the design before committing GPU time.** {doc}`../profiling/methodology`
explains the minimum sweep needed for each model and why a single fixed
`num_microbatches` is the most common identifiability trap.

---

## OOM as an inequality constraint: a bound, not a failure

When a profiling candidate runs out of memory, the `RunDescriptor` in its `ProfileRecord`
carries `status = "oom"` and no timing payload. From the cost model's perspective, an
OOM is not missing data. It is a lower bound on the true peak memory: whatever the actual
footprint was, it exceeded the device capacity.

The memory fitter records OOM candidates as inequality constraints alongside the
equality constraints from successful runs. A sweep that never OOMs has not found the
ceiling, because the fitted memory model's upper bound is the highest successful data point
rather than the device limit. In the FSDP sweep for Qwen3-8B on 8x H20, the token range's upper
bound was established precisely because 32768 and 65536 tokens per rank produced 0
successes against 27 OOM failures. Those OOMs are what bounded the model's valid domain from
above.

OOM status must come from a marker file the launcher writes on a caught allocation
failure, not from scanning a log for the phrase "out of memory". A log can contain that
phrase for reasons unrelated to an actual OOM (a warning, a copied error message from a
different rank), and a false positive can only raise the fitted memory floor, making the
model quietly more conservative in a way no metric reports.

---

## Compact glossary

```{glossary}
ProfileRecord
    One validated profiling observation. The JSONL boundary between an adapter and a
    cost model. Defined in `srl/profiler/schema.py`.

CostArtifact
    The JSON envelope produced by `srl-regress`. Contains one or more
    `ArtifactEntry` objects, one per cell.

ArtifactEntry
    Coefficients, feature ranges, and diagnostics for one discrete cell (topology +
    configuration combination).

cell
    One (parallelism, selector) combination fitted independently. A lookup that finds no
    matching cell raises rather than extrapolating to a neighbour.

selector
    The set of discrete keys that identify a cell within a fixed parallelism.

feature_ranges
    The `(min, max)` input bounds stored per feature in an `ArtifactEntry`. Queries
    outside these bounds are refused unless extrapolation is explicitly permitted.

slice column
    A kernel table column that must match exactly; no interpolation is performed along it.

axis column
    A kernel table column that the utility interpolator varies continuously along.

SOL (speed of light)
    The analytical roofline lower bound: `max(compute_time, memory_time)`.

utility
    `SOL / measured_latency`. Spans 4.65x across the GEMM table against 2268x for the
    raw latency, which is what makes it the easier quantity to interpolate.

floor
    A per-call dispatch cost placed inside `max(compute, memory, floor)`, not added
    as a constant. Fitted per backend and op by `kernels fit-floor`.

version
    One published weight generation. Incremented by the simulator after each training
    step.

true staleness
    `consume_version - v_min`: gap from the oldest weight version that produced any
    tokens for a request.

reported staleness
    `consume_version - v_max`: gap from the most recent weight version. What a real
    system observes.

group
    The set of responses sharing a prompt. Required for group-relative advantage
    (DAPO, GRPO).

design rank
    The column rank of the regression's design matrix. A rank-deficient design
    causes the fitter to raise rather than returning degenerate coefficients.

OOM constraint
    A profiling result where the system ran out of memory. Treated as a lower bound on
    peak memory in the fit, not as missing data.
```
