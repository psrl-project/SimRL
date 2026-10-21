# Megatron training cost model (Hydraulis)

This page documents the Hydraulis cost model fitted from Megatron-Bridge profiles
({doc}`/pipeline/profiling/megatron`): the latency decomposition, the memory formula, how fitting and
lookup work, and the composed alternative. Measurement rules are in
{doc}`/pipeline/profiling/methodology`. Accuracy is in {doc}`/reference/validation`.

## Latency model

The form follows Appendix C of the Hydraulis paper (https://arxiv.org/abs/2412.07894) and is
restructured so that every cost with a different scaling law gets its own column. One design row
per candidate:

```text
row = (1, weighted_tokens, weighted_slots, weighted_sum_squared_tokens)

latency = per_step_overhead
        + token_cost              * weighted_tokens
        + microbatch_overhead     * weighted_slots
        + token_quadratic_cost    * weighted_sum_squared_tokens
```

`LATENCY_TERM_COUNT` is 4 and the coefficient order is fixed by `LATENCY_COEFFICIENT_NAMES`, so
the fit and the predictor cannot disagree about which coefficient multiplies which column.
`latency_from_coefficients` is the single evaluator shared with the FSDP model.

The features are built by `latency_features` for the critical DP rank:

```text
stage_share = 1 / PP

weighted_tokens = (sum over microbatches of sum_tokens
                   + (PP - 1) * max_microbatch_tokens) * stage_share

weighted_sum_squared_tokens = (sum over microbatches of sum_squared_tokens
                   + (PP - 1) * max_microbatch_sum_squared_tokens) * stage_share

weighted_slots = (num_microbatches + PP - 1) * stage_share
slots          = num_microbatches + PP - 1
```

Read this carefully, because it is where a plausible reimplementation goes wrong.

- **The token terms are sums over the microbatches a stage actually runs**, not the largest
  microbatch times the slot count. A stage executes every microbatch it is given, so a step costs
  the sum of their costs. The `PP - 1` bubble slots it cannot fill are charged at the largest
  microbatch, because that sets the rate at which the pipeline fills and drains. Charging every
  slot the maximum treats a ragged batch as though every sequence were the longest: it
  overpredicts and flattens the prediction to near-constant, since the longest sequence barely
  moves between steps while the token total varies widely. Profiling measures uniform
  microbatches, where the two forms coincide, which is why a formula written the old way still
  fitted correctly and could not be reimplemented faithfully.
- **Every term carries `1 / PP`.** A 1F1B stage owns `num_layers / PP` of the model, so its cost
  per slot falls by `PP` while the slot count rises to `nmb + PP - 1`. Both factors are known from
  the topology, which is what makes pipeline parallelism derivable rather than something to
  profile. Point-to-point activation sends between stages are folded into the fitted
  per-microbatch cost rather than modelled.
- **The quadratic term is $\sum_i s_i^2$, not $(\sum_i s_i)^2$.** Attention does not cross a
  `cu_seqlens` boundary, so a packed microbatch costs the sum of each sequence's square. Squaring
  the packed total overcharges by the sequence count. `SequenceStats.sum_squared_tokens` records
  the right quantity, and a THD sweep with shapes that share a total but split it differently is
  what makes the column identifiable rather than collinear with the linear one.
- **The per-microbatch overhead** is charged once per slot, covering tensor setup, optimizer
  updates and communication that do not grow with sequence length. Measured per-microbatch times
  grow by less than the token count when sequence length doubles, and that sub-linear growth is
  exactly this term.

`tokens` and `sum_squared_tokens` (the per-microbatch maxima) are kept separately in
`LatencyFeatures` because they are what the feature ranges gate on: exceeding a profiled range is
a property of one microbatch, not of the batch total.

### Why this restructure

The original Appendix C form folds the pipeline bubble into the workload features `Q_eff` and
`T_eff`. Those two columns are collinear by construction at uniform microbatch shape, so a
quadratic term cannot be separated from a linear one. Separating the four costs above leaves each
coefficient identifiable in principle. Whether they are well conditioned in practice is a separate
question, and {doc}`/pipeline/profiling/methodology` covers it: the two token columns stay highly
correlated on a grid that varies length without varying the packing structure, and a coefficient
pushed to its bound of zero is a symptom of that, not a measurement.

### The quadratic term is domain-dependent

Whether a quadratic token term earns its place depends on the sequence lengths swept. At short
lengths attention is a small share of a layer under full activation recompute, so the term
absorbs residual and leave-one-out can prefer the linear form. At the lengths a real RL batch
produces, attention dominates enough that omitting the term is off by a factor. The fitted
`feature_ranges` are therefore also a statement about which form is valid where: a prediction
beyond the profiled length is extrapolating the functional form as well as the value.

### DP treatment

DP communication is not modelled. DP affects the local workload because the planner distributes
trajectory lengths across DP ranks and the critical rank's microbatches are the input, and the
fitted coefficients describe one rank running that work. The DP collective is assumed ideal, and
the artifact records `dp_latency: ideal-local-workload-scaling`.

## Memory model

Peak memory on the most loaded rank is

```text
predicted = static_inventory(parameter_count, parallelism, dtypes, distributed_optimizer)
          + activation_scale * activation_bytes(geometry, microbatches, parallelism)
          + residual_constant
```

and the result is rounded up to an integer byte count at predict time.

The activation feature is an **architecture-derived byte count**, not an opaque token count, so
`activation_scale` is a dimensionless correction that should land near one. A value far from one
means the geometry is wrong rather than the machine being unusual.

### Activation geometry

Under full recomputation with `recompute_method=uniform`, each transformer layer keeps only its
own input. Intra-layer tensors are rebuilt one layer at a time during backward. The output
projection is not recomputed, so vocabulary-sized logits survive the forward pass. Per token:

```text
checkpoints = (num_layers + 1) * hidden_size * activation_dtype_bytes / TP
logits      = vocab_size * activation_dtype_bytes * LOGIT_OVERHEAD
```

The layer term is divided by the tensor degree. The logit term is not, because each rank
materialises a full-vocabulary-sized loss footprint however the logits are sharded.

Under 1F1B the first stage keeps up to `min(nmb, PP)` microbatches in flight before its first
backward but holds no logits, while the last stage holds the logits with an in-flight depth of
one. `activation_bytes` therefore models both ends and returns the larger:

```text
first_stage = (checkpoints / PP) * min(nmb, PP)
last_stage  = (checkpoints / PP) + logits
```

Modelling only the first end, or scaling checkpoints by `min(nmb, PP)` on whichever stage is
heaviest, fails to reproduce the measured fact that the peak stage's footprint is insensitive to
the microbatch count at `PP > 1` while non-peak ranks do move.

### Static inventory

`srl/cost/training/inventory.py` sizes the Megatron distributed-optimizer layout on the heaviest
pipeline stage:

| pool | per-rank size | sharded by |
|---|---|---|
| model parameters | `local * parameter_bytes` | TP and PP |
| gradient buffer | `local * gradient_bytes` | TP and PP, **not DP** |
| fp32 master copy | `local * optimizer_state_bytes / DP`, only when the parameter dtype is narrower | TP, PP, DP |
| two Adam moments | `local * 2 * optimizer_state_bytes / DP` | TP, PP, DP |

The gradient buffer is the reduce-scatter source, so every rank holds it at full
model-parallel-local size. Only master weights and optimizer state are sharded by DP. Omitting the
gradient buffer from the DP divisor, or sharding it, changes the static term roughly twofold.
Moments are allocated lazily on the first optimizer step, so a snapshot taken before that step
reports less than steady state.

`local` is not `parameter_count / (TP * PP)`. The embedding sits on the first stage and the output
projection on the last, and neither is a transformer layer, so `heaviest_stage_parameters` adds
one vocabulary-sized embedding to a stage's share of the transformer layers and divides by TP.
Peak memory is set by the worst stage.

### Target and OOM constraints

The regression target is `peak_allocated_bytes`, the live-tensor high-water mark. Reserved bytes
are the caching allocator's segment pool: quantised, sticky, and under full recompute nearly
unresponsive to sequence length, which makes them a near-zero-signal target. Reserved is still
recorded. A caller that needs a provisioning number should add an allocator margin on top of the
prediction rather than treating this model's output as one.

Each OOM record contributes an inequality: the true peak was at least the device capacity. The fit
computes the deficit between that requirement and the fitted surface and raises
`residual_constant` by the largest deficit, so the shipped model never predicts that an
OOM-measured configuration fits. The increase is recorded in `oom_intercept_increase`.

## Cell key and lookup

`_CellKey` is

```text
(tp, pp, cp, ep, schedule, recompute, optimizer,
 parameter_dtype, gradient_dtype, optimizer_state_dtype,
 distributed_optimizer, parameter_count, activation geometry)
```

- `ep` is part of the key because a different expert degree means different all-to-all traffic and
  a different number of experts per rank. It is 1 for every dense model.
- `parameter_count` keeps a smoke run or a different checkpoint out of a production cell.
- The activation geometry keeps two models with the same parameter count but different depth or
  width apart.

**Latency pools pipeline degrees. Memory does not.** The latency features are per-layer and the
slot count is analytic, so every PP in a TP group is fitted together and the surviving entry is
addressed with `pp = 1`. `CostArtifact.entry_for(..., ignore_pp=True)` finds it for any requested
PP. Memory is fitted per PP, because the 1F1B peak depends on which stage is in flight and how
many microbatches it holds, which is not a per-stage division of one number.

At predict time DP is canonicalised to 1 and the query's own DP scales the local workload, so a
cell fitted at one DP answers queries at another. Every other selector must match exactly, and an
unrecognised combination raises `KeyError` rather than falling back to a neighbour.

`feature_ranges` gate the query: `tokens`, `pipeline_slots`, `sum_squared_tokens` and
`micro_batch_size` for latency, `activation_bytes` for memory. A value outside a
recorded range raises `TrainingOutOfDomainError` unless the name is listed under
`training_cost.extrapolate`. A permitted extrapolation is recorded in the estimate's diagnostics.

`profiled_dp` is recorded in `feature_ranges` and can be named under `extrapolate`, but no
prediction path compares a query against it. The memory model pools DP into one cell, so the
degree a cell was fitted at does not constrain the queries it answers.

## Fitting

```bash
srl-regress training \
    --latency-model hydraulis --memory-model hydraulis \
    --input  outputs/profile/qwen3_8b/megatron_bridge/profiles/megatron_bridge.jsonl \
    --output outputs/cost/qwen3_8b/training_hydraulis.json
```

- Records must be `training` records with `cp = 1` and schedule `1f1b`. Anything else is skipped
  during grouping and the fit raises if nothing remains.
- Each candidate contributes one design row: its retained step times reduced by `robust_step_time`,
  a median. Every step entering as its own row would let a noisy candidate outvote a clean one.
- The split is by `run_id` at `--validation-fraction` (default `0.2`).
- The solve is non-negative soft-L1 least squares on column-scaled, **relatively** weighted
  residuals, followed by one MAD-based refit. Relative residuals matter because a training sweep
  spans orders of magnitude: minimising absolute error optimises the largest candidates and lets
  the small ones absorb any relative error.
- `check_latency_rank` requires rank 4 and names the axes to vary when it is not reached.
  `--allow-rank-deficient` lowers the check. The memory design is `[activation_bytes, 1]` and
  requires rank 2.
- Each entry records `max_step_dispersion` and `median_step_dispersion` alongside the usual fit
  metrics, and whether its validation metrics are in sample.

## Artifact structure

One `CostArtifact` holds a `latency` entry per `(TP, EP, parameter count, geometry)` group and a
`memory` entry per full cell. The fields a reader should expect:

```javascript
{
  "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1, "ep": 1},
  "selector": {
    "component": "latency",            // or "memory"
    "schedule": "1f1b",
    "recompute": "full",
    "optimizer": "adam",
    "parameter_dtype": "bf16",
    "gradient_dtype": "fp32",
    "optimizer_state_dtype": "fp32",
    "distributed_optimizer": true
  },
  "coefficients": {
    // latency
    "per_step_overhead": 0.0, "token_cost": 0.0,
    "microbatch_overhead": 0.0, "token_quadratic_cost": 0.0
    // memory instead carries activation_scale, residual_constant, parameter_count
    // and geometry_* keys (num_layers, hidden_size, vocab_size, activation_dtype_bytes)
  },
  "feature_ranges": {
    "tokens": [0, 0], "pipeline_slots": [0, 0], "sum_squared_tokens": [0, 0],
    "micro_batch_size": [1, 1]
    // memory: activation_bytes, profiled_dp
  },
  "train_metrics": {"n_samples": 0, "design_rank": 4, "r_squared": 0.0, "mre_percent": 0.0},
  "validation_metrics": {"n_samples": 0},
  "diagnostics": {
    "formula": "per_step_overhead + (token_cost * (sum_over_microbatches(sum_tokens) + (PP - 1) * max_microbatch_tokens) + token_quadratic_cost * (sum_over_microbatches(sum_squared_tokens) + (PP - 1) * max_microbatch_sum_squared_tokens) + microbatch_overhead * (num_microbatches + PP - 1)) / PP",
    "pipeline_latency": "derived-per-layer-1f1b",
    "profiled_pipeline_degrees": [1],
    "dp_latency": "ideal-local-workload-scaling",
    "step_aggregation": "median-per-candidate",
    "validation_fraction": 0.2,
    "validation_is_in_sample": false,
    "max_step_dispersion": 0.0,
    "median_step_dispersion": 0.0
    // memory adds memory_metric, oom_constraints, oom_intercept_increase
  }
}
```

Read the `formula` string as documentation of the shipped form: it is written in the same terms
`LatencyFeatures.design_row` builds.

## Using the artifact

```yaml
training_cost: hydraulis
training_cost:
  artifact_path: outputs/cost/qwen3_8b/training_hydraulis.json
  parallelism: {tp: 1, pp: 1, dp: 8, cp: 1}
  schedule: 1f1b
  activation_recompute: full
  optimizer: adam
  parameter_dtype: bf16
  gradient_dtype: fp32
  optimizer_state_dtype: fp32
  distributed_optimizer: true
  extrapolate: []
```

Dtypes and `distributed_optimizer` may be omitted when the artifact holds a single value for them.
`from_artifact` infers it and raises when the artifact holds several. An unknown name in
`extrapolate` raises at construction rather than silently enabling nothing.

The model supports CP=1 and the 1F1B schedule. VPP, interleaved 1F1B and non-uniform pipeline
partitions are not modelled. Coverage gaps are listed in {doc}`/reference/limitations`.

## The kernel-composed alternative

`srl/cost/training/composed.py` defines `ComposedTraining`, a `TrainingLatencyModel` that sums
measured kernels instead of fitting a polynomial. It carries no coefficients and is not a Hydra
default: construct it in Python and pair it with a memory model, typically
`HydraulisMemoryModel` loaded from an artifact.

```python
from srl.cost.kernel.database import KernelDatabase
from srl.cost.training.composed import ComposedTraining

db = KernelDatabase.from_root("outputs/profile/qwen3_8b/kernel_tp2/kernel", gpu="h20")
model = ComposedTraining.from_checkpoint(
    "outputs/profile/qwen3_8b/kernel_tp2/kernel",
    "/path/to/Qwen3-8B",
    gpu="h20",
    tensor_parallel=2,
    recompute="full",
)
latency = model.predict(workload, TrainingParallelism(tp=2, pp=3, dp=1))
```

What it prices, and where its input comes from:

| term | source |
|---|---|
| `gemm_fwd`, `gemm_bwd` | the `gemm` table. Backward includes dgrad and wgrad, whose `(n, k)` slices differ from the forward's |
| `attn_fwd`, `attn_bwd` | the `attention_training` table, both directions measured. THD over `cu_seqlens`, not the rollout `context_attention` table |
| `elementwise` | the `elementwise` table, including the `tensor_plumbing` kind |
| `tp_collective` | `reduce_scatter` and `all_gather` from the `collective` table |
| `pp_comm` | `p2p` per microbatch per boundary |
| `bubble` | `(slots - num_microbatches)` idle slots at one stage-pass each |
| `logits` | the `gemm` table, once per step. A missing dgrad slice is skipped rather than clamped onto a neighbour |
| `optimizer` | the `optimizer_step` table over one rank's parameter shard. Disable with `charge_optimizer=False` |

Per-layer work is charged **per microbatch**, and only the bubble is charged per slot. Each slot
is one pipeline stage's work, so a per-model multiplier is scaled by `1 / PP`. Attention is priced
per sequence in the pack rather than over the pack total, and the token count one slot is priced
at is the mean across the plan, which reproduces the true sum for a uniform plan and stays linear
for a ragged one.

`decompose_microbatches` returns the terms as a dict, and `predict_microbatches` is defined as
their sum, so the decomposition can never drift from the prediction. `predict` plans microbatches
first. Validation calls `predict_microbatches` with the plan the measured run actually executed,
so a planning disagreement is not folded into a composition error.
`srl/cost/training/validate.py` scores it against training records with `score_composed_training`
and reports MRE, signed mean, worst candidate, and mean per-term shares.

The model is not wired into the Hydra training-cost group, and `srl-regress` does not produce an
artifact for it.
