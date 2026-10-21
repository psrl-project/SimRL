# FSDP2 / HSDP training cost model

This page documents the FSDP2 / HSDP cost model: its latency form, memory design, static
inventory, the three batching modes, fitting and lookup, and how it differs from the Megatron
model. Measurements are on {doc}`/pipeline/profiling/fsdp`. Accuracy is in
{doc}`/reference/validation`.

FSDP and Megatron have separate models because they shard differently and their schedules differ.
Megatron's distributed optimizer shards optimizer state only and keeps a full-size fp32 gradient
buffer as the reduce-scatter source. FSDP2 shards parameters, gradients and optimizer state
unconditionally. Their memory inventories differ about twofold at the same model size, and reusing
one for the other silently overstates memory.

## Latency model

FSDP has no pipeline, so the shared four-term form ({doc}`/pipeline/cost-models/training-megatron`) degenerates: `PP = 1`
makes the bubble zero and the stage share one, so the design row is

```text
row = (1, sum_over_microbatches(sum_tokens), num_microbatches,
       sum_over_microbatches(sum_squared_tokens))

latency = per_step_overhead
        + token_cost            * sum_tokens across microbatches
        + microbatch_overhead   * num_microbatches
        + token_quadratic_cost  * sum_squared_tokens across microbatches
```

`latency_features` is the same function the Megatron model uses, so the two cannot drift, and
`latency_from_coefficients` is the single evaluator. The design must reach rank 4. With a single
microbatch count the slot column is constant and collinear with the intercept, so
`check_latency_rank` raises and names the axes to vary. At least two, and preferably three,
microbatch counts are needed alongside at least three sequence lengths.

The quadratic term is present for the same reason as in the Megatron model: the same code path
serves short- and long-sequence regimes, and the term must be $\sum_i s_i^2$ because attention
does not cross a packed sequence boundary. Under BSHD with a micro-batch size of one every
microbatch holds one sequence and the two token columns are collinear. That is a property of the
sweep, not of the model, and the THD hint in {doc}`/pipeline/profiling/methodology` applies to FSDP as
well.

DP communication is not modelled. The planner distributes trajectory lengths across DP ranks and
evaluates the critical rank, and the coefficients describe that rank's work with the collective
assumed ideal (`dp_latency: ideal-local-workload-scaling`).

## Memory model

```text
predicted = static_inventory(parameter_count, world_size, dtypes, optimizer_offload)
          + microbatch_slope   * num_microbatches
          + activation_slope   * max_tokens_per_microbatch
          + residual_constant
```

and the result is multiplied by `reserved_over_allocated` when the selected target is allocated,
then rounded up.

### Why the microbatch term exists

Under gradient checkpointing activations are recomputed on the backward pass and are not kept
resident between forward and backward. What drives peak memory instead is the number of in-flight
autograd graphs: each pins its intermediate tensors until `backward()` completes, and they
accumulate as microbatches are queued. Measured at a fixed microbatch count, peak memory is flat
to four decimal places across a multi-fold token range, while one extra microbatch adds a
measurable step.

Fitting `max_tokens` alone against such a target is degenerate: the single feature explains
nothing, the optimiser assigns all variance to the constant, and the coefficient of determination
comes out catastrophically negative. The design therefore keeps **both** a microbatch term and a
token term, fitted non-negatively:

- With gradient checkpointing the constraint drives the token slope to approximately zero rather
  than distorting the microbatch slope.
- Without checkpointing activations are resident and the token term dominates.
- At long sequences the token term bites even with checkpointing, which is why it cannot simply be
  dropped.

The memory design is `[num_microbatches, max_tokens, 1]` and must reach rank 3.

### Target selection and the reserved ratio

Both `peak_allocated_bytes` and `peak_reserved_bytes` are fitted, and the winner is selected per
cell on **validation** coefficient of determination, before any OOM bump. Allocated is the better
regression target because it responds to load, while reserved is quantised and sticky. The
selection is still a measurement, so both are attempted and both scores are recorded in
`candidate_r_squared`. A cell whose two targets score identically (which a one-sample validation
set makes inevitable) records `validation_selection_informative: false`, and the tie keeps
allocated.

The simulator compares the prediction with device capacity, and it is reserved bytes that exhaust
a device. The artifact therefore stores the mean measured `reserved_over_allocated` ratio and,
when allocated was selected, `predict` multiplies by it before the capacity comparison. A single
scalar is the weak part of this model: fragmentation tracks allocation churn rather than peak, so
`reserved - allocated` may be nearer constant than proportional. The ratio is stored rather than
only applied so a later refit can switch to an additive term once several load points exist.

### OOM constraints

Each OOM record requires `peak > device capacity`. The fit raises `residual_constant` by the
largest deficit, converting the capacity target into the selected metric's units first (dividing
by the reserved ratio when allocated is the target), so the bump is not applied twice. The bump is
included in the shipped constant and the metrics are recomputed against it, so the advertised
quality describes the coefficients the entry actually carries.

A conservative intercept can make in-sample $r^2$ poor for a cell rich in OOM constraints. That is
the mechanism working: the successful records establish the slope and the OOM records establish a
floor the successes never reached.

## Static inventory

`srl/cost/training/fsdp_inventory.py`:

```text
local = parameter_count / world_size
total = local * master_bytes        // model parameters, master dtype
      + local * gradient_bytes      // sharded gradient
      + local * 2 * master_bytes    // two Adam moments, master dtype
```

- `world_size` is the whole mesh, `shard_size * replicate_size`. Replication duplicates a shard
  across rows rather than widening it, so a rank's share depends only on how many ranks exist.
- The module is built in the **master dtype** (`optimizer_state_dtype`), because AdamW's moments
  inherit the dtype of the parameters it is handed. There is no separate master-weight tensor:
  unlike Megatron, the module itself is already the master.
- The `parameter_dtype` compute copy is transient. It is produced by `MixedPrecisionPolicy` for
  the all-gather and freed per block under `reshard_after_forward=True`, so it contributes **no
  bytes** to the resident total. `compute_element_bytes` exposes it for activation accounting only.
- With `optimizer_offload: true`, `CPUOffloadPolicy` moves the moments to pinned host memory.
  `optimizer_moments` becomes zero, their bytes are reported in `offloaded_bytes`, and the
  parameter and gradient shards stay on device.

For fp32 master weights (`optimizer_state_dtype: fp32`), fp32 gradients and no offload the total
is four bytes of parameters plus four of gradients plus eight of Adam moments, i.e. sixteen bytes
per parameter over the world size. For any other dtype combination it is not, which is why the
inventory is computed rather than assumed. The compute dtype (`parameter_dtype`) never enters the
resident total.

## Batching modes and the planner

How a global batch's trajectory lengths become microbatches sets the token count a rank runs,
which is the input the cost model evaluates. `MicrobatchPlanner` implements three modes:

| mode | behaviour |
|---|---|
| `pad_to_max` | split into groups of `micro_batch_size`. Each group is padded to its longest sequence |
| `token_budget` | group to `max_tokens_per_rank`. Still padded within a group |
| `packed_no_pad` | groups of `micro_batch_size` are concatenated end to end. Only true tokens are charged |

The planner:

1. Partitions the lengths across DP ranks with verl's workload proxy `24576 * L + L^2`, using
   greedy longest-processing-time placement. The proxy is reproduced from verl because a different
   one would partition differently and every predicted microbatch count would be systematically
   wrong.
2. Selects the critical rank, the one with the largest proxy sum, since a step ends when its
   slowest rank ends.
3. Splits that rank's lengths according to the mode.

In `token_budget` the count is `min(len(lengths), ceil(total / budget))` and `micro_batch_size` is
deliberately ignored: verl reads it only on the static path (`use_dynamic_bsz=False`), so using it
here would invent microbatches the engine never runs. The budget is a target, not a guarantee. One
sequence longer than the budget yields one oversized microbatch, as it does in verl.

At `micro_batch_size = 1` the three modes coincide: one sequence per microbatch leaves padding
nothing to fill. Even above one, the balanced partition groups similar lengths together and
suppresses the difference. The modes diverge only with `micro_batch_size > 1` **and** a partition
that mixes short and long sequences in one microbatch.

## Cell key and lookup

`_CellKey` is

```text
(shard_size, replicate_size, recompute, optimizer,
 parameter_dtype, gradient_dtype, optimizer_state_dtype,
 optimizer_offload, reshard_after_forward, parameter_count)
```

Every entry's `parallelism` is canonicalised to `tp = pp = cp = dp = 1`. The mesh travels in the
selector, because sharding and replication are configuration rather than data partitioning.

At predict time the model enforces two things the fitted form cannot express:

- The schedule must be `fsdp`, and `workload.recompute` must equal the cell's recompute policy.
- `parallelism.dp` must equal the cell's `world_size`. A cell is a measurement of exactly one mesh,
  and the static pool scales as `1 / world_size`, so answering a smaller-dp query against a larger
  mesh would under-report memory by exactly that factor. That is a fabrication rather than an
  extrapolation, and it would silently defeat the simulator's OOM guard.

`feature_ranges` gate the query: `tokens`, `num_microbatches` and `sequences_per_microbatch` for
latency, `num_microbatches` and `activation_tokens` for memory. A value outside a recorded range
raises unless the name is listed in `training_cost.extrapolate`. A permitted extrapolation is
recorded in the estimate's diagnostics. The names are per component, and an unknown name raises at
construction. {doc}`/pipeline/simulation/domain-gates` covers the semantics. The FSDP default file
`config/defaults/training_cost/fsdp.yaml` lists each selectable name and what it is safe to
extrapolate.

Note that the `tokens` gate is the **largest** microbatch on the critical rank. Extrapolating it
means predicting a shape the hardware may not be able to run at all, so it should only be listed
when the grid reaches the target range.

## Fitting

```bash
srl-regress training --latency-model fsdp --memory-model fsdp \
    --input  outputs/profile/qwen3_8b/fsdp/profiles/fsdp.jsonl \
    --output outputs/cost/qwen3_8b/training_fsdp.json
```

- FSDP latency and memory must be selected together. Selecting one raises.
- `_fsdp_records` refuses a record that is not an FSDP record rather than skipping it, rejects
  `sequence_parallel != 1`, requires `tp = pp = cp = 1`, requires `shard_size * replicate_size` to
  equal `dp`, and requires a `parameter_count`. Silently dropping a foreign record would hide a
  mis-pointed input file.
- Each candidate contributes one row, its step times reduced to a median. The split is by run id at
  `--validation-fraction` (default `0.2`).
- The solve matches the Megatron model: scaled non-negative soft-L1 with relative residuals, one
  MAD refit, and a rank check that `--allow-rank-deficient` lowers.
- Every entry records step-time dispersion and whether its validation metrics are in sample, along
  with `memory_metric`, `candidate_r_squared`, `reserved_over_allocated`, `oom_constraints` and
  `oom_intercept_increase`.

## Using the artifact

```yaml
training_cost: fsdp
training_cost:
  artifact_path: outputs/cost/qwen3_8b/training_fsdp.json
  parallelism: {tp: 1, pp: 1, dp: 8, cp: 1}
  shard_size: 8
  replicate_size: 1
  activation_recompute: full
  optimizer: adam
  batching_mode: token_budget
  max_tokens_per_rank: 10240
  micro_batch_size: 1
  extrapolate: []
```

`shard_size` and `replicate_size` select the cell. Omit either only when the artifact holds one
value for it. Dtypes, `optimizer_offload` and `reshard_after_forward` behave the same way.

`batching_mode` is a load-time choice, not a fitted coefficient:

```bash
TRAINING_BACKEND=fsdp BATCHING_MODE=packed_no_pad \
TRAINING_ARTIFACT=outputs/cost/qwen3_8b/training_fsdp.json \
bash examples/simulate/simulate.sh
```

Coverage gaps (`optimizer_offload: true` never profiled, sequence parallelism out of scope, a
narrow microbatch range) are in {doc}`/reference/limitations`.
