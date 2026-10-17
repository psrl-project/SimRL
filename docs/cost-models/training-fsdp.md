# FSDP2 / HSDP training cost model

This page documents the FSDP2 / HSDP cost model: its latency form, memory design, static
inventory, three batching modes, and the accuracy obtained on Qwen3-8B / 8x H20.

For how the measurements are taken (sweep axes, warmup policy, NCCL OOM classification,
output layout) see {doc}`../profiling/fsdp`. For general measurement rules, see
{doc}`../profiling/methodology`. The `CostArtifact` schema is in {doc}`../reference/schemas`.

These are **two separate pages for a reason.** Megatron's distributed optimizer shards
optimizer state only and keeps a full-size fp32 gradient buffer as the reduce-scatter source.
FSDP2 shards parameters, gradients, and optimizer state unconditionally. Their memory
inventories differ by roughly twofold at the same model size, and their memory features
differ too. Megatron's activation feature is architecture-derived, while FSDP's feature
design was dictated by a direct measurement showing memory flat to four decimal places across
a 4x token range. Reusing one inventory for the other quietly overstates FSDP memory by
about 2x.

---

## Latency model

With no pipeline (`PP = 1`), the pipeline-slot count in the shared latency form collapses
from $\text{nmb} + \text{PP} - 1$ to $\text{nmb}$. The FSDP latency formula is therefore:

$$
\text{step} = c_{\text{overhead}}
    + \text{nmb} \times \bigl(c_{\text{token}} \times T_{\max}
                              + c_{q} \times S_{\max}
                              + c_{\text{mb}}\bigr)
$$

where:

- $c_{\text{overhead}}$ is a per-step fixed overhead paid once regardless of the batch.
- $\text{nmb}$ is the number of microbatches (gradient-accumulation steps) on the critical DP
  rank.
- $T_{\max}$ is the maximum token count across all microbatches on the critical rank.
- $S_{\max}$ is that microbatch's `sum_squared_tokens`, meaning $\sum_i s_i^2$ over its packed
  sequences.
- $c_{\text{token}}$ is a per-token cost (seconds per token).
- $c_{q}$ is a per-squared-token cost, capturing attention.
- $c_{\text{mb}}$ is a per-microbatch overhead charged once per microbatch regardless of
  sequence length.

In code (`srl/cost/training/latency.py`, shared with the Megatron model):

```python
latency = coefficients["per_step_overhead"] + features.slots * (
    coefficients["token_cost"] * features.tokens
    + coefficients["token_quadratic_cost"] * features.sum_squared_tokens
    + coefficients["microbatch_overhead"]
)
```

This is exactly the shared latency structure, with `slots = nmb` rather than `nmb + PP - 1`.
Both backends carry all four terms. The quadratic term was originally rejected on a Megatron
sweep that stopped at 2048 tokens, where attention is only about 3.7% of a layer's FLOPs, and
is required once the profiled sequence length is long enough for attention to matter.
{doc}`training-megatron` has the domain evidence.

**Packing is why the quadratic term must be $\sum_i s_i^2$ here.** Of this sweep's successful
records, 74 microbatches hold one sequence, 54 hold two, and 38 hold four. Squaring the packed
total instead of summing the squares would overcharge those by the sequence count: four packed
2048-token sequences give 1.68e7 against 6.71e7, a 4x error. This sweep exercises the
difference, so getting it wrong was a live bug rather than a hypothetical one.

### The `token_cost` coefficient is physical evidence

Both token coefficients were fitted independently for each of the three shard-degree cells,
on data no other cell saw (Qwen3-8B / 8x H20 / 2026-09-03):

| cell | `token_cost` (µs per token) | `token_quadratic_cost` |
|---|---|---|
| shard=2 | 458.9 | 1.609e-8 |
| shard=4 | 467.0 | 1.473e-8 |
| shard=8 | 466.7 | 1.482e-8 |

Consistency to 2% on the linear term across three independent fits is evidence that the
regression recovered a physical quantity, the per-token compute cost on this hardware, rather
than a curve fitted to noise. The quadratic term agrees to within 9%, which is weaker but
still far tighter than noise would give.

The per-step and per-microbatch overheads are less stable across cells because they absorb
all-reduce timing and gradient communication, which do depend on the mesh.

### Design-matrix rank requirement

The latency design row for one candidate is
`[1, nmb * tokens, nmb, nmb * sum_squared_tokens]`. When `nmb` is fixed across all candidates
the third column is a scalar multiple of the first, capping the design at rank 3 where the fit
needs rank 4. The fit then cannot separate `per_step_overhead` from `microbatch_overhead` and
raises an error. Sweeping `sequence_lengths` alone does not fix this, because it varies the
token columns and leaves the slot column degenerate. **At least two distinct
`num_microbatches` values are required in every sweep you intend to fit.** See
{doc}`../profiling/fsdp` for the identifiability discussion.

### DP treatment

DP communication is not modelled. The planner distributes trajectory lengths across DP ranks
and uses the critical rank's microbatches. The fitted latency coefficients describe one rank's
share of per-step work, and the collective is assumed ideal. Recorded in the artifact as
`"dp_latency": "ideal-local-workload-scaling"`. See {doc}`../reference/limitations` for scope.

---

## Memory model

### Why memory tracks microbatches, not tokens

Under gradient checkpointing, activations are recomputed on the backward pass and are not kept
resident between the forward and the backward. What accumulates instead is the autograd graph
for each microbatch. Each graph holds its intermediate tensors until `backward()` completes,
so the graphs pile up when `num_microbatches > 1`.

Measured at shard=8, gradient checkpointing on, Qwen3-8B / 8x H20:

| nmb | tokens | peak allocated |
|---|---|---|
| 1 | 512 | 21.6982 GiB |
| 1 | 2048 | 21.6978 GiB |
| 1 | 8192 | 30.5970 GiB |
| 2 | 512 | 22.2782 GiB |
| 2 | 2048 | 22.2779 GiB |
| 4 | 512 | 22.2782 GiB |
| 4 | 2048 | 22.2779 GiB |

Two effects are visible, and the first is what drove the feature design.

**Across a 4x token range at fixed `nmb`, memory does not move**: 21.6982 against 21.6978 GiB,
flat to four decimal places and very slightly *down*. Going from one microbatch to two adds
about 580 MiB, and going from two to four adds nothing measurable, since the graph count that
matters is bounded by what is in flight rather than by the total. A model fitted only on
`max(sum_tokens)` is therefore degenerate: the single feature explains nothing, the optimizer
assigns all variance to the constant, and both allocated and reserved variants score
catastrophically negative $r^2$ ($-2.6 \times 10^9$ allocated, $-3.4 \times 10^5$ reserved).

**The token axis does eventually bite, past the checkpointed regime.** At 8192 tokens the same
`nmb=1` cell reaches 30.60 GiB, and at 16384 it reaches 47.01 GiB. That is why the token term
stays in the design rather than being dropped: it is genuinely zero over the short-sequence
range and genuinely dominant at the long end.

### Memory feature design

The residual (peak minus static inventory) is fitted against two features:
`num_microbatches` and `max_tokens_per_microbatch`. The design row is
`[num_microbatches, max_tokens, 1]`.

$$
\text{peak} = \text{static\_inventory}
    + c_{\text{nmb}} \times \text{nmb}
    + c_{\text{act}} \times T_{\max}
    + c_{\text{residual}}
$$

Both coefficients are fitted with a non-negative constraint. In a purely short-sequence
checkpointed regime that constraint drives $c_{\text{act}}$ toward zero rather than letting it
distort $c_{\text{nmb}}$. Without gradient checkpointing, or at the long sequences shown above,
activations are resident and $T_{\max}$ drives memory, so the token term must remain in the
design because the same code path handles both regimes.

### Memory target selection

Both `peak_allocated_bytes` and `peak_reserved_bytes` are fitted as regression targets, and
the winner is selected per cell on validation $r^2$ before the OOM bump. Selection is
per-cell and recorded in the artifact's `diagnostics["memory_metric"]` and
`diagnostics["candidate_r_squared"]`.

The simulator compares the prediction against device capacity, and an OOM is caused by
reserved bytes rather than allocated ones. To bridge the gap the artifact also stores the
measured `reserved_over_allocated` ratio. At predict time, when the selected target is
`peak_allocated_bytes`, the prediction is multiplied by this ratio before the capacity
comparison. The regression keeps the well-behaved target, and the capacity comparison keeps
the quantity that actually causes OOM.

Measured ratios (Qwen3-8B / 8x H20 / 2026-08-31):

| cell | selected target | r² (selected) | reserved / allocated |
|---|---|---|---|
| shard=2 | reserved | 0.61 | 1.148 |
| shard=4 | allocated | (see accuracy table) | 1.356 |
| shard=8 | allocated | (see accuracy table) | 1.901 |

shard=2 chose `reserved` because its allocated $r^2$ was only 0.30. The allocator caching
structure is cell-dependent.

---

## Static inventory

The static inventory is `16p / world_size`, covering parameters, gradients, and both Adam
moments, all sharded uniformly across the entire mesh.

FSDP2 shards all three resident pools. Each `fully_shard` unit reduce-scatters per-layer
into already-sharded gradients, so no full-size buffer exists anywhere. Measured on 8 ranks
with a 1.630B-parameter model under AdamW, the allocator reported 3.101 GiB against an
analytic `16p / world_size = 3.037 GiB` (ratio 0.979).

`world_size` is the whole mesh, meaning `shard_size * replicate_size`. Replication duplicates a
shard across rows rather than widening it, so a rank's share depends only on how many ranks
there are in total, not on how they are laid out.

There is **no `distributed_optimizer` flag** for FSDP. Sharding is unconditional, which
is the FSDP2 semantic. Contrast with Megatron, where the distributed optimizer is a
switch: when off, the gradient buffer is unsharded and the optimizer state is replicated.
Applying the Megatron inventory to FSDP would overstate memory roughly twofold.

### Physical layout of resident tensors

Under `MixedPrecisionPolicy(param_dtype=bf16)` with `reshard_after_forward=True`:

- The module is built in the **master dtype** (`optimizer_state_dtype`, usually fp32). There
  is one resident parameter shard per rank, sized by the master dtype.
- The transient all-gather copy in the compute dtype (bf16) is freed after each transformer
  block under `reshard_after_forward=True`. It is **not** part of the static pool.
- Gradient shards are in `gradient_dtype` (usually fp32).
- Both Adam moments are in the master dtype.

No second master-weight tensor exists. Megatron needs one because its module lives in the
narrow dtype and the master is a separate fp32 buffer. FSDP2 has no such buffer: the module
itself is already the master.

In `srl/cost/training/fsdp_inventory.py`:

```python
local = parameter_count / world_size
total_bytes = (
    local * master_bytes          # model parameters (master dtype)
    + local * gradient_bytes      # gradient shard
    + local * 2 * master_bytes    # two Adam moments (master dtype)
)
```

The `parameter_dtype` (compute dtype) contributes **zero bytes** to `total_bytes` because the
all-gathered compute copy is transient. It is exposed as `compute_element_bytes` for
downstream activation accounting only.

---

## Batching modes and the microbatch planner

FSDP training in verl uses gradient accumulation over multiple microbatches. How the global
batch's real trajectory lengths are divided into microbatches determines the token count each
rank actually runs, which is the input the cost model evaluates.

Three modes are supported (`srl/cost/training/microbatch.py`):

| mode | description |
|---|---|
| `pad_to_max` | Split into groups of `micro_batch_size`. Each group is padded to its longest sequence. |
| `token_budget` | Group to a token budget per rank using verl's balancing proxy. Padded within each group. |
| `packed_no_pad` | Sequences are concatenated end-to-end within each microbatch, with no padding tokens charged. |

The measured latency spread between the extremes is 1.17x to 4.34x on DAPO-like bimodal
lengths, so the batching mode is a first-class input to the cost model rather than a detail.
The 1.17x to 4.34x range comes from kernel-level packing gain on deliberately ragged
microbatches. See {doc}`../reference/limitations` for when it does and does not transfer to a
balanced planner.

For each mode the planner:
1. Distributes the global batch's sequence lengths across DP ranks using verl's workload
   proxy ($24576 \times L + L^2$) with a greedy longest-processing-time partition.
2. Identifies the critical DP rank, the one with the largest total workload proxy sum.
3. Converts the critical rank's lengths into microbatches according to the chosen mode.

The workload proxy is reproduced exactly from verl so that the planner's partition agrees
with verl's partition. A different proxy would produce different microbatch counts and
every predicted cost would be systematically wrong.

At `micro_batch_size=1` all three modes produce identical results: one sequence per
microbatch means padding has nothing to fill. Even at `mbs > 1` the balanced partition
groups similar lengths together and suppresses the mode difference. To observe the modes
diverge you need `mbs > 1` **and** a partition that mixes short and long sequences in the
same microbatch.

Batching mode is set at artifact load time:

```bash
# verl default: token_budget with 10240 tokens per rank
TRAINING_BACKEND=fsdp BATCHING_MODE=token_budget MAX_TOKENS_PER_RANK=10240 \
TRAINING_ARTIFACT=outputs/cost/qwen3_8b/training_fsdp.json \
bash examples/simulate/simulate.sh

# charged on true tokens, no pad overhead
TRAINING_BACKEND=fsdp BATCHING_MODE=packed_no_pad \
TRAINING_ARTIFACT=outputs/cost/qwen3_8b/training_fsdp.json \
bash examples/simulate/simulate.sh
```

---

## Fitting

```bash
srl-regress training --latency-model fsdp --memory-model fsdp \
    --input  outputs/profile/qwen3_8b/fsdp/profiles/fsdp.jsonl \
    --output outputs/cost/qwen3_8b/training_fsdp.json
```

The fitter partitions records by the `shard_size` tag (and the full discrete cell key:
`recompute`, `optimizer`, dtypes, `optimizer_offload`, `reshard_after_forward`,
`parameter_count`) and fits each cell independently. Records for different shard degrees are
never mixed. The memory fit additionally selects the better target per cell.

For both latency and memory, each candidate contributes one design row, its step times
reduced to a median. OOM records contribute inequality constraints that lift the memory
intercept rather than being discarded as failures.

---

## Accuracy (measured, 108 records, Qwen3-8B / 8x H20)

73 success / 34 OOM / 1 failed, fitted with `--validation-fraction 0.2`. Every entry carries
`validation_is_in_sample: false`, so the held-out column is a real split. See
{doc}`../reference/validation` for how this compares with the other models.

| component | cell | train n | rank | train r² | train MRE | held-out n | held-out MRE |
|---|---|---|---|---|---|---|---|
| latency | shard=2 | 15 | 4 | 0.99877 | 4.236 % | 4 | 6.519 % |
| latency | shard=4 | 22 | 4 | 0.99999 | 0.777 % | 5 | 0.820 % |
| latency | shard=8 | 22 | 4 | 0.99999 | 0.648 % | 5 | 1.394 % |
| memory | shard=2 | 15 | 3 | −8.189 | 5.053 % | 4 | 4.897 % |
| memory | shard=4 | 22 | 3 | 0.885 | 4.673 % | 5 | 5.981 % |
| memory | shard=8 | 22 | 3 | 0.981 | 4.268 % | 5 | 5.388 % |

Latency error here is **lower** than the Megatron model's 3.8% to 10.9%, which is the reverse
of what an earlier version of this page reported. Two things explain it. This sweep has 15 to
22 candidates per cell against Megatron's 8, and its token domain reaches 16384 against 8192,
so the two token columns are better separated and the quadratic term is properly identified
rather than trading against the linear one.

Held-out error tracking in-sample error this closely is the practical definition of
generalisation. The model is not memorising its training set.

### The shard=2 memory r² of −8.19 is the diagnostic working

A negative $r^2$ means the prediction is worse than simply guessing the target mean, which
would be a failure if the regression had only seen successes. The shard=2 cell has 16 OOM
inequality constraints against 15 successful records. The inequalities lift the intercept about
$5.1 \times 10^9$ bytes above what the successes alone support, and the raised intercept then
over-predicts those successes by that margin.

This is **intentional and correct**. The successful records established the slope, and the OOM
records established a floor the successes never reached. The fitted model is deliberately
conservative: it will not predict that a workload fits on device when the OOM evidence says
otherwise. The MRE of 5.05% is evaluated against the successful records and describes the
quality of the slope, which is what varies in real queries.

An earlier version of the artifact claimed $r^2 = 1.0$ for this cell. It shipped coefficients
it had not computed metrics for, so the reported quality was meaningless. The honest answer
is $r^2 = -8.19$.

### Relationship between MRE and the OOM bump

The OOM bump raises `residual_constant` in the artifact. The $r^2$ and MRE in the accuracy
table are computed against the **final shipped constant** rather than the
regression-converged one, so the advertised quality describes the coefficients the model
actually uses at prediction time.

---

## Artifact structure

```javascript
{
  "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1},
  "selector": {
    "component": "latency",          // or "memory"
    "schedule": "fsdp",
    "shard_size": 8,
    "replicate_size": 1,
    "recompute": "full",
    "optimizer": "adam",
    "parameter_dtype": "bf16",
    "gradient_dtype": "fp32",
    "optimizer_state_dtype": "fp32",
    "optimizer_offload": false,
    "reshard_after_forward": true
  },
  "coefficients": {
    "per_step_overhead": 0.07527,        // latency only, seconds
    "token_cost": 0.0004589,             // latency only, seconds per token
    "token_quadratic_cost": 1.609e-08,   // latency only, seconds per squared token
    "microbatch_overhead": 0.4181,       // latency only, seconds per microbatch
    "microbatch_slope": 993942060.1,     // memory only, bytes per microbatch
    "activation_slope": 181751.1,        // memory only, bytes per activation token
    "residual_constant": 84505316328.1,  // memory only, bytes
    "parameter_count": 8190735360.0,     // memory only
    "world_size": 8.0                    // memory only
  },
  "feature_ranges": {
    "tokens": [512.0, 16384.0],                    // latency
    "num_microbatches": [1.0, 4.0],                // latency and memory
    "sequences_per_microbatch": [1.0, 4.0],        // latency
    "activation_tokens": [512.0, 65536.0]          // memory
  },
  "diagnostics": {
    "memory_metric": "peak_reserved_bytes",    // memory only, chosen per cell
    "reserved_over_allocated": 1.148,          // memory only, applied at predict time
    "oom_constraints": 16,                     // memory only
    "oom_intercept_increase": 5109677791.8,    // memory only, bytes
    "validation_fraction": 0.2,
    "validation_is_in_sample": false
  }
}
```

---

## Mesh constraint at prediction time

A cell is a measurement of exactly one HSDP mesh. The predictor enforces:

```python
if parallelism.dp != self.world_size:
    raise TrainingOutOfDomainError(...)
```

A query for a different `dp` is rejected rather than silently returning the wrong answer.
The sharded static inventory scales as $1/\text{world\_size}$, so answering a dp=4 query
against an 8-rank artifact would under-report memory by exactly twofold without the check.
For extrapolation and out-of-domain semantics see {doc}`../simulation/domain-gates`.

---

## Using the artifact

In the Hydra simulation config:

```yaml
training_cost: fsdp
training_cost:
  artifact_path: outputs/cost/qwen3_8b/training_fsdp.json
  parallelism:
    tp: 1
    pp: 1
    dp: 8
    cp: 1
  shard_size: 8
  replicate_size: 1
  activation_recompute: full
  batching_mode: token_budget
  max_tokens_per_rank: 10240
  micro_batch_size: 1
  extrapolate: [num_microbatches]
```

The `extrapolate: [num_microbatches]` permits out-of-range microbatch queries, which is
necessary when the simulator's batch-size and DP combination requires more microbatches than
were profiled. Fitting on `nmb ∈ {1, 2}` and predicting the 23 measured `nmb=4` candidates gave
a median absolute error of 7.7%, which is above the model's own in-domain error but still a
usable bound, so the upward extrapolation is defensible. Out-of-domain queries are never
silent: the exceedance is recorded in
`TrainingCostEstimate.diagnostics["extrapolation_notes"]`.

Known gaps in coverage, including the narrow `nmb` range [1, 4], `optimizer_offload=True`
never profiled, and Ulysses sequence parallelism out of scope, are documented in
{doc}`../reference/limitations`.
