# FSDP2 / HSDP training cost model

This page documents the FSDP2 / HSDP cost model — its latency form, memory design, static
inventory, three batching modes, and the accuracy obtained on Qwen3-8B / 8x H20 / 2026-08-31.

For how the measurements are taken — sweep axes, warmup policy, NCCL OOM classification,
output layout — see {doc}`../profiling/fsdp`. For general measurement rules, see
{doc}`../profiling/methodology`. The `CostArtifact` schema is in {doc}`../reference/schemas`.

These are **two separate pages for a reason.** Megatron's distributed optimizer shards
optimizer state only and keeps a full-size fp32 gradient buffer as the reduce-scatter
source. FSDP2 shards parameters, gradients, and optimizer state unconditionally. Their
memory inventories differ by roughly twofold at the same model size, and their memory
features differ too: Megatron's activation feature is architecture-derived, while FSDP's
feature design was dictated by a direct measurement that showed memory flat to four decimal
places across a 4× token range. Reusing one inventory for the other quietly overstates
FSDP memory ~2×.

---

## Latency model

With no pipeline (`PP = 1`), the pipeline-slot count in the shared latency form collapses
from $\text{nmb} + \text{PP} - 1$ to $\text{nmb}$. The FSDP latency formula is therefore:

$$
\text{step} = c_{\text{overhead}}
    + \text{nmb} \times \bigl(c_{\text{token}} \times T_{\max} + c_{\text{mb}}\bigr)
$$

where:

- $c_{\text{overhead}}$ is a per-step fixed overhead paid once regardless of the batch.
- $\text{nmb}$ is the number of microbatches (gradient-accumulation steps) on the
  critical DP rank.
- $T_{\max}$ is the maximum token count across all microbatches on the critical rank.
- $c_{\text{token}}$ is a per-token cost (seconds per token).
- $c_{\text{mb}}$ is a per-microbatch overhead charged once per microbatch regardless of
  sequence length.

In code (`dryrun/cost/training/fsdp.py`):

```python
latency = coefficients["per_step_overhead"] + features.slots * (
    coefficients["token_cost"] * features.tokens + coefficients["microbatch_overhead"]
)
```

This is exactly the shared latency structure from `dryrun/cost/training/latency.py`, with
`slots = nmb` rather than `nmb + PP - 1`. Reusing that form ensures the quadratic token
term that was measured and rejected for Megatron cannot reappear here under a different
name. See {doc}`../profiling/methodology` for the rejection evidence.

### The `token_cost` coefficient is physical evidence

The `token_cost` coefficient was fitted independently for each of the three shard-degree cells.
The results (Qwen3-8B / 8x H20 / 2026-08-31):

| cell | token\_cost (µs per token) |
|---|---|
| shard=2 | 583.8 |
| shard=4 | 600.5 |
| shard=8 | 601.5 |

Consistency to 3 % across three cells that never saw each other's data is evidence the fit
recovered a physical quantity — the per-token compute cost on this hardware — rather than a
curve fitted to noise. The per-step overhead and per-microbatch overhead are less stable
across cells because they absorb all-reduce timing and gradient communication, which do
depend on the mesh.

### Design-matrix rank requirement

The latency design row for one candidate is `[1, nmb * tokens, nmb]`. When `nmb` is fixed
across all candidates the third column is a scalar multiple of the first, capping the design
at rank 2. The fit then cannot separate `per_step_overhead` from `microbatch_overhead` and
raises an error. Sweeping `sequence_lengths` alone does not fix this: it varies the second
column but leaves the third degenerate. **At least two distinct `num_microbatches` values
are required in every sweep you intend to fit.** See {doc}`../profiling/fsdp` for the
identifiability discussion.

### DP treatment

DP communication is not modelled. The planner distributes trajectory lengths across DP ranks
and uses the critical rank's microbatches. The fitted latency coefficients describe one rank's
share of per-step work; the collective is assumed ideal. Recorded in the artifact as
`"dp_communication": "ignored"`. See {doc}`../reference/limitations` for scope.

---

## Memory model

### Why memory tracks microbatches, not tokens

Under gradient checkpointing, activations are recomputed on the backward pass and are not kept
resident between the forward and the backward. What accumulates instead is the autograd graph
for each microbatch: each graph holds its intermediate tensors until `backward()` completes,
so the graphs pile up when `num_microbatches > 1`.

Measured at shard=8, gradient checkpointing on (Qwen3-8B / 8x H20 / 2026-08-31):

| nmb | tokens | peak allocated |
|---|---|---|
| 1 | 512 | 9.4784 GiB |
| 1 | 1024 | 9.4784 GiB |
| 1 | 2048 | 9.4785 GiB |
| 2 | 512 | 10.0585 GiB |
| 2 | 1024 | 10.0585 GiB |
| 2 | 2048 | 10.0585 GiB |

Memory is flat to four decimal places across a 4× token range; one extra microbatch adds
~580 MiB. A model fitted only on `max(sum_tokens)` is degenerate: the single feature
explains nothing, the optimizer assigns all variance to the constant, and both allocated and
reserved variants score catastrophically negative $r^2$ ($-2.6 \times 10^9$ allocated,
$-3.4 \times 10^5$ reserved) in that configuration.

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

Both coefficients are fitted with a non-negative constraint. Under gradient checkpointing the
non-negative constraint drives $c_{\text{act}}$ toward zero rather than allowing it to distort
$c_{\text{nmb}}$. Without gradient checkpointing activations are resident and $T_{\max}$ drives
memory; the token term must therefore remain in the design even when its coefficient is near
zero in the checkpointed case, because the same code path must handle both policies.

### Memory target selection

Both `peak_allocated_bytes` and `peak_reserved_bytes` are fitted as regression targets, and
the winner is selected per cell on validation $r^2$ before the OOM bump. Selection is
per-cell and recorded in the artifact's `diagnostics["memory_metric"]` and
`diagnostics["candidate_r_squared"]`.

The simulator compares the prediction against device capacity, and an OOM is caused by
reserved bytes rather than allocated ones. To bridge the gap the artifact also stores the
measured `reserved_over_allocated` ratio. At predict time, when the selected target is
`peak_allocated_bytes`, the prediction is multiplied by this ratio before the capacity
comparison. The regression keeps the well-behaved target; the capacity comparison keeps
the quantity that actually causes OOM.

Measured ratios (Qwen3-8B / 8x H20 / 2026-08-31):

| cell | selected target | r² (selected) | reserved / allocated |
|---|---|---|---|
| shard=2 | reserved | 0.61 | 1.148 |
| shard=4 | allocated | (see accuracy table) | 1.356 |
| shard=8 | allocated | (see accuracy table) | 1.901 |

shard=2 chose `reserved` because its allocated $r^2$ was only 0.30 — the allocator caching
structure is cell-dependent.

---

## Static inventory

The static inventory is `16p / world_size` — parameters, gradients, and both Adam moments,
all sharded uniformly across the entire mesh.

FSDP2 shards all three resident pools. Each `fully_shard` unit reduce-scatters per-layer
into already-sharded gradients; no full-size buffer exists anywhere. Measured on 8 ranks
with a 1.630B-parameter model under AdamW, the allocator reported 3.101 GiB against an
analytic `16p / world_size = 3.037 GiB` (ratio 0.979).

`world_size` is the whole mesh — `shard_size × replicate_size`. Replication duplicates a
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

In `dryrun/cost/training/fsdp_inventory.py`:

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

Three modes are supported (`dryrun/cost/training/microbatch.py`):

| mode | description |
|---|---|
| `pad_to_max` | Split into groups of `micro_batch_size`. Each group is padded to its longest sequence. |
| `token_budget` | Group to a token budget per rank using verl's balancing proxy. Padded within each group. |
| `packed_no_pad` | Sequences are concatenated end-to-end within each microbatch; no padding tokens are charged. |

The measured latency spread between the extremes is 1.17× to 4.34× on DAPO-like bimodal
lengths, so the batching mode is a first-class input to the cost model rather than a detail.
The 1.17–4.34× range comes from kernel-level packing gain on deliberately ragged microbatches;
see {doc}`../reference/limitations` for when it does and does not transfer to a balanced
planner.

For each mode the planner:
1. Distributes the global batch's sequence lengths across DP ranks using verl's workload
   proxy ($24576 \times L + L^2$) with a greedy longest-processing-time partition.
2. Identifies the critical DP rank — the one with the largest total workload proxy sum.
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
dryrun-regress training --latency-model fsdp --memory-model fsdp \
    --input  outputs/profile/qwen3_8b/fsdp/profiles/fsdp.jsonl \
    --output outputs/cost/qwen3_8b/training_fsdp.json
```

The fitter partitions records by the `shard_size` tag (and the full discrete cell key:
`recompute`, `optimizer`, dtypes, `optimizer_offload`, `reshard_after_forward`,
`parameter_count`) and fits each cell independently. Records for different shard degrees are
never mixed. The memory fit additionally selects the better target per cell.

For both latency and memory, each candidate contributes one design row — its step times
reduced to a median. OOM records contribute inequality constraints that lift the memory
intercept rather than being discarded as failures.

---

## Accuracy (measured, in-sample, 108 records, Qwen3-8B / 8x H20 / 2026-08-31)

73 success / 34 OOM / 1 failed. All figures are in-sample with `--validation-fraction 0.0`.
For the global in-sample caveat see {doc}`../reference/validation`.

| component | cell | n | rank | r² | MRE |
|---|---|---|---|---|---|
| latency | shard=2 | 15 | 3 | 0.9648 | 10.50 % |
| latency | shard=4 | 22 | 3 | 0.9737 | 8.94 % |
| latency | shard=8 | 22 | 3 | 0.9729 | 10.65 % |
| memory | shard=8 | 22 | 3 | 0.9810 | 4.27 % |
| memory | shard=4 | 22 | 3 | 0.8846 | 4.67 % |
| memory | shard=2 | 15 | 3 | −8.19 | 5.05 % |

Latency MRE of 9–11 % is higher than the Megatron model's 1–3 %. This reflects the wider
sweep (15–22 candidates per cell vs 6) and the more variable communication cost that
depends on the HSDP mesh topology, not just the sequence length.

### The shard=2 memory r² of −8.19 is the diagnostic working

A negative $r^2$ means the prediction is worse than the target mean — which would be a
failure if the regression had only seen successes. The shard=2 cell has 16 OOM inequality
constraints against 15 successful records. The inequalities lift the intercept approximately
$5.1 \times 10^9$ bytes above what the successes alone support, and the inflated intercept
then over-predicts the successful records by that margin.

This is **intentional and correct**. The successful records established the slope; the OOM
records established a floor that the successful records did not reach. The fitted model is
conservative: it will never predict that a workload fits on device when the OOM evidence says
it does not. The MRE of 5.05 % is evaluated against the successful records and describes the
quality of the slope, which is what actually varies in production queries.

An earlier version of the artifact claimed $r^2 = 1.0$ for this cell — it shipped
coefficients it had not computed metrics for, and the reported quality was nonsense. The
honest answer is $r^2 = -8.19$.

### Relationship between MRE and the OOM bump

The OOM bump raises `residual_constant` in the artifact. The $r^2$ and MRE in the accuracy
table are computed against the **final shipped constant**, not the regression-converged
constant, so the advertised quality correctly describes the coefficients the model uses
at prediction time.

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
    "per_step_overhead": 0.089,      // latency only; seconds
    "token_cost": 0.000601,          // latency only; seconds per token
    "microbatch_overhead": 0.014,    // latency only; seconds per microbatch
    "microbatch_slope": 608000000,   // memory only; bytes per microbatch
    "activation_slope": 0.0,         // memory only; near-zero under checkpointing
    "residual_constant": 3200000000, // memory only; bytes
    "parameter_count": 1630601216.0, // memory only
    "world_size": 8.0                // memory only
  },
  "feature_ranges": {
    "tokens": [512.0, 16384.0],            // latency
    "num_microbatches": [1.0, 4.0],        // latency and memory
    "activation_tokens": [512.0, 16384.0]  // memory
  },
  "diagnostics": {
    "memory_metric": "peak_allocated_bytes",   // memory only
    "reserved_over_allocated": 1.901,          // memory only; applied at predict time
    "oom_constraints": 16,                     // memory only
    "oom_intercept_increase": 5100000000.0,    // memory only; bytes
    "validation_is_in_sample": true
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
necessary when the simulator's batch-size / DP combination requires more microbatches than
were profiled. Fitting on `nmb ∈ {1, 2}` and predicting the 23 measured `nmb=4` candidates
gave a median absolute error of 7.7 %, on par with the model's own 9–11 % in-domain MRE,
so the upward extrapolation is defensible. Out-of-domain queries are never silent: the
exceedance is recorded in `TrainingCostEstimate.diagnostics["extrapolation_notes"]`.

Known gaps in coverage — the narrow `nmb` range [1, 4], `optimizer_offload=True` not
profiled, Ulysses sequence parallelism out of scope — are documented in
{doc}`../reference/limitations`.
