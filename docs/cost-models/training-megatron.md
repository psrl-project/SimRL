# Megatron training cost model (Hydraulis)

This page documents the Hydraulis V1 cost model that is fitted from Megatron-Bridge profiles.
It covers the latency decomposition, the memory formula, how fitting works, and the accuracy
figures obtained on Qwen3-8B / 8x H20 / 2026-08-31.

For how the measurements are taken — warmup policy, dispersion reporting, sweep axes, output
layout — see {doc}`../profiling/megatron`. For general measurement rules and traps, see
{doc}`../profiling/methodology`. The `CostArtifact` schema is documented in
{doc}`../reference/schemas`.

---

## Latency model

The Hydraulis form follows Appendix C of the Hydraulis paper (https://arxiv.org/abs/2412.07894)
and is the **only authoritative statement** of the latency formula in this repository.

For a data-parallel rank, the predicted step time is:

$$
\text{step} = c_{\text{overhead}}
    + (\text{nmb} + \text{PP} - 1)
      \times \bigl(c_{\text{token}} \times T_{\max} + c_{\text{mb}}\bigr)
$$

where:

- $c_{\text{overhead}}$ is a per-step fixed overhead, independent of the workload.
- $\text{nmb}$ is the number of microbatches on the critical DP rank.
- $\text{PP}$ is the pipeline-parallel degree.
- $\text{nmb} + \text{PP} - 1$ is the number of **pipeline slots** on the 1F1B critical path:
  a stage runs its $\text{nmb}$ microbatches and then drains a bubble of $\text{PP} - 1$ empty
  slots it cannot fill.
- $T_{\max}$ is the maximum token count across all microbatches on the critical rank.
- $c_{\text{token}}$ is a per-token cost (seconds per token).
- $c_{\text{mb}}$ is a per-microbatch overhead charged once per slot regardless of token count.

In code (`dryrun/cost/training/hydraulis.py`):

```python
latency = coefficients["per_step_overhead"] + features.slots * (
    coefficients["token_cost"] * features.tokens
    + coefficients["microbatch_overhead"]
)
```

### Why this form and not the original Hydraulis form

The original Hydraulis Appendix C form writes the workload as $Q_{\text{eff}}$ and
$T_{\text{eff}}$, folding the pipeline bubble into the workload features. That created two
problems: the $Q_{\text{eff}}$ and $T_{\text{eff}}$ columns are collinear by construction
at uniform microbatch shape, and the quadratic $Q$ term cannot be separately identified
from the linear $T$ term in that case.

The restructured form separates three physically distinct costs. A training step is:

1. A **per-step overhead** paid once regardless of the batch.
2. A **per-token cost** proportional to the tokens in the largest microbatch.
3. A **per-microbatch overhead** paid once per slot, covering all tensor setup, optimizer
   updates, and communication that do not grow linearly with sequence length.

Measured per-microbatch times grow by factors of 1.0 to 1.9 when sequence length doubles
(pure token-proportional work would give 2.0). The sub-linear growth is precisely the
per-microbatch overhead that a token-proportional term alone cannot express.

### Why no quadratic token term

A $c_q \times T_{\max}^2$ term was tried. It improved in-sample MRE from 6.60 % to 4.65 %
at TP1/PP1 but worsened leave-one-out MRE from 10.24 % to 10.38 % — the signature of a
term absorbing residual rather than a mechanism. The implied coefficient corresponded to
roughly 18 000 % of the H20's bf16 throughput. Direct kernel measurements show that attention
accounts for only 2–3 % of a step under full activation recomputation. Do not re-add this term
without sequence lengths long enough for attention to dominate, and check the implied
utilization. See {doc}`../profiling/methodology` for the full table.

### DP treatment

DP communication is **not modelled** in V1. DP still affects the local workload: the planner
distributes trajectory lengths across DP ranks and uses the critical rank's microbatches as the
input. The fitted coefficients are for a single rank running the per-step work; the DP
collective is assumed ideal. This is recorded in the artifact as
`"dp_communication": "ignored"` and is a known limitation — see {doc}`../reference/limitations`.

---

## Memory model

Peak memory on the heaviest pipeline stage is:

$$
\text{peak\_reserved} =
    \text{parameter\_inventory}(\text{TP, PP, DP, dtypes, distributed\_optimizer})
    + c_{\text{act}} \times A_{\text{feature}}
    + c_{\text{residual}}
$$

where:

- `parameter_inventory` is an **analytic formula** for the static state, described below.
- $A_{\text{feature}}$ is the activation feature — an architecture-derived byte count, not an
  opaque token count — so the fitted $c_{\text{act}}$ is a dimensionless correction near one.
- $c_{\text{residual}}$ is a fitted constant that captures allocator overhead and modelling
  errors not attributable to activations.
- OOM records impose the inequality $\text{peak} > \text{device capacity}$, which is used to
  raise the intercept when the regression-fitted constant sits below the required floor.

The memory target is `peak_allocated_bytes`, not `peak_reserved_bytes`. Under full activation
recomputation, doubling the sequence length at TP1/PP1/DP8 moved `allocated` by +304 MiB and
`reserved` by −4 MiB, giving $r^2 = 0.196$ on `reserved`. Fitting `reserved` is wrong because
reserved bytes are an allocator caching artefact rather than a footprint. See
{doc}`../profiling/methodology` for the general reserved-vs-allocated lesson.

### Parameter inventory

`parameter_inventory` (in `dryrun/cost/training/inventory.py`) accounts for four distinct
resident pools, sized by the most-loaded pipeline stage:

| pool | size per rank | conditions |
|---|---|---|
| model parameters | $p_{\text{stage}} \times \text{param\_bytes}$ | always; TP and PP partition |
| gradient buffer | $p_{\text{local}} \times \text{grad\_bytes}$ | always; **not sharded by DP** |
| fp32 master parameters | $p_{\text{local}} \times 4$ bytes | when `param_dtype` is not fp32 |
| Adam moments (×2) | $p_{\text{local}} / \text{DP} \times \text{optstate\_bytes}$ | `distributed_optimizer=True` |

The gradient buffer is **not sharded by DP**. Megatron keeps a full-size fp32 buffer as the
reduce-scatter source, so every rank holds it whole. Measured at Qwen3-8B TP1/PP1/DP8 the
allocator reported `param_data` at 15.256 GiB ($= 2p$) and `grad_data` at 30.513 GiB
($= 4p$) with no data-parallel division — while the fp32 master shard was
3.814 GiB ($= 4p/\text{DP}$). The inventory formula is derived from those measurements, not
assumed from the architecture.

The heaviest pipeline stage holds the embedding (first stage) or the output projection
(last stage) on top of its share of transformer layers. At PP=4 on Qwen3-8B, the uniform
assumption ($p/(\text{TP} \times \text{PP})$) predicted 15.26 GiB where the heaviest rank
held 19.32 GiB; accounting for the embedding brings the prediction to 17.57 GiB.

### Activation feature

The activation feature is:

$$
A_{\text{feature}} = \min(\text{nmb}, \text{PP}) \times T_{\max} \times
    [(\text{num\_layers} + 1) \times \text{hidden} + \text{vocab}]
    \times \text{param\_element\_bytes}
$$

The $\min(\text{nmb}, \text{PP})$ cap reflects the 1F1B pipeline: at most PP microbatches
are in-flight simultaneously (one per stage), so activation memory does not grow linearly
in $\text{nmb}$ beyond that.

### A note on memory accuracy

The memory MRE of ~0.00 % (all four cells at rank 2, $r^2 \approx 1.0000$) does **not** mean
the activation term is well determined. It means the `parameter_inventory` term already
accounts for nearly all the variance, and the residual fit is picking up a small constant.
At the measured Qwen3-8B sequence lengths (512–2048 tokens) the activation contribution is
small relative to the static inventory, so the $c_{\text{act}}$ coefficient is poorly
constrained even when the total prediction is accurate. Extending the sweep to longer
sequences would tighten the activation coefficient — but the fit's headline numbers give no
indication that the coefficient is loose. Treat the activation term as useful structure, not
as a calibrated activation predictor.

---

## Fitting

```bash
dryrun-regress training \
    --latency-model hydraulis \
    --memory-model hydraulis \
    --input  outputs/profile/qwen3_8b/megatron_bridge/profiles/megatron_bridge.jsonl \
    --output outputs/cost/qwen3_8b/training_hydraulis.json
```

The fitter partitions records by `(TP, PP, CP, schedule, recompute, optimizer, dtypes,
distributed_optimizer, parameter_count)` and fits each cell independently. Records for
different parallelism configurations are never mixed.

Each candidate contributes **one design row**, not one per retained step — the step times
are reduced to their median first. A candidate whose steps are noisy would otherwise
outvote a clean candidate with equal parameter count. The median is unbiased against the
right-skew that stragglers produce.

Outlier rejection uses a robust non-negative least-squares fit with IRLS soft-L1 weighting.
The design matrix rank is checked: a rank below 3 for the latency model raises an error rather
than returning ill-conditioned coefficients, because the identifiability failure needs to be
fixed in the sweep, not papered over.

### Cell key and lookup

At prediction time the call identifies the cell via an exact selector match on all discrete
fields: `(TP, PP, schedule, recompute, optimizer, dtypes, distributed_optimizer)`. DP is the
only analytical extension — it scales the local workload but does not change the cell. An
unrecognised cell fails immediately. For extrapolation and out-of-domain semantics see
{doc}`../simulation/domain-gates`.

---

## Accuracy (measured, in-sample, Qwen3-8B / 8x H20 / 2026-08-31)

24 candidates, 6 per cell. All figures are in-sample with `--validation-fraction 0.0`.
For the global in-sample caveat see {doc}`../reference/validation`.

| component | cell | n | rank | r² | MRE |
|---|---|---|---|---|---|
| latency | TP1/PP1 | 6 | 3 | 0.9994 | 2.61 % |
| latency | TP1/PP4 | 6 | 3 | 0.9986 | 2.28 % |
| latency | TP4/PP1 | 6 | 3 | 0.9966 | 2.31 % |
| latency | TP8/PP1 | 6 | 3 | 0.9998 | 1.12 % |
| memory | all four | 2 | ~1.0000 | ~0.00 % | (see note above) |

The latency MRE of 1–3 % across four independently fitted cells is genuine model signal.
All four latency designs reached rank 3, meaning `per_step_overhead`, `token_cost`, and
`microbatch_overhead` are separately identified in every cell.

These are 6-sample in-sample numbers. The model has not been evaluated on held-out
configurations. Known gaps in coverage — in particular the narrow sequence range
[512, 2048] versus real DAPO batches at ~9470 tokens — are documented in
{doc}`../reference/limitations`.

---

## Artifact structure

The artifact (a `CostArtifact` JSON) is produced at `outputs/cost/qwen3_8b/training_hydraulis.json`
by the command above. It contains one entry per `(component, cell)` pair. Each entry carries:

```javascript
{
  "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1},
  "selector": {
    "component": "latency",     // or "memory"
    "schedule": "1f1b",
    "recompute": "full",
    "optimizer": "adam",
    "parameter_dtype": "bf16",
    "gradient_dtype": "fp32",
    "optimizer_state_dtype": "fp32",
    "distributed_optimizer": true
  },
  "coefficients": {
    "per_step_overhead": 0.123,   // seconds
    "token_cost": 0.00048,        // seconds per token
    "microbatch_overhead": 0.034  // seconds per slot
  },
  "feature_ranges": {
    "tokens": [512.0, 2048.0],
    "pipeline_slots": [2.0, 8.0],
    "micro_batch_size": [1.0, 1.0]
  },
  "train_metrics": {"r_squared": 0.9994, "mre_percent": 2.61, "design_rank": 3},
  "diagnostics": {
    "formula": "per_step_overhead + (nmb + PP - 1)*(token_cost*tokens + microbatch_overhead)",
    "dp_latency": "ideal-local-workload-scaling",
    "validation_is_in_sample": true
  }
}
```

The `feature_ranges` become hard out-of-domain gates at prediction time unless the
corresponding name is listed in `training_cost.extrapolate`. See
{doc}`../simulation/domain-gates`.

---

## Using the artifact

In the Hydra simulation config:

```yaml
training_cost: hydraulis
training_cost:
  artifact_path: outputs/cost/qwen3_8b/training_hydraulis.json
  parallelism:
    tp: 1
    pp: 1
    dp: 8
    cp: 1
  schedule: 1f1b
  activation_recompute: full
  distributed_optimizer: true
```

Or from the command line:

```bash
TRAINING_BACKEND=hydraulis \
TRAINING_ARTIFACT=outputs/cost/qwen3_8b/training_hydraulis.json \
bash examples/simulate/simulate.sh
```

The artifact supports only CP=1 and 1F1B schedule. VPP, interleaved 1F1B, and non-uniform
pipeline partitions are deferred; see {doc}`../reference/limitations`.
