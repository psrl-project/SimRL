# Megatron training cost model (Hydraulis)

This page documents the Hydraulis cost model that is fitted from Megatron-Bridge profiles.
It covers the latency decomposition, the memory formula, how fitting works, and the accuracy
figures obtained on Qwen3-8B / 8x H20 / 2026-09-03.

For how the measurements are taken (warmup policy, dispersion reporting, sweep axes, output
layout) see {doc}`../profiling/megatron`. For general measurement rules and traps, see
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
      \times \bigl(c_{\text{token}} \times T_{\max}
                   + c_{q} \times S_{\max}
                   + c_{\text{mb}}\bigr)
$$

where:

- $c_{\text{overhead}}$ is a per-step fixed overhead, independent of the workload.
- $\text{nmb}$ is the number of microbatches on the critical DP rank.
- $\text{PP}$ is the pipeline-parallel degree.
- $\text{nmb} + \text{PP} - 1$ is the number of **pipeline slots** on the 1F1B critical path.
  A stage runs its $\text{nmb}$ microbatches and then drains a bubble of $\text{PP} - 1$ empty
  slots it cannot fill.
- $T_{\max}$ is the maximum token count across all microbatches on the critical rank.
- $S_{\max}$ is that microbatch's `sum_squared_tokens`, meaning $\sum_i s_i^2$ over the
  sequences packed into it. This is the attention term.
- $c_{\text{token}}$ is a per-token cost (seconds per token).
- $c_{q}$ is a per-squared-token cost, capturing work quadratic in sequence length.
- $c_{\text{mb}}$ is a per-microbatch overhead charged once per slot regardless of token count.

In code (`srl/cost/training/latency.py`, shared with the FSDP model):

```python
latency = coefficients["per_step_overhead"] + features.slots * (
    coefficients["token_cost"] * features.tokens
    + coefficients["token_quadratic_cost"] * features.sum_squared_tokens
    + coefficients["microbatch_overhead"]
)
```

The design row is therefore
`[1, slots*tokens, slots, slots*sum_squared_tokens]`, and a healthy fit reaches rank 4.

### Why this form and not the original Hydraulis form

The original Hydraulis Appendix C form writes the workload as $Q_{\text{eff}}$ and
$T_{\text{eff}}$, folding the pipeline bubble into the workload features. That created two
problems. The $Q_{\text{eff}}$ and $T_{\text{eff}}$ columns are collinear by construction at
uniform microbatch shape, and the quadratic $Q$ term cannot then be separately identified
from the linear $T$ term.

The restructured form separates four physically distinct costs. A training step is:

1. A **per-step overhead** paid once regardless of the batch.
2. A **per-token cost** proportional to the tokens in the largest microbatch.
3. A **per-squared-token cost** proportional to $\sum_i s_i^2$, which is attention.
4. A **per-microbatch overhead** paid once per slot, covering tensor setup, optimizer
   updates, and communication that do not grow with sequence length.

Measured per-microbatch times grow by factors of 1.0 to 1.9 when sequence length doubles,
where pure token-proportional work would give 2.0. The sub-linear growth is precisely the
per-microbatch overhead that a token-proportional term alone cannot express.

### The quadratic term is domain-dependent

This page previously said not to add a quadratic token term. That conclusion was correct for
the domain it was tested in and wrong for the domain the simulator actually needs.

Tested against sequence lengths capped at 2048, the linear form won on leave-one-out error
and the quadratic term looked like it was absorbing residual: in-sample error improved from
6.60% to 4.65% while leave-one-out worsened from 10.24% to 10.38%. Attention is only about
3.7% of a layer's FLOPs at 2048 tokens under full activation recomputation, so there was
genuinely almost no quadratic signal to find.

Extending the sweep to 8192 reverses it. Attention reaches 13.3% of a layer's FLOPs there,
and the same code on the same cells gives, by NNLS leave-one-out:

| cell | seq ≤ 2048, linear / quadratic | seq ≤ 8192, linear / quadratic |
|---|---|---|
| TP1/PP1 | 14.17% / 16.29% | **60.48% / 5.52%** |
| TP4/PP1 | 20.13% / 18.00% | **72.85% / 26.54%** |

The disagreement is about the domain, not about the model. A form fitted without the
quadratic term is off by a factor at the sequence lengths a real DAPO batch produces.

**The term must be $\sum_i s_i^2$, not $(\sum_i s_i)^2$.** Attention does not cross sequence
boundaries, so a microbatch holding several packed sequences costs the sum of their squares.
Squaring the packed total overcharges by the sequence count: eight 2048-token sequences give
3.36e7 against 2.68e8, an 8x error. `SequenceStats.sum_squared_tokens` already records the
correct quantity.

### DP treatment

DP communication is **not modelled**. DP still affects the local workload, since the planner
distributes trajectory lengths across DP ranks and uses the critical rank's microbatches as
the input. The fitted coefficients describe a single rank running the per-step work, and the
DP collective is assumed ideal. This is recorded in the artifact as
`"dp_latency": "ideal-local-workload-scaling"` and is a known limitation. See
{doc}`../reference/limitations`.

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
- $A_{\text{feature}}$ is the activation feature, an architecture-derived byte count rather
  than an opaque token count, so the fitted $c_{\text{act}}$ is a dimensionless correction
  near one.
- $c_{\text{residual}}$ is a fitted constant capturing allocator overhead and modelling
  errors not attributable to activations.
- OOM records impose the inequality $\text{peak} > \text{device capacity}$, which raises the
  intercept when the regression-fitted constant sits below the required floor.

The memory target is `peak_allocated_bytes`, not `peak_reserved_bytes`. Under full activation
recomputation, doubling the sequence length at TP1/PP1/DP8 moved `allocated` by +304 MiB and
`reserved` by minus 4 MiB, giving $r^2 = 0.196$ on `reserved`. A feature that moves the wrong
way when the input doubles is not a target: reserved bytes are an allocator caching artefact
rather than a footprint. See {doc}`../profiling/methodology` for the general lesson.

### Parameter inventory

`parameter_inventory` (in `srl/cost/training/inventory.py`) accounts for four distinct
resident pools, sized by the most-loaded pipeline stage:

| pool | size per rank | conditions |
|---|---|---|
| model parameters | $p_{\text{stage}} \times \text{param\_bytes}$ | always, TP and PP partition |
| gradient buffer | $p_{\text{local}} \times \text{grad\_bytes}$ | always, **not sharded by DP** |
| fp32 master parameters | $p_{\text{local}} \times 4$ bytes | when `param_dtype` is not fp32 |
| Adam moments (×2) | $p_{\text{local}} / \text{DP} \times \text{optstate\_bytes}$ | `distributed_optimizer=True` |

The gradient buffer is **not sharded by DP**. Megatron keeps a full-size fp32 buffer as the
reduce-scatter source, so every rank holds it whole. Measured at Qwen3-8B TP1/PP1/DP8 the
allocator reported `param_data` at 15.256 GiB ($= 2p$) and `grad_data` at 30.513 GiB
($= 4p$) with no data-parallel division, while the fp32 master shard was 3.814 GiB
($= 4p/\text{DP}$). The inventory formula is derived from those measurements rather than
assumed from the architecture.

The heaviest pipeline stage holds the embedding (first stage) or the output projection
(last stage) on top of its share of transformer layers. At PP=4 on Qwen3-8B, the uniform
assumption ($p/(\text{TP} \times \text{PP})$) predicted 15.26 GiB where the heaviest rank
held 19.32 GiB. Accounting for the embedding brings the prediction to 17.57 GiB.

### Activation feature

The activation feature is:

$$
A_{\text{feature}} = \min(\text{nmb}, \text{PP}) \times T_{\max} \times
    [(\text{num\_layers} + 1) \times \text{hidden} + \text{vocab}]
    \times \text{param\_element\_bytes}
$$

The $\min(\text{nmb}, \text{PP})$ cap reflects the 1F1B pipeline. At most PP microbatches are
in flight simultaneously, one per stage, so activation memory does not grow linearly in
$\text{nmb}$ beyond that.

### A note on memory accuracy

Memory MRE lands between 0.389% and 2.744% across the four cells. That does **not** mean the
activation term is well determined. It means `parameter_inventory` already accounts for nearly
all the variance and the residual fit is picking up a small correction. Even with the sweep
extended to 8192 tokens, the activation contribution stays modest relative to the static
inventory, so $c_{\text{act}}$ remains loosely constrained while the total prediction is
accurate. Nothing in the headline numbers signals that the coefficient is loose. Treat the
activation term as useful structure rather than as a calibrated activation predictor.

---

## Fitting

```bash
srl-regress training \
    --latency-model hydraulis \
    --memory-model hydraulis \
    --input  outputs/profile/qwen3_8b/megatron_bridge/profiles/megatron_bridge.jsonl \
    --output outputs/cost/qwen3_8b/training_hydraulis.json
```

The fitter partitions records by `(TP, PP, CP, schedule, recompute, optimizer, dtypes,
distributed_optimizer, parameter_count)` and fits each cell independently. Records for
different parallelism configurations are never mixed.

Each candidate contributes **one design row**, not one per retained step, because the step
times are reduced to their median first. A candidate whose steps are noisy would otherwise
outvote a clean candidate with equal parameter count. The median is unbiased against the
right-skew that stragglers produce.

Outlier rejection uses a robust non-negative least-squares fit with IRLS soft-L1 weighting.
The design matrix rank is checked: a rank below 4 for the latency model raises an error rather
than returning ill-conditioned coefficients, because an identifiability failure needs to be
fixed in the sweep rather than papered over.

### Cell key and lookup

At prediction time the call identifies the cell via an exact selector match on all discrete
fields: `(TP, PP, schedule, recompute, optimizer, dtypes, distributed_optimizer)`. DP is the
only analytical extension, scaling the local workload without changing the cell. An
unrecognised cell fails immediately. For extrapolation and out-of-domain semantics see
{doc}`../simulation/domain-gates`.

---

## Accuracy (measured, Qwen3-8B / 8x H20 / 2026-09-03)

40 candidates, 10 per cell, fitted with `--validation-fraction 0.2`. Every entry carries
`validation_is_in_sample: false`, so the held-out column is a real split.

| component | cell | train n | rank | train r² | train MRE | held-out n | held-out MRE |
|---|---|---|---|---|---|---|---|
| latency | TP1/PP1 | 8 | 4 | 0.99998 | 3.845 % | 2 | 2.322 % |
| latency | TP1/PP4 | 8 | 4 | 0.99145 | 5.485 % | 2 | 4.809 % |
| latency | TP4/PP1 | 8 | 4 | 0.99418 | 8.877 % | 2 | 23.282 % |
| latency | TP8/PP1 | 8 | 4 | 0.99711 | 10.937 % | 2 | 9.010 % |
| memory | TP1/PP1 | 8 | 2 | 0.95387 | 0.773 % | 2 | 2.475 % |
| memory | TP1/PP4 | 8 | 2 | 0.91117 | 2.744 % | 2 | 2.373 % |
| memory | TP4/PP1 | 8 | 2 | 0.96450 | 0.745 % | 2 | 1.297 % |
| memory | TP8/PP1 | 8 | 2 | 0.98982 | 0.389 % | 2 | 0.809 % |

All four latency designs reached rank 4, so `per_step_overhead`, `token_cost`,
`token_quadratic_cost`, and `microbatch_overhead` are separately identified in every cell.

### Do not quote the 23.28% held-out figure without its context

**A two-point held-out set is not a generalisation estimate.** With 10 candidates per cell,
`--validation-fraction 0.2` reserves exactly two, so a single unlucky point dominates the
reported MRE. TP4/PP1 reports 23.282% on those two points while its leave-one-out error over
the same cell is 12.46%.

Prefer leave-one-out on a grid this small. By that measure the four latency cells total
**9.39%**, ranging from 4.02% at TP1/PP1 to 13.67% at TP8/PP1.

### Why the coefficients are less certain than the predictions

The two token columns are highly collinear. Over the 40 profiled candidates,
`corr(slots*tokens, slots*sum_squared_tokens) = 0.963` and the design's condition number is
2.3e8. Five sequence-length points at `micro_batch_size=1` are not enough to separate the
linear term from the quadratic one, and NNLS resolves the ambiguity by pushing one toward
zero. TP8/PP1's `token_cost` collapsed to 1.97e-13, which is not a physical statement about
that topology.

Predictions stay accurate because the two terms compensate for each other over the fitted
range. Individual coefficients should not be read as measured per-token costs. Separating
them properly needs either a `micro_batch_size` dimension or more length points.

Remaining coverage gaps, including the real DAPO tail at 11133 tokens against a fitted domain
of 8192, are in {doc}`../reference/limitations`.

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
    "per_step_overhead": 1e-10,           // seconds
    "token_cost": 0.0005456,              // seconds per token
    "token_quadratic_cost": 1.030e-07,    // seconds per squared token
    "microbatch_overhead": 0.1616         // seconds per slot
  },
  "feature_ranges": {
    "tokens": [512.0, 8192.0],
    "pipeline_slots": [1.0, 4.0],         // [4.0, 7.0] in the PP=4 cell
    "micro_batch_size": [1.0, 1.0]
  },
  "train_metrics": {"r_squared": 0.99998, "mre_percent": 3.845, "design_rank": 4, "n_samples": 8},
  "validation_metrics": {"r_squared": 0.99350, "mre_percent": 2.322, "n_samples": 2},
  "diagnostics": {
    "formula": "per_step_overhead + (num_microbatches + PP - 1) * (token_cost * max_microbatch_tokens + token_quadratic_cost * max_microbatch_sum_squared_tokens + microbatch_overhead)",
    "step_aggregation": "median-per-candidate",
    "dp_latency": "ideal-local-workload-scaling",
    "validation_fraction": 0.2,
    "validation_is_in_sample": false
  }
}
```

The values above are the TP1/PP1 latency entry from the shipped artifact. Read
`per_step_overhead: 1e-10` as zero: NNLS is constrained non-negative and pinned this term to
its floor, meaning the fit found no fixed per-step cost distinguishable from the
per-microbatch one.

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

The artifact supports only CP=1 and the 1F1B schedule. VPP, interleaved 1F1B, and non-uniform
pipeline partitions are deferred. See {doc}`../reference/limitations`.
