# Rollout cost models

Three fitted rollout models are provided: `roofline`, `psrl`, and `distserve`. All three are
trained on vLLM profiling data (see {doc}`../profiling/vllm`), stored as a `CostArtifact` JSON,
and loaded by the simulator without a code change. This page covers the formulas, the regression
command, and the accuracy of the fitted models. The CostArtifact schema in full is in
{doc}`../reference/schemas`; the in-sample caveat that applies to all three models is discussed
in {doc}`../reference/validation`.

## Fitting

All three models are fitted with a single command:

```bash
dryrun-regress rollout --model roofline \
    --input  outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
    --output outputs/cost/qwen3_8b/rollout_roofline.json
# --model accepts: roofline | psrl | distserve
# --validation-fraction 0.0  disables held-out scoring (reports in-sample only)
# --block-size 16            KV block size used in A_p denominator
```

The command reads the vLLM JSONL produced by `dryrun-profile vllm collect`, groups records by
parallelism topology, fits one entry per topology, and writes a `CostArtifact` JSON. A compact
example of the artifact format:

```javascript
{
  "kind": "rollout", "model_type": "roofline", "artifact_version": "1.0",
  "profile_digest": "sha256:9b7f45b7...",   // links this artifact to the measured data
  "entries": [{
    "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1, "world_size": 1},
    "coefficients": {"F": 0.00624718, "W": 0.00595378, "G": 0.00010058,
                     "A_p": 3.40461e-08, "A_d": 7.62594e-08, "b": 16.0},
    "feature_ranges": {"n_tokens": [1.0, 65536.0], "sum_l2": [0.0, 5.37e8],
                       "ctxsum": [0.0, 466089.0], "request_count": [1.0, 64.0]},
    "train_metrics": {"n_samples": 49208, "design_rank": 4,
                      "mre_percent": 5.25, "r_squared": 0.99989},
    "diagnostics": {"fit_method": "nonnegative-soft-l1"}
  }]
}
```

## Measured accuracy

Fitted on the Qwen3-8B / H20 / TP=1 vLLM JSONL (49,208 scheduler-iteration records,
2026-08-31). Both models were fitted with `--validation-fraction 0.0`, so the figures are
in-sample fit quality, not generalisation evidence.

| Model | MRE | r² | Notes |
|-------|-----|-----|-------|
| `roofline` | 5.25% | 0.9999 | in-sample, 49,208 points |
| `psrl` | 5.16% | 0.9999 | in-sample, 49,208 points |
| `distserve` | fitted, not scored | — | identifiability limit (see below) |

## The three models

### Unified roofline (`roofline`)

```text
latency = max(F, W + G*n_tokens + A_p * sum_l2 / b + A_d * ctxsum)
```

`F` is the low-load floor: a constant GPU overhead that dominates when the arithmetic workload is
small. The linear branch `W + G*n_tokens + ...` takes over once the step is large enough to be
compute- or memory-bound. The fit keeps all coefficients non-negative via soft-L1 regression,
then refits with MAD-based outlier rejection so that a small number of anomalous scheduler
iterations do not skew the floor estimate.

Features:
- `n_tokens`: total tokens in the scheduler iteration (prefill or decode).
- `sum_l2 / b` ($\sum L_i^2 / b$): the prefill attention term, where $L_i$ is the sequence
  length of request $i$ and $b$ is the KV block size. This term appears only in prefill and
  mixed steps.
- `ctxsum` ($\sum (L_i + k)$ over decode requests at KV depth $k$): the decode attention term.
  This term appears only in decode and mixed steps.

A mixed step simply includes both attention features — the model sees both `sum_l2` and
`ctxsum` simultaneously.

### PSRL fitted (`psrl`)

PSRL separates each phase into an attention term and a non-attention term, and fits them
independently:

```text
prefill = (p_attn_b + p_attn_k * sum_l2)
        + max(p_other_floor, p_other_b + p_other_k * n_tokens)

decode  = (d_attn_b + d_attn_k * ctxsum)
        + max(d_other_floor, d_other_b + d_other_k * n_requests)
```

For a mixed step the two attention terms are summed and the non-attention term takes the maximum
of the two phases' contributions. The maximum avoids charging the shared scheduler overhead twice:
the GPU runs one scheduler iteration, not two.

The per-phase floor (`p_other_floor`, `d_other_floor`) models the same physical effect as the
unified roofline's `F`, but independently for each phase. This can capture a different low-load
regime for decode (where the batch is small and dispatch cost dominates) versus prefill (where the
batch typically fills the arithmetic pipelines).

Decode uses `n_requests` as the non-attention feature rather than `n_tokens`, because decode
throughput is limited by the number of KV cache lookups, which scales with requests rather than
with output token count.

### DistServe (`distserve`)

DistServe maps the roofline form into architecture constants from DistServe Appendix A. Instead
of treating the coefficients as free parameters, it derives them from hidden size `h`,
intermediate size `m`, and five hardware constants:

```text
gemm = 4*h*h + 2*h*m

latency = max(F, (C3 + C4*gemm)
             + C1 * gemm * n_tokens
             + C2 * 3*h * sum_l2 / b
             + C5 * 3*h * ctxsum)
```

The intent is that `C1`–`C5` are hardware properties rather than per-model properties, enabling
cross-model transfer: calibrate on one model, predict on another. In practice a single-model fit
can only identify the combined overhead term `C3 + C4 * gemm`, not `C3` and `C4` independently
(they are linearly dependent on a single architecture). The artifact records the minimum-norm
canonical solution and includes a `c3_c4_identifiability` diagnostic. Cross-model joint
calibration that resolves the ambiguity is listed as a deferred stage in the design document.

DistServe does not produce a held-out accuracy score from a single-model fit because the
underlying coefficients are not uniquely identified — the fit result is the same as roofline with
renamed parameters. Score the roofline artifact instead if a number is needed.

## Sharing the feature vocabulary

All three models operate on the same four input features:

| Feature | Symbol | Phase |
|---------|--------|-------|
| Total tokens in step | `n_tokens` | all |
| Sum of squared sequence lengths | `sum_l2` | prefill, mixed |
| Sum of context depths | `ctxsum` | decode, mixed |
| Request count | `request_count` | decode (PSRL only) |

These features are extracted by the vLLM profiler adapter from the scheduler iteration log and
stored verbatim in each `ProfileRecord`. The feature ranges in the artifact record the observed
minimum and maximum, and the simulator refuses to query outside that range unless extrapolation
is explicitly permitted — see {doc}`../reference/schemas` for the extrapolation flag.

## `decode_coeffs()` and the simulator's closed-form advance

Every rollout model must implement `decode_coeffs(n_d, ctxsum) -> (F, alpha, beta)`, which
expresses one decode step as:

$$\tau(k) = \max\bigl(F,\ \alpha + \beta \cdot k\bigr)$$

where $k$ is the step index within a pure-decode segment and $n_d$ is the fixed batch size.
The simulator uses this to integrate a run of $k$ steps in closed form — summing the arithmetic
series exactly rather than iterating over every token — which is what makes long rollouts fast
to simulate.

For the unified roofline:

$$F = F,\quad \alpha = W + G \cdot n_d + A_d \cdot \text{ctxsum},\quad \beta = A_d \cdot n_d$$

For PSRL, $F = 0$ (no separate floor at the advance level; it is absorbed into $\alpha$), and
the attention slope drives $\beta = \text{d\_attn\_k} \cdot n_d$.

`ComposedRollout` does not use this interface because a composed step cost cannot be written
as `max(F, alpha + beta*k)` — the kernel sum is a more complex function of KV depth. It uses a
numerical `LatticeIntegrator` instead. See {doc}`../development/adapters-and-cost-models` for
the protocol that all cost models must satisfy.

## Choosing a model

All three models are close in accuracy on the Qwen3-8B / H20 sweep. The practical differences:

- `roofline` is a single five-parameter fit that works in all three phases simultaneously.
  It is the default choice and the fastest to fit.
- `psrl` fits phases independently, which means more parameters but also a better-conditioned
  per-phase regression when the dataset is large and the phase balance is uneven. Its 5.16% MRE
  is marginally better than `roofline`'s 5.25% on the Qwen3-8B dataset.
- `distserve` is the same as `roofline` numerically on a single-model fit. Its value emerges
  only when cross-architecture calibration is implemented. Until then, use `roofline` or `psrl`.

To use a fitted artifact in a simulation, set `rollout_cost.name` in the YAML and point it at the
JSON file. The simulator loads the entry matching the requested parallelism and calls `predict()`
or the closed-form advance; no regression code runs at simulation time.
