# Rollout cost models

Three fitted rollout models are provided: `roofline`, `psrl` and `distserve`. All three are fitted
from vLLM profiling data ({doc}`../profiling/vllm`), stored as a `CostArtifact` JSON, and loaded
by the simulator without a code change. This page covers their formulas, the regression command,
and how each model's pure-decode terms are derived.

{doc}`composed-rollout` covers the fourth rollout model, which sums measured kernels instead of
fitting.

## Fitting

```bash
srl-regress rollout --model roofline \
    --input  outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
    --output outputs/cost/qwen3_8b/rollout_roofline.json
# --model accepts: roofline | psrl | distserve
# --validation-fraction 0.0  default; set a fraction to reserve whole runs for validation
# --block-size 16            KV block size used in the prefill attention term
# --hidden-size / --intermediate-size   distserve only; default to the checkpoint's values
```

The command reads the JSONL, keeps successful rollout records, groups them by `parallelism`,
fits one entry per topology, and writes the artifact. A run split, when requested, is by
`run_id`, so one benchmark launch cannot straddle train and validation. When the records all come
from the vLLM backend it also writes diagnostic plots under `{output stem}_plots/`.

Every entry records `train_metrics`, `validation_metrics`, `feature_ranges` (the measured
minimum and maximum of `n_tokens`, `sum_l2`, `ctxsum` and `request_count`) and `diagnostics`
naming the fit method. The feature ranges are what the simulator gates queries on.

All three fits use non-negative least squares with a soft-L1 loss, then one MAD-based refit that
drops rows beyond 4.5 robust standard deviations when enough rows remain. Coefficients are
constrained non-negative, so a coefficient pinned at zero is a statement about identifiability,
not a measurement.

## Shared feature vocabulary

| feature | symbol | phase |
|---|---|---|
| tokens in the step | `n_tokens` | all |
| sum of squared prefill extents | `sum_l2` | prefill, mixed |
| sum of decode context depths | `ctxsum` | decode, mixed |
| requests in the step | `request_count` | decode (PSRL), and `ComposedRollout`'s logits shape |

These are filled by the vLLM adapter from the scheduler iteration log, as described on
{doc}`../profiling/vllm`. A pure prefill step has `ctxsum = 0`, a pure decode step has
`sum_l2 = 0`, and a mixed step has both. The models that distinguish phases zero the term that
does not belong.

## Unified roofline (`roofline`)

```text
latency = max(F, W + G * n_tokens + A_p * sum_l2 / b + A_d * ctxsum)
```

- `F` is a low-load floor: a fixed GPU overhead that dominates when the arithmetic workload is
  small.
- `W` is the linear branch's intercept, `G` the per-token cost, `A_p` the prefill-attention
  coefficient over `sum_l2 / b`, and `A_d` the decode-attention coefficient over `ctxsum`.
- `b` is the KV block size, recorded in the coefficients and in the entry selector.

The floor enters as a `max`, not as an additive term, so it cannot double-count in the
compute-bound regime. The fit is nonlinear only through the `max`. It initialises from a scan over
candidate floors, each solved as a non-negative linear least-squares problem, and keeps the
combination with the lowest squared error before the robust refit.

`LinearLPS` subclasses the roofline with `F = 1/r1`, `G = 1/(r1*K)` and zero attention terms. It
is a limited-processor-sharing model used to validate the simulator's closed-form advance against
a known answer, not a fitted model.

## PSRL fitted (`psrl`)

PSRL separates each phase into an attention term and a non-attention term and fits the phases
independently:

```text
prefill = (p_attn_b + p_attn_k * sum_l2) + max(p_other_floor, p_other_b + p_other_k * n_tokens)
decode  = (d_attn_b + d_attn_k * ctxsum) + max(d_other_floor, d_other_b + d_other_k * requests)
```

For a mixed step the two attention terms are summed and the non-attention term takes the
**maximum** of the two phases' contributions, since one scheduler iteration runs once. Decode's
non-attention feature is the request count rather than the token count, because decode throughput
is limited by the number of KV-cache lookups.

The fit needs at least five prefill and five decode observations per topology and raises
otherwise. Mixed observations are not fitted directly: they are scored by the mixed rule above.
Each entry records the prefill and decode design ranks separately, and its `train_metrics` rank
is the smaller of the two.

`psrl` has no `artifact_path` default. It must be fitted before use, and
`build_rollout_cost` raises if the path is missing.

## DistServe (`distserve`)

DistServe maps the roofline form onto architecture constants from DistServe Appendix A. Rather
than free per-model coefficients it derives them from hidden size `h`, intermediate size `m` and
five effective hardware constants:

```text
gemm = 4*h*h + 2*h*m

W   = C3 + C4 * gemm
G   = C1 * gemm
A_p = C2 * 3*h
A_d = C5 * 3*h

latency = max(F, W + G*n_tokens + A_p*sum_l2/b + A_d*ctxsum)
```

`fit` first performs the roofline fit and then inverts it into `C1..C5`. A single-model fit can
identify the combined overhead `C3 + C4 * gemm` but not `C3` and `C4` separately, because the two
columns are linearly dependent for one architecture. The artifact records the minimum-norm
canonical solution (`C3 = W / (1 + gemm^2)`, `C4 = W * gemm / (1 + gemm^2)`), the effective
overhead, and a `c3_c4_identifiability: single-model-canonical-minimum-norm` diagnostic. Its
`from_artifact` reads `h`, `m` and the five constants back.

Because the constants are not separately identified from one model, a single-model DistServe
artifact predicts what the corresponding roofline artifact predicts. Its value would come from
calibrating several architectures jointly, which is not implemented.

## The pure-decode closed form

Every rollout model implements

```text
decode_coeffs(n_d, ctxsum) -> (F, alpha, beta)
```

which expresses one decode step at a fixed batch of `n_d` requests with total context `ctxsum`
as

```text
tau(k) = max(F, alpha + beta * k)
```

where `k` counts steps within a pure-decode segment. `srl/cost/rollout/advance.py` sums this
exactly. The cumulative cost splits at the crossing step `k0 = ceil((F - alpha) / beta)`:

```text
Delta(k) = k0 * F + (k - k0) * (alpha + beta * k0) + beta * (k - k0) * (k - k0 - 1) / 2
```

`crossing_step`, `elapsed` and `steps_within` all implement this. `steps_within` inverts the sum
for a time budget. Every branch corrects its integer division by stepping forward while the
result fits and back if it overshoots, because `budget // tau` can land one step short of an exact
quotient. Functions are exact under `Fraction` inputs.

The coefficients per model:

| model | `F` | `alpha` | `beta` |
|---|---|---|---|
| `roofline` | `F` | `W + G*n_d + A_d*ctxsum` | `A_d*n_d` |
| `distserve` | `F` | `W + G*n_d + A_d*ctxsum` | `A_d*n_d` |
| `psrl` | `0` | `d_attn_b + d_attn_k*ctxsum + max(d_other_floor, d_other_b + d_other_k*n_d)` | `d_attn_k*n_d` |
| `composed` | `0` | step cost at the segment start | secant slope over one stride |

PSRL returns `F = 0` because its floor is already inside the non-attention term. Returning both
would count it twice.

`saturated(n_tokens, sum_l2, ctxsum, n_reqs)` reports whether the step's non-floor branch is
active. For the roofline and DistServe it compares the raw linear expression against `F`. PSRL
compares the non-attention branch against its per-phase floor, and for a mixed step requires both
phases to be saturated. The simulator reads it to know whether a step is above the low-load
regime.

`ComposedRollout` does not use `advance.py`, because a kernel sum is not `max(F, alpha + beta*k)`.
Its `decode_coeffs` returns a floor of zero and a secant slope, which keeps the interface usable,
but its segments are integrated numerically on a lattice ({doc}`composed-rollout`).

## Choosing a model

- `roofline` is one five-parameter fit that covers all three phases at once. It is the default
  rollout model and the one with no artifact requirement.
- `psrl` fits the phases separately, which gives more parameters and a better-conditioned
  per-phase regression when the data are large and the phase balance is uneven. It needs at least
  five observations per pure phase.
- `distserve` is numerically the roofline on a single-model fit. Use it when the artifact's
  architecture-constant form is what a downstream tool expects.
- `composed` needs no fit at all and generalises across models, GPUs and degrees without a
  server sweep, provided the kernel tables exist for that GPU and that TP degree.

To use a fitted artifact, point `rollout_cost.artifact_path` at it. The simulator loads the entry
matching the requested `parallelism` and calls `predict()` or the closed-form advance. No
regression code runs at simulation time.
