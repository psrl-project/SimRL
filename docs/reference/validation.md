# Validation status

This page collects every measured accuracy figure across all cost models, with sample counts,
whether the set was held out, and what the numbers mean.

The short version: **the two training models carry genuine held-out splits, the three rollout
regressions do not, and `ComposedRollout` is scored end to end against a real baseline rather
than by a regression metric.** Where a held-out set is only two points, that is stated,
because a two-point split is closer to noise than to evidence.

---

## Rollout cost models

Measured on Qwen3-8B / H20 / TP=1 from 49,208 samples in
`outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl`.

| Model | Design rank | $r^2$ | MRE | Held out? |
|---|---|---|---|---|
| `roofline` | 4 | 0.99989 | 5.252% | No (`n_samples=0` in `validation_metrics`) |
| `psrl` | 3 | 0.99992 | 5.164% | No (`n_samples=0` in `validation_metrics`) |
| `distserve` | 4 | 0.99989 | 5.252% | No |

The `distserve` figures are byte-identical to `roofline` because DistServe re-expresses the
same roofline fit through architecture constants (`h`, `m`, `C1` to `C5`) rather than fitting
independently. Two identical rows in a summary table look like a copy-paste error. They are
not. The mechanism differs while the numerical fit is the same.

All three rollout artifacts record `validation_metrics.n_samples == 0`, which is how the code
encodes that no held-out records were reserved. These are in-sample only.

---

## `ComposedRollout`

`ComposedRollout` has no regression step and therefore no `CostArtifact`. Its accuracy is
measured end to end by `dryrun-profile kernels validate` against a real vLLM profiling
baseline. See {doc}`../cost-models/composed-rollout` for the segment integrator, the
elementwise design decision, and the known limitations.

Measured 2026-08-25, H20 / Qwen3-8B / TP=1, `pchip` surface:

| Phase | Buckets | MRE |
|---|---|---|
| prefill | 29 | 3.72% |
| decode | 216 | 4.84% |
| mixed | 5 | 0.48% |

The mixed figure rests on six raw iterations and is indicative rather than a validated claim.

Measured 2026-09-03, H20 / Qwen3-8B / **TP=2**, `pchip` surface, against a serial baseline
collected on the same grid (`outputs/profile/qwen3_8b/vllm_tp2_serial/`, 49,211 iterations,
`max_parallel: 1`):

| Phase | Buckets | MRE | Status |
|---|---|---|---|
| **prefill, whole grid** | 29 | **3.60%** | validated |
| **decode, `requests >= 5`** | 137 | **2.96%** | validated |
| mixed | 7 | 3.64% | indicative |
| decode, whole grid | 204 | 9.51% | includes the small-batch gap |

Prefill needs no exclusion. Its six `requests = 1` buckets originally read 12.50% because they
were collected with `num-prompts: 1`, so each recorded iteration was a single cold request
carrying about 78 ms of server-side work that no steady-state step pays. Re-measured with
`num-prompts: 8` at the same shape, `requests = 1` prefill went from 46.98% to 3.98% MRE with
the prediction unchanged, taking the whole grid to 3.60%, comparable to the 3.72% at TP=1.

Decode's remaining gap is a *different* problem and a real model error. At `requests 1-4` the
model over-predicts, with a median ratio of 1.31. At one request, 8.4x more KV moves the
measured step only 1.09x, so per-step overhead dominates and summing kernels over-counts.
Re-measuring does not help. This needs a model term. See {doc}`limitations`.

An earlier parallel baseline (`vllm_tp2/`, `max_parallel: auto`) scored decode at 16.39%
because one of its four concurrent servers degraded mid-run. The serial re-measurement
confirmed it: `requests = 16` moved from median ratio 0.513 to 0.993 while every other batch
stayed put. See {doc}`limitations`, {doc}`../profiling/tensor-parallel`, and
`examples/profile/kernel/TP2_VALIDATION.md`.

This is the first end-to-end score of the communication path. At TP=1 `comm_queries` returns
empty, so the 3.72% / 4.84% / 0.48% figures above never exercised a collective. The collective
contributes 13.2% of a TP=2 decode step, and pricing it as zero degrades median decode error
from 3.07% to 10.70%.

For the per-kernel single-surface holdout numbers, see {doc}`../profiling/kernels`, which owns
that table.

---

## Megatron / Hydraulis training cost model

Measured 2026-09-03 on Qwen3-8B / 8x H20 from
`outputs/profile/qwen3_8b/megatron_bridge/profiles/megatron_bridge.jsonl`. 40 candidates, 10
per cell, all `success`. Fitted with `--validation-fraction 0.2`, so every entry carries
`validation_is_in_sample: false`.

### Latency (4 cells)

| Cell | Train n | Train $r^2$ | Train MRE | Held-out n | Held-out MRE |
|---|---|---|---|---|---|
| TP=1 / PP=1 | 8 | 0.99998 | 3.845% | 2 | 2.322% |
| TP=1 / PP=4 | 8 | 0.99145 | 5.485% | 2 | 4.809% |
| TP=4 / PP=1 | 8 | 0.99418 | 8.877% | 2 | 23.282% |
| TP=8 / PP=1 | 8 | 0.99711 | 10.937% | 2 | 9.010% |

### Memory (4 cells)

| Cell | Train n | Train $r^2$ | Train MRE | Held-out n | Held-out MRE |
|---|---|---|---|---|---|
| TP=1 / PP=1 | 8 | 0.95387 | 0.773% | 2 | 2.475% |
| TP=1 / PP=4 | 8 | 0.91117 | 2.744% | 2 | 2.373% |
| TP=4 / PP=1 | 8 | 0.96450 | 0.745% | 2 | 1.297% |
| TP=8 / PP=1 | 8 | 0.98982 | 0.389% | 2 | 0.809% |

**Do not read the 23.282% as a generalisation estimate.** A 20% split of 10 candidates
reserves two points, so a single unlucky point sets the number. That cell's leave-one-out error
is 12.46%. Across all four latency cells, leave-one-out totals **9.39%**, from 4.02% at
TP1/PP1 to 13.67% at TP8/PP1. Prefer those figures.

Memory MRE below 3% does not mean the activation term is well determined. The analytic
`parameter_inventory` already accounts for nearly all the variance, so the residual fit is
picking up a small correction and $c_{\text{act}}$ stays loosely constrained. See
{doc}`../cost-models/training-megatron`.

---

## FSDP2/HSDP training cost model

Measured on Qwen3-8B / 8x H20 from `outputs/profile/qwen3_8b/fsdp/profiles/fsdp.jsonl`.
108 candidates: 73 success, 34 OOM, 1 failed. Fitted with `--validation-fraction 0.2`, and
every entry carries `validation_is_in_sample: false`.

### Latency

| Cell | Train n | Train $r^2$ | Train MRE | Held-out n | Held-out MRE |
|---|---|---|---|---|---|
| shard=2 | 15 | 0.99877 | 4.236% | 4 | 6.519% |
| shard=4 | 22 | 0.99999 | 0.777% | 5 | 0.820% |
| shard=8 | 22 | 0.99999 | 0.648% | 5 | 1.394% |

### Memory

| Cell | Train n | Train $r^2$ | Train MRE | Held-out n | Held-out MRE |
|---|---|---|---|---|---|
| shard=2 | 15 | −8.189 | 5.053% | 4 | 4.897% |
| shard=4 | 22 | 0.885 | 4.673% | 5 | 5.981% |
| shard=8 | 22 | 0.981 | 4.268% | 5 | 5.388% |

This is the best-evidenced model on the project. It has the largest per-cell sample counts,
four or five held-out points rather than two, and held-out error that tracks in-sample error
closely, which is the practical definition of generalisation.

The shard=2 memory $r^2$ of $-8.19$ is **not** a fitting failure. It is the 16 OOM inequality
constraints raising the intercept above what the 15 successful records alone support, which
makes the model deliberately conservative about predicting that a workload fits. See
{doc}`../cost-models/training-fsdp`.

---

## What these numbers mean for a simulation

Three distinctions matter more than the numbers themselves.

**In-sample against held-out.** An in-sample figure tells you how well the model fits data it
was trained on, not whether it generalises. Treat the three rollout MREs as lower bounds on
run-time error.

**Held-out against leave-one-out.** On a grid of 10 candidates per cell, a 20% split leaves
two points and reports something dominated by chance. The Megatron TP4/PP1 discrepancy (23.28%
held-out against 12.46% leave-one-out) is the concrete case. Sample counts are given in every
table above so this is visible rather than implied.

**In-domain against extrapolated.** Every figure here describes queries inside the fitted
feature ranges. A real DAPO batch exceeds the Megatron `tokens` domain of 8192, reaching 11133
at the tail, and no accuracy figure on this page covers that region. The simulator refuses
such queries unless extrapolation is requested by feature name, and records each one in the
telemetry. See {doc}`../simulation/domain-gates` and {doc}`limitations`.
