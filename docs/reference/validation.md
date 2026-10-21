# Validation

This page does two things. It describes how accuracy is measured and gated in the code, and it
lists the figures that can be reproduced today from the pinned tests and the local fit artifacts.
Anything that cannot be reproduced from those is not quoted here.

---

## What kind of evidence each number is

In-sample
: The error of a fit on the records it was fitted to. It shows the functional form can represent
  the data and says little about unseen queries. An artifact entry marks it with
  `validation_metrics.n_samples == 0`, or `validation_is_in_sample: true` for the training models.

Held-out split
: `srl-regress --validation-fraction F` reserves a fraction of the records before fitting.
  Records are split by run, not by row, so repeats of one candidate stay on the same side. The
  reserved records populate `validation_metrics`. Small cells reserve very few points, so
  report the count next to every held-out figure.

Leave-one-out holdout
: `srl-profile kernels holdout` removes one measured kernel shape at a time, rebuilds the
  utility surface from the rest and predicts the removed shape. It scores interpolation of a
  single kernel, not a whole engine iteration.

End to end
: `srl-profile kernels validate` sums the kernel tables into `ComposedRollout` predictions and
  scores them against iterations a real vLLM server recorded. Neither the tables nor the dispatch
  floors were fitted to that baseline, so this is the closest thing to an independent test.

Every figure below says which of these it is, the sample count, and where it comes from.

---

## The gates in the code

### Kernel holdout (`srl/cost/kernel/holdout.py`)

- A shape is scored only if it is bracketed: after removing it, the remaining points still span it
  on every axis. A shape outside the span would be a frontier hold, which is designed to be
  conservative and would be scored for the wrong reason.
- A slice needs at least four interpolable points, and a row with a zero coordinate cannot be
  placed in log2 space, so it is excluded from scoring and kept only in the utility range.
- `--fraction` defaults to 1.0, which scores every bracketed shape. A smaller fraction samples a
  subset and the CLI warns that the result understates the error.
- `HoldoutResult.passes(gate)` is true only if at least one point was scored and the mean relative
  error is below the gate. The CLI default gate is 5.0 percent, and the command exits non-zero
  if any `(op, backend)` row fails. A row with no scorable points is reported as a coverage
  problem and not as an accuracy one.
- The dispatch floor used for each op comes from `floors.json`, which `fit-floor` chooses by
  minimising this same holdout error over a candidate list.

### Composed rollout validation (`srl/cost/kernel/validate.py`)

- A baseline holds tens of thousands of near-repeating iterations, so they are bucketed by phase,
  request count, mean context and token count. Context is rounded to a linear multiple (256),
  never to a power of two, and each bucket's latency is the median of its iterations.
- Phases are scored separately because their regimes differ. A single blended figure would hide
  a systematic error in one of them.
- The default `--min-samples 20` drops thinly populated buckets, which can remove prefill and
  mixed phases from a decode-dominated baseline. The CLI warns when that happens. Use
  `--min-samples 1` to score them, and treat a phase with few buckets as indicative.
- The default gate is 6.0 percent mean relative error per phase.

### Cost-model fits (`srl-regress`)

- Training fits are weighted by relative residuals, and a latency design matrix that does not
  reach full rank raises unless `--allow-rank-deficient` is passed.
- Every entry stores the `feature_ranges` it was fitted over. A query outside them raises
  `TrainingOutOfDomainError` unless the feature is named in `training_cost.extrapolate`, and each
  permitted exceedance is recorded in the `train` telemetry stream.
- `ComposedTraining` is scored by `score_composed_training` in `srl/cost/training/validate.py`,
  which also reports the signed mean and the share of the prediction each term contributes. A
  signed mean equal to the negated mean error means every record is under-predicted, which
  indicates a missing additive term and not noise.

---

## Figures that reproduce

All figures are for Qwen3-8B on H20 and use the vLLM profile and kernel tables under `outputs/`,
which are gitignored. The pinned values live in tests that skip when that data is absent, so a
fresh clone cannot reproduce them without re-running the sweeps. Re-measuring moves wall-clock
latencies slightly, so re-running a sweep can legitimately shift a figure by a few hundredths of a
percentage point.

### Rollout regressions (in-sample)

From `outputs/cost/qwen3_8b/rollout_*.json`, fitted on `outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl`
(49,208 successful iterations, TP=1) with the default `--validation-fraction 0`.

| Model | Samples | Design rank | MRE | Held out |
|---|---|---|---|---|
| `roofline` | 49,208 | 4 | 5.252% | No |
| `psrl` | 49,208 | 3 | 5.164% | No |
| `distserve` | 49,208 | 4 | 5.252% | No |

`distserve` reproduces the `roofline` figure because it re-expresses the same fit through
architecture constants. These are lower bounds on error on unseen queries.

### Kernel holdout, TP=1 tree

Pinned in `tests/test_kernel_baseline_regression.py`, `--fraction 1.0`, leave-one-out over
bracketed points. Tolerance 0.01 percentage points.

| Op | Backend | Points scored | MRE | Under the 5.0% gate |
|---|---|---|---|---|
| `gemm` | vllm | 64 | 3.91% | yes |
| `context_attention` | vllm | 59 | 6.46% | no |
| `generation_attention` | vllm | 72 | 11.55% | no |
| `context_attention` | flashinfer | 59 | 10.08% | no |
| `generation_attention` | flashinfer | 72 | 6.42% | no |

The decode attention table is timed by CUDA graph replay because the serving engine replays a
captured graph for decode batches. The surface over that table has real curvature to fit, which
is why its holdout error is the largest of the five while the end-to-end decode error below is
small.

### Composed rollout against a vLLM baseline, TP=1

Pinned in the same test. `pchip` surface, `--min-samples 1`, baseline
`outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl`. Not fitted to this baseline.

| Phase | Buckets | MRE |
|---|---|---|
| decode | 216 | 3.02% |
| prefill | 29 | 3.84% |
| mixed | 5 | 0.56% |

The mixed phase rests on five buckets and is indicative only.

### Kernel holdout and composed rollout, TP=2 tree

Pinned in `tests/test_collective_baseline_regression.py`. The holdout rows use `--fraction 1.0`.
The collective rows have 96 points each, spanning world sizes 2, 4 and 8 and both dtypes in one
table keyed by `(world_size, nodes)`.

| Op | Backend | Points | MRE |
|---|---|---|---|
| `gemm` | vllm | 174 | 3.72% |
| `context_attention` | vllm | 59 | 6.59% |
| `generation_attention` | vllm | 72 | 9.50% |
| `context_attention` | flashinfer | 59 | 9.83% |
| `generation_attention` | flashinfer | 72 | 6.36% |
| `elementwise` | vllm | 65 | 6.71% |
| `all_reduce` | vllm, nccl | 96 each | 6.31%, 4.66% |
| `reduce_scatter` | vllm, nccl | 96 each | 4.71%, 10.17% |
| `all_gather` | vllm, nccl | 96 each | 3.16%, 7.11% |
| `broadcast` | vllm, nccl | 96 each | 4.73%, 6.52% |
| `p2p` | nccl | 96 | 4.97% |

End to end at TP=2 against the serial vLLM TP=2 baseline, whole grid, `pchip`,
`--min-samples 1`. The baseline is the file `outputs/baseline_a0/vllm_tp2_spliced.jsonl`, in which
the single-request prefill points are replaced by a re-measurement made with enough prompts per
iteration to drop the cold-start overhead that a one-prompt run records.

| Phase | Buckets | MRE |
|---|---|---|
| prefill | 29 | 4.02% |
| decode | 204 | 4.03% |
| mixed | 7 | 4.55% |

### Cross-node collective tables

Not pinned by a test. `holdout_report` over the local `outputs/profile/qwen3_8b/kernel_multinode/kernel`
tree, `--fraction 1.0`, 64 points per row, NCCL backend only (vLLM disables its custom all-reduce
across nodes, so there is no `vllm` row).

| Op | MRE |
|---|---|
| `all_gather` | 9.86% |
| `all_reduce` | 14.15% |
| `broadcast` | 17.31% |
| `p2p` | 3.44% |
| `reduce_scatter` | 11.77% |

Apart from `p2p`, these are less accurate than the intra-node rows above. See {doc}`/reference/limitations`.

### `ComposedTraining` against Megatron THD steps

Pinned in `tests/test_training_composed_accuracy.py`: 22 profiled records at TP=2, PP=3 with full
recompute and THD packing, scored against kernel tables for TP=2. Mean relative error 8.13 percent
and signed mean -8.13 percent, tolerance 0.5 percentage points. Every record is under-predicted,
so the remainder is a missing additive term and not noise. The test also pins each term's share of
the prediction, because a pooled error cannot distinguish an improvement from one term growing to
cover another's omission.

### Megatron training model (held-out split)

From `outputs/cost/qwen3_8b/training_hydraulis.json`, fitted from
`outputs/profile/qwen3_8b/megatron_bridge/profiles/megatron_bridge.jsonl` (40 candidates, all
successful) with `--validation-fraction 0.2`. Every entry carries `validation_is_in_sample: false`.
Each cell holds 8 training and 2 held-out points, so a held-out figure here is set by two
measurements.

| Cell | Latency train MRE | Latency held-out MRE | Memory train MRE | Memory held-out MRE |
|---|---|---|---|---|
| TP=1 PP=1 | 3.845% | 2.322% | 0.773% | 2.475% |
| TP=1 PP=4 | 5.485% | 4.809% | 2.744% | 2.373% |
| TP=4 PP=1 | 8.877% | 23.282% | 0.745% | 1.297% |
| TP=8 PP=1 | 10.937% | 9.010% | 0.389% | 0.809% |

Do not read 23.282 percent as a generalisation estimate. With two held-out points one unlucky
shape sets it.

### FSDP2 and HSDP training model (held-out split)

From `outputs/cost/qwen3_8b/training_fsdp.json`, fitted from
`outputs/profile/qwen3_8b/fsdp/profiles/fsdp.jsonl` (108 candidates: 73 success, 34 OOM, 1
failed) with `--validation-fraction 0.2`. Every entry carries `validation_is_in_sample: false`.
OOM candidates enter the memory fit as inequality bounds.

| Cell | Latency train n | Latency train MRE | Latency held-out n | Latency held-out MRE |
|---|---|---|---|---|
| shard 2, replicate 4 | 15 | 4.236% | 4 | 6.519% |
| shard 4, replicate 2 | 22 | 0.777% | 5 | 0.820% |
| shard 8, replicate 1 | 22 | 0.648% | 5 | 1.394% |

| Cell | Memory train n | Memory train MRE | Memory held-out n | Memory held-out MRE |
|---|---|---|---|---|
| shard 2, replicate 4 | 15 | 5.053% | 4 | 4.897% |
| shard 4, replicate 2 | 22 | 4.673% | 5 | 5.981% |
| shard 8, replicate 1 | 22 | 4.268% | 5 | 5.388% |

The training $r^2$ of the shard-2 memory cell is negative (about $-8.19$). That is the OOM
inequality constraints raising the intercept above what the successful records alone support, so
the model is deliberately conservative about predicting that a workload fits. It is not a fit
failure.

---

## Reading these numbers

- A latency or memory figure describes queries inside the fitted `feature_ranges`. Outside them
  the simulator refuses unless extrapolation is requested by feature name, so no figure here covers
  an extrapolated query.
- A per-kernel holdout figure and an end-to-end figure answer different questions, and they can
  move in opposite directions. Moving a table from event timing to graph replay raised the decode
  attention holdout error and lowered the end-to-end decode error.
- Cross-hardware transfer of the kernel tables is not validated. Every figure above is H20.
- Whether a simulated system matches a real run is a separate question from any cost-model
  figure. `srl-compare` reports where a simulation differs from a converted real run.
