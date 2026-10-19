# Cost Models

A cost model turns measured profiling output into a prediction of latency or peak memory. Two
kinds exist:

- **Fitted models** reduce many records to a small set of coefficients and store them in a
  `CostArtifact` JSON. `srl-regress` produces the artifact, and the simulator loads it.
- **Composed models** hold no coefficients. They query measured parquet kernel tables and sum
  per-step kernel latencies, so there is no regression step and no artifact.

Rollout and training are independent. A simulation selects one model of each, from separate Hydra
groups, and a rollout model never has to be fitted for a training model to be used.

## Rollout models

| Hydra default (`rollout_cost=`) | Kind | Fitted from | Needs `artifact_path` |
|---|---|---|---|
| `roofline` | fitted | vLLM JSONL | no; YAML constants are defaults |
| `psrl` | fitted | vLLM JSONL | yes |
| `distserve` | fitted | vLLM JSONL, via a roofline calibration | no; YAML constants are defaults |
| `composed` | kernel sum | kernel parquet tables | no; needs `kernel_table_root`, `gpu`, `model_path` |

`composed` ignores `artifact_path` even though its YAML file carries the key, because there is no
regression to load. `psrl` requires one.

## Training models

| Hydra default (`training_cost=`) | Kind | Fitted from | Needs `artifact_path` |
|---|---|---|---|
| `fixed` | constant | nothing | no |
| `analytical` | analytic roofline | nothing | no |
| `linear` | analytic | nothing | no |
| `hydraulis` | fitted | Megatron-Bridge JSONL | yes |
| `fsdp` | fitted | FSDP JSONL | yes |

`fixed`, `analytical` and `linear` carry hand-set constants, which is what makes
{doc}`../getting-started/walkthrough/00-first-simulation` runnable without a GPU. The two fitted
models fall under `TRAINING_MODEL_REGISTRY` and are interchangeable only within their own backend:
a `hydraulis` model fits records from Megatron-Bridge, and an `fsdp` model fits records from the
FSDP launcher, and neither accepts the other's records.

There is also a **kernel-composed training latency model**, `ComposedTraining` in
`srl/cost/training/composed.py`. It sums measured kernels like `ComposedRollout` does, carries no
coefficients, and can be paired with `HydraulisMemoryModel` for memory. It is a library class
rather than a Hydra default, so it is constructed in Python and scored through
`srl/cost/training/validate.py`; it is not selectable as `training_cost=composed` and is not
exposed by `srl-regress`. {doc}`training-megatron` describes it and its per-term decomposition.

## Which regress subcommand applies

Only two regressions exist:

```bash
# rollout, from vLLM records
srl-regress rollout --model {roofline|psrl|distserve} \
    --input  outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
    --output outputs/cost/qwen3_8b/rollout_roofline.json

# training, from Megatron-Bridge or FSDP records
srl-regress training --latency-model {hydraulis|fsdp} --memory-model {hydraulis|fsdp} \
    --input  outputs/profile/qwen3_8b/megatron_bridge/profiles/megatron_bridge.jsonl \
    --output outputs/cost/qwen3_8b/training_hydraulis.json
```

`srl-regress rollout` accepts `--block-size` (default 16), `--validation-fraction` (default
`0.0`), and, for `distserve`, `--hidden-size` and `--intermediate-size`.
`srl-regress training` accepts `--validation-fraction` (default `0.2`) and
`--allow-rank-deficient`, and requires the FSDP latency and memory models to be selected together.

The kernel pipeline has no regression. Its tables feed `ComposedRollout` and `ComposedTraining`
directly, and its own accuracy is measured by `srl-profile kernels holdout` and
`srl-profile kernels validate` ({doc}`../profiling/kernels`).

## What a fitted artifact contains

A `CostArtifact` is JSON with `kind`, `model_type`, the model and hardware descriptors, a
`profile_digest` linking it to the measured records, and a list of `entries`. Each entry carries:

- `parallelism`: the mesh the entry describes.
- `selector`: the discrete configuration keys (component, schedule, recompute, dtypes, and so on).
- `coefficients`: the fitted or calibrated values.
- `feature_ranges`: the profiled span of each feature.
- `train_metrics` and `validation_metrics`: sample count, rank, $r^2$ and MRE.
- `diagnostics`: how the fit was done, and what it assumed.

Lookup is **exact**: `CostArtifact.entry_for` requires exactly one entry to match the requested
parallelism and every selector key, and raises otherwise. A cell that averaged over tensor-parallel
sizes would describe none of them, so there is no nearest-neighbour fallback.

`feature_ranges` become prediction-time gates. A query outside a recorded range raises
`TrainingOutOfDomainError` unless that feature is named in `training_cost.extrapolate`, and a
permitted extrapolation is always recorded in the estimate's diagnostics. Hydraulis latency pools
pipeline degrees into one entry per TP group and derives PP from the topology; FSDP memory does
not pool, because its peak depends on which stage is in flight. The full schema is in
{doc}`../reference/schemas`.

Fitting defaults and the split:

- Training fits split records by **run id**, not by row, so repeats of one candidate stay on one
  side. Each candidate contributes one row, its retained step times reduced to a median.
- A fit that reaches fewer than the required rank raises rather than returning unseparable
  coefficients. `--allow-rank-deficient` lowers the check for an exploratory fit.
- `srl-regress training` records whether its validation metrics are in-sample
  (`validation_is_in_sample`), because the split degenerates at a small fraction, a single run or
  a validation set that would leave no training rows.
- `srl-regress rollout` with vLLM records also writes diagnostic plots beside the artifact,
  under `{artifact_stem}_plots/`.

## Accuracy

Accuracy figures depend on the GPU, the model and the sweep, and are not derivable from this
repository. {doc}`../reference/validation` owns them, states how each kind of evidence was
produced, and lists what can be reproduced today.

```{toctree}
:maxdepth: 2

rollout
composed-rollout
training-megatron
training-fsdp
prior-art
```
