# Cost Models

A cost model turns measured profiling records into a small set of coefficients that predict
latency or peak memory, together with the input range those coefficients are valid over.

## Which model for which measurement

| Model | Kind | Fitted from | Config group |
|---|---|---|---|
| `roofline` | rollout | vLLM JSONL | `rollout_cost=roofline` |
| `psrl` | rollout | vLLM JSONL | `rollout_cost=psrl` |
| `distserve` | rollout | vLLM JSONL | `rollout_cost=distserve` |
| `composed` | rollout | kernel perf tables (**no fit**) | `rollout_cost=composed` |
| `hydraulis` | training | Megatron-Bridge JSONL | `training_cost=hydraulis` |
| `fsdp` | training | FSDP JSONL | `training_cost=fsdp` |
| `fixed`, `linear`, `analytical` | either | nothing, hand-set constants | e.g. `training_cost=fixed` |

The last row matters for getting started. `fixed` and `roofline` carry default coefficients
and need no profiling data at all, which is what makes
{doc}`../getting-started/walkthrough/00-first-simulation` runnable without a GPU.

## Fitting

Three of the four pipelines end in a regression:

```bash
# rollout
dryrun-regress rollout --model roofline \
    --input  outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
    --output outputs/cost/qwen3_8b/rollout_roofline.json

# training
dryrun-regress training --latency-model fsdp --memory-model fsdp \
    --input  outputs/profile/qwen3_8b/fsdp/profiles/fsdp.jsonl \
    --output outputs/cost/qwen3_8b/training_fsdp.json
```

The output is a `CostArtifact` JSON file whose schema is documented in
{doc}`../reference/schemas`. Its important property is that coefficients are stored **per
cell**, one set per distinct topology and configuration, because a fit that averaged over
tensor-parallel sizes would describe none of them. A lookup that finds no matching cell
raises rather than falling back to a neighbour.

`ComposedRollout` is the exception. It reads measured kernel tables directly and sums
per-step kernel latencies, so there is no regression and no artifact.

## Accuracy

Each model page carries its own measured accuracy. {doc}`../reference/validation` collects
all of them in one table alongside sample counts and which figures rest on a genuine
held-out split.

```{toctree}
:maxdepth: 2

rollout
composed-rollout
training-megatron
training-fsdp
prior-art
```
