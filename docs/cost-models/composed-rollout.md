# ComposedRollout

Every fitted rollout model ({doc}`rollout`) is regressed from end-to-end vLLM server
measurements, so it must be refitted for a new model, GPU or parallelism degree.
`ComposedRollout` instead predicts one inference engine iteration by **summing the measured
latencies of the individual kernels** the engine runs. Its input is the kernel perf tables of
{doc}`../profiling/kernels`, so there is no `srl-regress` step and no fitted artifact.

The kernel tables are stored per GPU, and every query is normalised by the target GPU's
speed-of-light roofline, so changing GPU requires a new utility surface, not a new end-to-end vLLM
sweep. Changing the model or the TP degree does require new tables, because kernel shapes are
sharded and looked up at measurement time.

## Prerequisites

```bash
bash examples/profile/kernel/profile_qwen3_8b.sh all
```

That writes parquet tables under `outputs/profile/qwen3_8b/kernel_tp1/kernel/` and runs the
interpolation gate. Per-kernel accuracy, the table layout, floor fitting, and the holdout and
validate mechanics are covered on {doc}`../profiling/kernels`.

## Configuration

```yaml
defaults:
  - rollout_cost: composed

rollout_cost:
  name: composed
  parallelism:
    tp: 1
  kernel_table_root: outputs/profile/qwen3_8b/kernel_tp1
  gpu: H20
  model_path: $MODEL_PATH
  attention_backend: ""       # empty = the backend vLLM's selector picks
  utility_surface: pchip
  decode_integration_stride: 1024
  saturation_step_threshold: 4611686018427387904
```

| field | meaning |
|---|---|
| `parallelism.tp` | Tensor-parallel degree of the simulated engine. `ComposedRollout` raises if it disagrees with the degree its `ModelOps` was built for, because `ModelOps` decides both the sharded GEMM shapes and whether a communication term is emitted. `pp`, `dp` and `cp` are accepted and unused. |
| `kernel_table_root` | The directory that **contains** `kernel/`; tables resolve at `{kernel_table_root}/kernel/{gpu}/`. |
| `gpu` | GPU slug selecting the subdirectory. It must match the recorded device. |
| `model_path` | HuggingFace checkpoint. The architecture is read here to derive each step's GEMM shapes, head counts and, for a hybrid model, its layer kinds. |
| `attention_backend` | `vllm`, `flashinfer`, or empty. Empty uses the database default, the backend vLLM's own selector picks, which is what a server ran. Overriding it compares against kernels that never ran. |
| `utility_surface` | `pchip` (default) or `shepard`. |
| `decode_integration_stride` | Steps between lattice nodes in the decode segment integrator. |
| `saturation_step_threshold` | Per-request KV depth at which the closed-form decode advance is enabled. The default disables it; see below. |

`kernel_table_root`, `gpu` and `model_path` are required. The example script wires this from
`ROLLOUT_MODEL=composed`, `KERNEL_ROOT`, `GPU` and `MODEL_PATH`.

## What one step costs

`predict` sums, in this order:

```text
step = sum over projections   (layers * gemm latency)              # plus the logits GEMM once
     + sum over collectives   (multiplier * comm latency)          # empty at tp = 1
     + attention_layers * per-layer attention latency
     + gdn_layers       * per-layer gated-delta-net latency        # 0 for a dense model
     + expert weight-read excess                                  # 0 for a dense model
```

- **GEMMs.** Each projection is queried at `(m = tokens, n, k)` with the layer count as its
  multiplier. The logits projection runs once per step over `request_count` tokens, because an
  engine computes logits only for each sequence's last token. A mixture-of-experts block issues
  one grouped GEMM per layer rather than one per activated expert: decomposing it per expert
  charges launch floor per tiny GEMM and is badly wrong in the other direction.
- **Attention.** `context_attention` and `generation_attention` are queried per layer, and each
  is multiplied by its **own** layer count. A dense model has `attention_layers = num_layers` and
  `gdn_layers = 0`; a hybrid model such as Qwen3.5-35B-A3B runs softmax attention on 10 of its 40
  layers and a gated delta net on the other 30, and charging all 40 as one kind is a fourfold
  error either way. The per-layer cost is
  - for prefill: `batch_size = prefill_ctx^2 / sum_l2`, `seq_len = sum_l2 / prefill_ctx`;
  - for decode: `batch_size = request_count`, `step = ctxsum / request_count`;
  - for a mixed step: both terms, summed.
- **Mean-extent rule.** `sum_l2` is the sum of squared attention extents and `prefill_ctx` their
  plain sum, so the prefilling set's weighted mean extent is `sum_l2 / prefill_ctx` and its count
  is `prefill_ctx^2 / sum_l2`. When `prefill_ctx` is zero the caller did not separate the two and
  `n_tokens` stands in. Dividing `n_tokens` by `request_count` instead would count decoders as
  prompt requests and price one long chunk as many tiny prompts.
- **Collectives.** At `tp > 1`, `ModelOps.comm_queries` adds two all-reduces per layer, after the
  attention output projection and after the MLP down projection, with the message sized
  `tokens * hidden * dtype bytes`. They are priced from the `vllm` table, which times vLLM's own
  dispatcher; `nccl` is the control. At `tp = 1` the query list is empty. See
  {doc}`../profiling/tensor-parallel`.
- **GDN.** Priced separately from softmax attention because its cost function is different: the
  chunked prefill is varlen and linear in the packed token total, and the recurrent decode has no
  history-depth axis at all.
- **Expert weight read.** A grouped expert GEMM reads every activated expert's weights whatever
  its token share, while an `m = tokens` GEMM query charges only the selected copies.
  `_expert_weight_seconds` adds the excess, priced at the device's memory bandwidth rather than
  through a table, because no measured GEMM row carries more than one weight copy. It is charged
  in full rather than amortised per expert's token share: dividing by tokens per expert is
  physically defensible for prefill but raised decode error, and decode is where the term is
  load-bearing.

**Elementwise is measured but not summed.** Rope, norms, activations and `tensor_plumbing` have a
perf table, and `ComposedRollout` deliberately does not add them. The measured GEMM and attention
latencies were collected from a real server whose elementwise kernels ran between them, so that
wall-clock time is already absorbed into the fitted utility values; adding a separate term would
double-count it. The table exists for `ComposedTraining`, which runs without CUDA graphs and
therefore pays the cost genuinely. A test pins the omission, and the fuzzier argument for it is
that adding the term has been observed to move every phase's error and every median ratio away
from 1 rather than toward it. Re-measure before wiring it in.

## Decode segments and the lattice integrator

A decode segment is a run of `k` steps at a fixed batch with no prefill mixed in and no request
entering or leaving. Fitted models integrate it in closed form ({doc}`rollout`); a kernel sum is
not `max(F, alpha + beta * k)`, so `ComposedRollout` integrates numerically.

The obvious integrator, one query per step, is exact but costly. Sampling at a stride and
trapezoid-summing per query is **not monotone** in `k`: on a spiked cost curve it can produce `k`
where `total(k) < total(k - 1)`. A non-monotone total has no unique inverse, and the simulator
compares the returned step count against a budget by integer identity, so an inexact inverse
silently changes which branch it takes.

`LatticeIntegrator` (`srl/cost/rollout/segment.py`) is monotone by construction. Lattice nodes at
`0, s, 2s, ...` are sampled once. Node `i` carries a fixed rate, the trapezoid average of the
per-step cost at nodes `i` and `i+1`, and a cumulative total, and a query interpolates linearly
inside one interval:

```text
total(k) = cumulative[i] + rates[i] * (k - i*s),   i = k // s
```

Since `rates[i]` does not depend on `k`, the total grows at a fixed positive slope inside every
interval and is continuous across them, so it is strictly increasing and bisection inverts it
exactly. `stride = 1` makes each interval one step and the sum exact. Above one, the trapezoid
average is what earns the accuracy, and the integrator samples lazily: only the nodes a query
reaches are evaluated.

The error this introduces is largest for a **cold** segment, one starting right after prefill with
little KV behind it, because that is where the per-step cost is steepest across a stride.
{doc}`../reference/validation` carries the measured figures against a step-by-step reference.

### The closed-form fast path is off

`decode_coeffs` can return `(0, intercept, secant slope)` sampled at offsets zero and `stride`, and
`saturation_step_threshold` decides when a segment uses it instead of the lattice. The default is
`1 << 62`, so no segment can reach it and every decode segment integrates on the lattice.

It is disabled on purpose. Within one segment the GEMM utilities are constant and the
generation-attention utility is nearly flat at deep KV, which is exactly when a secant is a good
approximation. But a real workload spends most of its predictions on prefill, which segment
integration never touches, so the saved decode predictions are a small share of all predictions.
The measured deviation of the approximation is of the same order as the model's own error against
a real server, which makes it a first-order contaminant rather than a saving. A per-request KV
depth is also a crude proxy: the bad region is where the generation-attention utility is
non-monotone in depth, not a depth threshold. Set a finite threshold only for a workload genuinely
dominated by long deep-KV decode runs, and re-measure end to end before trusting it.

## Prediction memoisation

A composed prediction costs several table lookups against interpolated surfaces, orders of
magnitude more than evaluating a fitted formula, so each model memoises:

- `step_time` memoises the whole `RolloutWorkload` in a bounded cache. Two call sites in the
  advance path resolve the same batch state in sequence, and two integrators built for the same
  `(n_d, ctxsum)` would otherwise evaluate identical offsets twice.
- GEMM plus communication terms are memoised on `(token_count, request_count)`, and the attention
  term on the integers it reads: `(context extent pair, generation depth, request_count)`. These
  keys repeat far more often than whole workloads do.

Both caches evict an arbitrary entry when full and are **bit-exact**: entries are exact values, so
eviction costs recomputation, never a different answer. The attention key is derived identically
in the C++ engine, so a change to it must be mirrored there.

## Scoring it against a real server

```bash
srl-profile kernels validate \
    --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml \
    --baseline outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
    --model-path $MODEL_PATH \
    --surface pchip --min-samples 1 --gate 100 \
    --output /tmp/validate-pchip.json
```

Pass `--min-samples 1` to score prefill and mixed as well as decode; at the default of 20 a
typically decode-dominated baseline has no bucket that deep for the other two phases, and the tool
warns about each phase it dropped. Score against the `vllm` tables, which are the kernels the
server ran. The bucketing rules and the report fields are on {doc}`../profiling/kernels`.

Accuracy figures by phase and by degree are in {doc}`../reference/validation`, along with what kind
of evidence each number is. The known limitations of the composition, including the mixed-step
approximation, small-batch over-prediction from per-call floors, and the TP degrees without an
end-to-end baseline, are in {doc}`../reference/limitations`.

## Known approximations

- **Mixed steps.** Only the two pure attention phases are measured. A mixed step is priced as
  context attention plus generation attention at the step's batch size, which is accurate for the
  near-uniform batches a profiling baseline contains. A chunked-prefill step is not that shape, and
  because attention is quadratic in sequence length, spreading one request's long chunk across the
  batch understates its context term severalfold. Measuring mixed batches directly is the fix and
  is not implemented.
- **Per-call floors.** A composed step sums one isolated latency per kernel, each with its own
  fitted floor, while a server replays the step inside a CUDA graph where much of that dispatch cost
  disappears. The excess is a fixed absolute amount, largest as a share of a small-batch step.
- **Overlap.** Collectives are timed in isolation, so compute/communication overlap is not
  represented. The sign of the residual is the diagnostic to read.
- **Hybrid models.** Gated-delta-net layers carry a much larger per-call floor than softmax
  attention, and charging it once per layer overstates prefill, whose real step runs the layers
  back to back on one stream. Decode, which is the dominant share of engine time, is the phase this
  model gets right.
