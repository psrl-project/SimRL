# Kernel profiling

The kernel pipeline writes parquet *performance tables* instead of fitting a parametric model.
{doc}`../cost-models/composed-rollout` and `ComposedTraining` read those tables directly and sum
measured kernel latencies per step, so a composed model needs this sweep to have been run.

The pipeline has no server, no KV-cache manager and no scheduler. Kernels are timed on synthetic
tensors sized from the checkpoint architecture, so the KV-overflow, preemption and
startup-serialisation constraints of the vLLM sweep do not apply and the grid can be as dense as
GPU time allows.

## Commands

```bash
# measure every registered family, then fit floors and run the interpolation gate
bash examples/profile/kernel/profile_qwen3_8b.sh all

# one family only, e.g. to re-sweep collectives without re-measuring compute
FAMILY=collective bash examples/profile/kernel/profile_qwen3_8b.sh run
```

`TP` selects the config file (`qwen3_8b_h20_tp${TP}.yaml`). The individual subcommands:

```bash
srl-profile kernels run       --config CFG [--family all|<family>]
srl-profile kernels fit-floor --config CFG [--candidates-us US ...]
srl-profile kernels holdout   --config CFG [--fraction 1.0] [--seed 0] [--gate 5.0] [--output PATH]
srl-profile kernels verify-shapes [--config CFG | --model-path PATH]
srl-profile kernels validate  --config CFG --baseline VLLM_JSONL \
                                  [--model-path PATH] [--surface pchip|shepard] \
                                  [--min-samples 20] [--gate 6.0] [--output PATH]
```

Run `run`, then `fit-floor`, then `holdout`. `fit-floor` is not optional: latency does not fall
to zero at tiny shapes, and a floor-taking op without one is systematically wrong in the
overhead-dominated regime. A missing `floors.json` falls back to `FALLBACK_FLOORS_S` with a
warning, and every op that takes a floor and has no fallback entry gets a second warning.

`verify-shapes` touches no GPU and needs only a checkpoint. It builds the model on the meta
device, enumerates its 2-D weights, applies the engine's fusion rules and compares them with the
shapes the architecture family declares. It exits non-zero on a mismatch or on heterogeneous
layers, because a family multiplies one layer's shapes by the layer count.

`validate` builds `ComposedRollout` from the config's `tensor_parallel` degree and the given
kernel root, and scores it against a vLLM baseline. See {doc}`tensor-parallel`.

## What is measured

`run` iterates the registered families, and for each family calls the first op's `measure` once.
Families are directories under the table root. Ops are the rows inside them.

| family | op | slice columns | axis columns | floor | graph-replayed |
|---|---|---|---|---|---|
| `gemm` | `gemm` | `gemm_dtype, n, k` | `m` | no | no |
| `attention` | `context_attention` | `num_key_value_heads, head_dim, attn_dtype, kv_cache_dtype, block_size, window_size` | `num_heads, batch_size, isl` | yes | no |
| `attention` | `generation_attention` | same | `num_heads, batch_size, step` | yes | yes |
| `elementwise` | `elementwise` | `dtype, op_kind` | `n_elements` | yes | no |
| `collective` | `all_reduce`, `reduce_scatter`, `all_gather`, `broadcast`, `p2p` | `dtype, world_size, nodes` | `message_bytes` | yes | no |
| `attention_training` | `training_attention_forward`, `training_attention_backward` | `num_key_value_heads, head_dim, attn_dtype, attn_mask_type` | `num_heads, total_tokens` | yes | no |
| `gdn` | `gdn_context` | `num_key_heads, num_value_heads, head_dim, attn_dtype, chunk_size` | `total_tokens` | yes | no |
| `gdn` | `gdn_decode` | same | `batch_size` | yes | yes |
| `optimizer` | `optimizer_step` | `optimizer, state_kind` | `param_count` | yes | no |

Slice columns are categorical and matched exactly at query time. Axis columns are numeric and
interpolated in log2 space. A query whose slice was never measured raises a coverage error
rather than blending into a neighbouring slice.

Notes on individual families:

- **`gemm`** covers forward, dgrad, wgrad and logits shapes in one table. Per-`(n, k)` slicing
  makes a dgrad's swapped `(k, n)` pair a *different slice* from its forward projection, and a
  wgrad row is `(n, k, tokens)` because that is what `F.linear` is actually called with. The
  table stores latency only. One surface therefore prices both a training and an inference step.
  GEMM takes no floor.
- **`attention`** measures both phases against one paged KV cache, through two backends per grid
  point (below).
- **`elementwise`** measures five `op_kind` slices: `rms_norm`, `rotary_embedding`,
  `silu_and_mul`, `qk_norm`, and `tensor_plumbing`. The last is not a model kernel. It stands for
  the `aten::copy_`, `aten::cat`, `aten::to`, `aten::mul` and `aten::add_` launches a Megatron
  step issues to stage buffers around Transformer Engine's fused modules, and it is priced by
  count rather than by bandwidth. `ComposedRollout` does **not** add an elementwise term.
  `ComposedTraining` does. See {doc}`../cost-models/composed-rollout`.
- **`collective`** is the tensor-parallel and training-communication family. Its op list,
  topology axes and dispatch measurement are the subject of {doc}`tensor-parallel`.
- **`attention_training`** prices Transformer Engine's THD-packed attention in both directions.
  Backward is measured rather than derived, because the measured backward/forward ratio is above
  two and drifts with sequence length. `attn_mask_type` is a slice column because THD requires
  `padding_causal` and that masking selects a different kernel path.
- **`gdn`** prices a gated delta net, the linear-attention layer a hybrid model uses for most of
  its depth. Its prefill is varlen and linear in the token total, so it has one axis
  `total_tokens`. Its decode state is a fixed size whatever the KV depth, so it has one axis
  `batch_size` and no `step`.
- **`optimizer`** prices Transformer Engine's `FusedAdam`, which is what megatron-core calls, over
  the parameter shard one rank updates. It is the only op with no token axis: an Adam step costs
  the same however many tokens produced the gradients.

`graph_replayed` selects the timing helper. A kernel the serving engine replays from a captured
CUDA graph is timed by graph replay, so the table does not charge a per-call Python dispatch cost
the engine never pays. An eager kernel is timed with CUDA events. vLLM's
`cudagraph_mode=FULL_AND_PIECEWISE` splits the two phases: a decode batch replays one full graph,
while prefill and mixed batches leave the splitting ops, including softmax attention and GDN,
outside it. That is why `generation_attention` and `gdn_decode` replay but
`context_attention` and `gdn_context` do not. Attention's decode rows are a known,
deliberately unconverted case: one table serves all phases and the composed model reads it for
all of them, so converting it would fix decode and break prefill by the same factor.

## Grids

Each family reads its own block from the YAML. A family that is absent is simply not swept.
Adding an op adds a block under `sweep/` and nothing in the config schema.

| block | keys |
|---|---|
| `gemm` | `token_counts`, `logits_request_counts` |
| `attention` | `batch_sizes`, `context_seq_lens`, `generation_steps`, `block_size`, `window_size`, `max_context_tokens`, `max_generation_kv_tokens` |
| `elementwise` | `token_counts` |
| `collective` | `message_bytes`, `topologies`, `hostnames`, `dtypes` |
| `attention_training` | `total_tokens`, `num_heads`, `mask_types` |
| `gdn` | `total_tokens`, `batch_sizes` |
| `optimizer` | `param_counts` |

Two attention fields truncate the grid before it is measured, both counted in KV elements, which
cost four bytes each at bfloat16 because K and V are both stored:

- `max_context_tokens` drops a context point whose `batch_size * seq_len` exceeds it. The default
  is `262144` tokens.
- `max_generation_kv_tokens` drops a generation point whose
  `batch_size * step * num_key_value_heads * head_dim` exceeds it. The default is `1 << 34`
  elements, deliberately generous: the largest configured decode point sits well inside the
  memory an H20 has free after bfloat16 weights. Lower them explicitly for a smaller card. A
  budget that is too small drops sweep points, and a sparse grid starves the holdout gate of the
  interior points it needs.

`measurement` sets `warmup_iterations` (default 5), `active_iterations` (default 20),
`dtype` and `kv_cache_dtype`.

## Output layout

```
{output_dir}/kernel/{gpu_slug}/{family}/{framework}/{version}/{op}_perf.parquet
```

`kernel_source` is a **column** of each table, not a path segment, so one table may mix kernels
(the collective tables normally do). `framework` and `version` name what timed the kernel:
`vllm` for GEMM, elementwise, softmax attention and GDN. `flashinfer` for the FlashInfer
attention tables. `nccl` for the collective control tables. `megatron` for training
attention and the optimizer, because Transformer Engine is what megatron-core calls.

Every table carries `framework, version, device, op_name, kernel_source, <op columns>, latency_s`.
Writes are atomic: rows go to a sibling `.tmp` file which then replaces the target, so a sweep
that dies partway cannot leave a truncated parquet that reads as a small table. A row whose keys
do not exactly match the op's declared columns is rejected at write time.

`floors.json` is written into each version directory that holds at least one floor-taking op and
is keyed by **op name**:

```javascript
{
  "context_attention": 0.000032,
  "generation_attention": 0.000022,
  "fitted_mre_percent": {"context_attention": 5.74, "generation_attention": 6.63},
  "candidates_us": [0, 2, 4, 6, 8, 12, 16, 22, 32, 45, 60, 80],
  "gpu": "h20",
  "backend": "vllm"
}
```

A file still carrying the retired phase keys `context` or `generation` is rejected with an error
rather than read leniently. Reading it as if it were op-keyed would apply a zero floor, which is
a silent accuracy loss. `holdout` and `validate` default their reports to
`{table_root}/holdout_report.json` and `{table_root}/validate_report.json`.

## Attention backends

Every attention grid point is measured twice: once through the backend vLLM's own selector picks
(`framework: vllm`) and once through FlashInfer (`framework: flashinfer`). The selection is made
by calling vLLM's selector, not by re-deriving its priority order, so it cannot drift from the
installed vLLM.

This makes the backend choice a number rather than an argument, and it also records what the
server actually ran. On H20 the selector resolves to FlashAttention-3 vendored inside vLLM,
which serves both prefill and decode. The vLLM table is the default for every downstream query
because it describes the kernel the server executes. The FlashInfer rows answer what a switch
would buy. Scoring a vLLM baseline against them, or overriding `attention_backend` in a
simulation, compares against a backend that never ran. See {doc}`../reference/limitations` for
the FlashInfer GQA restriction that can leave one table unwritten.

## Utility, SOL and the dispatch floor

Utility is `SOL / measured_latency`. SOL is the roofline upper bound from the GPU's peak
throughput (`srl/cost/kernel/sol.py`) and the packaged spec in `cost/kernel/data/gpu_specs.yaml`.
A utility near 1 means the kernel runs at the hardware ceiling. A value above roughly 1.2 means
the SOL formula is missing work, not that the GPU beat its spec.

Interpolating utility rather than latency is what makes a sparse table sufficient. The roofline
carries the orders of magnitude analytically, so the surface only has to resolve the residual:
an efficiency. Past the measured range both surfaces clamp to the boundary utility rather than
extrapolating, because extrapolating an efficiency is defensible and inventing one is not.

The floor is a per-call dispatch cost that a pure roofline does not express, and it enters as

```text
latency = max(compute, memory, floor)
```

not as `floor + roofline`, which would double-count it in the regime where it binds. `fit-floor`
scans `CANDIDATE_FLOORS_US` for each floor-taking op in each version directory, evaluates the
full bracketed holdout set at each candidate and writes the MRE-minimising value. The MRE-versus-
floor curve is unimodal in practice, so a scan finds the minimum. Ops declaring `fits_floor =
False` (GEMM) are skipped.

The floor is a property of the backend's dispatch path, not of the silicon, so a `floors.json`
fitted for one backend version is loaded only for that version's directory. Refit it whenever
the backend, its version or the GPU changes.

## Interpolation surfaces

`KernelDatabase` takes `utility_surface`:

- `pchip` (default for `ComposedRollout` and `kernels validate`): monotone-preserving cubic
  Hermite splines, hand-rolled. It is exact for one or two varying axes. Above two it defers to
  Shepard. Constant axes do not count, so an attention slice that pins `num_heads` is
  effectively two-dimensional. Past the measured range it clamps.
- `shepard`: the log2-space modified-Shepard blend over all measured points, defined for any
  number of axes.

An axis value that is zero cannot be placed in log2 space. Such rows stay in the measured
utility range diagnostic but are excluded from interpolation scoring. Generation attention at
`step = 0` is the canonical case.

## Holdout gate

```bash
srl-profile kernels holdout --config CFG --fraction 1.0 --gate 5.0
```

For each `(op, backend)` table, `holdout` keeps only slices with at least four points, keeps only
points that are *bracketed* after their own removal, removes a random `ceil(fraction * n)` of
them one at a time, rebuilds the surface from the rest and predicts the removed point. Only
interior points are scored, because removing a point that is no longer bracketed turns the query
into a frontier hold, which measures a different mechanism than interpolation accuracy. The gate
passes when the mean relative error is below `--gate` and at least one point was scored. A result
with zero scored points is reported as a *coverage* problem, not an accuracy problem.

Details that matter when reading a report:

- **`--fraction` defaults to `1.0`.** Any value below it samples a subset and warns, because a
  small sample can flatter a surface: a handful of points may score well where the full set does
  not. Treat `1.0` as the only figure worth quoting.
- Each result also reports `p90`, `max`, the worst coordinate, the utility range and the number
  of skipped slices. A skipped slice is one too small or too degenerate to hold a point out and
  still interpolate, so a high skip count means the MRE describes the well-covered part of the
  table.
- The held-out score is computed with the Shepard surface (`UtilityGrid`), whatever surface
  `ComposedRollout` will later query, so the gate measures interpolation quality rather than the
  accuracy of the surface a simulation uses.
- Floors used during scoring come from the version directory's `floors.json`, so run `fit-floor`
  first.

Per-op per-backend gate figures for the reference GPU and model live in
{doc}`../reference/validation`, which owns accuracy.

## End-to-end validation

`kernels holdout` tests single-kernel self-consistency. `kernels validate` tests a different and
harder claim: whether summing those kernels reproduces a real vLLM engine iteration.

```bash
srl-profile kernels validate \
    --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml \
    --baseline outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
    --model-path $MODEL_PATH \
    --surface pchip --min-samples 1 --gate 100 \
    --output /tmp/validate.json
```

The baseline is a vLLM JSONL ({doc}`vllm`). Bucketing is what makes the comparison meaningful:

- Iterations are grouped by `(phase, request_count, mean context, token count)`.
- The context and token keys are rounded to a **linear** multiple (`DEFAULT_CONTEXT_BUCKET =
  256`), never to a power of two. Snapping to powers of two once floored a real mean context and
  produced an error that did not exist.
- A bucket's latency is the **median** of its iterations, so one slow step cannot move it.
- A bucket holding fewer than `--min-samples` iterations is dropped. The default is `20`. On a
  decode-dominated baseline that can hide prefill and mixed entirely, and the tool warns once per
  phase it dropped. Pass `--min-samples 1` to score all three.

Results are reported per phase, each with bucket count, MRE, p90, max, the worst bucket and the
median predicted/measured ratio. Phases are separate because they are different regimes:
prefill attention is quadratic in sequence length, decode attention is linear in KV history.

## Using the tables in a simulation

```yaml
rollout_cost:
  name: composed
  kernel_table_root: outputs/profile/qwen3_8b/kernel_tp1
  gpu: H20
  model_path: $MODEL_PATH
  attention_backend: ""      # empty = the backend vLLM's selector picks
  utility_surface: pchip
```

or, through the example script:

```bash
ROLLOUT_MODEL=composed \
KERNEL_ROOT=outputs/profile/qwen3_8b/kernel_tp1/kernel \
GPU=h20 MODEL_PATH=$MODEL_PATH \
bash examples/simulate/simulate.sh
```

`kernel_table_root` is the directory that *contains* `kernel/`, so `ComposedRollout` resolves
tables at `{kernel_table_root}/kernel/{gpu}/`. `gpu` must match the slug in the tree. Leave
`attention_backend` empty unless deliberately comparing backends. A `tp > 1` simulation needs a
tree swept at that degree, and the example script checks that the path names it. See
{doc}`tensor-parallel`.

For adding a new op, the op protocol, and the shape families that decide which kernels a model
runs, see {doc}`../development/kernel-ops` and {doc}`../development/model-families`.
