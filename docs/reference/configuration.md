# Configuration reference

`srl` is a [Hydra](https://hydra.cc/) app. Configuration is assembled from *config groups*:
named directories under `srl/config/defaults/` where each YAML file is one selectable
option. The top-level `simulate.yaml` sets the default selections and the top-level `job` and
`rollout` blocks.

Cross-references: {doc}`../simulation/policies` for policy semantics,
{doc}`../simulation/domain-gates` for `extrapolate` and `batching_mode`,
{doc}`../workload/index` for workload sources.

---

## Top-level `simulate.yaml`

The default selections and the `job` / `rollout` blocks live here.

```{literalinclude} ../../srl/config/defaults/simulate.yaml
:language: yaml
```

---

## `rollout_cost` group

Selects the rollout cost model. Default: `roofline`.

Available options: `roofline`, `psrl`, `distserve`, `composed`.

### `roofline`

Parametric roofline model fitted from vLLM profiling JSONL. No profiling needed when
using the default hand-set constants. Replace them with a fitted artifact from
`srl-regress rollout --model roofline` for a calibrated result.

```{literalinclude} ../../srl/config/defaults/rollout_cost/roofline.yaml
:language: yaml
```

Key fields:

`F`
: Fixed floor latency in seconds. The `max(F, …)` ensures the model never predicts
  zero for an empty batch.

`W`, `G`
: Per-step and per-token weights in the memory-bandwidth term.

`A_p`, `A_d`
: Prefill (sum-of-squared-lengths) and decode (context-sum) attention coefficients.

`b`
: KV block size used when deriving `ctxsum` from the attention features.

### `psrl`

Same roofline formula, but the artifact is loaded from a file fitted against PSRL-specific
profiling records. The YAML is identical to `roofline` except `name: psrl`.

### `distserve`

DistServe C3/C4 formula. Re-expresses the same roofline fit through architecture constants
(`h`, `m`, `C1` through `C5`). Default constants are placeholders. Replace with fitted values.

```{literalinclude} ../../srl/config/defaults/rollout_cost/distserve.yaml
:language: yaml
```

### `composed`

`ComposedRollout`: sums measured per-kernel latencies rather than evaluating a parametric
formula. No `srl-regress` step. Requires the kernel profiling sweep to have been run first.
See {doc}`../cost-models/composed-rollout` for the full accuracy discussion.

```{literalinclude} ../../srl/config/defaults/rollout_cost/composed.yaml
:language: yaml
```

Key fields beyond the parallelism block:

`kernel_table_root`
: Root of the kernel profiling output tree. Tables are resolved at
  `{kernel_table_root}/kernel/{gpu}/`.

`gpu`
: GPU name that selects which subdirectory to load (e.g. `H20`).

`model_path`
: Path to the HuggingFace checkpoint. Used to derive GEMM shapes from the architecture.
  Use `$MODEL_PATH` rather than a hard-coded path.

`attention_backend`
: Override the attention backend (`vllm` or `flashinfer`). Leave empty to use the
  vLLM-selected backend (recommended for comparisons against a real server).

`utility_surface`
: Interpolation strategy: `pchip` (default, more accurate on H20) or `shepard`.

`decode_integration_stride`
: Lattice stride for the segment integrator. `1024` costs at most ~1.3% accuracy
  versus a step-by-step reference. Set to `1` for an exact sum.

`saturation_step_threshold`
: Disabled by default (set to max int64). A finite value enables the closed-form
  fast path for deep-KV decode segments. The measured trade-off is 1.2x speedup
  for 4.6% additional deviation on the real DAPO trace.

---

## `training_cost` group

Selects the training cost model. Default: `fixed`.

Available options: `fixed`, `analytical`, `linear`, `hydraulis`, `fsdp`.

### `fixed`

Constant step time and peak memory. No profiling needed. The simplest option for
getting-started runs.

```{literalinclude} ../../srl/config/defaults/training_cost/fixed.yaml
:language: yaml
```

### `analytical`

Roofline training cost: `max(compute_time, memory_time) * overhead_factor`. Useful for
rough estimates without profiling data.

```{literalinclude} ../../srl/config/defaults/training_cost/analytical.yaml
:language: yaml
```

### `linear`

`latency = time_per_token * total_tokens`, `peak_memory = base_memory_bytes / world_size`.

```{literalinclude} ../../srl/config/defaults/training_cost/linear.yaml
:language: yaml
```

### `hydraulis`

Fitted Megatron / Hydraulis V1 training cost. Requires a `CostArtifact` from
`srl-regress training --latency-model hydraulis`. See {doc}`../cost-models/training-megatron`.

```{literalinclude} ../../srl/config/defaults/training_cost/hydraulis.yaml
:language: yaml
```

`extrapolate`
: List of feature names for which out-of-domain queries are permitted. An unknown name
  raises at construction. See {doc}`../simulation/domain-gates` for which features are
  defensible to extrapolate. Selectable names: `micro_batch_size`, `pipeline_slots`,
  `tokens`, `activation_bytes`, `profiled_dp`.

### `fsdp`

Fitted FSDP2/HSDP training cost. Requires a `CostArtifact` from
`srl-regress training --latency-model fsdp`. See {doc}`../cost-models/training-fsdp`.

```{literalinclude} ../../srl/config/defaults/training_cost/fsdp.yaml
:language: yaml
```

`shard_size`, `replicate_size`
: Required. Identify the HSDP mesh cell to load from the artifact. Must satisfy
  `parallelism.dp == shard_size * replicate_size`.

`batching_mode`
: How real trajectory lengths become microbatches. `pad_to_max` (default) pads to the
  longest sequence in each microbatch. `token_budget` reproduces verl's dynamic batching with
  `max_tokens_per_rank`, and `packed_no_pad` charges only true token counts.
  See {doc}`../simulation/domain-gates` for the measured spread and when the modes differ.

`extrapolate`
: As for `hydraulis`. Selectable names: `num_microbatches`, `sequences_per_microbatch`,
  `tokens`, `activation_tokens`. Only `num_microbatches` is defensible for upward
  extrapolation without additional profiling.

---

## `policy` group

Selects the staleness / scheduling policy. Default: `psrl`.

Available options: `psrl`, `areal`, `roll`, `slime`, `verl`.
See {doc}`../simulation/policies` for the full behavioural description of each.

### `psrl`

Reserve/Occupy/Consume staleness inventory. Requires the external `psrl` package. When the
package is unavailable, constructing this policy raises `ImportError`. See
{doc}`../reference/limitations` for the dependency note.

```{literalinclude} ../../srl/config/defaults/policy/psrl.yaml
:language: yaml
```

### `areal`, `roll`

Simpler staleness policies with no additional configuration fields beyond `name`.

### `slime`

Loop-barrier policy: push new weights every `update_weights_interval` training steps.

```{literalinclude} ../../srl/config/defaults/policy/slime.yaml
:language: yaml
```

### `verl`

Overage-ratio policy. `staleness_threshold: null` falls back to `job.max_staleness`.

```{literalinclude} ../../srl/config/defaults/policy/verl.yaml
:language: yaml
```

---

## `workload` group

Selects the workload source. Default: `uniform`.

Available options: `uniform`, `bimodal`, `lognormal`, `normal`, `powerlaw`, `dapo_trace`.

### `uniform` (and other parametric distributions)

```{literalinclude} ../../srl/config/defaults/workload/uniform.yaml
:language: yaml
```

`source: distributions` means lengths are sampled from the named distribution at run time.
`bimodal`, `lognormal`, `normal`, and `powerlaw` follow the same structure with
distribution-specific parameters in the `output` block. See {doc}`../workload/index`.

### `dapo_trace`

```{literalinclude} ../../srl/config/defaults/workload/dapo_trace.yaml
:language: yaml
```

`source: trace` reads pre-collected `(prompt, n responses)` groups from a JSONL artifact.
`trace.path` is the merged joint trace produced by `examples/workload/collect_joint_trace.sh`.
`n_groups: 0` reads all groups. Set a positive value to cap the trace.
