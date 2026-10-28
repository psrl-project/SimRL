# Configuration reference

`srl` is a [Hydra](https://hydra.cc/) app. The primary config is
`srl/config/defaults/simulate.yaml`. Four config groups, each a directory of YAML files under
`srl/config/defaults/`, supply the pieces that vary: `policy`, `rollout_cost`, `training_cost`
and `workload`. The remaining blocks (`job`, `rollout`, `sync_cost`, `recompute_cost` and a few
scalars) live in `simulate.yaml` itself.

```bash
srl policy=verl rollout_cost=roofline training_cost=fixed workload=bimodal job.batch_size=32
```

How the pieces fit:

- The composed config is in Hydra struct mode, so an unknown key is an error. A key that exists
  in no YAML or schema is appended with a leading plus sign, as `+log_exclude` is.
- `srl/config/schema.py` holds the typed dataclasses that mirror these blocks, and
  `register_configs()` stores them in Hydra's ConfigStore. Only the `policy` group is validated
  against a dataclass of the same name, and that is also where policy keys that have no YAML line
  come from. The other groups are validated by the YAML alone, so a value of the wrong type is not
  rejected at composition.
- `srl --help` lists the options of each group. Entries ending in `_schema` are the ConfigStore
  nodes and are not meant to be selected.

Behaviour behind these keys is described in {doc}`/pipeline/simulation/policies`,
{doc}`/pipeline/simulation/routing`, {doc}`/pipeline/simulation/domain-gates` and {doc}`/pipeline/workload/index`.

---

## `simulate.yaml`

```{literalinclude} ../../srl/config/defaults/simulate.yaml
:language: yaml
```

Notes that the comments do not carry:

`job.batch_size` and `job.group_size`
: `batch_size` counts prompt groups per training step, so responses per step is
  `batch_size * group_size`. `group_size: null` adopts the workload's arity. A synthetic
  distribution workload has group size 1. An explicit value that disagrees with the workload is an
  error.

`rollout.kv_blocks` and `rollout.kv_cache_sizing`
: With `kv_cache_sizing.model_path` set, the block count comes from the model instead of the
  literal. `probe: true` starts a real vLLM engine on a cache miss and caches the answer (see
  `srl-profile kv-probe` in {doc}`/reference/cli`). `probe: false` computes it from `config.json` and the
  checkpoint's safetensors index, which is only reliable for dense models. `max_model_len` is
  taken as `job.max_prompt_length + job.max_output_length` and is 32768 when either cap is null.
  The engine's dtype is `rollout_cost.dtype` when that key is appended and `bfloat16` otherwise.

`rollout.router.strategy`
: One of `request_num_balance`, `round_robin`, `random`, `kv_cache_aware`, `prefix_aware`,
  `throughput_balance`, `length_aware` or `inventory_deadline`. The commented list in
  `simulate.yaml` omits `inventory_deadline`, which needs `policy=psrl`.

`rollout.router.eviction`
: One of `vllm_lru`, `longest_ctx` or `latest_deadline`. `latest_deadline` needs the Python engine.

`sync_cost`
: `name` is `fixed`, the only model.   `weight_load_time` is read from this block whatever the `name`.

`recompute_cost`
: `name` is `fixed`, `analytical` or `training_forward`. `training_forward` reads the model's
  hidden size, vocabulary size and parameter count from the training artifact, or from the
  checkpoint named in `rollout_cost.model_path` or `rollout.kv_cache_sizing.model_path` when the
  training model has no artifact. The `hidden_size`, `vocab_size` and `parameter_count` keys at the
  end of the YAML block are not read by any code. The geometry has no configured source by design.

`log_telemetry` and `log_exclude`
: With `log_telemetry=false` nothing is written under `logs/` and no plots are drawn.
  `log_exclude` takes a comma-separated string of stream names or prefixes to skip, for example
  `+log_exclude='instance,rollout_step'`.

---

## `rollout_cost` group

Selects the rollout latency model. The default is `roofline`. Every option carries the
`parallelism` block (`tp`, `pp`, `dp`, `cp`, all 1 by default). The loader passes it to
`from_artifact`, and `composed` reads `tp`.

When `artifact_path` is set, `roofline`, `psrl` and `distserve` load the fitted coefficients from
that `CostArtifact` and ignore the constants in the YAML. `psrl` has no constants and requires an
artifact.

### `roofline`

```{literalinclude} ../../srl/config/defaults/rollout_cost/roofline.yaml
:language: yaml
```

`F`, `W` and `G` are the floor, per-step and per-token terms. `A_p` and `A_d` are the prefill and
decode attention coefficients, and `b` is the KV block size used for the attention features. Fit
them with `srl-regress rollout --model roofline`.

### `psrl`

```{literalinclude} ../../srl/config/defaults/rollout_cost/psrl.yaml
:language: yaml
```

The two-stage fitted model from `srl-regress rollout --model psrl`. It always needs an artifact.

### `distserve`

```{literalinclude} ../../srl/config/defaults/rollout_cost/distserve.yaml
:language: yaml
```

The DistServe form with architecture constants `h` and `m` and coefficients `C1` to `C5`. The YAML
values are placeholders. Fit real ones with `srl-regress rollout --model distserve`.

### `composed`

```{literalinclude} ../../srl/config/defaults/rollout_cost/composed.yaml
:language: yaml
```

Sums measured per-kernel latencies and needs no regression step. `kernel_table_root`, `gpu` and
`model_path` are required and an empty one raises.

`kernel_table_root`
: The directory that contains `<gpu>/<family>/<backend>/<version>/*_perf.parquet`, which is the
  `kernel` directory a `srl-profile kernels run` writes.

`gpu`
: GPU name or slug that selects the subdirectory, and the bundled hardware spec.

`model_path`
: Checkpoint used to derive the per-layer kernel shapes at `parallelism.tp`.

`attention_backend`
: Force a backend such as `vllm` or `flashinfer`. Empty uses the backend vLLM's own selector would
  pick.

`utility_surface`
: `pchip` or `shepard`.

`decode_integration_stride`
: Lattice stride of the decode segment integrator. 1 is an exact step-by-step sum.

`saturation_step_threshold`
: Per-request KV depth above which a decode segment uses the closed form instead of the lattice.
  The default is the largest int64, which disables it.

---

## `training_cost` group

Selects the training latency and memory model. The default is `fixed`. Every option carries the
`parallelism` block, `micro_batch_size`, `schedule`, `activation_recompute`, `optimizer` and
`gpu_memory_mib`. A non-null `gpu_memory_mib` makes the simulator raise when the predicted peak
memory exceeds it.

`fixed`, `analytical` and `linear` need nothing but their constants. `hydraulis` and `fsdp` load a
`CostArtifact` from `artifact_path` and refuse queries outside the profiled feature ranges unless
the feature is listed in `extrapolate`. An unknown name in `extrapolate` raises at construction,
and every permitted exceedance is recorded in the `train` stream's `extrapolation_notes`.

`ComposedTraining` is not selectable through this group. See {doc}`/reference/limitations`.

### `fixed`

```{literalinclude} ../../srl/config/defaults/training_cost/fixed.yaml
:language: yaml
```

### `analytical`

```{literalinclude} ../../srl/config/defaults/training_cost/analytical.yaml
:language: yaml
```

### `linear`

```{literalinclude} ../../srl/config/defaults/training_cost/linear.yaml
:language: yaml
```

### `hydraulis`

The Megatron model. `max_tokens_per_microbatch` switches microbatching from a fixed sequence
count to a token budget and is the already-multiplied value (`ppo_max_token_len_per_gpu` times the
sequence-parallel size).

```{literalinclude} ../../srl/config/defaults/training_cost/hydraulis.yaml
:language: yaml
```

### `fsdp`

The FSDP2 and HSDP model. `shard_size` and `replicate_size` pick which profiled mesh to answer for,
and `parallelism.dp` must equal their product. `optimizer_offload` and `reshard_after_forward`
must match a profiled cell.

`batching_mode`
: `pad_to_max` charges padding, `token_budget` reproduces dynamic batching with
  `max_tokens_per_rank`, and `packed_no_pad` charges only real tokens.

```{literalinclude} ../../srl/config/defaults/training_cost/fsdp.yaml
:language: yaml
```

---

## `policy` group

Selects the system preset. The default is `psrl`. Each option is a typed dataclass under
`srl/config/policies/` plus a YAML file of the same name. The code path from a selection to a
running policy is:

1. Hydra composes the dataclass defaults with the YAML into a plain node.
2. `preset_config` rebuilds the dataclass from that node, which restores the methods Hydra drops.
3. `preset.build(max_inflight=rollout.max_inflight)` constructs the `StalenessPolicy`.
4. `preset.horizon` becomes `SimConfig.staleness_horizon`, the bound policies read as
   `SimState.max_staleness`. `None` means the system declares no bound.
5. `preset.ingress_window_groups(batch_size, group_size, n_instances)` becomes the number of prompt
   groups the dataset may feed into the router before any training step publishes. It is a flow
   limit and not a staleness bound. Only `slime` reads the extra two arguments.

| Preset | Keys (defaults) | `horizon` | Ingress window, in groups |
|---|---|---|---|
| `psrl` | `staleness` (2), `proactive_filter` (false), `proactive_threshold` (0) | `staleness` | `(staleness + 1) * batch_size` |
| `areal` | `max_head_offpolicyness` (0) | the same key | `(2 * max_head_offpolicyness + 1) * batch_size` |
| `roll` | `async_generation_ratio` (1.0) | `ceil(ratio)` | `(ceil(ratio) + 2) * batch_size` |
| `slime` | `server_concurrency` (64), `group_prefetch_factor` (1), `flush_cache_interval` (1) | none | `2 * max(threshold, batch_size) + batch_size`, with `threshold` as `max(1, server_concurrency * n_instances // group_size) * group_prefetch_factor` |
| `verl` | `staleness_threshold` (2.0), `require_batches` (1) | `ceil(threshold)` | `(ceil(threshold) + 1) * batch_size` |
| `verl_v1` | `num_warmup_batches` (1.0), `max_off_policy_threshold` (8), `max_off_policy_strategy` (drop), `pause_mode` (abort) | `max_off_policy_threshold` | `(max(h, ceil(num_warmup_batches)) + 2) * batch_size`, with `h` the threshold or 1 |
| `skyrl` | `max_staleness_steps` (4), `pause_mode` (in_place) | `max_staleness_steps` | `(max_staleness_steps + 2) * batch_size` |
| `openrlhf` | `async_queue_size` (1), `partial_rollout_enable` (false) | none | `(async_queue_size + 1) * batch_size` |
| `dressage` | `keep_versions` (0), `max_active_groups` (null), `output_queue_size` (1000), `max_partial_rollout_preempts` (null), `pause_mode` (in_place) | `keep_versions`, none when 0 | `cap + 2 * batch_size`, with `cap` as `max_active_groups` or `batch_size` |
| `nemo_rl` | `sampler` (in_order), `max_staleness_versions` (1), `max_lookahead_versions` (1), `sample_freshest_first` (false), `max_inflight_prompts` (32), `max_buffered_rollouts` (64), `pause_mode` (abort) | `max_staleness_versions` for `windowed` and `weight_fifo`, `max_lookahead_versions` for `in_order`, none for `ready_first` | `max_buffered_rollouts + batch_size` |
| `trl` | `max_staleness` (4), `max_inflight_tasks` (-1), `pause_mode` (in_place) | `max_staleness` | `cap + batch_size`, with `cap` as `max_inflight_tasks` or `max(max_staleness, 1) * batch_size` when negative |
| `prime_rl` | `max_off_policy_steps` (8), `max_inflight` (1024), `target_lag` (1), `action` (drop) | `max_off_policy_steps` | `min(max_inflight, max((target_lag + 1) * batch_size, (max_off_policy_steps + 2) * batch_size))` |

`rollout.max_inflight` is forwarded to the presets whose policy takes a cap on concurrently
generating groups. `prime_rl` has its own `policy.max_inflight` instead. `pause_mode` is `abort`
(a weight publish interrupts and re-enqueues every request, losing its KV) or `in_place`
(requests keep running under the new weights).

`psrl` is built on the external `psrl` package's staleness inventory. The `inventory_deadline`
router strategy and the forecast modules also require `policy=psrl`. Defaults for each preset are
the YAML files in `srl/config/defaults/policy/`.

---

## `workload` group

Selects where request lengths come from. The default is `uniform`.

| Option | `source` | Content |
|---|---|---|
| `uniform`, `bimodal`, `lognormal`, `normal`, `powerlaw` | `distributions` | Prompt and output lengths sampled independently from a named distribution |
| `dapo_trace` | `trace` | Joint `(prompt, n responses)` groups read from a trace file |

With `source: distributions` the `output` and `prompt` blocks each take `name` (`uniform`,
`bimodal`, `lognormal`, `normal`, `powerlaw` or `fixed`), `n` and `seed`, plus the parameters of
that distribution. The prompt block defaults to a fixed length of 512.

```{literalinclude} ../../srl/config/defaults/workload/uniform.yaml
:language: yaml
```

`bimodal` takes `short_len`, `long_len` and `long_frac`. `lognormal` takes `mu` and `sigma` of the
underlying normal. `normal` takes `mu`, `sigma` and a lower bound `lo`. `powerlaw` takes `alpha`,
`lo` and `hi`. Per-request caps are `job.max_prompt_length` and `job.max_output_length`, not
part of the distribution.

With `source: trace` the run reads groups from a trace written by `srl-workload` (see
{doc}`/pipeline/workload/trace-collection`). The group arity comes from the trace.

```{literalinclude} ../../srl/config/defaults/workload/dapo_trace.yaml
:language: yaml
```

`trace.path` is required and `trace.n_groups: 0` reads every group.
