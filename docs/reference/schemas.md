# Schemas

Four on-disk contracts connect the layers: the versioned profile JSONL that adapters write and
fitters read, the `CostArtifact` JSON that fitters write and the simulator reads, the kernel perf
tables that the kernel profiler writes and the composed cost models read, and the telemetry streams
the simulator writes. The first three are defined in code and validated on every read and write.

---

## `ProfileRecord`: the JSONL profiling boundary

Defined in `srl/profiler/schema.py`. Each line of a profile file is one record, serialised by
`write_records` in `srl/profiler/io.py` as canonical JSON (sorted keys, compact separators).
`write_records` replaces the destination atomically unless `append=True`. `read_records` and
`iter_records` re-validate every line and name the file and line number on failure.
`profile_digest` is a SHA-256 over the canonical lines and is what a `CostArtifact` records as its
provenance.

### Envelope

`schema_version`
: `"1.0"`. Any other value is rejected.

`record_type`
: `"rollout"` or `"training"`. It must match the type of `payload`.

`run`
: A `RunDescriptor`. `run_id` (non-empty), `repeat` (non-negative), `status` (`success`, `oom`,
  `failed` or `skipped`), `started_at` (ISO 8601 UTC), `duration_s`, `raw_artifact` and `error`
  (nullable).

`model`
: A `ModelDescriptor`. `name` (non-empty) and optional `revision`, `architecture`,
  `parameter_count`, `num_layers`, `hidden_size`, `num_attention_heads`, `num_query_groups`, plus an
  `extra` dict that carries further config keys such as `head_dim`, `intermediate_size`,
  `vocab_size` and the keys that make an unsupported architecture refusable. Counts must be positive.

`hardware`
: A `HardwareDescriptor`. `gpu_name`, `gpu_count`, `gpu_memory_mib`, `node_count`,
  `gpus_per_node`, `interconnect` and `extra`.

`software`
: A `SoftwareDescriptor`. `backend` and `backend_version` (both non-empty), `source_commit`,
  `python_version`, `torch_version`, `cuda_version`, `command` (list of strings) and `extra`.

`parallelism`
: A `Parallelism`. `tp`, `pp`, `dp`, `cp`, `ep` (all positive, default 1) and `world_size`.

`payload`
: A `RolloutPayload` or a `TrainingPayload`.

`tags`
: A flat `dict[str, str]` the adapter fills. Training records carry the discrete cell keys here
  (for FSDP, `shard_size`, `replicate_size`, `optimizer_offload` and `reshard_after_forward`).

Invariants enforced at construction:

- `world_size == tp * pp * dp * cp`. `ep` is not a factor, because expert groups are carved out
  of the data-parallel dimension.
- When `ep > 1` and `dp > 1`, `world_size` must be divisible by `ep * pp`.
- `parallelism.world_size == hardware.gpu_count`.
- For training records, `global_batch_size == micro_batch_size * dp * num_microbatches`.

### `RolloutPayload`

One scheduler iteration of a vLLM server.

| Field | Type | Meaning |
|---|---|---|
| `phase` | `prefill`, `decode` or `mixed` | Batch composition |
| `request_count` | int, positive | Requests in the iteration |
| `input_tokens` | int | Prompt tokens processed |
| `output_tokens` | int | Tokens generated |
| `n_tokens` | int, positive | Tokens in the iteration |
| `sum_l` | float | Sum of sequence lengths |
| `sum_l2` | float | Sum of squared lengths, the prefill attention feature |
| `ctxsum` | float | Sum of context depths, the decode attention feature |
| `latency_s` | float | Iteration latency in seconds |
| `ttft_s`, `itl_s` | float arrays | Optional per-request latencies |
| `request_input_lengths`, `request_output_lengths` | int arrays | Optional, one entry per request when present |
| `source_request_ids` | int array | Optional benchmark request ids |
| `batch_step_index` | int or null | Iteration index within the run |

### `TrainingPayload`

One profiled training candidate.

| Field | Type | Meaning |
|---|---|---|
| `micro_batch_size` | int | Sequences per microbatch |
| `global_batch_size` | int | Sequences per step across all ranks |
| `num_microbatches` | int | Microbatches per rank per step |
| `schedule` | `1f1b` or `fsdp` | Pipeline schedule. `fsdp` marks a backend with no pipeline |
| `recompute`, `optimizer` | str | Discrete cell keys, for example `full` and `adam` |
| `microbatches` | `SequenceStats[]` | One entry per microbatch |
| `step_times_s` | float array | Per-step wall time, warmup steps included |
| `timers_s` | dict of float arrays | Optional named timers |
| `rank_memory` | `RankMemory[]` | Per-rank allocated and reserved peaks |
| `peak_allocated_bytes`, `peak_reserved_bytes` | int or null | When `rank_memory` is present they must equal its maximum |
| `parameter_dtype`, `gradient_dtype`, `optimizer_state_dtype` | str | Default `bf16`, `fp32`, `fp32` |
| `distributed_optimizer` | bool | Whether optimizer state is sharded across data parallel ranks |
| `parameter_count` | int or null | Model parameter count |
| `warmup_steps` | int | Leading steps of `step_times_s` to drop when fitting |
| `packing` | `bshd` or `thd` | Sequence layout the step was measured under |

`packing` matters because the two layouts price attention differently. BSHD charges the padded
length of each sequence and THD charges the sum of per-sequence squares, so a fit must not mix
them. Under `bshd` every microbatch has exactly `micro_batch_size` sequences. A `thd` microbatch is
one packed sample whatever its sequence count.

`SequenceStats` holds `count`, `sum_tokens`, `sum_squared_tokens`, `max_tokens` and an optional
`lengths` tuple. When `lengths` is present the other four are checked against it. `RankMemory`
holds `rank`, `allocated_bytes` and `reserved_bytes`, with `reserved_bytes >= allocated_bytes`.

### Example (rollout, abridged)

```javascript
{
  "schema_version": "1.0",
  "record_type": "rollout",
  "run": {"run_id": "vllm-097f7b34d69c83cee6e0", "repeat": 0, "status": "success",
          "started_at": "2026-08-18T16:37:15.855992+00:00", "duration_s": 0.29,
          "raw_artifact": "raw/vllm_timing/...", "error": null},
  "model": {"name": "Qwen3-8B", "architecture": "Qwen3ForCausalLM", "parameter_count": 8190735360,
            "num_layers": 36, "hidden_size": 4096, "num_attention_heads": 32,
            "num_query_groups": 8, "extra": {"head_dim": 128, "vocab_size": 151936}},
  "hardware": {"gpu_name": "H20", "gpu_count": 1, "gpu_memory_mib": 97871, "node_count": 1,
               "gpus_per_node": 1, "interconnect": "PCIe", "extra": {}},
  "software": {"backend": "vllm", "backend_version": "0.26.0",
               "command": ["vllm", "bench", "sweep", "serve"], "extra": {}},
  "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1, "ep": 1, "world_size": 1},
  "payload": {"phase": "prefill", "request_count": 1, "input_tokens": 256, "output_tokens": 0,
              "n_tokens": 256, "sum_l": 256.0, "sum_l2": 65536.0, "ctxsum": 0.0,
              "latency_s": 0.02506, "batch_step_index": 0},
  "tags": {"phase": "prefill", "source": "vllm-scheduler-iteration-log"}
}
```

---

## `CostArtifact`: the regression output

Defined in `srl/cost/artifact.py`. `srl-regress` writes it with an atomic `save` and the
simulator reads it with `CostArtifact.load`.

### Envelope

| Field | Type | Notes |
|---|---|---|
| `artifact_version` | str | `"1.0"`. Any other value raises |
| `created_at` | str | ISO 8601 UTC |
| `kind` | str | `rollout` or `training`. Anything else raises |
| `model_type` | str | `roofline`, `psrl`, `distserve`, `hydraulis` or `fsdp`. `ComposedRollout` declares `composed`, but nothing writes an artifact of that type, because a kernel sum has no coefficients to record |
| `profile_digest` | str | `sha256:` over the input JSONL |
| `hardware` | dict | The profile's `HardwareDescriptor` fields. `gpu_count` is the list of distinct counts seen |
| `model` | dict | The profile's `ModelDescriptor` fields |
| `diagnostics` | dict | Fit-level notes, for example the supported schedules and mesh axes |
| `entries` | `ArtifactEntry[]` | At least one. One per fitted cell |

`entry_for(parallelism, *, selector=None, ignore_pp=False)` returns the single entry whose
parallelism and selector match exactly, and raises `KeyError` when zero or several match. There is
no nearest-cell fallback. With `ignore_pp=True` the match uses `tp`, `cp` and `ep` only, which is
how the training latency form handles a pipeline depth it derives instead of looks up.

### `ArtifactEntry`

| Field | Type | Notes |
|---|---|---|
| `parallelism` | dict | `tp`, `pp`, `dp`, `cp`, `ep`, `world_size` |
| `selector` | dict | Discrete keys beyond parallelism, such as `component`, `schedule`, `recompute`, `optimizer`, the dtypes and the FSDP mesh |
| `coefficients` | dict of float | Named fitted weights |
| `feature_ranges` | dict of `[min, max]` | Domain each feature was fitted over. The domain gate reads it |
| `train_metrics` | `FitMetrics` | Fit quality on the training split |
| `validation_metrics` | `FitMetrics` | Held-out quality. `n_samples: 0` means nothing was held out |
| `diagnostics` | dict | Includes `validation_is_in_sample` for the training models |

`FitMetrics` holds `r_squared`, `rmse`, `mre_percent`, `max_error`, `n_samples`, `n_outliers` and
`design_rank`.

### Example (rollout roofline, abridged)

```javascript
{
  "artifact_version": "1.0",
  "kind": "rollout",
  "model_type": "roofline",
  "profile_digest": "sha256:9b7f45b7d83d02ef...",
  "hardware": {"gpu_name": "H20", "gpu_count": [1], "gpu_memory_mib": 97871, "...": "..."},
  "model": {"name": "Qwen3-8B", "parameter_count": 8190735360, "...": "..."},
  "entries": [
    {
      "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1, "world_size": 1},
      "selector": {"block_size": 16},
      "coefficients": {"F": 0.00625, "W": 0.00595, "G": 0.000101,
                       "A_p": 3.40e-08, "A_d": 7.63e-08, "b": 16.0},
      "feature_ranges": {"n_tokens": [1.0, 65536.0], "request_count": [1.0, 64.0],
                         "sum_l2": [0.0, 536870912.0], "ctxsum": [0.0, 466089.0]},
      "train_metrics": {"n_samples": 49208, "mre_percent": 5.25, "design_rank": 4, "...": "..."},
      "validation_metrics": {"n_samples": 0, "...": "..."},
      "diagnostics": {"fit_method": "nonnegative-soft-l1", "grouped_holdout": true}
    }
  ]
}
```

### Example (FSDP latency cell, abridged)

```javascript
{
  "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1, "world_size": 1},
  "selector": {
    "component": "latency", "schedule": "fsdp", "recompute": "full", "optimizer": "adam",
    "shard_size": 8, "replicate_size": 1,
    "parameter_dtype": "bf16", "gradient_dtype": "fp32", "optimizer_state_dtype": "fp32",
    "optimizer_offload": false, "reshard_after_forward": true
  },
  "coefficients": {"per_step_overhead": 0.0276, "token_cost": 0.000467,
                   "microbatch_overhead": 0.2246, "token_quadratic_cost": 1.48e-08},
  "feature_ranges": {"tokens": [512.0, 16384.0], "num_microbatches": [1.0, 4.0],
                     "sequences_per_microbatch": [1.0, 4.0]},
  "train_metrics": {"n_samples": 22, "design_rank": 4, "...": "..."},
  "validation_metrics": {"n_samples": 5, "design_rank": 4, "...": "..."},
  "diagnostics": {
    "validation_fraction": 0.2,
    "validation_is_in_sample": false,
    "...": "..."
  }
}
```

The Megatron (`hydraulis`) latency cell uses the same four coefficients with `pipeline_slots` in
place of `num_microbatches`. The authoritative form is `latency_from_coefficients` in
`srl/cost/training/latency.py`, not the `formula` string a `diagnostics` block may carry, which
can lag the code. Memory cells have `component: "memory"` and different coefficients.

---

## Kernel perf tables

`srl-profile kernels run` writes one parquet table per kernel op. They are not `ProfileRecord`
JSONL, because a kernel table prices one shape and carries no workload. `PerfTableWriter`
(`srl/profiler/kernel/measurements.py`) writes and `KernelTable` (`srl/cost/kernel/table.py`)
reads.

```text
<root>/<gpu>/<family>/<backend>/<version>/<op>_perf.parquet
<root>/<gpu>/<family>/<backend>/<version>/floors.json
```

`<gpu>` is the GPU slug (`h20`), `<family>` and `<op>` come from the op registry, `<backend>`
names the kernel provider (`vllm`, `flashinfer`, `nccl`, `megatron`), and `<version>` is that
provider's version. A table has these columns, in this order:

- Provenance: `framework`, `version`, `device`, `op_name`, `kernel_source`.
- The op's slice columns, matched exactly when a query is priced.
- The op's axis columns, interpolated in log2 space.
- `latency_s`, last.

Writes are atomic. A reader raises `KernelCoverageError` for a missing file, a missing column (the
message names the `srl-profile kernels run --family` command that re-measures it), or a
non-positive latency. A repeated shape keeps its fastest measurement. Latency is stored and
utility is not, because utility depends on the GPU spec being priced against.

`floors.json` sits beside the tables a floor was fitted for. Its keys are op names with a float
dispatch floor in seconds, plus the metadata keys `fitted_mre_percent`, `candidates_us`, `gpu` and
`backend`. A file that uses the retired phase keys (`context`, `generation`) raises instead of
being read. A missing file falls back to the hard-coded `FALLBACK_FLOORS_S` and logs a warning.

`holdout` and `validate` write `holdout_report.json` and `validate_report.json` next to the
tables. Each is a JSON list of the result dataclasses in `srl/cost/kernel/holdout.py` and
`srl/cost/kernel/validate.py`.

---

## Telemetry streams

`SimTelemetry` in `srl/telemetry/writer.py` owns these schemas, and `JsonlStore` maps each stream
name to `<run>/logs/<name>.jsonl`. A name containing `/` becomes a subdirectory. The store holds
no history in memory, flushes every record, and skips any stream named by `log_exclude`. Only
committed engine state is written, never an event that a speculative step may still undo. Plots and
`srl-compare` read these files and nothing else. {doc}`/pipeline/simulation/outputs` describes how to read
them.

| Stream | One row per | Main fields |
|---|---|---|
| `instance/<id>` | Change in an instance's occupancy | `t`, `instance_id`, `tp`, `num_running_reqs`, `num_waiting_reqs`, `kv_cache_usage`, `kv_blocks_used`, `total_ctx`, `running_len`, `waiting_len` |
| `rollout_step` | Engine step record | `t`, `instance_id`, `phase`, `steps`, `n_prefill`, `n_prefill_hit`, `n_prefill_computed`, `n_decode`, `n_tokens`, `dt`, `saturated`, `saturated_steps` |
| `gateway` | Change in the router queue | `t`, `queue_len`, `pending_request_num`, `running_request_num`, `queue_by_version`, `n_routed`, `n_reroutes`, `n_cross_instance_reroutes` |
| `buffer` | Change in the staleness inventory | `t`, `train_version`, `rollout_version`, `buffers`, a list of `buffer_id`, `ready_num_entries`, `n_occupied`, `n_reserved` and `reserved_group_ids`. Written only by policies that keep an inventory |
| `request` | Lifecycle event of a request | `t`, `event`, request identity and group fields, `admit_time`, `dispatch_time`, `complete_time`, `consume_version`, `prompt_len`, `target_len`, `generated`, `ctx`, `v_traj`, `v_min`, `v_max`, `staleness_true`, `staleness_reported`, `num_preemptions`, `num_aborts`, `num_reroutes`, `reprefill_tokens`, `prefix_hit_tokens`, `drop_reason`, `retry` |
| `train` | Training step | `t`, `start`, `end`, `dt`, `version`, `n_groups`, `group_size`, `n_reqs`, `total_tokens`, `train_time`, `peak_memory_bytes`, the four training parallelism degrees, `micro_batch_size`, `schedule`, `activation_recompute`, `sync_time`, `recompute_time`, `n_inflight`, `n_ready`, `n_pending`, `n_dropped_step`, and `extrapolation_notes` when a gate was relaxed |

`request.event` is one of `enqueue`, `admit`, `complete`, `drop` or `consume`. `staleness_true`
uses the oldest weight version any token was generated under and `staleness_reported` uses the
newest. A `drop_reason` is the reason the policy supplied, or the generic `l2_cancel` or `l3_stale` when it supplied none.

`srl-psrl-plot` writes a real PSRL run into the same `instance/<id>`, `gateway` and `train` shapes,
so one plotter draws both.
