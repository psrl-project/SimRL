# Schemas

Three on-disk contracts hold everything between the measurement and the simulation layers.

---

## `ProfileRecord`: the JSONL profiling boundary

Defined in `srl/profiler/schema.py`. Every line of a profiling JSONL file is one
`ProfileRecord` serialised with `json.dumps(record.to_dict())`.

### Common envelope

Every record carries these top-level fields:

`schema_version`
: Currently `"1.0"`. Records with a different version are rejected by `ProfileRecord.from_dict`.

`record_type`
: `"rollout"` or `"training"`. Must match the payload type.

`run`
: A `RunDescriptor` with `run_id` (non-empty string), `repeat` (non-negative int),
  `status` (`"success"`, `"oom"`, `"failed"`, or `"skipped"`), `started_at` (ISO 8601
  UTC), `duration_s` (float or null), `raw_artifact` (path or null), and `error` (string
  or null).

`model`
: A `ModelDescriptor` with `name`, optional `revision`, `architecture`, `parameter_count`,
  `num_layers`, `hidden_size`, `num_attention_heads`, `num_query_groups`, and an `extra`
  dict for additional architecture fields.

`hardware`
: A `HardwareDescriptor` with `gpu_name`, `gpu_count`, `gpu_memory_mib`, `node_count`,
  `gpus_per_node`, `interconnect`, and `extra`.

`software`
: A `SoftwareDescriptor` with `backend`, `backend_version`, `source_commit`,
  `python_version`, `torch_version`, `cuda_version`, `command` (list of strings), and
  `extra`.

`parallelism`
: A `Parallelism` with `tp`, `pp`, `dp`, `cp`, and `world_size`. Writer invariant:
  `world_size == tp * pp * dp * cp`.

`payload`
: Either a `RolloutPayload` or a `TrainingPayload` (see below).

`tags`
: A flat `dict[str, str]` for arbitrary key/value annotations written by the adapter.

### `RolloutPayload`

Written by the vLLM adapter for each scheduler batch-step observation.

| Field | Type | Meaning |
|---|---|---|
| `phase` | `"prefill"`, `"decode"`, or `"mixed"` | Batch composition |
| `request_count` | int | Number of requests in the batch |
| `input_tokens` | int | Total input tokens |
| `output_tokens` | int | Total output tokens generated this step |
| `n_tokens` | int | Total tokens processed (input + output) |
| `sum_l` | float | Sum of current sequence lengths |
| `sum_l2` | float | Sum of squared lengths (prefill attention feature) |
| `ctxsum` | float | Sum of context depths (decode attention feature) |
| `latency_s` | float | Wall-clock step latency in seconds |
| `ttft_s` | float[] | Per-request time-to-first-token (optional) |
| `itl_s` | float[] | Per-request inter-token latencies (optional) |
| `request_input_lengths` | int[] | Per-request input lengths (optional) |
| `request_output_lengths` | int[] | Per-request output lengths (optional) |
| `source_request_ids` | int[] | Benchmark-assigned request IDs (optional) |
| `batch_step_index` | int or null | Step index within the run |

### `TrainingPayload`

Written by the Megatron and FSDP adapters for each profiled training candidate.

Additional writer invariant: `global_batch_size == micro_batch_size * dp * num_microbatches`.

| Field | Type | Meaning |
|---|---|---|
| `micro_batch_size` | int | Sequences per microbatch |
| `global_batch_size` | int | Total sequences across all ranks |
| `num_microbatches` | int | Gradient-accumulation steps per rank |
| `schedule` | str | `"1f1b"` or `"fsdp"` |
| `recompute` | str | Activation recompute policy (e.g. `"full"`, `"none"`) |
| `optimizer` | str | Optimizer name (e.g. `"adam"`) |
| `microbatches` | `SequenceStats[]` | One stats struct per microbatch |
| `step_times_s` | float[] | Per-step latencies (all steps, including warmup) |
| `timers_s` | dict | Named timer arrays (optional, Megatron-specific) |
| `rank_memory` | `RankMemory[]` | Per-rank allocated/reserved peaks |
| `peak_allocated_bytes` | int or null | Max allocated across ranks |
| `peak_reserved_bytes` | int or null | Max reserved across ranks |
| `parameter_dtype` | str | Compute parameter dtype (e.g. `"bf16"`) |
| `gradient_dtype` | str | Gradient dtype |
| `optimizer_state_dtype` | str | Optimizer state dtype |
| `distributed_optimizer` | bool | Whether optimizer state is DP-sharded (Megatron) |
| `parameter_count` | int or null | Dense model parameter count |
| `warmup_steps` | int | Number of warmup steps at the start of `step_times_s` |

`SequenceStats` holds `count`, `sum_tokens`, `sum_squared_tokens`, `max_tokens`, and an
optional `lengths` tuple. `RankMemory` holds `rank`, `allocated_bytes`, and
`reserved_bytes` (invariant: `reserved_bytes >= allocated_bytes`).

### Compact example (rollout)

```javascript
{
  "schema_version": "1.0",
  "record_type": "rollout",
  "run": {
    "run_id": "vllm_tp1_decode_b16_s512_0",
    "repeat": 0,
    "status": "success",
    "started_at": "2026-08-25T10:00:00+00:00",
    "duration_s": 0.045
  },
  "model": {
    "name": "Qwen3-8B",
    "architecture": "Qwen3ForCausalLM",
    "parameter_count": 8190735360,
    "num_layers": 36,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_query_groups": 8
  },
  "hardware": {
    "gpu_name": "H20",
    "gpu_count": 1,
    "gpu_memory_mib": 97871
  },
  "software": {
    "backend": "vllm",
    "backend_version": "2.11.0+cu128"
  },
  "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1, "world_size": 1},
  "payload": {
    "phase": "decode",
    "request_count": 16,
    "input_tokens": 0,
    "output_tokens": 16,
    "n_tokens": 16,
    "sum_l": 8208.0,
    "sum_l2": 0.0,
    "ctxsum": 8208.0,
    "latency_s": 0.0312
  },
  "tags": {"sweep": "tp1"}
}
```

---

## `CostArtifact`: the regression output

Defined in `srl/cost/artifact.py`. Written by `srl-regress` and read by the simulator.

### Outer envelope

| Field | Type | Notes |
|---|---|---|
| `artifact_version` | str | Currently `"1.0"` |
| `created_at` | str | ISO 8601 UTC timestamp of when the artifact was written |
| `kind` | str | `"rollout"` or `"training"` |
| `model_type` | str | Formula name: `"roofline"`, `"psrl"`, `"distserve"`, `"hydraulis"`, `"fsdp"` |
| `profile_digest` | str | SHA-256 of the input JSONL, so re-fitting a changed input produces a new digest |
| `hardware` | dict | Copy of the `HardwareDescriptor` from the profiling records |
| `model` | dict | Copy of the `ModelDescriptor` |
| `diagnostics` | dict | Fit-level diagnostics |
| `entries` | `ArtifactEntry[]` | One per fitted cell |

`entry_for(parallelism, selector=...)` performs an exact match and raises `KeyError` if
zero or more than one entry matches. There is no silent fallback to a neighbour cell.

### `ArtifactEntry`

| Field | Type | Notes |
|---|---|---|
| `parallelism` | dict | TP/PP/DP/CP and world_size |
| `selector` | dict | Discrete cell keys beyond parallelism |
| `coefficients` | dict[str, float] | Named regression weights |
| `feature_ranges` | dict[str, [min, max]] | Valid input domain per feature |
| `train_metrics` | `FitMetrics` | In-sample fit quality |
| `validation_metrics` | `FitMetrics` | Held-out quality, or zero-sample if in-sample |
| `diagnostics` | dict | Entry-level diagnostics including `validation_is_in_sample` |

`FitMetrics` fields: `r_squared`, `rmse`, `mre_percent`, `max_error`, `n_samples`,
`n_outliers`, `design_rank`.

### Compact example (rollout / roofline)

```javascript
{
  "artifact_version": "1.0",
  "kind": "rollout",
  "model_type": "roofline",
  "profile_digest": "sha256:9b7f45b7d...",
  "hardware": {"gpu_name": "H20", "gpu_count": 1, "gpu_memory_mib": 97871, ...},
  "model": {"name": "Qwen3-8B", "parameter_count": 8190735360, ...},
  "entries": [
    {
      "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1, "world_size": 1},
      "selector": {"block_size": 16},
      "coefficients": {
        "F": 0.006247, "W": 0.005954, "G": 0.000101,
        "A_p": 3.40e-8, "A_d": 7.63e-8, "b": 16.0
      },
      "feature_ranges": {
        "n_tokens":      [1.0, 65536.0],
        "request_count": [1.0, 64.0],
        "sum_l2":        [0.0, 536870912.0],
        "ctxsum":        [0.0, 466089.0]
      },
      "train_metrics": {
        "r_squared": 0.99989, "mre_percent": 5.252, "n_samples": 49208, "design_rank": 4
      },
      "validation_metrics": {"n_samples": 0, ...},
      "diagnostics": {"validation_is_in_sample": true}
    }
  ]
}
```

### Compact example (training / FSDP latency cell)

```javascript
{
  "artifact_version": "1.0",
  "kind": "training",
  "model_type": "fsdp",
  "entries": [
    {
      "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1, "world_size": 1},
      "selector": {
        "component": "latency",
        "schedule": "fsdp",
        "shard_size": 8,
        "replicate_size": 1,
        "recompute": "full",
        "optimizer": "adam",
        "parameter_dtype": "bf16",
        "gradient_dtype": "fp32",
        "optimizer_state_dtype": "fp32",
        "optimizer_offload": false,
        "reshard_after_forward": true
      },
      "coefficients": {
        "per_step_overhead": 0.089,
        "token_cost": 0.000601,
        "microbatch_overhead": 0.014
      },
      "feature_ranges": {
        "tokens":              [512.0, 16384.0],
        "num_microbatches":    [1.0, 4.0],
        "sequences_per_microbatch": [1.0, 1.0]
      },
      "train_metrics": {
        "r_squared": 0.9729, "mre_percent": 10.65, "n_samples": 22, "design_rank": 3
      },
      "validation_metrics": {
        "r_squared": 0.9900, "mre_percent": 5.615, "n_samples": 5
      },
      "diagnostics": {
        "validation_fraction": 0.2,
        "validation_is_in_sample": false,
        "dp_latency": "ideal-local-workload-scaling"
      }
    }
  ]
}
```

---

## Telemetry streams

The simulator writes four JSONL streams under `{output_dir}/logs/` when `log_telemetry: true`.
The streams are the primary output of a simulation run, and `srl-plot` reads them to produce
figures without re-simulating. Schema details (field names, units, and the meaning of each event
type) are documented in {doc}`../simulation/outputs`, which owns the telemetry topic.

| File | One row per |
|---|---|
| `instance.jsonl` | Rollout engine step (engine iteration) |
| `rollout_step.jsonl` | Completed or dropped request |
| `train.jsonl` | Training step |
| `request.jsonl` | Request lifecycle event |
