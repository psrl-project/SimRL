# Megatron-Bridge profiling

This page covers the Megatron-Bridge training sweep: what the sweep axes mean, why SimRL owns
the entry point, how to run it, what lands on disk, and what each sidecar file contains. The
formulas fitted from that output are in {doc}`../cost-models/training-megatron`. General
measurement rules, including warmup and the OOM-as-data contract, are in {doc}`methodology`.

## Why SimRL owns the entry point

The Megatron-Bridge distribution ships its own `scripts/performance` CLI, and it is not used
here. That CLI parses its own known arguments and drops the rest, so recompute mode, output paths
and profiling flags could vanish without an error.

SimRL's adapter resolves the library's published recipe
(`megatron.bridge.recipes.<family>.<name>`) and applies the candidate to the resulting
`ConfigContainer` in Python, where a wrong field raises. The torchrun child is
`srl.profiler.adapters.megatron.launch`; every setting it needs travels in `launch_spec.json`
inside the candidate's run directory, and nothing is marshalled through a CLI.

The recipe is inferred from the checkpoint's `model_type`, architecture name and parameter count,
matched against a table of known recipe sizes within a tolerance. A model whose size matches no
recipe is refused, and `model_family_name` + `model_recipe_name` in the YAML name one explicitly.
Both must be set together.

The official callback API (`megatron.bridge.training.callbacks`) brackets steps and reads
allocator peaks, so the collector never parses a log or a TensorBoard scalar.

## Quick start

```bash
export MODEL_PATH=/path/to/Qwen3-8B
CONFIG=examples/profile/megatron/qwen3_8b.yaml bash examples/profile/megatron/profile_qwen3_8b.sh generate
CONFIG=examples/profile/megatron/qwen3_8b.yaml bash examples/profile/megatron/profile_qwen3_8b.sh run
CONFIG=examples/profile/megatron/qwen3_8b.yaml bash examples/profile/megatron/profile_qwen3_8b.sh collect
```

The wrapper defaults `CANDIDATE_INDEX` to `0`, so a bare `run` cannot launch the whole sweep.
Run every candidate with `CANDIDATE_INDEX=all`, or select one by position or id through the CLI:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CANDIDATE_INDEX=all \
    bash examples/profile/megatron/profile_qwen3_8b.sh run

srl-profile megatron run --config CFG --index 42
srl-profile megatron run --config CFG --candidate-id a3f7c12d9e8b4501
```

`run` accepts `--dry-run` (prepare and print the command) and `--resume` (skip candidates whose
`status.json` says `success`). It is a Python loop over candidates, not a single `torchrun`: a
failing candidate is logged, counted and skipped so the rest of the sweep continues.

`collect` defaults its output to `{output_dir}/profiles/megatron_bridge.jsonl`. `--candidates`,
`--index` and `--candidate-id` work the same way as on `run`.

`collect` raises when a candidate marked `success` carries no step times or no rank memory, so a
launcher that failed to write its sidecar cannot become a record with no measurement. Setting
`profiling.allow_incomplete: true` turns those cases into skips and also lets `collect` skip a
candidate that was never prepared at all.

### Multi-node

A mesh wider than one host's GPU count passes `--nnodes N --node-rank K --master-addr HOST
--master-port PORT` to `srl-profile megatron run`. Every host runs the same command with its own
rank. There is no separate Slurm entry point; a Slurm job array calling the same command works.

### Smoke test

`qwen3_8b_smoke.yaml` uses a two-layer override on one GPU. Its records carry a different
`parameter_count`, and the cost-model cell key includes that count, so smoke records cannot enter
a production fit. The collector also rewrites the record's layer count and parameter count to what
actually ran, so the activation footprint is priced against the measured model.

## Sweep axes

`generate` reads `search_space:` from the YAML and enumerates candidates
(`enumerate_training_candidates`). Constraints are applied, not assumed:

- TP must divide `num_attention_heads` and be compatible with `num_query_groups`.
- PP must divide `num_layers`.
- `global_batch_size` is derived as `micro_batch_size * dp * num_microbatches`.
- `world_size` is the mesh size. With `data_parallel_sizes` absent, DP absorbs whatever TP, PP
  and CP leave of the world. Pin DP when a sweep is meant to isolate TP or PP, otherwise every
  TP or PP comparison is also a DP comparison with a different gradient-reduction volume and a
  different per-rank token count.
- `expert_parallel_sizes` defaults to `[1]`. EP does not consume ranks (Megatron carves expert
  groups out of DP) but it changes the all-to-all volume and the expert count per rank, so it is
  part of the cell key. Megatron requires `expert_parallel_size * pp` to divide the mesh, and the
  launcher pins expert tensor parallelism to 1; combinations that violate this are dropped at
  enumeration rather than raising minutes into distributed init.

Key fields:

| key | meaning |
|---|---|
| `tensor_parallel_sizes`, `pipeline_parallel_sizes`, `context_parallel_sizes`, `expert_parallel_sizes`, `data_parallel_sizes` | topology axes; CP must be 1 |
| `micro_batch_size` | sequences per microbatch; must be 1 under THD |
| `num_microbatches` | local microbatches per DP rank. Defaults to `[1, *pipeline_parallel_sizes]` when absent |
| `sequence_lengths` | one uniform sequence per microbatch entry: `[L] * micro_batch_size` |
| `sequence_shapes` | mutually exclusive with `sequence_lengths`; each entry is one *packed microbatch* |
| `recompute` (or `recompute_modes`) | default `full` |
| `packing` | `bshd` (default) or `thd` |

The example `qwen3_8b.yaml` sweeps TP `[1, 4, 8]`, PP `[1, 4]`, microbatches `[1, 4]` and five
sequence lengths. `qwen3_8b_thd.yaml` pins a TP2/PP3 mesh and sweeps THD packs.

### BSHD versus THD

- `bshd` pads every sequence in a microbatch to the longest, so attention costs
  `O(max_len^2)` per sequence. With micro-batch size one, `sum_squared_tokens` is identically
  `sum_tokens^2`, so the linear and quadratic token columns are collinear and a fit cannot tell
  "many tokens" from "long sequences".
- `thd` concatenates the microbatch behind `cu_seqlens`, as verl does with
  `use_remove_padding=True`, and costs `O(sum_i s_i^2)`. Shapes that share a token total but
  split it differently separate the two columns. A THD candidate trains one packed row per
  microbatch, so `micro_batch_size` is forced to 1.

The two layouts are different functions of the same batch. A model fitted on one does not price
the other, so fit THD measurements for a verl-style step.

### Why the microbatch count must vary

The shared latency row is `(1, weighted_tokens, weighted_slots, weighted_sum_squared_tokens)`
and must reach rank 4 ({doc}`../cost-models/training-megatron`). At a fixed microbatch count the
slot column is constant and collinear with the intercept, so any number of sequence lengths
alone caps the design below the rank the fit needs. Use at least two microbatch counts, spaced
widely, and at least three sequence lengths.

### Coverage

The fitted `feature_ranges` are hard out-of-domain gates unless a feature is listed under
`training_cost.extrapolate`. `tokens` is the **largest** microbatch, so one long sequence in a
batch sets it. Extend the sequence axis to cover the token counts the simulator will request, or
list the feature explicitly and accept the recorded extrapolation. See
{doc}`../simulation/domain-gates`.

## Steps and warmup

The YAML's `training:` block sets `max_steps` and `warmup_steps`; the shipped example uses `12`
and `4`. `max_steps` must exceed `warmup_steps`. The launcher writes every step into the sidecar
and the collector slices the warmup off, so the window can change without re-profiling. See
{doc}`methodology`.

Each step is bracketed by `on_train_step_start` and `on_train_step_end` with a
`torch.cuda.synchronize()` on each side and `time.perf_counter()` in between. No barrier is
taken, so each rank's timing reflects its own work rather than the slowest rank's. A synchronized
wall clock is used rather than CUDA events because the model has a per-step overhead term that
must include host-side optimizer work and the gradient reduction.

The launcher resets peak memory counters at train start, after the model and optimizer exist.
Resetting does not free memory, so the static pool is immediately re-counted; the construction
spike is discarded.

## Output layout

```text
{output_dir}/
├── control/megatron_bridge/candidates.json    generated candidate list
├── raw/megatron_bridge[/{experiment_name}]/
│   └── {candidate_id}/
│       ├── launch_spec.json        everything the torchrun child reads
│       ├── manifest.json           immutable record, written by prepare
│       ├── ConfigContainer.yaml    what Megatron actually resolved
│       ├── srl_metrics.json        measurements, written by rank 0
│       ├── oom_rank{N}.json        per-rank OOM markers
│       ├── status.json             success | oom | failed
│       └── run.log
└── profiles/megatron_bridge.jsonl  collected records
```

The candidate list lives under `control/`, not at the output root. The raw root gains an
`experiment_name` segment when the YAML sets one, so two experiments can share an output
directory without colliding.

### `launch_spec.json`

Written before `torchrun` starts. It carries the recipe family and name, the mesh, the batch
shape, `seq_length`, `packing`, the per-sequence lengths a THD pack uses, `train_iters`,
`warmup_steps`, `vocab_size`, the recompute settings, the dtype settings and `run_dir`. The
recompute pair is always `recompute_method: uniform` with `recompute_num_layers: 1`, which keeps
the activation peak close to linear in tokens. The spec also records
`derives_position_ids_from_input_ids` for a vision-language forward, which cannot be pipelined by
this profiler and is therefore profiled at PP=1.

### `manifest.json`

Not mutated after preparation. It records the adapter, launcher arguments, the candidate object,
the requested and effective sequence shape (padded for BSHD, concatenated for THD), the recompute
method and layer count, the layer-count override, the command, and the pinned backend version and
source commit.

### `ConfigContainer.yaml`

Rank zero serialises the resolved `ConfigContainer`, and the collector compares it against the
candidate: TP, PP, CP, `seq_length`, `micro_batch_size`, `global_batch_size`, recompute
granularity, and, for a model that declares experts, `expert_model_parallel_size` and
`expert_tensor_parallel_size = 1`. A mutation that did not stick is caught here rather than
producing a record attributed to the wrong configuration.

### `srl_metrics.json`

Written atomically by rank 0 in `on_train_end`:

- `step_times_s`: every step's wall-clock seconds, warmup included.
- `timers_s`: normally empty for Megatron. The collector accepts the field so both training
  backends share a payload shape.
- `rank_memory`: one entry per rank with `rank`, `allocated_bytes`, `reserved_bytes`,
  `static_allocated_bytes` and `local_parameter_count`. The static figure is read per rank, so a
  heavier pipeline stage's share is visible.
- `microbatch_sequence_lengths`: the shapes actually trained. Under BSHD this is
  `[seq_length] * micro_batch_size`; under THD the per-sequence lengths, padded to the TP degree
  exactly as the launcher aligns them.
- `parameter_count`: the global count, summed across the model-parallel group so it is invariant
  across TP and PP degrees.
- `static_reserved_bytes_at_train_start`: diagnostic, for confirming that the peak reset kept the
  static pool.

The listener also verifies that Megatron derived the candidate's microbatch count and raises if
it did not, since Megatron computes that count from global batch, micro batch and DP rather than
accepting it directly.

### `oom_rank{N}.json`

Each rank writes its own marker when it catches `torch.OutOfMemoryError`, with no collective
taken, because an OOM is usually local to one rank while the others are blocked. The marker
carries `rank`, `torch_oom`, `device_free_bytes`, `device_total_bytes`,
`device_reserved_bytes`, `seq_length`, `num_microbatches` and the message, truncated.

The second detection route is a `RuntimeError` naming NCCL and `unhandled cuda error` (or
`out of memory`) on a device with under 2% free memory. NCCL allocates its buffers outside the
PyTorch allocator, so exhaustion surfaces there as a plain `RuntimeError`. The free-memory reading
is what distinguishes it from a driver fault; anything else re-raises and stays `failed`. The
collector decides OOM from markers, never from log text. See {doc}`methodology`.

### `status.json`

Written by the shared runner with `status: "success"`, `"oom"` or `"failed"`, plus the exit code,
timestamps, duration and command.

### Optional trace

Setting `SIMRL_TRACE_STEP=N` traces step `N` on rank 0 with `torch.profiler` and writes
`trace_step{N}.json` (Chrome trace) and `kernel_totals_step{N}.json` (summarised kernel time) into
the candidate directory. It is off by default because the profiler perturbs the steps the latency
model is fitted against, so a traced run should not also be the measured run. The summary is what
identifies kernels the composed training model does not know about.

## Fitting

```bash
srl-regress training \
    --latency-model hydraulis --memory-model hydraulis \
    --input  outputs/profile/qwen3_8b/megatron_bridge/profiles/megatron_bridge.jsonl \
    --output outputs/cost/qwen3_8b/training_hydraulis.json
```

`--validation-fraction` defaults to `0.2`, the split is by run id, and `--allow-rank-deficient`
lowers the design-rank requirement. Formulas, the cell key and lookup are in
{doc}`../cost-models/training-megatron`. The `ProfileRecord` schema is in
{doc}`../reference/schemas`.
