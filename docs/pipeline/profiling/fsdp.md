# FSDP2 / HSDP profiling

This page covers the FSDP2 / HSDP pipeline: how to configure and run a sweep, which axes must
vary or the fit will refuse the data, and what lands on disk. The formulas are in
{doc}`/pipeline/cost-models/training-fsdp`. General measurement rules are in {doc}`/pipeline/profiling/methodology`.

## Commands

```bash
export MODEL_PATH=/path/to/Qwen3-8B
CFG=examples/profile/fsdp/qwen3_8b.yaml

srl-profile fsdp generate --config $CFG
srl-profile fsdp run      --config $CFG [--resume] [--dry-run] [--index N | --candidate-id ID]
srl-profile fsdp collect  --config $CFG [--index N | --candidate-id ID] [--output PATH]
```

`generate` writes the candidate list to `{output_dir}/control/fsdp/candidates.json`. `run` walks
the list, launches one `torchrun` child per candidate and records a status for each. A failing
candidate is logged and skipped. `collect` normalises every candidate directory into
`{output_dir}/profiles/fsdp.jsonl` by default. `--candidates PATH` overrides the manifest path on
all three.

`collect` raises when a candidate marked `success` carries no step times, no rank memory, or no
`observed_mesh`, so a launcher that failed to write its sidecar cannot become a record with no
measurement or an unverified topology. `profiling.allow_incomplete: true` turns those cases into
skips and also lets `collect` skip a candidate that was never prepared.

`examples/profile/fsdp/qwen3_8b_smoke.yaml` uses a two-layer override, so smoke records carry a
different `parameter_count` and the cell key keeps them out of a production fit.

`--resume` skips a candidate whose `status.json` says `success`, which means it **re-runs every
OOM candidate**. On a high-memory grid that repeats runs you already have complete information
about. Use `--index` or `--candidate-id` to run one candidate instead.

## Sweep axes

`generate` reads `search_space:`:

| axis | meaning |
|---|---|
| `world_size` | the fixed mesh size, defaults to `hardware.gpu_count` |
| `shard_sizes` | the HSDP shard degree. `replicate = world_size / shard`, and a shard degree that does not divide the world is skipped |
| `num_microbatches` | gradient-accumulation depth. **Must vary. See below.** |
| `n_seqs_per_rank` | sequences per rank per microbatch, i.e. the micro-batch size |
| `sequence_lengths` | tokens per sequence, every microbatch a rank runs is this shape |
| `gradient_checkpointing` | pinned `true` in the example, matching the RL recipe |
| `optimizer_offload` | pinned `false` in the example, see {doc}`/reference/limitations` |
| `reshard_after_forward` | whether parameters are freed after each block's forward |

`shard_size` and `replicate_size` are a two-dimensional HSDP mesh: parameters are sharded within
each row and replicated across rows. `shard == world` is pure FSDP. Sharding and replication are
configuration rather than data partitioning, so every rank still consumes a distinct microbatch
and the profile topology is pure DP. The mesh travels into the artifact as selectors.

`sequence_parallel` is not implemented at present. Any value other than 1 is rejected rather than
silently ignored.

### Why the microbatch count must vary

The latency design row is `(1, weighted_tokens, weighted_slots, weighted_sum_squared_tokens)`
with `slots = num_microbatches` for FSDP, and must reach rank 4. With `nmb` fixed, the slot column
is a scalar multiple of the intercept and the design caps below that. The memory design is
`[num_microbatches, max_tokens, 1]` and needs rank 3, so a single `nmb` value leaves its first
column collinear with the intercept as well. `check_latency_rank` and the memory rank check raise
rather than returning coefficients they cannot separate.

Sweeping sequence lengths alone does not fix this: it moves the token columns and leaves the slot
column degenerate. Use at least two, ideally three, widely spaced microbatch counts. The example
uses `[1, 2, 4]`. Running `nmb` identically shaped microbatches under gradient accumulation is
what real FSDP training does, so the axis costs no measurement fidelity.

## The launcher

Each candidate is one `torchrun` child (`srl.profiler.adapters.fsdp.launch`) driven by a
`launch_spec.json` written by `prepare`. The child:

1. Joins the process group, sets the device from `LOCAL_RANK` (not the global rank modulo the
   device count, which breaks on an unequal share per node).
2. Builds the model from `AutoConfig` alone, with no weights: timing and memory are governed by
   architecture, so no trained checkpoint is needed beyond `config.json`.
3. Builds the module in the **master dtype** (`optimizer_state_dtype`), because AdamW's moments
   inherit the dtype of the parameters it is handed, and the static inventory is sized from the
   declared dtype. A mismatch is raised after the first optimizer step by
   `check_optimizer_state_dtype`.
4. Enables non-reentrant gradient checkpointing and pins `model.train()`. HuggingFace gates
   recompute on `self.training`, so a model left in eval silently turns checkpointing into a
   no-op while the candidate still claims it.
5. Shards each transformer block with `fully_shard`, then the root. `reshard_after_forward` is
   applied to the blocks. The root always uses `False`, because its parameters are needed
   immediately when backward begins and resharding it adds a fixed cost to one side of the very
   comparison the axis is meant to isolate. `MixedPrecisionPolicy` uses `compute_dtype` for the
   all-gathered parameters and `gradient_dtype` for the reduction.
6. Runs `max_steps` accumulated steps. The timing bracket spans the whole step, accumulation
   included, because the per-step overhead term is defined as the cost paid once per step. No
   barrier is taken.
7. After the first step, records the live static allocation. It must be read after that step,
   because AdamW allocates its moments lazily.
8. After the timed loop, runs one extra step under `torch.profiler` and records collective kernel
   occupancy. Profiling inside the loop would inflate a retained sample. Outside it, it can
   contaminate none.
9. Gathers each rank's peak and has rank 0 write `srl_metrics.json`.

The synthetic batch is one uniform padded microbatch, repeated `num_microbatches` times.
Profiling measures uniform shapes only. Forming ragged microbatches from real trajectory lengths
is the simulate-side planner's job (`MicrobatchPlanner`).

`prepare` also records the effective mesh the child built as `observed_mesh`, and `collect`
compares it with the candidate. A record whose observed mesh disagrees is rejected with a hard
error, because the cell key is the topology.

## Output layout

```text
{output_dir}/
├── control/fsdp/candidates.json
├── raw/fsdp[/{experiment_name}]/{candidate_id}/
│   ├── launch_spec.json
│   ├── manifest.json
│   ├── srl_metrics.json
│   ├── oom_rank{N}.json
│   ├── status.json
│   └── run.log
└── profiles/fsdp.jsonl
```

There is no `ConfigContainer`. `srl_metrics.json` carries:

```javascript
{
  "step_times_s": [/* every step, warmup included */],
  "timers_s": {"all_gather": [...], "reduce_scatter": [...]},
  "rank_memory": [{"rank": 0, "allocated_bytes": 0, "reserved_bytes": 0}, ...],
  "microbatch_sequence_lengths": [[L, ...], ...],     // one entry per microbatch
  "parameter_count": 0,
  "observed_mesh": {"replicate_size": 1, "shard_size": 8},
  "static_allocated_bytes_at_train_start": 0
}
```

`timers_s` holds summed `self_device_time_total` per collective kind from the profiled step. That
is kernel *occupancy*, not exposed wall time: with `reshard_after_forward=True` most of the
all-gather overlaps compute on another stream, so the sum is an upper bound on what the step pays.
It is not read by the regression. It is ground truth for a future kernel-summation model.

`static_allocated_bytes_at_train_start` is a diagnostic despite its name: it is captured after the
first optimizer step, not at train start, because the moments are allocated lazily. It is
parameters plus gradients plus moments at whatever dtype the optimizer actually produced, which
follows configuration, so compare it against the analytic inventory rather than expecting it to
predict one.

## OOM classification

OOM candidates are wanted data: they bound the memory model from below, and an OOM at a
high-memory point is how the upper end of the token and microbatch axes is established.

Most OOMs raise `torch.OutOfMemoryError` and are caught by exact type. Memory exhaustion can also
surface as a NCCL buffer allocation failure: a step completes, then a collective tries to
allocate its buffers and the device is full. The child catches `DistBackendError` and classifies
it as OOM only when the message contains `failed to cuda calloc` or `out of memory`, recording
`oom_source: nccl_alloc` in the marker. Any other `DistBackendError`, such as a timeout or a
rendezvous failure, stays `failed`: widening the match would turn genuine network faults into
memory bounds.

`collect` decides OOM from the marker files, never from log text, for the reason
{doc}`/pipeline/profiling/methodology` gives: a false OOM can only raise the fitted memory floor.

Stale markers are deleted before a launch. If a re-run is killed after that deletion but before
new markers are written, `status.json` still reads `oom` while `collect` reports `failed`. The
two disagreeing is the signature. Re-run that candidate with `--candidate-id` rather than
accepting the classification.

## Fitting

```bash
srl-regress training --latency-model fsdp --memory-model fsdp \
    --input  outputs/profile/qwen3_8b/fsdp/profiles/fsdp.jsonl \
    --output outputs/cost/qwen3_8b/training_fsdp.json
```

FSDP latency and memory must be fitted together. The fitter groups records by the full discrete
cell key including `shard_size` and `replicate_size`, refuses a record whose schedule is not
`fsdp`, whose `sequence_parallel` is not `1`, or whose `tp`/`pp`/`cp` are not 1, and checks that
`shard_size * replicate_size` agrees with `dp`. `--validation-fraction` defaults to `0.2` and the
split is by run. Memory fits both peak metrics and selects one per cell. See
{doc}`/pipeline/cost-models/training-fsdp` for that rule and for what the selected target means at
predict time.
