# FSDP2 / HSDP profiling

This page covers the FSDP2 / HSDP pipeline: how to configure and run a sweep, what axes
to vary and, critically, which axis must vary or the fit will refuse the data, and where
the output lands. Read {doc}`index` first for the three-verb contract and the cost table,
and {doc}`methodology` before committing to a full sweep.

---

## Commands

```bash
# Set MODEL_PATH to your Qwen3-8B (or other HuggingFace) checkpoint directory.
export MODEL_PATH=/path/to/Qwen3-8B

srl-profile fsdp generate --config examples/profile/fsdp/qwen3_8b.yaml
srl-profile fsdp run      --config examples/profile/fsdp/qwen3_8b.yaml [--resume]
srl-profile fsdp collect  --config examples/profile/fsdp/qwen3_8b.yaml [--output PATH]
```

A smoke run uses the two-layer override config, which keeps candidates under a minute and
verifies mesh construction and sidecar writing before investing the full sweep's time:

```bash
srl-profile fsdp generate --config examples/profile/fsdp/qwen3_8b_smoke.yaml
srl-profile fsdp run      --config examples/profile/fsdp/qwen3_8b_smoke.yaml
srl-profile fsdp collect  --config examples/profile/fsdp/qwen3_8b_smoke.yaml
```

Records from the smoke run carry a different `parameter_count` (two layers, not
thirty-six), so the cost-model cell key separates them automatically and they can never
contaminate a production fit.

---

## Running a single candidate

`--resume` skips candidates already marked `success`, but it **re-runs every OOM
candidate** because the status file says `oom` rather than `success`. On a sweep with many
high-memory candidates this re-executes tens of runs you already have complete information
about. Use `--index` or `--candidate-id` to re-run one specific candidate instead:

```bash
# By position in the generated list (0-indexed).
srl-profile fsdp run --config examples/profile/fsdp/qwen3_8b.yaml --index 42

# By the 16-character SHA-256 prefix written into candidates.json.
srl-profile fsdp run --config examples/profile/fsdp/qwen3_8b.yaml --candidate-id a3f7c12d9e8b4501
```

Both flags run exactly one candidate, write its output, and exit. After collecting a
partial sweep you can inspect which candidates are still missing and target them
individually rather than re-launching the whole grid.

---

## Sweep axes

The full config lives at `examples/profile/fsdp/qwen3_8b.yaml`. The `search_space` block
controls what `generate` enumerates:

| axis | values in full sweep | notes |
|---|---|---|
| `shard_sizes` | `[2, 4, 8]` | The HSDP shard degree. `replicate = world_size / shard`. |
| `num_microbatches` | `[1, 2, 4]` | **Must vary. See below.** |
| `n_seqs_per_rank` | `[1, 2, 4]` | Sequences per rank per microbatch (micro-batch size). |
| `sequence_lengths` | `[512, 2048, 8192, 16384]` | Tokens per sequence. |
| `gradient_checkpointing` | `[true]` | Pinned true, matching the DAPO training recipe. |
| `optimizer_offload` | `[false]` | Pinned false. See {doc}`../reference/limitations`. |
| `reshard_after_forward` | `[true]` | Controls whether parameters are freed after each block's forward. |

`shard_size` and `replicate_size` together describe a two-dimensional HSDP mesh:
parameters are sharded within each row of `shard_size` ranks and replicated across the
`replicate_size` rows. A shard degree that does not divide `world_size` describes no
runnable topology and is silently skipped during `generate`. With `world_size: 8` the
three values `[2, 4, 8]` produce `replicate` degrees of 4, 2, and 1 respectively, and `shard=8`
is pure FSDP (one mesh dimension, replicate=1).

### Why `num_microbatches` must vary

The latency model has three terms: a per-step overhead, a per-token cost, and a
per-microbatch overhead. Its design-matrix row for one candidate is
`[1, nmb * tokens, nmb]`. When `nmb` is fixed across all candidates, the third column is a
scalar multiple of the first column, and the design caps at rank two. The fit then cannot
separate `per_step_overhead` from `per_microbatch_overhead` and refuses the data.

Sweeping `sequence_lengths` alone does not fix this, because it varies the second column but
leaves the third degenerate against the first. **You must include at least two distinct
values for `num_microbatches` in every sweep you intend to fit.** The smoke config uses
`[1, 2]`, and the full sweep uses `[1, 2, 4]`. The same constraint applies to the memory
model: its design row is `[num_microbatches, tokens, 1]`, and a single `nmb` value leaves
the first column collinear with the intercept.

Running `nmb` identically-shaped microbatches with gradient accumulation is exactly what
real FSDP training does, so the axis costs no measurement fidelity.

---

## Timing (measured, Qwen3-8B / 8x H20 / 2026-08-31)

108 candidates total (73 `success` / 34 `oom` / 1 `failed`, counted from the collected
JSONL), median **183 s per candidate**, total **6.9 h**. Of each candidate's 183 s,
roughly 60 s (31%) is actual training, and the remaining 123 s is process startup:
spawning eight ranks, importing torch and transformers, building a 36-layer model, and
forming the NCCL group.

Cutting `max_steps` is not the lever it appears to be: reducing from 12 to 8 saves about
2 s out of 196 s (1%) while halving the retained samples the median and outlier rejection
depend on. See {doc}`index` for the full startup-cost analysis.

::{warning}
That single `failed` candidate is an artefact of an interrupted re-run, not a distinct
failure mode, and it is worth understanding because it costs you data silently.

`run` deletes any stale OOM marker files before launching a candidate, and `collect`
decides OOM **from those markers** rather than by scanning the log. That is deliberate, since a
false OOM adds an inequality that can only raise the fitted memory floor. If a re-run is
killed (`SIGTERM`) after the markers are cleared but before the child writes new ones, the
candidate's own `status.json` still reads `oom` while `collect` sees no markers and reports
`failed`. The memory lower bound that candidate contributed is then dropped from the fit.

Its `status.json` and its collected record disagreeing is the signature. If you see it,
re-run that candidate to completion with `--candidate-id` rather than accepting the
`failed` classification.
::

---

## Token ceiling

Beyond 16384 tokens/rank the hardware cannot run the profiled configuration:

| sequence_length | successes | OOMs |
|---|---|---|
| ≤ 16384 | 73 | 35 |
| 32768 | 0 | 18 |
| 65536 | 0 | 9 |

There is no ground truth at 32768 or above, so neither interpolation nor extrapolation
toward those values is meaningful. The `sequence_lengths` axis is capped at 16384 in the
production config for this reason.

OOM rate does not grow with `num_microbatches`. Across the 108 candidates, the success
counts at `nmb = 1`, `nmb = 2`, and `nmb = 4` were 26, 24, and 23 respectively. Gradient
accumulation repeats identical work and does not raise the activation peak, so adding
microbatches does not push the device closer to exhaustion.

For `num_microbatches` extrapolation semantics see {doc}`../simulation/domain-gates`.

---

## Output layout

```
outputs/profile/qwen3_8b/fsdp/
├── candidates.json               ← generated candidate list with candidate_id per row
├── profiles/
│   └── fsdp.jsonl                ← collected ProfileRecord stream
└── raw/fsdp/
    └── <candidate_id>/
        ├── manifest.json         ← spec echoed back by the launcher
        ├── srl_metrics.json   ← timings, memory peaks, mesh (written by rank 0)
        ├── status.json           ← "success" / "oom" / "failed"
        └── run.log               ← torchrun stdout/stderr
```

There is no `ConfigContainer.yaml`. Instead each candidate's raw directory carries
`srl_metrics.json`, which includes an `observed_mesh` field that `collect` cross-checks
before accepting the record. A record whose `observed_mesh` disagrees with its candidate
spec is rejected with a hard error rather than silently entering the fit attributed to the
wrong topology.

### The `srl_metrics.json` sidecar

Rank zero writes this file after the timed loop completes:

```javascript
{
  "step_times_s": [4.95, 0.083, 0.077, ...],        // all 12 steps, warmup included
  "timers_s": {"all_gather": [...], "reduce_scatter": [...]},
  "rank_memory": [{"rank": 0, "allocated_bytes": ..., "reserved_bytes": ...}, ...],
  "microbatch_sequence_lengths": [[512], [512]],     // one entry per microbatch
  "parameter_count": 1630601216,
  "observed_mesh": {"replicate_size": 1, "shard_size": 8},
  "static_allocated_bytes_at_train_start": 3329769472
}
```

`step_times_s` retains **all** steps including warmup. The **collector** slices them off
at `collect` time using `config.warmup_steps`, so the retained window can be changed
without re-profiling. The `timers_s` entries record NCCL kernel occupancy (summed
`self_device_time_total` per kind) from a separate profiled step run after the timed
loop. They are not used by the regression but are preserved as ground truth for a
future kernel-summation model.

For the full `ProfileRecord` schema see {doc}`../reference/schemas`.

---

## Fitting

```bash
srl-regress training --latency-model fsdp --memory-model fsdp \
    --input  outputs/profile/qwen3_8b/fsdp/profiles/fsdp.jsonl \
    --output outputs/cost/qwen3_8b/training_fsdp.json
```

The fitter partitions records by `shard_size` tag and fits each cell independently, which
is why the accuracy table in {doc}`../cost-models/training-fsdp` reports three latency rows
and three memory rows. Records from different shard degrees are not mixed.

For the model coefficients, fitted accuracy, the shard=2 `r²=-8.19` story, and the
`reserved`/`allocated` target selection see {doc}`../cost-models/training-fsdp`.

---

## OOM classification and the NCCL calloc trap

OOM candidates are wanted data, providing lower bounds that constrain the memory
model's intercept. Treating them as failures to skip would discard precisely the
high-memory end of the sweep that the model relies on as a floor.

Most OOM candidates raise `torch.OutOfMemoryError` directly. However, memory exhaustion
can also surface as a NCCL buffer allocation failure: training completes successfully,
then the collective inside `on_train_end` tries to allocate a 10 MiB NCCL buffer and
fails because the device is full. In the live sweep this produced five `DistBackendError`
records at `shard_size=2, replicate_size=4`, all dying with `Failed to CUDA calloc
10485760 bytes`.

The launcher handles this by text-matching the error message against known allocation
phrases (`failed to cuda calloc`, `out of memory`). When the message matches, the
candidate is classified as OOM and `oom_source` is set to `nccl_alloc` in the per-rank
marker file. A `DistBackendError` whose message does not match, such as a timeout, a rendezvous
failure, or a real network fault, stays classified as `failed` and is not treated as a
memory bound. The `oom_source` field makes the classification route auditable in the raw
artifacts.

---

## What the launcher does not depend on

The profiler constructs a HuggingFace model from `AutoConfig` without loading weights:
timing and memory are governed by architecture, not by trained values. Verl is absent from
the profiling environment. The relevant FSDP2 semantics (mesh, per-block wrapping,
mixed-precision policy, offload policy) are applied directly through torch APIs. The
profile therefore does not pin to a verl version and does not require a trained checkpoint
at `$MODEL_PATH`, since only the architecture `config.json` is read.
