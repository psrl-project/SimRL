# Megatron-Bridge profiling

This page covers the Megatron-Bridge training sweep: what the sweep axes mean, why SimRL owns the
entry point, how to run it, what ends up on disk, and what every sidecar file contains. For the
formulas that regression fits to that output, see {doc}`../cost-models/training-megatron`. For
general measurement rules (warmup policy, dispersion reporting, OOM as data) see {doc}`methodology`.

---

## Why SimRL owns the entry point

The Megatron-Bridge distribution ships a `scripts/performance` CLI. It is not used here. That CLI
parses its own known arguments and silently drops everything else, so recompute mode, output-path
settings, and profiling flags never reached the underlying `ConfigContainer`. Its `--gpu` choices
also exclude H20 outright.

SimRL's adapter takes the library's published recipe (`megatron.bridge.recipes.<family>`) and
applies the candidate configuration in Python, where a mistyped field raises an error rather than
disappearing. The torchrun child is `srl.profiler.adapters.megatron.launch`, launched via
`srl-profile megatron run`. Every setting the child needs travels in `launch_spec.json`, and nothing
is marshalled through a CLI that could silently lose it.

The official callback API (`megatron.bridge.training.callbacks`) is used to bracket steps and read
CUDA allocator peaks. This replaces TensorBoard scalar scraping and memory-snapshot pickles: the
sidecar `srl_metrics.json` contains exactly the fields the collector needs, in a stable schema,
without parsing logs.

---

## Quick start

Install the pinned environment and export the checkpoint path once before running any sweep:

```bash
conda activate simrl
bash third_party/install.sh
pip install -e ".[dev]"

export MODEL_PATH=/path/to/Qwen3-8B    # local HuggingFace checkpoint directory
```

The three-verb contract ({doc}`index`) applies without modification:

```bash
# config default: examples/profile/megatron/qwen3_8b.yaml
# smoke override: CONFIG=examples/profile/megatron/qwen3_8b_smoke.yaml

bash examples/profile/megatron/profile_qwen3_8b.sh generate
bash examples/profile/megatron/profile_qwen3_8b.sh run       # add --resume to continue
bash examples/profile/megatron/profile_qwen3_8b.sh collect
```

The script defaults to candidate index zero, so an accidental bare invocation of `run` cannot
launch the entire sweep. To run all candidates, set `CANDIDATE_INDEX=all` before calling the
`run` step:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CANDIDATE_INDEX=all \
    bash examples/profile/megatron/profile_qwen3_8b.sh run
```

### Smoke test first

Before launching any full-model candidate, prove the chain on one GPU with a two-layer model:

```bash
CUDA_VISIBLE_DEVICES=0 CONFIG=examples/profile/megatron/qwen3_8b_smoke.yaml \
    bash examples/profile/megatron/profile_qwen3_8b.sh run
```

The smoke config (`examples/profile/megatron/qwen3_8b_smoke.yaml`) uses `num_layers_override: 2`
and `gpu_count: 1`. Those records must never be mixed into a production fit, because the cost-model cell
key includes `parameter_count`, which separates them automatically.

### Multi-node runs

Multiple nodes use the same torchrun launcher path. Pass `--nnodes N --node-rank K` directly to
`srl-profile megatron run`. There is no separate Slurm entry point. Use a Slurm job array that
calls the same command with the appropriate `--node-rank`.

---

## Sweep axes and constraints

The default config (`examples/profile/megatron/qwen3_8b.yaml`) sweeps:

| axis | default values | notes |
|---|---|---|
| `tensor_parallel_sizes` | `[1, 4, 8]` | must divide `num_attention_heads` and `num_key_value_heads` |
| `pipeline_parallel_sizes` | `[1, 4]` | must divide `num_layers` uniformly |
| `num_microbatches` | `[1, 4]` | local per-DP-rank depth, where GBS = MBS x DP x `num_microbatches` |
| `sequence_lengths` | `[512, 1024, 2048, 4096, 8192]` | five values, reaching what a real batch needs |
| `recompute` | `full` | discrete selector, where "none" is a separate cell |
| `micro_batch_size` | `1` | fixed |

`hardware.gpu_count` is the fixed world size. DP is `world_size / (TP × PP)`. The same topology is
never repeated across candidates.

### Why both `num_microbatches` and `sequence_lengths` are required

The Hydraulis latency design row is `[1, slots * tokens, slots, slots * sum_squared_tokens]` where
`slots = num_microbatches + PP - 1`. At a fixed `num_microbatches`, the `slots` column is constant
and therefore collinear with the intercept, capping the design matrix at rank 3 where the fit needs
rank 4. Reaching full rank requires at least two distinct `num_microbatches` values and at least
three `sequence_lengths`. Do not prune either axis below those minimums.

The fitted `feature_ranges` become hard out-of-domain gates at prediction time, so extend both axes
to cover whatever token counts the simulator will actually request, once the sweep is green.

### Coverage note

The sweep covers `sequence_lengths` up to 8192, widened from an earlier 2048 because a real DAPO
batch exceeded it. The tail still sits outside: the DAPO trace's longest sequence is 11133 tokens,
about 1.36x past the domain. That residual gap is a known limitation. See
{doc}`../reference/limitations`.

---

## Timing (Qwen3-8B / 8x H20 / 2026-09-03, 40 candidates)

| stat | value |
|---|---|
| candidates | 40 |
| median wall clock per candidate | 169.8 s |
| of which is actual training | ~16 s (9%) |
| median training step | 1.34 s |
| total | 2.23 h |

Roughly 154 s of each 170 s is fixed process overhead: spawning eight processes, importing torch
and Megatron in every rank, building the full 36-layer model, forming the NCCL group, and tearing
it all down. That startup cost is near-identical between Megatron and FSDP. Reducing `max_steps`
is therefore not the meaningful lever it appears to be. See the sweep-cost discussion in
{doc}`index`.

### Steps and warmup

Each candidate runs `max_steps: 12` steps and discards `warmup_steps: 4`. A single warmup step was
insufficient: under one warmup step the first profiled step was the slowest in 18 of 24 runs, where
chance alone would give about 8. Dropping the second step as well cut the median spread within a
candidate from 16.4% to 0.5%. The eight retained samples then support a median and outlier
rejection, which two samples cannot. The per-step cost is well under a second, so the extra steps
are nearly free relative to the 143 s startup.

The measurement bracket covers `on_train_step_start` through `on_train_step_end` with
`torch.cuda.synchronize()` on each side and `time.perf_counter()` for elapsed time. No barrier is
taken, so each rank's timing reflects its own forward-backward-optimizer work rather than the
slowest rank's.

---

## Output layout

After a successful sweep, the output tree rooted at `output_dir` (default:
`outputs/profile/qwen3_8b/megatron_bridge/`) looks like:

```text
outputs/profile/qwen3_8b/megatron_bridge/
├── control/megatron_bridge/
│   └── candidates.json          ← full candidate list written by generate
├── raw/megatron_bridge/
│   └── <candidate_id>/
│       ├── launch_spec.json     ← everything the torchrun child reads
│       ├── manifest.json        ← immutable record of the candidate (written by prepare)
│       ├── ConfigContainer.yaml ← what Megatron actually resolved (cross-checked by collect)
│       ├── srl_metrics.json  ← the measurement sidecar (written by rank 0 on success)
│       ├── oom_rank{N}.json     ← per-rank OOM marker (written on torch.OutOfMemoryError)
│       ├── status.json          ← success | oom | failed
│       └── run.log
└── profiles/
    └── megatron_bridge.jsonl    ← normalised output consumed by regression
```

### `launch_spec.json`

Serialised by `prepare` before `torchrun` starts. Contains every field the child needs: recipe
family and name, parallelism, `seq_length`, `micro_batch_size`, `global_batch_size`,
`num_microbatches`, recompute settings, dtype flags, `vocab_size`, `train_iters`, `warmup_steps`,
and `run_dir`. Nothing is passed through `sys.argv`.

### `manifest.json`

Written alongside `launch_spec.json` and never mutated after that. Records the adapter name,
backend, launcher arguments, the candidate as a structured object, the effective (padded) sequence
shape, recompute method and layer count, and the backend source commit. If `num_layers_override` is
set (smoke runs), that value is recorded so the collector can use the overridden layer count when
building the architecture descriptor.

### `ConfigContainer.yaml`

Rank zero serialises the resolved `ConfigContainer` at the path declared in `launch_spec.json`
(`run_dir/ConfigContainer.yaml`). The collector cross-checks it against the candidate: TP, PP, CP,
`seq_length`, `micro_batch_size`, `global_batch_size`, and recompute granularity must all match.
A mutation that was applied in Python but failed to stick is caught here rather than silently
producing a record attributed to the wrong configuration.

### `srl_metrics.json`

Written atomically by rank 0 in `on_train_end`. Contains:

- `step_times_s`: wall-clock seconds per retained step, after the collector slices off the
  warmup window. The
  collector applies the warmup slice
- `rank_memory`: one entry per rank with `rank`, `allocated_bytes`, `reserved_bytes`,
  `static_allocated_bytes`, and `local_parameter_count`
- `microbatch_sequence_lengths`: the shapes actually trained (used to detect mismatches)
- `parameter_count`: global parameter count summed across model-parallel groups
- `static_reserved_bytes_at_train_start`: a diagnostic, the allocator state just before the first
  step, used to confirm that peak-reset kept the static pool

The collector reads `srl_metrics.json` if it exists and otherwise falls back to TensorBoard
scalars, and the sidecar is always preferred.

### `oom_rank{N}.json`

Written by each rank independently when `torch.OutOfMemoryError` is caught. No collective is taken,
because a CUDA out-of-memory failure is usually local to one rank while the others are blocked
inside a collective. Each marker records `rank`, `torch_oom: true`, `device_free_bytes`,
`device_total_bytes`, `device_reserved_bytes`, `seq_length`, `num_microbatches`, and the error
message (truncated to 2000 characters).

The collector reads these markers to classify the candidate as OOM, not the run log. A log that
merely mentions "out of memory" is not used, because such a match would add an inequality constraint
that can only raise the fitted memory floor. OOM records are wanted data, not failures to discard.
NCCL allocation failures and driver-level CUDA errors are distinct faults and remain classified as
`failed`, not `oom`.

### `status.json`

Written by `execute` with `status: "success"`, `"oom"`, or `"failed"`. The `run` verb's `--resume`
flag skips candidates with `status: "success"` and re-runs all others, including OOM candidates.

---

## Fitting the collected profiles

```bash
srl-regress training \
    --latency-model hydraulis \
    --memory-model hydraulis \
    --input  outputs/profile/qwen3_8b/megatron_bridge/profiles/megatron_bridge.jsonl \
    --output outputs/cost/qwen3_8b/training_hydraulis.json
```

For the Hydraulis formulas, fitted coefficient meanings, and held-out accuracy numbers, see
{doc}`../cost-models/training-megatron`. For the `ProfileRecord` schema that
`megatron_bridge.jsonl` conforms to, see {doc}`../reference/schemas`.
