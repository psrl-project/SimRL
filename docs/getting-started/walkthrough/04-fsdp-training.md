# Stage 4: FSDP2 training

The same measurement as {doc}`03-megatron-training`, under a different backend. FSDP2 shards
parameters, gradients, and optimizer state alike. Megatron's distributed optimizer shards the
optimizer state only and keeps a full-size gradient buffer. The static memory footprints
differ accordingly, which is why this is a separate model and not a reused one.

It is also the sweep with the most candidates, and the one whose memory model is fitted with
out-of-memory results as inequality constraints.

## What it needs

A checkpoint and eight GPUs. The grid is `examples/profile/fsdp/qwen3_8b.yaml`:

```yaml
search_space:
  world_size: 8
  shard_sizes: [2, 4, 8]              # pure FSDP and both HSDP meshes
  num_microbatches: [1, 2, 4]
  n_seqs_per_rank: [1, 2, 4]
  sequence_lengths: [512, 2048, 8192, 16384]
  gradient_checkpointing: [true]
  optimizer_offload: [false]
  reshard_after_forward: [true]

training:
  attn_implementation: flash_attention_2
  max_steps: 12
  warmup_steps: 4
```

`srl-profile fsdp generate` produces **108 candidates**: three shard sizes, three microbatch
counts, three sequences per rank, and four sequence lengths. `replicate_size` is derived as
`world_size / shard_size`, so `shard_sizes: [2, 4, 8]` gives the meshes 2x4, 4x2, and 8x1.

Two axes are pinned on purpose. `optimizer_offload: [false]` is a first-order effect on both
latency and memory that deserves its own cells, and sweeping it would double the run.
`gradient_checkpointing: [true]` matches what a production trainer runs, so the profiled cell is
the one that will execute. A cell is selected by exactly these discrete values at simulation
time (see below), so profile `true` separately if you need it.

## The commands

```bash
CFG=examples/profile/fsdp/qwen3_8b.yaml
srl-profile fsdp generate --config $CFG
srl-profile fsdp run      --config $CFG --index 0           # one candidate; see the wrapper below
srl-profile fsdp collect  --config $CFG

srl-regress training --latency-model fsdp --memory-model fsdp \
    --input  outputs/profile/qwen3_8b/fsdp/profiles/fsdp.jsonl \
    --output outputs/cost/qwen3_8b/training_fsdp.json
```

The wrapper script is the practical entry point:

```bash
bash examples/profile/fsdp/profile_qwen3_8b.sh generate
CANDIDATE_INDEX=all bash examples/profile/fsdp/profile_qwen3_8b.sh run
CANDIDATE_INDEX=all bash examples/profile/fsdp/profile_qwen3_8b.sh collect
```

It takes `generate`, `run`, or `collect`, reads `CONFIG` (default `examples/profile/fsdp/qwen3_8b.yaml`),
and **defaults `CANDIDATE_INDEX` to 0**, so a bare `run` launches one candidate. The
underlying flags are `--candidates PATH`, `--candidate-id ID | --index N`, `--dry-run`, and
`--resume`. `--dry-run` logs the `torchrun ... -m srl.profiler.adapters.fsdp.launch --spec
<launch_spec.json>` command per candidate without starting it. `--resume` skips candidates
already marked `success`, so it **re-runs OOM and failed ones**.

**Smoke-test first.** `examples/profile/fsdp/qwen3_8b_smoke.yaml` builds a small model and
finishes quickly. A long sweep is an expensive way to discover a path typo. A candidate's
`run.duration_s` is dominated by process startup (spawning eight ranks, importing torch,
building the model, forming the NCCL group) and not by the retained steps, so reducing
`max_steps` barely shortens a sweep. Compare `run.duration_s` with `sum(step_times_s)` in
`fsdp.jsonl` to see the ratio on your machine.

## What lands on disk

```text
outputs/profile/qwen3_8b/fsdp/
├── control/fsdp/candidates.json      the candidate list
├── raw/fsdp/<candidate_id>/          launch_spec.json, srl_metrics.json, run.log, status.json,
│                                     and oom_rank<N>.json on a caught out-of-memory
└── profiles/fsdp.jsonl               one normalised record per candidate
```

A sweep produces a mix of statuses. The record set fitted on this page held 108 records:
**73 `success`, 34 `oom`, 1 `failed`.**

The OOM records are wanted data, not waste. An OOM is not a missing measurement, it is an
inequality: whatever the true peak was, it exceeded device capacity. The memory fitter keeps
those as inequality constraints beside the equalities from successful runs, and they bound the
valid domain from above. **A sweep that never runs out of memory has not found the ceiling.**

An OOM record carries the full configuration and a reason, with no usable timings:

```javascript
{
  "run": {
    "status": "oom",
    "error": "CUDA out of memory on rank(s) 0, 1, 2, 3, 4, 5, 6, 7.",
    "raw_artifact": "raw/fsdp/<candidate_id>",
    ...
  },
  "payload": { ... }   // configuration present, step_times_s empty
}
```

Status is decided by the marker files and nothing else. A rank that catches
`torch.OutOfMemoryError` writes `oom_rank<N>.json`. If the process exits non-zero with no
marker the candidate is `failed`, and a log that mentions "out of memory" does not change
that. A false OOM could only raise the fitted memory floor, so the classification is
deliberately strict. A candidate whose memory exhaustion surfaced some other way is `failed`
and its lower bound is lost. {doc}`../../profiling/fsdp` has the classification rules.

A successful record's payload has the same shape as Megatron's, with one difference that
matters:

```javascript
{
  "num_microbatches": 2,
  "microbatches": [
    {
      "count": 4,                             // four sequences packed into one microbatch
      "lengths": [2048, 2048, 2048, 2048],
      "max_tokens": 8192,
      "sum_tokens": 8192,
      "sum_squared_tokens": 16777216          // 4 * 2048^2, NOT 8192^2
    },
    ...
  ],
  "step_times_s": [...],
  "warmup_steps": 4,
  "peak_allocated_bytes": ...,
  "peak_reserved_bytes": ...,
  "rank_memory": [...]
}
```

**This sweep genuinely packs**: among the successful records, microbatches hold 1, 2, or 4
sequences (`n_seqs_per_rank`). That is where `sum_squared_tokens` differs from `sum_tokens`
squared, by a factor of the sequence count. Four 2048-token sequences give 1.68e7 against
6.71e7, a 4x difference in the attention term. The FSDP record's `tags` carry the mesh:
`shard_size`, `replicate_size`, `optimizer_offload`, `reshard_after_forward`,
`attn_implementation`.

## The fitted artifact

`model_type: fsdp`, six entries: one latency and one memory per shard degree. Fitting the
record set above gave:

| component | mesh (shard x replicate) | design rank | train MRE | held-out MRE (samples) |
|---|---|---|---|---|
| latency | 2 x 4 | 4 | 3.9% | 7.8% (4) |
| latency | 4 x 2 | 4 | 0.85% | 1.05% (5) |
| latency | 8 x 1 | 4 | 0.76% | 1.18% (5) |
| memory | 2 x 4 | 3 | 5.2% | 5.1% (4) |
| memory | 4 x 2 | 3 | 3.7% | 5.0% (5) |
| memory | 8 x 1 | 3 | 5.7% | 4.7% (5) |

Latency for the 8x1 mesh:

```javascript
{
  "coefficients": {
    "per_step_overhead":    0.03209,
    "token_cost":           0.0004646,     // seconds per token
    "token_quadratic_cost": 1.516e-08,
    "microbatch_overhead":  0.219
  },
  "feature_ranges": {
    "tokens":                   [512.0, 16384.0],
    "sum_squared_tokens":       [262144.0, 1073741824.0],
    "num_microbatches":         [1.0, 4.0],
    "sequences_per_microbatch": [1.0, 4.0]
  },
  "selector": {
    "component": "latency", "schedule": "fsdp",
    "shard_size": 8, "replicate_size": 1,
    "recompute": "full", "optimizer": "adam",
    "optimizer_offload": false, "reshard_after_forward": true,
    "parameter_dtype": "bf16", "gradient_dtype": "fp32", "optimizer_state_dtype": "fp32"
  },
  "diagnostics": {
    "formula": "per_step_overhead + token_cost * sum_over_microbatches(sum_tokens) + token_quadratic_cost * sum_over_microbatches(sum_squared_tokens) + microbatch_overhead * num_microbatches",
    "slots": "num_microbatches",
    "validation_is_in_sample": false,
    ...
  }
}
```

The form is the same four-term shape as Megatron's, with `slots = num_microbatches` since
there is no pipeline to drain, and the token terms summed over the rank's real microbatches.

Memory is structurally different:

```javascript
{
  "coefficients": {
    "parameter_count":   8190735360.0,
    "world_size":        8.0,
    "microbatch_slope":  1115412488.7,      // bytes per microbatch
    "activation_slope":  183843.5,          // bytes per activation token
    "residual_constant": 84375275674.7
  },
  "feature_ranges": {"num_microbatches": [1.0, 4.0], "activation_tokens": [512.0, 65536.0]},
  "diagnostics": {
    "memory_metric": "peak_reserved_bytes",            // chosen per cell, see below
    "reserved_over_allocated": 1.148,
    "oom_constraints": 16,
    "oom_intercept_increase": 5.14e9,                  // bytes the OOM evidence added
    "candidate_r_squared": {"peak_allocated_bytes": 0.534, "peak_reserved_bytes": 0.648},
    "static_inventory": "fsdp2-sharded-parameters-gradients-and-optimizer"
  },
  "train_metrics": {"n_samples": 15, "r_squared": -8.653, "mre_percent": 5.184}
}
```

(The entry above is the 2 x 4 mesh.) Three things there look wrong and are not.

**Memory scales with microbatch count, not tokens.** Under gradient checkpointing, activations
are recomputed on the backward pass instead of kept resident, and what accumulates is one
autograd graph per microbatch. A feature set built only on the largest token count fits a flat
target and degenerates, so the residual carries both `microbatch_slope * num_microbatches` and
`activation_slope * max_tokens`, both non-negative.

**The memory target is chosen per cell.** Both `peak_allocated_bytes` and `peak_reserved_bytes`
are fitted and the better validation score wins. The stored `reserved_over_allocated` ratio
lets an allocated-target prediction still be compared with device capacity, since an OOM is
triggered by reserved bytes. In the fit above the 2 x 4 and 4 x 2 meshes selected reserved and
the 8 x 1 mesh selected allocated.

**A negative $r^2$ on a cell with OOM constraints is those constraints working.** The 2 x 4
cell has 16 inequality constraints against 15 successes. They lift the intercept by about
5.1e9 bytes above what the successes alone support, so the model over-predicts those successes
by design and will not claim a workload fits when the OOM evidence says otherwise. The MRE
describes the slope, which is what varies in real queries.
{doc}`../../cost-models/training-fsdp` has the derivation.

## How to tell it worked

```bash
python -c "
import json
a = json.load(open('outputs/cost/qwen3_8b/training_fsdp.json'))
for e in a['entries']:
    s = e['selector']
    print(f\"{s['component']:8s} shard={s['shard_size']}x{s['replicate_size']}\"
          f\"  rank={e['train_metrics']['design_rank']}\"
          f\"  train {e['train_metrics']['mre_percent']:6.3f}%\"
          f\"  held-out {e['validation_metrics']['mre_percent']:6.3f}%\")
"
```

Check three things. **`design_rank: 4`** on every latency cell, since a pinned
`num_microbatches` would cap it and make the whole sweep unfittable (the fitter refuses it
without `--allow-rank-deficient`). **`validation_is_in_sample: false`** with a non-zero
`validation_metrics.n_samples`, confirming a real split. And **`oom_constraints`** above zero
on memory entries that sit near the capacity edge, which says the sweep found the ceiling.

Fitted independently per mesh, `token_cost` should come out near the same value on meshes that
never saw each other's data. When it does, the regression recovered a physical quantity. When
one cell drifts well away from the others, look at its sample count first.

## Use it

An FSDP artifact holds one cell per mesh, and a cell answers only for the mesh it measured, so
the config names it:

```bash
srl rollout_cost=roofline rollout_cost.artifact_path=outputs/cost/qwen3_8b/rollout_roofline.json \
    training_cost=fsdp \
    training_cost.artifact_path=outputs/cost/qwen3_8b/training_fsdp.json \
    training_cost.shard_size=8 training_cost.replicate_size=1 \
    workload=dapo_trace workload.trace.path=outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl \
    job.n_versions=3 job.batch_size=4 \
    job.max_prompt_length=1024 job.max_output_length=10240
```

Omitting `shard_size` on a multi-mesh artifact raises `Artifact has multiple shard_size values;
select one explicitly.` `training_cost.parallelism.dp` (default 8 in `fsdp.yaml`) must equal
`shard_size * replicate_size`. `dp=4` against the 8x1 cell raises `This FSDP cell was measured
on a 8x1 shard-by-replicate mesh, which is dp=8, so it cannot answer for dp=4`.
`optimizer_offload`, `reshard_after_forward`, `activation_recompute`, `optimizer`, and the
three dtypes must match a profiled cell too.

`training_cost.batching_mode` has no counterpart in the Megatron path. Profiling measured
uniform shapes, so how real trajectory lengths become microbatches is a simulate-time choice:

- `pad_to_max` (default) pads each microbatch to its longest sequence and charges the pad tokens.
- `token_budget` groups to `training_cost.max_tokens_per_rank` (verl's default is 10240), still
  padded within a microbatch.
- `packed_no_pad` concatenates with no pad tokens and charges the true token counts.

All three can return identical results, and must at `micro_batch_size=1`, so agreement is not
evidence that the wiring is wrong: balanced grouping puts similar lengths together, which flattens
the padding difference a second time. {doc}`../../simulation/domain-gates` explains where the
modes diverge, and {doc}`06-simulate` shows the full simulation.

## Next

{doc}`05-workload-trace` measures what the request shapes actually look like, which is what
decides whether these fitted domains are wide enough.
