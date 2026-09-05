# Stage 4: FSDP2 training

The same measurement as {doc}`03-megatron-training`, under a different backend. FSDP2
shards parameters, gradients, and optimizer state. Megatron's distributed optimizer shards
optimizer state only and keeps a full-size fp32 gradient buffer. Their memory footprints
differ by roughly twofold at the same model size, which is why this is a separate model
rather than a reused one.

This is the longest sweep on the project, about seven hours, and it is the only model with
a genuine held-out split on every cell.

## What it needs

A checkpoint and eight GPUs. The grid is in `examples/profile/fsdp/qwen3_8b.yaml`:

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

108 candidates. `replicate_size` is derived, so `shard_sizes: [2, 4, 8]` gives the meshes
2x4, 4x2, and 8x1.

Two pinned axes are worth knowing about. `optimizer_offload` is fixed to `false` because
offloading is a first-order effect on both latency and memory and deserves its own cells,
but sweeping it would double a seven-hour run, and one complete surface beats two
half-covered ones. `gradient_checkpointing` is fixed to `true` because that is what the
production trainer runs, so the profiled cell matches what will actually execute.

## The commands

```bash
CFG=examples/profile/fsdp/qwen3_8b.yaml
dryrun-profile fsdp generate --config $CFG
dryrun-profile fsdp run      --config $CFG            # --resume, or --candidate-id ID
dryrun-profile fsdp collect  --config $CFG

dryrun-regress training --latency-model fsdp --memory-model fsdp \
    --input  outputs/profile/qwen3_8b/fsdp/profiles/fsdp.jsonl \
    --output outputs/cost/qwen3_8b/training_fsdp.json
```

**Smoke-test first.** `examples/profile/fsdp/qwen3_8b_smoke.yaml` builds a two-layer model
and finishes in minutes. Seven hours is a long time to discover a path typo.

## How long it takes

Measured 2026-08-31 on 8x H20, from `run.duration_s`:

| | |
|---|---|
| candidates | 108 |
| median per candidate | 183.0 s |
| total | 6.87 h |
| median training step | 4.64 s |

About 136 s of each 183 s is process startup, the same fixed cost as Megatron's 143 s. The
sweep is longer than Megatron's because it has 108 candidates against 40 and a wider grid,
not because FSDP is slower per candidate.

## What lands on disk

```text
outputs/profile/qwen3_8b/fsdp/
├── control/fsdp/                 candidate list, per-candidate status, OOM markers
├── raw/fsdp/                     per-candidate step logs
└── profiles/fsdp.jsonl           108 records
```

The outcome: **73 `success`, 34 `oom`, 1 `failed`.**

The 34 OOM records are wanted data, not waste. An OOM is not a missing measurement, it is
an inequality: whatever the true peak was, it exceeded device capacity. The memory fitter
records those as inequality constraints alongside the equalities from successful runs, and
they are what bounds the model's valid domain from above. **A sweep that never OOMs has not
found the ceiling.**

An OOM record carries the full configuration and a reason, with no usable timings:

```javascript
{
  "run": {
    "status": "oom",
    "error": "CUDA out of memory on rank(s) 0, 1, 2, 3, 4, 5, 6, 7.",
    "duration_s": 182.166
  },
  "payload": { ... }   // configuration present, step_times_s empty
}
```

That status must come from a marker file the launcher writes on a caught allocation
failure, never from grepping a log for "out of memory". A log can contain that phrase for
unrelated reasons, and a false positive can only *raise* the fitted memory floor, making
the model quietly more conservative in a way no metric reports.

One further trap: with `shard=2` the static state alone occupies 64% of device capacity,
and running out of memory there surfaced as an **NCCL allocation failure** rather than
`torch.OutOfMemoryError`. Those candidates were initially classified `failed`, which threw
away the lower-bound data. {doc}`../../profiling/fsdp` has the classification rules.

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
  "step_times_s": [...],                      // 8 retained of 12
  "warmup_steps": 4,
  "peak_allocated_bytes": ...,
  "peak_reserved_bytes": ...,
  "rank_memory": [...]
}
```

**This sweep genuinely packs**, unlike the Megatron one: of the successful records, 74
microbatches hold one sequence, 54 hold two, and 38 hold four. That is exactly where
`sum_squared_tokens` differs from `sum_tokens` squared, by a factor of the sequence count.
Four 2048-token sequences give 1.68e7 against 6.71e7, a 4x difference in the attention
term. Getting this wrong was a live bug, not a hypothetical one.

## The fitted artifact

Six entries, one latency and one memory per shard degree.

Latency, `shard=2`:

```javascript
{
  "coefficients": {
    "per_step_overhead":    0.07527,
    "token_cost":           0.0004589,     // seconds per token
    "token_quadratic_cost": 1.609e-08,
    "microbatch_overhead":  0.4181
  },
  "feature_ranges": {
    "tokens":                   [512.0, 16384.0],
    "num_microbatches":         [1.0, 4.0],
    "sequences_per_microbatch": [1.0, 4.0]
  },
  "selector": {
    "component": "latency", "schedule": "fsdp",
    "shard_size": 2, "replicate_size": 4,
    "recompute": "full", "optimizer": "adam",
    "optimizer_offload": false, "reshard_after_forward": true,
    "parameter_dtype": "bf16", "gradient_dtype": "fp32", "optimizer_state_dtype": "fp32"
  },
  "train_metrics":      {"n_samples": 15, "design_rank": 4, "r_squared": 0.99877, "mre_percent": 4.236},
  "validation_metrics": {"n_samples": 4,  "design_rank": 4, "r_squared": 0.99633, "mre_percent": 6.519},
  "diagnostics": {
    "formula": "per_step_overhead + num_microbatches * (token_cost * max_microbatch_tokens + token_quadratic_cost * max_microbatch_sum_squared_tokens + microbatch_overhead)",
    "slots": "num_microbatches",
    "validation_fraction": 0.2,
    "validation_is_in_sample": false
  }
}
```

The formula is the same four-term shape as Megatron's, with `slots = num_microbatches`
rather than `num_microbatches + PP - 1`, because there is no pipeline to drain.

Memory is structurally different, and the interesting part is its target selection:

```javascript
{
  "coefficients": {
    "parameter_count":     8190735360.0,
    "world_size":          8.0,
    "microbatch_slope":    993942060.1,      // bytes per microbatch
    "activation_slope":    181751.1,         // bytes per activation token
    "residual_constant":   84505316328.1
  },
  "feature_ranges": {
    "num_microbatches":  [1.0, 4.0],
    "activation_tokens": [512.0, 65536.0]
  },
  "diagnostics": {
    "memory_metric": "peak_reserved_bytes",         // per-cell choice, see below
    "reserved_over_allocated": 1.148,
    "oom_constraints": 16,
    "oom_intercept_increase": 5109677791.8,         // bytes the OOM evidence added
    "static_inventory": "fsdp2-sharded-parameters-gradients-and-optimizer",
    "candidate_r_squared": {"peak_allocated_bytes": 0.301, "peak_reserved_bytes": 0.615}
  },
  "train_metrics": {"n_samples": 15, "r_squared": -8.189, "mre_percent": 5.053}
}
```

Three things in there look wrong and are not.

**Memory scales with microbatch count, not tokens.** Under gradient checkpointing,
activations are recomputed on the backward pass rather than kept resident. What accumulates
instead is one autograd graph per microbatch. Measured directly, allocated memory was flat
to four decimal places across a 4x token range while `num_microbatches` moved it. A feature
set built on `max(sum_tokens)` fits nothing here, and a negative $r^2$ was the diagnostic
doing its job.

**The memory target is chosen per cell.** Megatron fits `peak_allocated_bytes`, while here
`shard=2` selected `peak_reserved_bytes` because its allocated $r^2$ was only 0.30 against
0.61 for reserved. Both targets are fitted and the better one wins, with the measured
`reserved_over_allocated` ratio stored so an allocated-target prediction can still be
compared against device capacity. Predicting allocated directly would under-report OOM,
since the failure is triggered by reserved bytes.

**The `shard=2` memory $r^2$ of $-8.19$ is the OOM constraints working.** That cell has 16
inequality constraints against 15 successes. The inequalities lift the intercept about
5.1e9 bytes above what the successes alone support, and the raised intercept then
over-predicts those successes by that margin. The model is deliberately conservative: it
will not predict that a workload fits when the OOM evidence says otherwise. The 5.05% MRE
describes the slope, which is what varies in real queries.
{doc}`../../cost-models/training-fsdp` has the full derivation.

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
`num_microbatches` would cap it at 3 and make the whole seven-hour sweep unfittable.
**`validation_is_in_sample: false`** with a non-zero `validation_metrics.n_samples`,
confirming a real split. And **`oom_constraints`** above zero on the memory entries, which
says the sweep found the ceiling.

The `token_cost` coefficient is the strongest single piece of evidence. Fitted
independently per cell, it came out at 458.9, 467.0, and 466.7 µs per token across three
meshes that never saw each other's data. Agreement to 2% across independent fits means the
regression recovered a physical quantity rather than a curve. `token_quadratic_cost` agrees
too, at 1.609e-8, 1.473e-8, and 1.482e-8. {doc}`../../cost-models/training-fsdp` owns
those tables.

## Use it

```bash
TRAINING_BACKEND=fsdp BATCHING_MODE=pad_to_max \
TRAINING_ARTIFACT=outputs/cost/qwen3_8b/training_fsdp.json \
bash examples/simulate/simulate.sh
```

`BATCHING_MODE` has no counterpart in the Megatron path. Profiling measured uniform
shapes, so how real trajectory lengths become microbatches is a simulate-time choice:
`pad_to_max` charges for pad tokens, `token_budget` reproduces verl's dynamic batching, and
`packed_no_pad` charges real tokens only. Be aware that all three can legitimately return
identical results, and must at `micro_batch_size=1`. Balanced grouping puts similar lengths
together, which flattens the padding difference a second time.
{doc}`../../simulation/domain-gates` explains when they diverge.

## Next

{doc}`05-workload-trace` measures what the request shapes actually look like, which is what
decides whether these fitted domains are wide enough.
