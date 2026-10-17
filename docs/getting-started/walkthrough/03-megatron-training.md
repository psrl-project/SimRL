# Stage 3: Megatron training

Rollout is half the loop. This stage measures the other half: a real training step under
Megatron-Bridge, across four parallelism topologies, and fits a latency model and a
memory model from it.

## What it needs

A checkpoint and eight GPUs on one node. Weights are never loaded (`load_weights=false`),
so the checkpoint is read for its architecture only.

The grid is in `examples/profile/megatron/qwen3_8b.yaml`:

```yaml
search_space:
  tensor_parallel_sizes: [1, 4, 8]
  pipeline_parallel_sizes: [1, 4]
  micro_batch_size: 1
  num_microbatches: [1, 4]
  sequence_lengths: [512, 1024, 2048, 4096, 8192]
  recompute: full

training:
  max_steps: 12
  warmup_steps: 4
```

Four of the six `(TP, PP)` combinations are feasible at world size 8, giving **40
candidates**: `tp1/pp1/dp8`, `tp1/pp4/dp2`, `tp4/pp1/dp2`, and `tp8/pp1/dp1`, ten each.

Three settings in that config are load-bearing, and each one was learned by getting it
wrong:

**`max_steps: 12` with `warmup_steps: 4`.** Under a single warmup step the first profiled
step was the slowest in 18 of 24 runs, where chance alone would give about 8. One step does
not reach steady state. Discarding the second as well cut the median spread within a
candidate from 16.4% to 0.5%. The eight retained samples then support a median, which two
samples cannot. The extra steps are nearly free, since a candidate spends about 2.5
minutes in distributed setup and well under a second per step.

**Two `num_microbatches` values, not one.** The latency design row is
`[1, slots*tokens, slots, slots*sum_squared_tokens]` where `slots = num_microbatches +
PP - 1`. At a fixed microbatch count the slot column is constant, hence collinear with the
intercept, so any number of sequence lengths caps the design at rank 2 where the fit needs
rank 4. `{1, 4}` beats `{1, 2}` at the same candidate count because it opens the slot lever
wider. A third value buys nothing.

**Sequence lengths reaching 8192.** The `tokens` feature is the *largest* microbatch, not
the mean, so one long sequence in a batch sets it. The DAPO trace's median sequence is 8828
tokens and its longest is 11133, so an earlier grid stopping at 2048 was 5.4x too small and
every simulated step raised `TrainingOutOfDomainError`.

## The commands

```bash
bash examples/profile/megatron/profile_qwen3_8b.sh generate
bash examples/profile/megatron/profile_qwen3_8b.sh run       # --resume to continue
bash examples/profile/megatron/profile_qwen3_8b.sh collect

srl-regress training --latency-model hydraulis --memory-model hydraulis \
    --input  outputs/profile/qwen3_8b/megatron_bridge/profiles/megatron_bridge.jsonl \
    --output outputs/cost/qwen3_8b/training_hydraulis.json
```

Smoke-test first on `examples/profile/megatron/qwen3_8b_smoke.yaml`, which is minutes
rather than hours.

A note on `--resume`: it skips candidates already marked `success`, which means it
**re-runs every out-of-memory candidate**. That is rarely what you want on a second pass,
since an OOM is a result, not a failure to retry.

## How long it takes

Measured 2026-09-03 on 8x H20, read from `run.duration_s` in the JSONL:

| | |
|---|---|
| candidates | 40 |
| median per candidate | 169.8 s |
| total | 2.23 h |
| median training step | 1.34 s |

Roughly 143 s of each 170 s is fixed process overhead: spawning eight processes, importing
torch and transformers in every rank, building a 36-layer model, forming the NCCL group,
tearing it down. Actual measurement is about 11 seconds of it.

The consequence is that **cutting `max_steps` is not the lever it looks like**. Going from
12 steps to 8 saves about 2 s out of 170, roughly 1%, while halving the sample count the
median depends on. The real lever would be amortising startup across candidates that share
a topology, which is not implemented, and is not free: candidates would then share
allocator state, compromising the independence of the memory measurement.

## What lands on disk

```text
outputs/profile/qwen3_8b/megatron_bridge/
├── control/                                    candidate list, per-candidate status
├── raw/megatron_bridge/<hash>/                 resolved ConfigContainer.yaml, step logs
└── profiles/megatron_bridge.jsonl              40 records, all status: success
```

One record's payload:

```javascript
{
  "num_microbatches": 1,
  "micro_batch_size": 1,
  "global_batch_size": 8,
  "microbatches": [                        // one entry per microbatch on this rank
    {
      "count": 1,                          // sequences packed into it
      "lengths": [512],
      "max_tokens": 512,
      "sum_tokens": 512,
      "sum_squared_tokens": 262144         // Sum(s_i^2), the attention feature
    }
  ],
  "step_times_s": [0.5031, 0.5069, 0.5515, 0.5820, ...],   // 8 retained of 12
  "warmup_steps": 4,
  "peak_allocated_bytes": 63099938816,     // 58.77 GiB, what the memory model fits
  "peak_reserved_bytes":  66790096896,     // 62.20 GiB, an allocator artefact
  "rank_memory": [                         // all eight ranks, so PP imbalance is visible
    {"rank": 0, "allocated_bytes": 63098900992, "reserved_bytes": 66790096896},
    {"rank": 1, "allocated_bytes": 63098277376, "reserved_bytes": 66787999744},
    ...
  ],
  "schedule": "1f1b",
  "recompute": "full",
  "optimizer": "adam",
  "parameter_dtype": "bf16",
  "gradient_dtype": "fp32",
  "optimizer_state_dtype": "fp32",
  "distributed_optimizer": true,
  "parameter_count": 8190735360,
  "timers_s": {}
}
```

Two fields in there are the ones to understand.

**`sum_squared_tokens`, not the square of `sum_tokens`.** Attention does not cross sequence
boundaries, so a microbatch holding several packed sequences costs the sum of their squared
lengths. Squaring the packed total instead overcharges by the sequence count: eight
2048-token sequences give 3.36e7 against 2.68e8, an 8x error. The Megatron sweep here uses
`micro_batch_size: 1` so every microbatch holds one sequence and the two agree, but the
FSDP sweep in {doc}`04-fsdp-training` does pack, where it matters.

**Both `allocated` and `reserved` are recorded, and the model fits `allocated`.** Reserved
bytes are the caching allocator's high-water mark, not a footprint. Under full activation
recomputation, doubling the sequence length moved `allocated` by +304 MiB and `reserved` by
*minus* 4 MiB, giving $r^2 = 0.196$ on reserved against 1.0 on allocated. A feature that
moves the wrong way when the input doubles is not a target.

## The fitted artifact

`outputs/cost/qwen3_8b/training_hydraulis.json` holds eight entries, one per
`(component, cell)`: four latency and four memory.

```javascript
{
  "coefficients": {
    "per_step_overhead":     1e-10,
    "token_cost":            0.0005456,     // seconds per token
    "token_quadratic_cost":  1.030e-07,     // against sum_squared_tokens
    "microbatch_overhead":   0.1616         // seconds per pipeline slot
  },
  "feature_ranges": {
    "tokens":           [512.0, 8192.0],
    "pipeline_slots":   [1.0, 4.0],
    "micro_batch_size": [1.0, 1.0]
  },
  "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1},
  "selector": {
    "component": "latency",                 // or "memory"
    "schedule": "1f1b", "recompute": "full", "optimizer": "adam",
    "parameter_dtype": "bf16", "gradient_dtype": "fp32",
    "optimizer_state_dtype": "fp32", "distributed_optimizer": true
  },
  "train_metrics":      {"n_samples": 8, "design_rank": 4, "r_squared": 0.99998, "mre_percent": 3.845},
  "validation_metrics": {"n_samples": 2, "design_rank": 4, "r_squared": 0.99350, "mre_percent": 2.322},
  "diagnostics": {
    "formula": "per_step_overhead + (num_microbatches + PP - 1) * (token_cost * max_microbatch_tokens + token_quadratic_cost * max_microbatch_sum_squared_tokens + microbatch_overhead)",
    "step_aggregation": "median-per-candidate",
    "dp_latency": "ideal-local-workload-scaling",
    "validation_fraction": 0.2,
    "validation_is_in_sample": false
  }
}
```

The four coefficients each name something physical: a cost paid once per step, work
proportional to tokens, work quadratic in sequence length (attention), and a fixed cost
per pipeline slot. {doc}`../../cost-models/training-megatron` derives the form and explains
why the quadratic term is needed above 2048 tokens and harmful below it.

Cells are fitted independently and looked up by exact match. A query for `tp2` raises
rather than interpolating between `tp1` and `tp4`, because the token cost per rank and the
communication overhead differ qualitatively across those regimes, and an average would
describe neither. {doc}`../concepts` covers cells and selectors.

## How to tell it worked

```bash
python -c "
import json
a = json.load(open('outputs/cost/qwen3_8b/training_hydraulis.json'))
for e in a['entries']:
    p, s = e['parallelism'], e['selector']
    print(f\"{s['component']:8s} tp{p['tp']}pp{p['pp']}  rank={e['train_metrics']['design_rank']}\"
          f\"  train {e['train_metrics']['mre_percent']:6.3f}%  val {e['validation_metrics']['mre_percent']:6.3f}%\")
"
```

**`design_rank: 4` in every cell is the first thing to check.** Four columns, rank 4, so
all four coefficients are separately identified. A rank of 3 means the grid collapsed, most
often because `num_microbatches` was pinned, and the fit is then not trustworthy no matter
what the MRE says.

The figures this artifact carries, and how to read them, are in
{doc}`../../cost-models/training-megatron`. One caution belongs here: the `tp4/pp1` cell
reports 23.28% held-out MRE, which is an artifact of reserving 20% of 10 candidates. Two
points, where a single miss dominates. Its leave-one-out error is 12.46%. **On a grid this
small, prefer leave-one-out to a two-point split.**

## Use it

```bash
TRAINING_BACKEND=hydraulis \
TRAINING_ARTIFACT=outputs/cost/qwen3_8b/training_hydraulis.json \
bash examples/simulate/simulate.sh
```

## Next

{doc}`04-fsdp-training` measures the same thing under FSDP2, which shards differently and
therefore needs a separate model rather than a reused one.
