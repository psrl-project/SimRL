# Stage 3: Megatron training

Rollout is half the loop. This stage measures the other half: a real training step under
Megatron-Bridge, across several parallelism topologies, and fits a latency model and a memory
model from it.

## What it needs

A checkpoint and eight GPUs on one node (`hardware.gpu_count` is the world size, and DP is
whatever TP and PP leave over). Weights are never loaded: the launcher sets
`pretrained_checkpoint = None`, so the checkpoint is read for its architecture only.

The grid is `examples/profile/megatron/qwen3_8b.yaml`:

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

`srl-profile megatron generate` turns that into **40 candidates**. Of the six `(TP, PP)`
combinations, four are feasible at world size 8 (`tp1/pp1/dp8`, `tp1/pp4/dp2`, `tp4/pp1/dp2`,
`tp8/pp1/dp1`), and each is crossed with two microbatch counts and five sequence lengths for
ten candidates apiece. The Megatron recipe (`qwen3_8b`) and the architecture constraints are
inferred from the checkpoint's `config.json`.

Three settings in that config are load-bearing:

**`max_steps: 12` with `warmup_steps: 4`.** The first profiled steps are not steady state.
Discarding four leaves eight samples, enough for a median, which the fit uses per candidate
because step times are right-skewed: interference makes a step slower, never faster than the
hardware allows.

**Two `num_microbatches` values.** The latency design row is
`[1, tokens, slots, sum_squared_tokens]` with `slots = num_microbatches + PP - 1`. At a fixed
microbatch count the slot column is constant, so it is collinear with the intercept and the
design cannot reach rank 4 whatever the sequence lengths. Vary both axes. `srl-regress`
refuses a rank-deficient cell unless told `--allow-rank-deficient`.

**Sequence lengths that reach the traffic you will simulate.** The `tokens` feature is the
*largest* microbatch, so one long sequence in a batch sets it, and a query outside the fitted
range is refused (stage 6). The grid above stops at 8192; a real DAPO trace has longer
sequences.

## The commands

```bash
bash examples/profile/megatron/profile_qwen3_8b.sh generate
CANDIDATE_INDEX=all bash examples/profile/megatron/profile_qwen3_8b.sh run
CANDIDATE_INDEX=all bash examples/profile/megatron/profile_qwen3_8b.sh collect

srl-regress training --latency-model hydraulis --memory-model hydraulis \
    --input  outputs/profile/qwen3_8b/megatron_bridge/profiles/megatron_bridge.jsonl \
    --output outputs/cost/qwen3_8b/training_hydraulis.json
```

The wrapper takes `generate`, `run`, or `collect`, reads `CONFIG` (default
`examples/profile/megatron/qwen3_8b.yaml`), and **defaults `CANDIDATE_INDEX` to 0** so an
accidental `run` cannot launch the whole sweep. Set `CANDIDATE_INDEX=all` for the full sweep.
Extra arguments after the action are forwarded to `srl-profile`, so the underlying flags are:

```text
srl-profile megatron run --config CFG [--candidates PATH] [--candidate-id ID | --index N]
                         [--dry-run] [--resume]
                         [--nnodes N --node-rank R --master-addr HOST --master-port PORT]
```

`--dry-run` writes each candidate's `launch_spec.json` and logs the `torchrun` command without
starting it:

```text
torchrun --standalone --nnodes 1 --nproc-per-node 8 -m srl.profiler.adapters.megatron.launch \
    --spec outputs/profile/qwen3_8b/megatron_bridge/raw/megatron_bridge/<id>/launch_spec.json
```

Smoke-test first on `examples/profile/megatron/qwen3_8b_smoke.yaml` (a two-layer model), and
use `CUDA_VISIBLE_DEVICES=0 CONFIG=examples/profile/megatron/qwen3_8b_smoke.yaml ...` to prove
the chain on a single GPU before launching a full-model candidate.

A mesh wider than one host takes the same launcher on every host with its own `--node-rank`
and a shared `--master-addr`/`--master-port`; there is no separate scheduler path.

`--resume` skips a candidate whose `status.json` says `success`, which means it
**re-runs every out-of-memory or failed candidate**. That is rarely what you want on a second
pass, since an OOM is a result.

## What lands on disk

```text
outputs/profile/qwen3_8b/megatron_bridge/
├── control/megatron_bridge/candidates.json     the candidate list
├── raw/megatron_bridge/<candidate_id>/         one directory per candidate
│   ├── launch_spec.json                        what the launcher was told to run
│   ├── ConfigContainer.yaml                    the resolved Megatron-Bridge config
│   ├── srl_metrics.json                        step times and per-rank memory, written by the launcher
│   ├── oom_rank<N>.json                        only if a rank caught an out-of-memory
│   ├── run.log
│   └── status.json                             success | oom | failed, plus duration
└── profiles/megatron_bridge.jsonl              one normalised record per candidate
```

`candidates.json` is `{"schema_version": 1, "candidates": [...]}`, each entry carrying
`parallelism`, `micro_batch_size`, `global_batch_size`, `num_microbatches`, `sequence_shape`,
`schedule`, `recompute`, `packing`, and a `candidate_id`. Status comes from the OOM marker
files: only a rank that caught `torch.OutOfMemoryError` writes one, and a log that merely
mentions the phrase stays `failed`.

The launcher owns the training loop, so step times and per-rank CUDA peaks are measured through
Megatron's callback API in-process. It does not use the vendored `scripts/performance` CLI.

One record's payload (abridged):

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
  "step_times_s": [ ... ],                 // retained steps after warmup
  "warmup_steps": 4,
  "peak_allocated_bytes": 63099938816,     // what the memory model fits
  "peak_reserved_bytes":  66790096896,     // caching-allocator high-water mark
  "rank_memory": [                         // every rank, so PP imbalance is visible
    {"rank": 0, "allocated_bytes": 63098900992, "reserved_bytes": 66790096896}, ...
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

**`sum_squared_tokens`, not the square of `sum_tokens`.** Attention does not cross sequence
boundaries, so a microbatch holding several packed sequences costs the sum of their squared
lengths. Squaring the packed total overcharges by the sequence count. With
`micro_batch_size: 1` every microbatch holds one sequence and the two agree;
{doc}`04-fsdp-training` packs.

**Both `allocated` and `reserved` are recorded, and the Megatron memory model fits
`allocated`.** Reserved bytes are the allocator's high-water mark, not a footprint, and can
move the wrong way when the input changes.

### Packing: BSHD or THD

The grid above profiles `packing: bshd` (the default), where every sequence in a microbatch is
padded to the longest, so `sum_squared_tokens` equals `sum_tokens` squared and the linear and
quadratic token columns are collinear. A real RL trainer packs sequences behind `cu_seqlens`
(THD) and runs the varlen attention kernel, whose cost is `O(sum(s_i^2))`.
`examples/profile/megatron/qwen3_8b_thd.yaml` shows the THD variant: `search_space.packing: thd`,
a fixed `data_parallel_sizes`, and `sequence_shapes` where each entry is one packed microbatch.
A coefficient fitted on BSHD does not price a THD step. Fit from the layout your trainer uses.

## The fitted artifact

A fit of the sweep above produces a `CostArtifact` with `model_type: hydraulis` and seven
entries, one `(component, cell)` each:

| component | cell | design rank | train MRE | held-out samples |
|---|---|---|---|---|
| latency | `tp1` (pp 1 and 4 pooled) | 4 | 13.8% | 4 |
| latency | `tp4` | 4 | 6.2% | 2 |
| latency | `tp8` | 4 | 9.5% | 2 |
| memory | `tp1/pp1` | 2 | 1.4% | 2 |
| memory | `tp1/pp4` | 2 | 3.4% | 2 |
| memory | `tp4/pp1` | 2 | 0.6% | 2 |
| memory | `tp8/pp1` | 2 | 0.4% | 2 |

Latency cells are keyed by TP (and CP, EP) only, and the pipeline depth is derived: every
workload term is divided by `PP` and the slot count is `num_microbatches + PP - 1`, so one
profiled PP per TP is enough to predict another. Memory is not a per-stage division, so it is
looked up by the exact `(TP, PP)`. A query for an unprofiled TP raises `KeyError` rather than
interpolating.

A latency entry:

```javascript
{
  "coefficients": {
    "per_step_overhead":     0.02789,
    "token_cost":            0.0003619,     // seconds per token
    "token_quadratic_cost":  8.182e-08,     // against sum_squared_tokens
    "microbatch_overhead":   0.2458         // seconds per pipeline slot
  },
  "feature_ranges": {
    "tokens":             [512.0, 8192.0],
    "sum_squared_tokens": [262144.0, 268435456.0],
    "pipeline_slots":     [1.0, 7.0],
    "micro_batch_size":   [1.0, 1.0]
  },
  "selector": {
    "component": "latency", "schedule": "1f1b", "recompute": "full", "optimizer": "adam",
    "parameter_dtype": "bf16", "gradient_dtype": "fp32", "optimizer_state_dtype": "fp32",
    "distributed_optimizer": true
  },
  "diagnostics": {
    "formula": "per_step_overhead + (token_cost * (sum_over_microbatches(sum_tokens) + (PP - 1) * max_microbatch_tokens) + token_quadratic_cost * (sum_over_microbatches(sum_squared_tokens) + (PP - 1) * max_microbatch_sum_squared_tokens) + microbatch_overhead * (num_microbatches + PP - 1)) / PP",
    "profiled_pipeline_degrees": [1, 4],
    "validation_fraction": 0.2,
    "validation_is_in_sample": false,
    ...
  }
}
```

The four coefficients name something physical: a cost paid once per step, work proportional
to tokens, work quadratic in sequence length (attention), and a fixed cost per pipeline slot.
The token terms are sums over the rank's real microbatches, with the `PP - 1` bubble slots
charged at the largest microbatch. A memory entry has coefficients `activation_scale`,
`parameter_count`, `residual_constant` and the model-geometry terms, and feature ranges for
`activation_bytes` and `profiled_dp`. {doc}`../../cost-models/training-megatron` derives both.

## How to tell it worked

```bash
python -c "
import json
a = json.load(open('outputs/cost/qwen3_8b/training_hydraulis.json'))
for e in a['entries']:
    p, s = e['parallelism'], e['selector']
    print(f\"{s['component']:8s} tp{p['tp']}pp{p['pp']}  rank={e['train_metrics']['design_rank']}\"
          f\"  train {e['train_metrics']['mre_percent']:6.3f}%\"
          f\"  val {e['validation_metrics']['mre_percent']:6.3f}% (n={e['validation_metrics']['n_samples']})\")
"
```

**`design_rank: 4` on every latency cell is the first thing to check**, and `2` on a memory
cell (two fitted columns). A lower latency rank means the grid collapsed, most often because
`num_microbatches` was pinned, and the fit is untrustworthy whatever the MRE says.

Do not over-read the held-out column. The split reserves about 20% of runs, which on a few
candidates per cell is two to four points, and a single miss dominates. The split is by
`run_id`, so repeats of one configuration cannot straddle it. On a grid this small, prefer the
train metrics plus a leave-one-out check of your own to a two-point split.

## Use it

```bash
srl rollout_cost=roofline rollout_cost.artifact_path=outputs/cost/qwen3_8b/rollout_roofline.json \
    training_cost=hydraulis \
    training_cost.artifact_path=outputs/cost/qwen3_8b/training_hydraulis.json \
    training_cost.parallelism.tp=1 training_cost.parallelism.dp=8 \
    training_cost.activation_recompute=full \
    workload=bimodal job.n_versions=3 job.batch_size=16
```

The `selector` keys of the cell are config keys of `training_cost`
(`activation_recompute`, `optimizer`, the three dtypes, `distributed_optimizer`), and a
mismatch finds no cell. The response count per step (`batch_size * group_size`) must be
divisible by `dp * micro_batch_size`: `batch_size=4` with `dp=8` raises
`global_batch_size must be divisible by dp * micro_batch_size`. {doc}`06-simulate` shows the
domain gates firing on a real trace.

## Next

{doc}`04-fsdp-training` measures the same thing under FSDP2, which shards differently and
therefore needs a separate model rather than a reused one.
