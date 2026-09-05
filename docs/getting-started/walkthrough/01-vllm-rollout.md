# Stage 1: vLLM rollout

The first real measurement. A vLLM server is started, benchmarked across a grid of
prompt lengths and concurrencies, and every scheduler iteration it logs becomes one
record. A regression then reduces those records to five coefficients.

This is the cheapest profiling stage, about six minutes, and it is the one that makes
simulated rollout latency reflect your hardware rather than a default.

## What it needs

A checkpoint and at least one GPU:

```bash
export MODEL_PATH=/path/to/Qwen3-8B
```

The sweep grid lives in `examples/profile/vllm/qwen3_8b_tp1.yaml`: 24 prefill points
crossing six prompt lengths with four concurrencies, plus six decode points.

```{literalinclude} ../../../examples/profile/vllm/qwen3_8b_tp1.yaml
:language: yaml
:start-at: "phases:"
:end-before: "  decode:"
```

Two details in that grid matter. Every point sets `num-prompts` equal to
`max-concurrency`, so one point costs a single scheduler step in prefill rather than
several request waves. And decode uses one long output per `(input, concurrency)` pair,
because decode walks context forward across steps and a single request therefore
sweeps the whole KV range for free.

## The command

```bash
bash examples/profile/vllm/profile_qwen3_8b.sh all
```

That runs the three verbs every profiling backend exposes, in order:

```bash
dryrun-profile vllm generate --config CFG   # enumerate sweep points, no GPU touched
dryrun-profile vllm run      --config CFG   # start servers, benchmark
dryrun-profile vllm collect  --config CFG   # normalise into JSONL
```

`generate` is free and writes a job manifest, so it is the cheapest way to see what a
sweep will attempt before committing GPU hours. `collect` re-reads whatever the run left
behind, so post-processing choices like the warmup window can change without
re-measuring.

## How long it takes

About 6 minutes for 30 sweep points on 8x H20, measured 2026-08-31. The eight GPUs run
eight servers concurrently. `max_concurrent_startups: 8` throttles boot rather than
benchmarking, because parallel imports of torch plus a checkpoint read from network
storage are slow enough to trip `server_ready_timeout`.

## What lands on disk

```text
outputs/profile/qwen3_8b/vllm_tp1/
├── control/vllm/{prefill,decode}/     job manifest, per-shard serve and bench params, logs
├── raw/vllm_timing/                   what `vllm bench sweep serve` wrote
└── profiles/vllm.jsonl                49,208 normalised records
```

**30 sweep points produce 49,208 records** because one record is one *scheduler
iteration*, not one sweep point and not one request. A decode point generating 8192
tokens logs 8192 iterations. The split is 49,144 decode, 58 prefill, and 6 mixed. All
49,208 carry `status: success`.

That ratio is the point of the design. A single benchmark run yields tens of thousands of
independent (batch shape, latency) observations, which is why a six-minute sweep can fit
a five-coefficient model well.

One record, the first line of `profiles/vllm.jsonl`:

```javascript
{
  "record_type": "rollout",
  "schema_version": "1.0",
  "payload": {
    "phase": "prefill",
    "latency_s": 0.02506,       // this scheduler iteration's elapsed time
    "n_tokens": 256,            // tokens processed in the iteration
    "request_count": 1,
    "input_tokens": 256,
    "output_tokens": 0,
    "sum_l": 256.0,             // sum of prefill lengths
    "sum_l2": 65536.0,          // sum of squared prefill lengths, the attention term
    "ctxsum": 0.0,              // summed KV context of decoding requests, zero here
    "batch_step_index": 0
  },
  "run": {
    "run_id": "vllm-097f7b34d69c83cee6e0",
    "status": "success",        // success | oom | failed | skipped
    "duration_s": 0.2908,
    "repeat": 0,
    "started_at": "2026-08-18T16:37:15.855992+00:00",
    "raw_artifact": "raw/vllm_timing/qwen3-8b-prefill-timing-s000/..."
  },
  "model": {
    "name": "/path/to/Qwen3-8B",
    "architecture": "Qwen3ForCausalLM",
    "parameter_count": 8190735360,
    "num_layers": 36, "hidden_size": 4096,
    "num_attention_heads": 32, "num_query_groups": 8,
    "extra": {"head_dim": 128, "intermediate_size": 12288, "vocab_size": 151936}
  },
  "hardware": {"gpu_name": "H20", "gpu_count": 1, "gpu_memory_mib": 97871, ...},
  "software": {"backend": "vllm", "backend_version": "0.26.0", "source_commit": "568afb3a1...", ...},
  "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1, "world_size": 1},
  "tags": {"source": "vllm-scheduler-iteration-log", "scheduler_iteration": {...}, ...}
}
```

Five metadata groups accompany every payload: `run`, `model`, `hardware`, `software`,
`parallelism`. They are what make a record self-describing, so a JSONL file from one
machine can be told apart from another without external bookkeeping. `world_size` is
validated to equal `tp * pp * dp * cp`.

The same shape with `phase: decode` carries the complementary features, `ctxsum: 1025.0`
and `sum_l2: 0.0`, since a pure decode iteration has no prefill work. {doc}`../concepts`
covers `ProfileRecord` as a schema, {doc}`../../reference/schemas` has every field.

## Fitting the cost model

```bash
dryrun-regress rollout --model roofline \
    --input  outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
    --output outputs/cost/qwen3_8b/rollout_roofline.json
```

Seconds, no GPU. The result is a `CostArtifact` with one entry:

```javascript
{
  "kind": "rollout",
  "model_type": "roofline",
  "profile_digest": "sha256:9b7f45b7d83d...",   // changes if the input changes
  "entries": [{
    "coefficients": {
      "F": 0.006247,        // per-iteration floor, seconds
      "W": 0.005954,        // fixed per-step work
      "G": 0.0001006,       // seconds per token
      "A_p": 3.405e-08,     // prefill attention, against sum_l2 / block_size
      "A_d": 7.626e-08,     // decode attention, against ctxsum
      "b": 16.0             // block size
    },
    "feature_ranges": {
      "n_tokens":      [1.0, 65536.0],
      "request_count": [1.0, 64.0],
      "sum_l2":        [0.0, 536870912.0],
      "ctxsum":        [0.0, 466089.0]
    },
    "train_metrics": {
      "n_samples": 49208, "design_rank": 4,
      "r_squared": 0.99989, "mre_percent": 5.252,
      "rmse": 0.001543, "max_error": 0.08408, "n_outliers": 53
    },
    "validation_metrics": {"n_samples": 0, ...},   // nothing held out
    "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1},
    "selector": {"block_size": 16}
  }],
  ...
}
```

Predicted latency is
`max(F, W + G*n_tokens + A_p*sum_l2/b + A_d*ctxsum)`.

Two fields deserve attention:

**`feature_ranges` is a hard gate, not documentation.** A query with `n_tokens` above
65536 raises rather than returning a number, because the fit says nothing about that
region. {doc}`../../simulation/domain-gates` covers how to permit extrapolation when you
mean to.

**`validation_metrics.n_samples: 0` means nothing was held out.** The 5.252% MRE is
in-sample. That is the honest reading, and the code encodes it as a zero sample count
rather than by omitting the field. {doc}`../../reference/validation` lists which models on
this project have a genuine split and which do not.

Two other rollout models fit from the same JSONL:

```bash
bash examples/regress/regress_rollout_psrl.sh
bash examples/regress/regress_rollout_distserve.sh
```

`psrl` reaches 5.164% MRE at design rank 3. `distserve` produces figures identical to
`roofline` because it re-expresses the same fit through architecture constants rather
than fitting independently. {doc}`../../cost-models/rollout` has all three formulas.

## How to tell it worked

Read `train_metrics` out of the artifact:

```bash
python -c "
import json
e = json.load(open('outputs/cost/qwen3_8b/rollout_roofline.json'))['entries'][0]
print(e['train_metrics'])
print(e['feature_ranges'])
"
```

`design_rank: 4` against four fitted terms means every coefficient is separately
identified. A rank below the term count means two features moved together across the
whole sweep and the solver cannot tell their contributions apart. `n_samples` in the tens
of thousands confirms `collect` read the iteration log rather than only per-request
summaries.

An MRE in the 5% range on this grid is normal. An MRE above 15% usually means the sweep
hit preemption: a decode point whose `max-concurrency * (input + output)` exceeds the KV
cache gets truncated at its first preemption, and the truncated tail pollutes the fit.

## Use it

```bash
ROLLOUT_MODEL=roofline bash examples/simulate/simulate.sh
```

## Next

{doc}`02-kernels` measures the same engine from underneath, one CUDA kernel at a time.
That path costs an hour instead of six minutes and needs no regression at all, and it is
what lets you price a configuration the end-to-end sweep never ran.
