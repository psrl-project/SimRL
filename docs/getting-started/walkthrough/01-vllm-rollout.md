# Stage 1: vLLM rollout

The first real measurement. A vLLM server is started, benchmarked across a grid of prompt
lengths and concurrencies, and every scheduler iteration it logs becomes one record. A
regression then reduces those records to a handful of coefficients.

This is the cheapest profiling stage that touches a real engine, and it is what makes
simulated rollout latency reflect your hardware instead of a default.

## What it needs

A checkpoint and at least one GPU:

```bash
export MODEL_PATH=/path/to/Qwen3-8B
```

The sweep grid is `examples/profile/vllm/qwen3_8b_tp1.yaml`. Under `phases:` it lists 24
prefill points (six prompt lengths crossed with four concurrencies) and six decode points:

```{literalinclude} ../../../examples/profile/vllm/qwen3_8b_tp1.yaml
:language: yaml
:start-at: "phases:"
:end-before: "  decode:"
```

Two details matter. Every point sets `num-prompts` equal to `max-concurrency`, so a prefill
point costs one scheduler step instead of several request waves. And decode uses one long
output per `(input, concurrency)` pair, because decode walks the context forward across
steps, so a single request sweeps the whole KV range.

Servers always start with `--no-async-scheduling`: with async scheduling an iteration's
elapsed time stops measuring its own forward pass. `serve.args` and `serve.parameters` in the
config become the `vllm serve` flags, and `bench.args` the `vllm bench serve` flags.

## The commands

```bash
bash examples/profile/vllm/profile_qwen3_8b.sh all
```

`profile_qwen3_8b.sh` takes `generate`, `run`, `collect`, or `all`, and reads `TP` (default 1,
selecting `qwen3_8b_tp${TP}.yaml`), `CONFIG`, `PHASE` (`all`, `prefill`, or `decode`), and
`MAX_PARALLEL`. `all` runs the three verbs every profiling backend exposes, in order:

```bash
srl-profile vllm generate --config CFG   # enumerate the sweep, no GPU touched
srl-profile vllm run      --config CFG   # start servers and benchmark
srl-profile vllm collect  --config CFG   # normalise into JSONL
```

`generate` is free. It writes a job manifest and logs the exact `vllm bench sweep serve`
command for each shard, so it is the cheapest way to see what a sweep will attempt. `run`
accepts `--phase`, `--dry-run`, `--resume`, and `--max-parallel auto|N`. `collect` re-reads
whatever `run` left behind and accepts `--output`, so a post-processing change does not need
re-measuring.

With several GPUs visible, TP=1 points are sharded across them and benchmarked concurrently,
each shard with its own port, experiment directory, and scheduler log. The shards benchmark in
parallel but boot at most `run.max_concurrent_startups` at a time (the example config sets 8,
the code default is 1), because simultaneous imports of torch, vLLM, and a checkpoint from
shared storage can exceed `server_ready_timeout`.

## What lands on disk

```text
outputs/profile/qwen3_8b/vllm_tp1/
├── control/vllm/{prefill,decode}/   job manifest, per-shard serve and bench params
├── raw/vllm_timing/                 what `vllm bench sweep serve` wrote
└── profiles/vllm.jsonl              normalised records
```

`output_dir` in the config is relative to the config file, which is why the example YAML
spells it `../../../outputs/...`.

**One record is one scheduler iteration**, not one sweep point and not one request. A decode
point that generates 8192 tokens logs 8192 iterations, so a 30-point sweep yields tens of
thousands of independent (batch shape, latency) observations. That is why a short sweep can
fit a few coefficients well.

Collection truncates a run at its first preemption. Once the KV cache fills, vLLM evicts a
request and later re-prefills its prompt plus the tokens it had generated. From then on the
running requests no longer share one age, so `ctxsum` describes no later step. Keep
`max-concurrency * (input + output)` inside the KV cache or the tail of that point is lost.

One record, abridged:

```javascript
{
  "record_type": "rollout",
  "schema_version": "1.0",
  "payload": {
    "phase": "prefill",         // prefill | decode | mixed
    "latency_s": 0.02506,       // this scheduler iteration's elapsed time
    "n_tokens": 256,            // tokens processed in the iteration
    "request_count": 1,
    "input_tokens": 256,
    "output_tokens": 0,
    "sum_l": 256.0,             // sum of prefill lengths
    "sum_l2": 65536.0,          // sum of squared prefill lengths, the attention term
    "ctxsum": 0.0,              // summed KV context of decoding requests
    "batch_step_index": 0
  },
  "run": {"run_id": "vllm-...", "status": "success", "repeat": 0, "duration_s": 0.29, ...},
  "model": {"name": "/path/to/Qwen3-8B", "architecture": "Qwen3ForCausalLM",
            "parameter_count": 8190735360, "num_layers": 36, "hidden_size": 4096, ...},
  "hardware": {"gpu_name": "H20", "gpu_count": 1, "gpu_memory_mib": 97871, ...},
  "software": {"backend": "vllm", "backend_version": "0.26.0", "source_commit": "568afb3a...", ...},
  "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1, "world_size": 1},
  "tags": {"source": "vllm-scheduler-iteration-log", "feature_source": "equal-length-formula", ...}
}
```

`sum_l2` and `ctxsum` are filled from equal-length formulas, since every request in a point
has the same length. A pure decode record has `sum_l2: 0.0` and a non-zero `ctxsum`.
{doc}`../concepts` covers `ProfileRecord` as a schema and {doc}`../../reference/schemas` has
every field.

## Fitting the cost model

```bash
srl-regress rollout --model roofline \
    --input  outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
    --output outputs/cost/qwen3_8b/rollout_roofline.json
```

This takes seconds and needs no GPU. Besides the artifact it writes diagnostic plots to
`<output stem>_plots/` (here `outputs/cost/qwen3_8b/rollout_roofline_plots/`) when every
record came from vLLM. The artifact is a `CostArtifact` with one entry per profiled
topology. Fitting an earlier Qwen3-8B/H20 sweep gave:

```javascript
{
  "kind": "rollout",
  "model_type": "roofline",
  "artifact_version": "1.0",
  "profile_digest": "sha256:...",     // changes if the input changes
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
    "train_metrics": {"n_samples": 49208, "design_rank": 4, "r_squared": 0.99989,
                      "mre_percent": 5.252, "rmse": 0.001543, "max_error": 0.08408, "n_outliers": 53},
    "validation_metrics": {"n_samples": 0, ...},    // nothing held out by default
    "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1, "ep": 1, "world_size": 1},
    "selector": {"block_size": 16}
  }],
  ...
}
```

Predicted latency is `max(F, W + G*n_tokens + A_p*sum_l2/b + A_d*ctxsum)`, with the prefill
attention term applied to `prefill` and `mixed` steps and the decode term to `decode` and
`mixed` ones.

`--validation-fraction` defaults to 0.0 for rollout fits, so `validation_metrics.n_samples:
0` means nothing was held out and the MRE above is in-sample. Pass for example
`--validation-fraction 0.2` for a split that is grouped by run, so repeats of one point
cannot leak across it. On the same data, this gave a held-out MRE of 1.958% over 8202 samples
(the training MRE moves to 6.076% because the split changes what is trained on).
{doc}`../../reference/validation` lists which models on this project have a genuine split.

Two other rollout models fit from the same JSONL:

```bash
bash examples/regress/regress_rollout_psrl.sh
bash examples/regress/regress_rollout_distserve.sh
```

`psrl` fits ten coefficients in separate prefill and decode forms (design rank 3 on the sweep
above). `distserve` re-expresses the roofline fit through the architecture constants `h` and
`m`, so its fitted error equals the roofline's. All three scripts read `INPUT` and write
`OUTPUT`, both overridable, and their defaults are the paths on this page.
{doc}`../../cost-models/rollout` has the formulas.

## How to tell it worked

```bash
python -c "
import json
e = json.load(open('outputs/cost/qwen3_8b/rollout_roofline.json'))['entries'][0]
print(e['train_metrics'])
print(e['feature_ranges'])
"
```

`design_rank: 4` against four fitted linear terms means every coefficient is separately
identified. A smaller rank means two features moved together across the whole sweep and the
solver cannot split their contributions. An `n_samples` in the tens of thousands confirms
`collect` read the iteration log, not just per-request summaries. An MRE far above the 5% of
the sweep above usually means a decode point hit preemption and the truncated tail polluted
the fit.

## Use it

Pass the artifact to the cost model:

```bash
srl rollout_cost=roofline rollout_cost.artifact_path=outputs/cost/qwen3_8b/rollout_roofline.json \
    workload=bimodal job.n_versions=3 job.batch_size=16 rollout.n_instances=2
```

When `artifact_path` is set the artifact replaces the `F`, `W`, `G`, `A_p`, `A_d`, `b`
overrides, and the lookup is exact on `rollout_cost.parallelism` (`tp`, `pp`, `dp`, `cp`),
so a TP=2 simulation needs a TP=2 cell in the artifact. {doc}`06-simulate` combines this with
a fitted training model and a workload trace.

## Next

{doc}`02-kernels` measures the same engine from underneath, one CUDA kernel at a time. It
needs no regression at all, and it is what lets you price a configuration the end-to-end
sweep never ran.
