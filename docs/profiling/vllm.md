# vLLM rollout profiling

The vLLM pipeline measures end-to-end scheduler iterations from an official
`vllm bench sweep serve` run. Its output is a JSONL file of `ProfileRecord` objects, one per
scheduler iteration, which `srl-regress rollout` fits into a rollout cost model
({doc}`../cost-models/rollout`). It is also the baseline that `srl-profile kernels validate`
scores the kernel-composed model against ({doc}`kernels`).

Unlike the training pipelines, a server stays alive across many benchmark points, so the
per-candidate process cost described on {doc}`index` does not apply.

## Commands

Install the pinned vLLM and Megatron-Bridge packages (`third_party/install.sh`) and the project
in the active Python environment, then name the checkpoint once per shell:

```bash
export MODEL_PATH=/path/to/Qwen3-8B
```

The convenience wrapper runs all three verbs in sequence, with `TP`, `CONFIG`, `PHASE` and
`MAX_PARALLEL` selecting the config, phase and shard limit:

```bash
bash examples/profile/vllm/profile_qwen3_8b.sh all
```

Or call the verbs individually:

```bash
srl-profile vllm generate --config examples/profile/vllm/qwen3_8b_tp1.yaml [--phase all|prefill|decode]
srl-profile vllm run      --config examples/profile/vllm/qwen3_8b_tp1.yaml \
                              [--phase ...] [--resume] [--dry-run] [--max-parallel auto|N]
srl-profile vllm collect  --config examples/profile/vllm/qwen3_8b_tp1.yaml \
                              [--phase ...] [--output PATH]
```

- `generate` touches no GPU. It writes the serve and bench parameter files and the job manifests
  under `control/vllm/` and logs the exact `vllm bench sweep serve` command of every job.
- `--dry-run` prepares the same jobs and prints what `run` would launch without starting a
  server.
- `--resume` skips a job whose experiment directory already holds all its expected
  `run=*.json` files (serve points x bench points x `num_runs`). A partially finished job is
  relaunched with the official sweep's own `--resume`, so completed benchmark points inside it
  are not repeated.
- `--max-parallel` overrides `run.max_parallel` from the YAML. `CUDA_VISIBLE_DEVICES` scopes the
  GPU pool.
- `collect --output` redirects the JSONL. The default is `{output_dir}/profiles/vllm.jsonl`.

### Fitting the cost model

```bash
srl-regress rollout --model roofline \
    --input  outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
    --output outputs/cost/qwen3_8b/rollout_roofline.json
```

`--model` accepts `roofline`, `psrl` or `distserve`. The formulas are on
{doc}`../cost-models/rollout`.

### Probing KV capacity

```bash
srl-profile kv-probe --model $MODEL_PATH --tp 1 [--gpu-memory-utilization 0.9] \
    [--max-model-len 32768] [--dtype bfloat16] [--refresh]
```

`kv-probe` starts a real vLLM engine in a subprocess, reads back the KV block count it allocates
and stores it in an on-disk cache (`~/.cache/srl/kv_capacity.json`, overridable with
`SIMRL_KV_CACHE`). The simulator reads that cache, so a GPU box can populate it for a GPU-less
one. vLLM sizes its cache from memory left after weights, its own activation peak, non-torch
allocations and CUDA graphs. Those cannot be derived from `config.json`, which is why capacity
is read from an engine. The cache is keyed on the checkpoint fingerprint, TP degree, memory
budget, max model length, dtype, GPU model, vLLM version, block size, `max-num-batched-tokens`
and `max-num-seqs`. A key miss re-probes rather than interpolating. `--refresh` clears the file
first, for when the answer changed under a stable key such as a rebuilt vLLM.

## Config

The YAML has these top-level keys (`VLLMSweepConfig`):

| key | meaning |
|---|---|
| `model`, `hardware` | checkpoint (path or hub id) and the hardware descriptor |
| `output_dir`, `experiment_name` | tree root and the prefix of every raw experiment directory |
| `serve.args`, `serve.parameters` or `serve.grid` | fixed `vllm serve` flags, and the serve points to sweep |
| `bench.args` | fixed `vllm bench serve` flags |
| `phases.prefill`, `phases.decode` | benchmark points, as `parameters:` lists or a `grid:` cross product |
| `num_runs` | repeats of each benchmark point |
| `server_ready_timeout` | seconds to wait for a server to report ready |
| `run.max_parallel` | `auto` or a positive integer |
| `run.max_concurrent_startups` | how many shards may sit in server startup at once |
| `run.startup_stagger_seconds` | margin added to the startup wait |
| `environment` | environment variables for the sweep |
| `executable` | the `vllm` binary |

The config validator rejects a sweep that cannot be interpreted:

- `bench.args.random-range-ratio` must be `0`. The attention features below are derived from the
  equal-length constraint it enforces.
- `serve.args` must not enable a profiler. Timing uses scheduler iteration logs.
- `serve.args` must not set `async-scheduling` or `no-async-scheduling`. The adapter always
  passes `--no-async-scheduling` (next section).
- A prefill point must request exactly one output token, and a decode point more than one.
- At least one phase must be configured.

## Sweep shape

The reference config `examples/profile/vllm/qwen3_8b_tp1.yaml` has two phases.

**Prefill** points are a grid of prompt length and concurrency with `random-output-len: 1`. Each
point is one scheduler step in which all requests arrive together, are prefilled in a single
batch and return one token each. One sweep point therefore maps to one data record, which is why
the prefill phase must vary both prompt length and concurrency: with only one step per point
there is nothing to walk across time.

**Decode** points use one prompt length, a long output and several concurrencies. Each step
generates one token per live request, so a long output walks `ctxsum` from its initial value up
to the final KV depth. That walk is what the decode attention feature captures, so one long
output per concurrency is sufficient.

Keep `max-concurrency x (input + output)` within KV-cache capacity, or collection truncates the
run at the first preemption (below). Use more than one prompt per point whenever a figure from
that point will be quoted: a single-prompt benchmark records a single cold iteration, which can
carry fixed server-side work that no steady-state step pays.

## Parallel sharding and startup serialisation

`run` partitions work across the visible GPUs. For each serve point it needs
`world_size = tp x pp x dp` GPUs and creates

```text
shards = min(benchmark points, max_parallel, visible GPUs // world_size)
```

shards. Benchmark points are dealt round-robin to shards, each shard gets its own GPU set,
port, experiment directory and sweep log, and shard ids are `s000`, `s001`, and so on. If
the effective parallelism is one, each phase runs as a single unsharded job. A serve point that
needs more GPUs than are visible is rejected at preparation time.

Concurrent shards boot **one at a time** by default. Parallel imports of torch, vLLM and the
checkpoint from a shared filesystem are slow enough to exceed `server_ready_timeout`, so a startup
gate lets only `run.max_concurrent_startups` shards hold a startup permit, from launch until the
server reports ready. The default is `1`. Raise it when the checkpoint and environment live on a
fast local disk. If a shard fails, the others continue, and re-running with `--resume` retries
only the unfinished ones.

The runner forces `PYTHONUNBUFFERED=1` and overrides a `VLLM_LOGGING_LEVEL` that would suppress
iteration logs (such as `WARN`) back to `INFO`, because vLLM only emits `Iteration(...)` lines at
`INFO`.

## Async scheduling is disabled

Every server runs with `--no-async-scheduling` and `--enable-logging-iteration-details`. Async
scheduling overlaps a step's execution with its neighbours' scheduling, so the logged elapsed
time stops measuring that step's own forward pass: a large prefill can report almost nothing
while its cost is absorbed into a later step. Collection also guards against any step that still
looks impossible, as described next.

## Which iterations collection trusts

`collect` parses the sweep log into one `_IterationDetail` per scheduler step (fields `index`,
`ctx_requests`, `ctx_tokens`, `gen_requests`, `gen_tokens`, `elapsed_ms`) and keeps only the steps
that the equal-length formulas can describe.

**Truncation.** A run is cut at the first step that violates the equal-length assumption, and
every later step is dropped. There are three reasons, each logged as a warning that names the run,
the iteration and how many steps were kept and dropped:

- `context-longer-than-prompt`: a step schedules more context tokens than its context requests can
  hold at the configured input length, so a preempted request is re-prefilling its prompt plus
  tokens it already generated.
- `context-beyond-prompt-token-budget`: cumulative scheduled context tokens exceed the total
  prompt tokens the run could ever submit, which is again recompute of generated tokens.
- `multiple-generation-tokens-per-request`: a step generated other than one token per generation
  request, so the running set no longer shares one age.

After any of these, in-flight requests no longer share a common age and `ctxsum` cannot describe
any later step.

**Unmeasured steps.** A step is dropped individually, without ending the run, when its elapsed
time is an async-overlap artifact:

- `elapsed_ms < 1.0`, or
- the implied rate `(ctx_tokens + gen_tokens) / elapsed` exceeds `_MAX_MEASURED_TOKENS_PER_SECOND`
  (`1e6` tokens per second).

The absolute floor alone is not enough, because an overlap artifact on a large prefill lands just
above it yet reports an impossible rate. The rate cap is deliberately loose, far above anything
real hardware reaches, so it rejects artifacts without discarding genuine steps.

**Phase filter.** A prefill config keeps steps that are not pure decode, and a decode config
keeps steps that are not pure prefill. Mixed steps are kept in both.

Missing scheduler logs fail collection with an error naming the run. There is no fallback to
client-side TTFT or ITL: those include queueing, networking and event-loop delay, are preserved in
the benchmark JSON for diagnostics, and are never turned into synthetic GPU steps.

## Attention feature formulas

The adapter fills `sum_l2` and `ctxsum` from closed-form equal-length expressions, not from
profiler annotations.

For a **prefill** step (or a prefill chunk under `max-num-batched-tokens`) with $R$ context
requests and $T$ context tokens, the remainder of $T / R$ is spread one token per request, matching
how a token budget splits across concurrent prefills. With `base, rem = divmod(T, R)`:

```text
n_tokens = T
sum_l2   = rem * (base + 1)^2 + (R - rem) * base^2     # = T^2 / R when rem == 0
```

For a **decode** step with $R$ generation requests, $k$ steps after the first generation step of
the run, and prompt length $L$, each request attends a context of $L + 1 + k$:

```text
n_tokens = R
ctxsum   = R * (L + 1 + k)
```

$k$ is the engine's own iteration index minus the index of the first step that generated, so
dropping an untrusted step in the middle of a run does not shift the ages of the steps after it.
A **mixed** step carries both `sum_l2` and `ctxsum`. Features enter the cost models as described
on {doc}`../cost-models/rollout`.

## Output layout

```text
{output_dir}/
├── raw/vllm_timing/{experiment_name}-{phase}-timing[-sNNN]/   official bench JSON, run=*.json
├── control/vllm/{phase}/
│   ├── timing[_sNNN]_serve_params.json
│   ├── timing[_sNNN]_bench_params.json
│   ├── timing[_sNNN]_manifest.json
│   ├── timing[_sNNN]_sweep.log                                 scheduler iteration log
│   └── jobs_manifest.json
└── profiles/vllm.jsonl                                         normalised records
```

A shard suffix (`-s000`, `_s000`, and so on) appears on the raw directory, params files, manifest
and sweep log of every parallel shard. `collect` discovers the shard directories itself and merges
them into one flat JSONL, so the output is the same however many shards ran.

One record is written **per scheduler iteration**, not per request, so a sweep of a few dozen
points expands to many thousands of records: every decode run logs one record per generated token
per concurrency point.

## Record format

Records are `ProfileRecord` objects with `record_type: "rollout"`. The load-bearing parts:

```javascript
{
  "record_type": "rollout",
  "run": {"run_id": "vllm-<hash of phase and benchmark path>", "status": "success",
          "raw_artifact": "raw/vllm_timing/...", "repeat": 0},
  "model": { /* architecture read from the checkpoint */ },
  "hardware": { /* gpu_count is set to the run's world size */ },
  "software": {"backend": "vllm", "backend_version": "...", "source_commit": "..."},
  "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1},
  "payload": {
    "phase": "prefill",          // prefill | decode | mixed, from the step itself
    "latency_s": 0.0,            // the regression target
    "n_tokens": 0,
    "sum_l": 0.0, "sum_l2": 0.0, "ctxsum": 0.0,
    "request_count": 0,
    "input_tokens": 0, "output_tokens": 0,
    "ttft_s": [], "itl_s": []    // kept for diagnostics, never regressed
  },
  "tags": {
    "phase": "prefill",          // the config phase being collected
    "observed_phase": "prefill", // the step's own phase
    "source": "vllm-scheduler-iteration-log",
    "latency_source": "scheduler-iteration-elapsed-ms",
    "feature_source": "equal-length-formula",
    "timing_benchmark_artifact": "raw/vllm_timing/.../run=0.json",
    "scheduler_iteration": {"index": 0, "ctx_requests": 0, "ctx_tokens": 0,
                            "gen_requests": 0, "gen_tokens": 0, "elapsed_ms": 0.0}
  }
}
```

`software.backend_version` and `source_commit` are the pinned constants in
`srl/profiler/adapters/vllm/config.py`, not a reading of the installed package. The full
`ProfileRecord` schema is in {doc}`../reference/schemas`.

## Baselines for validating the kernel model

A baseline that will be used to score `ComposedRollout` (see {doc}`tensor-parallel`) has to be a
clean one:

- Collect it serially (`run.max_parallel: 1`, `run.max_concurrent_startups: 1`, as in
  `examples/profile/vllm/qwen3_8b_tp2_serial.yaml`). Servers sharing a machine and a filesystem
  can degrade one another mid-run without any error.
- The signature of that contamination is non-monotonicity, not outliers. At the same per-request
  depth a larger batch must be slower than a smaller one. When it is faster, the smaller point
  was slowed by something else, and filtering outliers will not repair it. Re-measure.
- Keep the grid identical across TP degrees if the degrees will be compared bucket for bucket.
  The TP2 configs repeat the TP1 `phases:` block for that reason.

## Smoke-testing a new config

Copy the YAML and reduce each phase grid, `num_runs` and `bench.args.num-prompts`. Keep at least
one output-one prefill point and one decode point with more than one output token. Preview the
sweep before touching a GPU:

```bash
bash examples/profile/vllm/profile_qwen3_8b.sh generate
srl-profile vllm run --config examples/profile/vllm/qwen3_8b_tp1.yaml --dry-run
```
