# vLLM rollout profiling

The vLLM pipeline measures end-to-end scheduler iterations from an official
`vllm bench sweep serve` run. Its output is a JSONL file of `ProfileRecord` objects,
one per scheduler step, which `srl-regress rollout` then fits into a rollout cost
model. Of the four profiling pipelines, this one is by far the cheapest, roughly six minutes
for a 30-point sweep, because the adapter runs an existing vLLM benchmark
rather than launching independent training workers. See {doc}`index` for the three-verb
contract and the side-by-side sweep timing across all four pipelines.

## Commands

Install the pinned vLLM and Megatron-Bridge packages before first use, then set
`MODEL_PATH` to the checkpoint once per shell:

```bash
conda activate simrl
bash third_party/install.sh
pip install -e ".[dev]"
export MODEL_PATH=/path/to/Qwen3-8B
```

The convenience wrapper runs all three verbs in sequence:

```bash
bash examples/profile/vllm/profile_qwen3_8b.sh all
```

Or call the verbs individually when you want to inspect candidates before committing
GPU time, resume a partial sweep, or redirect the collected output:

```bash
srl-profile vllm generate --config examples/profile/vllm/qwen3_8b_tp1.yaml
srl-profile vllm run      --config examples/profile/vllm/qwen3_8b_tp1.yaml \
                              [--resume] [--dry-run] [--max-parallel auto|N]
srl-profile vllm collect  --config examples/profile/vllm/qwen3_8b_tp1.yaml \
                              [--output PATH]
```

`generate` touches no GPU. `--dry-run` prints what `run` would launch without
starting a server. `--resume` skips candidates already marked `success`, so it
**re-runs every previously failed point**, so use `--dry-run` to confirm the
remaining list before a second pass.

### Fitting the cost model

Once collection finishes, fit a rollout cost model from the JSONL:

```bash
srl-regress rollout --model roofline \
    --input  outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
    --output outputs/cost/qwen3_8b/rollout_roofline.json
```

`--model` accepts `roofline`, `psrl`, or `distserve`. Rollout model formulas and their
fitted accuracy are covered in {doc}`../cost-models/rollout`.

## Sweep shape

The reference config `examples/profile/vllm/qwen3_8b_tp1.yaml` divides the sweep into
two phases.

**Prefill phase**, 24 points: six prompt lengths (256, 512, 1024, 2048, 4096, 8192
tokens) × four concurrencies (1, 4, 16, 64 requests). Each point is a single-output
benchmark (`random-output-len: 1`), so every run is one scheduler step: all `R`
requests arrive together, the engine processes them in one prefill batch, and returns
one token each. That direct mapping, one sweep point to one data record, is why the
prefill phase must vary both prompt length and concurrency. With only one scheduler
step per point there is nothing to walk across time.

**Decode phase**, 6 points: one prompt length (1024 tokens), one output length (8192
tokens), six concurrencies (1, 4, 16, 32, 48, 64 requests). A decode point is long by
design: each step generates one token per live request, so an 8192-token output walks
`ctxsum` from its initial value up to the final KV-cache context depth. That walk is
exactly what the decode attention feature captures, which is why one long-output point
with varying concurrency is sufficient for this phase.

Measured on Qwen3-8B / 8x H20 / 2026-08-31:

| Phase | Wall clock |
|---|---|
| Prefill sweep | ~1 min |
| Decode sweep | ~5 min |
| `collect` | seconds |
| `srl-regress rollout` (49 k points) | ~10 s |

**Total ≈ 6 minutes.** The 30 sweep points expand to 49,208 `ProfileRecord` objects
because each scheduler iteration in every decode run becomes one record.

## Parallel sharding and startup serialization

When multiple GPUs are visible and the config specifies `tensor-parallel-size: 1`, the
adapter shards the benchmark points across GPU slots and runs them concurrently.
`--max-parallel auto` (the default) fills all visible slots, while `--max-parallel 1`
restores serial execution. The YAML controls the same policy via `run.max_parallel`.

Concurrent shards boot **one at a time** regardless of `max_parallel`. Parallel imports
of torch, vLLM, and the checkpoint from a shared filesystem are slow enough that eight
simultaneous imports blew past `server_ready_timeout` in testing, and admitting one startup
at a time keeps each server inside the timeout budget. Raise `run.max_concurrent_startups`
when the checkpoint and environment live on a fast local disk. Each shard gets an
isolated server port, experiment directory, and scheduler log. If a shard fails, the
others continue, and re-running the same command retries only the unfinished shards.

`CUDA_VISIBLE_DEVICES` scopes the GPU pool. `MAX_PARALLEL=1` in the environment is
equivalent to `--max-parallel 1` for scripts that wrap the CLI.

## Async scheduling must be disabled

Every server runs with `--no-async-scheduling`. This is enforced by the adapter
config validator. Specifying `async-scheduling` or `no-async-scheduling` in
`serve.args` raises a validation error. The reason is subtle: async scheduling
overlaps a step's execution with its neighbours' scheduling, so the elapsed time
logged for that step no longer measures its own forward pass. A 65k-token prefill can
report microseconds while its real cost is absorbed into the next step's timestamp.
Steps that still carry an impossibly short elapsed time despite `--no-async-scheduling`
are dropped at collection and a warning is emitted.

## Preemption truncation

Collection truncates a decode run at its first preemption. Once the KV cache fills,
vLLM evicts requests and later re-prefills each with its prompt plus the tokens it
already generated. From that point the in-flight requests no longer share a common age,
so `ctxsum` cannot describe any subsequent step. The warning names the run and reports
how many steps were kept and dropped.

To avoid losing the tail of a decode point, keep `max-concurrency × (input + output)`
within the KV-cache capacity. The config comment on `qwen3_8b_tp1.yaml` states this
constraint directly. If a point is routinely truncated, reduce its `max-concurrency` or
`random-output-len`.

## Attention feature formulas

The adapter fills `sum_l2` and `ctxsum` from closed-form equal-length expressions
rather than profiler annotations. The formulas are derived from the equal-length
constraint that `random-range-ratio: 0.0` enforces, and that constraint is validated
at config load time.

For a **prefill** step (or a prefill chunk under `max-num-batched-tokens`): `R` requests
each of length `L`, so `T = R * L` total tokens, and the squared-length sum
concentrates at the equal-length value:

```
n_tokens = T
sum_l2   = T² / R
```

That closed form holds when `R` divides `T`. When it does not, the adapter does not round:
`_balanced_chunk_sum_l2` spreads the remainder one token per request, matching how a token
budget is actually split across concurrent prefills. With `base, rem = divmod(T, R)`:

```
sum_l2 = rem * (base + 1)² + (R - rem) * base²
```

which reduces to `T² / R` exactly when `rem == 0`.

For a **decode** step: `R` generation requests on the `k`-th step after prefill
(0-based). Each request holds a prompt of length `L` plus the `k` tokens already
generated, so its KV-cache context is `L + k` entries long. One new token is added
this step, making the attended context length `L + 1 + k`. Summing over requests:

```
n_tokens = R
ctxsum   = R * (L + 1 + k)
```

A **mixed** step that contains both prefill and generation tokens contributes both
`sum_l2` and `ctxsum`. The configured phase collection still discards the opposite pure
phase. A decode-phase config keeps mixed steps but drops any pure-prefill steps.

Client TTFT and ITL include queuing, networking, event-loop delay, and possible chunk
coalescing. They are preserved in the untouched benchmark JSON for diagnostics but are
never clustered into synthetic GPU steps. Missing scheduler logs fail collection rather
than falling back silently to client timing.

For how these features enter the rollout cost model equations, see
{doc}`../cost-models/rollout`. For the general rules around warmup and dispersion
statistics, see {doc}`methodology`.

## Output layout

```
outputs/profile/qwen3_8b/vllm_tp1/
├── raw/vllm_timing/qwen3-8b-{prefill,decode}-timing[-sNNN]/   ← official bench JSON
├── control/vllm/{prefill,decode}/                              ← scheduler iteration logs
└── profiles/vllm.jsonl                                         ← 49,208 records
```

Parallel shards append a shard suffix (`-s000`, `-s001`, …) to their raw and control
directories. `collect` merges them by their original serve/bench combination path, so
the final JSONL is always a flat file regardless of how many shards ran.

One record is written **per scheduler iteration**, not per request. This is why 30
sweep points expand to 49 k records: the decode phase alone logs one record per
generated token per concurrency point.

## Record format

A representative record from the prefill phase (comments annotate non-obvious fields):

```javascript
{
  "record_type": "rollout",
  "schema_version": "1.0",
  "run": {
    "run_id": "vllm-097f7b34d69c83cee6e0",
    "status": "success",
    "duration_s": 0.29,
    "raw_artifact": "raw/vllm_timing/...",
    "repeat": 0
  },
  "model": {
    "name": "...",           // MODEL_PATH, never an absolute path in your config
    "num_layers": 36, "hidden_size": 4096,
    "num_attention_heads": 32, "num_query_groups": 8,
    "parameter_count": 8190735360
  },
  "hardware": {"gpu_name": "H20", "gpu_count": 1, "gpu_memory_mib": 97871},
  "software": {"backend": "vllm", "backend_version": "0.26.0"},
  "parallelism": {"tp": 1, "pp": 1, "dp": 1, "cp": 1, "world_size": 1},
  "payload": {
    "phase": "prefill",
    "latency_s": 0.02506,    // ← the regression target
    "n_tokens": 256,
    "sum_l": 256.0,
    "sum_l2": 65536.0,       // = n_tokens² / R = 256² / 1
    "ctxsum": 0.0,
    "request_count": 1,
    "input_tokens": 256, "output_tokens": 0,
    "ttft_s": [], "itl_s": []  // kept for diagnostics, never regressed
  },
  "tags": {
    "latency_source": "scheduler-iteration-elapsed-ms",
    "feature_source": "equal-length-formula",
    "scheduler_iteration": {"index": 0, "ctx_tokens": 256, "gen_tokens": 0,
                             "elapsed_ms": 25.06}
  }
}
```

The full `ProfileRecord` schema is documented in {doc}`../reference/schemas`.

## Smoke-testing a new config

To reduce a config for a quick sanity check, copy the YAML and reduce each phase
grid, `num_runs`, and `bench.args.num-prompts`. Keep at least one output-one prefill
point and one decode point with more than one output token. Preview what the sweep
would do before touching a GPU:

```bash
bash examples/profile/vllm/profile_qwen3_8b.sh generate
srl-profile vllm run --config examples/profile/vllm/qwen3_8b_tp1.yaml --dry-run
```
