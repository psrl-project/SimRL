# Group-aware DAPO workload traces and an end-to-end Qwen3-8B tp=1 simulation case

**Date:** 2026-08-24
**Status:** Approved for implementation
**Scope:** Task A only (group-aware workload). Tasks B and C are named here as
follow-ups but are explicitly out of scope.

## Problem

We want to simulate one concrete workload end to end: Qwen3-8B at rollout tp=1
over `dapo-math-17k`, matching the PSRL DAPO recipe. Today that is impossible
for three reasons.

1. **No real workload data.** `examples/simulate/distributions.yaml` holds
   synthetic `normal` parameters. The two fitting scripts under
   `examples/workload/` point at stale paths (`/jizhicfs/lhy/data/...`,
   `/apdcephfs_zwfy10_303541817/.../Qwen3-8B`) that do not match the target
   model or dataset.

2. **The prompt/output join is fictional.** `Simulator._next_prompt_length` and
   `_next_length` independently index two separate lists by `rid % len(...)`
   (`srl/simulator/simulator.py:120-127`). For synthetic distributions this
   is harmless. For real data it fabricates pairings that never occurred, and it
   cannot express DAPO's `n=8` group structure at all.

3. **`simulate_with_fitted_cost.sh` hardcodes `workload=normal`** (line 68)
   while reading distribution parameters from a YAML file. Any trace-based
   distribution config is rejected by Hydra, because keys like `trace_path` do
   not exist on the `normal` config group.

### Why groups matter

DAPO samples `n=8` responses per prompt and computes group-relative advantages.
A group is therefore a real structural unit of the workload, not a detail:
the 8 responses share one prompt, and their lengths vary widely within the group
(measured: one prompt produced 6300..10240 tokens across its 8 responses).

The naive fix (derive `group_id` from `rid // 8`) is a silent correctness trap.
When `len(trace)` is not a multiple of the group size, the `rid % len` wraparound
splits a group across the boundary. Both follow-up tasks (B: whole-group batch
consumption, C: shared-prefix prefill) depend on group boundaries being exact, so
**group boundaries must come from the data, not from arithmetic.**

## Measured facts

All verified on this cluster on 2026-08-24 before writing this spec.

**Dataset** (`dapo-math-17k.parquet`): 1,791,700 rows but only **17,917 unique
prompts**, each repeated ~58-59× per row group (2 row groups). Unique identity is
`extra_info.index`. `prompt` is a single-message `[{content, role: "user"}]` list.
Prompt tokens over unique prompts with the Qwen3-8B chat template: mean 154,
median 141, p95 254, max 1273, min 72: only 0.1% exceed the recipe's
`max_prompt_length=1024`.

Consequence: `load_prompt_messages` random-samples *rows*, so sampling without
deduplicating on `extra_info.index` draws ~100× duplicate prompts.

**Output lengths** (vLLM tp=1, one H20, temperature 1.0, top_p 1.0,
`max_tokens=10240`, 64 completions over 8 unique prompts): mean 8218, median
8642, p95 10240, min 3347, max 10240. **31% truncate at the cap.** Qwen3-8B is a
thinking model; a short cap truncates everything and is useless for fitting.

**Throughput:** ~1886 output tokens/s at concurrency 16 (128 completions in
494s). A single GPU would need ~9 hours for 8192 completions.

**vLLM API:** `/v1/completions` accepts `n` and `return_token_ids: true`,
returning per-choice `token_ids` plus `prompt_token_ids`. Verified
`len(prompt_token_ids) == 196` matches client-side tokenization exactly.

> ⚠️ **`usage.completion_tokens` is the SUM over all `n` choices** (measured 4800
> for 8×600), not a per-response length. `collect_output_lengths` currently reads
> exactly this field (`srl/workload/inference.py:137`). Adding `n` without
> changing that line would silently record 8× lengths. Per-response lengths must
> come from `len(choice["token_ids"])`.

**Cluster:** two hosts, `29.162.247.148` (local) and `28.49.16.220`, 8 H20 each =
16 GPUs. Passwordless SSH works. `/apdcephfs_zwfy/...` (repo, model, dataset) and
the conda env with identical vLLM `0.26.1.dev0+g568afb3a1` are visible on both.

## Design

### 1. Joint group trace format

One JSONL artifact, one line per group, replacing the two independent traces on
this path.

```json
{"_meta": true, "n_groups": 1024, "group_size": 8, "dataset": "...", "model": "...", "tp": 1, "max_tokens": 10240, "seed": 42, "stats": {...}}
{"group_id": 0, "prompt_len": 411, "output_lens": [6300, 6636, 7581, 7750, 8272, 9834, 10240, 10240]}
```

`prompt_len` and `output_lens` come from the same HTTP response, so the join is
observed rather than fabricated, and the group boundary is explicit in the data.
`stats` carries prompt- and output-length summaries plus the truncation fraction.

The existing single-dimension trace format and `from_trace` stay as they are:
synthetic and single-dimension workflows keep working unchanged. This is not a
compatibility shim: the joint trace is a genuinely different artifact (a joint
distribution rather than a marginal one), so both formats have a reason to exist.

### 2. Distributed collection

A worker is a `(host, gpu_index)` pair addressed by a flat worker index.
Prompts are split deterministically as `prompts[worker_index:n_workers]`.

```
hosts.txt                    1024 deduped unique prompts
    │                                │
    ▼                                ▼
16 workers = (host, gpu)  →  prompts[i:16]  (deterministic from index i)
    │
    ▼
each worker: CUDA_VISIBLE_DEVICES=<gpu> vllm serve <model> --port <base+gpu> (tp=1)
             per prompt: n=8, return_token_ids=true, max_tokens=10240
             writes shards/worker-<NN>.jsonl
    │
    ▼
merge → joint.jsonl   (validates every expected shard is present)
```

- **Deterministic sharding by index** means rerunning worker 7 reproduces
  exactly worker 7's prompts, so one failed shard retries without disturbing the
  others. This mirrors the existing `examples/profile/vllm` shard precedent.
- **Local host runs directly; remote hosts via SSH.** One code path
  parameterized by host, with the `ssh` prefix omitted for the local host.
- **Per-worker shard files** avoid concurrent writes to one file. The shared
  filesystem means no artifact shipping.
- **Merge validates completeness** and names any missing shard rather than
  silently emitting a short trace.
- **Startup is staggered** per host. Concurrent loads of an 8-shard checkpoint
  from a shared filesystem are slow enough to trip readiness timeouts: the same
  hazard already documented in `examples/profile/README.md`.

`hosts.txt` is one host per line with an optional GPU count:

```
# host [gpus]
29.162.247.148 8
28.49.16.220 8
```

Prompt sampling deduplicates on `extra_info.index` before subsampling, so each
group is a distinct problem: matching a real DAPO step of 64 different problems.

### 3. Config surface

An explicit `workload.source` switch, so "these two blocks are actually one
joint distribution" cannot hide inside two independent-looking blocks:

```yaml
# srl/config/defaults/workload/dapo_trace.yaml
source: trace                # trace | distributions
trace:
  path: outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl
  n_groups: 0                # 0 = all groups
```

Existing groups (`normal`, `uniform`, ...) gain `source: distributions`.
`WorkloadConfig` grows `source: str` and a `trace: JointTraceConfig` field.

### 4. Simulator changes

- `Request.group_id: int = 0`: a new field on the request data model.
- A `GroupedWorkload` source in `srl/workload/` that owns the group list and
  yields `(group_id, prompt_len, output_len)` per request, wrapping at group
  boundaries.
- `Simulator` consumes that one source, replacing the two independent
  `rid % len(...)` lookups. `SimConfig` carries the grouped workload instead of
  a bare `prompt_lengths` list.
- `job.max_prompt_length` / `max_output_length` clipping still applies per
  request via the existing `clip_length`.

Requests within a group stay adjacent in admission order, which is what lets
task B consume whole groups and task C recognize a shared prefix without
changing the trace format again.

### 5. Case scripts

```
examples/workload/dapo_qwen3_8b/
├── README.md          # provenance, measured stats, runtime, caveats
├── hosts.txt          # the two hosts
└── collect_joint_trace.sh
examples/simulate/
├── dapo_qwen3_8b.yaml            # the simulate case config
└── simulate_with_fitted_cost.sh  # fixed: no hardcoded workload=normal
```

Simulation parameters follow the PSRL recipe
(`psrl/examples/dapo_trainer/qwen3_8b_megatron.sh`) at full scale: 16 rollout
instances, `max_prompt_length=1024`, `max_output_length=10240`,
`max_staleness=2`, partial rollout on, rollout tp=1, batch 64 prompts × n=8.
Rollout cost uses the existing tp=1 artifact
`outputs/cost/qwen3_8b/rollout_psrl.json`.

`n_versions=16`, chosen so the run nominally consumes 16 × 512 = 8192 responses : 
the exact length of the collected trace. However, the simulator admits ahead of
consumption, so `next_rid` reaches 9728 (1536 past the trace boundary), meaning
wraparound **does** occur. It is safe: wraparound lands on a group boundary by
construction, and all consumed batches come from the un-wrapped prefix. A case
that avoids wraparound entirely would require a longer trace or fewer versions.

Training cost stays `training_cost=fixed` with the script's existing
`TRAINING_LATENCY_S` default. No fitted Qwen3-8B training artifact exists under
`outputs/cost/qwen3_8b/`, and fitting one is a separate profiling exercise. The
case therefore measures rollout behaviour against a fixed training stall, which
is stated in the README so the training latency is not mistaken for a measured
value.

### 6. The `simulate_with_fitted_cost.sh` fix

Line 68's hardcoded `workload=normal` is replaced by a workload group selected
from the case config, so a trace-based workload is expressible. `DIST_FILE`
handling stays for the synthetic path.

## Out of scope

- **Task B: whole-group batch consumption.** `job.batch_size` should mean
  prompts per step, and a batch should consume all 8 responses of a group or
  none. Touches all five policies plus the PSRL `StalenessInventory` bridge
  (which keys buffer entries by `prompt_id = rid`).
- **Task C: shared prefix / prefix caching.** The 8 responses share a prompt
  prefix, so prefill should be paid once per group. Touches engine KV
  accounting (`blocks_for(r.prefilled)`) and the chunked-prefill loop.
- **DAPO dynamic-sampling filter** (dropping all-correct/all-wrong groups):
  needs rewards the simulator does not model.

### Documented caveats until B and C land

These go in the case README, stated plainly rather than papered over:

1. `job.batch_size` counts **responses**, not prompts. The case uses 512 = 64 × 8.
2. Prefill is charged **8× per group** rather than once, so rollout time is
   pessimistic by roughly one prefill per extra group member. With mean prompt
   154 tokens against mean output 8218, the error is small here, but it is real.

## Verification

1. Unit tests: joint-trace round-trip; group boundary integrity under wraparound
   when `n_groups` does not divide the request count; per-choice length
   extraction (guarding the `usage.completion_tokens` trap); deterministic
   shard split covering every prompt exactly once; merge detecting a missing
   shard.
2. A tiny end-to-end collection (2 workers × 2 prompts, small `max_tokens`)
   asserting a valid joint trace.
3. Full collection: 1024 groups / 8192 completions across 16 GPUs (~35-40 min),
   then confirm measured stats land near the probe values (mean output ~8200,
   ~31% truncation).
4. Full simulation on the collected trace, reporting throughput and the
   staleness distribution.
5. `pytest` green and `ruff` clean.
