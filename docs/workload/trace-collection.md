# Joint trace collection

A joint trace records what a real model actually produced: each prompt's measured token
length, all *n* of its response lengths, and the group boundary tying them together.
Collecting one means running real inference, so it costs GPU time — about 49 minutes on 16
H20s for the reference trace below.

Use this path when you need fidelity to a specific model and dataset, or when your
algorithm computes group-relative advantage. See {doc}`index` for when parametric
distributions are the better choice.

---

## What you need

A HuggingFace checkpoint, a parquet dataset, and one or more hosts with GPUs.

The dataset must have a `prompt` column holding chat messages as
`list[{"role", "content"}]`, and an `extra_info` column containing an `index` key. That
index is what deduplication keys on, and it matters more than it sounds: `dapo-math-17k`
has 1.79M rows but only 17,917 distinct problems, so sampling without deduplicating first
would draw the same prompt about a hundred times.

---

## Host inventory

`examples/workload/hosts.txt` lists one host per line with an optional GPU count. Blank
lines and `#` comments are ignored, and a missing count defaults to 1:

```text
# host [gpus]
10.0.0.1 8
10.0.0.2 8
```

Expand it into the worker table before committing to a run:

```bash
dryrun-workload list-workers --hosts-file examples/workload/hosts.txt
```

Each row is `index host gpu port`, one worker per GPU, with ports counting up from
`--base-port` (8700 by default):

```text
0 10.0.0.1 0 8700
1 10.0.0.1 1 8701
2 10.0.0.1 2 8702
3 10.0.0.1 3 8703
```

---

## Collect

```bash
HOSTS_FILE=examples/workload/hosts.txt \
DATASET=$DATASET_PATH \
MODEL=$MODEL_PATH \
ENV_SCRIPT=/path/to/env-activation.sh \
bash examples/workload/collect_joint_trace.sh
```

`DATASET`, `MODEL` and `ENV_SCRIPT` are required, and the script aborts naming whichever is
missing. `ENV_SCRIPT` is sourced on **every worker**, so it must be a path that exists on
the workers rather than only on the host you launch from.

Set `DRY_RUN=1` to print the per-worker commands without starting a single server. Do that
first on any new inventory.

Other knobs: `SAMPLE_SIZE` (1024), `GROUP_N` (8 responses per prompt), `MAX_TOKENS`
(10240), `TP` (1), `CONCURRENCY` (16), `BASE_PORT` (8700), `SEED` (42), `OUT_DIR`, and
`STARTUP_STAGGER_S` (25).

That last one is not arbitrary. Concurrent loads of a multi-shard checkpoint from a shared
filesystem are slow enough to trip the server readiness timeout, so boots are serialised
per host.

### What the script does

1. Expands the hosts file into workers via `list-workers`.
2. Skips any worker whose shard is already complete, so a rerun resumes.
3. Launches `dryrun-workload collect-groups` per worker — locally when the host matches,
   over `ssh` otherwise.
4. Waits for all workers, reporting log paths for any that failed.
5. Merges shards by `group_id` into one trace, then runs `verify-collection`.

Prompts are assigned by strided sharding — worker *i* of *n* takes prompts
`i, i+n, i+2n, …` — so each worker's slice is deterministic and the merge can restore the
original order by sorting on `group_id`.

---

## Output

```text
outputs/workload/qwen3_8b_dapo_tp1/
├── joint.jsonl                 <- the merged trace
└── shards/
    ├── worker-00.jsonl         <- one shard per GPU
    ├── worker-00.log           <- worker stdout/stderr
    └── worker-00.server.log    <- vLLM server output
```

The trace is JSONL with a metadata header on line 1 and one group per line after it.

The header carries provenance and summary statistics: `model`, `dataset`, `tokenizer`, `n`,
`tp`, `seed`, `temperature`, `top_p`, `n_groups`, `group_size`, `n_shards`,
`max_output_tokens`, and a `stats` block with mean, median, p95, min, max and a 20-bin
histogram for both prompt and output lengths, plus `truncated_fraction` for outputs.

Each subsequent line is one group:

```json
{"group_id": 0, "prompt_len": 93, "output_lens": [7143, 7677, 7558, 6236, 7230, 7740, 7644, 6818]}
```

`output_lens` holds exactly `group_size` entries, and every group must agree on that count —
a ragged trace is rejected rather than written.

---

## Verifying a collection

A shard counts as complete only when its body holds at least the `n_groups_requested`
recorded in its own header. Checking the header count alone would accept a run that was
interrupted after writing the header but before finishing the records.

```bash
dryrun-workload check-shard --path outputs/workload/.../shards/worker-00.jsonl

dryrun-workload verify-collection \
    --trace outputs/workload/.../joint.jsonl \
    --expected-groups 1024 \
    --shard-dir outputs/workload/.../shards \
    --n-workers 16
```

`check-shard` communicates through its exit status, which is what makes the resume logic
work. `verify-collection` names the specific workers responsible for any shortfall, rather
than only reporting that the total is wrong.

Shards are also cross-checked on merge: every shard must be present, and all must agree on
`max_output_tokens`. Provenance fields identical across shards are copied to the merged
header, and a disagreement is an error — two shards generated under different settings do
not belong in one trace.

---

## Simulating from it

```bash
dryrun workload=dapo_trace \
    workload.trace.path=outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl \
    job.n_versions=16 job.batch_size=512 rollout.n_instances=16 \
    job.max_prompt_length=1024 job.max_output_length=10240
```

`workload.trace.n_groups=0` reads every group; a positive value caps how many are loaded,
which is useful for a quick smoke run.

Note that `job.batch_size` counts **responses**, not prompts. With 8 responses per prompt,
`batch_size=512` consumes 64 prompts per training step.

---

## The reference trace

Collected 2026-08-25: Qwen3-8B on `dapo-math-17k`, 1024 prompts x 8 responses across 16 H20
GPUs in 49 minutes, `max_tokens=10240`, temperature 1.0, TP=1.

| | mean | median | p95 | min | max |
|---|---|---|---|---|---|
| prompt tokens | 149.7 | 138 | 245.8 | 75 | 893 |
| output tokens | 7804.3 | 8689.5 | 10240 | 989 | 10240 |

**40.8% of responses hit the 10240-token cap.** That single number shapes how this workload
behaves in simulation, and it is why a smooth two-parameter distribution cannot stand in for
this trace — see {doc}`index` for the parametric approximation that can.

Two modelling caveats apply to traces generally and are recorded in
{doc}`../reference/limitations`: prefill is charged once per response rather than once per
shared prompt, so rollout time is slightly pessimistic, and whole-group batch consumption is
not implemented.
