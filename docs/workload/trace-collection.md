# Joint trace collection

A joint trace records what a real model produced: each prompt's measured token length, all *n*
of its response lengths, and the group boundary tying them together. Collecting one means
running real inference, so it costs GPU time.

Use this path when you need fidelity to a specific model and dataset, or when your algorithm
computes group-relative advantage. {doc}`index` covers when a parametric distribution is the
better choice.

---

## What you need

A HuggingFace checkpoint, a parquet dataset, one or more hosts with GPUs, and the pinned stack
from {doc}`../getting-started/installation` (vLLM serves the model, `transformers` renders the
chat template, `pyarrow` reads the dataset, `aiohttp` sends requests).

The dataset needs a `prompt` column of chat messages (`list[{"role", "content"}]`) and an
`extra_info` column with an `index` key. Prompts are deduplicated on that index, in
`load_unique_prompt_messages`, before any sampling: `dapo-math-17k` repeats every problem about
a hundred times, so sampling rows would draw mostly duplicates. The distinct prompts are then
subsampled to `--sample-size` with a seeded RNG, so a fixed `SEED` reproduces the same prompts.

---

## Host inventory

`examples/workload/hosts.txt` lists one host per line with an optional GPU count. Blank lines and
`#` comments are ignored, and a missing count means 1:

```text
# host [gpus]
10.0.0.1 2
10.0.0.2
```

Expand it into the worker table before committing to a run:

```bash
srl-workload list-workers --hosts-file examples/workload/hosts.txt
```

Each row is `index host gpu port`, one worker per GPU. Ports start at `--base-port` (default
8700) on each host, since every host has its own port space:

```text
0 10.0.0.1 0 8700
1 10.0.0.1 1 8701
2 10.0.0.2 0 8700
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

`DATASET`, `MODEL`, and `ENV_SCRIPT` are required and the script aborts naming whichever is
missing. `ENV_SCRIPT` is sourced on **every worker** before the collector starts, so it must be
a path that exists on the workers and not only on the host you launch from. A worker whose host
is one of this machine's addresses runs locally. Any other host is reached over `ssh` with
`BatchMode`.

Set `DRY_RUN=1` to print the per-worker commands without starting a server. Do that first on
any new inventory:

```text
[remote] worker-0 host=10.0.0.1 gpu=0 port=8700: ssh -o BatchMode=yes ... "source /path/env.sh ...;
  cd <repo> && python -m srl.cli.workload collect-groups --dataset ... --tokenizer ... --model ...
  --output <OUT_DIR>/shards/worker-00.jsonl --worker-index 0 --n-workers 3 --sample-size 1024
  --n 8 --max-tokens 10240 --tp 1 --concurrency 16 --port 8700 --gpu 0 --seed 42
  --serve-arg gpu_memory_utilization=0.85 --serve-arg max_model_len=12288"
```

The knobs, with their defaults:

| Variable | Default | Meaning |
|---|---|---|
| `SAMPLE_SIZE` | 1024 | distinct prompts across all workers |
| `GROUP_N` | 8 | responses sampled per prompt |
| `MAX_TOKENS` | 10240 | generation cap per response |
| `TP` | 1 | `--tensor-parallel-size` of each server |
| `CONCURRENCY` | 16 | in-flight prompts per worker |
| `BASE_PORT` | 8700 | first port on each host |
| `SEED` | 42 | prompt-subsampling seed |
| `STARTUP_STAGGER_S` | 25 | seconds between worker launches |
| `TOKENIZER` | `MODEL` | tokenizer used to render the chat template |
| `OUT_DIR` | `outputs/workload/qwen3_8b_dapo_tp1` | where the shards and trace go |
| `SHARD_DIR`, `TRACE` | `$OUT_DIR/shards`, `$OUT_DIR/joint.jsonl` | override either path |

`STARTUP_STAGGER_S` is not arbitrary: concurrent loads of a multi-shard checkpoint from shared
storage can exceed the server readiness timeout, so launches are spaced out. The script pins
each worker to a single GPU, so it is a `TP=1` collector. For a larger `--tp` call
`srl-workload collect-groups` yourself without `--gpu`. `max_model_len` is set to
`MAX_TOKENS + 2048` so the longest response and its prompt fit.

### What the script does

1. Expands the hosts file into workers with `list-workers`.
2. Skips any worker whose shard already exists and is complete, so a rerun resumes.
3. Launches `srl-workload collect-groups` per worker, locally or over `ssh`.
4. Waits for all workers and reports the log path of any that failed. A failed worker does not
   cancel the others. Rerun the script and only the incomplete shards retry.
5. Merges the shards by `group_id` with `merge-shards`, then runs `verify-collection`.

`Ctrl-C` kills the vLLM servers the script started, scoped to `vllm serve <MODEL> --port <N>`
for the script's own ports so that unrelated servers are left alone.

### What a worker does

Prompts are assigned by strided sharding: worker *i* of *n* takes positions `i, i+n, i+2n, ...`
of the sampled list, so a worker's slice depends on its index alone and a failed shard retries
without touching the others. `group_id` is the prompt's position in the full sampled list,
which is what lets the merge restore the order by sorting.

Each worker renders the chat template (falling back to concatenating message contents when the
tokenizer has none), starts `vllm serve`, and sends one `/v1/completions` request per prompt
with `n` responses and `return_token_ids: true`. Lengths are read from the returned token ids:
the prompt length from `prompt_token_ids`, and each response length from that choice's
`token_ids`. A prompt whose request fails is logged and skipped, and the worker raises only if
every request failed. The shard file is rewritten from scratch each attempt.

The CLI behind it:

```text
srl-workload collect-groups --dataset PARQUET --tokenizer PATH --model PATH --output SHARD.jsonl
    [--worker-index 0] [--n-workers 1] [--sample-size 1024] [--n 8] [--max-tokens 10240]
    [--temperature 1.0] [--top-p 1.0] [--tp 1] [--concurrency 16] [--port 8700] [--seed 42]
    [--gpu N] [--server-ready-timeout 1800] [--serve-arg KEY=VALUE ...]
```

---

## Output

```text
outputs/workload/qwen3_8b_dapo_tp1/
├── joint.jsonl                 the merged trace
└── shards/
    ├── worker-00.jsonl         one shard per GPU
    ├── worker-00.log           worker stdout and stderr
    └── worker-00.server.log    vLLM server output
```

The trace is JSONL with a metadata header on line 1 and one group per line after it.

A **shard** header carries the provenance of one worker: `kind`, `dataset`, `tokenizer`,
`model`, `tp`, `n`, `max_output_tokens`, `temperature`, `top_p`, `seed`, `worker_index`,
`n_workers`, and `n_groups_requested`, plus `n_groups`, `group_size`, and a `stats` block. The
**merged** header keeps the provenance fields that agree across shards, adds `n_shards`, and
carries the `stats` block recomputed over all groups: for both `prompt` and `output` the mean,
median, p95, min, max, and a 20-bin histogram, plus `truncated_fraction` (responses at or above
`max_output_tokens`) under `output`.

Each subsequent line is one group:

```json
{"group_id": 0, "prompt_len": 93, "output_lens": [7143, 7677, 7558, 6236, 7230, 7740, 7644, 6818]}
```

`output_lens` holds exactly `group_size` entries and every group must agree. A ragged trace is
rejected when written.

---

## Verifying a collection

A shard counts as complete only when its body holds at least the `n_groups_requested` recorded
in its own header. Checking the header count alone would accept a run interrupted after the
header was written but before the records were.

```bash
srl-workload check-shard --path outputs/workload/qwen3_8b_dapo_tp1/shards/worker-00.jsonl

srl-workload verify-collection \
    --trace outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl \
    --expected-groups 1024 \
    --shard-dir outputs/workload/qwen3_8b_dapo_tp1/shards \
    --n-workers 16
```

`check-shard` prints nothing and communicates through its exit status (0 when complete), which
is what the script's resume logic keys on. `verify-collection` prints
`Trace complete: 1024/1024 groups collected.` on success. On a shortfall it exits 1, reports the
gap, and lists each incomplete worker:

```text
Merged trace is short by 2 groups (10/12).
Short shards:
  worker-03: 40/64 groups
```

so you learn which workers to rerun, not only that the total is wrong.

`merge-shards` can also be run by hand:

```bash
srl-workload merge-shards --shard-dir SHARDS --n-workers 16 --output joint.jsonl \
    [--max-output-tokens 10240]
```

It checks that every expected shard exists and holds groups, and that all shards agree on
`max_output_tokens`. Two shards generated under different caps do not belong in one trace.
`--max-output-tokens` is an assertion against the shards' own value, not an override. Omit it
to let the shards be authoritative.

---

## Simulating from it

```bash
srl workload=dapo_trace \
    workload.trace.path=outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl \
    job.n_versions=16 job.batch_size=64 rollout.n_instances=16 \
    job.max_prompt_length=1024 job.max_output_length=10240
```

`workload.trace.n_groups=0` reads every group, and a positive value caps how many are loaded,
which is useful for a quick smoke run. `job.batch_size` counts **groups**: with 8 responses per
group, `batch_size=64` consumes 64 prompts (512 responses) per training step.
{doc}`../getting-started/walkthrough/06-simulate` runs a trace against fitted cost models.

---

## The reference trace

The reference trace used across this site was collected with the defaults above from Qwen3-8B
on `dapo-math-17k`: 1024 prompts times 8 responses across 16 workers, `max_tokens=10240`,
temperature 1.0, TP=1. It lives at `outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl` in the
working tree it was produced in. `outputs/` is not tracked by git, so a fresh clone must
collect its own.

| | mean | median | p95 | min | max |
|---|---|---|---|---|---|
| prompt tokens | 149.7 | 138 | 245.8 | 75 | 893 |
| output tokens | 7804.3 | 8689.5 | 10240 | 989 | 10240 |

**40.8% of responses hit the 10240-token cap.** That single number shapes how this workload
behaves in simulation, and it is why a smooth two-parameter distribution cannot stand in for
it. {doc}`index` shows the parametric approximation that can.

Two modelling points apply to traces generally. The simulator holds lengths and provenance, not
token ids, so responses of one group share prompt blocks in the KV cache but two different
prompts never do ({doc}`../simulation/prefix-cache`). And a trace that is reconstructed from a
real run's logs instead of collected here can have fabricated grouping, which
{doc}`../reference/limitations` describes.
