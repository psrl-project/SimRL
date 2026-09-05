# Stage 5: collect a workload trace

Stages 1 through 4 measured how fast the system runs a given shape. This stage measures
what the shapes actually are: how long prompts are, how long responses turn out to be, and
which responses belong to the same prompt.

It is optional. A parametric distribution works without it. But group-relative advantage
(DAPO, GRPO) normalises rewards within a group, and that normalisation is undefined if the
group boundary is unknown, so a parametric workload where every response is independent
cannot faithfully represent a GRPO job.

## What it needs

A checkpoint, a dataset, and a set of hosts. Collection generates from a real model on
real prompts, so there is no offline shortcut.

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export DATASET_PATH=/path/to/dapo-math-17k.parquet
```

Hosts go in `examples/workload/hosts.txt`, one per line with a GPU count:

```text
# host [gpus]
29.162.247.148 8
28.49.16.220 8
```

One vLLM server per GPU, all generating in parallel.

## The command

```bash
HOSTS_FILE=examples/workload/hosts.txt \
DATASET=$DATASET_PATH MODEL=$MODEL_PATH \
ENV_SCRIPT=/path/to/env-activation.sh \
bash examples/workload/collect_joint_trace.sh
```

`DRY_RUN=1` prints the per-worker commands without launching anything, which is worth doing
first.

`ENV_SCRIPT` is sourced on every remote worker, so it has to be a path that exists on the
workers rather than only on the machine you start from.

The defaults collect 1024 groups of 8 responses with a 10240-token cap:

```bash
SAMPLE_SIZE=1024 GROUP_N=8 MAX_TOKENS=10240 TP=1 CONCURRENCY=16 SEED=42
```

`STARTUP_STAGGER_S=25` serialises server boots per host. Concurrent loads of a five-shard
checkpoint from shared storage are slow enough to trip the readiness timeout.

One thing about the dataset. `dapo-math-17k` has 1.79M rows but only **17,917 unique
prompts**, so sampling rows without deduplicating first gives you the same prompt many
times over. The collector deduplicates.

## How long it takes

About 49 minutes for 1024 groups of 8 on 16 GPUs, measured 2026-08-25. That is 8192
generations. Per-GPU throughput on H20 was roughly 1886 tokens/s.

## What lands on disk

```text
outputs/workload/qwen3_8b_dapo_tp1/
├── joint.jsonl                    the trace
└── shards/
    ├── worker-00.jsonl            per-worker output
    ├── worker-00.log              collector log
    ├── worker-00.server.log       vLLM server log
    └── ...                        16 workers
```

**The trace's first line is a metadata header, and every line after it is one group.** That
structure matters when reading the file by hand: skip line one, or check for `_meta`.

The header, from the trace collected above:

```javascript
{
  "_meta": true,
  "n_groups": 1024,
  "group_size": 8,
  "stats": {
    "n_groups": 1024,
    "group_size": 8,
    "max_output_tokens": 10240,
    "prompt": {
      "mean": 149.705, "median": 138.0, "p95": 245.85, "min": 75, "max": 893,
      "histogram": {"bin_edges": [75.0, 115.9, ...], "counts": [264, 432, 187, 80, ...]}
    },
    "output": {
      "mean": 7804.28, "median": 8689.5, "p95": 10240.0, "min": 989, "max": 10240,
      "histogram": {"bin_edges": [989.0, 1451.55, ...], "counts": [14, 48, 114, ..., 3548]}
    },
    "truncated_fraction": 0.407592
  }
}
```

A data line:

```json
{"group_id": 0, "prompt_len": 93, "output_lens": [7143, 7677, 7558, 6236, 7230, 7740, 7644, 6818]}
```

Three fields per group: the group id, the one prompt length, and the *n* response lengths.
That is the whole point of the format. The group boundary survives, and so does the
correlation between a prompt and its own responses.

## What the header tells you

Two numbers in there decide which parametric family could approximate this distribution,
and both rule out the obvious choice.

**The mean (7804) is below the median (8690).** That is left skew. A lognormal is
right-skewed by construction, so no choice of parameters makes it fit. Fitting one anyway
would misrepresent the shape.

**40.8% of responses hit the cap.** The distribution is truncated, not merely skewed: a
large point mass sits at exactly 10240, visible in the histogram as 3548 counts in the last
bin against 223 in the one before it. No smooth two-parameter family reproduces that.

The honest parametric approximation is `bimodal`, and
{doc}`../../workload/index` shows how to set its parameters from these statistics. There is
deliberately **no distribution-fitting command**: estimating a family from a trace and then
sampling from it throws away the correlation and grouping that made the trace worth
collecting. If you have a trace, replay it.

## How to tell it worked

```bash
python -c "
import json
with open('outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl') as f:
    meta = json.loads(f.readline())['stats']
    rows = [json.loads(l) for l in f]
print(f\"{len(rows)} groups, group_size={meta['group_size']}\")
print(f\"prompt  mean {meta['prompt']['mean']:.1f}  median {meta['prompt']['median']}\")
print(f\"output  mean {meta['output']['mean']:.1f}  median {meta['output']['median']}\")
print(f\"truncated {meta['truncated_fraction']:.1%}\")
assert all(len(r['output_lens']) == meta['group_size'] for r in rows)
"
```

Every group should carry exactly `group_size` responses. A short group means a worker died
mid-collection and its shard was merged incomplete.

A `truncated_fraction` near zero means the cap never bound and the tail is real. Near one
means the cap is doing all the work and the trace mostly records your `MAX_TOKENS` setting
rather than the model's behaviour.

## The consequence for stages 3 and 4

This trace is what tells you whether the fitted training domains are wide enough. The
median sequence here is 8828 tokens (prompt 138 plus output 8690) and the longest is 11133.
The Megatron latency model was fitted over `tokens` in [512, 8192], so the tail sits 1.36x
outside it, and its memory model over `activation_bytes` up to 2.941e9 against a real
3.997e9.

Both are out of domain, and the simulator refuses rather than guessing. Stage 6 covers how
to permit that bounded extrapolation deliberately.

{doc}`../../workload/trace-collection` has the collection internals.

## Next

{doc}`06-simulate` puts all of it together.
