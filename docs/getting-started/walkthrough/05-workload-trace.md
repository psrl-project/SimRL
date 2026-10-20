# Stage 5: collect a workload trace

Stages 1 through 4 measured how fast the system runs a given shape. This stage measures what
the shapes actually are: how long prompts are, how long responses turn out to be, and which
responses belong to the same prompt.

It is optional. A parametric distribution runs without it. But GRPO and DAPO normalise
rewards within a group, and a parametric workload where every response is independent
(`group_size` 1) cannot represent that. {doc}`../../workload/index` compares the two sources.

## What it needs

A checkpoint, a parquet dataset, and one or more GPU hosts. Collection generates from a real
model on real prompts, so there is no offline shortcut.

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export DATASET_PATH=/path/to/dapo-math-17k.parquet
```

The dataset needs a `prompt` column of chat messages (`list[{"role", "content"}]`) and an
`extra_info` column holding an `index`. Prompts are deduplicated on that index before
sampling: `dapo-math-17k` repeats each problem about a hundred times, so sampling rows would
draw the same prompt over and over.

Hosts go in `examples/workload/hosts.txt`, one per line with an optional GPU count (a missing
count means one GPU):

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

`DATASET`, `MODEL`, and `ENV_SCRIPT` are required and the script aborts naming whichever is
missing. `ENV_SCRIPT` is sourced on every worker, so it must be a path that exists on the
workers and not only on the machine you launch from. `DRY_RUN=1` prints the per-worker
command without starting a server, and is worth doing first on any new inventory.

The defaults collect 1024 groups of 8 responses with a 10240-token cap:

```bash
SAMPLE_SIZE=1024 GROUP_N=8 MAX_TOKENS=10240 TP=1 CONCURRENCY=16 SEED=42 \
BASE_PORT=8700 STARTUP_STAGGER_S=25
```

`STARTUP_STAGGER_S` waits between worker launches so that concurrent loads of a multi-shard
checkpoint from shared storage do not trip the readiness timeout. Each worker is the CLI
`srl-workload collect-groups`. {doc}`../../workload/trace-collection` documents the script's
steps, the flags, and the verification commands.

## What lands on disk

```text
outputs/workload/qwen3_8b_dapo_tp1/
├── joint.jsonl                    the merged trace
└── shards/
    ├── worker-00.jsonl            one shard per GPU
    ├── worker-00.log              collector log
    ├── worker-00.server.log       vLLM server log
    └── ...
```

**The first line of the trace is a metadata header, and every line after it is one group.**
Skip line one, or check for `_meta`, when reading the file by hand.

The header from a trace collected this way (Qwen3-8B on `dapo-math-17k`, histograms omitted):

```javascript
{
  "_meta": true,
  "n_groups": 1024,
  "group_size": 8,
  "stats": {
    "n_groups": 1024,
    "group_size": 8,
    "max_output_tokens": 10240,
    "prompt": {"mean": 149.705, "median": 138.0, "p95": 245.85, "min": 75, "max": 893,
               "histogram": {"bin_edges": [...], "counts": [...]}},
    "output": {"mean": 7804.28, "median": 8689.5, "p95": 10240.0, "min": 989, "max": 10240,
               "histogram": {"bin_edges": [...], "counts": [...]},
               "truncated_fraction": 0.4075927734375}
  },
  "model": "/path/to/Qwen3-8B", "dataset": "...", "tokenizer": "...",
  "n": 8, "tp": 1, "seed": 42, "temperature": 1.0, "top_p": 1.0,
  "kind": "joint_group_trace", "n_shards": 16
}
```

A data line:

```json
{"group_id": 0, "prompt_len": 93, "output_lens": [7143, 7677, 7558, 6236, 7230, 7740, 7644, 6818]}
```

Three fields per group: the id, the one prompt length, and the *n* response lengths. The group
boundary survives, and so does the pairing between a prompt and its own responses.

## What the header tells you

Two numbers decide which parametric family could approximate this distribution, and both rule
out the obvious choice.

**The mean (7804) is below the median (8690).** That is left skew. A lognormal is right-skewed
by construction, so no choice of parameters makes it fit.

**40.8% of responses hit the cap.** The distribution is truncated, not merely skewed: a point
mass sits at exactly 10240, visible in the histogram as 3548 counts in the last bin against 223
in the one before it.

The closest parametric shape is `bimodal`, and {doc}`../../workload/index` shows how to set
it from these statistics. There is deliberately **no distribution-fitting command**:
estimating a family from a trace and then sampling from it discards the correlation and
grouping that made the trace worth collecting. If you have a trace, replay it.

## How to tell it worked

```bash
srl-workload verify-collection \
    --trace outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl --expected-groups 1024 \
    --shard-dir outputs/workload/qwen3_8b_dapo_tp1/shards --n-workers 16
```

prints `Trace complete: 1024/1024 groups collected.` and exits 0, or names the short workers
and exits 1. The collection script runs this itself as its last step. To read the header by
hand:

```bash
python -c "
import json
with open('outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl') as f:
    meta = json.loads(f.readline())['stats']
    rows = [json.loads(l) for l in f]
print(f\"{len(rows)} groups, group_size={meta['group_size']}\")
print(f\"prompt  mean {meta['prompt']['mean']:.1f}  median {meta['prompt']['median']}\")
print(f\"output  mean {meta['output']['mean']:.1f}  median {meta['output']['median']}\")
print(f\"truncated {meta['output']['truncated_fraction']:.1%}\")
assert all(len(r['output_lens']) == meta['group_size'] for r in rows)
"
```

Every group carries exactly `group_size` responses. The writer rejects a ragged trace. A
`truncated_fraction` near zero means the cap never bound and the tail is real. Near one means
the cap is doing all the work and the trace mostly records your `MAX_TOKENS` setting.

## The consequence for stages 3 and 4

This trace tells you whether the fitted training domains are wide enough. The median sequence
per response here is about 8800 tokens (prompt plus output) and the longest is 11133. The
Megatron latency cell fitted in {doc}`03-megatron-training` covers `tokens` up to 8192, so the
tail sits outside it, and the FSDP cells cover up to 16384.

A query outside a fitted range is refused, not guessed. {doc}`06-simulate` shows the refusal
and how to permit a bounded extrapolation deliberately.

## Next

{doc}`06-simulate` puts all of it together.
