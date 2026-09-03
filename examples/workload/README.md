# DAPO Qwen3-8B workload trace

This example collects a joint `(prompt, n responses)` workload trace from
Qwen3-8B on `dapo-math-17k`, then uses it as the simulator's
`workload=dapo_trace` source. Unlike independent length distributions, the
joint format preserves each prompt's measured token length, its eight response
lengths, and the group boundary required by DAPO.

## Collection

The default host inventory is [`hosts.txt`](hosts.txt): two hosts with eight
H20 GPUs each. Override it with `HOSTS_FILE=/path/to/hosts.txt`.

```bash
bash examples/workload/collect_joint_trace.sh
```

The script:

1. deduplicates DAPO prompts by `extra_info.index`;
2. samples 1024 prompts and assigns them to 16 workers with deterministic
   strided sharding;
3. starts one vLLM server per GPU and generates `n=8` responses per prompt;
4. writes `shards/worker-NN.jsonl`, merges them by `group_id`, and validates
   the merged body count.

The resulting trace is:

```text
outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl
```

Collection settings such as `DATASET`, `MODEL`, `SAMPLE_SIZE`, `GROUP_N`,
`MAX_TOKENS`, `CONCURRENCY`, and `OUT_DIR` can be overridden with environment
variables. Use `DRY_RUN=1` to inspect worker commands without launching
servers.

Completed shards are skipped on rerun. A shard is complete only when its body
contains at least the `n_groups_requested` recorded in its header; truncated or
short shards are recollected. The final `verify-collection` gate reports the
workers responsible for any merged-trace shortfall.

## Simulation

```bash
bash examples/simulate/dapo_qwen3_8b.sh
```

The default recipe uses 64 prompts × 8 responses per training step
(`batch_size=512` responses), 16 rollout instances, prompt cap 1024, response
cap 10240, and `max_staleness=2`.

## Recorded run

The 2026-08-25 collection produced 1024 groups / 8192 responses across 16 H20
GPUs in 49 minutes:

- prompt tokens: mean 149.7, median 138, p95 246, max 893;
- output tokens: mean 7804, median 8690, p95 10240, min 989;
- 40.8% of responses reached the 10240-token cap.

The corresponding 16-version simulation consumed all 8192 responses with no
drops, reached 24,904 output tokens/s, and observed maximum staleness 2.

## Current limitations

- `job.batch_size` counts responses, not prompts; whole-group batch consumption
  is not implemented yet.
- Prompt prefill is charged once per response rather than once per shared
  prompt, so rollout time is slightly pessimistic.
- The Qwen3-8B training cost remains fixed rather than fitted from a measured
  training artifact.
