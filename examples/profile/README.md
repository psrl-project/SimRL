# Qwen3-8B profiling

These workflows call pinned official benchmark entrypoints. DryRun-RL preserves their raw output and writes a
validated JSONL normalization as the only regression input.

Install pinned vLLM and Megatron-Bridge into the current conda environment first:

```bash
conda activate dryrun-rl
bash third_party/install.sh
pip install -e ".[dev]"
```

The example configs read the checkpoint from `MODEL_PATH`, so export it once:

```bash
export MODEL_PATH=/path/to/Qwen3-8B
```

## vLLM rollout

`qwen3_8b_tp1.yaml` defines separate output-one prefill and long-output decode workloads.
For each phase the adapter launches one official unprofiled timing sweep. Collection reads
scheduler `Iteration(...)` logs for step elapsed time and fills `sum_l2` / `ctxsum` from
equal-length formulas (`sum_l2` from as-equal-as-possible packing of `T` tokens across
`R` prefill requests; `ctxsum = R*(L+1+k)` on the `k`-th decode step). Client TTFT/ITL
remain in the untouched benchmark JSON for diagnostics but are not clustered into
synthetic GPU steps.

Servers always run with `--no-async-scheduling`. Async scheduling overlaps a step with
its neighbours, so an iteration's elapsed time stops measuring its own forward pass: a
65k-token prefill can report microseconds while its cost lands on a later step. Steps
that still report an impossibly short elapsed time are dropped at collection.

Collection also truncates a run at its first preemption. Once the KV cache fills, vLLM
evicts requests and later re-prefills each one's prompt *plus* the tokens it already
generated; from then on the running requests no longer share one age, so `ctxsum` cannot
describe any later step. Keep `max-concurrency * (input + output)` inside the KV cache to
avoid losing the tail of a decode point — the warning names the run and the step count it
dropped.

Preview commands without starting a server:

```bash
bash examples/profile/vllm/profile_qwen3_8b.sh generate
bash examples/profile/vllm/profile_qwen3_8b.sh run --dry-run
```

Run and collect both phases:

```bash
bash examples/profile/vllm/profile_qwen3_8b.sh all
```

When multiple GPUs are visible, TP=1 benchmark points are sharded across them and
run concurrently. Set `CUDA_VISIBLE_DEVICES` to choose the GPU pool, or force the
legacy serial layout with `MAX_PARALLEL=1` (equivalent to
`dryrun-profile vllm run --max-parallel 1`). `--max-parallel auto` is the
default; a workflow can set the same policy with `run.max_parallel: auto|N`.
Each parallel worker gets an isolated server port, experiment directory,
and scheduler log.

Shards benchmark in parallel but boot one at a time, because concurrent imports of
torch, vLLM, and the checkpoint from a shared filesystem are slow enough to trip
`server_ready_timeout` and kill the shard before it benchmarks anything. Raise
`run.max_concurrent_startups` when the checkpoint and environment sit on a fast local
disk. A shard that still fails no longer cancels the others; rerunning the same command
retries only the unfinished shards.

The normalized output is `outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl`. Official benchmark JSON remains
under `raw/vllm_timing/` and scheduler logs under `control/vllm/`. Parallel experiment directories and
control files carry an `sNNN` shard suffix; collection merges them by their
original serve/bench combination path.

For a smoke test, copy the YAML and reduce each phase grid, `num_runs`, and `bench.args.num-prompts`. Keep at least
one output-one prefill point and one decode point with more than one output token.

Prefill must enumerate prompt length and concurrency because each request is a single scheduler step. Decode already
emits one record per generation step, so the decode grid only needs a few starting contexts, one medium output
length to walk `ctxsum`, and a concurrency sweep up to `max-num-seqs`.

## vLLM rollout, without a server

`kernel/qwen3_8b_h20_tp1.yaml` profiles attention and GEMM kernels directly instead of
launching `vllm bench sweep serve`. There is no server, KV cache manager, or scheduler
involved, so the KV-cache-overflow, preemption, and startup-serialization constraints
above do not apply, and the sweep grid can be far denser. See
[`kernel/README.md`](kernel/README.md).

```bash
bash examples/profile/kernel/profile_qwen3_8b.sh all
```

## Megatron-Bridge training

The YAML only needs a HuggingFace checkpoint path. Architecture constraints and the Megatron-Bridge recipe are
inferred from `config.json`. `hardware.gpu_count` is the fixed world size; remaining ranks after TP and PP become
DP. Each candidate discards four warmup steps and records eight, with micro-batch size 1 and full activation
recomputation. The same topology is not repeated.

`num_microbatches` is the local 1F1B depth on one DP rank. Global batch is `MBS * DP * num_microbatches`. A value
smaller than PP leaves the pipeline underfilled, so peak activation memory is lower than a full `PP` in-flight
window.

DryRun owns the training entry point. `prepare` writes a `launch_spec.json` and `run` starts
`torchrun -m dryrun.profiler.adapters.megatron.launch`, which applies the candidate to the library recipe's
`ConfigContainer` in Python. The vendored `scripts/performance` CLI is not used: it parses only its own known
arguments and discards the rest, so recompute, profiling, and output-path settings never reached the model, and its
`--gpu` choices exclude H20 outright. Because the launcher owns the loop, step times and per-rank CUDA peaks are
measured through the official callback API and written to `dryrun_metrics.json`, which replaces TensorBoard scalar
scraping and memory-snapshot pickles.

Generate candidates from architecture, topology, batch, and pipeline constraints:

```bash
bash examples/profile/megatron/profile_qwen3_8b.sh generate
```

Prove the chain on one GPU with a two-layer model before launching a full-model candidate:

```bash
CUDA_VISIBLE_DEVICES=0 CONFIG=examples/profile/megatron/qwen3_8b_smoke.yaml \
  bash examples/profile/megatron/profile_qwen3_8b.sh run
```

The script defaults to candidate index zero so an accidental invocation cannot launch the entire sweep:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash examples/profile/megatron/profile_qwen3_8b.sh run
bash examples/profile/megatron/profile_qwen3_8b.sh collect
```

Set `CANDIDATE_INDEX=all` only for a provisioned sweep. Multiple nodes use the same launcher through
`torchrun --nnodes N --node-rank`; there is no separate Slurm path.

The sweep covers three sequence lengths at MBS=1 and local microbatch counts `{1, 4}` across four `(TP, PP)` cells.
Both axes are required: the latency design row is `[1, slots * tokens, slots]` with `slots = num_microbatches + PP - 1`,
so at a fixed microbatch count the slot column is constant and collinear with the intercept, capping the design at
rank two where the fit needs rank three. Fitted `feature_ranges` become hard out-of-domain gates at prediction time,
so extend both axes to whatever the simulator needs once the sweep is green.

Each candidate runs twelve steps and discards four. One warmup step was not enough: the first profiled step was the
slowest in 18 of 24 runs, where chance alone would give about 8, and dropping the second step as well cut the median
spread within a candidate from 16.4% to 0.5%. The retained steps are reduced by a median rather than a mean, since
interference inflates individual steps but nothing makes a step faster than the hardware allows.

OOM candidates are expected data for memory regression and are preserved instead of being discarded. The launcher
catches `torch.OutOfMemoryError` and writes a per-rank `oom_rank{N}.json` marker; classification reads those markers
rather than scanning the log, because a log that merely mentions the phrase would add an inequality constraint that
can only raise the fitted memory floor.
