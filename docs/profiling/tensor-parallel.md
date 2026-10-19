# Tensor parallelism and communication

Everything else in this documentation prices a single-rank engine. This page is the path for an
engine sharded across GPUs, where a step pays for collectives as well as compute. It is the
`collective` family of the kernel pipeline plus the measurement-time sharding rules that make a
per-degree kernel tree necessary.

## What changes at `tp > 1`

| quantity | how `tp` enters |
|---|---|
| GEMM shapes | `num_attention_heads`, `num_query_groups`, `intermediate_size` and the vocabulary projection are divided by `tp`. `hidden_size` is not, because it is the width of the collective-reassembled activation. |
| attention head counts | both head counts are divided by `tp`, and an indivisible degree raises rather than rounding. |
| collectives | `ModelOps.comm_queries` adds two all-reduces per layer (after the attention output projection and after the MLP down projection), sliced on `world_size` and priced from the `all_reduce` table. |

All three happen at **measurement** time, not at predict time: the perf table stores the sharded
shapes, so a `tp=2` prediction read against a `tp=1` tree prices every projection at twice its
real width, and the resulting error looks like a model defect rather than a wiring mistake. That
is why a compute tree is **per degree** (`kernel_tp1`, `kernel_tp2`, ...).

The collective count is emitted as `(query, multiplier)` data rather than written inline, so a
layer-count error cannot look like a plausible measurement.

## The collective family

Five ops, all in the `collective` family, are measured by one sweep:

| op | per-rank traffic | used by |
|---|---|---|
| `all_reduce` | full message | vLLM rollout, after each attention and MLP block |
| `reduce_scatter` | `tokens * hidden / tp` | Megatron sequence-parallel dgrad and the DP gradient reduction |
| `all_gather` | `tokens * hidden / tp` (gather to full) | Megatron sequence-parallel wgrad input, and the DP gradient reduction |
| `broadcast` | one message | pipelined broadcast |
| `p2p` | one directed transfer | a pipeline stage boundary send/recv |

They exist as separate ops because one table cannot price another op: a reduce-scatter over
`tokens*hidden/tp` bytes priced from an all-reduce table would be wrong in kernel identity,
message size and count at once. Each op differs only in how much traffic one rank moves, which
it declares as a traffic factor. All five share one roofline binding and one floor policy.

The slice and axis columns are:

- slice (matched exactly): `dtype`, `world_size`, `nodes`.
- axis (interpolated): `message_bytes`.

`world_size` and `nodes` are slice columns on purpose. A 2-rank all-reduce and an 8-rank one run
different ring schedules, and 8 ranks on one node and 8 across two traverse different links in
different counts. An unmeasured topology therefore raises a coverage error rather than
interpolating from a different one.

`message_bytes` is swept directly rather than derived from token counts, because the ops do not
carry the same size at the same token count: an SP-sharded reduce-scatter moves
`tokens*hidden/tp` where an all-reduce moves `tokens*hidden`, and a Megatron gradient bucket is
sized in parameters, not tokens. `dtypes` includes `float32` because Megatron reduces gradients
in fp32, so a bf16-only table cannot price the DP reduction.

A fitted floor is essential here rather than a refinement: every collective pays a launch and
handshake cost that dominates entirely at the small messages a decode step and an SP-sharded
layer live in.

## The collective measured is vLLM's dispatcher, not NCCL

`CudaCommunicator.all_reduce` does not go straight to NCCL. It tries NCCL symmetric memory, then
vLLM's custom one-shot/two-shot P2P kernel, and reaches `torch.distributed.all_reduce` only as a
last resort. For a decode step's message sizes the custom kernel covers the range, so a `tp=2`
server issues **zero** NCCL all-reduces.

The sweep therefore writes two tables from one measurement per topology:

- `framework: vllm` times `get_tp_group()`'s own entry points. This is what
  `CommQuery(backend="vllm")`, the default, reads.
- `framework: nccl` times `torch.distributed` as a **control**, selectable with
  `CommQuery(backend="nccl")`. It keeps the cost of an earlier choice honest: the model once read
  the control table and priced a kernel the server never invoked.

`kernel_source` records which backend actually fired on each row, read off the live communicator
rather than re-derived from vLLM's size thresholds. Because the dispatch splits by message size at
every world size, that column normally reads `mixed`.

Ops the dispatcher does not expose are not measured under the `vllm` framework. `p2p` is the
case: vLLM's pipeline send/recv uses a separate group with its own buffers, so a `vllm` row would
misattribute the kernel. Crossing a node boundary disables both the custom kernel and symmetric
memory, so a cross-node topology produces an `nccl`-only table and the sweep logs why.

To see which backend a given GPU and vLLM version dispatches:

```bash
torchrun --standalone --nproc-per-node 2 examples/profile/kernel/dispatch_probe.py
```

## Running a sweep at a degree

### 1. Compute kernels at the degree you will simulate

```bash
export MODEL_PATH=/path/to/Qwen3-8B
TP=2 bash examples/profile/kernel/profile_qwen3_8b.sh all
```

This writes a `kernel_tp2` tree whose GEMM and attention rows are already sharded.

### 2. Collectives

The `collective:` block declares its own `topologies` as `(world_size, nodes)` pairs rather than
reading `parallelism.tensor_parallel`, so **one run writes world sizes 2, 4 and 8 into one
table**:

```bash
FAMILY=collective bash examples/profile/kernel/profile_qwen3_8b.sh run
```

Every collective is timed inside one process group per topology, because group setup and the NCCL
handshake dominate a small-message measurement. The `tp4` and `tp8` configs therefore exist only
for compute sweeps at those degrees.

Cross-node coverage needs hostnames and passwordless ssh. `collective_multinode.yaml` declares the
hosts, and the driver refuses a cross-node topology when `NCCL_IB_DISABLE` would fall back to TCP,
forwards every inherited `NCCL_*` variable to the remote ranks so both ends of a ring run the same
fabric configuration, and logs the ones that change what is measured:

```bash
srl-profile kernels run      --config examples/profile/kernel/collective_multinode.yaml --family collective
srl-profile kernels fit-floor --config examples/profile/kernel/collective_multinode.yaml
srl-profile kernels holdout   --config examples/profile/kernel/collective_multinode.yaml
```

`torch.multiprocessing.spawn` cannot produce a cross-node group, since it binds `MASTER_ADDR` to
loopback; that is why the child is launched under `torchrun` and, across hosts, under ssh.

### 3. Collect a real baseline at the same degree

```bash
CONFIG=examples/profile/vllm/qwen3_8b_tp2_serial.yaml \
    bash examples/profile/vllm/profile_qwen3_8b.sh all
```

Use `max_parallel: 1` for any baseline whose figures will be quoted: concurrent servers sharing a
machine and a filesystem can degrade one another mid-run with no error, and the diagnostic is
non-monotonicity rather than an outlier. The grid should be the same as the `tp=1` grid if the two
degrees will be compared bucket for bucket. {doc}`vllm` covers both points.

### 4. Score it, and attribute the error

```bash
CFG=examples/profile/kernel/qwen3_8b_h20_tp2.yaml
BASE=outputs/profile/qwen3_8b/vllm_tp2_serial/profiles/vllm.jsonl

srl-profile kernels validate --config $CFG --baseline $BASE \
    --model-path $MODEL_PATH --surface pchip --min-samples 1 --gate 100

python examples/profile/kernel/decompose_tp.py --config $CFG --baseline $BASE \
    --model-path $MODEL_PATH
```

`validate` takes the degree from the config's `tensor_parallel`, so the config must be the one
swept at that degree. `--min-samples 1` is required to score all three phases; the default of 20
silently drops prefill and mixed from a typical decode-dominated baseline, and the tool warns
about each dropped phase. `--gate 100` observes rather than fails.

`decompose_tp.py` splits each bucket's prediction into GEMM, attention and collective, and prices
the collective three ways: dispatched (`vllm`), the NCCL control, and zero. That separates three
questions one MRE conflates: how large the collective term is, how much the backend choice
matters, and whether the residual looks like unmodelled overlap. The collective is timed in
isolation, so an unmodelled compute/communication overlap would show up as over-prediction by
roughly the collective's share, and fitting the comm term downward would be curve-fitting over a
known mechanism.

Accuracy figures for each degree are in {doc}`../reference/validation`. `kernels validate` writes
`{table_root}/validate_report.json` by default.

### 5. Simulate

```bash
ROLLOUT_MODEL=composed ROLLOUT_TP=2 \
KERNEL_ROOT=outputs/profile/qwen3_8b/kernel_tp2/kernel \
GPU=h20 MODEL_PATH=$MODEL_PATH \
    bash examples/simulate/simulate.sh
```

`ROLLOUT_TP` defaults to 1 and only the composed model reads it; the fitted models were
regressed at one degree and carry no notion of a second, so setting it there is silently
ignored. `pp`, `dp` and `cp` are accepted in `rollout_cost.parallelism`, but the composed model
uses `tp` only. `ComposedRollout` raises if the degree passed to it disagrees with the degree its
`ModelOps` was built for.

The example script refuses a `tp > 1` run whose `KERNEL_ROOT` has no `collective/` directory, or
whose path does not name the degree, rather than letting a coverage error surface after the
workload is built.

## Training communication

`reduce_scatter`, `all_gather` and `p2p` are measured for the training models, not for rollout.
`TrainingOps` prices sequence-parallel collectives per layer as `(query, multiplier)` data, with
different counts for the forward, the recompute replay and the backward, and `ComposedTraining`
charges pipeline boundary transfers per microbatch and the 1F1B bubble as idle slots. See
{doc}`../cost-models/training-megatron`.

## What is not modelled

- **Compute/communication overlap.** A collective is timed in isolation. The direction of the
  residual in a validation run is the evidence to read: uniform over-prediction by roughly the
  collective's share would mean overlap is missing.
- **Per-call floors at small batch.** A composed step sums one isolated latency per kernel, each
  with its own floor, while a server replays the step inside a CUDA graph where much of that cost
  disappears. Small-batch decode is where this is largest. See
  {doc}`../reference/limitations`.
- **Degrees without an end-to-end baseline.** A collective table and a compute tree are not
  themselves evidence that composition holds at that degree. Scoring it needs a vLLM baseline at
  the same degree.
- **Cross-node collectives for rollout.** vLLM keeps its tensor-parallel group inside one node,
  which is why `CommQuery` passes `nodes=1` explicitly rather than defaulting it. Cross-node
  tables exist for training communication.
