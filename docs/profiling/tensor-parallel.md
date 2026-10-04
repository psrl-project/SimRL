# Tensor parallelism and communication

Everything else in this documentation prices a single-rank engine. This page is the full path for
an engine sharded across GPUs, where a step pays for collectives as well as compute.

Read this before running anything at `tp>1`. Two of the steps below are not optional, and
skipping either produces a number that looks plausible and is wrong.

## What a `tp>1` step actually costs

`ModelOps.comm_queries` emits **two all-reduces per layer**, one after the attention block's
output projection, one after the MLP's down projection. For Qwen3-8B that is 72 collectives in
every step. They are not a correction term: measured on H20 at `tp=2`, the collective is **13.2%
of a decode step** (up to 21.0% at small batch), and pricing it as zero moves median decode
error from 3.07% to 10.70%.

Three things change when the sweep runs at `tp>1`, and all three happen at *measurement* time,
not at predict time:

| Quantity | How `tp` enters |
|---|---|
| GEMM shapes | `num_attention_heads`, `num_query_groups`, `intermediate_size` and `vocab_size` are divided by `tp`. `hidden_size` is **not**, because it is the width of the collective-reassembled activation. |
| Attention head counts | `rank_head_counts` divides both head counts, and indivisibility is refused rather than rounded. |
| Collectives | Added by `comm_queries`, priced from the `all_reduce` table, sliced on `world_size`. |

This is why a kernel tree is **per degree**. A `tp=2` prediction read against the `tp=1` tree
prices every projection at twice its real width, and the resulting error looks like a model defect
rather than a wiring mistake. `simulate.sh` checks the `KERNEL_ROOT` path names the degree and
refuses otherwise.

## The collective measured is not NCCL

The `all_reduce` table is timed through `get_tp_group().all_reduce`, vLLM's own
`CudaCommunicator`, so the kernel timed is the kernel a server dispatches. It is deliberately
*not* `torch.distributed.all_reduce`, which was the original choice and was wrong.

Measured with `examples/profile/kernel/dispatch_probe.py` on H20 (sm_90), message size
`tokens * hidden * 2` bytes:

| Message size | `world_size=2` | `world_size=4` | `world_size=8` |
|---|---|---|---|
| 8-128 KiB | `custom_all_reduce` | `custom_all_reduce` | `custom_all_reduce` |
| 256 KiB to 4 MiB | `custom_all_reduce` | `custom_all_reduce` | `torch_symm_mem` |
| 8-32 MiB | `torch_symm_mem` | `torch_symm_mem` | `torch_symm_mem` |
| 64-128 MiB | `torch_symm_mem` / `pynccl` | `pynccl` | `torch_symm_mem` / `pynccl` |

`torch.distributed.all_reduce` never appears at any world size or message size. A Qwen3-8B decode
step lives in 8-512 KiB, which is entirely `custom_all_reduce`: on sm_90 that path's ceiling at two
ranks is 8 MiB, far above what a decode step needs. NCCL symmetric memory never fires despite
`VLLM_ALLREDUCE_USE_SYMM_MEM` defaulting on, because `is_symmetric_memory_enabled()` also requires
`VLLM_USE_NCCL_SYMM_MEM`, which defaults to 0.

Both tables are written from one sweep. `vllm` is what the cost model reads, and `nccl` is retained as
a control, selectable with `CommQuery(backend="nccl")`, so the cost of the original choice stays a
number rather than an argument:

| Sweep | `vllm` holdout MRE | `nccl` holdout MRE |
|---|---|---|
| `tp=2` | 6.01% | 4.96% |
| `tp=4` | 3.76% | 7.56% |
| `tp=8` | **9.30%** | **10.66%** |

At `tp=2` the two agree within ~6% across the decode range because both sit on the same ~18 us
launch floor. The old table was wrong in *identity* rather than badly wrong in *latency* at that degree.
The penalty grows with world size: `nccl` goes from beating `vllm` at `tp=2` to losing at
`tp=4` and `tp=8`, because its latencies increasingly describe a schedule the server does
not run.

::{warning}
The `tp=8` row was corrected on 2026-10-04. It previously read 7.56% / 24.80%, taken from
that tree's `holdout_report.json`. That report's mtime (`2026-09-02T16:46:49.5`) **predates
the parquet tables it scores** (`16:46:50.5`), whereas `tp=4` has the correct order: a
re-sweep overwrote the tables and `holdout` was never re-run. Scoring the tables that are
actually on disk gives 9.30% / 10.66%, reproducibly. Every other figure in both trees
reproduces bit-exactly, which rules out a code regression.

The qualitative claim survives: the dispatched backend still wins at `tp=8` and the control
still degrades with world size. Its magnitude was overstated, and "a utility floor of
0.000" was a property of the discarded data. See
`docs/superpowers/specs/2026-10-04-collective-ops-design.md` section 4 and
`tests/test_collective_baseline_regression.py`.
::

Since dispatch splits by message size at every world size, `kernel_source` reads `mixed` on these
tables. The per-size backend is recoverable from `examples/profile/kernel/DISPATCH.md` and from the
profiler's log.

## Running it end to end

### 1. Sweep the kernels at the degree you will simulate

```bash
export MODEL_PATH=/path/to/Qwen3-8B
TP=2 bash examples/profile/kernel/profile_qwen3_8b.sh all
```

That runs `run` → `fit-floor` → `holdout` in that order, which is mandatory:
`fit-floor` must precede `holdout`, and a floor-less `all_reduce` SOL errs by 220x at 4096 bytes.

To add collective coverage at another world size without re-sweeping compute (seconds, not hours):

```bash
for TP in 4 8; do
    TP=$TP FAMILY=collective bash examples/profile/kernel/profile_qwen3_8b.sh all
done
```

`world_size` is a **slice** column, so an unmeasured degree raises a coverage error rather than
interpolating. That is intended: a 2-rank and an 8-rank all-reduce run different schedules over
different topologies.

Check the dispatch if you changed GPU or vLLM version:

```bash
torchrun --standalone --nproc-per-node 2 examples/profile/kernel/dispatch_probe.py
```

### 2. Collect a real baseline at the same degree: serially

```bash
CONFIG=examples/profile/vllm/qwen3_8b_tp2_serial.yaml \
    bash examples/profile/vllm/profile_qwen3_8b.sh all
```

::{warning}
Use `max_parallel: 1` for any baseline whose figures will be quoted. The first TP=2 baseline ran
four servers across 8 GPUs and one CephFS mount, and one degraded mid-run: at per-request depth
8,960, `reqs=16` measured 35.76 ms against `reqs=32`'s 19.24 ms: half the batch, half the total
KV, nearly double the time. That inflated whole-grid decode error to 16.39% while the uncontended
buckets scored 3.18%. The serial re-measurement confirmed it: `reqs=16` moved from median ratio
0.513 to 0.993 while every other batch stayed put, and whole-grid decode fell to 9.51%.

The diagnostic is **monotonicity, not outliers**: a larger batch at deeper KV must be slower.
Scoring against each bucket's *minimum* latency moved the error only from 16.39% to 16.51%, which
proves sustained degradation rather than transient spikes, so filtering outliers does not fix it
and re-measurement is the only remedy.
::

The grid must match the `tp=1` grid if you want the two comparable bucket for bucket.
`qwen3_8b_tp2.yaml` and `qwen3_8b_tp2_serial.yaml` both hold a `phases:` block byte-identical to
`qwen3_8b_tp1.yaml` for exactly this reason.

### 3. Score it, and attribute the error

```bash
CFG=examples/profile/kernel/qwen3_8b_h20_tp2.yaml
BASE=outputs/profile/qwen3_8b/vllm_tp2_serial/profiles/vllm.jsonl

dryrun-profile kernels validate --config $CFG --baseline $BASE \
    --model-path $MODEL_PATH --surface pchip --min-samples 1 --gate 100

python examples/profile/kernel/decompose_tp.py --config $CFG --baseline $BASE \
    --model-path $MODEL_PATH
```

`--min-samples 1` is required or prefill and mixed are silently dropped (the default is 20).
`--gate 100` observes rather than fails.

`decompose_tp.py` splits each bucket's prediction into GEMM, attention and collective, then prices
the collective three ways: dispatched, NCCL control, and zero. That separates three questions
that a single MRE conflates: how large the collective term is, how much the backend choice
matters, and whether the residual looks like unmodelled overlap.

### 4. Simulate

```bash
ROLLOUT_MODEL=composed ROLLOUT_TP=2 \
KERNEL_ROOT=outputs/profile/qwen3_8b/kernel_tp2/kernel \
GPU=h20 MODEL_PATH=$MODEL_PATH \
    bash examples/simulate/simulate.sh
```

`ROLLOUT_TP` defaults to 1. Only the composed model reads it, since the fitted models were regressed at
one degree and carry no notion of a second, so setting it there is silently ignored. `pp`, `dp` and
`cp` are accepted in `rollout_cost.parallelism` but the composed model ignores them too.

`simulate.sh` refuses a `tp>1` run whose `KERNEL_ROOT` has no `collective/` directory, or whose
path does not name the degree, rather than letting a mid-simulation coverage error surface after
the workload is built.

## Accuracy at `tp=2`, and what it rests on

Measured 2026-09-03 against the serial baseline, H20 / Qwen3-8B, `pchip`:

| Selection | Buckets | MRE |
|---|---|---|
| prefill, whole grid | 29 | **3.60%** |
| decode, `requests >= 5` | 137 | **2.96%** |
| mixed | 7 | 3.64% |
| decode, whole grid | 204 | 9.51% |

The excluded-region figures beat the 4.84% decode / 3.72% prefill published at `tp=1`, and they are
the first end-to-end evidence that kernel composition survives tensor parallelism. At `tp=1`
`comm_queries` returns empty, so no earlier figure exercised a collective at all.

Prefill needs no exclusion. Its six `requests = 1` buckets originally scored 12.50% because they
were collected with `num-prompts: 1`, so each recorded iteration was a single cold request carrying
~78 ms of server-side work that no steady-state step pays. Fitting
`latency = fixed + per_token * tokens` found that offset directly (+68.2 ms in the serial run against
-1.3 ms in the parallel one, absent at `requests = 3` on the same server), and it implied a 256-token
prefill 3.8x *slower* on two GPUs than on one. Re-measured with `num-prompts: 8` at the same shape,
`requests = 1` prefill went from 46.98% to 3.98% MRE with the prediction unchanged, and 256 tokens now
measures 18.26 ms against 25.47 ms at `tp=1`, a sane 1.4x speedup.

Decode's remaining gap is a different thing and a **real model error**: `requests 1-4`
over-predicts (median ratio 1.31, and 1.1312 at `tp=1`, so it is not a TP effect). It predicts *too
much*, so it is not a missing overhead term. A composed step sums 253 isolated kernel latencies
(145 GEMM, 72 all-reduce, 36 attention), each carrying its own fitted per-call floor, and the excess at
`reqs=1` is 1.641 ms over 253 launches, or 6.5 µs each, which is essentially the 6.0 µs all-reduce
floor. The real server replays the step inside a CUDA graph where those costs largely vanish. Since
GEMM and comm are batch-independent (3.123 ms at one request, 3.180 ms at sixteen, because a decode GEMM is
memory-bound), that fixed excess is 31% of a 5.2 ms single-request step and under 3% of a 23 ms
batch-64 step. See {doc}`../reference/limitations` for the fix candidates.

Note the direction of the residual. Every phase's signed error is **negative** (-2.31% decode,
-2.63% prefill, -2.23% mixed), so the model slightly *under*-predicts. This was expected to go the
other way: the collective is timed in isolation, so an unmodelled compute/communication overlap
should have produced over-prediction by roughly the collective's 13.2% share. It did not. Overlap
is therefore not the dominant term at this degree, though it remains unquantified.

## What is still missing

- **Cross-node collectives are now measured** (2026-10-04), which closes what this list
  previously called "the first thing to fix when DP or PP communication is modelled". `nodes`
  is a slice column alongside `world_size`, `GpuSpec` carries a separate
  `inter_node_bandwidth_bytes_per_s`, and the sweep runs over real nodes via ssh rather than
  `127.0.0.1`. Measured at `(16, 2)` and `(24, 3)` on 24 H20s: a cross-node ring is 2.1-3.8x
  slower than intra-node in the floor-dominated regime and 1.34-1.84x at 64 MiB and above.

  Two caveats carry forward. The cross-node tables score 9.9-17.3% MRE against 3.2-10.2%
  intra-node, and monotonicity counts rule out contention, so that is a real property of the
  surface rather than a bad run: treat cross-node pricing as roughly +-15%. And cross-node
  rows carry **no `vllm` column**, because vLLM disables its custom all-reduce for a group
  spanning nodes (`custom_all_reduce.py:90`) and falls through to pynccl, which the control
  table already measures. See
  `docs/superpowers/specs/2026-10-04-collective-ops-design.md`.
- **`tp=4` and `tp=8` have collective tables but no end-to-end baseline.** Scoring either needs a
  vLLM sweep at that degree, and a compute sweep too, since GEMM and attention shapes are sharded.
- **No overlap model.** See above.
- **Small-batch decode over-prediction.** `requests 1-4` predicts high (median ratio 1.31 at
  `tp=2`, 1.1312 at `tp=1`) because the step sums 253 per-call floors that CUDA graph replay
  removes. The largest genuine model error remaining, and not a parallelism effect.
- **`num-prompts` must exceed 1 on any point whose figure will be quoted.** A `num-prompts: 1`
  benchmark records exactly one cold iteration, which carried a ~78 ms fixed cost here and made
  prefill look 3.5x worse than it is. `qwen3_8b_tp2_reqs1.yaml` is the corrected form.

See {doc}`../reference/limitations` for the full list, and
`examples/profile/kernel/TP2_VALIDATION.md` for the derivation behind these numbers.
