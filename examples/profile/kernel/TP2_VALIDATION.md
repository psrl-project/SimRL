# TP=2 end-to-end validation of the composed kernel model, Qwen3-8B on H20

> **Superseded in part.** The figures below come from the first, *parallel* baseline
> (`vllm_tp2/`). It was re-measured serially on 2026-09-03 (`vllm_tp2_serial/`), which confirmed
> the contamination diagnosis and produced the numbers now quoted everywhere. See
> **"The serial re-measurement"** at the end. This section is retained because the contrast
> between the two runs is the evidence that the diagnosis was right.

Baseline: `outputs/profile/qwen3_8b/vllm_tp2/profiles/vllm.jsonl`, 49,213 iterations from the
grid in `examples/profile/vllm/qwen3_8b_tp2.yaml`, which is byte-identical to the TP=1 grid so
the two are comparable bucket for bucket. Scored with `pchip`, `--min-samples 1`.

## Headline

| Phase | Buckets | TP=2 MRE | TP=1 MRE | median ratio (TP=2) |
|---|---|---|---|---|
| prefill | 28 | 6.95% | 3.72% | 0.9648 |
| decode | 203 | 16.39% | 4.84% | 0.9720 |
| mixed | 5 | 2.51% | 0.48% | 0.9766 |

Taken at face value this says TP=2 is 3.4x worse at decode. That reading is wrong, and the
decomposition below says why: 16.39% is a mean over three disjoint regimes, one of which is a
contaminated measurement rather than a model error.

## Decode error resolves into three groups

| Selection | Buckets | MRE | median ratio |
|---|---|---|---|
| all | 203 | 16.39% | 0.9720 |
| **reqs >= 32** | **103** | **3.18%** | **0.9707** |
| reqs 5-16 | 34 | 43.51% | **0.5084** |
| reqs 1-4 | 66 | 23.02% | 1.3009 |

**At reqs >= 32 the model is accurate: 3.18%, better than the 4.84% published at TP=1.** This is
the regime a real rollout runs in, and it is the regime where the collective term is largest in
absolute terms. So the kernel-composition approach does hold up under tensor parallelism.

**The reqs 5-16 group is a bad measurement, not a bad prediction.** A ratio of 0.5084 means the
measured step took almost exactly twice the predicted time, and the data contradicts itself:

| per-request depth | reqs=16 measured | reqs=32 measured |
|---|---|---|
| 1,089 | 6.12 ms | 7.21 ms |
| 8,960 | **35.76 ms** | **19.24 ms** |

At depth 1,089 the ordering is correct (16 requests cost less than 32). At depth 8,960 half the
batch takes nearly double the time, at twice the total KV. No cost model should reproduce that,
and TP=1's same batch range scores 0.9960. The four TP=2 servers shared 8 GPUs and one CephFS
mount; the reqs=16 server (GPUs 2,3) degraded partway through its run. Corroborating: reqs=4, 16
and 32 all show max/median latency ratios of 7.5x-20x within a bucket, against 1.5x-1.6x at
reqs=48/64.

Re-scoring against each bucket's *minimum* latency (a contended step is slower, never faster)
moves decode only from 16.39% to 16.51%, so this is a sustained degradation over that server's
run rather than transient spikes, which is why the fix is to re-measure, not to filter outliers.

**The reqs 1-4 group is the small-batch defect already documented at TP=1**, where the ratio is
1.1312. It worsens to 1.3009 at TP=2. `docs/reference/limitations.md` already records this with
three causes tested and ruled out (CUDA graph capture, validity of summing kernels, interpolation
at m=1), leaving kernel overlap in the real server as the remaining candidate. TP=2 adds 72
collectives to over-predict with, which is consistent with the effect growing rather than being
new.

## The overlap hypothesis is not supported

The plan predicted that if the collective is timed in isolation and the server overlaps it, the
model would *over*-predict at TP=2 by roughly the collective term. Measured, every phase's median
ratio is **below** 1.0 (0.9648 / 0.9720 / 0.9766): the model slightly *under*-predicts. At
reqs >= 32 the ratio is 0.9707 with the collective contributing 13.1% of the step, so an
unmodelled overlap saving of that size would have shown up as a ratio near 1.13, not 0.97.

This does not prove overlap is absent; it proves it is not the dominant term at this degree. The
collective is still timed in isolation and that remains an unquantified approximation.

## Collective term size, and the cost of the original mistake

From `examples/profile/kernel/decompose_tp.py` (median over buckets):

| Phase | comm share of step | median comm share max | MRE with `vllm` comm | with `nccl` comm | with no comm |
|---|---|---|---|---|---|
| decode | 13.1% | 21.0% | 3.98% | 2.90% | 10.41% |
| prefill | 4.2% | 7.1% | 3.52% | 4.19% | 8.24% |
| mixed | 3.9% | 5.0% | 2.34% | 2.08% | 6.13% |

(These are *median* absolute errors, which is why decode reads 3.98% here against the 16.39% mean
above: the median bucket is accurate and a minority are far off, exactly as the group table
shows.)

Two things follow:

1. **Modelling the collective at all matters.** Dropping it degrades decode from 3.98% to 10.41%
   and prefill from 3.52% to 8.24%. A comm-free TP>1 model would be visibly wrong.
2. **At TP=2 the choice between the dispatched kernel and the NCCL control barely matters for
   accuracy**: 3.98% vs 2.90% at decode, with the NCCL table 1.12x slower on the comm term.
   Both are pinned to the same ~18 us launch floor in the 8-512 KiB decode range. The original
   table was wrong in *identity*, not badly wrong in *latency*, at this degree.

That last point deserves emphasis because it tempers the original motivation. The
custom-vs-NCCL distinction is real (measured, `torch.distributed.all_reduce` is never what a
server dispatches) but at TP=2 on H20 it is worth ~1pp of end-to-end MRE, not the dominant error.
Where it does bite is the per-kernel holdout as world size grows: the NCCL control reaches 10.66%
MRE at world_size 8 against 9.30% for the dispatched table, having *beaten* it at world_size 2.
(These world_size 8 figures were corrected on 2026-10-04; the previously published
24.80% / 7.56% came from a holdout report that predates the tables it scores. See the
warning in `docs/profiling/tensor-parallel.md`.)

## Status

- TP=2 intra-node is now **validated end to end** at reqs >= 32: 3.18% decode, 6.95% prefill.
- The reqs 5-16 decode buckets are **not usable** and should be re-measured with
  `max_parallel=1` (serial servers, no cross-server contention) before quoting a whole-grid
  decode number for TP=2.
- The headline 16.39% should not be quoted as the TP=2 decode accuracy without this
  decomposition attached, and equally the 3.18% should not be quoted without saying it excludes
  two batch ranges.


---

## The serial re-measurement (2026-09-03)

Re-ran the same grid with `max_parallel: 1` (`examples/profile/vllm/qwen3_8b_tp2_serial.yaml`),
one server on GPUs 0-1, ~40 minutes. 49,211 iterations against the parallel run's 49,213.

**The monotonicity violation is gone.** At matched per-request depth:

| per-request depth | reqs=16 parallel | reqs=16 serial | reqs=32 serial |
|---|---|---|---|
| 1,089 | 6.12 ms | 6.17 ms | 7.19 ms |
| 9,152 | **35.76 ms** | **12.43 ms** | 19.50 ms |

reqs=16 is now correctly *faster* than reqs=32 at the same depth, and its median ratio moves from
0.513 to 0.993. That is the diagnosis confirmed: the parallel run's reqs=16 server was degraded,
and no cost model should have reproduced it.

Decode median ratio by batch, both runs:

| reqs | 1 | 4 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|
| parallel | 1.318 | 1.066 | **0.513** | 0.966 | 0.976 | 0.969 |
| serial | 1.314 | 1.073 | **0.993** | 0.968 | 0.974 | 0.975 |

Every other batch is unchanged, which is what makes the reqs=16 fix attributable to contention
rather than to anything about the model.

### Accuracy on the serial baseline

| Selection | Buckets | MRE |
|---|---|---|
| decode, whole grid | 204 | 9.51% |
| **decode, `requests >= 5`** | **137** | **2.96%** |
| prefill, whole grid | 29 | 12.50% |
| **prefill, `requests > 1`** | **23** | **3.50%** |
| mixed | 7 | 3.64% |

Decode improved from 16.39% to 9.51% on the whole grid, and the usable-region figure improved from
3.18% over 103 buckets to **2.96% over 137**: more buckets *and* lower error, because the
reqs=16 group rejoined the usable set instead of being excluded.

### The prefill regression is a 68 ms offset in six buckets, not a model gap

Prefill went from 6.95% to 12.50% on the serial run. It is entirely `reqs=1`, whose median ratio
moved 0.815 to 0.515. Fitting `latency = fixed + per_token * tokens` over the six `reqs=1` prefill
shapes isolates it:

| run | fitted fixed cost | fitted per-token |
|---|---|---|
| tp1 | -21.5 ms | 119.5 us |
| tp2 parallel | -1.3 ms | 73.2 us |
| **tp2 serial** | **+68.2 ms** | 63.1 us |

The per-token slope is sane everywhere and matches between the two tp=2 runs (73.2 / 63.1 us). Only
the serial run carries a fixed offset, and only at `reqs=1`. On that same serial server:

| group | fitted fixed cost | fitted per-token |
|---|---|---|
| `reqs=1` | **+68.2 ms** | 63.1 us |
| `reqs=3` | -6.3 ms | 61.6 us |
| `reqs=15` | -15.7 ms | 57.3 us |

Two direct comparisons make it concrete. Same server, same total token count, different request
count: `reqs=1 / 2048 tokens` measures 197.26 ms while `reqs=8 / 2048 tokens` measures 118.54 ms.
And in execution order, a `reqs=3 / 768-token` step (56.54 ms) sits between two `reqs=1 / 256-token`
steps (95.99 and 90.01 ms): fewer tokens taking more time, seconds apart on one server.

The decisive sanity check is against tp=1. A 256-token prefill measures 25.47 ms at tp=1 on one GPU
and **96.72 ms** at tp=2 serial on two: 3.8x *slower* on twice the hardware, which no real compute
does. At 4096 tokens and above both tp=2 runs agree closely (0.718 and 0.713 of tp=1) and show the
expected 1.4-1.6x speedup. So the offset is confined to the small-token, single-request corner.

**This is a measurement artifact in six buckets, not a missing model term.** A genuine fixed
per-step cost would appear at every request count and in both tp=2 runs; it appears in neither.
The likely mechanism is that `reqs=1` points set `num-prompts: 1`, so the recorded iteration is a
single isolated request whose measured span picks up server-side work that a steady stream amortises
:  and the serial run, having booted a fresh server for the whole phase, is the configuration where
that shows up unmasked. It was not diagnosed further because the fix is the same either way: those
six buckets need re-measuring with more prompts per point before prefill `reqs=1` can be scored.

An earlier revision of this document called this a "missing fixed per-step cost" that the model
should learn. That was wrong, and the evidence above is what refutes it: a real fixed cost cannot be
absent at `reqs=3` on the same server, absent in the parallel run, and absent at tp=1.

### What to quote

**decode 2.96% at `requests >= 5`, prefill 3.50% at `requests > 1`, mixed 3.64%**: with the
exclusion stated. The two exclusions are *different* problems and should not be described as one:

- prefill `reqs=1` (6 buckets) is a measurement artifact, per above. Re-measure.
- decode `reqs 1-4` (66 buckets) is a real model error, over-predicting at median ratio 1.31
  against 1.1312 at tp=1. It is the pre-existing small-batch limitation, already documented with
  three causes tested and ruled out, and tp=2 amplifies it. Note the direction: the model predicts
  *more* than measured, so it is not a missing cost term at all: if anything the model is
  double-counting something the server overlaps.

Collective decomposition on the serial baseline, median over buckets:

| Phase | comm share | MRE with `vllm` comm | with `nccl` comm | with no comm |
|---|---|---|---|---|
| decode | 13.2% | 3.07% | 2.45% | 10.70% |
| prefill | 4.3% | 2.63% | 2.48% | 6.49% |
| mixed | 3.9% | 2.23% | 2.08% | 6.10% |

Unchanged conclusions: modelling the collective matters (decode 3.07% against 10.70% without it),
the dispatched-vs-NCCL choice is worth about 0.6pp at `tp=2`, and every phase still
under-predicts (signed -2.31% / -2.63% / -2.23%), so unmodelled overlap is still not the dominant
term.


---

## Why outlier rejection is not the fix (measured)

The natural reaction to both problems is to discard the bad points as outliers, in `validate` or in
the profiler. Measured, that does almost nothing, because the errors are not carried by a minority
of iterations.

`load_baseline` already takes the **median** of each bucket, which is inherently outlier-resistant.
Re-scoring with two more aggressive aggregators:

| Phase | median (current) | median of best half | minimum |
|---|---|---|---|
| decode | 9.51% | 9.18% | 10.58% |
| prefill | 12.50% | 12.34% | 12.23% |

Trimming the slow half of every bucket buys 0.3pp at decode and 0.2pp at prefill. For comparison,
excluding the two known-bad *batch groups* takes decode to 2.96% and prefill to 3.50%. The error
lives in which shapes were measured, not in stray iterations within a shape.

The reason is visible in the raw data. Prefill `reqs=1 / 256 tokens` holds four iterations at
90.0, 96.0, 97.5 and 104.0 ms: there is no minority to drop, the whole bucket sits at that level.
Across all 205 buckets with 4+ iterations, trimming the slow half moves the bucket median by 1.1%
at the median bucket and 5.7% at the very worst.

The same conclusion held for the contamination this document opens with: scoring the parallel run
against each bucket's *minimum* latency moved decode only from 16.39% to 16.51%. A degraded server
and an unmeasured shape both produce a consistently wrong bucket, and neither is reachable by
robust statistics.

**So the rule is: use the median (already the default), and spend the effort on coverage instead.**
Outlier rejection is worth adding only where a bucket genuinely holds a stray minority, and on
this grid, at 4 iterations per bucket, the median already handles that. Adding a trimming step would
buy tenths of a percentage point while making every published figure depend on a new tunable
threshold.


---

## vLLM measured vs kernel-composed, side by side

Every MRE on this page is already a comparison between two things, and it is worth naming them
explicitly because they are not symmetric:

- **vLLM end-to-end**: a real `vllm serve` process, `elapsed_ms` read per scheduler iteration from
  its own log. This is the *measurement*, and it can itself be wrong (see the two contamination
  findings above).
- **kernel-composed**: 5 GEMM lookups + 1-2 attention lookups + 1 collective lookup, summed. This
  is the *prediction*.

`srl-profile kernels validate` buckets the former and scores the latter against it.

### Prefill (tp=2 serial)

| reqs | tokens | vLLM ms | composed ms | error |
|---:|---:|---:|---:|---:|
| 1 | 256 | 96.72 | 17.72 | -81.7% |
| 1 | 2048 | 197.26 | 116.09 | -41.1% |
| 1 | 8192 | 588.85 | 511.18 | -13.2% |
| **8** | **2048** | **118.54** | **112.58** | **-5.0%** |
| 3 | 3072 | 187.05 | 171.02 | -8.6% |
| 3 | 12288 | 712.87 | 706.49 | -0.9% |
| 15 | 15360 | 850.01 | 827.65 | -2.6% |
| 32 | 65536 | 3607.69 | 3589.26 | -0.5% |

The two bolded-adjacent rows are the argument: `reqs=1 / 2048 tokens` errs -41.1% while
`reqs=8 / 2048 tokens` errs -5.0%. Identical total work on the same server, and the only difference
is how many requests it was spread over. Above `reqs=3` the error is -0.4% to -8.6% and shrinks as
the step grows, which is the signature of a *fixed* discrepancy diluted by real work.

### Decode (tp=2 serial)

| reqs | per-request depth | vLLM ms | composed ms | error |
|---:|---:|---:|---:|---:|
| 1 | 1088 | 4.995 | 6.894 | **+38.0%** |
| 1 | 9152 | 5.450 | 6.952 | +27.6% |
| 4 | 1089 | 5.350 | 6.832 | +27.7% |
| 4 | 9152 | 7.745 | 7.157 | -7.6% |
| 16 | 5120 | 9.400 | 9.391 | **-0.1%** |
| 16 | 9152 | 12.430 | 12.354 | -0.6% |
| 32 | 9152 | 19.500 | 19.007 | -2.5% |
| 64 | 9152 | 35.460 | 34.767 | -2.0% |
| 64 | 1090 | 10.850 | 10.602 | -2.3% |

From `reqs=16` upward the agreement is -0.1% to -3.3% and flat in KV depth, which is the regime a
real rollout runs in.

### The two `reqs=1` anomalies are opposite in sign, so they are not one cause

An earlier revision of this document claimed a single missing fixed cost explained both. The
absolute gap (`measured - predicted`) refutes that:

| run | phase, reqs=1 | gap, median | gap, range |
|---|---|---|---|
| tp=2 serial | prefill | **+78.0 ms** | +60.1 .. +81.2 |
| tp=2 serial | decode | **-1.7 ms** | -1.90 .. -1.50 |
| tp=1 | prefill | -7.5 ms | -13.5 .. -1.0 |
| tp=1 | decode | -1.2 ms | -1.48 .. -0.80 |

Both tp=2 gaps are near-constant, and both are constant in the *absolute* sense that identifies a
fixed term rather than a scaling error. But they have **opposite signs and differ 46x in magnitude**,
so they cannot share a mechanism:

- **prefill +78 ms** exists only in the tp=2 serial run. tp=1 shows -7.5 ms and the tp=2 parallel run
  shows -1.3 ms fitted. It is a measurement artifact, and it is what the `num-prompts: 8`
  re-measurement addresses.
- **decode -1.7 ms** exists at tp=1 too (-1.2 ms). It is a real, TP-independent property of the
  model: at `reqs=1` the composed sum slightly *exceeds* what vLLM measures. Note what vLLM measures
  there: 8.4x more KV changes the step by only 1.09x (tp=1: 1.11x), against 3.27x at `reqs=64`. A
  single-request decode step is dominated by per-step overhead and the GPU work is nearly invisible,
  so summing kernels over-counts. Re-measuring will not change this; it is the known small-batch
  limitation, and the fix would be a model term, not a better baseline.

The practical consequence: `reqs=1` prefill is worth re-measuring and `reqs=1` decode is not.


---

## The re-measurement: prefill `reqs=1` was the baseline's fault, and it is fixed

Ran the six `max-concurrency: 1` prefill points again with `num-prompts: 8` instead of `1`
(`examples/profile/vllm/qwen3_8b_tp2_reqs1.yaml`, ~7 minutes, 2 GPUs). The measured *shape* is
unchanged: concurrency is still 1, so each iteration is still a single-request batch, but the
server runs eight requests in sequence, so the recorded iterations are steady-state rather than a
single cold one.

| tokens | old (num-prompts 1) | new (num-prompts 8) | composed prediction | old error | new error |
|---:|---:|---:|---:|---:|---:|
| 256 | 96.72 ms | **18.26 ms** | 17.72 ms | -81.7% | **-3.0%** |
| 512 | 91.62 ms | 33.39 ms | 31.54 ms | -65.6% | -5.5% |
| 1024 | 132.77 ms | 62.15 ms | 58.55 ms | -55.9% | -5.8% |
| 2048 | 197.26 ms | 121.72 ms | 116.09 ms | -41.1% | -4.6% |
| 4096 | 319.26 ms | 248.03 ms | 241.36 ms | -24.4% | -2.7% |
| 8192 | 588.85 ms | 522.81 ms | 511.18 ms | -13.2% | -2.2% |

**`reqs=1` prefill MRE: 46.98% -> 3.98%.** Nothing about the model changed. The 256-token point went
from 96.72 ms to 18.26 ms against an unchanged prediction of 17.72 ms, and the ~78 ms constant offset
is gone at every token count. The +78 ms was `num-prompts: 1` server-side work that no steady-state
iteration pays, and it was in the *measurement*, exactly as the fit predicted.

The sanity check that failed before now passes: a 256-token prefill at tp=2 measures 18.26 ms
against 25.47 ms at tp=1 on one GPU: a 1.4x speedup on twice the hardware, where the old number
implied 3.8x *slower*.

### Final tp=2 accuracy, splicing the re-measured buckets in

| Phase | Buckets | before | **after** |
|---|---:|---:|---:|
| prefill | 29 | 12.50% | **3.60%** |
| decode | 204 | 9.51% | 9.51% |
| mixed | 7 | 3.64% | 3.64% |

Prefill now needs **no exclusion at all**: 3.60% over the whole grid, comparable to the 3.72%
published at tp=1. Decode is untouched, as expected: its `reqs=1` gap is -1.7 ms of *over*-prediction
and is a model property present at tp=1 too, not a baseline artifact. Re-measuring decode would not
have moved it, which is why it was not re-measured.

So the tp=2 picture reduces to one remaining caveat rather than two:

- **prefill: 3.60%, whole grid, no caveat.**
- **decode: 2.96% at `requests >= 5`**, with `requests 1-4` over-predicting because a single-request
  decode step is dominated by per-step overhead that summing kernels cannot see (8.4x more KV moves
  the measured step only 1.09x). That is the pre-existing small-batch limitation and needs a model
  term, not a better measurement.
- **mixed: 3.64%**, 7 buckets, indicative.


---

## What the small-batch decode error actually is (and why "add an overhead term" is wrong)

An earlier note in this file, and a summary message, proposed fixing the remaining decode error by
adding a per-step fixed-overhead term. **That is backwards.** The model *over*-predicts at small
batch, so adding anything makes it worse:

| reqs | depth | measured | predicted | predicted - measured |
|---:|---:|---:|---:|---:|
| 1 | 5120 | 5.240 ms | 6.881 ms | **+1.641 ms** |
| 4 | 5120 | 6.340 ms | 7.033 ms | +0.693 ms |
| 16 | 5120 | 9.400 ms | 9.391 ms | -0.009 ms |
| 64 | 5120 | 23.290 ms | 22.643 ms | -0.647 ms |

### Where the excess sits

Decomposing the predicted step at depth 5120:

| reqs | GEMM | attention | comm | total | measured |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.123 | 2.490 | 1.268 | 6.881 | 5.240 |
| 16 | 3.180 | 4.685 | 1.527 | 9.391 | 9.400 |
| 64 | 4.745 | 16.716 | 1.183 | 22.643 | 23.290 |

GEMM is **3.123 ms at one request and 3.180 ms at sixteen**: essentially flat, and comm likewise.
That is not a table error: a decode GEMM is memory-bound, reading the whole weight matrix to compute
one token's worth of output, so `m=1` and `m=16` genuinely cost the same. The per-shape numbers
confirm it (`m=1, n=12288, k=4096` = 38.71 us against 41.14 us at `m=16`).

So the model's step cost has a floor of about 4.4 ms of GEMM+comm that does not shrink with batch,
and at `reqs=1` the *entire real step* is 5.240 ms. Attention is the only term that scales, which is
why the error vanishes by `reqs=16` and inverts slightly beyond it.

### The mechanism: 253 per-call floors the server does not pay

A composed decode step sums **253 kernel latencies**: 145 GEMM (4 per layer plus `lm_head`),
72 all-reduce (2 per layer), 36 attention. Each was timed as an isolated call, so each carries its
own launch and dispatch cost. The fitted floors say how much: `all_reduce` 6.0 us,
`generation_attention` 22.0 us.

The excess at `reqs=1` is 1.641 ms over 253 launches = **6.5 us per launch**, which is the
all-reduce floor almost exactly. The real server replays this whole step inside a **CUDA graph**,
where those per-call costs are largely eliminated, which is the same trap the elementwise table
already documents from the other side (timing the `CustomOp` module instead of the raw op cost
13.2x, because the server replays it in a graph).

At `reqs=1` the fixed part is ~84% of the predicted step (4.4 of 5.2 ms measured), so a per-launch
error of 6.5 us across 253 launches is 31% of the step. At `reqs=64` the same absolute excess is
under 3% of a 23 ms step and is swamped by real attention work. That is exactly the observed shape.

### What a fix would have to look like

Not an additive overhead term. The candidates, in order of how well they match the evidence:

1. **A per-call floor discount for graph replay.** The model already knows each op's fitted floor.
   Charging the floor once per step rather than once per kernel: or scaling it by a measured
   graph-replay factor: attacks the term the evidence points at. It needs one new measurement:
   the same kernel sequence timed inside a captured graph versus as isolated calls.
2. **Model the overlap explicitly.** The collective is timed in isolation and the server can overlap
   it with compute. This is already a documented unquantified approximation, and at `reqs=1` comm is
   1.268 ms of a 5.240 ms step, so full overlap alone would account for most of the excess.
3. **Do nothing and keep the domain restriction.** `requests >= 5` scores 2.96%, and a real rollout
   runs batched. This is the honest status quo and costs nothing.

**This is the decision taken on 2026-09-03: option 3.** Neither fix is scheduled. Option 1 would add
a fitted graph-replay coefficient, and a term that improves in-sample error without representing a
mechanism is exactly what this project rejected the quadratic token term for. Option 2 is the more
principled change and stays on the deferred list with topology-aware communication.

What would settle between (1) and (2) is one experiment: capture a CUDA graph over one decoder
layer's kernel sequence at `reqs=1`, time the graph replay, and compare against the sum of the same
kernels timed individually. If the graph is faster by roughly `n_launches * floor`, it is (1). If the
gap is closer to the collective's contribution, it is (2). Neither is implemented.

**Until then the correct statement is that the composed model is validated for batched decode
(`requests >= 5`, 2.96%) and over-predicts single-request decode by ~30%, with the mechanism
identified as per-call floors that CUDA graph replay removes.**
