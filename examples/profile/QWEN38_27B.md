# Qwen3.8-27B profiling and parallelism plan

Qwen3.8-27B on 24 H20s, profiled with the same three workflows as Qwen3-8B: `vllm/`,
`megatron/` and `kernel/`. The checkpoint directory is `Qwen3.8-27B`, and its `config.json`
declares the `qwen3_5` architecture (`Qwen3_5ForConditionalGeneration`), which is why the
family and recipe names carry the `qwen35` prefix. This note records what the architecture
forces, what was measured, and the parallelism and GPU split that follow. The commands live in
[`vllm/profile_qwen35_27b.sh`](vllm/profile_qwen35_27b.sh),
[`megatron/qwen35_27b.yaml`](megatron/qwen35_27b.yaml) and
[`kernel/profile_qwen35_27b.sh`](kernel/profile_qwen35_27b.sh), and the plan is reproduced by
`examples/analysis/qwen35_27b_parallel_plan.py`.

## Architecture, and the three caps it imposes

| | |
|---|---|
| Layers | 64, of which 16 run softmax attention (`full_attention_interval: 4`) and 48 run a gated delta net |
| Hidden | 5120, 24 query heads, **4 KV heads**, `head_dim` 256 |
| Attention | `attn_output_gate`, so `q_proj` is `12288 x 5120` and the fused QKV is `14336 x 5120` |
| Linear attention | 16 key heads, 48 value heads, head dim 128, `in_proj_qkvz` is `16384 x 5120` |
| MLP | 17408 |
| Vocabulary | 248320 |
| Parameters | 27.78B in the checkpoint, 29.07B as Megatron builds it (MTP layer and vision tower included) |

`kernels verify-shapes` compares every declared projection against the real module graph and
all ten match, including the five linear-attention ones.

Three hard caps follow, and each was hit rather than assumed:

1. **TP <= 4.** Four KV heads. `tp=8` is refused by the shape derivation and by Megatron
   (`TP size 8 should be less than or equal to num_query_groups 4`).
2. **PP <= 2.** The multimodal provider derives rotary positions from `input_ids` when
   `position_ids` is absent, and Megatron passes all-None to a middle pipeline stage.
   `tp4/pp2` trains; `tp2/pp4` dies on rank 3 with `'NoneType' object has no attribute
   'shape'`. Only the first and last stages are safe.
3. **CP = 1.** The vendored Megatron asserts `context_parallel_size == 1` whenever a gated
   delta-net layer is present: `Gated delta net does not support context parallel for now`.
   Every `cp=2` candidate fails in `ConfigContainer.validate()` before any GPU work. The
   newer Megatron in the base environment does implement head-parallel GDN CP, so this is a
   version limit rather than an architectural one.

A fourth fact drives long-context memory: the last stage holds a vocabulary-sized tensor.
The measured activation at 16k is 15.1 GiB, which is `vocab * 4 bytes * seq`, the full
vocabulary in fp32, replicated per rank. The library recipe also auto-detects MTP from the
checkpoint (`mtp_num_hidden_layers: 1`) and trains it. Whether that adds a second copy is
untested: the no-MTP probe failed on the PP=2 position-ids path before it could measure, so
the MTP share below is an inference from the recipe, not a measurement.

## Inference: TP4 at every context

Measured on 4 GPUs, `max-model-len 131072`, `gpu-memory-utilization 0.9`. Weights take
51.7 GiB at TP1, 25.9 at TP2 and 12.9 at TP4, which sets the KV capacity.

Prefill, whole-replica throughput in tokens/s:

| input | TP1 c=1 | TP4 c=1 | TP4 c=16 | TP4 c=64 |
|---|---|---|---|---|
| 1k | 941 | 799 | | 9438 |
| 4k | 2485 | 7157 | | 9971 |
| 16k | 2566 | 8669 | 9531 | |
| 32k | | 8577 | 9088 | |
| 65k | | 7924 | | |

Decode, whole-replica output tokens/s:

| context | TP1 c=8 | TP4 c=1 | TP4 c=2 | TP4 c=4 | TP4 c=8 | TP4 c=16 | TP4 c=32 |
|---|---|---|---|---|---|---|---|
| 1k | 314 | 114 | | | 767 | | 1993 |
| 16k | | 106 | | 362 | | 698 | |
| 32k | | 97 | | 308 | 454 | | |
| 65k | | 83 | 152 | 232 | | | |
| 98k | | 72 | 128 | | | | |
| 115k | | 68 | 119 | | | | |

Prefill scales sublinearly with TP: TP4 delivers 3.5x of TP1 on 4x the GPUs, so per-GPU
prefill throughput is essentially flat. Decode is 2.2-2.4x of TP1 at 4x the GPUs, so TP4 is
slightly worse per GPU. What TP4 buys is latency (16k TTFT 1.9 s against 6.4 s) and KV
capacity, and the second is what makes 64k and 128k servable at all.

KV capacity per replica, from a 0.9 utilization budget less weights and 3 GiB of non-torch
overhead:

| TP | weights | KV budget | per token | tokens held | sequences at 16k | at 64k | at 128k |
|---|---|---|---|---|---|---|---|
| 1 | 51.7 GiB | 30.8 GiB | 64 KiB | 504k | 31 | 8 | 4 |
| 2 | 25.9 GiB | 56.6 GiB | 32 KiB | 1.85M | 113 | 28 | 14 |
| 4 | 12.9 GiB | 69.6 GiB | 16 KiB | 4.56M | 278 | 69 | 35 |

**Recommendation: TP4 at every context length.** TP1 and TP2 buy nothing per GPU and cost
KV capacity; TP4 is the only setting that fits a useful number of long sequences.

## Training: TP4/PP2, fitted rather than hand-derived

Measured on H20, `tp4/pp2/cp1`, full activation recompute, bf16 parameters, fp32 gradients,
distributed Adam:

| GPUs | sequence | microbatches | DP | step time | peak allocated |
|---|---|---|---|---|---|
| 8 | 16384 | 1 | 1 | 8.32 s | 78.6 GiB |
| 8 | 16384 | 4 | 1 | 21.62 s | 78.7 GiB |
| 8 | 24576 | 1 | 1 | 13.21 s | 86.1 GiB |
| 8 | 32768 | 1 | 1 | OOM | predicted 95.6 GiB on a 95 GiB device |
| 16 | 32768 | 1 | 2 | 18.54 s | 74.6 GiB |
| 16 | 32768 | 4 | 2 | 48.08 s | 74.8 GiB |
| 16 | 65536 | 1 | 2 | stall (600 s NCCL all-gather timeout at 90 GiB of 95) | - |

Fitting these through `HydraulisTrainingCost` (`srl.cost.training`, the same artifact class
every other model in this codebase uses, not a one-off script) needed a third sequence
length: 16384 and 32768 are an exact factor of two apart, which leaves the design's linear
and quadratic token columns collinear and the fit rank-deficient (`check_latency_rank`
caught this directly: "two sequence lengths leave the linear and quadratic token columns
collinear"). Adding the 24576 point broke the collinearity. Fitting against the five
successful points plus the one OOM boundary gives, at `tp4/pp2/cp1`:

- **Latency.** MRE 1.51% against the five measured steps (held in-sample, five points is
  too few to also hold out a fold). `per_step_overhead` fits to zero: this workload has no
  measurable fixed cost per step, only a per-microbatch one (`microbatch_overhead=1.81 s`)
  and small token-linear and token-quadratic rates.
- **Memory.** MRE 1.02% against the five peaks, and the OOM boundary is satisfied as an
  inequality (the fitted model predicts 95.6 GiB for the 8-GPU 32768-token candidate that
  actually OOM'd on a 95 GiB device). Fitting this needed one adjacent bug fix: an OOM
  candidate's `parameter_count` was defaulting to the checkpoint's *declared* count
  (27.78B) rather than the *measured* 29.07B Megatron actually builds (MTP head and vision
  tower included), which silently split one real topology into two artifact cells that
  `entry_for` could not tell apart. Fixed in `collect.py` (report `None` rather than a
  guessed count when a candidate never measures one) and in `hydraulis.py`
  (`_training_records` backfills a missing count from a sibling at the same TP/PP before
  falling back to the checkpoint's declared value).

The peak is dominated by the per-rank, non-TP-divided vocabulary logits: at 16k the
activation share is 15.1 GiB, matching `vocab * 4 bytes(fp32) * seq` almost exactly, and
growing linearly with sequence length. That term is what keeps rising with context no
matter how many GPUs join, because it is not divided by tensor, pipeline, or data
parallelism.

## Context parallelism: never measured, now an explicit, flagged extrapolation

The vendored Megatron asserts `context_parallel_size == 1` whenever a gated-delta-net layer
is present, so **no `cp > 1` training step has ever run**, on this checkpoint or any other
this codebase fits. `HydraulisTrainingCost.fit` reflects that structurally: `_training_records`
drops every `cp != 1` record before fitting, so an artifact can only ever contain `cp == 1`
cells. A `cp > 1` query is therefore refused by default (`ValueError`, not a silent guess),
and only answered when the caller opts in with `HydraulisTrainingCost.from_artifact(...,
allow_context_parallel=True)`. That path applies the division the vendored Megatron's own
head-parallel CP would apply once it is wired up (`GatedDeltaNet.cp_size`, see
`mamba_context_parallel.py`'s `_all_to_all_cp2hp`/`_all_to_all_hp2cp` conversions):

- activation (`activation_bytes`) and the optimizer shard (`parameter_inventory`) divide by
  `cp`, same as the 16k-vs-32k-at-dp2 comparison already confirms for `dp`;
- the latency model's token-proportional and quadratic terms divide by `cp` (ring attention
  spreads the fixed `O(sum(s_i^2))` pairwise cost evenly across the `cp` ranks that jointly
  hold the sequence; the fixed per-step and per-microbatch costs do not shrink);
- `TrainingOps` divides the gated-delta-net head count by `cp` on top of `tp` (the scan
  processes the full local sequence for its share of heads, it does not get fewer tokens),
  and refuses a `cp` that does not divide the TP-local head count, the same assertion
  Megatron itself makes.

Every prediction that uses `cp > 1` carries an explicit note
(`"cp=N is an analytical projection from the cp=1 fit, not a measured cell"`) in
`TrainingCostEstimate.diagnostics["extrapolation_notes"]`, and is marked `*` in every table
below. CP is admissible up to 4 at TP4: it must divide both `linear_num_key_heads / tp = 4`
and `linear_num_value_heads / tp = 12`, so `gcd(4, 12) = 4` bounds it.

Twenty-one unit tests (`tests/test_training_context_parallel.py`) check this machinery
directly: that `cp=1` is never refused and `cp>1` is refused by default; that `cp=1`
predictions are byte-identical whether or not the model was constructed with
`allow_context_parallel`; that the memory projection halves the activation term (not the
static state) at `cp=2` and quarters it at `cp=4`; that the latency projection halves the
token-proportional part at `cp=2` while leaving the per-microbatch and per-step parts alone;
that `TrainingOps` divides the GDN head count by `cp` and refuses a non-dividing one; and
that `ComposedTraining` (the kernel-table-composed model, exercised here with a stub
database so the arithmetic is checked independent of kernel accuracy) divides GEMM,
elementwise and logits tokens by `cp`, divides the attention *result* by `cp` rather than
its input length (squaring a halved length would scale the quadratic term by `cp^2` rather
than `cp`), and leaves `cp=1` completely unaffected.

**This is model-side reasoning, not a measurement.** No real step at `cp > 1` backs any `*`
number below, on this architecture or any other.

## Per-context plan: parallelism, GPU split, and step time

Workload: 32 prompts x 16 responses (group size) = **512 trajectories per step**, prompt and
response each half the context cap, reproduced by
`python examples/analysis/qwen35_27b_parallel_plan.py --markdown`. Every number is read from
`HydraulisTrainingCost` against the fitted artifact
(`outputs/profile/qwen35_27b/hydraulis_training.json`) for training, and from the measured
vLLM benchmarks for rollout. Under staleness 1, rollout(k+1) and train(k) run concurrently
on separate GPU pools, so a step costs the larger phase, **but only when the machine is
actually partitioned**: a plan that cannot spare GPUs for a concurrent rollout replica falls
back to training the whole machine and then rolling out on the whole machine in sequence,
marked `!` below, which does not realize staleness 1's overlap and is a time-shared total,
not a steady-state per-step figure. GPU splits are restricted to powers of two: 512
trajectories only divide evenly across a data-parallel degree that is itself a power of
two, and `TrainingWorkload` refuses an uneven split rather than silently approximating it.

**Read the microbatch-count column before trusting a `train s` figure.** The fit covers 1
and 4 microbatches per rank; a real step at these GPU counts runs 64-512 (`TRAJECTORIES /
dp`), a 16x-128x extrapolation past what was measured. It is a *linear* extrapolation of a
quantity 1F1B defines as linear (`slots = nmb + pp - 1`, the same reasoning that already
lets this model skip measuring every pipeline depth), not a curve-fit guess, but it has not
been checked against a real step with more than 4 microbatches on this checkpoint.

### 16k

- **Inference: TP4.** 698 output tokens/s per 4-GPU replica at 16 concurrent, KV holds 4.56M
  tokens or 278 sequences.
- **Training: tp4/pp2/cp1**, no extrapolation beyond the microbatch count above.

| GPUs | train(tp,pp,cp,dp) | microbatches/rank | split | peak | train | rollout | **step** |
|---|---|---|---|---|---|---|---|
| 16 | (4,2,1,1) | 512 | 8/8 | 79.4 GiB | 2173 s | 3238 s | **3238 s** |
| 32 | (4,2,1,2) | 256 | 16/16 | 59.1 GiB | 1089 s | 1619 s | **1619 s** |
| 64 | (4,2,1,4) | 128 | 32/32 | 49.0 GiB | 547 s | 809 s | **809 s** |
| 128 | (4,2,1,8) | 64 | 64/64 | 43.9 GiB | 275 s | 405 s | **405 s** |

### 32k

- **Inference: TP4.** 454 output tokens/s per replica at 8 concurrent, 139 sequences fit.
- **Training: tp4/pp2/cp1.** 16 GPUs minimum: 8 GPUs OOMs (measured, and the fitted model
  predicts 95.6 GiB on a 95 GiB device for exactly that candidate).

| GPUs | train(tp,pp,cp,dp) | microbatches/rank | split | peak | train | rollout | **step** |
|---|---|---|---|---|---|---|---|
| 16 | (4,2,1,2) | 256 | 16/16 | 75.3 GiB | 2425 s | 4852 s | **7277 s !** |
| 32 | (4,2,1,2) | 256 | 16/16 | 75.3 GiB | 2425 s | 4852 s | **4852 s** |
| 64 | (4,2,1,2) | 256 | 16/48 | 75.3 GiB | 2425 s | 1617 s | **2425 s** |
| 128 | (4,2,1,8) | 64 | 64/64 | 60.0 GiB | 613 s | 1213 s | **1213 s** |

16 GPUs cannot spare any for a concurrent rollout replica once training takes its minimum
16, so that row is time-shared rather than partitioned.

### 64k

- **Inference: TP4.** 232 output tokens/s per replica at 4 concurrent, 69 sequences fit.
  Inference alone is unaffected by any of this: it was actually run end to end.
- **Training: cp=1 does not fit at any GPU count** (16-GPU 32768-token stalled in a 600 s
  NCCL timeout at 90 GiB of 95, and the per-rank logits term does not shrink with more
  GPUs). Every plan below uses `cp=2`, the **analytical extrapolation** flagged above.

| GPUs | train(tp,pp,cp,dp) | microbatches/rank | split | peak | train | rollout | **step** |
|---|---|---|---|---|---|---|---|
| 16 | (4,2,2,1) * | 512 | 16/16 | 75.3 GiB | 6756 s | 18545 s | **25300 s !** |
| 32 | (4,2,2,1) * | 512 | 16/16 | 75.3 GiB | 6756 s | 18545 s | **18545 s** |
| 64 | (4,2,2,1) * | 512 | 16/48 | 75.3 GiB | 6756 s | 6182 s | **6756 s** |
| 128 | (4,2,2,2) * | 256 | 32/96 | 65.1 GiB | 3384 s | 3091 s | **3384 s** |

### 128k

- **Inference: TP4.** 119 output tokens/s per replica at 2 concurrent, 35 sequences fit.
  This is the only phase of the four that is actually measured at 128k.
- **Training: cp=4 required**, the admissible maximum (`gcd(linear_num_key_heads / tp,
  linear_num_value_heads / tp) = gcd(4, 12) = 4`). 16 GPUs cannot even form one `tp4 * pp2 *
  cp4 = 32`-rank mesh, so it is infeasible outright, not merely slow.

| GPUs | train(tp,pp,cp,dp) | microbatches/rank | split | peak | train | rollout | **step** |
|---|---|---|---|---|---|---|---|
| 16 | - | - | - | - | - | - | **infeasible** |
| 32 | (4,2,4,1) * | 512 | 32/32 | 65.1 GiB | 10587 s | 35712 s | **46299 s !** |
| 64 | (4,2,4,1) * | 512 | 32/32 | 65.1 GiB | 10587 s | 35712 s | **35712 s** |
| 128 | (4,2,4,1) * | 512 | 32/96 | 65.1 GiB | 10587 s | 11904 s | **11904 s** |

### Regularities across the table

Rollout dominates once the context passes 16k, because decode throughput per replica falls
faster (698 -> 119 tokens/s, a 5.9x drop from 16k to 128k) than the per-step token volume
grows (8x). Training wants as much data parallelism as it can get, because the fit found
almost no fixed per-step cost (`per_step_overhead` fits to zero) and a comparatively large
per-microbatch one (`microbatch_overhead = 1.81 s`), so halving the microbatch count per
rank by doubling `dp` is close to a direct halving of `train_s`: this is also why several
rows keep `train_gpus` at the smallest feasible mesh rather than growing it with `world`,
preferring to hand the rest to rollout instead. And whenever training alone needs every GPU
the world has (32k on 16, 64k on 16, 128k on 32), the row is necessarily time-shared: there
is nothing left to run a concurrent rollout replica on, so staleness 1's overlap is not
available at that GPU count regardless of how the trainer is configured.

## What would change the answer

1. **Actually enable GDN context parallelism in the vendored Megatron.** Everything at 64k
   and 128k above is the analytical `cp>1` projection, never a measurement. Landing the
   newer base environment's head-parallel CP (`GatedDeltaNet.cp_size`,
   `_all_to_all_cp2hp`/`_all_to_all_hp2cp`) into the pinned Megatron would let a real sweep
   confirm or correct the division-by-`cp` assumption this plan currently takes on faith,
   and would let `HydraulisTrainingCost.fit` add real `cp=2`/`cp=4` cells instead of the
   caller having to opt into `allow_context_parallel=True`.
2. **Stop training MTP, if it is what doubles the logits.** The recipe enables the head from
   the checkpoint, and an RL run does not need it. If it accounts for half the per-token
   activation, every ceiling in this plan would roughly double. This is unverified: at
   `mtp_num_layers=0` the PP=2 candidate dies on the `position_ids is None` branch before
   measuring, so the provider needs a fix and a single 16k no-MTP run before the lever can
   be counted on.
3. **Use a chunked or fused cross-entropy.** The peak is the logits tensor, so any loss that
   does not materialize the full vocabulary per rank raises every ceiling more than any
   parallelism knob.
4. **Measure a step with more than 4 microbatches per rank.** Every table above extrapolates
   the fitted latency form 16x-128x past the measured microbatch-count range. The
   extrapolation is linear by 1F1B's own definition of `slots`, not a curve fit, but
   nothing here has checked it against a real step at that scale.

## Calibration summary

- Shapes: 10 of 10 declared projections match the real module graph (`kernels
  verify-shapes`).
- Rollout: 131,217 scheduler-iteration records from the TP4 sweep, plus TP1 baselines. Used
  directly as measured throughput, not through a fitted model.
- Training: `HydraulisTrainingCost` fitted from five real steps at `tp4/pp2/cp1` spanning
  three sequence lengths (16384, 24576, 32768) and two microbatch counts (1, 4), needed to
  reach the rank-4 design the latency form requires. Latency MRE 1.51%, memory MRE 1.02%
  against those five points; the one measured OOM boundary (32768 tokens, 8 GPUs) is
  satisfied as an inequality (predicted 95.6 GiB on a 95 GiB device). Fixed one real bug in
  the fitting pipeline along the way: an OOM candidate's `parameter_count` defaulted to the
  checkpoint's declared count instead of Megatron's measured one, silently splitting one
  topology into two artifact cells (`collect.py`, `hydraulis.py::_training_records`).
- Context parallelism: structurally unmeasurable on this stack (the vendored Megatron
  refuses `cp > 1` for a GDN layer), so modelled as an explicit, opt-in analytical
  projection rather than a fitted cell. 21 unit tests
  (`tests/test_training_context_parallel.py`) check the projection's arithmetic and that it
  changes nothing at `cp=1`.
- Microbatch count: the fit's design covers `nmb in {1, 4}`; the plan above evaluates it at
  `nmb` up to 512. Linear in the model's own terms, unverified at that scale.

## Kernel tables

`srl-profile kernels run` at TP4 wrote gemm, attention, elementwise and gdn tables and
`fit-floor` fitted their floors. The holdout gate does **not** pass for this model:

| family | held-out points | MRE | p90 | max |
|---|---|---|---|---|
| gemm | 221 | 6.8% | 18.3% | 67.2% |
| context_attention | 20 | 10.0% | 26.8% | 34.4% |
| generation_attention | 88 | 15.3% | 32.8% | 47.6% |
| gdn_context | 9 | 3.7% | 13.5% | 13.5% |
| gdn_decode | 8 | 9.4% | 16.6% | 16.6% |
| elementwise | 30 | 8.2% | 17.5% | 38.4% |

The dense grid improved GEMM from 8.7% to 6.8% but left 34 slices unbracketed, and the
generation-attention surface got worse as the grid grew, so this is not only a sampling
problem: the `qwen35_dense` shapes include extreme aspect ratios (`in_proj_ba` is 24 rows
wide, `in_proj_qkvz` 16384) that the utility surface does not fit well. The plan above does
not depend on these tables. It uses the measured end-to-end steps and the calibrated memory
model. Treat the kernel tables as unusable for this checkpoint until the surface is
reworked.
