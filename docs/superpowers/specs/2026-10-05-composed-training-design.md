# Spec B: Kernel-Composed Training Latency

**Date:** 2026-10-05
**Status:** implemented. B0-B5 delivered; B4's DP term remains unvalidated by design (§3 B8).
**Repo:** `/apdcephfs_zwfy/share_303760348/lhy/SimRL`
**Depends on:** `2026-10-04-collective-ops-design.md` (Spec A), implemented and gated.

## 0. Result, and where this design was wrong

`ComposedTraining` predicts the 22 out-of-sample Megatron THD step times at **MRE 12.63%**,
signed mean **-12.63%**, against the fitted polynomial's 2.73% in-sample. No fitted
coefficients. `tests/test_training_composed_accuracy.py` pins it.

**One unexplained term carries the whole residual**, and it is the open item: every record
under-predicts by 4.7-25.6% of its own prediction, proportional to total predicted time
(c = 0.185, r^2 = 0.932). Measurement excludes the GEMM table (`te.Linear` vs `F.linear` is
1.001x), the optimizer step, recompute, and the unmodelled residual adds. At ~16 ms per
layer against ~29 ms of modelled GEMM per layer it is too large to be a pointwise kernel,
and every table it would live in holds 2.8-5.7% on holdout. Resolving it needs a profiler
trace rather than more composition.

The path there, because the terms that mattered were not the terms this design predicted:

| MRE | signed | change |
|---|---|---|
| 18.85% | -18.85% | GEMM + rollout attention + collectives |
| 17.46% | -17.46% | + elementwise (-1.39pp) |
| 17.23% | -17.23% | + optimizer step (-0.23pp) |
| 7.63% | +0.27% | + measured THD attention, both directions (**-9.60pp**) |
| 7.62% | +0.30% | + pipeline p2p sends (-0.01pp) |
| **12.63%** | **-12.63%** | + per-sequence attention (**+5.01pp**) |

The last row raises the pooled MRE and is still correct. Attention does not attend across a
`cu_seqlens` boundary, so a pack's quadratic work is `sum_i s_i^2`; charging `(sum_i s_i)^2`
over-predicted packed records by roughly what the unexplained term under-predicts, and the
7.62% was two errors cancelling. The validation set isolates this directly — five records
hold 12288 tokens as (12288), (10240,2048), (6144,6144), (4096x3), (3072x4), and measured
step time falls monotonically across them while the pack-total model predicted all five
identically. The correction removes 1.02-1.10x of that measured effect and orders the five
at r = 0.9995. Reverting it to recover the lower number would mean keeping a known-wrong
term because it happened to offset an unknown one.

Elementwise and the optimizer step are the two terms §2 identifies as missing, and together
they closed 1.6pp of a 19pp gap. The gap was almost entirely **attention priced from the
wrong table**, and §1.2(c) of this document is the reason it was priced that way.

**§1.2(c) is wrong for the layout training uses.** The claim — cuDNN FusedAttention on
Hopper+, not flash — came from calling TE's selector with `attn_mask_type="causal"`. THD
requires `padding_causal` (`DotProductAttention.forward` asserts `"padding" in
attn_mask_type`), and a probe built with `causal` reports *every* backend unavailable, which
reads as a statement about the hardware. Measured with the real module and real packed
inputs, TE logs `Running with FlashAttention backend (version 2.8.3)` and disables
FusedAttention with "no backend supports the provided input". All 112 rows of the new table
carry `kernel_source=FlashAttention`.

Two consequences, both measured on H20 at tp=2 (16 query heads, 4 KV heads, head_dim 128):

- **The forward was 1.59-1.84x low.** The rollout `context_attention` table measures vLLM's
  prefill kernel over a BSHD batch; its grid varies a batch dimension a THD pack does not
  have, so every training query sat at the edge of the measured surface. 0.566 ms vs 1.040 ms
  at 4096 tokens; 15.113 ms vs 24.058 ms at 22528.
- **The backward was 2.24-2.42x low.** B3's `attn_bwd_fwd_ratio=1.0` rested on a cuDNN
  benchmark for a layout that is not run here. The real ratio exceeds 2 because the backward
  computes dQ, dK and dV against the forward's single output and FlashAttention
  re-materializes the probabilities rather than storing them. It also drifts with sequence
  length, so no scalar is right and the backward gets its own measured table.

Combined, the attention term was understated 2.35-2.60x while being 15.2% of the prediction.

**What found it was the signed error, not the magnitude.** For as long as every one of the
22 records under-predicted, the signed and absolute means were equal — the signature of a
missing additive term rather than a noisy one. The per-term decomposition then said which
terms were large enough to be it, and the residual's proportionality to work across a 192x
token range excluded everything charged once per step. That is the §3 B7 decision doing the
job it was added for, and it is the strongest argument in this document for a kernel sum
over a fitted polynomial: a 17% residual on a fit is a coefficient to adjust, while a 17%
residual here named the op.

**One measurement nearly went in wrong twice.** A `.post1` flash-attn wheel is rejected by
TE's `<= 2.8.3` upper bound (PEP 440 orders `2.8.3.post1` *above* it), TE falls back to
`UnfusedDotProductAttention`, and the only signal is a debug-level log line. The first run
of the new sweep measured that fallback. `_normalize_flash_attn_version` is the existing fix;
the profiler now calls it before importing TE and refuses to write a table whose
`kernel_source` is the unfused path.

Also corrected: the optimizer sweep's 4.096B-parameter grid point recorded a kernel that
never ran. Its state needs ~115 GiB against H20's 95 GiB, the allocating `step()` OOM'd
inside TE, the exception was swallowed, and the timed step iterated an empty state list in
0.60 ms where 2.048B parameters took 21.56 ms — an implied 190 TB/s on a 4.0 TB/s card.
Nothing downstream rejects a *fast* row: a floor fit accepts it and the holdout MRE
improves, because the point is easy to interpolate. The profiler now verifies the state
allocation.

Final per-term shares, which is what a future change should be compared against rather than
the pooled MRE:

| term | share | term | share |
|---|---|---|---|
| `gemm_bwd` | 34.6% | `tp_collective` | 7.8% |
| `gemm_fwd` | 34.4% | `logits` | 1.8% |
| `attn_bwd` | 10.5% | `elementwise` | 1.6% |
| `attn_fwd` | 9.1% | `optimizer` | 0.3% |
| | | `pp_comm` | 0.0% |

Holdouts for the three new ops: `training_attention_forward` 5.66% (48 points),
`training_attention_backward` 3.83% (48), `optimizer_step` 2.84% (12). `gemm` holds at
3.72% and the end-to-end rollout numbers are unchanged, so the rollout path is untouched.

## 1. Context

`srl/cost/training/latency.py` predicts a training step from four fitted coefficients
over whole-step wall times. Its own docstring names the alternative: "feeding training GEMMs
through the same kernel tables is the principled next step rather than another fitted
polynomial term." The rollout side already does exactly that and reaches 2.96-3.64% MRE at
tp=2 by summing measured kernels.

Spec A removed the blocking gap: the collective family now covers the five collectives both
training backends issue, keyed by `(world_size, nodes)`. What remains is the composition
itself.

### 1.1 Why a kernel sum rather than a better fit

The fitted model is accurate where it was measured (2.73% in-sample on the THD grid) and
structurally silent outside it. Three recorded limitations are all of that kind:

- `docs/reference/limitations.md` forbids a quadratic token term beyond the profiled range,
  because the fitted coefficient implies ~18000% of H20's bf16 throughput: it absorbs
  residual rather than representing attention.
- PP is derivable only because the features are per-layer and the bubble is analytic; a
  schedule other than 1F1B has no representation at all.
- Every new `(tp, recompute, dtype)` cell needs its own profiling campaign.

A kernel sum has the opposite failure mode: it is wrong in ways that are attributable to a
named op. That is the whole reason to build it, and it is also why the gate below reports a
per-term decomposition rather than one number.

### 1.2 What this spec found before designing anything

Three facts, each measured rather than assumed, and each one changing the design:

**(a) Three of four backward dgrad shapes were not in the measured table.** The original
kernel spec stated backward reuses forward "at zero extra measurement cost". That was true
while `gemm` sliced on `(gemm_dtype,)` alone; per-`(n, k)` slicing ended it, because a
dgrad's swapped `(k_fwd, n_fwd)` is a different slice. Verified by querying the live table:
`qkv_proj`, `gate_up_proj` and `down_proj` dgrads all raise `KernelCoverageError`, and only
`o_proj` resolves, because its `n` and `k` happen to be measured in both orders.
`gemm_all_shapes` emitted forward + wgrad + logits only.

**(b) The wgrad triple itself was wrong.** `gemm_wgrad_shapes` emitted
`(n_fwd, tokens, k_fwd)`. The table's row means `F.linear(A[m,k], W[n,k]) -> C[m,n]`
(`gemm.py:69-72`), and `dW` has the weight's shape, so the correct row is
`(n_fwd, k_fwd, tokens)`. The FLOP count is identical because the product is symmetric,
which is why nothing caught it: it was a lookup into the wrong slice, not an implausible
cost. Confirmed against `F.linear`: at `o_proj` tp=2 the old triple yields a `[4096, 4096]`
output where `dW` is `[4096, 2048]`. Two of Qwen3-8B's four projections were affected.

**(c) Attention on this hardware is cuDNN, not flash.** ⚠️ **This is wrong for THD, and
§0 records the correction and what it cost.** The claim below is retained because it is the
reasoning that produced a 9.6pp error, and the failure mode — a backend probe built with
reconstructed parameters that disagree with the real call — generalizes past this instance.

The original text: Transformer Engine disables FlashAttention on Hopper+ deliberately
(`dot_product_attention/utils.py:923`: "give FusedAttention preference on Hopper+ for
performance reasons"). Verified by calling TE's own selector with Megatron's THD attention
parameters: `flash=False`, `cc=(9,0)`. So the attention ops to measure are
`fused_attn_fwd` / `fused_attn_bwd`, and measuring `flash_attn_varlen_*` would time a
kernel this machine never runs. This is the same class of error as the one
`docs/profiling/tensor-parallel.md` records for `torch.distributed.all_reduce`.

What was actually wrong: "Megatron's THD attention parameters" were reconstructed by hand
for the probe, and used `attn_mask_type="causal"` where THD requires `padding_causal`. With
that mask TE reports `flash=False` **and** `fused=False` — no backend at all, which reads as
a hardware fact rather than as an invalid parameter set. Run against the real module with
real packed inputs, TE selects FlashAttention 2.8.3. The lesson the design should have
carried: a selector probe is only evidence if the call it describes is one that runs, so the
backend is now read from TE's own `_attention_backends` dict *after* a real forward, and the
sweep refuses to write a table whose backend is the unfused fallback.

(a) and (b) are fixed as prerequisite work, since no composition can be built on a table
that refuses three quarters of its backward queries.

## 2. The per-layer op list, measured against megatron-core

For one Qwen3 layer at `tp=2`, `sequence_parallel=True`, `recompute_granularity="full"`.
`s·b` is the token count, `h` the full hidden size. Verified against
`megatron/core/tensor_parallel/layers.py` and Transformer Engine's
`module/layernorm_linear.py`, `module/linear.py`, `module/base.py`.

### GEMMs, 4 per layer forward

| projection | parallel | n (per rank) | k (per rank) |
|---|---|---|---|
| `qkv` | column | `(nq + 2·nkv)·hd / tp` | `h` |
| `o_proj` | row | `h` | `nq·hd / tp` |
| `gate_up` | column | `2·ffn / tp` | `h` |
| `down_proj` | row | `h` | `ffn / tp` |

`m = s·b` for all four: the all-gather restores the full sequence before the GEMM.
Backward adds a dgrad `(m, k_fwd, n_fwd)` and a wgrad `(n_fwd, k_fwd, m)` per projection.

### Collectives: 4 forward, 6 backward, and forward is paid twice

| pass | all_gather | reduce_scatter | sizes |
|---|---|---|---|
| forward | 2 (qkv in, fc1 in) | 2 (proj out, fc2 out) | AG `s·b·h/tp` -> `s·b·h`; RS the reverse |
| recompute | 2 | 2 | identical: the replay re-issues them |
| backward | 4 | 2 | see below |

Backward's asymmetry is load-bearing and would be easy to get wrong. Column-parallel layers
issue **both** an all-gather (to rebuild `ln_out` for wgrad,
`layernorm_linear.py:390,659`) and a reduce-scatter of the dgrad (`:736`). Row-parallel
layers issue **only** an all-gather of `grad_output` (`module/base.py:1083,1089`), because
their dgrad is already column-sharded. So 2 column layers x 2 + 2 row layers x 1 = 6
all-gathers... **4**, because each column layer's AG and each row layer's AG is one each:
2 (column, for wgrad) + 2 (row, for grad_output) = 4 all-gathers, and 2 reduce-scatters
(column only).

Every forward collective carries the **full** hidden size; `tp` shards the sequence
dimension, not the hidden one. Totals per layer per step under full recompute:
**8 all-gather + 6 reduce-scatter**.

The recompute multiplier is `CheckpointFunction.backward` re-calling `ctx.run_function`
under `enable_grad` (`tensor_parallel/random.py:621`), which replays the whole
`custom_forward` loop (`transformer_block.py:483-514`), collectives included. Treating
recompute as compute-only would understate the collective term by a third.

### Elementwise

Qwen3 specifics that change the count:

- The two RMSNorms are **fused into** `TELayerNormColumnParallelLinear`
  (`layernorm_linear.py:202`), not separate kernels. Backward has two `rmsnorm_bwd`.
- `q_norm` and `k_norm` are **two additional** RMSNorm kernels over `head_dim`
  (`qk_layernorm=True` for Qwen3), applied at `attention.py:1489,1492`.
- RoPE is **two** kernels (query, key) unless `apply_rope_fusion` is set; it is not by
  default.
- SwiGLU is **three** kernels (chunk + silu + mul) unless `bias_activation_fusion`, which
  defaults `False` (`transformer_config.py:423`).
- Two residual adds, each one fused `bias_dropout_add_fused_train` kernel.

Elementwise **is summed** for training, unlike the rollout model. The rollout omission is
justified by CUDA-graph replay absorbing elementwise cost into the fitted GEMM and attention
utilities (measured: adding it moved decode 4.84% -> 11.18%). Megatron runs no CUDA graphs
on this path, so the cost is genuinely exposed. This is the one place where the two composed
models legitimately differ, and `docs/superpowers/specs/2026-08-26-kernel-op-abstraction-design.md`
anticipated it: "The table exists for the FSDP training backend, which has no CUDA graphs
and will need an explicit term."

### Optimizer, once per step

With the distributed optimizer: grad reduce-scatter over the DP group
(`param_and_grad_buffer.py:608`), fp32 master copy, grad-norm (`multi_tensor_l2norm` plus
two scalar all-reduces, `clip_grads.py:112,131,134`), fused Adam on the `1/dp` shard, and a
param all-gather (`distrib_optimizer.py:2667` -> `param_and_grad_buffer.py:418`).

The two scalar all-reduces are deliberately **not** modelled: a scalar collective is pure
launch floor, which the fitted per-call floor already represents, and adding two per step
against a multi-second step is below the noise of every other term.

## 3. Design decisions

| # | Decision | Rationale |
|---|---|---|
| B1 | `TrainingOps`, a sibling to `ModelOps`, emitting a declarative op list per microbatch | Same reason `ModelOps` exists: an inline multiplier is one indentation level from a plausible 36x error. Training has three multipliers (layers, recompute, fwd/bwd) so the risk is higher |
| B2 | `ComposedTraining` implements `TrainingLatencyModel` only; memory reuses `HydraulisMemoryModel` | Memory is already analytic (`inventory.py` + `activation.py`) and needs no kernels. Composing it would replace a working model with a worse one |
| B3 | Two new ops: `attention_backward` and `optimizer_step` | The only measurement families training needs that rollout does not. Both go through the existing `KernelOp` protocol |
| B4 | `attention_backward` measures **cuDNN** `fused_attn_bwd`, with a `kernel_source` tag | §1.2(c): flash is disabled on sm90 by TE. Measuring flash would time a kernel that never runs |
| B5 | Fix `gemm_wgrad_shapes` and add `gemm_dgrad_shapes`; re-sweep the GEMM table | §1.2(a,b). Prerequisite: three quarters of backward queries raise without it |
| B6 | PP stays analytic via the existing `pipeline_slots`; the bubble is not composed | Already validated: PP is derivable because the features are per-layer. Reusing it keeps B's novelty confined to the per-microbatch cost |
| B7 | Report a per-term decomposition alongside the MRE | A kernel sum's value is attributable error. One pooled number throws that away |
| B8 | DP collective terms are implemented but **flagged unvalidated** | All 22 THD records are `dp=1`, so the term cannot be scored. Spec A gated the tables; only the composition is untested |

### B2 in detail

`ComposedTraining` is a `TrainingLatencyModel`, and `build_training_cost` pairs it with
`HydraulisMemoryModel` from an existing artifact. That keeps three properties: the OOM
check still compares against the quantity that causes OOM, the memory model's own
validation stays valid, and a composed-latency run can be A/B'd against a fitted-latency run
with memory held identical.

Consequence to state plainly: `ComposedTraining` needs **no `fit()`**. It has no
coefficients. It is constructed from a kernel table root plus a model checkpoint, like
`ComposedRollout`. The memory artifact is a separate input.

## 4. Architecture

```
srl/cost/kernel/ops/
├── attention_backward.py   AttentionBackwardKernelOp   (new; family "attention")
└── optimizer.py            OptimizerStepKernelOp       (new; family "optimizer")

srl/profiler/kernels/
├── attention_bwd.py        cuDNN fused_attn_bwd timing  (new)
└── optimizer.py            fused Adam timing            (new)

srl/cost/training/
├── training_ops.py         TrainingOps: the per-microbatch op list   (new)
└── composed.py             ComposedTraining(TrainingLatencyModel)    (new)
```

`TrainingOps` mirrors `ModelOps`: construct from a `ModelDescriptor` plus a degree, shard at
construction, and expose `*_queries(token_count)` returning `(query, multiplier)` pairs. It
reuses `transformer_gemm_shapes`, `rank_head_counts` and `model_head_dim` rather than
re-deriving any sharding, for the reason `model_ops.py` already records: a composition that
shards differently from the sweep queries shapes that were never measured.

### Interfaces preserved

`ComposedTraining.predict(workload, parallelism) -> float` is the existing abstract method
(`cost/training/base.py:207`). `TrainingWorkload`, `TrainingParallelism` and
`MicrobatchPlanner` are untouched: the planner already turns ragged RL lengths into a
microbatch plan, and `ComposedTraining` prices each plan entry. No simulator change.

## 5. Phasing

| Phase | Scope | Gate | Outcome |
|---|---|---|---|
| B0 | Fix wgrad triple, add dgrad shapes, re-sweep GEMM | `gemm` holdout within +-0.5pp of 3.9019%; all 12 fwd/dgrad/wgrad queries resolve for Qwen3-8B tp=2 | done, `9a2dd9d`. Holdout 3.72% over 174 points (62 before; the old table's 43-of-51 junk slices were the artifact of the wgrad bug) |
| B3 | `TrainingOps` + `ComposedTraining`, no DP term | Per-term decomposition and MRE on the 22 THD records, against the fitted model's 2.73% | done, `bc0d64e`. Shipped at 17.46% with attention mispriced; see B1 |
| B2 | `optimizer_step` op + measurement | Holdout MRE reported; floor fitted | done, `60045bf`. Holdout 2.84% over 12 points, floor 12us. Closed 0.23pp: the term is 0.2% of a step, which is the result |
| B1 | `attention_backward` op + measurement | Holdout MRE reported with `point_count > 0`; `kernel_source` names the backend TE selects | done. Both directions measured as the `attention_training` family: forward 5.66%, backward 3.83%, 48 points each, `kernel_source=FlashAttention`. **Closed 9.60pp** |
| B4 | DP collective terms, flagged unvalidated | Constructs and predicts at `dp>1`; `dp=1` numbers unchanged | `dp_comm_queries` raises `NotImplementedError` rather than returning a guess. No `dp>1` THD record exists, so an implemented-but-unscored term would be indistinguishable from a validated one |
| B5 | Docs | Every figure traces to a B0-B4 command | done: §0 of this file, and the pinned figures live in `tests/test_training_composed_accuracy.py` so they cannot drift from the tables silently |

B0 was strictly first, as planned. The rest did not run in order: B3 shipped before B1 and
B2 on the assumption that attention was adequately priced by the rollout table, and B1 then
turned out to carry 9.6 of the remaining 17.2pp. Running B3 early is what surfaced that, so
the inversion was productive rather than a mistake — but the design's phase ordering
implies B1 and B2 are refinements of a working composition, and B1 was a correctness fix.

B0 is strictly first: it is a correctness fix to committed code, and the accuracy numbers in
`tests/test_collective_baseline_regression.py` must be re-pinned from the corrected sweep
before anything is built on top.

## 6. Verification

1. **Unit, no GPU.** `TrainingOps` multipliers: 4 GEMMs per layer forward, dgrad and wgrad
   per projection, 8 all-gather + 6 reduce-scatter per layer under full recompute, 4+4 without.
   A test asserts the per-layer count is **not** multiplied by `num_layers` twice, the same
   check `ModelOps.comm_queries` carries.
2. **Shape convention.** Both backward triples validated against `F.linear`'s output shape
   rather than FLOP arithmetic, since FLOPs cannot distinguish the orderings. This is the
   check that would have caught §1.2(b).
3. **Coverage.** Every query `TrainingOps` emits for Qwen3-8B at tp=2 resolves against the
   re-swept table with no clamping. A test enumerates them and asserts no
   `KernelCoverageError`.
4. **Holdout** for the two new ops, with `point_count > 0`.
5. **Composed vs real, the number that matters.** Score `ComposedTraining.predict` against
   all 22 THD records (`outputs/profile/qwen3_8b/megatron_tp2_pp3_thd/`), reporting MRE
   **and** the per-term decomposition. Compare against the fitted model's 2.73%, noting that
   the fit saw these targets and the composition did not.
6. **Order of magnitude, as an early falsification.** A crude sum already gives 898 ms
   against a measured 1287 ms at 4096 tokens, so the composition starts ~30% low with
   collectives, elementwise and optimizer unmodelled. If the complete sum lands *above*
   measured, a multiplier is doubled somewhere.
7. **No regression.** The rollout path must be untouched: `tests/test_collective_baseline_regression.py`
   and the composed-rollout tests pass with values re-pinned only where B0's re-sweep moved them.

All GPU work requires `source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh`.

## 7. Deliberately out of scope

- **Compute/communication overlap.** Megatron's `overlap_grad_reduce` and
  `overlap_param_gather` make this a real question, and the profiled runs may have them on.
  The sum is a plain sum; if the composition over-predicts, overlap is the first suspect and
  the per-term decomposition is what would show it.
- **DP end-to-end validation.** No `dp>1` THD records exist. B4 implements the term and says
  it is unvalidated rather than implying otherwise.
- **FSDP.** The shared `TrainingOps` is written so FSDP can reuse it, but only Megatron is
  validated here.
- **Pipeline schedules other than 1F1B**, context parallelism, and MoE: all already blocked
  upstream.
- **The psrl end-to-end comparison.** A recorded finding is that verl bills ~34 s of
  optimizer offload to `update_actor` that the profiler structurally cannot see. B validates
  against the profiler's own step times, where that cost is absent. Conflating the two would
  make a composition error and an accounting difference indistinguishable.

## 8. Risks

| risk | mitigation |
|---|---|
| A multiplier is doubled (layers x recompute x fwd/bwd) | Declarative op list with explicit multipliers; unit tests on counts; verification 6 catches over-prediction |
| Measuring flash instead of cuDNN attention | B4 pins the backend and tags `kernel_source`; TE's own selector is the authority, queried not assumed |
| The re-swept GEMM table moves the validated rollout numbers | B0 gates on `gemm` holdout within +-0.5pp and re-runs the rollout end-to-end score |
| Backward collective counts wrong (the 4 vs 6 asymmetry) | Counts derived from TE source with file:line, and asserted in unit tests |
| Elementwise double-counts, as it would for rollout | Different hardware path: no CUDA graphs in Megatron. If the composition over-predicts, the decomposition isolates the elementwise share directly |
| DP term silently treated as validated | B8 flags it; docs state the distinction, as Spec A's `tp>1` note does |
