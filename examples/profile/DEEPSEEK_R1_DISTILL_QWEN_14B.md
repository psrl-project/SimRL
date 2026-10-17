# DeepSeek-R1-Distill-Qwen-14B profiling

The Qwen3-8B workflows (`kernel/`, `vllm/`, `megatron/`) applied to
`/apdcephfs_zwfy/share_303760348/lhy/models/DeepSeek-R1-Distill-Qwen-14B`. Read
[`README.md`](README.md) and [`kernel/README.md`](kernel/README.md) first: everything there
about why each sweep is shaped the way it is still applies. This file records only what is
different about this checkpoint, and what measuring it revealed.

```bash
export MODEL_PATH=/apdcephfs_zwfy/share_303760348/lhy/models/DeepSeek-R1-Distill-Qwen-14B
```

## The architecture, and what it changes

| | Qwen3-8B | DeepSeek-R1-Distill-Qwen-14B |
|---|---|---|
| architecture | `Qwen3ForCausalLM` | `Qwen2ForCausalLM` |
| parameters | 8.19B | 14.77B |
| layers | 36 | 48 |
| hidden / intermediate | 4096 / 12288 | 5120 / 13824 |
| query / KV heads | 32 / 8 (ratio 4) | 40 / 8 (**ratio 5**) |
| head_dim | 128 | 128 |
| `q_norm`/`k_norm` | yes | **no** |
| KV per token (bf16) | 0.1406 MiB | 0.1875 MiB |
| bf16 weights | 15.3 GiB | 27.5 GiB |

Both are dense gated pre-norm blocks with GQA, so `families_dense.py` already registers
`Qwen2ForCausalLM` and no new kernel-shape family was needed. `kernels verify-shapes`
confirms all five declared GEMM shapes against the real module graph:

```
Qwen2ForCausalLM -> family 'dense_llama', 48 layers.
  qkv_proj (7168, 5120)  o_proj (5120, 5120)  gate_up_proj (27648, 5120)
  down_proj (5120, 13824)  lm_head (152064, 5120)        all OK
```

Two consequences are worth knowing before reading any number below.

**The GQA ratio is 5, and FlashInfer cannot decode it.** FlashInfer's paged decode kernel
dispatches the ratio as a template parameter and compiles specializations only for
`{1, 2, 3, 4, 8}` (`DISPATCH_GQA_GROUP_SIZE` in `flashinfer/data/include/flashinfer/utils.cuh`).
At 40/8 this model lands on 5 and the generated kernel raises `Unsupported group_size: 5`.
Tensor parallelism does not rescue it: `rank_head_counts` shards query and KV heads by the
same degree, so 40/8, 20/4, 10/2 and 5/1 are all ratio 5. This is a property of the
architecture, not of a topology choice.

`profile_flashinfer_attention` now detects this, warns, and skips the generation phase
rather than letting it abort the sweep after the context phase has already been measured.
No FlashInfer `generation_attention` table exists for this model, and a query against one
raises a coverage error — which is the honest outcome. It costs the cost model nothing:
**vLLM's own selector picks `FLASH_ATTN` on this H20 regardless**, so the `vllm` rows are
the kernels a real server runs, and those are what `AttentionQuery` reads by default. The
FlashInfer context table is still measured and is still only a "what would a backend switch
buy" diagnostic.

**No `q_norm`/`k_norm`.** `_has_qk_layernorm` returns False for a Qwen2 config, so the
elementwise grid omits the two narrow per-head RMSNorm kernels Qwen3 pays for. Nothing to
configure; it falls out of the checkpoint.

## Kernel profiling

```bash
TP=1 bash examples/profile/kernel/profile_deepseek_r1_distill_qwen_14b.sh all
TP=2 bash examples/profile/kernel/profile_deepseek_r1_distill_qwen_14b.sh all
```

Configs: `kernel/deepseek_r1_distill_qwen_14b_h20_tp{1,2}.yaml`. TP1 carries the rollout
families; TP2 adds `attention_training` and `optimizer`, because tp=2 is the degree the
training sweep runs at. One `collective` sweep covers world sizes 2, 4 and 8 from its own
`topologies` block, so there are no tp=4/tp=8 configs.

Grid budgets transfer unchanged from Qwen3-8B: the KV geometry that sets them (8 KV heads at
head_dim 128) is identical, so `max_context_tokens` and `max_generation_kv_tokens` need no
adjustment despite the larger model.

Three grids are model-specific and must stay in sync with the training sweep:

- `attention_training.num_heads: [5, 10, 20, 40]` — the per-rank query head counts this
  model's 40 heads actually produce at tp=8/4/2/1. Not the powers of two a 32-head model
  gives; listing 32 or 16 would write rows keyed on a head count no query can reach.
- `optimizer.param_counts` includes `1651673088` and `2040956928`, the two shards the
  tp=2/pp=4 mesh really produces. A pipeline does not split parameters evenly — the first
  and last stages also carry the embedding and LM head — so dividing 14.77B by 8 gives
  1.846B, a count **no rank has**. Both values were read from
  `rank_memory[].local_parameter_count` in a profiled run.
- `gemm.token_counts` (tp2) lists the non-power-of-two packed totals the THD grid reaches,
  because a wgrad query carries `k = token_count` and an unmeasured value raises.

### Holdout gate, TP1, `--fraction 1.0`

Reported at full fraction deliberately; see `kernel/README.md` on why `--fraction 0.2`
flatters the surface. The `vllm` rows are the ones the cost model reads.

| op | backend | points | MRE | p90 | Qwen3-8B MRE |
|---|---|---|---|---|---|
| `gemm` | vllm | 104 | **3.21%** | 8.94% | 3.91% |
| `all_gather` | vllm | 96 | 3.42% | 8.31% | — |
| `broadcast` | vllm | 96 | 4.25% | 9.77% | — |
| `reduce_scatter` | vllm | 96 | 4.57% | 13.58% | — |
| `all_reduce` | vllm | 96 | 5.20% | 12.52% | 6.01% (tp2) |
| `elementwise` | vllm | 65 | 5.29% | 16.40% | 5.01% |
| `context_attention` | vllm | 59 | 6.15% | 19.48% | 5.74% |
| `generation_attention` | vllm | 72 | 8.73% | 22.88% | 6.63% |
| `context_attention` | flashinfer | 59 | 8.94% | 26.33% | 9.87% |
| `generation_attention` | flashinfer | — | n/a (group_size 5) | — | 6.01% |

The surface tracks the Qwen3-8B baseline op for op; GEMM is better. Every dispatched
collective is inside or near the 5% gate. The two attention rows above it are the same ones
`kernel/README.md` already documents as above target there, with the same cause (grid
density at small shapes against a dispatch floor), so this is not a new regression. The
`nccl` rows are kept as a control and are worse (`all_reduce` 12.91%), which is expected and
harmless: vLLM's dispatcher never selects raw NCCL in the message-size range a decode step
lives in.

### Holdout gate, TP2, `--fraction 1.0`

TP2 adds the three families the training cost model queries. All three are inside the gate,
which matters more than the rollout rows here, since TP2 is the degree the Megatron sweep
runs at.

| op | backend | points | MRE | p90 | utility |
|---|---|---|---|---|---|
| `optimizer_step` | megatron | 14 | **2.30%** | 7.80% | 0.431-0.744 |
| `training_attention_backward` | megatron | 48 | **3.37%** | 9.10% | 0.320-0.613 |
| `training_attention_forward` | megatron | 48 | **4.57%** | 12.76% | 0.228-0.614 |
| `generation_attention` | vllm | 72 | 3.85% | 11.40% | 0.212-0.389 |
| `gemm` | vllm | 174 | 3.95% | 11.41% | 0.388-0.963 |
| `broadcast` | vllm | 96 | 4.36% | 8.74% | 0.076-0.817 |
| `all_reduce` | vllm | 96 | 4.69% | 12.58% | 0.140-0.781 |
| `reduce_scatter` | vllm | 96 | 5.02% | 15.33% | 0.237-0.751 |
| `elementwise` | vllm | 65 | 5.13% | 16.45% | 0.189-1.038 |
| `context_attention` | vllm | 59 | 6.63% | 18.62% | 0.162-0.936 |
| `all_gather` | vllm | 96 | 7.81% | 18.41% | 0.027-0.305 |
| `context_attention` | flashinfer | 59 | 9.68% | 29.12% | 0.219-1.119 |

`all_gather` is the one dispatched collective meaningfully above the gate at this degree
(7.81% against 3.42% at TP1) and its utility range explains why: 0.027-0.305, an op that
never approaches the roofline, so the surface is interpolating over a regime the SOL model
describes poorly. It is also the collective a TP=2 forward pass uses least.

`generation_attention` improves markedly from TP1 (3.85% against 8.73%): halving the per-rank
head count moves the decode kernel off the dispatch floor where TP1's small shapes sit.

## Megatron-Bridge training

```bash
srl-profile megatron generate --config examples/profile/megatron/deepseek_r1_distill_qwen_14b_thd.yaml \
    --candidates outputs/profile/deepseek_r1_distill_qwen_14b/megatron_tp2_pp4_thd/control/megatron_bridge/candidates.json
srl-profile megatron run     --config ... --candidates ...
srl-profile megatron collect --config ... --candidates ...
```

`megatron/deepseek_r1_distill_qwen_14b_thd.yaml`, 24 candidates. THD packing rather than
BSHD, for the reason `qwen3_8b_thd.yaml` sets out: under BSHD `sum_squared_tokens` is
identically `sum_tokens^2`, the linear and quadratic token columns are collinear, and a
BSHD-fitted coefficient overpredicted real packed steps by 57-60% on Qwen3-8B.

**TP2/PP4 is the analog of Qwen3-8B's TP2/PP3**, chosen to hold per-stage depth fixed: 48
layers over 4 stages is the same 12 layers per stage that 36 over 3 gives, at the same
tensor-parallel degree. It also fills the node exactly (2 × 4 × 1 = 8) where TP2/PP3 left
two GPUs idle, and keeps the per-rank shard inside the 2.048B ceiling the `optimizer` grid
measures to. Measured peak is 52 GiB of a 95 GiB card.

### The recipe table

`infer_bridge_recipe` previously knew only Qwen3 and computed its recipe name as
`f"qwen3_{round(params / 1e9)}b"`. That arithmetic names recipes that do not exist:
Megatron-Bridge spells sub-billion sizes `600m` and `1p7b`, and it ships `qwen25_14b` but no
`qwen2_14b`, so a 14.77B Qwen2 checkpoint would have resolved to a `qwen2_15b` nobody wrote
— failing inside the torchrun child, after the candidate had already paid for distributed
setup. It is now an explicit table keyed on `model_type` with an 8% size tolerance, and
`model_family_name` / `model_recipe_name` in the YAML override it (both together, or
neither). This checkpoint resolves to `qwen25_14b`; Qwen3-8B still resolves to `qwen3_8b`.

Qwen2 and Qwen2.5 share one HuggingFace `model_type` and nothing in `config.json` separates
them, so size is the only discriminator. That is sufficient here: the two generations'
recipes differ only in `ddp.check_for_nan_in_grad` and in the hub id whose architecture
`_local_checkpoint` replaces with the profiled checkpoint anyway.

### Identifiability is working

The grid exists to separate the linear token column from the quadratic one. Two candidates
at `nmb=1` with the same 12288-token total but different `sum(s_i^2)`:

| packed microbatch | sum tokens | sum s² | step |
|---|---|---|---|
| `[3072, 3072, 3072, 3072]` | 12288 | 3.8e7 | 6.743 s |
| `[12288]` | 12288 | 1.5e8 | 7.462 s |

Same tokens, 4x the attention work, 10.7% more step time. A fit with only one of these
shapes could not tell those apart.

### The fitted artifact

```bash
srl-regress training --latency-model hydraulis --memory-model hydraulis \
    --input  outputs/profile/deepseek_r1_distill_qwen_14b/megatron_tp2_pp4_thd/profiles/megatron_bridge.jsonl \
    --output outputs/cost/deepseek_r1_distill_qwen_14b/training_hydraulis.json
```

All 24 candidates succeeded; none OOM'd, so `allow_incomplete: true` was not exercised.

| | 14B TP2/PP4 | Qwen3-8B TP2/PP3 |
|---|---|---|
| `token_cost` | 5.227e-04 | 2.909e-04 |
| `token_quadratic_cost` | 1.142e-08 | 5.723e-09 |
| design rank | 4 (full) | 4 (full) |
| train MRE | 3.02% (n=19) | 2.73% (n=17) |
| holdout MRE | 5.57% (n=5) | 1.37% (n=4) |
| holdout r² | 0.9949 | 0.9999 |
| `activation_scale` | 2.205 | 0.632 |
| domain: tokens | 4096-22528 | 4096-22528 |
| domain: pipeline slots | 4-67 | 3-66 |

The design reaches full rank, which is what the three microbatch counts and the
equal-total/different-split shapes were for, and the quadratic column survived with a
non-zero coefficient rather than collapsing into the linear one.

Both coefficients are larger than the 8B fit's, in the right direction but by more than
FLOP counting alone predicts. PP4 against PP3 puts the same 12 layers on a stage in both
cases, so the per-stage work ratio comes from width only: 1.43x for the linear projections
and 1.25x for attention. Measured, `token_cost` is 1.80x and the quadratic 2.00x. The gap is
unexplained here and worth knowing before these coefficients are read as physical constants
— plausible causes are lower kernel efficiency at this model's shapes and the full-recompute
forward being charged differently, but neither was isolated. What the numbers do support is
the fit's conditioning and its in-domain accuracy, not a claim that the coefficients are
transferable across architectures.

Holdout MRE is higher than the 8B figure, on five points spanning a 118-fold step-time range
(2.2 s to 263 s). Read it as "within a few percent" rather than as a stable number.

**`feature_ranges` are hard gates, not hints.** A query outside tokens 4096-22528 or slots
4-67 raises `TrainingOutOfDomainError` rather than extrapolating, and `tokens` is the
*largest* microbatch, so one long sequence in a batch sets it. Extend the sweep rather than
loosening the gate.

## vLLM rollout

```bash
TP=1 bash examples/profile/vllm/profile_deepseek_r1_distill_qwen_14b.sh all
TP=2 bash examples/profile/vllm/profile_deepseek_r1_distill_qwen_14b.sh all
```

Configs: `vllm/deepseek_r1_distill_qwen_14b_tp{1,2}.yaml`, with grids byte-identical to each
other and to the Qwen3-8B pair, so per-phase MRE stays comparable across checkpoint and
topology.

**The tp=1 decode tail will truncate, and that is expected.** 27.5 GiB of weights plus
0.1875 MiB per KV token leaves about 309k tokens at `gpu-memory-utilization: 0.9`, where
`max-concurrency: 64` at 1024 + 8192 wants 590k. Collection truncates such a point at its
first preemption and warns with the run name and dropped step count; the retained prefix
stays valid because every request in it still shares one age, which is what `ctxsum`
assumes. Treat the concurrency-64 decode row as covering fewer context values than the
concurrency-16 row rather than as an equal sample. Qwen3-8B reaches 501k on the same grid,
so it crosses the same boundary later — a difference of degree, not a new failure mode.

At tp=2 the pooled cache reaches about 767k tokens and no truncation is expected. A dropped
-step warning there is a finding about that topology, not the known tp=1 limit.

### What the sweeps produced

| | TP1 | TP2 |
|---|---|---|
| iteration records | 49202 | 49233 |
| prefill / decode / mixed | 54 / 49143 / 5 | 27 / 204 / 6 buckets |
| preemption truncations | 3 (concurrency 32, 48, 64) | **0** |

The capacity arithmetic held: at tp=1 the three largest decode points each kept 8192 steps
and dropped 560, 3457 and 4899, and at tp=2 nothing truncated at all.

### End-to-end validation: `kernels validate`

The holdout gate asks whether single-kernel interpolation is self-consistent. This asks the
harder question — whether summing those kernels reproduces a real vLLM engine iteration.
Run with `--min-samples 1` so prefill and mixed are scored rather than dropped, and against
the `vllm` rows, which are the kernels the server actually ran.

| phase | TP1 buckets | TP1 MRE | TP2 buckets | TP2 MRE | Qwen3-8B TP1 |
|---|---|---|---|---|---|
| prefill | 25 | **2.82%** | 27 | 6.32% | 3.72% |
| decode | 215 | **5.60%** | 204 | 7.38% | 4.84% |
| mixed | 5 | **1.24%** | 6 | 3.78% | 0.48% |

TP1 lands in the same band as the Qwen3-8B reference, prefill better and decode marginally
worse. Median ratios are 0.99-1.02 at TP1, so the composed model is not biased in either
direction — the error is spread, not a systematic offset.

TP2 is 1.2-3x worse per phase, with median ratios drifting to 0.96 on prefill and mixed: a
slight *under*-prediction. The likely cause is that TP2 adds a per-layer `all_reduce` to
every step, so the composed sum now depends on the collective table, and `all_gather` /
`all_reduce` are the ops whose holdout error is worst at this degree (7.81% / 4.69%) because
their utility never exceeds 0.31. The mixed row rests on six buckets either way and should
not be read as a validation on its own.

### Regressed rollout artifacts, and which one to use

Three families are fitted from the same `vllm.jsonl`, at both degrees:

```bash
srl-regress rollout --model psrl \
    --input  outputs/profile/deepseek_r1_distill_qwen_14b/vllm_tp1/profiles/vllm.jsonl \
    --output outputs/cost/deepseek_r1_distill_qwen_14b/rollout_psrl.json
# distserve additionally needs --hidden-size 5120 --intermediate-size 13824
```

Replaying every one of the ~49k measured iterations back through `predict()` — not the
fitter's own self-report — gives per-phase MRE:

| model | prefill | decode | mixed |
|---|---|---|---|
| **psrl TP1** | **1.41%** | **3.18%** | 0.56% |
| **psrl TP2** | 9.48% | **4.66%** | 1.52% |
| roofline TP1 | 3.00% | 6.31% | 0.38% |
| roofline TP2 | **49.41%** | 2.39% | 21.81% |

**Use `psrl`.** It is what `examples/simulate/simulate.sh` reads by default (`${COST_DIR}/rollout_psrl.json`,
hence the unsuffixed copy fitted at TP1), and it is the only family healthy at both degrees.
Qwen3-8B's psrl baseline is 5.16% overall, so TP1 here is better and TP2 comparable.

**`roofline` and `distserve` are unusable at TP2** and the artifacts are kept only as a
control. Their reported `r_squared` is 0.28 with a *lower* MRE of 2.45%, which looks like a
contradiction and is the tell: a handful of very expensive steps dominate the sum of squares
while the bulk of cheap decode steps dominate the mean relative error. The ~50% prefill
error is the mechanism — `A_p` fits to 3.58e-07 against the 8B's 3.40e-08, a factor of 10.5
— and it bites on the 65536-token chunked-prefill steps, which really do take 7.5 s at TP2
(verified against the raw records; the model under-predicts them at 3.8 s rather than
inventing a large number). `distserve` shares roofline's prefill term and so shares the
failure, byte for byte.

## Where the results land

Mirroring `outputs/profile/qwen3_8b/` and `outputs/cost/qwen3_8b/`.

```
outputs/profile/deepseek_r1_distill_qwen_14b/
├── kernel_tp1/
│   ├── kernel/h20/{gemm,attention,elementwise,collective}/<backend>/<version>/*_perf.parquet
│   ├── kernel/h20/.../floors.json          # fitted dispatch floors, per backend+version
│   ├── kernel/holdout_report.json
│   └── collective_runs/
├── kernel_tp2/
│   └── kernel/h20/{...,attention_training,optimizer}/...   # plus the two training families
├── megatron_tp2_pp4_thd/
│   ├── profiles/megatron_bridge.jsonl      # 24 records, the regression input
│   ├── raw/megatron_bridge/<candidate_id>/ # per-candidate metrics, ConfigContainer, run.log
│   └── control/megatron_bridge/candidates.json
└── vllm_tp{1,2}/
    ├── profiles/vllm.jsonl                 # ~49k iteration records each
    ├── raw/vllm_timing/                    # untouched official benchmark JSON
    └── control/vllm/                       # scheduler logs, serve/bench params

outputs/cost/deepseek_r1_distill_qwen_14b/
├── training_hydraulis.json                 # the fitted latency + memory artifact
├── validate_tp1.json                       # composed-vs-vLLM report
└── validate_tp2.json
```

Under `attention/flashinfer/0.6.14/` both TP directories hold `context_attention_perf.parquet`
and **no** `generation_attention_perf.parquet`. That absence is the group-size skip described
above, not a failed or interrupted run.

## Measure on an idle node, and check that it is idle

The first local attempt at the TP2 sweep OOM'd, and the cause was another session profiling
Qwen3.5-35B-A3B on GPUs 0-3 of the same host. The TP1 collective tables and the TP2 GEMM
table from that window were **discarded and re-measured**, because a kernel timing taken
beside someone else's model is contention rather than hardware, and nothing in the output
would have said so — the tables would simply have been slow.

`nvidia-smi` alone is not enough when a sweep is between points and the GPU reads idle.
Check for live compute processes:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

The kernel and training sweeps also must not overlap with each other on one node, for the
same reason.
