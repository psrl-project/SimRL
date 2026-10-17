# Kernel-Level Cost Model Design

Date: 2026-08-24
Status: Approved design, pending implementation plan

## 1. Context

SimRL predicts rollout and training latency from cost models fitted to profiling data.
Today two end-to-end paths exist and work: `srl-profile vllm` drives a real vLLM server and
`srl-profile megatron` drives Megatron-Bridge; `srl-regress` fits `UnifiedRoofline` /
`PSRLFitted` / `DistServe` / `HydraulisTrainingCost` from those records.

A third path was added recently under `srl/profiler/kernels/`, intended to predict latency by
profiling individual kernels instead of a whole server. It does not work as a cost model, for
three independent reasons.

### 1.1 The current kernel path destroys its own measurements

The flow is `kernels run` → raw JSON → `kernels synthesize` → `kernel.jsonl` → `srl-regress`.
`compose.py:fit_composed_terms` collapses the entire attention sweep into four scalars
(`prefill_attn_b/k`, `decode_attn_b/k`) by ordinary least squares, then
`synthesize.py:synthesize_rollout_records` evaluates those scalars on a denser grid to emit
synthetic `ProfileRecord`s, and `srl-regress rollout` fits coefficients *again* from that
synthetic data.

The second regression is close to an identity transform: the data is noise-free and was generated
by the very linear model being fitted. `synthesize` exists only to bridge into `srl-regress`'s
input format. Meanwhile the first regression is lossy in a way that cannot be recovered: tile
quantization, FlashAttention's 128-alignment staircase, and the latency-bound regime at low token
counts are all flattened into a straight line before any consumer sees them.

`synthesize.py` and `compose.py` are therefore both deleted by this design, not refactored.

### 1.2 The attention measurement does not match vLLM

`attention.py:90` calls `torch.nn.functional.scaled_dot_product_attention` on dense
`[B, H, kv_len, D]` tensors. vLLM serves decode from a **paged** KV cache reached through a block
table, which is a different memory-access pattern, not a different kernel. The file's own docstring
concedes it is "a safe lower bound". `linear.py` has the same class of problem: it times
`torch.nn.Linear` calls one at a time and never shards shapes by TP, while vLLM runs fused QKV and
fused gate-up projections on per-rank shapes.

### 1.3 The CPU gap is a hardcoded constant

`overhead.py` returns a fixed `scheduling_overhead_s = 0.0005` to stand in for everything CUDA-event
timing cannot see: the Python scheduler loop, sampler, input preparation, CUDA graph replay, and
kernel launch gaps.

### 1.4 Intended outcome

Keep both end-to-end paths exactly as they are. Add a kernel-based path that is a genuine
alternative source of cost models, able to predict rollout *and* training latency, and report a
credible prediction error against real hardware. First phase fixes hardware (H20), model
(Qwen3-8B), and parallelism (TP ∈ {1, 2}).

## 2. Reference implementations

Two mature simulators were studied at `/apdcephfs_zwfy/share_303760348/lhy/simulator/`. Both made
the same choice on the decision that matters most, and it is the opposite of the current code's.

**Neither stores fitted coefficients. Both keep raw `(shape → latency)` rows and interpolate at
query time.**

- aiconfigurator: `systems/data/{gpu}/{family}/{backend}/{version}/{op}_perf.parquet`.
  Verified schemas: GEMM is 107,225 rows of `(m, n, k, gemm_dtype) → latency` plus a
  `kernel_source` column recording the actual kernel
  (`torch.nn.functional.linear` vs `CutlassFP8ScaledMMLinearKernel`); `context_attention` is 50,830
  rows keyed by `(batch_size, isl, num_heads, num_key_value_heads, head_dim, attn_dtype,
  kv_cache_dtype)` with `kernel_source` = `vllm_flash_attn_fa3`.
- Frontier: wide CSVs at `data/profiling/compute/{device}/{model}/{attention,mlp}.csv`, columns
  `time_stats.{op}.{min,max,mean,median,std}` plus identity columns
  (`n_embd`, `n_q_head`, `n_kv_head`, `block_size`, `num_tensor_parallel_workers`, `max_model_len`,
  `batch_size`, `prefill_chunk_size`, `kv_cache_size`, `is_prefill`, `attention_backend`).

Other findings that shape this design:

| Aspect | aiconfigurator | Frontier |
|---|---|---|
| Query | log2-space kNN + Modified-Shepard weights, **SOL-normalized utility** | sklearn RandomForest per (device, model, TP) |
| Past measured range | `latency = SOL(query) / weighted_avg_utility` | RF extrapolation (unreliable) |
| Cross-hardware | Yes, via SOL using target GPU's `tc_flops` / `mem_bw` | **No**, retrain per device |
| Attention backend | Asks vLLM's own selector (`current_platform.get_attn_backend_cls`), profiles whatever it picked, records it in `kernel_source` | `flashinfer.BatchPrefillWithPagedKVCacheWrapper`, but imports vLLM's `get_kv_cache_layout()` and `reshape_and_cache_flash` for layout parity |
| Op granularity | Unified `gemm(m,n,k)` table; `context_attention` / `generation_attention` | Per-semantic-op tables (`attn_pre_proj`, `mlp_up_proj`, …) |
| CPU overhead | Not modeled | **Measured table** keyed by `(model, batch_size, tp, num_prefill_tokens, num_decode_tokens, scheduling_mode)` |
| Calibration | No global fudge; `_correct_sol` raises measurements below theory | **Per-op `calibration_scale`** (e.g. `_attn_pre_proj_calibration_scale`) |
| Compute/comm overlap | `operations/overlap.py` | Not modeled, plain sum |
| Training (fwd/bwd) | **No** | **No** |

TP sharding happens at op-construction time, not lookup time. From
`aic-core/src/aiconfigurator_core/sdk/models/gpt.py`:

```python
ops.GEMM("context_qkv_gemm", self._num_layers,
         self._num_heads * self._head_size // tp_size + self._head_size * num_kv_heads_per_gpu * 2,
         h, gemm_quant_mode, seq_split=cp),
ops.ContextAttention("context_attention", self._num_layers,
                     self._num_heads // tp_size, num_kv_heads_per_gpu, ...),
ops.CustomAllReduce("context_ar_1", self._num_layers, h, tp_size),
ops.P2P("context_p2p", pp_size - 1, h, pp_size),
```

Latency is `sum(op.query(db) * op.scale_factor for op in ops)`, where `scale_factor` is the layer
count. This declarative op list (not a coefficient converter) is the "kernel level → vLLM/Megatron
level" translation layer.

**Neither reference models training.** The Megatron side of this design has no prior art to copy.

## 3. Design decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Store raw `(shape → latency)` in parquet; interpolate at query time | Preserves non-linearity both references keep; ends the fit→synthesize→fit loop |
| D2 | Interpolate **SOL-normalized utility**, not latency | Only mechanism with theoretical support for crossing hardware; yields readable diagnostics |
| D3 | Ask vLLM's selector for the backend; also profile FlashInfer; tag rows with `kernel_source` | Auto-tracks vLLM backend changes and answers "what if we switched backends" |
| D4 | One unified GEMM table `(m, n, k, dtype)` | Lets training dgrad/wgrad reuse the forward table at zero extra measurement cost |
| D5 | `decode_coeffs` returns a local tangent | Preserves the simulator's O(1) closed-form decode advance without touching `advance.py` |
| D6 | Parquet for measurement tables, coexisting with `CostArtifact` JSON for coefficients | Different semantics; JSON is wrong for tens of thousands of rows |
| D7 | Delete the hardcoded `scheduling_overhead_s`; measure CPU overhead, fit the residual | A constant cannot represent a workload-dependent gap |
| D8 | Report prefill / decode / mixed error **separately**; never a pooled number | See §7: the existing baseline is 99.9% decode, so a pooled metric hides prefill entirely |

### D3 in detail

Two rows are measured per attention shape and distinguished by `kernel_source`:

1. Whatever vLLM's own selector picks: resolved via `current_platform.get_attn_backend_cls(...)`,
   with the KV cache built per `backend_cls.get_required_kv_cache_layout()` and run through
   `impl_cls`, as aiconfigurator's `collector/vllm/collect_attn.py` does. Tagged with the concrete
   backend and version (e.g. `vllm_flash_attn_fa3`).
2. FlashInfer's paged wrappers (`BatchPrefillWithPagedKVCacheWrapper`,
   `BatchDecodeWithPagedKVCacheWrapper`), tagged `flashinfer_<version>`.

**The query path defaults to row 1**: the model must predict the backend vLLM actually runs.
Row 2 is selected only by an explicit `attention_backend` override in the composed cost config, so
that "how much would switching backends buy" is answerable without changing the default prediction.

Both paths write paged KV cache with a block table; neither uses dense tensors. This is what fixes
§1.2.

### D2 in detail

Parquet stores measured latency. On load, `utility = SOL(shape, gpu_spec) / latency_measured`
(a dimensionless efficiency in roughly (0, 1]). Interpolation happens in utility space; queries
return `latency = SOL(query, target_gpu) / interp(utility)`. Past the measured frontier, hold the
boundary utility rather than extrapolating latency.

This matters concretely here: the target GPU is **H20**, whose BF16 compute is modest while HBM3
bandwidth is very high: a flops/bandwidth ratio unlike H100's. Transferring latency across such
hardware is meaningless; transferring utility is defensible.

**Limitation resolved (Fix B, 2026-08-24): D2 degrades in the dispatch-limited regime; floor in `max` fixes it.**

The Phase 1a gate confirmed D2 is a large win for compute- and bandwidth-limited kernels: GEMM
8.48% via utility vs 31.64% via raw latency (3.7x improvement); context attention 20.85% vs
43.39%. However, D2 *loses* for generation attention: 34.89% via utility vs 18.66% via raw
latency. The reason is structural. D2 silently assumes SOL approaching zero implies latency
approaching zero. When a kernel is dispatch-limited, SOL goes to zero while measured latency
plateaus at a constant per-call overhead (~70 us for vLLM, ~10 us for FlashInfer). Utility
collapses toward zero too: at batch 1, step 0, utility is 0.000075 (SOL 0.005 us against
68.6 us measured). A multiplicative normalizer cannot represent an additive floor, so the
interpolation surface becomes extremely steep in exactly the regime where Phase 1b needs it
most: decode attention is 99.9% of the existing vLLM baseline.

The decisive evidence is a flat latency table: batch=1 measures ~70 us at step 128 and still
~70 us at step 8192 (64x more KV work). That is `max(floor, work)`, not `floor + work`. Adding
a constant before deriving utility barely helps; placing the floor inside the max restores
utility to order 1 in the overhead-limited regime without changing the cross-hardware-transfer
property where D2 is sound.

Fix B (2026-08-24) adds `floor_s` as a third field in `SolComponents` and makes
`latency_s = max(math_s, memory_s, floor_s)`. Per-backend, per-phase floors fitted from real
H20 measurements are stored in `ATTENTION_FLOOR_S` in `sol.py`:

| backend | context | generation |
|---|---|---|
| vLLM | 32 us | 22 us |
| FlashInfer | 4 us | 6 us |

Post-fix gate on the existing H20 parquet at `--fraction 1.0`:

| family | backend | as-shipped MRE | with max-floor MRE |
|---|---|---|---|
| `context_attention` | vllm | 20.85% | 5.74% |
| `context_attention` | flashinfer | 11.39% | 9.87% |
| `generation_attention` | vllm | 34.89% | 6.63% |
| `generation_attention` | flashinfer | 20.08% | 6.01% |

`context_attention/flashinfer` at 9.87% remains above the 5% gate; the 4 us floor is correct
for the dispatch mechanism but insufficient to collapse residual variance at very small shapes.
Densifying the grid in the overhead-limited regime is the next lever. GEMM is unchanged at 4.16%.

### D5 in detail

`instance.py:354` and `instance.py:460` take a closed-form fast path. `advance.py` requires step
cost to be exactly `tau(k) = max(F, alpha + beta * k)` so that a k-step decode run collapses to an
arithmetic-series sum instead of k cost-model calls. `PSRLFitted.decode_coeffs` can supply this
because decode latency is exactly linear in `ctxsum`, which grows by `n_d` per step.

A lookup table has no such closed form. Resolution:

```python
def decode_coeffs(self, n_d, ctxsum):
    t0 = self._decode_step(n_d, ctxsum)
    t1 = self._decode_step(n_d, ctxsum + n_d * _TANGENT_STRIDE)
    return 0.0, t0, (t1 - t0) / _TANGENT_STRIDE
```

Decode latency is near-linear in `ctxsum` (KV bytes read scale linearly), so tangent error stays
small. `advance.py` and `instance.py` are untouched.

### D4 in detail: the training-side lever

Backward dgrad and wgrad are both GEMMs, differing only in how `(m, n, k)` is permuted. With a
unified GEMM table, training reuses forward measurements for free:

| Training component | Source |
|---|---|
| Forward GEMM | GEMM table |
| Backward dgrad / wgrad | **Same GEMM table, permuted `(m, n, k)`**: zero new measurement |
| Attention forward | `context_attention` table |
| Attention backward | New measurement (FlashAttention backward) |
| Recompute | Re-evaluate the recomputed layers' forward op list |
| TP all-reduce | comm table (2 per layer forward + 2 per layer backward) |
| Gradient reduce-scatter / parameter all-gather | comm table |
| Optimizer step | New measurement (Adam, bandwidth-bound) |
| PP bubble | **Analytical**: `(pp - 1)` microbatch slots, not measured |

Only three new measurement families are needed for the whole training side.

**Amendment (Phase 1a): per-(n,k) slicing is required and wgrad shapes must be measured explicitly.**

Phase 1a holdout analysis found that interpolation accuracy with the original `(gemm_dtype,)` slicing
was 8.48% MRE (above the 5% gate) because a small-m query's nearest neighbour in log2 space was a
point from a *different* projection that happened to be geometrically closer across the shared grid.
The fix is to move `n` and `k` from `axis_columns` into `slice_columns`, so each `(dtype, n, k)` pair
interpolates 1-D in `m`. This reduces GEMM MRE to 4.08% and makes the 5% gate pass.

Per-(n,k) slicing changes the backward-reuse story. Dgrad permutes to `(n, k)` pairs that exist in
the forward slices and therefore still requires no new measurement. However, wgrad's `n_wgrad` equals
the token count, which is not one of the measured `(n, k)` projection pairs: a wgrad lookup would
land in an unmeasured slice after slicing. Wgrad shapes are therefore measured explicitly alongside the
forward shapes in `gemm_all_shapes`, adding 62 new measurement points for Qwen3-8B / TP=1 (64 total
wgrad shapes minus 2 that coincide with forward shapes).

## 4. Architecture

```
srl/profiler/kernels/          measure only → parquet
├── attention.py       context / generation; vLLM-selected backend and FlashInfer
├── attention_bwd.py   FlashAttention backward                        [training]
├── gemm.py            unified (m, n, k, dtype) table
├── elementwise.py     rope / rmsnorm / activation / add / kv_cache_save
├── collective.py      allreduce / allgather / reduce_scatter / p2p
├── optimizer.py       Adam step                                      [training]
└── overhead.py        CPU: schedule / sampler / prepare_inputs        [rollout]
     → outputs/kernel/{gpu}/{family}/{backend}/{version}/{op}_perf.parquet

srl/cost/kernel/               lookup + SOL (new)
├── hardware.py   GPU specs: per-dtype tc_flops, mem_bw, link_bw
├── sol.py        analytical SOL per op family
├── table.py      KernelTable: load parquet, derive utility
├── interp.py     log2-space kNN + Shepard weights; utility hold past frontier
├── database.py   KernelDatabase.query_{gemm,attention,comm,elementwise}
└── model_ops.py  declarative op list + TP sharding rules

srl/cost/rollout/composed.py   ComposedRollout(RolloutCostModel)
srl/cost/training/composed.py  ComposedTraining(TrainingCostModel)
```

`roofline.py`, `psrl.py`, `distserve.py`, `hydraulis.py` are **not modified**. The composed models
are two additional registry keys in `ROLLOUT_MODEL_REGISTRY` and `TRAINING_MODEL_REGISTRY`.

### Interfaces preserved

`ComposedRollout` implements the existing `RolloutCostModel` ABC
(`srl/cost/rollout/base.py:49`): `predict(RolloutWorkload)`, `decode_coeffs(n_d, ctxsum)`,
`saturated(...)`. `step_time` is inherited. `ComposedTraining` implements
`estimate(TrainingWorkload, TrainingParallelism) -> TrainingCostEstimate`
(`srl/cost/training/base.py:153`). Neither `instance.py` nor `simulator.py` changes.

### CLI surface

```
srl-profile kernels run       --config <yaml> [--family gemm|attention|comm|...]
srl-profile kernels inspect   --config <yaml>          # coverage / utility report, no GPU
srl-regress   rollout    --model {roofline,psrl,distserve}      real vLLM e2e     (unchanged)
srl-regress   training   --latency-model hydraulis              real Megatron e2e (unchanged)
srl-regress   calibrate  --composed <tables> --against <jsonl>  kernel-sum → real gap (new)
```

`kernels synthesize` is removed. Note there is no regression on the kernel tables themselves : 
utility is a deterministic transform of measurement. The only thing fitted is the residual between
the sum of kernel times and real end-to-end latency, which is what `calibrate` does, following
Frontier's per-op `calibration_scale`.

## 5. Files

### New

- `srl/cost/kernel/{hardware,sol,table,interp,database,model_ops}.py`
- `srl/cost/rollout/composed.py`, `srl/cost/training/composed.py`
- `srl/profiler/kernels/{gemm,elementwise,attention_bwd,optimizer}.py`
- `srl/cli/`: `kernels inspect` and `regress calibrate` handlers
- `examples/profile/kernel/qwen3_8b_h20_tp1.yaml`, `..._tp2.yaml`

### Deleted

- `srl/profiler/kernels/compose.py` (regression at the wrong layer, §1.1)
- `srl/profiler/kernels/synthesize.py` (bridge to a format we no longer target)
- `cmd_kernels_synthesize` in `srl/cli/profile.py`
- `examples/regress/regress_rollout_{psrl,roofline}_kernel.sh` (fit-on-synthetic-data scripts)

### Rewritten

- `srl/profiler/kernels/attention.py`: paged KV via vLLM's selected backend, replacing dense SDPA
- `srl/profiler/kernels/linear.py` → `gemm.py`: unified table, per-rank shapes
- `srl/profiler/kernels/overhead.py`: measure and tabulate CPU overhead
  (schedule / sampler / prepare_inputs, keyed as in Frontier's schema); the file keeps its name but
  `resolve_scheduling_overhead` and the `DEFAULT_SCHEDULING_OVERHEAD_S` constant are removed (D7)
- `srl/profiler/kernels/measurements.py`: parquet writer instead of JSON bundle
- `srl/profiler/kernels/config.py`: sweep grids per op family; drop `synthesis`

Docs to refresh once behavior changes: `examples/profile/README.md`, `examples/regress/README.md`
(both already carry uncommitted edits describing the `synthesize` flow being removed here).

### Reused (do not reimplement)

- `srl/profiler/schema.py`: `ModelDescriptor`, `HardwareDescriptor`, `Parallelism`,
  `ProfileRecord`; `ProfileValidationError` for all validation
- `srl/profiler/kernels/_timing.py`: `time_cuda_call`, `torch_dtype`
- `srl/profiler/kernels/config.py`: `model_head_dim`, `model_intermediate_size`
- `srl/cost/artifact.py`: `CostArtifact`, `ArtifactEntry`, `FitMetrics` for the calibrate output
- `srl/cost/rollout/_fit.py`: `robust_nonnegative_fit`, `split_by_run` for calibrate
- `srl/cost/training/inventory.py`: `parameter_inventory` for training memory
- `srl/component/rollout/engine/advance.py`: untouched, satisfied via D5

## 6. Phasing

| Phase | Scope | Exit criterion |
|---|---|---|
| 1a | `hardware` / `sol` / `table` / `interp` / `database`; GEMM + attention families; vLLM backend alignment | Single-kernel query error < 5% on held-out in-grid points |
| 1b | `model_ops` + `ComposedRollout` + CPU overhead table + D5 tangent | **Report prefill / decode / mixed MRE% separately** vs real vLLM |
| 1c | `regress calibrate` | Per-phase error converges to an acceptable level |
| 2 | `ComposedTraining`: bwd via D4, recompute, PP bubble, optimizer | Report iteration-time error vs real Megatron |

## 7. Verification

### Baseline: audited, and skewed

`outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl`: 49,208 records, H20, Qwen3-8B,
vllm 0.26.0, per-scheduler-iteration timing
(`latency_source: scheduler-iteration-elapsed-ms`). Suitable in kind, but:

| Phase | Records | Coverage |
|---|---|---|
| prefill | **58** | `n_tokens` 256–65536, `request_count` 1–42 |
| decode | **49,144** | `request_count` almost entirely {1, 4, 16, 32, 48, 64}, ~8,190 each |
| mixed | **6** |: |

Prefill is 0.1% of the file, and TP is 1 only. Prefill is exactly where kernel composition is
hardest (quadratic `sum_l2` attention plus large GEMMs). A pooled error metric would be dominated
by 49,144 decode rows. This also reframes the existing `UnifiedRoofline` fit reporting
R² = 0.9999 on this data: that number is substantially a statement about decode.

Hence D8, plus a top-up sweep.

### Steps

1. **Unit**: `sol.py` closed forms against hand-computed values; `interp.py` on synthetic grids
   with known answers. Four cases: exact grid hit; in-grid blend; past-frontier utility hold; and a
   *ragged* table, meaning a corner-truncated grid where one axis's branch has no data at a
   coordinate the other does (attention sweeps produce this whenever `batch_size × seq_len` exceeds
   memory at large shapes): the surviving branch must be SOL-corrected rather than silently
   returning the wrong coordinate's value.
2. **Held-out kernel accuracy**: profile a grid, withhold points, report per-family MRE%.
   Gate for 1a.
3. **Top-up real-machine sweep**: a prefill-weighted vLLM sweep plus a TP=2 run, since 58 prefill
   points and zero TP=2 points cannot support a conclusion.
4. **Composed vs real (the number that matters)**: for each record in the baseline plus the
   top-up, compare `ComposedRollout.predict` against measured `latency_s`. Report
   **prefill MRE% / decode MRE% / mixed MRE% separately**, with counts. Gate for 1b.
5. **Simulator equivalence**: run `srl` end-to-end with `rollout_cost=composed` against
   `rollout_cost=psrl`; confirm the D5 tangent does not distort long decode segments (compare
   `rollout_step.jsonl` step counts and total time).
6. **Training**: `ComposedTraining` vs real Megatron iteration time from
   `srl-profile megatron`, reported per (TP, PP, recompute) cell. Gate for phase 2.

All profiling and simulation runs require
`source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh`.

### Environment (verified)

`vllm 0.26.1.dev0+g568afb3a1`, `flashinfer-python 0.6.14`, `torch 2.11.0+cu128`,
`megatron-core 0.17.1`, `pyarrow 23.0.1`, `pandas 2.3.3`, `scipy 1.17.1`, `numpy 1.26.4`.
`flash-attn` is **not** installed as a standalone package: attention profiling must go through
vLLM's vendored `vllm_flash_attn` or FlashInfer. Hardware: 2 × NVIDIA H20, 97,871 MiB each.

## 8. Deferred

- Compute/communication overlap (both references either omit it or bolt it on late; plain sum first)
- MoE op families (Qwen3-8B is dense)
- Cross-hardware validation: D2 makes it *possible*, but with one GPU type available it stays
  unproven and must not be claimed
- Context parallelism (`cp > 1`), already blocked in the training schema
- Pipeline schedules other than 1F1B
