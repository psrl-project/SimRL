# Kernel profiling

The kernel pipeline is the only profiling path in SimRL with no regression step. Instead of
fitting a parametric model, it writes parquet *performance tables*, one per `(op, backend)` pair,
and {doc}`../cost-models/composed-rollout` reads those tables directly, summing measured kernel
latencies per inference step. A simulation that uses `ComposedRollout` therefore requires this
sweep to be run first.

The pipeline is also structurally different from the vLLM, Megatron, and FSDP sweeps: there is no
server, no KV-cache manager, and no scheduler. Kernels are timed with synthetic tensors sized from
the checkpoint architecture, so none of the KV-cache-overflow, preemption, or startup-serialisation
constraints of the server sweep apply and the grid can be as dense as GPU time allows.

## Running the sweep

```bash
# Profile all families and run the holdout accuracy gate.
bash examples/profile/kernel/profile_qwen3_8b.sh all

# Profile only (no gate).
bash examples/profile/kernel/profile_qwen3_8b.sh run

# Gate only (requires a completed run).
bash examples/profile/kernel/profile_qwen3_8b.sh holdout
```

`TP` selects the config. The default is `TP=1`:

```bash
TP=1 bash examples/profile/kernel/profile_qwen3_8b.sh all   # qwen3_8b_h20_tp1.yaml
TP=2 bash examples/profile/kernel/profile_qwen3_8b.sh run   # qwen3_8b_h20_tp2.yaml (needs 2 GPUs)
```

The individual subcommands are also available directly:

```bash
srl-profile kernels run       --config CFG [--family all|gemm|attention|elementwise|collective]
srl-profile kernels fit-floor --config CFG [--candidates-us 0 2 4 6 8 12 16 22 32 45 60 80]
srl-profile kernels holdout   --config CFG [--fraction 1.0] [--gate 5.0] [--output PATH]
srl-profile kernels validate  --config CFG --baseline VLLM_JSONL [--surface pchip|shepard] [--gate PCT]
```

Run them in that order. `fit-floor` is not optional: attention latency does not fall to zero at
tiny shapes, and fitting the dispatch floor moved context attention from 20.85% to 5.74% MRE and
generation attention from 34.89% to 6.63%. Those are not marginal improvements. Skipping the step
means the surface is systematically wrong in the overhead-dominated regime.

### Timing (Qwen3-8B, H20, 2026-08-25)

| Step | Wall clock |
|---|---|
| Attention run (2 backends × 144 points) | ~20 min |
| GEMM run (134 shapes) | ~20 min |
| Elementwise run | minutes |
| `fit-floor` / `holdout` / `validate` | minutes each |

**Total ~1 hour.** This is the second-cheapest pipeline after the vLLM sweep.

## Output layout

Each profiled `(op, backend)` pair writes one parquet file:

```
{output_dir}/kernel/{gpu}/{family}/{backend}/{version}/{op}_perf.parquet
```

For TP=1 on H20 the tree looks like this:

```
outputs/profile/qwen3_8b/kernel_tp1/kernel/h20/
├── gemm/vllm/2.11.0+cu128/gemm_perf.parquet                      (134 rows)
├── attention/vllm/0.26.1.dev0+.../
│   ├── context_attention_perf.parquet      (59 rows)
│   ├── generation_attention_perf.parquet   (72 rows)
│   └── floors.json
├── attention/flashinfer/0.6.14/
│   ├── context_attention_perf.parquet      (59 rows)
│   ├── generation_attention_perf.parquet   (72 rows)
│   └── floors.json
└── elementwise/vllm/.../elementwise_perf.parquet                  (60 rows)
```

The directory path encodes GPU / family / backend / version, which is what scopes a `floors.json`
to the backend it was fitted for. `ComposedRollout` resolves paths from `kernel_table_root` plus
the `gpu` field in the simulation config, and no other path configuration is needed.

Config files and their output dirs:

| File | TP | GPU count | Output dir |
|------|----|-----------|------------|
| `qwen3_8b_h20_tp1.yaml` | 1 | 1 | `outputs/profile/qwen3_8b/kernel_tp1` |
| `qwen3_8b_h20_tp2.yaml` | 2 | 2 | `outputs/profile/qwen3_8b/kernel_tp2` |

## Column schemas

`gemm_perf.parquet` columns:

```
framework, version, device, op_name, kernel_source, m, n, k, gemm_dtype, latency_s
```

A few rows from the Qwen3-8B TP=1 table (H20, vllm 2.11.0+cu128):

```
m=1   n=4096   k=4096    bfloat16   12 us
m=1   n=4096   k=12288   bfloat16   39 us
m=1   n=6144   k=4096    bfloat16   20 us
```

`context_attention_perf.parquet` and `generation_attention_perf.parquet` share this schema:

```
framework, version, device, op_name, kernel_source,
batch_size, num_heads, num_key_value_heads, head_dim,
isl, step, attn_dtype, kv_cache_dtype, block_size, window_size, latency_s
```

Context rows use `isl` (input sequence length) as the axis, and generation rows use `step`
(current KV depth). Each attention grid point is measured twice and produces two rows: one for the
vLLM-selected backend and one for FlashInfer.

`floors.json` is written beside each attention table:

```javascript
{
  "context_attention": 0.000032,      // seconds
  "generation_attention": 0.000022,   // seconds
  "fitted_mre_percent": {
    "context_attention": 5.7432,
    "generation_attention": 6.6293
  },
  "candidates_us": [0, 2, 4, 6, 8, 12, 16, 22, 32, 45, 60, 80],
  "gpu": "h20",
  "backend": "vllm"
}
```

## Grid memory budgets

Two fields in the YAML truncate the grid to avoid OOM:

- `max_context_tokens`: drops context-attention points where `batch_size * seq_len` exceeds this
  value. Default 262,144 (1 GiB of KV at bfloat16).
- `max_generation_kv_tokens`: drops generation-attention points where the total KV cache exceeds
  this many elements. Default `1 << 34` (a 64 GiB cache). That is deliberately generous: the
  largest configured generation point (batch 64, step 32768, 8 KV heads, head dim 128) is
  2³¹ elements (8 GiB), well inside the ~75 GiB an H20 has free after loading Qwen3-8B bfloat16
  weights. Lower this explicitly when profiling a smaller GPU.

## Attention backends and why vLLM wins on H20

Each attention grid point is measured twice: once on the backend vLLM's own selector picks, and
once on FlashInfer. On H20 (SM 9.0, Hopper) with FlashInfer 0.6.14 installed, the selector picks
`FLASH_ATTN`, which resolves to FlashAttention-3 vendored inside vLLM (`kernel_source =
vllm_flash_attn_fa3`). It serves both prefill and decode, since FA3's varlen path covers both
phases.

The reason FlashInfer loses despite being installed and valid is priority ordering, not
availability. `_get_backend_priorities` in `vllm/platforms/cuda.py` returns
`[FLASHINFER, FLASH_ATTN, ...]` when `device_capability.major == 10` (Blackwell) and
`[FLASH_ATTN, FLASHINFER, ...]` on everything else. `get_attn_backend_cls` takes the
lowest-priority-number valid candidate. On H20 that is priority 0 (`FLASH_ATTN`), not priority 1
(`FLASHINFER`):

```
VALID backends in priority order (lower = preferred):
   0  FLASH_ATTN      <- selected
   1  FLASHINFER      <- installed and valid, but outranked
   2  TRITON_ATTN
   3  FLEX_ATTENTION
```

On Blackwell hardware, or with an explicit `--attention-backend flashinfer` flag, the FlashInfer
rows become the active backend and the vLLM rows answer a counterfactual. Re-verify the priority
order per GPU generation rather than assuming it stays the same.

The FlashInfer rows show the backend switch is shape-dependent rather than a uniform win. Measured
ratios of FlashInfer to vLLM latency on the context grid span 0.210 to 1.053: at batch 1 /
seq 128, FlashInfer is 4.8x faster (14.4 us against 68.7 us), and by seq 2048 it has crossed over
to marginally slower (327 us against 311 us). Most of the small-shape advantage is vLLM's larger
per-call dispatch cost rather than a better kernel. One consequence: vLLM's larger constant floor
flattens its utility surface and makes it easier to interpolate, which is part of why
`context_attention/vllm` scores 5.74% MRE while `context_attention/flashinfer` scores 9.87%.

## Dispatch floors and why they matter

Kernel latency at small shapes is dominated by a constant per-call dispatch cost that the pure
roofline SOL does not model. Real H20 data shows decode latency flat at approximately 70 us across
a 64x increase in KV work at batch 1, which is conclusive that the cost structure is
`max(math_work, memory_work, floor)`, not `floor + roofline`. The floor is a lower bound, not an
addend, because adding it would double-count in the overhead-dominated regime.

`srl-profile kernels fit-floor` fits per-backend, per-op floor values from real measurements,
writing them to `floors.json`. The fallback when that file is absent is `FALLBACK_FLOORS_S` in
`sol.py`. Fitted values on H20 (Qwen3-8B, TP=1, 2026-08-24):

| Backend | `context_attention` floor | `generation_attention` floor |
|---------|--------------------------|------------------------------|
| vLLM | 32 us | 22 us |
| FlashInfer | 4 us | 6 us |
| elementwise (vLLM) | 2 us | not measured |
| all_reduce (NCCL) | 2 us | not measured |

The vLLM floor is far higher because vLLM's Python dispatch path is much more expensive than
FlashInfer's. The fitted floors eliminated the largest sources of pre-fix error: attention MRE
went from 20.85% → 5.74% (context/vllm) and 34.89% → 6.63% (generation/vllm).

`floors.json` is keyed by op name. An older format keyed by phase (`context`, `generation`) is
rejected with an explicit error message rather than being read leniently. A leniently-read
old-format file applies a zero floor, which measures 34.89% MRE for generation attention.

## Holdout accuracy gate

`kernels holdout` reserves a random interior fraction of grid points and evaluates how well the
utility interpolator predicts them from the remaining data. The gate passes when the mean relative
error (MRE) over held-out interior points is below 5% for every family. A passing gate means
single-kernel interpolation is self-consistent on this grid and this GPU. It says nothing about
whether summing kernels reproduces a real vLLM engine iteration. That harder question is answered
by `kernels validate` (see below).

### Always report at `--fraction 1.0`

The default `--fraction 0.2` scores a random fifth of eligible points, and a small sample can look
far better than the surface really is. On the first H20 sweep, GEMM measured 3.38% MRE over 13
sampled points but 8.48% over all 64. Using `--fraction 0.2` for a go/no-go decision is therefore
not sound. Run holdout at `--fraction 1.0` and treat that as the authoritative number.

### Current gate results (Qwen3-8B, H20, 2026-08-25)

Measured at `--fraction 1.0 --gate 5.0`, and reproduced on 2026-08-31 to four significant
figures. This is the canonical accuracy table for this GPU and model. Three inconsistent subsets
previously existed across multiple docs and this table supersedes all of them.

The rows come from two sweeps. `gemm`, both attention families and `elementwise` are from the
TP=1 config, while `all_reduce` is from the TP=2 config, because a collective needs at least two ranks
and cannot be measured at world size 1.

| Op | Backend | Sweep | Points | MRE | p90 | max | Utility range | Gate (5%) |
|----|---------|-------|--------|-----|-----|-----|---------------|-----------|
| `gemm` | vllm | TP=1 | 64 | 3.91% | 7.70% | 47.98% | 0.473-0.949 | PASS |
| `elementwise` | vllm | TP=1 | 52 | 5.01% | 13.83% | 25.37% | 0.162-0.718 | fail |
| `context_attention` | vllm | TP=1 | 59 | 5.74% | 17.27% | 25.92% | 0.204-0.942 | fail |
| `generation_attention` | flashinfer | TP=1 | 72 | 6.01% | 14.61% | 39.67% | 0.203-0.497 | fail |
| `all_reduce` | vllm | TP=2 | 13 | 6.01% | 12.29% | 21.74% | 0.283-0.632 | fail |
| `generation_attention` | vllm | TP=1 | 72 | 6.63% | 17.57% | 28.46% | 0.213-0.465 | fail |
| `context_attention` | flashinfer | TP=1 | 59 | 9.87% | 32.69% | 47.12% | 0.183-0.951 | fail |

### The collective is measured through vLLM's dispatcher

`all_reduce` is timed by calling `get_tp_group().all_reduce`, which is vLLM's own
`CudaCommunicator` and therefore selects whatever kernel a real server would select. It is *not*
timed with `torch.distributed.all_reduce`, which was the original choice and was wrong: measured
on H20 at world sizes 2, 4 and 8, that call is never what a server dispatches.

Dispatch measured with `examples/profile/kernel/dispatch_probe.py` (full table in
`examples/profile/kernel/DISPATCH.md`). For Qwen3-8B a message is `tokens * 4096 * 2` bytes, so a
decode step at batch 1-64 lives in 8-512 KiB:

| Message size | world_size=2 | world_size=4 | world_size=8 |
|---|---|---|---|
| 8-128 KiB | `custom_all_reduce` | `custom_all_reduce` | `custom_all_reduce` |
| 256 KiB to 4 MiB | `custom_all_reduce` | `custom_all_reduce` | `torch_symm_mem` |
| 8-32 MiB | `torch_symm_mem` | `torch_symm_mem` | `torch_symm_mem` |
| 64-128 MiB | `torch_symm_mem` / `pynccl` | `pynccl` | `torch_symm_mem` / `pynccl` |

The whole decode range is vLLM's custom one-shot/two-shot P2P kernel, because on sm_90 the custom
path's ceiling at two ranks is 8 MiB, far above what a decode step needs. NCCL symmetric memory
never fires despite `VLLM_ALLREDUCE_USE_SYMM_MEM` defaulting on, because
`is_symmetric_memory_enabled()` also requires `VLLM_USE_NCCL_SYMM_MEM`, which defaults to 0.

Both tables are written from one sweep, so the cost of the original choice is a number rather than
an argument. `nccl` is retained as a control and is selectable with `CommQuery(backend="nccl")`:

| Sweep | `vllm` MRE | `nccl` MRE |
|---|---|---|
| TP=2 | 6.01% | 4.96% |
| TP=4 | 3.76% | 7.56% |
| TP=8 | **9.30%** | **10.66%** |

At TP=2 the two tables agree within about 6% in the decode range, because both are pinned to the
same 18 us launch floor, so the original table was wrong in *identity* rather than badly wrong in latency
there. The penalty grows with world size: the NCCL control beats the dispatched table at TP=2
and loses at TP=4 and TP=8, because its latencies increasingly describe a schedule the
server does not run.

The TP=8 row was corrected on 2026-10-04: it previously read 7.56% / 24.80%, from a
`holdout_report.json` whose mtime predates the tables it scores. See the warning in
{doc}`tensor-parallel`.

Because the dispatch splits by message size at every world size, `kernel_source` reads `mixed` on
these tables and the per-size backend is recoverable from `DISPATCH.md` and the profiler's log.

::{note}
The committed `holdout_report.json` under the TP=1 tree holds only five of these rows. It was
written before `elementwise` and `all_reduce` were swept and never regenerated, so re-running
`kernels holdout` produces six rows for TP=1 where the file has five. The table above is the
regenerated result, not the stale file.
::

Only `gemm` passes the strict 5% gate. The vLLM-backend attention families sit between 5.74% and
6.63%, close enough that the end-to-end composed model still reaches below 5% on every phase
(see {doc}`../cost-models/composed-rollout`). `context_attention/flashinfer` at 9.87% is the
outlier and its cause is grid density, not a floor issue: a 0-to-80 us floor sweep confirmed that
4 us is already optimal. The fix is denser `context_seq_lens` points below seq_len=1024.

`gemm` reports 30 skipped slices at TP=1 and 46 at TP=2. A slice is skipped when it holds too few
points to hold one out and still interpolate, so a high skip count means the 3.91% is computed over
the well-covered slices rather than the whole table.

`all_reduce` is measured at world_size 2, 4 and 8, each in its own config
(`qwen3_8b_h20_tp{2,4,8}.yaml`, where the TP=4 and TP=8 ones sweep the collective family only). A query
at an unmeasured world size raises a coverage error rather than extrapolating, because world_size
is a *slice* column: a 2-rank and an 8-rank all-reduce run different schedules over different
topologies, so interpolation between them is not meaningful. This is intended behaviour, not a gap
in the gate logic.

### What changed from the as-shipped baseline

Before floor fitting, the numbers were considerably worse:

| Family | Backend | MRE (before fix) | MRE (after fix) |
|--------|---------|------------------|-----------------|
| `context_attention` | vllm | 20.85% | 5.74% |
| `generation_attention` | vllm | 34.89% | 6.63% |
| `context_attention` | flashinfer | 11.39% | 9.87% |
| `generation_attention` | flashinfer | 20.08% | 6.01% |
| `gemm` | vllm | 8.48% | 3.91% |

The attention failures all share one root cause, the constant dispatch cost that the pure
roofline does not model, fixed by `max(math, memory, floor)`. The GEMM improvement is unrelated:
the as-shipped table keyed its slice only on dtype, so all four Qwen3-8B projections shared one
grid. In log2 space the nearest neighbour to `(m=16, n=4096, k=4096)` was `(m=16, n=6144,
k=4096)`, a different projection with utility 0.563 against the true 0.815. Slicing per `(n, k)`
reduced GEMM from 8.48% to 4.08% (p90 from 26.48% to 9.62%) with zero new measurements. The fix
then shipped as 3.91% after the full `--fraction 1.0` re-run.

## Utility and calibrated ranges

Utility is `SOL / measured_latency`, where SOL (speed of light) is the roofline upper bound
derived from the GPU's peak FLOP/s and memory bandwidth. A utility of 1.0 means the kernel runs at
the hardware ceiling. Utilities above 1.2 indicate a bug in the SOL formula, not a measurement
outlier.

Calibrated ranges on H20 with Qwen3-8B, TP=1 (2026-08-25):

- **GEMM:** 0.47-0.95, rising monotonically with token count (m=1 is ~0.62, m=8192 is ~0.94).
- **Context attention:** 0.013-0.951. The very low end includes the overhead-limited regime at
  small seq_len where dispatch cost dominates.
- **Generation attention:** 0.014-0.436. Low utility at small batch sizes is physically real:
  roughly 70 us of constant per-call overhead inside vLLM's attention path is not modeled by the
  roofline SOL.

The utility representation has a practical consequence. Across the whole GEMM table the
interpolator learns a surface spanning 4.65x where raw latency spans 2268x, and within a single
slice the gap is wider still (1.69x against 566x for `(n=4096, k=4096)`). Pooled across all five
ops the measured latencies span 58,323x. The roofline carries every order of magnitude
analytically, so the interpolator only has to resolve the residual variation above it.

## End-to-end validation: `kernels validate`

`kernels holdout` tests single-kernel self-consistency. `kernels validate` tests whether summing
those kernels reproduces a real vLLM engine iteration, a different and harder claim. Run it after
the holdout gate passes (or is accepted):

```bash
srl-profile kernels validate \
    --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml \
    --baseline outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
    --model-path $MODEL_PATH \
    --surface pchip \
    --min-samples 1 \
    --gate 100 \
    --output /tmp/validate.json
```

The baseline is a vLLM JSONL profiling run (see {doc}`vllm` for how to produce it). The tool
groups iterations into buckets of near-identical shape and computes the MRE of `ComposedRollout`'s
predictions against the median measured latency of each bucket.

**Always score against the `vllm` rows.** Those are the kernels the server actually ran. The
FlashInfer rows answer what a backend switch would buy, so scoring against them on a vLLM server
measures a backend that never ran. `AttentionQuery.backend` defaults to the vLLM table for exactly
this reason.

**Pass `--min-samples 1` to score all three phases.** At the default `--min-samples 20`, prefill
and mixed are dropped from the report because neither accumulates 20 identical iterations in a
typical vLLM run. The tool warns explicitly about each dropped phase.

Note: `outputs/profile/qwen3_8b/kernel_tp1/kernel/validate_report.json` currently holds a single
degenerate decode bucket at 68.73% MRE and should be regenerated with the command above before
quoting any number from it.

The end-to-end accuracy figures (prefill 3.72%, decode 4.84%, mixed 0.48% on H20, Qwen3-8B,
TP=1, 2026-08-25) belong to {doc}`../cost-models/composed-rollout`, which owns the full
discussion of segment integration, phase decomposition, and known limitations of the composition.

## Using the tables in a simulation

The kernel pipeline requires no regression step. Set `ROLLOUT_MODEL=composed` and point the
simulation at the kernel root:

```bash
ROLLOUT_MODEL=composed \
KERNEL_ROOT=outputs/profile/qwen3_8b/kernel_tp1/kernel \
GPU=h20 \
bash examples/simulate/simulate.sh
```

Or configure it directly in the simulation YAML:

```yaml
rollout_cost:
  name: composed
  kernel_table_root: outputs/profile/qwen3_8b/kernel_tp1
  gpu: H20
  model_path: $MODEL_PATH
  attention_backend: ""   # empty = vLLM-selected backend (recommended)
  utility_surface: pchip
```

Leave `attention_backend` empty to use the vLLM-selected backend. On H20 that resolves to
`FLASH_ATTN` / FA3. Any end-to-end comparison against a real server must use the same backend the
server used, and overriding it breaks the comparison.

## What this pipeline does NOT cover

The following questions are answered elsewhere and are not duplicated here:

- **How to add a kernel op** (`OP_REGISTRY`, `KernelOp` protocol, `measure()`, the floor policy
  `max(math, memory, floor)`, and why elementwise ops are measured but deliberately not summed into
  a rollout step): {doc}`../development/kernel-ops`
- **Model shapes and architecture dispatch** (`ModelFamily`, shape derivation, MoE and MLA
  refusal): {doc}`../development/model-families`
- **TP>1 being end-to-end unvalidated** and the other known accuracy limitations of
  `ComposedRollout`: {doc}`../reference/limitations`
- **ComposedRollout's end-to-end accuracy** and segment integration design:
  {doc}`../cost-models/composed-rollout`
