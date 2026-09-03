# Qwen3-8B kernel profiling

This is a direct-kernel profiling path — no vLLM server, no KV cache manager, and no scheduler.
Attention and GEMM kernels are profiled with synthetic tensors sized from the checkpoint architecture,
so none of the KV-cache-overflow, preemption, or startup-serialization constraints of the server
sweep apply. The grid can be as dense as GPU time allows.

## Running

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
TP=2 bash examples/profile/kernel/profile_qwen3_8b.sh all   # qwen3_8b_h20_tp2.yaml (2 GPUs)
```

`FAMILY` restricts `run` to one family. The tp=4 and tp=8 configs exist only to complete the
`all_reduce` world_size slices, so they sweep the collective alone -- seconds rather than hours:

```bash
TP=4 FAMILY=collective bash examples/profile/kernel/profile_qwen3_8b.sh all
TP=8 FAMILY=collective bash examples/profile/kernel/profile_qwen3_8b.sh all
```

## Output layout

Each profiled `(op, backend)` pair writes one parquet file:

```
{output_dir}/kernel/{gpu}/{family}/{backend}/{version}/{op}_perf.parquet
```

For example, the TP=1 GEMM table lands at:

```
outputs/profile/qwen3_8b/kernel_tp1/kernel/h20/gemm/vllm/2.11.0+cu128/gemm_perf.parquet
```

Attention produces two rows per grid point: one for the vLLM-selected backend and one for
FlashInfer. The vLLM-selected backend is the default for all downstream queries.

## Utility

Utility is `SOL / measured`, where SOL (Speed of Light) is a roofline upper bound derived
from the GPU's peak FLOP/s and memory bandwidth. A utility of 1.0 means the kernel is
exactly at the hardware ceiling. Utilities above 1.2 indicate a bug in the SOL formula, not
a measurement problem.

Calibrated ranges on H20 with Qwen3-8B:
- GEMM: 0.47–0.95, rising monotonically with token count (m=1 is ~0.62, m=8192 is ~0.94).
- Context attention: 0.013–0.951 (full grid including overhead-limited regime at small seq_len).
- Generation attention: 0.014 (batch 1, step 1024) up to 0.436 (batch 64, step 16384).

The very low generation-attention utility at small batch sizes is a known, physically real
effect. Roughly 70 us of constant per-call overhead inside vLLM's attention path is not
modeled by the roofline SOL (the timing harness floor is only ~4.8 us). This is a limit of
the roofline model, not a measurement error.

## Grid memory budgets

Two fields in the YAML truncate the grid to avoid OOM:

- `max_context_tokens`: drops context attention points where
  `batch_size * seq_len > max_context_tokens` (262144 by default, 1 GiB of KV at bfloat16).
- `max_generation_kv_tokens`: drops generation attention points where the total KV cache
  exceeds this many elements (`1 << 34` by default, a 64 GiB cache). The default is
  deliberately generous: the largest configured generation point (batch 64, step 32768,
  8 KV heads, head_dim 128) is 2^31 elements (8 GiB), well inside the ~75 GiB an H20
  has free after Qwen3-8B bfloat16 weights. Lower this explicitly when profiling a
  smaller GPU.

## Attention backends

Each attention grid point is measured twice: once on the backend vLLM's own selector picks,
and once on FlashInfer.

On this hardware the selector picks **`FLASH_ATTN`**, which resolves to FlashAttention-3
vendored inside vLLM (`kernel_source = vllm_flash_attn_fa3`). It serves **both prefill and
decode** — FA3's varlen path covers the two phases, so there is no separate prefill kernel.

**Installing FlashInfer does not change that, and the reason is hardware-dependent.**
Selection is priority-ordered, not availability-ordered: `_get_backend_priorities` in
`vllm/platforms/cuda.py` returns, for non-MLA models, `[FLASHINFER, FLASH_ATTN, ...]` when
`device_capability.major == 10` (Blackwell) and `[FLASH_ATTN, FLASHINFER, ...]` on everything
else, and `get_attn_backend_cls` takes the lowest-priority-number valid candidate. On this
H20 (SM 9.0, Hopper) FlashInfer 0.6.14 is installed and reports as *valid*, but it sits at
priority 1 behind FLASH_ATTN at priority 0, so FA3 wins:

```
VALID backends in priority order (lower = preferred):
   0  FLASH_ATTN      <- selected
   1  FLASHINFER      <- installed and valid, but outranked
   2  TRITON_ATTN
   3  FLEX_ATTENTION
```

So a FlashInfer-first server is what you get on Blackwell, or by passing an explicit
`--attention-backend`. Re-verify this per GPU generation rather than assuming it.

That distinction matters when validating against a real server. **Any end-to-end comparison
must read the `vllm` rows**, because those are the kernels the server actually executed;
scoring against the FlashInfer rows would be measuring a backend that never ran.
`AttentionQuery.backend` defaults to the vLLM table for exactly this reason, and the
end-to-end scripts assert the default resolves there rather than trusting it to stay put.

The FlashInfer rows answer a different question — what a backend switch would buy — and they
show it is shape-dependent rather than a uniform win. Measured ratios of FlashInfer to vLLM
latency on the context grid span 0.210 to 1.053: at batch 1 / seq 128 FlashInfer is 4.8x
faster (14.4 us against 68.7 us), and by seq 2048 it has crossed over to marginally slower
(327 us against 311 us). Most of the small-shape advantage is vLLM's larger per-call dispatch
cost rather than a better kernel.

One consequence worth knowing: vLLM's larger constant floor *flattens* its utility surface
and so makes it easier to interpolate, while FlashInfer's low overhead exposes the true kernel
behaviour including that crossover. FlashInfer context is therefore the least accurate family
in the gate (9.87% against vLLM's 5.74%) despite being the faster backend, and the remedy is
a denser `context_seq_lens` grid below 1024 rather than a floor change.

## Holdout accuracy gate

`kernels holdout` reserves a random interior fraction of grid points and evaluates how well
the utility interpolator predicts them from the remaining data. The gate passes when the
mean relative error (MRE) over held-out interior points is below 5% for every family.

A passing gate means single-kernel interpolation is accurate **on this grid and this GPU**.
It says nothing about whether summing kernels reproduces vLLM end-to-end — that comparison is
`kernels validate` (see below), measured at 3.72% prefill / 4.84% decode / 0.48% mixed on H20
with Qwen3-8B TP=1 (2026-08-25). See `docs/composed-rollout.md` for the full accuracy table,
the segment integrator design, and the known limitations. `kernels validate` also says nothing
about cross-hardware transfer, which the cost model makes possible but which remains unvalidated
with one GPU type.

Report the gate at `--fraction 1.0`. The default `--fraction 0.2` scores a random fifth of
the eligible points, and a small sample can look far better than the surface really is: on
the first H20 sweep GEMM measured 3.38% MRE over 13 sampled points but 8.48% over all 64.

### Current result on H20 (Qwen3-8B): every op except context_attention/flashinfer is under 7%

Measured at `--fraction 1.0`. The `gemm` and attention rows are TP=1; `all_reduce` is TP=2,
since no collective runs at world size 1.

| op | backend | points | MRE | p90 | utility range |
|---|---|---|---|---|---|
| `gemm` | vllm | 64 | 3.91% | 7.70% | 0.473-0.949 |
| `elementwise` | vllm | 52 | 5.01% | 13.83% | 0.162-0.718 |
| `context_attention` | vllm | 59 | 5.74% | 17.27% | 0.204-0.942 |
| `generation_attention` | flashinfer | 72 | 6.01% | 14.61% | 0.203-0.497 |
| `all_reduce` | vllm | 13 | 6.01% | 12.29% | 0.283-0.632 |
| `generation_attention` | vllm | 72 | 6.63% | 17.57% | 0.213-0.465 |
| `context_attention` | flashinfer | 59 | 9.87% | 32.69% | 0.183-0.951 |

Fitted dispatch floors: vllm `context_attention` 32 us / `generation_attention` 22 us,
flashinfer 4 us / 6 us, `elementwise` 2 us, `all_reduce` 2 us.

`context_attention/flashinfer` is the one op above the 5% target, and the cause is grid density
below `seq_len` 1024 rather than a floor misfit: a 0-80 us floor sweep confirmed 4 us is already
optimal. The fix is denser `context_seq_lens` points below 1024.

`all_reduce` is measured at world_size 2, 4 and 8, each in its own config. A query at an
unmeasured degree raises a coverage error rather than extrapolating. That is intended:
`world_size` is a slice column, because a 2-rank and an 8-rank all-reduce run different schedules.

The collective is timed through `get_tp_group().all_reduce` -- vLLM's own dispatcher -- and **not**
`torch.distributed.all_reduce`. Measured on H20 at all three degrees, that call is never what a
server selects: the custom P2P kernel covers the whole 8-512 KiB range a decode step lives in.
Both tables are written; `vllm` is what the cost model reads and `nccl` is kept as a control.
Holdout, dispatched vs control: 6.01%/4.96% at tp=2, 3.76%/7.56% at tp=4, 7.56%/24.80% at tp=8.
See `docs/profiling/tensor-parallel.md` and `DISPATCH.md`.

See `docs/kernel-ops.md` for how the op abstraction works and how to add a family, and
`docs/composed-rollout.md` for why the `elementwise` table is measured but deliberately not
summed into a rollout step.

#### As-shipped numbers (before any fixes)

| family | backend | points | MRE |
|---|---|---|---|
| `gemm` | vllm | 64 | 8.48% |
| `context_attention` | flashinfer | 59 | 11.39% |
| `context_attention` | vllm | 59 | 20.85% |
| `generation_attention` | flashinfer | 72 | 20.08% |
| `generation_attention` | vllm | 72 | 34.89% |

The failures come from two distinct mechanisms, each with its own zero-new-measurement
remedy.

#### Attention families: dispatch-floor resolved by max-floor in SOL

The dominant cause for all four attention rows is a constant per-call dispatch cost that
the pure roofline SOL does not model. Real H20 data shows decode latency flat across a 64x
increase in KV work at small batch sizes (batch=1 measures ~70 us at step 128 and still
~70 us at step 8192). This is conclusive evidence the cost is `max(floor, roofline_work)`,
not `floor + roofline_work`.

The fix (Fix B, 2026-08-24) adds `floor_s` as a third component inside `SolComponents.latency_s`,
making SOL `max(math_s, memory_s, floor_s)`. Per-backend, per-op floors are fitted from
real H20 measurements into `floors.json`, with `FALLBACK_FLOORS_S` in `sol.py` as the
hardcoded fallback when that file is absent:

| backend | context floor | generation floor |
|---|---|---|
| vLLM | 32 us | 22 us |
| FlashInfer | 4 us | 6 us |

These are fitted values from H20 hardware (Qwen3-8B, TP=1, 2026-08-24). The vLLM floor
is far higher because vLLM's Python/dispatch path is much more expensive than FlashInfer's.

The `context_attention/flashinfer` family at 9.87% remains above the 5% gate. The 4 us
floor is correct for the dispatch mechanism but insufficient to collapse residual variance
at small shapes where both math and memory work are also near-zero. Densifying the grid
in that regime is the next lever; no new floor tuning is warranted.

#### GEMM: cross-projection neighbour contamination

Error is concentrated at small token counts (m ≤ 256: MRE 12.7%; m > 256: MRE 1.4%). The
tile-cliff observation is directionally real but names the wrong fix. The actual mechanism is
neighbour identity: the unified GEMM table keys its slice only on dtype, so all four
Qwen3-8B projections share one grid. In log2 space the nearest neighbour to
`(m=16, n=4096, k=4096)` is `(m=16, n=6144, k=4096)` — a different projection with utility
0.563 against the true 0.815 — closer than the same-projection neighbour. Slicing per
`(n, k)`, i.e. pure 1-D interpolation in m with zero new measurement, takes GEMM from
8.48% MRE to 4.08% MRE (p90 from 26.48% to 9.62%), through the 5% gate. Raising the
neighbour count barely helps (8 neighbours gives 7.16%), which confirms the problem is
neighbour identity, not neighbour count.

## End-to-end validation: `kernels validate`

`kernels holdout` tests whether single-kernel interpolation is self-consistent.
`kernels validate` tests whether summing those kernels reproduces a real vLLM engine iteration —
a different and harder claim.

```bash
dryrun-profile kernels validate \
    --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml \
    --baseline outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
    --model-path /path/to/Qwen3-8B \
    --surface pchip \
    --min-samples 1 \
    --gate 100 \
    --output /tmp/validate.json
```

The baseline is a vLLM JSONL profiling run. The tool groups iterations into buckets of
near-identical shape and computes the mean relative error (MRE) of `ComposedRollout`'s
predictions against the median measured latency of each bucket.

**Always compare against the `vllm` rows.** Those are the kernels the server actually ran.
FlashInfer rows answer what a backend switch would buy; scoring against them on a vLLM server
is measuring a backend that never ran.

**Use `--min-samples 1` to score all three phases.** At the default `--min-samples 20`,
prefill and mixed phases are dropped from the report because neither accumulates 20 identical
iterations in a typical vLLM run. The tool warns explicitly about each dropped phase.

**Never report the validation at `--fraction 0.2`.** That flag applies to the holdout gate,
not to validation. The validate command uses all iterations regardless.

### Current result on H20 (Qwen3-8B, TP=1, 2026-08-25)

Measured with `--min-samples 1 --gate 100` against `outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl`:

| phase | buckets | MRE (pchip) | MRE (shepard) | note |
|-------|---------|------------|--------------|------|
| prefill | 29 | 3.72% | 3.72% | 58 real iterations |
| decode | 216 | 4.84% | 4.95% | |
| mixed | 5 | 0.48% | 0.43% | 6 real iterations only |

All three phases are inside the 5% gate. The mixed result (0.48%) is encouraging but rests on
only six real baseline records. The baseline is 0.01% mixed; a simulator with chunked prefill
produces them constantly. Six samples is not a validation — see `docs/composed-rollout.md` for
the full discussion of limitations.

## Config files

| File | TP | GPU count | Output dir |
|------|----|-----------|------------|
| `qwen3_8b_h20_tp1.yaml` | 1 | 1 | `outputs/profile/qwen3_8b/kernel_tp1` |
| `qwen3_8b_h20_tp2.yaml` | 2 | 2 | `outputs/profile/qwen3_8b/kernel_tp2` |
