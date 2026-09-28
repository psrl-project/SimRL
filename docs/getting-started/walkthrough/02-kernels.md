# Stage 2: kernels

Stage 1 measured the inference engine from the outside: one number per scheduler
iteration. This stage measures it from underneath, timing each CUDA kernel a layer runs,
and then prices a step by summing them.

It costs about an hour instead of six minutes. What it buys is the ability to price a
shape the end-to-end sweep never ran, because the kernels a step is made of are the same
kernels at any batch shape.

**This is the one path with no regression step.** `ComposedRollout` reads the measured
tables directly.

## What it needs

A checkpoint and one GPU. The grid lives in
`examples/profile/kernel/qwen3_8b_h20_tp1.yaml`:

```yaml
gemm:
  token_counts: [1, 2, 4, 8, 16, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

attention:
  batch_sizes: [1, 2, 4, 8, 16, 32, 48, 64]
  context_seq_lens: [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
  generation_steps: [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
  block_size: 16

measurement:
  warmup_iterations: 5
  active_iterations: 20
```

## The commands, in this order

```bash
CFG=examples/profile/kernel/qwen3_8b_h20_tp1.yaml
dryrun-profile kernels run       --config $CFG
dryrun-profile kernels fit-floor --config $CFG
dryrun-profile kernels holdout   --config $CFG --fraction 1.0
dryrun-profile kernels validate  --config $CFG \
    --baseline outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl --min-samples 1
```

Or `bash examples/profile/kernel/profile_qwen3_8b.sh all`.

**The order is not optional.** `fit-floor` must precede `holdout`, because the attention
families would otherwise be scored against a zero dispatch floor and read far worse than
they are.

`--fraction 1.0` on the holdout matters too. A smaller fraction flatters the result: at
`--fraction 0.2` GEMM scored 3.38% where the full grid gives 8.48% on the same tables,
because holding out fewer points leaves the interpolator more neighbours to lean on.
Report the full fraction.

## How long it takes

About 1 hour for 338 shapes on 8x H20, measured 2026-08-31. `run` dominates. The three
scoring commands take seconds.

## What lands on disk

Parquet tables, not JSON, laid out by GPU, kernel family, backend, and backend version:

```text
outputs/profile/qwen3_8b/kernel_tp1/kernel/
├── holdout_report.json
├── validate_report.json
└── h20/
    ├── gemm/vllm/2.11.0+cu128/
    │   └── gemm_perf.parquet                       134 rows
    ├── attention/vllm/0.26.1.dev0+g568afb3a1/
    │   ├── context_attention_perf.parquet          59 rows
    │   ├── generation_attention_perf.parquet       80 rows
    │   └── floors.json
    ├── attention/flashinfer/0.6.14/
    │   ├── context_attention_perf.parquet          59 rows
    │   ├── generation_attention_perf.parquet       80 rows
    │   └── floors.json
    └── elementwise/vllm/0.26.1.dev0+g568afb3a1/
        ├── elementwise_perf.parquet                60 rows
        └── floors.json
```

472 rows in total. The version in the path is deliberate: a kernel measured against
vLLM 0.26 is not evidence about vLLM 0.27, and putting the version in the path means the
two cannot be silently mixed.

The GEMM table has ten columns:

```text
framework  version       device      op_name  kernel_source                m     n      k  gemm_dtype  latency_s
vllm       2.11.0+cu128  NVIDIA H20  gemm     torch.nn.functional.linear   1  4096   4096  bfloat16    0.000012
vllm       2.11.0+cu128  NVIDIA H20  gemm     torch.nn.functional.linear   1  4096  12288  bfloat16    0.000039
vllm       2.11.0+cu128  NVIDIA H20  gemm     torch.nn.functional.linear   1  6144   4096  bfloat16    0.000020
```

`m` is the token count and is interpolated along. `n`, `k`, and `gemm_dtype` are matched
exactly, never interpolated across, because a `(4096, 4096)` GEMM and a `(151936, 4096)`
GEMM can select entirely different tile sizes. {doc}`../concepts` covers the slice-versus-
axis distinction, and it is the same reason `all_reduce` slices on `world_size`.

Attention tables carry sixteen columns, including `batch_size`, `num_heads`,
`num_key_value_heads`, `head_dim`, `isl`, `step`, `block_size`, and `window_size`.

### Where the swept shapes come from, and what that costs you

Worth understanding before you spend the hour, because it decides when the sweep has to be
repeated.

`(n, k)` is **not** something you choose. It is computed from `config.json` by the model
family. For Qwen3-8B at TP=1 there are exactly five pairs:

| shape | n | k |
|---|---|---|
| `qkv_proj` | `(32 + 2*8) * 128` = 6144 | 4096 |
| `o_proj` | 4096 | `32 * 128` = 4096 |
| `gate_up_proj` | `2 * 12288` = 24576 | 4096 |
| `down_proj` | 4096 | 12288 |
| `lm_head` | 151936 | 4096 |

The sweep crosses those with the `token_counts` from the config, adds the backward (wgrad)
axis permutations and the logits row counts, and deduplicates. That is the 134 rows in 35
slices. At simulation time `ModelOps` builds its queries from the *same* function, so a
slice always hits.

The asymmetry between the two kinds of column is what you feel in practice:

```text
slice (4096, 4096) measured, m=1500 never measured  -> 3.724e-04 s, utility 0.9131
slice (5000, 4096) never measured, any m            -> KernelCoverageError
slice (4096, 4096), m=99999 past the measured 24576  -> 2.419e-02 s, utility 0.9372
```

An unmeasured `(n, k)` **raises**. An `m` past the top of the sweep holds the boundary
utility and reconstructs latency as `SOL(query)/utility`, because extrapolating an
efficiency is defensible where extrapolating a latency is not.

So `m` is the only cheap axis. A dozen token counts per slice cover any batch shape on that
projection. Everything else means re-sweeping: **a different model changes every `(n, k)`,
and so does a different TP degree**, since the shapes are sharded at profiling time. That is
what the per-degree trees and the guards in `simulate.sh` are for.

`floors.json` holds the fitted per-call dispatch cost:

```json
{
  "context_attention": 3.2e-05,
  "generation_attention": 2.2e-05,
  "fitted_mre_percent": {
    "context_attention": 5.7432,
    "generation_attention": 6.6293
  },
  "candidates_us": [0, 2, 4, 6, 8, 12, 16, 22, 32, 45, 60, 80],
  "gpu": "h20",
  "backend": "vllm"
}
```

A floor exists because at small batch sizes a decode attention kernel is dispatch-limited:
its latency barely moves across a 64x increase in KV length. The floor goes *inside*
`max(compute, memory, floor)` rather than being added, so it binds only in that regime.
GEMM needs no floor.

## What the tables are used for

`ComposedRollout` reads them straight, with no regression and no `CostArtifact`:

```bash
ROLLOUT_MODEL=composed \
KERNEL_ROOT=outputs/profile/qwen3_8b/kernel_tp1/kernel \
MODEL_PATH=$MODEL_PATH \
GPU=h20 \
bash examples/simulate/simulate.sh
```

What it interpolates is not latency but **utility**, the dimensionless ratio
`SOL / measured_latency`, where SOL is the analytic roofline. On this GEMM table latency spans
2268x while utility spans 4.65x, and inside the `(n=4096, k=4096)` slice it is 566x against
1.69x. Interpolating the narrow quantity and letting the analytic formula carry the orders of
magnitude is what makes this generalise. {doc}`../../cost-models/composed-rollout` has the segment
integrator and its limits.

## How to tell it worked

Per-kernel holdout, from `holdout_report.json`:

```bash
python -c "
import json
for r in json.load(open('outputs/profile/qwen3_8b/kernel_tp1/kernel/holdout_report.json')):
    print(f\"{r['op_name']:22s} {r['backend']:10s} MRE {r['mre_percent']:6.2f}%  p90 {r['p90_percent']:6.2f}%  n={r['point_count']}\")
"
```

For the tables above:

| Op | Backend | MRE | p90 | Points |
|---|---|---|---|---|
| `gemm` | vllm | 3.91% | 7.70% | 64 |
| `context_attention` | vllm | 5.74% | 17.27% | 59 |
| `generation_attention` | flashinfer | 6.01% | 14.61% | 72 |
| `generation_attention` | vllm | 6.63% | 17.57% | 72 |
| `context_attention` | flashinfer | 9.87% | 32.69% | 59 |

`utility_min` and `utility_max` in the same report are worth a glance. GEMM spans 0.473 to
0.949 over the holdout points, so the interpolator is working over a narrow range. Generation
attention tops out at 0.50, which says that kernel never approaches its roofline, and that is
expected for a memory-bound decode kernel. {doc}`../../profiling/kernels` owns the full table.

Then the end-to-end score, `validate`, which compares summed kernels against the real
vLLM baseline from stage 1. At TP=1 that reads 3.72% prefill over 29 buckets and 4.84%
decode over 216. {doc}`../../reference/validation` owns those figures.

## Tensor parallel needs its own tree

A `tp>1` step pays two all-reduces per layer, 72 of them for Qwen3-8B, which is 13.2% of
a decode step. Pricing them at zero degrades median decode error from 3.07% to 10.70%, so
the collective table is not optional.

```bash
TP=2 bash examples/profile/kernel/profile_qwen3_8b.sh all      # ~1 h, compute plus collective
for TP in 4 8; do                                              # seconds each, collective only
    TP=$TP FAMILY=collective bash examples/profile/kernel/profile_qwen3_8b.sh all
done
```

Kernel trees are **per degree** by convention (`kernel_tp1`, `kernel_tp2`, ...). GEMM
shapes and attention head counts are sharded at sweep time, so a tree swept at one degree
prices the wrong shapes at another. `simulate.sh` checks for this and refuses, because the
resulting error looks like a model defect rather than a wiring mistake.

Simulate at that degree:

```bash
ROLLOUT_MODEL=composed ROLLOUT_TP=2 \
KERNEL_ROOT=outputs/profile/qwen3_8b/kernel_tp2/kernel \
MODEL_PATH=$MODEL_PATH GPU=h20 bash examples/simulate/simulate.sh
```

Two things about the TP=2 baseline are worth knowing before you spend the hour.

**Run the baseline servers serially.** With `max_parallel: auto`, four concurrent vLLM
servers contaminate each other: one degraded mid-run and decode scored 16.39% instead of
9.51%. The tell was self-contradiction rather than an outlier, since the `requests = 16`
bucket read *faster* than it should have (median ratio 0.513, which then moved to 0.993 on
re-measurement). Use `examples/profile/vllm/qwen3_8b_tp2_serial.yaml`.

**Never benchmark a point with `num-prompts: 1`.** It records exactly one cold iteration
carrying about 78 ms of one-time server work that no steady-state step pays. Six prefill
points collected that way scored 46.98%. Re-measured with `num-prompts: 8` at the same shape
they scored 3.98%, with the prediction unchanged.

{doc}`../../profiling/tensor-parallel` has the full account.

## Next

{doc}`03-megatron-training` moves to the training side.
