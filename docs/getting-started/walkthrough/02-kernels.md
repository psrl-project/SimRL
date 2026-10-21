# Stage 2: kernels

Stage 1 measured the inference engine from the outside: one number per scheduler iteration.
This stage measures it from underneath, timing the CUDA kernels a layer runs, and then prices
a step by summing them.

What it buys is the ability to price a shape the end-to-end sweep never ran, because the
kernels a step is made of are the same kernels at any batch shape.

**This is the one path with no regression step.** `ComposedRollout` reads the measured
tables directly, so there is no `CostArtifact`.

## What it needs

A checkpoint and one GPU for the compute families. The grid is
`examples/profile/kernel/qwen3_8b_h20_tp1.yaml`. Its main blocks are:

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

It also declares `elementwise:` and `collective:` blocks. The collective block sweeps
`message_bytes` from 4 KiB to 512 MiB over `topologies: [[2, 1], [4, 1], [8, 1]]`
(`[world_size, nodes]`), so it needs as many GPUs on the host as its largest world size.

## The commands

```bash
CFG=examples/profile/kernel/qwen3_8b_h20_tp1.yaml
for fam in gemm attention elementwise; do
    srl-profile kernels run --config $CFG --family $fam
done
srl-profile kernels fit-floor --config $CFG
srl-profile kernels holdout   --config $CFG
srl-profile kernels validate  --config $CFG \
    --baseline outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl --min-samples 1
```

`--family` accepts `all`, `attention`, `attention_training`, `collective`, `elementwise`,
`gdn`, `gemm`, and `optimizer`, the families registered in `srl/cost/kernel/ops/`.
`--family all` visits the families the config declares and logs the rest as skipped. This config
has no `attention_training`, `gdn`, or `optimizer` block: those serve training-side pricing and
hybrid models. Naming a family the config does not declare raises `No sweep configured for op ...`.

`bash examples/profile/kernel/profile_qwen3_8b.sh all` wraps `run`, `fit-floor`, and `holdout`
with `TP=1|2|4|8` selecting `qwen3_8b_h20_tp${TP}.yaml` and `FAMILY=<name>` restricting `run`.
Its default is `FAMILY=all`, so apply the same caution about the config's blocks.

**The order is not optional.** `fit-floor` reads the tables `run` writes, and `holdout` scores
the surface using the floors it fits. Without `fit-floor`, queries use the fallback constants
in `srl/cost/kernel/sol.py` and a warning says so.

`holdout` and `validate` exit non-zero when a family exceeds `--gate` (defaults 5.0 and 6.0
percent), after writing their report. A failed gate is a finding, not a crash.

`holdout --fraction` defaults to 1.0. A smaller fraction scores a random subset and can
flatter the surface, so the command warns that its MRE understates the real error. Report the
full fraction.

`verify-shapes` needs no profiling at all. It compares the GEMM shapes the model family
declares against the checkpoint's real module graph:

```bash
srl-profile kernels verify-shapes --config $CFG
```

```text
Qwen3ForCausalLM -> family 'dense_llama', 36 layers.
  down_proj: declared (4096, 12288), observed (4096, 12288) OK.
  gate_up_proj: declared (24576, 4096), observed (24576, 4096) OK.
  lm_head: declared (151936, 4096), observed (151936, 4096) OK.
  o_proj: declared (4096, 4096), observed (4096, 4096) OK.
  qkv_proj: declared (6144, 4096), observed (6144, 4096) OK.
Declared shapes match the module graph.
```

## What lands on disk

Parquet tables, not JSON, laid out by GPU, kernel family, backend, and backend version, under
`<output_dir>/kernel/`:

```text
outputs/profile/qwen3_8b/kernel_tp1/kernel/
├── holdout_report.json
├── validate_report.json
└── h20/
    ├── gemm/vllm/2.11.0+cu128/gemm_perf.parquet
    ├── attention/vllm/<vllm version>/
    │   ├── context_attention_perf.parquet
    │   ├── generation_attention_perf.parquet
    │   └── floors.json
    ├── attention/flashinfer/0.6.14/
    │   ├── context_attention_perf.parquet
    │   ├── generation_attention_perf.parquet
    │   └── floors.json
    └── elementwise/vllm/<vllm version>/
        ├── elementwise_perf.parquet
        └── floors.json
```

The version in the path is deliberate: a kernel measured against one vLLM build is not
evidence about another, and the path keeps the two from mixing. When a backend directory holds
more than one version, `KernelDatabase` refuses to pick one.

The GEMM table has ten columns:

```text
framework  version       device      op_name  kernel_source               m  n     k     gemm_dtype  latency_s
vllm       2.11.0+cu128  NVIDIA H20  gemm     torch.nn.functional.linear  1  4096  4096  bfloat16    1.15e-05
vllm       2.11.0+cu128  NVIDIA H20  gemm     torch.nn.functional.linear  1  4096  12288 bfloat16    3.94e-05
vllm       2.11.0+cu128  NVIDIA H20  gemm     torch.nn.functional.linear  1  6144  4096  bfloat16    2.00e-05
```

`m` is the token count and is interpolated along. `n`, `k`, and `gemm_dtype` are matched
exactly. The attention tables add `num_key_value_heads`, `head_dim`, `attn_dtype`,
`kv_cache_dtype`, `block_size`, and `window_size` as the slice, with `num_heads`,
`batch_size`, and `isl` (context) or `step` (generation) as the axes. Elementwise has `dtype`,
`op_kind`, and `n_elements`. {doc}`/getting-started/concepts` covers the slice-versus-axis distinction.

### Where the swept shapes come from

`(n, k)` is not something you choose. It is computed from `config.json` by the model family
(`srl/profiler/kernel/grid/families_dense.py` for `dense_llama`, which covers
`Qwen3ForCausalLM`). For Qwen3-8B at TP=1 the forward projections are five pairs:

| shape | n | k |
|---|---|---|
| `qkv_proj` | `(32 + 2*8) * 128` = 6144 | 4096 |
| `o_proj` | 4096 | `32 * 128` = 4096 |
| `gate_up_proj` | `2 * 12288` = 24576 | 4096 |
| `down_proj` | 4096 | 12288 |
| `lm_head` | 151936 | 4096 |

The sweep crosses these with the configured `token_counts`, adds the backward-pass
permutations (dgrad and wgrad), and deduplicates. At simulation time `ModelOps` builds its
queries from the same function, so a slice always hits. The asymmetry between the two kinds
of column is what you feel in practice:

- an `m` between measured points interpolates utility.
- an `m` past the measured top holds the boundary utility and reconstructs latency as
  `SOL(query) / utility`.
- an `(n, k)` that was never measured raises `KernelCoverageError`.

So `m` is the cheap axis. **A different model changes every `(n, k)`, and so does a different
TP degree**, since shapes are sharded at profiling time. {doc}`/development/model-families`
covers adding a family.

`floors.json` holds the fitted per-call dispatch cost for each floor-taking op:

```json
{
  "context_attention": 3.2e-05,
  "generation_attention": 6e-06,
  "fitted_mre_percent": {"context_attention": 6.4592, "generation_attention": 11.5547},
  "candidates_us": [0, 2, 4, 6, 8, 12, 16, 22, 32, 45, 60, 80],
  "gpu": "h20",
  "backend": "vllm"
}
```

A floor exists because a small attention call is dispatch-limited: its latency barely moves
as the KV length grows. It goes inside `max(compute, memory, floor)` rather than being added,
so it binds only in that regime. GEMM declares no floor. The values are per backend and
version. In the tables above the vLLM generation floor came out at 6 us.

## What the tables are used for

`ComposedRollout` reads them directly, with no regression and no `CostArtifact`. Select it
with `rollout_cost=composed` and name the tables, the GPU slug, and the checkpoint:

```bash
srl rollout_cost=composed \
    rollout_cost.kernel_table_root=outputs/profile/qwen3_8b/kernel_tp1/kernel \
    rollout_cost.gpu=h20 \
    rollout_cost.model_path=$MODEL_PATH \
    workload=bimodal job.n_versions=3 job.batch_size=16 rollout.n_instances=2
```

`kernel_table_root`, `gpu`, and `model_path` are all required. Omitting one raises
`rollout_cost.<field> is required for the composed model`. Other keys in
`srl/config/defaults/rollout_cost/composed.yaml` are `attention_backend` (empty selects the
vLLM-selected backend), `utility_surface` (`pchip` or `shepard`), `decode_integration_stride`,
and `saturation_step_threshold`.

What it interpolates is not latency but **utility**, the dimensionless ratio
`SOL / measured_latency`, where SOL is the analytic roofline. Interpolating the narrow
quantity and letting the analytic formula carry the orders of magnitude is what makes it
generalise. {doc}`/pipeline/cost-models/composed-rollout` has the segment integrator and its
limits.

## How to tell it worked

Per-kernel holdout, from `holdout_report.json`:

```bash
python -c "
import json
for r in json.load(open('outputs/profile/qwen3_8b/kernel_tp1/kernel/holdout_report.json')):
    print(f\"{r['op_name']:22s} {r['backend']:10s} MRE {r['mre_percent']:6.2f}%  p90 {r['p90_percent']:6.2f}%  n={r['point_count']}\")
"
```

Each row reports `mre_percent`, `p90_percent`, `max_percent` with `worst_coords`,
`point_count`, `skipped_slices`, and `utility_min`/`utility_max`. A `point_count` of 0 is a
coverage problem (a grid too sparse or degenerate to hold a point out), not an accuracy
problem. On the tables above, `gemm [vllm]` scored 3.91% MRE over 64 held-out points and the
attention families 6% to 12%. {doc}`/pipeline/profiling/kernels` owns the full table.

Utility above roughly 1.2 means the SOL formula is missing work the kernel does, not that the
GPU beat its spec. Generation attention staying below 0.5 says that kernel never approaches
its roofline.

The end-to-end score is `validate`, which sums kernels per vLLM scheduler iteration and
compares against the stage 1 baseline, bucketed by phase:

```text
decode:  216 buckets, MRE 3.02%
mixed:     5 buckets, MRE 0.56%
prefill:  29 buckets, MRE 3.84%
```

`--min-samples` (default 20) drops buckets with few iterations, and a phase that holds no
bucket at that size is not scored at all. The command warns when that happens, and
`--min-samples 1` scores everything. {doc}`/reference/validation` owns the project's
accuracy figures.

## Tensor parallel needs its own tree

A `tp>1` step pays two all-reduces per layer, so the collective table is not optional, and
GEMM shapes and attention head counts are sharded at sweep time. Keep one kernel tree per TP
degree and simulate against the matching one:

```bash
# GPU kernels at tp=2, in qwen3_8b_h20_tp2.yaml (output_dir .../kernel_tp2)
for fam in gemm attention elementwise collective; do
    srl-profile kernels run --config examples/profile/kernel/qwen3_8b_h20_tp2.yaml --family $fam
done

srl rollout_cost=composed rollout_cost.parallelism.tp=2 \
    rollout_cost.kernel_table_root=outputs/profile/qwen3_8b/kernel_tp2/kernel \
    rollout_cost.gpu=h20 rollout_cost.model_path=$MODEL_PATH ...
```

The collective sweep writes world sizes 2, 4, and 8 into one table per backend (`vllm`,
which times vLLM's own dispatcher, and `nccl` as a control), so one run covers every TP degree
up to 8. The shipped compute configs are `qwen3_8b_h20_tp1.yaml` and `qwen3_8b_h20_tp2.yaml`.
The `tp4` and `tp8` configs declare only the collective block.

A tree swept at another degree does not price the wrong shapes silently: pointing a TP=2
simulation at the TP=1 tree raises `KernelCoverageError: No gemm rows for slice
('bfloat16', 3072, 4096)` followed by the slices that were measured.
{doc}`/pipeline/profiling/tensor-parallel` has the full account.

## Next

{doc}`/getting-started/walkthrough/03-megatron-training` moves to the training side.
