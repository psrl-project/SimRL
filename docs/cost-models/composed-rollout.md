# ComposedRollout

Every other rollout cost model in this project (`roofline`, `psrl`, `distserve`) is fitted to
end-to-end vLLM server measurements. That means running the vLLM profiling sweep and the
`dryrun-regress rollout` step before any simulation. `ComposedRollout` takes a different
approach: it predicts one inference engine iteration by **summing the measured latencies of the
individual GPU kernels** the engine runs. The kernel tables are measured once per GPU and are
parameterised by hardware speed-of-light (SOL), so a GPU change requires only a new calibrated
utility surface, not a new end-to-end vLLM sweep. There is no `dryrun-regress` step before
using it.

Per-kernel holdout accuracy, the performance table layout, dispatch floor fitting, and the
`kernels validate` and `kernels holdout` command mechanics are all covered in
{doc}`../profiling/kernels`, which owns those topics. This page covers the end-to-end model
accuracy, the segment integrator, the configuration fields, and the elementwise design decision.

## Prerequisites

The kernel tables must exist before the simulation. Run the kernel profiling sweep once per GPU
and model combination:

```bash
bash examples/profile/kernel/profile_qwen3_8b.sh all
```

That writes parquet tables under `outputs/profile/qwen3_8b/kernel_tp1/` and prints the holdout
accuracy gate. Check that the vLLM-backend families pass before running a simulation. See
{doc}`../profiling/kernels` for the gate results and what each family's MRE means.

## Configuration

Select `ComposedRollout` with `rollout_cost=composed` and fill in the required fields:

```yaml
defaults:
  - rollout_cost: composed

rollout_cost:
  name: composed

  parallelism:
    tp: 1          # TP=2 is validated above batch 32; see Limitations for the caveats

  kernel_table_root: outputs/profile/qwen3_8b/kernel_tp1
  gpu: H20
  model_path: $MODEL_PATH
  attention_backend: ""          # empty = vLLM-selected backend (recommended)
  utility_surface: pchip         # pchip (default) or shepard
  decode_integration_stride: 1024
```

Field descriptions:

- `parallelism.tp`: Tensor-parallel degree of the engine being simulated. Above 1 the step gains
  two all-reduces per layer, and `kernel_table_root` **must** point at a tree swept at the same
  degree: GEMM shapes and attention head counts are sharded at measurement time, so the `tp=1`
  tree prices every projection at twice its real width. `ComposedRollout` raises if this disagrees
  with the degree its `ModelOps` was built for. `pp`, `dp` and `cp` are accepted here and ignored.
  Via `simulate.sh` this is `ROLLOUT_TP`, which also checks the `KERNEL_ROOT` path names the
  degree. See {doc}`../profiling/tensor-parallel`.
- `kernel_table_root`: Root of the kernel profiling output tree. `ComposedRollout` resolves
  tables at `{kernel_table_root}/kernel/{gpu}/`. Use the repo-relative path produced by the
  profiling sweep. Do not hard-code an absolute path.
- `gpu`: GPU name that selects which subdirectory to load. Must match what the profiling run
  recorded (for example `H20` if the tree contains `kernel/h20/`).
- `model_path`: Path to the HuggingFace checkpoint directory. The model reads the architecture
  (hidden size, number of heads, GQA groups, intermediate size, number of layers) to derive the
  correct GEMM shapes for each step. Use `$MODEL_PATH` in scripts so the path is not hard-coded.
- `attention_backend`: Override the attention backend (`vllm` or `flashinfer`). Leave empty to
  use the backend vLLM's own selector picks. On H20 that resolves to `FLASH_ATTN` / FA3. Any
  end-to-end comparison against a real vLLM server must use the same backend the server used.
  overriding it compares against kernels that never ran.
- `utility_surface`: Interpolation strategy. `pchip` (monotone-preserving cubic Hermite splines,
  hand-rolled) is more accurate than `shepard` on every measured family at this grid density and
  is the default. `shepard` is the log2-space modified Shepard blend and remains available as a
  fallback when the `pchip` grid is sparse.
- `decode_integration_stride`: Steps between lattice nodes in the segment integrator (described
  below). `1024` uses at most six cost evaluations to integrate a 4096-step segment and costs at
  most ~1.3% accuracy versus a step-by-step reference at batch 16. Set to `1` for an exact sum
  at the cost of one kernel query per decode step.

## End-to-end accuracy

Measured on 2026-08-25, H20, Qwen3-8B, TP=1, `pchip` surface. The baseline is the vLLM JSONL
from `outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl`. The tool is `kernels validate`
run with `--min-samples 1 --gate 100` to include all three phases.

| Phase | Buckets | MRE (pchip) | MRE (shepard) | Note |
|-------|---------|-------------|---------------|------|
| prefill | 29 | 3.72% | 3.72% | 58 real iterations |
| decode | 216 | **4.84%** | 4.95% | |
| mixed | 5 | 0.48% | 0.43% | 6 real iterations, see Limitations |

The baseline is almost entirely decode. Prefill has 58 real iterations spread across 29 buckets.
The five mixed buckets collapse from six raw iterations total, so the 0.48% mixed figure is
indicative rather than a validated claim. See the mixed-step limitation below.

### TP=2

Measured 2026-09-03 against `outputs/profile/qwen3_8b/vllm_tp2_serial/profiles/vllm.jsonl` --
49,211 iterations, `max_parallel: 1`, on a grid byte-identical to the TP=1 one so the two compare
bucket for bucket.

| Selection | Buckets | MRE (pchip) | Note |
|---|---|---|---|
| prefill, whole grid | 29 | **3.60%** | no exclusion needed |
| decode, `requests >= 5` | 137 | **2.96%** | the validated figure |
| mixed | 7 | 3.64% | 7 real iterations |
| decode, whole grid | 204 | 9.51% | includes the small-batch gap |

This is the first score of the communication path at all. At TP=1 `comm_queries` returns empty,
so none of the figures above it exercised a collective. The collective contributes 13.2% of a TP=2
decode step (max 21.0%), and pricing it as zero moves median decode error from 3.07% to 10.70%, so
modelling it is load-bearing.

Prefill's original 12.50% was a **baseline artifact**, since fixed. Its six `requests = 1` points
used `num-prompts: 1`, making each recorded iteration a single cold request that carried ~78 ms of
server-side work. Re-measured with `num-prompts: 8` at the same shape they went from 46.98% to 3.98%
MRE with the prediction unchanged. Decode's remaining gap is a **real model error**: `requests 1-4`
over-predicts at median ratio 1.31. Note the direction: it predicts *too much*, so it is not a
missing cost term. A composed step sums 253 isolated kernel latencies each carrying its own fitted
per-call floor, and the excess at `reqs=1` is 6.5 µs per launch, essentially the 6.0 µs all-reduce
floor that CUDA graph replay removes on the real server. See {doc}`../reference/limitations`.

Note the direction of the residual. Every phase's signed error is **negative** (-2.31% decode,
-2.63% prefill, -2.23% mixed), so the model slightly *under*-predicts. This was expected to go the
other way: the collective is timed in isolation, so an unmodelled compute/communication overlap
should have produced over-prediction by roughly the collective's 13.2% share. It did not. Overlap
is therefore not the dominant term at this degree, though it remains unquantified.

::{warning}
An earlier baseline collected with `max_parallel: auto` put decode at 16.39%. One of its four
concurrent servers degraded mid-run: at per-request depth 8,960, `reqs=16` measured 35.76 ms
against `reqs=32`'s 19.24 ms, which is half the batch at half the total KV taking nearly double
the time.
The serial re-measurement confirmed it, moving `reqs=16` from median ratio 0.513 to 0.993 while
every other batch stayed put. Use `max_parallel: 1` for any baseline whose figures will be quoted.
::

Full analysis: `examples/profile/kernel/TP2_VALIDATION.md`.

`pchip` beats `shepard` on every phase: the utility surfaces in this sweep are well enough
behaved that monotone-preserving cubic splines outperform the distance-weighted blend.

### Relation to prior baselines

Before the pchip utility surface was introduced, a hand-composed decode baseline measured 5.00%
MRE. The current pchip result of 4.84% improves on that prior baseline. The 5.00% figure is
superseded and should not be quoted as a current result.

Note: `outputs/profile/qwen3_8b/kernel_tp1/kernel/validate_report.json` currently holds a
single degenerate decode bucket at 68.73% MRE and should be regenerated with the `kernels
validate` command above before quoting any number from it. The figures in the table above come
from a clean re-run.

## Segment integrator

A composed step cost is not `max(F, alpha + beta*k)`, so the closed-form pure-decode advance
used by the fitted rollout models does not apply. The latency of a decode segment, a run of `k`
steps at a fixed batch size over a growing KV history, must be integrated numerically.

### Why naive trapezoid fails

Naive trapezoid summation (one kernel query per step) is not monotone in `k`. On a spiked cost
curve, it produces values of `k` where `total(k) < total(k-1)`. A non-monotone total has no
unique inverse, which causes the simulator's `instance.py` to take the wrong branch when
comparing a step count against its time budget.

### LatticeIntegrator

`LatticeIntegrator` (`dryrun/cost/rollout/segment.py`) is monotone by construction. Lattice
nodes at offsets $0, s, 2s, \ldots$ (where $s$ is `decode_integration_stride`) are sampled once.
Each node carries a fixed per-step rate (the trapezoid average of the cost at that node and the
next) and a cumulative total:

$$\text{total}(k) = \text{cumulative}[i] + \text{rates}[i] \cdot (k - i \cdot s),
\quad \text{where}\ i = \lfloor k / s \rfloor$$

Because `rates[i]` is constant within an interval, the total grows at a fixed positive slope
inside every interval and is continuous and strictly increasing across them. Bisection therefore
inverts it exactly. The same property that makes it invertible also makes it safe to use in the
simulator: the step-count-within-budget query always has a unique answer.

### Accuracy vs cost trade-off

Measured against a step-by-step reference over 4096 steps on real Qwen3-8B kernel tables (H20,
TP=1). The starting KV context matters as much as the batch size, because a cold segment
(starting right after prefill, KV depth 0) has the steepest per-step cost increase:

| Starting context | Batch 16 | Batch 64 |
|---|---|---|
| 0 (cold, just after prefill) | 0.46% | 0.42% |
| 2048 | 0.07% | 0.01% |
| 16384 and beyond | ≤0.01% | ≤0.01% |

Over a shorter 2048-step segment the cold case reaches **1.31%** (batch 16) and 1.11% (batch
64). A cold segment is the common case immediately after prefill, so treat ~1.3% as the
conservative bound rather than the sub-0.1% a warm segment shows.

At `stride=1`, the integrator evaluates one kernel query per decode step and produces an exact
sum. At `stride=1024` (the default), it uses at most 5 queries for a 4096-step segment, a 1024x
reduction in kernel evaluations for a ~1.3% worst-case accuracy cost.

A single-tangent approximation (the degenerate case of `stride >= max_steps`) measures 7.08%
error at batch 16 and 13.68% at batch 64, which is why it is not a supported mode.

## Elementwise ops: measured, deliberately not summed

Rope, rmsnorm, activation, and Qwen3's `qk_norm` each have a perf table (op `elementwise`,
holdout MRE 5.01% over 52 points, fitted floor 2 us, H20 / TP=1 / 2026-08-25). They are
**not** added to `ComposedRollout.predict()`. This is a measurement-backed design decision, not
unfinished work.

Adding the elementwise term makes every phase worse:

| Phase | MRE (current) | MRE (with elementwise) | Median ratio (current) | Median ratio (with it) |
|-------|---------------|------------------------|------------------------|------------------------|
| decode | 4.84% | 11.18% | 1.0119 | 1.0639 |
| prefill | 3.72% | 5.84% | 1.0069 | 1.0305 |
| mixed | 0.48% | 1.89% | 0.9994 | 1.0143 |

Every phase degrades and every median ratio moves *away* from 1.0. A genuinely missing term
would improve at least one regime. The cause is the SOL/utility factorisation: the measured GEMM
and attention latencies were collected from a real server whose elementwise kernels were running
between them. That wall-clock time is already absorbed into the fitted utility values. Adding a
separate elementwise term double-counts it.

`tests/test_kernel_ops_elementwise.py:test_the_composed_model_does_not_add_an_elementwise_term`
pins this behaviour so a future change that wires the term in has to confront the reasoning
rather than silently double-count.

The table exists for two other consumers: the FSDP training backend, which runs without CUDA
graphs and will need an explicit elementwise term, and as the third op family that validates the
`KernelOp` abstraction beyond the two families that originally shaped it.

## Known limitations

The known limitations (mixed steps, the unusable TP=2 decode buckets below batch 32, decode
over-predicted by 13% at small batch, no CPU overhead term, cross-node TP being unmeasured,
cross-hardware transfer unvalidated, and
`context_attention/flashinfer` above the 5% holdout gate) are described with their measured
magnitudes and proposed fixes in {doc}`../reference/limitations`.

## Running validation

To score `ComposedRollout` against a real vLLM profiling baseline:

```bash
dryrun-profile kernels validate \
    --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml \
    --baseline outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
    --model-path $MODEL_PATH \
    --surface pchip \
    --min-samples 1 \
    --gate 100 \
    --output /tmp/validate-pchip.json
```

Pass `--min-samples 1` to score all three phases. At the default `--min-samples 20`, only
decode accumulates enough identical iterations from a typical vLLM run, so prefill and mixed are
dropped with an explicit warning. Score against the `vllm` rows, which are the kernels the server
actually ran. The FlashInfer rows exist to answer what a backend switch would buy, so scoring
against them on a vLLM server compares against a backend that never ran.
