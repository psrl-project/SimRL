# Prior art and comparisons

SimRL's kernel cost model was designed by reading two existing simulators, aiconfigurator and
Frontier, before committing to a design. This page records what was taken from them, what is
genuinely different, and which papers the other cost models follow.

The short version: **on the decision that matters most, SimRL and aiconfigurator made the same
choice, and Frontier made the opposite one.** The design decision to interpolate a
SOL-normalised utility rather than a latency was adopted from aiconfigurator deliberately rather
than arrived at independently.

## The decision that splits the three

All three predict one kernel's latency from a table of measurements. They differ in what the table
stores and what gets interpolated.

| | aiconfigurator | Frontier | SimRL |
|---|---|---|---|
| table holds | measured `(shape -> latency)` | measured `(shape -> latency)` | measured `(shape -> latency)` |
| interpolates | SOL-normalised utility | latency, via a random forest | SOL-normalised utility |
| query | log2-space kNN with modified-Shepard weights | random-forest regression with grid-searched hyper-parameters | log2-space interpolation; Shepard by default, monotone cubic splines where the grid allows |
| past the measured range | hold the boundary utility | forest extrapolation | hold the boundary utility |
| cross-hardware | in principle, through the target GPU's SOL | retrain per device | in principle, through the target GPU's SOL; not yet validated |
| source of shapes | analytic from the config | per-model profiling CSVs | analytic from the config, dispatched by an architecture family |

### Why utility rather than latency

A measured kernel latency spans orders of magnitude across shapes, while its efficiency against
the roofline spans a small factor. GEMM latency grows by orders of magnitude as `m` goes from one
token to tens of thousands, yet the fraction of the roofline it achieves moves by a small factor.
A regressor fitted on latency carries the whole dynamic range; interpolating utility leaves the
orders of magnitude to the analytic roofline and asks the surface only how efficient the kernel is.

The trade-off is that the SOL formula must be right, because an error in the roofline propagates
into every utility and therefore into every prediction. It is accurate for GEMM on the reference
hardware. For attention at small batch sizes a pure roofline understates cost by an order of
magnitude, which is what motivated the fitted dispatch floor below.

### What Frontier does instead

Frontier keeps no `(m, n, k)` table. Its profiling data is a wide CSV per device and model, with
one row per token count and one column group per semantic operator (attention pre-projection, MLP
up-projection, and so on). `num_tokens` is the only feature; model size, head count and tensor
degree identify the dataset rather than the features, and the estimator is a random forest fitted
on absolute latency. It also measures something neither of the others does: a CPU-overhead table
keyed on model, batch size, tensor degree, prefill and decode token counts and scheduling mode.

## What is actually different about SimRL

### A dispatch floor inside the max

Attention, elementwise, collective, gated-delta-net and optimizer kernels carry a per-call cost
the roofline does not express. SimRL fits one per backend and op and evaluates

```text
latency = max(compute, memory, floor)
```

The floor goes *inside* the max rather than being added, because adding it double-counts in the
regime where it binds. The evidence is direct: decode attention latency is flat across a large
increase in KV depth at small batch. `kernels fit-floor` fits the value per `(op, backend,
version)` and stores it in the version directory's `floors.json`; a missing file falls back to a
packaged map with a warning.

aiconfigurator has launch floors in its communication and MoE tables but not this mechanism for
attention. Frontier's per-operator calibration scales are a multiplicative correction, which
cannot reproduce a flat region.

### Shape derivation behind an architecture family

Both reference simulators derive their per-layer operator list analytically from `config.json`.
SimRL does the same, but through a `ModelFamily` chosen by the checkpoint's `architectures` entry
(`srl/profiler/kernel/grid/families_dense.py`, `families_moe.py`). An architecture with no
registered family is **refused**, because pricing an unsupported block with another family's
arithmetic produces a plausible number for kernels the model never runs.

Registered today:

- **Dense Llama-style blocks** — `families_dense.py`: four fused projections plus the vocabulary
  projection, matching what an inference engine runs rather than HuggingFace's seven separate
  weights.
- **Qwen3.5-MoE** — `families_moe.py`: a router projection, per-token routed expert GEMMs at the
  MoE width scaled by `num_experts_per_tok`, an optional shared expert, hybrid full-attention and
  linear-attention layers, and `attn_output_gate`, which doubles the query rows of the fused QKV
  projection. Each of those changes the measured shape set, which is why the block cannot reuse
  the dense arithmetic.

`kernels verify-shapes` is an optional self-check the other two do not have: it builds the model
on the meta device, enumerates every 2-D weight, applies the engine's fusion rules, and compares
against the declared shapes. The two sides are independent — one is arithmetic over the config,
the other a real tensor's shape — so it catches an arithmetic error and it catches layer
heterogeneity, which no config key reliably reports. It does not observe the engine: the fusion
rules are hard-coded, so it confirms that HuggingFace's separate projections sum to the declared
shapes, not that the engine fuses them. It is not wired into `kernels run`; nothing invokes it
unless you do, and `kernels run` reads the family arithmetic directly.

Families are also why the simulator covers the hybrid-attention and gated-delta-net kernels
(`gdn`), the fused optimizer step (`optimizer`), and the training-attention family
(`attention_training`) that prices Transformer Engine's THD layout rather than vLLM's prefill
kernel.

### Two collective tables

Which kernel a real server runs is not NCCL. The collective sweep therefore writes one table
through vLLM's own dispatcher and one through `torch.distributed` as a control, so the cost of
having chosen the wrong one is a number rather than an argument. This is covered on
{doc}`../profiling/tensor-parallel`.

## What is still refused

**Compressed-KV / MLA attention.** A checkpoint carrying `kv_lora_rank` or `q_lora_rank` is
refused by the dense family, because that block has no `(heads + 2 * kv_heads) * head_dim` qkv
projection and its cache is compressed. Both reference simulators cover MLA and SimRL does not:
this is a capability gap, not a design win. Closing it needs a checkpoint to validate against, and
a synthetic implementation would only test itself.

Refusing is honest here. An earlier "MoE is four dense GEMMs" approximation passed every unit
test because the tests built their own synthetic shapes, and the error stayed invisible until a
real module graph was compared against the declared one. The registered MoE family is the result
of doing that comparison rather than of guessing.

## How the shapes and the sweep relate

`(n, k)` is fully determined by the architecture and is looked up by exact match; only `m` is
interpolated. The consequences:

- A query at an unmeasured `(n, k)` raises `KernelCoverageError` rather than clamping to a
  neighbouring projection's curve. Per-`(n, k)` slicing exists precisely because blending two
  projections into one slice scored worse.
- An `m` past the measured frontier holds the boundary utility, because extrapolating an
  efficiency is defensible and extrapolating a latency is not.
- A different model changes every `(n, k)`; a different TP degree does too, since shapes are
  sharded at profiling time. That is what the per-degree kernel trees and the simulation-side
  guard on `KERNEL_ROOT` exist for.
- Forward, dgrad, wgrad and logits shapes all land in the one `gemm` table, which is what lets a
  single surface price both an inference and a training step.

A sweep therefore needs only a few dozen token counts per projection, not a cross product over
everything a reference simulator may cover across many models and quantisation modes.

### A note on fused shapes

vLLM fuses `q_proj`, `k_proj`, `v_proj` into one QKV projection and `gate_proj` with `up_proj`
into one merged column projection, so a layer runs fewer, wider GEMMs than HuggingFace declares.
The FLOPs are identical; the launch count is not. Pricing the unfused shapes would over-count
dispatch and understate each GEMM's efficiency, which is why the declared shapes are the
engine's.

## Training cost decomposition: Hydraulis

Neither reference simulator models training, so this half of the project has no prior art to copy.

The training latency model follows Appendix C of Hydraulis
([arXiv:2412.07894](https://arxiv.org/abs/2412.07894)), which decomposes a Megatron pipeline step
into a communication-aware sum over microbatches. SimRL restructures it into four separately named
terms — per-step overhead, per-token cost, per-microbatch overhead, and a quadratic attention
term — with the token terms summed over the microbatches a stage runs plus a pipeline bubble, and
scaled by the stage share. The FSDP model is the same row with the pipeline degenerating to
`slots = num_microbatches`. See {doc}`training-megatron` and {doc}`training-fsdp`.

## Rollout cost structure: DistServe

The DistServe rollout model ([arXiv:2401.09670](https://arxiv.org/abs/2401.09670)) separates
prefill and decode attention costs explicitly and calibrates effective hardware constants from
measured profiles rather than using vendor peaks. `srl/cost/rollout/distserve.py` is that
calibration. The `roofline` model generalises the structure to one formula covering mixed batches,
by summing a prefill term and a decode term, which the original formulation does not cover. See
{doc}`rollout`.

## Composed models and the prior art

The composed approach — sum measured kernels rather than fit an end-to-end curve — is the
alternative to all of the above and is what lets SimRL price a new model or degree without a
server sweep. Its lineage is the same kernel-table idea as aiconfigurator, with the FP32/utility
normalisation kept and the whole engine iteration assembled from it. See
{doc}`composed-rollout` and {doc}`../profiling/kernels`.
