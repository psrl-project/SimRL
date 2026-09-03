# Prior art and comparisons

DryRun-RL's kernel cost model was designed by reading two existing tools — aiconfigurator
and Frontier — before committing to a design. This page documents what was taken from
them, where the design deliberately diverges, and which systems influenced the cost model
formulas.

---

## Kernel cost models: aiconfigurator and Frontier

Both tools derive per-layer operation lists analytically from a model's `config.json`
rather than from an introspected module graph. That approach is fast and does not require
loading weights, but it carries a risk: if the operator fusion strategy changes between
framework versions, or if a model's `config.json` does not carry a flag that signals an
unusual architecture, the declared op list silently diverges from what the engine
actually runs.

The comparison table from `docs/kernel-ops.md`:

| Feature | aiconfigurator | Frontier | DryRun-RL |
|---|---|---|---|
| op list | one class per family, hand-written | declarative families via flags | one family per arch group |
| dispatch | `ARCHITECTURE_TO_MODEL_FAMILY` | `is_moe` / `model_arch` flags | `ModelDescriptor.architecture` |
| MoE | router GEMM + dispatch/combine | `gating + topk + grouped_gemm` | **refused** |
| MLA | `ContextMLA` / `GenerationMLA` ops | `LATENT_MLA_ATTENTION_FAMILY` | **refused** |
| heterogeneous layers | one op per variant with a count | per-architecture profile | **refused** (gate detects it) |
| unsupported arch | no family registered → raises | falls back to a generic profile | raises |
| shape validation | none | none | `kernels verify-shapes` |
| per-op cost | analytical table lookup | `RandomForest` on profiled latencies | roofline SOL / interpolated utility |

---

## Two deliberate divergences

### Refusing MoE and MLA rather than approximating

Neither MoE nor MLA is supported. A checkpoint carrying `num_experts`, `kv_lora_rank`,
or related keys in `config.json` is rejected at configuration time with a message naming
the key and the kernel that would have been omitted.

This is a capability gap relative to both alternatives, and it is an honestly large one:
mixture-of-experts models are common in current production deployments, and multi-head
latent attention appears in DeepSeek-style models. Closing these gaps requires a
checkpoint to validate the pricing against. Without one, any MoE or MLA family could
only be tested against synthetic descriptors — which is how a previous "MoE is four
dense GEMMs" approximation became an uncharacterised error: it passed all unit tests
because the tests constructed their own synthetic shapes.

The rule is: refusing is honest; guessing is not.

### Learning utility rather than latency

Frontier trains a `RandomForestRegressor` directly on measured latencies. Across a
kernel sweep, those latencies span roughly 300x — from a few microseconds for a small
all-reduce to milliseconds for a large prefill attention. The regressor must carry that
entire dynamic range.

DryRun-RL instead learns *utility* — the dimensionless ratio `SOL / measured_latency` —
and lets the analytical speed-of-light formula carry every order of magnitude. Utility
spans roughly 2x for GEMM (0.47 to 0.95 on H20), making it far easier to interpolate
accurately with a simple surface. The measured MRE figures for this approach are in the
3–10% range across all five ops; see `docs/kernel-ops.md` for the per-op breakdown.

The trade-off is that the SOL formula must be correct: an error in the roofline
propagates into every utility value and therefore into every prediction. For GEMM on
H20 the roofline is accurate enough that this is not a practical issue. For attention
at small batch sizes, the roofline alone understates cost by a factor of 10 or more —
which is what motivated the fitted dispatch floor (see {doc}`../getting-started/concepts`
under *SOL, utility, and floor*).

### Shape validation

Neither aiconfigurator nor Frontier validates that its declared shapes match the model
the engine actually runs. DryRun-RL provides `kernels verify-shapes`, which builds the
model on the meta device, enumerates every 2-D weight module, applies the engine's
fusion rules, and compares against the declared shapes. For Qwen3-8B, this revealed
that the engine fuses `q_proj`, `k_proj`, `v_proj` into one `QKVParallelLinear` and
`gate_proj` + `up_proj` into one `MergedColumnParallelLinear`, giving four GEMM shapes
per layer rather than the seven HuggingFace declares. Without the gate, a shape list
derived from `config.json` alone would overcount layers by 75%.

---

## Training cost decomposition: Hydraulis

The training latency model follows Appendix C of Hydraulis
([arXiv:2412.07894](https://arxiv.org/abs/2412.07894)), which decomposes a Megatron
pipeline step into a communication-aware sum over microbatches:

$$\text{latency} = a_Q \cdot Q_\text{eff} + b_T \cdot T_\text{eff}
    + c_\text{pipeline} \cdot (\text{num\_microbatches} + PP - 1) + \text{update\_overhead}$$

DryRun-RL fits this form against Megatron-Bridge profiling records. The FSDP latency
model uses a structurally similar form (`per_step_overhead + slots * (token_cost *
tokens + microbatch_overhead)`) because the same physical decomposition applies to a
non-pipelined backend: every term names a real cost. See {doc}`../cost-models/training-megatron`
and {doc}`../cost-models/training-fsdp` for the fitted coefficients and accuracy.

## Rollout cost structure: DistServe

The DistServe rollout model
([arXiv:2401.09670](https://arxiv.org/abs/2401.09670)) separates prefill and decode
attention costs explicitly. DryRun-RL's roofline model generalises this structure into
a single formula that handles mixed batches by summing `sum_l2` (prefill) and `ctxsum`
(decode) terms, while retaining the DistServe calibration approach of fitting effective
hardware constants from measured profiles rather than using vendor-published peaks.

The `distserve` cost model in `dryrun/cost/rollout/distserve.py` is DryRun-RL's
implementation of the DistServe calibration. The `roofline` model is a generalisation
that handles mixed-phase batches, which the original DistServe formulation does not
cover. See {doc}`../cost-models/rollout` for the formula and fitting details.
