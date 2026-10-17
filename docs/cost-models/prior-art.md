# Prior art and comparisons

SimRL's kernel cost model was designed by reading two existing simulators, aiconfigurator
and Frontier, before committing to a design. This page records what was taken from them, what
is genuinely different, and which papers the other cost models follow.

The short version: **on the decision that matters most, SimRL and aiconfigurator made the
same choice, and Frontier made the opposite one.** The design document
(`docs/superpowers/specs/2026-08-24-kernel-cost-model-design.md`, decision D2) adopted
aiconfigurator's approach deliberately rather than arriving at it independently.

---

## The decision that splits the three

All three predict one kernel's latency from a table of measurements. They differ in *what the
table stores* and *what gets interpolated*.

| | aiconfigurator | Frontier | SimRL |
|---|---|---|---|
| table holds | raw `(shape → latency)` | raw `(shape → latency)` | raw `(shape → latency)` |
| interpolates | **SOL-normalised utility** | latency, via RandomForest | **SOL-normalised utility** |
| query math | log2-space kNN, modified-Shepard weights | `RandomForestRegressor` + `GridSearchCV` | log2-space kNN, modified-Shepard weights |
| past the measured range | hold boundary utility, `latency = SOL/util` | RF extrapolation | hold boundary utility, `latency = SOL/util` |
| cross-hardware | yes in principle, via the target GPU's SOL | no, retrain per device | untested (H20 only) |

aiconfigurator's own source states this. From
`aic-core/rust/aiconfigurator-core/src/perf_database/perf_interp.rs`:

```text
3. beyond the range -> hold a boundary util, latency = SOL(query)/util.
   Multi-axis grids transfer the util from the nn_leaves nearest collected
   points in joint log2 space with tapered (modified-Shepard) weights
```

Utility, log2 spacing, Shepard tapering, and the boundary hold are all shared. SimRL's
`srl/cost/kernel/interp.py` is the same construction.

### Why utility rather than latency

Measured on the Qwen3-8B GEMM table (134 rows, H20), **latency spans 2268x while utility spans
4.65x**. Within one slice the gap is starker. For `(n=4096, k=4096)`, the attention output
projection:

| m | measured | SOL | utility | bound |
|---|---|---|---|---|
| 1 | 1.150e-05 | 8.393e-06 | 0.730 | memory |
| 8 | 1.050e-05 | 8.421e-06 | 0.802 | memory |
| 32 | 1.345e-05 | 8.520e-06 | 0.633 | memory |
| 64 | 2.620e-05 | 1.451e-05 | 0.554 | compute |
| 256 | 6.957e-05 | 5.804e-05 | 0.834 | compute |
| 2048 | 5.062e-04 | 4.643e-04 | 0.917 | compute |
| 24576 | 5.940e-03 | 5.572e-03 | 0.938 | compute |

Latency moves 566x across that column and utility moves 1.69x. A regressor fitted on latency
carries the whole dynamic range. Interpolating utility leaves the orders of magnitude to the
analytic roofline and asks the surface only *how efficient* the kernel is.

The trade-off is that the SOL formula must be right. An error in the roofline propagates into
every utility and therefore into every prediction. For GEMM on H20 it is accurate enough. For
attention at small batch sizes the pure roofline understates cost by more than 10x, which is
what motivated the fitted dispatch floor.

### What Frontier does instead

Frontier has no `(m, n, k)` table. Its profiling data is a wide CSV per `(device, model)` with
one row per `num_tokens` and one column group per *semantic* operator
(`time_stats.attn_pre_proj.median`, `time_stats.mlp_up_proj.median`, and so on). The training
call in `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` is:

```python
models[model_name] = self._train_model(
    model_name=model_name,
    df=compute_df,
    feature_cols=["num_tokens"],
    target_col=target_col,
)
```

`num_tokens` is the only feature. `n_embd`, `n_head`, and `num_tensor_parallel_workers` are not
features but the **identity of the dataset**: a different TP degree or model is a different CSV
and a different forest. The estimator is a `RandomForestRegressor` tuned by `GridSearchCV`, and
it fits absolute latency. It deliberately keeps no test split:

> we don't create a train/test split, because we want to use all data for training and we don't
> care about overfitting, because we only want to predict execution time within the same domain

Frontier compensates with per-operator `calibration_scale` constants, and it measures something
neither of the others does: a CPU-overhead table keyed on
`(model, batch_size, tp, num_prefill_tokens, num_decode_tokens, scheduling_mode)`.

---

## What is actually different about SimRL

Three things, none of which is the utility idea.

### A dispatch floor inside the max

Attention, elementwise, and collective ops carry a per-call cost the roofline does not express.
SimRL fits one per backend and op and evaluates
$\text{latency} = \max(\text{compute}, \text{memory}, \text{floor})$.

The floor goes *inside* the max rather than being added, because adding it would double-count in
the regime where it binds. The evidence is direct: at batch 1, decode attention latency stays
near 70 µs across a 64x increase in KV depth. Without the floor, generation attention scores
34.89% MRE against 6.63% with it.

aiconfigurator has launch floors in its communication and MoE tables, but not this mechanism for
attention. Frontier's per-operator calibration scales are a multiplicative correction, which
cannot reproduce a flat region.

### `kernels verify-shapes`

All three derive their per-layer operator list analytically from `config.json`. SimRL ships
an **optional** self-check that the other two do not have: build the model on the meta device,
enumerate every 2-D weight, apply the engine's fusion rules, and compare against the declared
shapes. On Qwen3-8B it costs 3.09 s and 0 MiB of VRAM.

```text
Qwen3ForCausalLM -> family 'dense_llama', 36 layers.
  down_proj:    declared (4096, 12288), observed (4096, 12288) OK.
  gate_up_proj: declared (24576, 4096), observed (24576, 4096) OK.
  lm_head:      declared (151936, 4096), observed (151936, 4096) OK.
  o_proj:       declared (4096, 4096),  observed (4096, 4096)  OK.
  qkv_proj:     declared (6144, 4096),  observed (6144, 4096)  OK.
```

Be precise about what this proves. The two sides are independent: `declared` is arithmetic over
`config.json`, `observed` is a real tensor's `weight.shape`. So it catches arithmetic that
disagrees with the checkpoint, and it catches **layer heterogeneity**, which no config key
reliably reports and which the dense family would otherwise flatten into one shape times
`num_layers`.

What it does not do:

- It does not observe the engine. The fusion rules in `_FUSION_RULES` are hard-coded, so the
  check confirms that HuggingFace's separate projections *sum to* the shapes we declare. If vLLM
  stopped fusing QKV, this gate would still pass and the pricing would still be wrong.
- It is not wired into `kernels run`. Nothing invokes it unless you do.
- It supports `tensor_parallel=1` only, since the module graph is unsharded.

It is a useful instrument, not an architectural advantage. The pricing path is hard-coded
arithmetic over `config.json`, exactly as in the other two.

### Refusing MoE and MLA

Neither is supported. A checkpoint carrying `num_experts`, `kv_lora_rank`, or a related key is
rejected at configuration time with a message naming the key and the kernel that would have been
omitted.

**This is a capability gap, not a design win.** aiconfigurator has `ContextMLA`,
`GenerationMLA`, `MoE`, and `MoEDispatch` operations, and Frontier has a
`LATENT_MLA_ATTENTION_FAMILY` plus an `is_moe` path. Both cover architectures we cannot price.

Closing the gap needs a checkpoint to validate against. Without one, an MoE family could only be
tested against synthetic descriptors, which is how a previous "MoE is four dense GEMMs"
approximation became an uncharacterised error: it passed every unit test, because the tests
built their own synthetic shapes. Refusing is honest. Guessing is not.

---

## How the shapes and the sweep relate

Worth stating plainly, because it determines when a sweep has to be repeated.

`(n, k)` is **fully determined by the architecture**, and it is looked up by exact match. For
Qwen3-8B at TP=1, `srl/profiler/kernels/families_dense.py` computes five pairs:

| shape | n | k |
|---|---|---|
| `qkv_proj` | `(32 + 2*8) * 128` = 6144 | 4096 |
| `o_proj` | 4096 | `32 * 128` = 4096 |
| `gate_up_proj` | `2 * 12288` = 24576 | 4096 |
| `down_proj` | 4096 | 12288 |
| `lm_head` | 151936 | 4096 |

The sweep crosses those with the configured `token_counts`, adds the wgrad axis permutations and
the logits row counts, and deduplicates: 134 triples in 35 slices. At query time `ModelOps` emits
`GemmQuery(m, n, k)` from the *same* function, so a slice always hits.

Only `m` is interpolated. The consequences, all measured against the shipped table:

```text
slice (4096, 4096) measured, m=1500 never measured  -> 3.724e-04 s, utility 0.9131
slice (5000, 4096) never measured, any m            -> KernelCoverageError
slice (4096, 4096), m=99999 past the measured 24576 -> 2.419e-02 s, utility 0.9372
```

An unmeasured `(n, k)` raises. An `m` past the frontier holds the boundary utility, because
extrapolating an efficiency is defensible while extrapolating a latency is not.

So `m` is the only cheap axis. A dozen token counts per slice cover any batch shape on that
projection. Everything else forces a re-sweep: **a different model changes every `(n, k)`, and a
different TP degree does too**, since the shapes are sharded at profiling time. That is what the
per-degree kernel trees (`kernel_tp1`, `kernel_tp2`, ...) and the guards in
`examples/simulate/simulate.sh` exist for.

This is also why the table is 134 rows where aiconfigurator's GEMM table is about 107,000. Not
frugality: the table only ever needs this model's five `(n, k)` pairs, while aiconfigurator
covers many models and quantisation modes.

### A note on fused shapes

vLLM fuses `q_proj`, `k_proj`, `v_proj` into one `QKVParallelLinear` and `gate_proj` + `up_proj`
into one `MergedColumnParallelLinear`, so a layer runs four GEMMs where HuggingFace declares
seven 2-D weights.

The **FLOPs are identical** either way: $\sum 2nk$ comes to 3.8588e+08 per token for both
groupings, because 4096 + 1024 + 1024 = 6144 rows and 12288 + 12288 = 24576 rows. What changes is
the **launch count**, 7 against 4, a factor of 1.75. That matters because each call pays its own
per-call overhead and because a narrower GEMM sits at lower utility, not because the arithmetic
volume differs. Pricing the unfused shapes would over-count dispatch, not work.

---

## Training cost decomposition: Hydraulis

Neither reference models training at all, so this half of the project has no prior art to copy.

The training latency model follows Appendix C of Hydraulis
([arXiv:2412.07894](https://arxiv.org/abs/2412.07894)), which decomposes a Megatron pipeline step
into a communication-aware sum over microbatches:

$$\text{latency} = a_Q \cdot Q_\text{eff} + b_T \cdot T_\text{eff}
    + c_\text{pipeline} \cdot (\text{num\_microbatches} + PP - 1) + \text{update\_overhead}$$

SimRL restructures this into four separately identifiable terms and fits it against
Megatron-Bridge records. The FSDP model shares the same form with `slots = num_microbatches`.
See {doc}`training-megatron` and {doc}`training-fsdp`.

## Rollout cost structure: DistServe

The DistServe rollout model ([arXiv:2401.09670](https://arxiv.org/abs/2401.09670)) separates
prefill and decode attention costs explicitly. SimRL's roofline model generalises that
structure into one formula covering mixed batches by summing a `sum_l2` term (prefill) and a
`ctxsum` term (decode), while keeping DistServe's approach of fitting effective hardware
constants from measured profiles rather than using vendor peaks.

`srl/cost/rollout/distserve.py` is the DistServe calibration itself. The `roofline` model is
the mixed-phase generalisation, which the original formulation does not cover. See
{doc}`rollout`.
