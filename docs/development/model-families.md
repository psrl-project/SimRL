# Model families

A `KernelOp` says how to price one kernel shape. A `ModelFamily` says which shapes one layer of a
model produces. The two are separate so a new architecture never touches a kernel op and a new
backend never touches architecture arithmetic.

The protocol lives in `srl/profiler/kernel/grid/families.py`. The families live beside it in
`families_dense.py` and `families_moe.py`.

---

## The `ModelFamily` protocol

`ModelFamily` is a `runtime_checkable` protocol.

| Method | Returns |
|---|---|
| `gemm_shapes(model, *, tensor_parallel)` | One layer's per-rank GEMM shapes, excluding the logits projection |
| `logits_shape(model, *, tensor_parallel)` | The vocabulary projection, which runs once per step |
| `head_counts(model, *, tensor_parallel)` | This rank's `(num_heads, num_kv_heads)` |

The input is the `ModelDescriptor` read from `config.json`, plus the TP degree. A family raises
`ProfileValidationError` for a missing dimension or one not divisible by `tensor_parallel`. It never
silently floor-divides.

`family_for(model)` dispatches on `model.architecture`, which is `config.json`'s `architectures[0]`.
The registry is populated at the bottom of `families.py`, which imports each family module, so
importing the module is enough.

---

## Registered architectures

`dense_llama` (`DenseLlamaFamily`) covers `Qwen3ForCausalLM`, `Qwen2ForCausalLM`,
`LlamaForCausalLM` and `MistralForCausalLM`. It produces four fused projections per layer:

| Shape | Content |
|---|---|
| `qkv_proj` | Fused query, key and value |
| `o_proj` | Attention output |
| `gate_up_proj` | Fused gate and up projections |
| `down_proj` | MLP down projection |

These are the serving engine's shapes, not the checkpoint's. HuggingFace keeps `q_proj`, `k_proj`
and `v_proj` separate and `gate_proj` and `up_proj` separate, seven 2-D weights per layer, while
vLLM fuses them into `QKVParallelLinear` and `MergedColumnParallelLinear`, four. `logits_shape`
returns the `lm_head` projection, which the engine runs once per step over each sequence's last
token.

`dense_llama` refuses a checkpoint carrying any of `num_experts`, `num_local_experts`,
`n_routed_experts`, `num_experts_per_tok`, `kv_lora_rank` or `q_lora_rank`, even when its
architecture string is registered. Each refusal names the kernel that would have been omitted. A
mixture-of-experts block priced as dense loses its router GEMM and its per-expert GEMMs and still
returns a plausible number. A compressed-KV block has no `(heads + 2 * kv_heads) * head_dim` qkv
projection at all.

`qwen35_moe` (`Qwen35MoeFamily`) covers `Qwen3_5MoeForConditionalGeneration` and
`Qwen3_5MoeForCausalLM`. It prices the attention stack of one full-attention layer, the router
projection, the per-expert gated pair at `moe_intermediate_size` with `n` scaled by
`num_experts_per_tok`, and the shared expert. `n_full_attention_layers` reports how many layers
that describes. Layers on the linear-attention interval are not priced as if they were attention:
their projections have a different shape set and the kernel table has no SSM scan family.

An architecture with no registered family is refused by `family_for` with the registered list in
the message.

---

## The introspection gate

The fused shapes are arithmetic over `config.json`, and nothing in the cost path can falsify it.
`srl-profile kernels verify-shapes` is the oracle:

```bash
srl-profile kernels verify-shapes --model-path "$MODEL_PATH"
srl-profile kernels verify-shapes --config CFG      # model from the config
```

`verify_shapes` builds the real model on the **meta device**, so no weights are allocated and no GPU
is touched, enumerates every 2-D weight and 3-D expert stack, applies the engine's fusion rules,
and compares them with the family's declared shapes. It prints one line per shape and exits non-zero
on a mismatch. It also reports any layer whose shape set differs from layer 0's, because a family
multiplies one layer's shapes by the layer count and a heterogeneous stack would be priced as though
every layer matched the first.

Three limits:

- **It compares against HuggingFace, not the engine.** `_FUSION_RULES` is a hand-written map, so the
  declared and observed sides share its assumption. The gate verifies that the `config.json`
  arithmetic matches the checkpoint's tensors and that the layers are homogeneous. It cannot notice
  that an engine stopped fusing.
- **It is manual.** No profiling run, collection step or simulation invokes it.
- **It supports `tensor_parallel=1` only.** The module graph is unsharded, so comparing it with
  sharded shapes would report a mismatch at every entry, and the function raises rather than
  pretending otherwise.

---

## Adding a family

**Step 1.** Create `srl/profiler/kernel/grid/families_<name>.py` with a class satisfying the
protocol. Return `GemmShape` values rather than plain tuples, so pricing and the introspection gate
agree about which dimension is `n` and which is `k`.

**Step 2.** Register the class for each architecture string it covers with
`register_family(architecture, instance)`. Registering the same architecture twice raises
`ProfileValidationError` rather than overwriting.

**Step 3.** Import the new module from `srl/profiler/kernel/grid/families.py`. That import is the
only registration path. If it is missing, `family_for` raises with an empty registry in a fresh
interpreter while every unit test passes, because each test imports the family module it exercises.

**Step 4.** Keep the module free of `torch` and vLLM. `tests/test_kernel_package_imports.py`
imports every module under `grid/` in its own subprocess, and the cost path imports `grid.gemm`
to build its queries.

A new architecture that runs a block shape already covered should be registered against the
existing family, not given a new class. A model whose block differs, in attention layout, MLP
structure or layer heterogeneity, needs its own family and a `verify-shapes` run against a real
checkpoint before its prices mean anything.
