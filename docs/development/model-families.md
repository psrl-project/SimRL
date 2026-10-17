# Model families

A `KernelOp` says *how to price a kernel shape*: it owns the roofline formula, the floor policy,
and the table schema. A `ModelFamily` says *which shapes a layer produces*: it translates a model
architecture into the exact per-rank GEMM shapes, head counts, and logits shape that an inference
engine executes. The two concerns are kept separate so that new hardware backends can add ops
without knowing anything about model architectures, and new architectures can be added without
touching a single op.

---

## The `ModelFamily` protocol

`ModelFamily` is a `runtime_checkable` protocol defined in `srl/profiler/kernels/families.py`.
Every family class must implement three methods:

| Method | Returns |
|---|---|
| `gemm_shapes(model, *, tensor_parallel)` | Per-layer GEMM shapes, excluding the logits projection. |
| `logits_shape(model, *, tensor_parallel)` | The vocabulary projection, which runs once per step. |
| `head_counts(model, *, tensor_parallel)` | `(num_heads, num_kv_heads)` for this rank. |

The input in every case is a `ModelDescriptor`, the frozen architecture snapshot read from
`config.json` at profiling time, plus the TP degree. The family raises `ProfileValidationError`
if any dimension is missing or not divisible by `tensor_parallel`, and it never silently
floor-divides.

---

## `dense_llama`: four fused shapes per layer

The one registered family, `DenseLlamaFamily` (in `srl/profiler/kernels/families_dense.py`),
covers `Qwen3ForCausalLM`, `Qwen2ForCausalLM`, `LlamaForCausalLM`, and `MistralForCausalLM`. It
produces four GEMM shapes per layer:

| Shape name | Description |
|---|---|
| `qkv_proj` | Fused query + key + value projection. |
| `o_proj` | Attention output projection. |
| `gate_up_proj` | Fused gate and up projections (gated MLP). |
| `down_proj` | MLP down projection. |

These four shapes are what the *inference engine* runs, not what the HuggingFace checkpoint
declares. HuggingFace keeps `q_proj`, `k_proj`, and `v_proj` separate and `gate_proj` / `up_proj`
separate, giving seven 2-D weight matrices per layer. vLLM fuses them via `QKVParallelLinear` and
`MergedColumnParallelLinear`, giving four. Using the unfused HuggingFace shapes would query GEMM
measurements at shapes that were never profiled, so the slice lookup would miss and raise. This has
been verified against both local checkpoints by `kernels verify-shapes`: summing HuggingFace's row
counts reproduces the fused `n` values exactly (4096 + 1024 + 1024 = 6144, and 12288 + 12288 =
24576).

Fusing does not change the arithmetic volume. Both groupings come to the same $\sum 2nk$ per token,
3.8588e+08 for Qwen3-8B. What it changes is the launch count, seven against four, so pricing the
unfused shapes would over-count per-call dispatch overhead and place each GEMM at a lower utility,
not inflate the FLOPs.

In addition, `logits_shape` returns the `lm_head` vocabulary projection, which vLLM runs once per
step (not per layer) via `ParallelLMHead`, projecting only each sequence's last token.

---

## Unsupported architectures are refused, not approximated

This is the central design choice of the family system. Before it existed, a mixture-of-experts
config was priced with the dense arithmetic: no router GEMM, no per-expert GEMMs, no error, and a
latency number plausible enough that nothing downstream would question it. A silently wrong latency
propagates into every schedule the simulator produces. A raised error does not.

`family_for(model)` raises `ProfileValidationError` for any architecture not in `FAMILY_REGISTRY`.
The dense family also refuses models carrying disqualifying config keys, even if their architecture
string is registered:

`num_experts`, `num_local_experts`, `n_routed_experts`, `num_experts_per_tok`
: Mixture of experts. Needs a router GEMM and per-expert GEMMs, neither of which dense
  pricing includes.

`kv_lora_rank`, `q_lora_rank`
: Compressed-KV attention (MLA). Has no `(heads + 2*kv_heads)*head_dim` QKV projection
  at all, so every GEMM shape this family derives would be wrong.

Each error message names the specific kernel that would be omitted. `srl/profiler/hf.py`
forwards these keys into `ModelDescriptor.extra` for the sole purpose of making the refusal
reachable. MoE and MLA families are deliberately not implemented: no checkpoint of either type is
available for validation, and an untested family is likely to produce the same kind of
plausible-but-wrong number that the refusal is designed to prevent.

---

## The introspection gate

The four fused shapes are derived from `config.json` arithmetic. Nothing in the fast pricing path
can verify that the arithmetic still matches what a real engine module graph actually runs. The
`kernels verify-shapes` subcommand is the oracle that can:

```bash
srl-profile kernels verify-shapes --model-path $MODEL_PATH
```

It builds the real model on the **meta device**, with no weights and no GPU memory, enumerates every 2-D
weight, applies the fusion rules in `_FUSION_RULES`, and compares the resulting shapes against the
family's declared shapes. For Qwen3-8B this takes about 3 seconds and uses 0 MiB of VRAM. The
current result on both local checkpoints is: all five shapes match, with 36 and 28 homogeneous
layers respectively.

The command also asserts that every layer fuses to the *same* shape set. No config key reliably
signals layer heterogeneity. Because `dense_llama` multiplies one layer's shapes by the layer
count, a heterogeneous model priced as though every layer matched the first would produce a
systematically wrong latency with no visible error. `introspect.py` imports `torch` and
`transformers`, so it is called only from the CLI, not from the cost path, which stays torch-free.

### What the gate does not cover

Three limits are worth knowing before treating a passing run as proof.

**It compares against HuggingFace, not against the engine.** `_FUSION_RULES` is a hard-coded map
(`qkv_proj` from `q_proj`/`k_proj`/`v_proj`, `gate_up_proj` from `gate_proj`/`up_proj`). The
declared side and the observed side therefore share the same assumption about how vLLM fuses. If
vLLM stopped fusing QKV, this gate would still pass while the pricing became wrong. What the gate
independently verifies is that the `config.json` arithmetic matches the checkpoint's real tensors,
and that the layers are homogeneous.

**It is manual.** Nothing in `kernels run`, `collect`, or the simulator invokes it. It runs when
you run it.

**It supports `tensor_parallel=1` only.** The module graph is unsharded, so comparing it against
sharded shapes would report a mismatch on every entry, and the function raises rather than
pretending otherwise.

---

## Adding a family

Three steps are required.

**Step 1.** Create `srl/profiler/kernels/families_<name>.py` with a class satisfying the
`ModelFamily` protocol. The class must implement `gemm_shapes`, `logits_shape`, and `head_counts`.
If layers are heterogeneous, return `(shape, count)` pairs rather than one shape per layer.
`ModelOps.gemm_queries` already handles `(query, multiplier)` pairs, so this is a protocol
widening, not a break.

**Step 2.** For each architecture string the family covers, call `register_family(architecture,
family_instance)` at module level. Registering the same architecture twice raises
`ProfileValidationError` rather than silently overwriting.

**Step 3.** Import the new module at the bottom of `srl/profiler/kernels/families.py`.

The last step is not optional and not obvious. `family_for` is called from the cost path, which
has no reason to know which specific family modules exist. Registration happens as a side effect of
importing the module. If the import is omitted from `families.py`, a fresh interpreter sees an
empty registry and `family_for` raises `Registered: none`, while every unit test passes, because
each test imports the family module it exercises. The `families.py` tail comment documents this
trap and shows the existing pattern for the dense family.
