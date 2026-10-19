# Kernel ops

A measured kernel family is one class implementing the `KernelOp` protocol in
`srl/cost/kernel/ops/base.py`. The class owns the perf-table schema, the speed-of-light formula,
the floor policy, the sweep parser and the GPU measurement. The rest of the pipeline, the floor
fitter, the holdout gate, the table writer, the database and the CLI, discovers ops through one
registry. Adding an op means writing one class and registering it.

For accuracy figures see {doc}`../reference/validation`. For the slice, axis, utility and floor
vocabulary see {doc}`../getting-started/concepts` and {doc}`../cost-models/prior-art`.

---

## Why an op is a class

`KernelOp.sol()` is the single binding from `(spec, slice_key, coords, floor_s)` to a roofline
cost. Utility is `SOL / measured latency`, and the holdout gate computes utility on both sides of
its round trip, so two copies of that binding could drift apart and silently change the measured
interpolation accuracy. There is one copy.

Families have different signatures. Each op unpacks its own `slice_key`, whose column order it
also declares, so no shared keyword bag is needed. `gemm` reads `(gemm_dtype, n, k)`, and
`all_reduce` reads `(dtype, world_size, nodes)`.

The protocol requires:

| Member | Purpose |
|---|---|
| `name` | Op name. Matches the parquet file stem |
| `family` | Directory that holds the table, e.g. `attention` |
| `slice_columns` | Columns matched exactly when a query is priced |
| `axis_columns` | Columns interpolated in log2 space |
| `fits_floor` | Whether the op has a per-backend dispatch floor |
| `graph_replayed` | Whether the serving engine replays this op from a captured CUDA graph |
| `sol(spec, *, slice_key, coords, floor_s)` | Roofline components for one shape |
| `parse_sweep(value)` | Parse the op's YAML sweep subtree |
| `table_columns()` | Non-provenance columns, latency last |
| `measure(model, config, *, writer_factory)` | Measure the family on the GPU and write tables |

`graph_replayed` is per op, not per family. vLLM's full-and-piecewise CUDA graph mode replays a
whole decode forward as one graph while prefill and mixed batches run eagerly for the ops outside
the graph. So `generation_attention` and `gdn_decode` are replayed, and `context_attention` and
`gdn_context` are not. Timing a replayed op with CUDA events charges the simulator a per-call host
cost the engine never pays. Timing an eager op by graph replay hides one it does pay.

---

## The registered ops

`ops/__init__.py` imports every op module, which is what populates `OP_REGISTRY`.

| Op | Family | Slice columns | Axis columns | Floor | Graph replay |
|---|---|---|---|---|---|
| `gemm` | `gemm` | `gemm_dtype, n, k` | `m` | no | no |
| `context_attention` | `attention` | `num_key_value_heads, head_dim, attn_dtype, kv_cache_dtype, block_size, window_size` | `num_heads, batch_size, isl` | fitted | no |
| `generation_attention` | `attention` | the same six | `num_heads, batch_size, step` | fitted | yes |
| `training_attention_forward` | `attention_training` | `num_key_value_heads, head_dim, attn_dtype, attn_mask_type` | `num_heads, total_tokens` | fitted | no |
| `training_attention_backward` | `attention_training` | the same four | `num_heads, total_tokens` | fitted | no |
| `elementwise` | `elementwise` | `dtype, op_kind` | `n_elements` | fitted | no |
| `all_reduce` | `collective` | `dtype, world_size, nodes` | `message_bytes` | fitted | no |
| `reduce_scatter` | `collective` | `dtype, world_size, nodes` | `message_bytes` | fitted | no |
| `all_gather` | `collective` | `dtype, world_size, nodes` | `message_bytes` | fitted | no |
| `broadcast` | `collective` | `dtype, world_size, nodes` | `message_bytes` | fitted | no |
| `p2p` | `collective` | `dtype, world_size, nodes` | `message_bytes` | fitted | no |
| `gdn_context` | `gdn` | `num_key_heads, num_value_heads, head_dim, attn_dtype, chunk_size` | `total_tokens` | fitted | no |
| `gdn_decode` | `gdn` | the same five | `batch_size` | fitted | yes |
| `optimizer_step` | `optimizer` | `optimizer, state_kind` | `param_count` | fitted | no |

`families()` lists the seven family directories: `attention`, `attention_training`, `collective`,
`elementwise`, `gdn`, `gemm` and `optimizer`.

`world_size` and `nodes` are slice columns rather than axes. A two-rank and an eight-rank
all-reduce run different ring schedules, and an intra-node and a cross-node one run over links
that differ by roughly an order of magnitude, so blending either into an interpolation would
misprice shapes that were never measured. A query at an unprofiled combination raises a coverage
error.

---

## Adding an op

**Step 1.** Write the class in `srl/cost/kernel/ops/<family>.py` and implement every member of the
protocol. If the op joins an existing family, the op shares that family's table directory and its
`measure()` call.

**Step 2.** Instantiate the class at module level and call `register_op`. A duplicate name raises
`KernelCoverageError` instead of overwriting.

**Step 3.** Add the module to the imports in `srl/cost/kernel/ops/__init__.py`. Registration is an
import side effect, and this file is the only path that runs it. `op_for` is called from the table
writer, `database.py` and `floor.py`, none of which know which ops exist. If the import is
missing, a fresh interpreter sees an empty registry while every unit test passes, because each
test imports the module it exercises. `tests/test_kernel_op_coverage.py` guards this with a
subprocess that compares the registry against the modules on disk.

**Step 4.** Add the family's sweep block to the kernel profile YAML under `examples/profile/kernel/`.
The floor fitter, the holdout gate, the table writer and the CLI need no edit. They loop over
`ops_in` and `ops_for_family` and read `fits_floor`.

An op declaring `fits_floor = True` must also have an entry in `FALLBACK_FLOORS_S`
(`srl/cost/kernel/sol.py`) under at least one backend. `tests/test_kernel_sol.py` derives this
check from the live registry. Without floors.json, a floor-taking op with no fallback entry takes
a zero floor, and a zero floor is not a refinement.

---

## The four lookups

Nothing indexes `OP_REGISTRY` directly.

| Lookup | Returns |
|---|---|
| `op_for(name)` | One op, raising `KernelCoverageError` on an unknown name |
| `ops_for_family(family)` | Every op writing into one directory, sorted by name |
| `families()` | The sorted distinct family directory names |
| `ops_in(version_dir)` | The ops whose `<name>_perf.parquet` exists in one version directory |

`ops_in` skips a parquet with no registered op so a stray file cannot break a report.
`test_kernel_op_coverage.py` is what fails on an op that exists but was never registered.

---

## `measure()` is per family

Both attention phases come from one backend sweep that allocates a KV cache once and times both
phases against it. The CLI calls `measure()` once per family, through `ops_for_family(family)[0]`.
Calling it per op would measure the grid twice and could select a different backend for each phase.
A family that holds a single op writes a single table.

For the GDN family, a model whose GQA ratio has no compiled FlashInfer decode kernel gets its
context table only. That is a property of the checkpoint, not of the TP choice.

---

## `floors.json`

A dispatch floor belongs to the backend's dispatch path. `fit-floor` scans candidate floors and
keeps, for each op, the one minimising the holdout error over all bracketed points. The result is
written to `{version_dir}/floors.json` beside the tables it describes, keyed by op name, alongside
the metadata keys `fitted_mre_percent`, `candidates_us`, `gpu` and `backend`.

An older file keyed by phase (`context`, `generation`) raises a `KernelCoverageError` that names
`fit-floor`. A missing file falls back to `FALLBACK_FLOORS_S` with a warning, and an op that the
fallback does not cover gets a second warning because it would otherwise take a zero floor.

The floor enters the SOL as `max(compute, memory, floor)`, not as an added constant. A decode
kernel that stays flat across a wide range of KV work is dispatch-bound, so the floor is a lower
bound.

---

## Measured shapes and utility surfaces

- `table.py` groups a table's rows into slices by the slice columns and interpolates the axis
  columns in log2 space. A duplicate shape keeps its fastest measurement.
- `interp.py` holds `UtilityGrid`, built from `UtilitySample`s. It raises `KernelCoverageError`
  when a query cannot be interpolated.
- `surface.py` holds the two surfaces in `SURFACE_REGISTRY`: `pchip` (default) and `shepard`.
- `database.py` answers a query as `SOL(shape, target_spec) / utility(shape)`. Splitting the
  prediction this way is what would let a table measured on one GPU price another, which is a
  property of the design and not a validated claim.

---

## Measuring elementwise kernels

Two traps make an eager microbenchmark report something the server never pays.

**The `CustomOp` wrapper.** vLLM wraps fused kernels in a `CustomOp` that replays inside a CUDA
graph. Calling the module in eager mode measures Python dispatch. The measurement code therefore
builds a `CompilationConfig(custom_ops=["all"])` context and checks the resolved dispatch path
(`forward_cuda`) rather than trusting a log line.

**Access paths.** Some kernels are reachable only through a wrapper. `silu_and_mul` is not on
`vllm._custom_ops` and is reached as `SiluAndMul().op` inside the config context.

`op_kind` is a slice column because normalisation over `head_dim` (Qwen3's `q_norm` and `k_norm`)
and normalisation over `hidden_size` land on different points of the bandwidth curve at the same
element count. Treating the two as one slice would query the wrong curve.

---

## The torch-free boundary

The cost path imports `cost.kernel.ops`, `cost.kernel.database`, `cost.kernel.table` and, for shape
derivation, `profiler.kernel.grid` and `profiler.kernel.utils.model_dims`. None of those may import
`torch` or vLLM, because a simulation prices kernels from parquet and must not pay the GPU stack's
import time.

An op that measures on a GPU imports its dependencies inside `measure()`, and its `parse_sweep`
imports the sweep module lazily. `tests/test_kernel_package_imports.py` discovers the modules under
`sweep/` and `grid/` plus `config.py`, `utils/model_dims.py` and `utils/validators.py` and imports
each in its own subprocess. A new grid or sweep is covered the moment it is added.
