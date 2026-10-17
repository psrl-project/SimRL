# Kernel ops

Every measured kernel family is one class. That class owns its perf-table schema, its speed-of-light
formula, its floor policy, its sweep grid specification, and how it is measured on the GPU. Adding a
new op therefore requires touching exactly one place: the new op class and its registration. Nothing
else in the codebase (the floor fitter, the holdout gate, the perf-table writer, the CLI) needs to
change.

For measured numbers and interpolation accuracy, see {doc}`../profiling/kernels`. For the slice,
axis, SOL, utility, and floor vocabulary, see {doc}`../getting-started/concepts`.

---

## Why `KernelOp` is a class, not a table of functions

The visible symptom of the old design was edit count: adding a kernel op used to mean touching ten
separate sites. The actual defect was narrower and more dangerous. `srl/cost/kernel/database.py`
and `srl/cost/kernel/holdout.py` each held their own independent implementation of the same
binding, the function mapping `(spec, slice_key, coords, floor_s)` onto a SOL call. Utility is
`SOL / measured_latency`, and the holdout gate computes utility on both sides of a round trip, so
any drift between those two copies would have silently changed the measured interpolation accuracy
without changing anything a unit test looked at.

`KernelOp.sol()` is that binding, once. There is no second copy to drift from.

Signature differences between families are absorbed cleanly without a shared `**kwargs` catch-all:
each op unpacks what it needs from *its own* `slice_key`, whose column order that same op declares.
`GemmKernelOp.sol` reads `(gemm_dtype, n, k)`, and `AllReduceKernelOp.sol` reads `(dtype, world_size)`.
Each op knows its own coordinate space.

The `KernelOp` protocol is defined in `srl/cost/kernel/ops/base.py`. Every op must implement:

| Attribute / method | Purpose |
|---|---|
| `name` | Op name, e.g. `gemm`. Matches the parquet filename stem. |
| `family` | Directory that holds this op's table, e.g. `attention`. |
| `slice_columns` | Columns requiring exact match. |
| `axis_columns` | Columns interpolated along. |
| `fits_floor` | Whether this op has a per-backend dispatch floor. |
| `sol(spec, *, slice_key, coords, floor_s)` | Return roofline `SolComponents`. |
| `parse_sweep(value)` | Parse this op's YAML sweep subtree. |
| `table_columns()` | Return non-provenance column names, latency last. |
| `measure(model, config, *, writer_factory)` | Measure on the GPU and write tables. |

---

## The registered ops

| Op | Family | Slice columns | Axis columns | Floor |
|---|---|---|---|---|
| `gemm` | `gemm` | `gemm_dtype, n, k` | `m` | no |
| `context_attention` | `attention` | 6 shared columns | `num_heads, batch_size, isl` | fitted |
| `generation_attention` | `attention` | 6 shared columns | `num_heads, batch_size, step` | fitted |
| `elementwise` | `elementwise` | `dtype, op_kind` | `n_elements` | fitted |
| `all_reduce` | `collective` | `dtype, world_size` | `message_bytes` | fitted |

The table shows structure, not accuracy. For interpolation MRE, p90, point counts, and the
`--fraction 1.0` gate results, see {doc}`../profiling/kernels`.

`all_reduce` places `world_size` in its slice columns rather than its axis columns because a 2-rank
and an 8-rank all-reduce run different ring schedules over different link topologies. The ring
lower bound `2*(N-1)/N * bytes / bandwidth` varies non-linearly with `N` even if the message size
is held constant. Blending across world sizes would systematically underestimate costs at
intermediate counts that were never profiled. A query at `tp=4` or `tp=8` therefore raises a
coverage error rather than extrapolating.

---

## Adding an op

Four steps are required, and the order is important.

**Step 1.** Write the class in `srl/cost/kernel/ops/<family>.py`, implementing all attributes
and methods in the `KernelOp` protocol. Choose the right `family` value: ops sharing a directory
are measured together and stored together.

**Step 2.** Instantiate the class at module level and call `register_op`. Duplicate registration
raises `KernelCoverageError` rather than silently overwriting, so two ops sharing a name cannot
exist.

**Step 3.** Add the module import to `srl/cost/kernel/ops/__init__.py`. This step is more
important than it looks and is the one most likely to be forgotten.

`op_for` is called from the perf-table writer, `database.py`, and `floor.py`. None of those
modules has any reason to know which specific ops exist, which is the registry's job. Registration
happens as an import side effect of the module that defines each op. If the import is omitted from
`ops/__init__.py`, those callers see an empty registry. This failure mode is invisible to unit
tests: each test imports the op module it exercises, so it populates the registry for itself. The
failure only appears when a fresh interpreter calls `op_for` without first importing the individual
module. The documented symptom is `op_for("gemm")` raising with `Registered ops: none` while every
test passes. The `ops/__init__.py` comment explains this explicitly and marks each import `noqa:
F401` because ruff cannot see that the import's purpose is its side effect.

**Step 4.** Add the family's sweep block to the kernel profile YAML configs under
`examples/profile/kernel/`. Floor fitting, holdout discovery, the perf-table writer, and the CLI
need no edits: they loop over `ops_in()` / `ops_for_family()` and check `fits_floor`. This was
verified empirically: the commits adding `elementwise` and `all_reduce` touched neither
`srl/cost/kernel/floor.py` nor `srl/cli/profile.py`.

If the op declares `fits_floor = True`, it must also appear in `sol.FALLBACK_FLOORS_S` under at
least one backend. `tests/test_kernel_sol.py` enforces this off the live registry. The reason the
fallback matters is quantified: an op with a floor policy but no entry in `FALLBACK_FLOORS_S` takes
a zero floor when `floors.json` is absent. For `elementwise` that costs 5.01% against 11.79% MRE. For
`all_reduce` the underlying SOL error reaches 220x at 4096 bytes.

---

## The four lookups

`OP_REGISTRY` is never indexed directly. Everything goes through four stable functions defined in
`srl/cost/kernel/ops/base.py`:

| Lookup | Returns |
|---|---|
| `op_for(name)` | One op, raising `KernelCoverageError` on an unknown name. |
| `ops_for_family(family)` | Every op writing into one directory, sorted by name. |
| `families()` | Sorted distinct family directory names across all registered ops. |
| `ops_in(version_dir)` | Ops whose `{name}_perf.parquet` exists in one version directory. |

Direct indexing is avoided because it makes the caller implicitly assume it knows what is
registered, which is exactly the assumption that allowed the stale-registry bug to go undetected. The four
functions make that assumption explicit via `KernelCoverageError`.

---

## `measure()` is per family, not per op

Both attention phases, context and generation, come from a single sweep that allocates one KV
cache and times both phases against it. Calling `measure()` once per op would measure the attention
grid twice and could select a different backend for each phase, producing tables that were measured
under inconsistent conditions. The CLI therefore calls `measure()` once per family, not once per
op. An op whose family holds only itself (GEMM, elementwise, collective) simply writes one table.

---

## `floors.json` is keyed by op name

A dispatch floor is a property of the backend's dispatch path. It is fitted per `(gpu, family,
backend, version)` and stored beside the tables it was fitted for. The keys in `floors.json` are op
names: `context_attention`, `generation_attention`, `elementwise`, `all_reduce`.

An older format keyed by phase (`context`, `generation`) is **rejected** with an explicit error
message naming `srl-profile kernels fit-floor`, never read leniently. The reason leniency is
refused: a leniently-read old-format file applies a zero floor, which produces generation attention
MRE of 34.89% against 6.63% at the fitted 22 µs. Silently incorrect is worse than explicitly
broken.

The floor goes inside `max(math, memory, floor)`, not added as a constant. Real H20 data shows
decode latency staying near-constant across a 64x increase in KV work at batch 1, because the kernel is
entirely dispatch-limited. That shape is conclusive that dispatch cost is a lower bound, not an
addend, because adding it as a constant would double-count in the overhead-dominated regime. The formula is
documented in {doc}`../getting-started/concepts`.

---

## Measuring elementwise kernels: two traps

**Trap 1: the `CustomOp` dispatch trap.** vLLM wraps each elementwise kernel in a `CustomOp` that
replays inside a CUDA graph. Timing the module in eager mode measures Python dispatch overhead
rather than the kernel:

| | 16 tokens | 4096 tokens |
|---|---|---|
| `RMSNorm(...)` module call | 60.33 µs | 342.48 µs |
| raw `vllm._custom_ops.rms_norm(...)` | 4.57 µs | 29.03 µs |
| ratio | 13.2x | 11.8x |

A 13x apparent speedup from routing the call differently is not a real kernel improvement. It is
measuring the wrong thing entirely.

**The fix is `CompilationConfig(custom_ops=["all"])`.** Without it, a `CustomOp` dispatches vLLM's
native PyTorch fallback, which is a different kernel from the fused one the server runs.

Verify that the config took effect by probing the dispatch path directly, not by watching the log.
With `custom_ops=["all"]`, `_forward_method` resolves to `forward_cuda`. With `[]` it resolves to
`forward_native`. The `Priority not set for op ... using native implementation` warning in vLLM
0.26.1.dev0 is not a reliable canary: it fires on module calls even when `custom_ops=["all"]` is
active and never fires on the raw-op path regardless of config.

**Trap 2: `silu_and_mul` access path.** `silu_and_mul` is not exposed on `vllm._custom_ops`.
Reach it as `SiluAndMul().op` after constructing the module inside the config context.

**Qwen3 runs six elementwise ops per layer, not four.** The four common ops are `rms_norm`,
`rotary_embedding`, `silu_and_mul`, and `o_proj` output normalization. Qwen3 adds `q_norm` and
`k_norm` over `head_dim`, which is a separate `op_kind` from `rms_norm` because it normalises over
`head_dim` rather than `hidden_size`, landing on a different point of the bandwidth curve at the
same token count. Treating `qk_norm` as a variant of `rms_norm` would query the wrong slice of the
table.

---

## Torch-free cost path

`srl/cost/kernel/ops/base.py` must stay importable without `torch`. A simulator run prices
kernels from parquet and should not pay a 4.2-second GPU-stack import to do it. Any op that must
measure on a GPU imports its GPU dependencies (`torch`, vLLM modules, FlashInfer) inside the
`measure()` and `parse_sweep()` method bodies. The `parse_sweep` and `measure` methods are
never called on the cost path. Only `sol()`, `slice_columns`, `axis_columns`, and `fits_floor` are.
`tests/test_kernel_ops_base.py` enforces this with a subprocess probe on a fresh interpreter.
`tests/test_kernel_shapes_import.py` enforces the same rule for the shape-derivation module.
