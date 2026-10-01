# Kernel Op Abstraction Design

**Date:** 2026-08-26
**Status:** approved, not implemented
**Repo:** `/apdcephfs_zwfy/share_303760348/lhy/DryRun-RL`

## Problem

Adding one kernel op to the cost model today requires editing **ten** sites:

| layer | file | what is op-specific |
|---|---|---|
| SOL | `cost/kernel/sol.py` | one free function per op (`gemm_sol`, `context_attention_sol`, `generation_attention_sol`) |
| schema | `cost/kernel/table.py` | one `TableSchema` constant per op, plus a `schema_for()` dict |
| query | `cost/kernel/database.py` | one Query dataclass per op, one `query_*` method per op, `_sol()` branching on `isinstance` then on `phase` |
| holdout | `cost/kernel/holdout.py` | `_make_sol_fn` re-dispatches on op-name strings; four `== "attention"` special cases |
| floor | `cost/kernel/floor.py` | `_ATTENTION_OPS` phase map, hardcoded `/ "attention"` path |
| write | `profiler/kernels/measurements.py` | `write_gemm` vs `write_attention`, one row dataclass each |
| sweep | `profiler/kernels/config.py` | `GemmSweep` and `AttentionSweep` as named fields on `KernelProfileConfig` |
| measure | `profiler/kernels/{gemm,attention}.py` | separate entry points with different signatures (`writer=` vs `writer_factory=`) |
| CLI | `cli/profile.py` | `--family choices=[...]` and one `if args.family in (...)` block per family |

The count is the visible symptom. The real defect is that `database._sol()` (line 298) and
`holdout._make_sol_fn()` (line 154) are **two independent implementations of the same
binding**: both map `(spec, slice_key, coords, floor_s)` onto a SOL call. They can drift,
and drift there silently changes utility on one side of the holdout round trip, which is
the exact quantity the accuracy gate measures. That is a correctness risk, not verbosity.

A second, subtler leak: `floors.json` is keyed by **phase** (`context`, `generation`) while
everything else is keyed by **op name** (`context_attention`). Four separate sites translate
between the two.

## Goal

Make one class per op the single source of truth for that op's schema, SOL formula,
coordinate mapping, floor policy, sweep config, and measurement routine. Then add two new
ops (`elementwise` and `collective`) through that abstraction, which is the only real
proof it generalizes beyond the two families that shaped it.

## Non-goals

- **Mixed-batch attention is not a new op.** A chunked-prefill step is one context call
  plus one generation call, which is already what `composed.py:_attention_seconds` does.
  It is runtime composition, not a new table.
- **Backward GEMM is not a new op.** `shapes.py:gemm_wgrad_shapes` already emits wgrad
  triples into the *same* `gemm` table via `gemm_all_shapes`; backward is the existing op
  with permuted axes.
- **No end-to-end validation at `tp>1`.** See "Deliberately unvalidated" below.
- No change to the physics: SOL stays `max(math, memory, floor)`, floors stay inside the
  `max` rather than added, tables keep storing latency rather than utility, and a query
  stays `SOL / interpolated_utility`.

## Execution order

The order is a safety property, not a preference. Stage 1 must be provably
behaviour-neutral, and that proof is only possible while the op set is unchanged.

| stage | content | gate |
|---|---|---|
| **0** | Capture the ten baseline numbers below into a committed file | the file exists before any source edit |
| **1** | The refactor: `ops/` package, `sol()` unification, floor re-keying, writer collapse, sweep-on-op, CLI derivation | every stage-0 number reproduces **identically** |
| **2** | `elementwise` op: schema, SOL, sweep, measurement, floor fit: **not** wired into `predict()` | holdout MRE with `point_count > 0`; a test asserts `predict()` has no elementwise term |
| **3** | `all_reduce` op: same, plus `ModelOps.comm_queries` and lifting the `tp=1` restriction | holdout MRE reported; `tp=2` constructs and predicts; tp=1 validate numbers **unchanged** |
| **4** | Docs: `docs/composed-rollout.md`, `.claude/codebase-map.md`, kernel README | every figure traces to a stage-2/3 command |

Stages 2 and 3 are independent of each other and both depend only on stage 1. Stage 3 must
not begin before stage 1's gate passes, because a collective term added mid-refactor would
make the baseline comparison meaningless.

## Architecture

### `KernelOp`: the single source of truth

```python
class KernelOp(Protocol):
    name: str                      # "gemm", "context_attention", "all_reduce", ...
    family: str                    # directory: "gemm", "attention", "collective", "elementwise"
    slice_columns: tuple[str, ...] # categorical, matched exactly
    axis_columns: tuple[str, ...]  # numeric, interpolated in log2 space
    fits_floor: bool               # does this op take a fitted dispatch floor?

    def sol(self, spec: GpuSpec, *, slice_key, coords, floor_s: float) -> SolComponents: ...
    def parse_sweep(self, value: dict) -> object: ...
    def measure(self, model, config, *, writer_factory) -> tuple[Path, ...]: ...
```

Layout:

```
dryrun/cost/kernel/ops/
├── base.py          KernelOp protocol + OP_REGISTRY + four lookups
├── gemm.py          GemmOp
├── attention.py     ContextAttentionOp, GenerationAttentionOp
├── elementwise.py   ElementwiseOp
└── collective.py    AllReduceOp
```

`base.py` exports exactly four lookups, and no other module may index `OP_REGISTRY`
directly:

| lookup | returns | replaces |
|---|---|---|
| `op_for(name) -> KernelOp` | one op, raising `KernelCoverageError` on an unknown name | `table.schema_for` |
| `ops_for_family(family) -> tuple[KernelOp, ...]` | every op writing into one directory | the `if args.family in (...)` blocks |
| `families() -> tuple[str, ...]` | sorted distinct family names | the `--family` choices literal |
| `ops_in(version_dir) -> tuple[KernelOp, ...]` | ops whose `{name}_perf.parquet` exists there | the parquet-stem parsing at `holdout.py:367` and `floor.py:86` |

`sol()` is the load-bearing member. After this change there is exactly one implementation
of the SOL binding, reached by both `database` and `holdout`, so the duplication becomes
structurally impossible rather than merely discouraged.

**Signature uniformity is achieved through `slice_key`, not through `**kwargs`.** GEMM's SOL
needs `dtype`; attention's needs five dtype and head parameters; collective's needs
`world_size`. Each op unpacks what it needs from **its own** `slice_key`, whose column order
that same op declares. So `GemmOp.sol` unpacks `(gemm_dtype, n, k)` and `AllReduceOp.sol`
unpacks `(dtype, world_size, hidden_size)`, with no shared signature to compromise. This is
why the abstraction is a class rather than a `sol_fn` field on `TableSchema`: a single
callable field would have forced `**kwargs`, losing the self-documenting signatures that
`context_attention_sol` has today.

`TableSchema` is absorbed: the op already declares `slice_columns` and `axis_columns`, so
`schema_for(op_name)` becomes `op_for(op_name)` and the three `*_SCHEMA` constants are
deleted rather than kept as aliases.

**The registry must stay torch-free.** `tests/test_kernel_package_imports.py` and
`tests/test_kernel_shapes_import.py` assert that importing the cost path does not load
`torch` (a 4.2 s import). `measure()` therefore imports its GPU dependencies inside the
method body, matching how `profiler/kernels/attention.py` already defers `_vllm_attn`.

### Profiler side

**Row writing collapses to one method.** `write_gemm` and `write_attention` differ only in
their columns dict, and that dict is exactly `op.slice_columns + op.axis_columns`:

```python
def write(self, op_name: str, rows: Sequence[Mapping[str, Any]]) -> Path:
    op = op_for(op_name)
    expected = set(op.slice_columns + op.axis_columns) | {"latency_s"}
    # Reject a missing or extra column here. write_attention currently trusts its caller
    # to populate eleven columns by hand, and a typo only surfaces when someone reads the
    # table: possibly hours after the sweep that produced it.
```

This is strictly safer than today, not merely shorter.

**`measure()` is deliberately coarse**: `(model, config, writer_factory) -> paths` rather
than a per-shape `measure_one(shape) -> latency`. The reason is collective: it spawns
`world_size` processes with `torch.multiprocessing` and only rank 0 retains timings, so
there is no per-shape call to hook. A fine-grained interface would leak that. The coarse one
also accommodates attention's existing `writer_factory` (the kernel source is known only
after backend selection) and GEMM's plain loop without bending either.

**Sweep config moves onto the op.** `KernelProfileConfig` loses its `gemm:` and `attention:`
named fields in favour of `config.sweep(op_name)`, each op parsing its own YAML subtree
through `parse_sweep`. Validation stays eager and keeps naming the offending field, because
it stays with the op. `shapes.py` already receives `sweep: GemmSweep` as an explicit
parameter, so it is unaffected.

**CLI dispatch derives from the registry.** `--family` choices become
`("all",) + families()`, and the per-family `if` blocks become a loop over
`ops_for_family()`. Adding an op requires zero CLI edits.

### Floors keyed by op, not by phase

`floors.json` is re-keyed from `{"context": ..., "generation": ...}` to
`{"context_attention": ..., "generation_attention": ...}`. This deletes all four
phase↔op translations (`floor.py:43`, `holdout.py:224`, `holdout.py:380`,
`database._sol`); the lookup becomes `floors.get(op.name)`.

Floor fitting then becomes op-agnostic:

```python
for op in ops_in(version_dir):
    if not op.fits_floor:
        continue
```

GEMM is skipped by declaring `fits_floor = False` rather than by being absent from a map,
and `elementwise` and `collective` receive fitted floors with **no new code in
`floor.py`**. That is the concrete test of whether this abstraction is real.

`holdout.py`'s discovery loop likewise stops special-casing `family_dir.name ==
"attention"` and asks each op instead.

`ATTENTION_FLOOR_S` in `sol.py` is re-keyed identically and renamed `FALLBACK_FLOORS_S`. It
remains a fallback with a named warning: a missing `floors.json` must degrade loudly, not
fail.

**This is a breaking format change,** which the project contract permits (no backward
compatibility; remove obsolete paths rather than adding migrations). A `floors.json`
carrying the old phase keys is **rejected with a message naming
`dryrun-profile kernels fit-floor`**, never silently misread: a lenient reader would apply
a zero floor and quietly degrade the gate by more than 20 percentage points (measured: floor
0 scores 34.89% MRE against 6.63% at the fitted 22 us).

### The two new ops

| | `elementwise` | `all_reduce` |
|---|---|---|
| family directory | `elementwise` | `collective` |
| slice columns | `(dtype, op_kind)` | `(dtype, world_size)` |
| axis columns | `(n_elements,)` | `(message_bytes,)` |
| SOL math term | none (no matmul) | none |
| SOL memory term | `bytes / memory_bw` | `2·(N−1)/N · bytes / link_bw` |
| `fits_floor` | `True` | `True` |
| measurement | single CUDA kernel, existing timing loop | multi-process NCCL spawn |

`op_kind` values: `rmsnorm`, `rope`, `silu_mul`. They share one schema because they share
one SOL shape (pure bandwidth) and differ only in bytes touched, which `n_elements`
already carries.

`GpuSpec.link_bandwidth_bytes_per_s` already exists, so collective's SOL needs no change to
`gpu_specs.yaml`.

`profiler/kernels/collective.py` already implements the NCCL spawn and returns a
`CollectiveProfile` of JSON measurements. It is rewritten to write a parquet perf table
through `AllReduceOp.measure` instead, so collective joins the same table tree, the same
utility surface, and the same holdout gate as every other op. The `CollectiveProfile` /
`CollectiveMeasurement` JSON dataclasses are deleted rather than kept alongside.

`ModelOps` gains `comm_queries(token_count)`: **two all-reduces per layer** (after
`o_proj`, after `down_proj`), multiplier `num_layers`. It is declared as data with an
explicit multiplier, for the same reason `logits_op` carries `×1`: an inline multiplier is
one indentation level away from a plausible-looking 36× error.

`ComposedRollout`'s `tensor_parallel != 1` rejection is removed, and `predict()` adds the
comm and elementwise terms.

## Behaviour-neutrality gate

The refactor touches code that currently passes a validated accuracy gate. The gate for the
refactor itself is that **the numbers do not move**. Baselines captured 2026-08-26 on H20,
Qwen3-8B, tp=1, `--fraction 1.0`:

| quantity | baseline | required after refactor |
|---|---|---|
| holdout `gemm/vllm` | 3.91% MRE, 64 points | identical |
| holdout `context_attention/vllm` | 5.74% MRE, 59 points | identical |
| holdout `generation_attention/vllm` | 6.63% MRE, 72 points | identical |
| holdout `context_attention/flashinfer` | 9.87% MRE, 59 points | identical |
| holdout `generation_attention/flashinfer` | 6.01% MRE, 72 points | identical |
| validate decode (pchip) | 4.84% MRE, 216 buckets | identical |
| validate prefill (pchip) | 3.72% MRE, 29 buckets | identical |
| validate mixed (pchip) | 0.48% MRE, 5 buckets | identical |
| fitted floors, vllm | context 32 us, generation 22 us | identical |
| fitted floors, flashinfer | context 4 us, generation 6 us | identical |

Not "close": identical. `sol()` performs the same arithmetic on the same inputs, so any
drift means the binding changed. This mirrors the golden-trace diff that gated the
`instance.py` rewrite in the previous phase.

The two new ops are measured and gated **only after** that proof lands, so their error
cannot be confused with refactor drift.

## Elementwise: measured, and deliberately not wired into `predict()`

This section replaces an earlier estimate that was wrong in both magnitude and direction.
The estimate assumed elementwise kernels scale with bandwidth, putting them at 0.03% of a
decode step. Measured on H20 against the real vLLM kernels, they are **flat at 4–5 us**
regardless of token count below ~128, so their *share* grows as the step shrinks.

### Three measurements that changed the decision

**1. The kernel must be timed raw, not through its vLLM module.** vLLM wraps each
elementwise kernel in a `CustomOp` module, and the server replays that inside a CUDA graph.
Timing the module in eager mode measures Python dispatch, not the kernel:

| | 16 tokens | 4096 tokens |
|---|---|---|
| `RMSNorm(...)` module call | 60.33 us | 342.48 us |
| raw `vllm._custom_ops.rms_norm(...)` | **4.57 us** | **29.03 us** |
| ratio | **13.2×** | 11.8× |

Profiling the module would have inflated the elementwise table by an order of magnitude.
Additionally, a `CustomOp` instantiated without `CompilationConfig(custom_ops=["all"])`
silently dispatches vLLM's **native PyTorch fallback** rather than the fused CUDA kernel
(it logs `Priority not set for op rms_norm, using native implementation`), which is a
different kernel from the one the server runs.

**2. Qwen3 runs six elementwise kernels per layer, not four**: two `rms_norm` (input and
post-attention), one `rotary_embedding`, one `silu_and_mul`, and `q_norm`/`k_norm` over
`head_dim` (Qwen3 has `qk_norm`). Measured per-layer total: **27.3 us** at decode token
counts, **356 us** at 4096 tokens.

**3. Adding it makes every phase worse, in the same direction.** Scored over the same 250
baseline buckets `kernels validate` uses, with the measured per-layer cost interpolated in
log2 space over a 15-point token grid:

| phase | buckets | MRE now | MRE with elementwise | median ratio now | with elementwise |
|---|---|---|---|---|---|
| decode | 216 | 4.84% | **11.18%** | 1.0119 | 1.0639 |
| prefill | 29 | 3.72% | **5.84%** | 1.0069 | 1.0305 |
| mixed | 5 | 0.48% | **1.89%** | 0.9994 | 1.0143 |

### Why this is double-counting, not an accuracy tradeoff

Every phase moves the same way, and every median ratio moves *away* from 1.0. A genuinely
missing term would move the error toward zero in at least one regime. It moves toward
over-prediction everywhere.

The cause is how utility is defined. Utility is `SOL / measured_latency`, and the measured
latency of each GEMM and attention kernel was collected from a server whose elementwise
kernels were running between them. **The elementwise cost is therefore already inside the
fitted GEMM and attention utilities.** `docs/composed-rollout.md` described this absorption
as partial; the measurement shows it is essentially complete.

### Decision

`elementwise` is implemented as a full op: schema, SOL, sweep, measurement, fitted floor,
holdout gate, and **is not added to `ComposedRollout.predict()`**. Concretely:

- The abstraction gets its third op family, which is the structural proof it generalizes
  beyond the two families that shaped the current code.
- The table exists for the FSDP training backend, which has **no CUDA graphs** and will
  need an explicit term. See
  `docs/superpowers/specs/2026-08-26-fsdp-training-backend-design.md`.
- The `4.84% → 11.18%` figure is recorded in `docs/composed-rollout.md` as the evidence
  that utility absorbs elementwise completely: replacing the current vaguer "partly
  absorbed" wording.
- Wiring it into `predict()` requires first refitting GEMM and attention utility against
  *kernel-only* latency so the terms stop overlapping. That is a separate piece of work
  with its own gates, and it is out of scope here.

A test asserts `predict()` does **not** include an elementwise term, so a future change that
wires it in has to confront this reasoning rather than silently double-count.

### The unexplained small-batch bias, recorded not fixed

Investigating the above surfaced a separate finding. The aggregate decode `median_ratio` of
1.0119 is an averaging artifact; the real structure is concentrated:

| batch | buckets | median ratio | median latency |
|---|---|---|---|
| reqs 1–4 | 66 | **1.1312** | 6.81 ms |
| reqs 5–16 | 35 | 0.9960 | 14.06 ms |
| reqs 17–48 | 68 | 1.0135 | 25.07 ms |
| reqs 49+ | 47 | 1.0072 | 46.03 ms |

Small-batch decode is over-predicted by **13%**. Three candidate causes were tested and
**ruled out**:

- **CUDA graph capture.** Eager and graph-replayed GEMM agree within 1–12% on the real
  shapes (`m=1, n=24576, k=4096`: 61.23 us eager, 60.73 us graph).
- **Summing kernels being invalid.** Four individually timed GEMMs × 36 layers = 4.644 ms;
  one fused 36-layer graph replay = 4.721 ms: agreement to **1.7%**.
- **Bad interpolation at `m=1`.** `m=1` is a *measured* point, not interpolated: the table
  value and the query result are bit-identical (62.28 us).

So the composition arithmetic is sound and every input is measured, yet batch-1 is 13% high.
The remaining candidate is kernel overlap in the real server, which per-kernel timing cannot
observe. This is recorded as a known limitation with its ruled-out causes; diagnosing it is
out of scope for this plan and must not be conflated with refactor drift.

## Deliberately unvalidated

**`ComposedRollout` at `tp>1` becomes constructible but its end-to-end accuracy is
unvalidated.** No vLLM `tp=2` profiling baseline exists (`outputs/profile/` holds only
tp=1 trees), so `kernels validate` cannot score it. What *is* gated is the collective
table's own interpolation accuracy, through the same holdout gate as every other op. The
documentation must state this distinction plainly rather than let "tp>1 now works" imply
end-to-end accuracy. Eight H20s are available, so a tp=2 baseline is feasible later; it is
out of scope here because it would mix a new measurement campaign into a refactor proof.

Also carried forward unchanged from the previous phase, and not addressed here:

- **Mixed batches are still approximated** as context + generation attention at the step's
  batch size, scored on only six real mixed iterations. Still the model's largest
  uncharacterised approximation.
- **`context_attention/flashinfer` remains at 9.87%**, above the 5% gate. The cause is grid
  density below `seq_len` 1024, not a floor misfit: a 0–80 us floor sweep confirmed 4 us is
  already optimal.
- **Cross-hardware transfer remains unvalidated**; only one GPU type is available.

## Testing

**Refactor tasks** are gated by the baseline table above, plus:

The refactor renames two APIs the test suite uses directly, so some test edits are
unavoidable. The permitted blast radius is named up front, and anything beyond it is a
signal that behaviour changed rather than structure:

| API | test files referencing it | permitted edit |
|---|---|---|
| `schema_for`, `GEMM_SCHEMA`, `CONTEXT_ATTENTION_SCHEMA`, `GENERATION_ATTENTION_SCHEMA` | `test_kernel_holdout`, `test_kernel_attention_vllm`, `test_kernel_floor`, `test_kernel_attention_flashinfer`, `test_kernel_measurements`, `test_kernel_table`, `test_kernel_gemm`, `test_kernel_surface_wiring` | mechanical rename to `op_for` / the op object |
| `write_gemm`, `write_attention` | `test_kernel_holdout`, `test_kernel_measurements`, `test_kernel_table`, `test_kernel_database`, `test_profiling_cli` | mechanical rewrite to `write(op_name, rows)` |

**No assertion may be weakened or deleted.** A test whose *expected value* changes means the
refactor was not behaviour-neutral, and the correct response is to fix the code, not the
test.

Additionally:

- `tests/test_kernel_package_imports.py` and `tests/test_kernel_shapes_import.py` must
  pass **unmodified**: the op registry must not pull in `torch`.
- A new test asserts `OP_REGISTRY` covers every parquet stem the perf-table tree contains,
  so a table with no op is a failure rather than a silent skip in `holdout_report`.
- A new test asserts an old-format `floors.json` (phase keys) is rejected with a message
  naming the `fit-floor` command, rather than read as a zero floor.

**New-op tasks** are gated by:

- `PerfTableWriter.write` rejecting a missing or extra column, per op, on synthetic rows.
- The holdout gate reporting a real MRE for `elementwise` and `all_reduce`: with the
  explicit check that `point_count > 0`, since a zero-point result is a coverage failure
  that currently only logs a warning.
- Floors being fitted for both new ops with no edit to `floor.py`, asserted by a test that
  fits floors for a synthetic op registered only inside the test.
- `ModelOps.comm_queries` returning two all-reduces per layer at the layer multiplier, and
  a test asserting the total is not multiplied by `num_layers` twice.
- `ComposedRollout(tensor_parallel=2)` constructing successfully and predicting a step
  strictly larger than the same shape at `tp=1` per rank plus zero comm.

## Risks

| risk | mitigation |
|---|---|
| Refactor silently changes utility | Byte-level baseline comparison on ten measured quantities, captured before any edit |
| `measure()` interface leaks for collective | Interface deliberately coarse (`-> paths`), chosen *because* of collective's multi-process shape |
| Op registry drags `torch` into the cost path | Existing import-guard tests must pass unmodified; GPU imports stay inside `measure()` |
| Old `floors.json` misread as zero floor | Explicit rejection with a directive message; measured cost of a silent zero floor is 34.89% vs 6.63% MRE |
| Elementwise double-counts work already in the fitted utilities | Measured at decode 4.84%→11.18%; the op is built and gated but not wired into `predict()`, and a test asserts that |
| Elementwise measured through its vLLM module instead of the raw op | Module wrapper costs 13.2× the kernel; the plan names `vllm._custom_ops` entry points and requires `custom_ops=["all"]` |
| `tp>1` reads as validated | Documentation states the kernel/e2e distinction explicitly |
