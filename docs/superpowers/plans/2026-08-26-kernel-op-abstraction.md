# Kernel Op Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one class per kernel op the single source of truth for its schema, SOL formula, floor policy, sweep config, and measurement routine, then add `elementwise` and `all_reduce` through that abstraction.

**Architecture:** A `KernelOp` protocol in a new torch-free `dryrun/cost/kernel/ops/` package absorbs `TableSchema` and replaces the two independent SOL bindings in `database._sol` and `holdout._make_sol_fn` with one `op.sol()`. `floors.json` is re-keyed from phase to op name, which deletes four phase-to-op translations. The profiler's per-family writer methods and sweep config fields collapse onto the same ops.

**Tech Stack:** Python 3.12, pyarrow 23.0.1, numpy 1.26.4, torch (profiler side only), vLLM (third_party checkout), Hydra/OmegaConf, pytest, ruff.

## Global Constraints

- Activate the environment before any run: `source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh`
- ruff: `line-length = 119`, `select = ["E", "F", "W", "I"]`, `known-first-party = ["dryrun"]`
- pytest: `testpaths = ["tests"]`; tests are flat `tests/test_*.py`, no subdirectories
- Suite baseline is **713 passed** measured 2026-08-27 at commit `70a5071`, with
  `--ignore=tests/test_fsdp_cost.py`. **Always pass that ignore flag**: another session is
  executing the FSDP/Megatron plan on this same branch and that file currently errors at
  collection. After each task assert two things rather than a memorised total: the suite is
  green, and its count grew by exactly that task's new-test count. A count that grew by less
  means a parametrization was dropped; one that grew by more means a test was collected twice.
  (An earlier draft of this plan said 635/725; both figures predate the other session's ~78
  tests and must not be used.)
- **Do not touch `dryrun/cost/training/`, `dryrun/profiler/adapters/`, or `tests/test_fsdp_*`.**
  Those belong to the concurrent FSDP/Megatron plan and are orthogonal to this one.
- **Run the baseline capture and the Stage 1 gate on idle GPUs.** `nvidia-smi` must show no
  concurrent load. `tests/test_kernel_attention_vllm.py:test_generation_latency_grows_with_kv_history`
  asserts a measured latency ratio above 1.20x and was observed failing under contention from
  the other session while passing in isolation. A timing failure is not a regression until you
  have checked for concurrent GPU work.
- Exactly **7 pre-existing ruff findings** exist, none in files this plan touches: `I001` in `dryrun/cost/training/hydraulis.py`, `dryrun/profiler/adapters/vllm/collect.py`, `tests/test_plots.py`, `tests/test_profile_adapters.py`, `tests/test_telemetry.py`, `tests/test_training_cost.py`; plus `F841` at `tests/test_training_cost.py:109`. Leave them. Everything you touch must be clean.
- Coding style per `.claude/coding-style.md`: identifier strings (dict keys, `==` comparands, argparse `choices`) never take a trailing period and never change casing; human-readable messages are capitalized and end with `.`; **one-line docstrings are the established convention: do not expand them**; type hints use `X | None` and lowercase builtins.
- Latency is in **seconds** everywhere, never milliseconds.
- No backward compatibility. Remove obsolete paths rather than adding fallbacks or migrations.
- Do not modify `dryrun/cost/kernel/{hardware,interp,surface}.py` or `dryrun/cost/rollout/{psrl,roofline,distserve,segment,advance}.py` behaviour.
- **Never run `git stash`, `git stash pop`, `git checkout: <path>`, `git restore`, `git reset`, `git clean`, `git add -A`, or `git add .`** The repository owner has extensive uncommitted work in `dryrun/workload/`, `dryrun/cli/workload.py`, `examples/`, and `tests/test_workload_*.py`. Stage only the explicit paths each task names.
- `.claude/settings.local.json` must stay gitignored and must never be committed.

### Verified facts: do not re-derive these

Every number below was measured on this machine at commit `72f00f5`.

**Naming collision, must be resolved in Task 2.** `dryrun/cost/kernel/model_ops.py:30` already
defines a frozen dataclass named `GemmOp`, meaning "one GEMM in the per-step op list": a
different concept from this plan's `GemmOp` (a kernel *family*). The existing name is read by
`tests/test_kernel_model_ops.py` (`.name`, `.n`, `.k`, `.per_layer`) and
`tests/test_rollout_composed.py` (`projection_ops`, `logits_op`). **This plan renames the
new class to `GemmKernelOp`** and leaves `model_ops.GemmOp` untouched. Every new op class
carries the `KernelOp` suffix: `GemmKernelOp`, `ContextAttentionKernelOp`,
`GenerationAttentionKernelOp`, `ElementwiseKernelOp`, `AllReduceKernelOp`.

**The `m=1` GEMM points are measured, not interpolated.** For slice `(bfloat16, 24576, 4096)`,
the table value and `query_gemm` result are bit-identical at `m` = 1, 2, 4, 16 (62.28, 61.50,
62.00, 62.53 us). Any behaviour-neutrality failure at these points is a real regression, not
an interpolation artifact.

**Summing per-kernel latencies is valid.** Four individually timed GEMMs at `m=1` sum to
129.00 us; one CUDA-graph replay of 36 layers of the same four is 4.721 ms against the summed
4.644 ms: agreement to **1.7%**. Do not "fix" the composition method.

**CUDA graphs do not explain any error.** Eager vs graph-replayed GEMM agree within 1-12%
(`m=1, n=24576, k=4096`: 61.23 us eager, 60.73 us graph).

**Elementwise must be timed through `vllm._custom_ops`, not the module.** `RMSNorm(...)`
called as a module costs 60.33 us at 16 tokens against 4.57 us for
`vllm._custom_ops.rms_norm(...)`: **13.2x**. The server replays the fused op inside a CUDA
graph, so the raw op is the honest measurement.

**A `CustomOp` needs `CompilationConfig(custom_ops=["all"])`.** Without it vLLM logs
`Priority not set for op rms_norm, using native implementation` and dispatches the native
PyTorch fallback, which is a different kernel from the server's.

**Qwen3 runs six elementwise kernels per layer**, not four: 2x `rms_norm` (input,
post-attention), 1x `rotary_embedding`, 1x `silu_and_mul`, plus `q_norm` and `k_norm` over
`head_dim` (Qwen3 has qk_norm). Measured per-layer totals: 27.32 us at 16 tokens, 356.47 us at
4096 tokens.

**`silu_and_mul` is not exposed on `vllm._custom_ops`.** Reach it as `SiluAndMul().op` after
constructing the module inside the config context; the module's `forward_cuda` calls
`self.op(out, x)`. `rms_norm` and `rotary_embedding` ARE on `vllm._custom_ops` with
signatures `rms_norm(out, input, weight, epsilon) -> None` and
`rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox, rope_dim_offset=0, inverse=False) -> None`.

**Elementwise must NOT be wired into `predict()`.** Measured over the same 250 baseline
buckets `kernels validate` uses, adding it takes decode 4.84% -> **11.18%**, prefill 3.72% ->
**5.84%**, mixed 0.48% -> **1.89%**, with every median ratio moving away from 1.0. It is
double-counting: utility is `SOL / measured`, and the measured GEMM and attention latencies
came from a server already running these kernels between them. See the spec's elementwise
section.

**`GpuSpec.link_bandwidth_bytes_per_s` already exists**, so collective SOL needs no change to
`dryrun/cost/kernel/data/gpu_specs.yaml`.

**8x H20 are available**, so `all_reduce` is measurable at world sizes 2, 4, 8.

**`profiler/kernels/collective.py` already implements the NCCL spawn** via
`torch.multiprocessing.spawn` with `_collective_worker`, returning `CollectiveProfile`. Task 11
rewrites it to write a parquet table instead; the JSON dataclasses are deleted.

## Task-to-stage map

The spec defines five stages whose ordering is a safety property, not a preference. This plan's
twelve tasks map onto them as follows. **The stage-1 gate between Task 9 and Task 10 is a hard
stop:** a new op measured before it lands would make the behaviour-neutrality comparison
meaningless, because its error could not be distinguished from refactor drift.

| stage | tasks | gate |
|---|---|---|
| **0**: baseline | Task 1 | The ten numbers are captured and committed before any source edit. |
| **1**: refactor | Tasks 2-9 | Every stage-0 number reproduces **identically** (the "Stage 1 gate" section after Task 9). |
| **2**: elementwise | Task 10 | Holdout MRE with `point_count > 0`; a test asserts `predict()` has no elementwise term. |
| **3**: all_reduce | Task 11 | Holdout MRE reported; `tp=2` constructs and predicts; tp=1 validate numbers unchanged. |
| **4**: docs | Task 12 | Every figure traces to a command run in that task's Step 1. |

Tasks 10 and 11 are independent of each other and both depend only on the stage-1 gate. Within
stage 1, the order is dependency-driven: Task 5 must precede Tasks 6-8 (they consume
`KernelTable.op`), and Task 7 must follow Task 6 (Task 6 leaves one test deliberately red that
Task 7's re-keying turns green).

## File Structure

| File | Responsibility |
|---|---|
| `dryrun/cost/kernel/ops/base.py` | new. `KernelOp` protocol, `OP_REGISTRY`, and the four lookups `op_for` / `ops_for_family` / `families` / `ops_in`. |
| `dryrun/cost/kernel/ops/gemm.py` | new. `GemmKernelOp`: schema, SOL, no floor. |
| `dryrun/cost/kernel/ops/attention.py` | new. `ContextAttentionKernelOp`, `GenerationAttentionKernelOp`: shared slice columns, fitted floor. |
| `dryrun/cost/kernel/ops/elementwise.py` | new. `ElementwiseKernelOp`: bandwidth-only SOL, fitted floor. |
| `dryrun/cost/kernel/ops/collective.py` | new. `AllReduceKernelOp`: link-bandwidth SOL, `world_size` in the slice. |
| `dryrun/cost/kernel/table.py` | modified. `TableSchema` and `schema_for` deleted; `KernelTable` takes a `KernelOp`. |
| `dryrun/cost/kernel/database.py` | modified. `_sol` delegates to `op.sol()`; floors keyed by op name. |
| `dryrun/cost/kernel/holdout.py` | modified. `_make_sol_fn` and the three `_sol_for_*` helpers deleted; discovery uses `ops_in`. |
| `dryrun/cost/kernel/floor.py` | modified. `_ATTENTION_OPS` deleted; fitting loops over `op.fits_floor`; old-format rejection. |
| `dryrun/cost/kernel/sol.py` | modified. `ATTENTION_FLOOR_S` re-keyed by op name and renamed `FALLBACK_FLOORS_S`. |
| `dryrun/cost/kernel/model_ops.py` | modified. Gains `comm_queries`. |
| `dryrun/cost/rollout/composed.py` | modified. `tp>1` allowed; comm term added. |
| `dryrun/profiler/kernels/measurements.py` | modified. `write_gemm`/`write_attention` collapse into `write`. |
| `dryrun/profiler/kernels/config.py` | modified. Named sweep fields replaced by `config.sweep(op_name)`. |
| `dryrun/profiler/kernels/elementwise.py` | new. Raw fused-op timing for the three elementwise kernels. |
| `dryrun/profiler/kernels/collective.py` | modified. Writes a perf table; JSON dataclasses deleted. |
| `dryrun/cli/profile.py` | modified. `--family` choices and dispatch derived from the registry. |

Four separations are deliberate. `ops/` is a package rather than one module because each op
carries a SOL formula with real physics and a docstring explaining it; one file per family
keeps each readable. `base.py` holds only the protocol and lookups so that importing it can
never pull a GPU dependency. `elementwise.py` is separate from `gemm.py` on the profiler side
because its timing needs a live `VllmConfig` context that GEMM does not. And `sol.py` keeps the
formulas as free functions called *by* the ops rather than inlined into them, so the existing
`tests/test_kernel_sol.py` continues to test the physics directly.

---

## Task 1: Capture the behaviour-neutrality baseline

Nothing may be edited before this lands. The baseline is the only evidence the refactor
preserved behaviour, and capturing it after an edit proves nothing.

**Files:**
- Create: `.superpowers/sdd/baseline/holdout.json`, `.superpowers/sdd/baseline/validate-pchip.json`, `.superpowers/sdd/baseline/validate-shepard.json`, `.superpowers/sdd/baseline/floors-vllm.json`, `.superpowers/sdd/baseline/floors-flashinfer.json`, `.superpowers/sdd/baseline/SUMMARY.md`
- Create: `tests/test_kernel_baseline_regression.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `tests/test_kernel_baseline_regression.py` asserting every later task reproduces the ten baseline quantities. Later tasks consume it by running it.

- [ ] **Step 1: Capture the four artifacts**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/DryRun-RL
mkdir -p .superpowers/sdd/baseline

dryrun-profile kernels holdout \
    --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml \
    --fraction 1.0 --gate 100 \
    --output .superpowers/sdd/baseline/holdout.json

for surface in pchip shepard; do
  dryrun-profile kernels validate \
      --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml \
      --baseline outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
      --model-path /apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B \
      --surface "${surface}" --min-samples 1 --gate 100 \
      --output ".superpowers/sdd/baseline/validate-${surface}.json"
done

cp outputs/profile/qwen3_8b/kernel_tp1/kernel/h20/attention/vllm/*/floors.json \
   .superpowers/sdd/baseline/floors-vllm.json
cp outputs/profile/qwen3_8b/kernel_tp1/kernel/h20/attention/flashinfer/*/floors.json \
   .superpowers/sdd/baseline/floors-flashinfer.json
```

- [ ] **Step 2: Verify the captured numbers match the expected baseline**

```bash
python - <<'PY'
import json
h = {(r["op_name"], r["backend"]): r for r in json.load(open(".superpowers/sdd/baseline/holdout.json"))}
expect_h = {
    ("gemm", "vllm"): (3.91, 64),
    ("context_attention", "vllm"): (5.74, 59),
    ("generation_attention", "vllm"): (6.63, 72),
    ("context_attention", "flashinfer"): (9.87, 59),
    ("generation_attention", "flashinfer"): (6.01, 72),
}
for key, (mre, n) in expect_h.items():
    got = h[key]
    assert abs(got["mre_percent"] - mre) < 0.01, f"{key}: {got['mre_percent']} != {mre}"
    assert got["point_count"] == n, f"{key}: {got['point_count']} != {n}"
v = {r["phase"]: r for r in json.load(open(".superpowers/sdd/baseline/validate-pchip.json"))}
for phase, mre, n in (("decode", 4.84, 216), ("prefill", 3.72, 29), ("mixed", 0.48, 5)):
    assert abs(v[phase]["mre_percent"] - mre) < 0.01, f"{phase}: {v[phase]['mre_percent']}"
    assert v[phase]["bucket_count"] == n
f = json.load(open(".superpowers/sdd/baseline/floors-vllm.json"))
assert f["context"] == 32e-6 and f["generation"] == 22e-6, f
f = json.load(open(".superpowers/sdd/baseline/floors-flashinfer.json"))
assert f["context"] == 4e-6 and f["generation"] == 6e-6, f
print("BASELINE MATCHES EXPECTED")
PY
```

Expected: `BASELINE MATCHES EXPECTED`. If any assertion fires, STOP and report: the plan's
baseline table is wrong and every later gate depends on it.

- [ ] **Step 3: Write the regression test**

`.superpowers/` is gitignored, so the numbers must also live in a committed test. Create
`tests/test_kernel_baseline_regression.py`:

```python
"""The refactor must not move any measured accuracy number.

These are the ten quantities the KernelOp refactor is gated on. They were captured at
commit 72f00f5 before any edit, on H20 with Qwen3-8B at tp=1 and --fraction 1.0. The
refactor rebinds where SOL formulas are looked up, not what they compute, so every number
here must reproduce exactly. A change means the binding changed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TABLES = _REPO / "outputs" / "profile" / "qwen3_8b" / "kernel_tp1" / "kernel"
_BASELINE = _REPO / "outputs" / "profile" / "qwen3_8b" / "vllm_tp1" / "profiles" / "vllm.jsonl"
_CHECKPOINT = "/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B"

pytestmark = pytest.mark.skipif(
    not _TABLES.is_dir() or not _BASELINE.is_file(),
    reason="Measured perf tables and the vLLM baseline are machine-local.",
)

# (op_name, backend) -> (mre_percent, point_count)
_HOLDOUT = {
    ("gemm", "vllm"): (3.91, 64),
    ("context_attention", "vllm"): (5.74, 59),
    ("generation_attention", "vllm"): (6.63, 72),
    ("context_attention", "flashinfer"): (9.87, 59),
    ("generation_attention", "flashinfer"): (6.01, 72),
}

# phase -> (mre_percent, bucket_count)
_VALIDATE_PCHIP = {
    "decode": (4.84, 216),
    "prefill": (3.72, 29),
    "mixed": (0.48, 5),
}

_FLOORS = {
    "vllm": {"context_attention": 32e-6, "generation_attention": 22e-6},
    "flashinfer": {"context_attention": 4e-6, "generation_attention": 6e-6},
}


def test_holdout_accuracy_has_not_moved():
    from dryrun.cost.kernel.holdout import holdout_report

    results = {(r.op_name, r.backend): r for r in holdout_report(_TABLES, gpu="h20", fraction=1.0)}
    for key, (mre, count) in _HOLDOUT.items():
        assert key in results, f"{key} disappeared from the holdout report"
        assert results[key].point_count == count, f"{key} point count moved"
        assert results[key].mre_percent == pytest.approx(mre, abs=0.01), f"{key} MRE moved"


def test_composed_model_accuracy_has_not_moved():
    from dryrun.cost.kernel.database import KernelDatabase
    from dryrun.cost.kernel.model_ops import ModelOps
    from dryrun.cost.kernel.validate import load_baseline, score_model
    from dryrun.cost.rollout.composed import ComposedRollout

    database = KernelDatabase.from_root(_TABLES, gpu="h20", utility_surface="pchip")
    ops = ModelOps.from_checkpoint(_CHECKPOINT, tensor_parallel=1)
    model = ComposedRollout(database, ops, tensor_parallel=1)
    buckets = load_baseline(_BASELINE, min_samples=1)
    results = {r.phase: r for r in score_model(model, buckets)}
    assert set(results) == set(_VALIDATE_PCHIP)
    for phase, (mre, count) in _VALIDATE_PCHIP.items():
        assert results[phase].bucket_count == count, f"{phase} bucket count moved"
        assert results[phase].mre_percent == pytest.approx(mre, abs=0.01), f"{phase} MRE moved"


@pytest.mark.parametrize("backend", sorted(_FLOORS))
def test_fitted_floors_have_not_moved(backend):
    """Floors are keyed by op name after this refactor; the values must not change."""
    from dryrun.cost.kernel.floor import load_floors

    backend_dir = _TABLES / "h20" / "attention" / backend
    versions = sorted(item for item in backend_dir.iterdir() if item.is_dir())
    assert len(versions) == 1, f"expected one version under {backend_dir}, got {versions}"
    floors = load_floors(versions[0], backend=backend, warn_missing=False)
    assert floors is not None, f"no floors.json under {versions[0]}"
    for op_name, value in _FLOORS[backend].items():
        assert floors[op_name] == pytest.approx(value, rel=1e-9), f"{backend}/{op_name} moved"
```

Note `test_fitted_floors_have_not_moved` asserts the **new** op-name keys. It fails until
Task 7 re-keys `floors.json`, which is correct: that task is what changes the format, and the
test is what pins the values across the change.

- [ ] **Step 4: Run the two tests that must already pass**

```bash
python -m pytest tests/test_kernel_baseline_regression.py -q \
    -k "holdout_accuracy or composed_model_accuracy"
```

Expected: **2 passed**. These two assert current behaviour, so they must pass now. If either
fails, the plan's baseline is wrong: STOP and report.

- [ ] **Step 5: Write the summary and commit**

```bash
python - <<'PY'
import json
from pathlib import Path
out = ["# Behaviour-neutrality baseline", "", "Captured at commit 72f00f5 before any edit.", ""]
out += ["| quantity | value |", "|---|---|"]
for r in json.load(open(".superpowers/sdd/baseline/holdout.json")):
    out.append(f"| holdout {r['op_name']}/{r['backend']} | {r['mre_percent']:.2f}% MRE, {r['point_count']} points |")
for surface in ("pchip", "shepard"):
    for r in json.load(open(f".superpowers/sdd/baseline/validate-{surface}.json")):
        out.append(f"| validate {r['phase']} ({surface}) | {r['mre_percent']:.2f}% MRE, {r['bucket_count']} buckets |")
for backend in ("vllm", "flashinfer"):
    f = json.load(open(f".superpowers/sdd/baseline/floors-{backend}.json"))
    out.append(f"| floors {backend} | context {f['context']*1e6:.0f} us, generation {f['generation']*1e6:.0f} us |")
Path(".superpowers/sdd/baseline/SUMMARY.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
PY
git add tests/test_kernel_baseline_regression.py
git commit -m "test: pin the ten accuracy numbers the KernelOp refactor must preserve

The refactor rebinds where SOL formulas are looked up, not what they compute,
so every measured accuracy number must reproduce exactly. Captured before any
edit, because a baseline captured afterwards proves nothing.

The floor assertions use op-name keys, which floors.json does not carry yet, so
that test fails until the re-keying task lands. That is deliberate: the test is
what pins the values across the format change.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

Note `.superpowers/` is gitignored, so only the test file is committed; the JSON artifacts stay
local as the raw evidence.

---

## Task 2: The `KernelOp` protocol and the four lookups

**Files:**
- Create: `dryrun/cost/kernel/ops/__init__.py`, `dryrun/cost/kernel/ops/base.py`
- Test: `tests/test_kernel_ops_base.py`

**Interfaces:**
- Consumes: `dryrun.cost.kernel.hardware.GpuSpec`, `dryrun.cost.kernel.interp.KernelCoverageError`, `dryrun.cost.kernel.sol.SolComponents`.
- Produces:
  - `KernelOp`: a `typing.Protocol` with attributes `name: str`, `family: str`, `slice_columns: tuple[str, ...]`, `axis_columns: tuple[str, ...]`, `fits_floor: bool`, and methods `sol(spec, *, slice_key, coords, floor_s) -> SolComponents`, `parse_sweep(value: dict) -> object`, `table_columns() -> tuple[str, ...]`.
  - `OP_REGISTRY: dict[str, KernelOp]`: empty in this task; populated by Tasks 3-4 and 10-11.
  - `register_op(op: KernelOp) -> None`
  - `op_for(name: str) -> KernelOp`: raises `KernelCoverageError` naming known ops.
  - `ops_for_family(family: str) -> tuple[KernelOp, ...]`: sorted by `name`.
  - `families() -> tuple[str, ...]`: sorted distinct family names.
  - `ops_in(version_dir: Path) -> tuple[KernelOp, ...]`: ops whose `{name}_perf.parquet` exists, sorted by `name`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kernel_ops_base.py`:

```python
"""The op registry and its four lookups."""

from __future__ import annotations

from pathlib import Path

import pytest

from dryrun.cost.kernel.interp import KernelCoverageError
from dryrun.cost.kernel.ops.base import (
    OP_REGISTRY,
    families,
    op_for,
    ops_for_family,
    ops_in,
    register_op,
)
from dryrun.cost.kernel.sol import SolComponents


class _FakeOp:
    """A minimal op used to exercise the registry without touching real tables."""

    def __init__(self, name: str, family: str, fits_floor: bool = False) -> None:
        self.name = name
        self.family = family
        self.slice_columns = ("dtype",)
        self.axis_columns = ("n_elements",)
        self.fits_floor = fits_floor

    def sol(self, spec, *, slice_key, coords, floor_s=0.0) -> SolComponents:
        return SolComponents(math_s=0.0, memory_s=float(coords[0]), floor_s=floor_s)

    def parse_sweep(self, value: dict) -> object:
        return value

    def table_columns(self) -> tuple[str, ...]:
        return self.slice_columns + self.axis_columns + ("latency_s",)


@pytest.fixture
def isolated_registry():
    """Restore the real registry after a test mutates it."""
    saved = dict(OP_REGISTRY)
    OP_REGISTRY.clear()
    yield OP_REGISTRY
    OP_REGISTRY.clear()
    OP_REGISTRY.update(saved)


def test_a_registered_op_is_retrievable_by_name(isolated_registry):
    op = _FakeOp("widget", "widgets")
    register_op(op)
    assert op_for("widget") is op


def test_an_unknown_op_name_is_rejected_naming_the_known_ones(isolated_registry):
    register_op(_FakeOp("widget", "widgets"))
    with pytest.raises(KernelCoverageError, match="widget"):
        op_for("wodget")


def test_registering_the_same_name_twice_is_rejected(isolated_registry):
    """A duplicate name would make op_for's answer depend on import order."""
    register_op(_FakeOp("widget", "widgets"))
    with pytest.raises(KernelCoverageError, match="already registered"):
        register_op(_FakeOp("widget", "other"))


def test_ops_are_grouped_by_family_and_sorted(isolated_registry):
    register_op(_FakeOp("beta", "shared"))
    register_op(_FakeOp("alpha", "shared"))
    register_op(_FakeOp("solo", "lonely"))
    assert [op.name for op in ops_for_family("shared")] == ["alpha", "beta"]
    assert [op.name for op in ops_for_family("lonely")] == ["solo"]


def test_an_unmeasured_family_yields_no_ops(isolated_registry):
    register_op(_FakeOp("widget", "widgets"))
    assert ops_for_family("absent") == ()


def test_families_are_distinct_and_sorted(isolated_registry):
    register_op(_FakeOp("beta", "shared"))
    register_op(_FakeOp("alpha", "shared"))
    register_op(_FakeOp("solo", "lonely"))
    assert families() == ("lonely", "shared")


def test_ops_in_finds_only_the_parquet_files_present(isolated_registry, tmp_path):
    register_op(_FakeOp("present", "f"))
    register_op(_FakeOp("absent", "f"))
    (tmp_path / "present_perf.parquet").write_bytes(b"")
    assert [op.name for op in ops_in(tmp_path)] == ["present"]


def test_ops_in_on_a_missing_directory_yields_nothing(isolated_registry, tmp_path):
    register_op(_FakeOp("present", "f"))
    assert ops_in(tmp_path / "nope") == ()


def test_ops_in_ignores_a_parquet_with_no_registered_op(isolated_registry, tmp_path):
    """A stray table must not crash discovery; the coverage test in Task 9 catches real gaps."""
    register_op(_FakeOp("known", "f"))
    (tmp_path / "known_perf.parquet").write_bytes(b"")
    (tmp_path / "mystery_perf.parquet").write_bytes(b"")
    assert [op.name for op in ops_in(tmp_path)] == ["known"]


def test_table_columns_puts_latency_last(isolated_registry):
    """PerfTableWriter builds its payload from this order, so it is part of the contract."""
    op = _FakeOp("widget", "widgets")
    assert op.table_columns() == ("dtype", "n_elements", "latency_s")


def test_the_registry_import_does_not_load_torch():
    """The cost path must not pay a 4.2 s GPU-stack import to price a kernel."""
    import subprocess
    import sys

    probe = (
        "import sys; import dryrun.cost.kernel.ops.base; "
        "assert 'torch' not in sys.modules, 'ops.base pulled in torch'; "
        "assert 'vllm' not in sys.modules, 'ops.base pulled in vllm'; print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_kernel_ops_base.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dryrun.cost.kernel.ops'`.

- [ ] **Step 3: Write the implementation**

Create `dryrun/cost/kernel/ops/__init__.py`:

```python
"""One class per kernel op, each owning its schema, SOL formula, and floor policy."""
```

Leave it at that single line. Populating `__init__` with re-exports would make importing any
op import all of them, and `ops_in` only needs the registry.

Create `dryrun/cost/kernel/ops/base.py`:

```python
"""
The contract one measured kernel op must satisfy, and the registry of them.

Op-specific knowledge used to be spread across ten sites: a SOL function in `sol.py`, a
`TableSchema` in `table.py`, a query dataclass and an `isinstance` branch in `database.py`, a
second SOL binding in `holdout.py`, a phase map in `floor.py`, a writer method in
`measurements.py`, a sweep field in `config.py`, a profiler entry point, and a CLI branch.
The count was the visible symptom; the defect was that `database._sol` and
`holdout._make_sol_fn` were two independent implementations of the same binding, and drift
between them would silently change utility on one side of the holdout round trip: the exact
quantity the accuracy gate measures.

`sol()` on the op is that binding, once. Signature differences between families are absorbed
by each op unpacking its own `slice_key`, whose column order that same op declares, so no
shared `**kwargs` signature is needed.

This module must stay importable without `torch`: a simulator run prices kernels from parquet
and must not pay a 4.2-second GPU-stack import to do it. Ops that measure on a GPU import
their dependencies inside the measurement call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..interp import KernelCoverageError
from ..sol import SolComponents


@runtime_checkable
class KernelOp(Protocol):
    """One measured kernel family: its table shape, its roofline, and its floor policy."""

    name: str
    family: str
    slice_columns: tuple[str, ...]
    axis_columns: tuple[str, ...]
    fits_floor: bool

    def sol(
        self,
        spec: Any,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """Return the roofline components of one shape in this op's own coordinates."""

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse this op's YAML sweep subtree, raising on an invalid grid."""

    def table_columns(self) -> tuple[str, ...]:
        """Return the non-provenance columns of this op's perf table, latency last."""


OP_REGISTRY: dict[str, KernelOp] = {}


def register_op(op: KernelOp) -> None:
    """
    Add one op to the registry.

    Args:
        op (KernelOp): The op to register.

    Raises:
        KernelCoverageError: If an op with the same name is already registered.
    """
    if op.name in OP_REGISTRY:
        raise KernelCoverageError(
            f"Kernel op {op.name!r} is already registered. Two ops sharing a name would make "
            f"op_for's answer depend on import order."
        )
    OP_REGISTRY[op.name] = op


def op_for(name: str) -> KernelOp:
    """
    Return the op measured under one name.

    Args:
        name (str): Op name, e.g. `gemm` or `generation_attention`.

    Returns:
        KernelOp: The matching op.

    Raises:
        KernelCoverageError: If no op is registered under `name`.
    """
    try:
        return OP_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(OP_REGISTRY)) or "none"
        raise KernelCoverageError(f"Unknown kernel op {name!r}. Registered ops: {known}.") from exc


def ops_for_family(family: str) -> tuple[KernelOp, ...]:
    """
    Return every op writing into one family directory.

    Args:
        family (str): Family directory name, e.g. `attention`.

    Returns:
        tuple[KernelOp, ...]: Sorted by op name; empty when the family has no ops.
    """
    return tuple(sorted((op for op in OP_REGISTRY.values() if op.family == family), key=lambda op: op.name))


def families() -> tuple[str, ...]:
    """Return the sorted distinct family directory names across every registered op."""
    return tuple(sorted({op.family for op in OP_REGISTRY.values()}))


def ops_in(version_dir: str | Path) -> tuple[KernelOp, ...]:
    """
    Return the ops whose perf table exists in one version directory.

    Replaces parsing an op name out of a parquet stem at each discovery site. A parquet with
    no registered op is skipped rather than raising, because a stray file must not break a
    report; `tests/test_kernel_op_coverage.py` is what fails on a genuinely unregistered op.

    Args:
        version_dir (str | Path): A `{root}/{gpu}/{family}/{backend}/{version}/` directory.

    Returns:
        tuple[KernelOp, ...]: Sorted by op name; empty when the directory is absent.
    """
    directory = Path(version_dir)
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            (op for op in OP_REGISTRY.values() if (directory / f"{op.name}_perf.parquet").is_file()),
            key=lambda op: op.name,
        )
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_kernel_ops_base.py -q
python -m pytest tests/ -q
ruff check dryrun/ tests/
```

Expected: **11 passed** in the new file, the suite grows by exactly the new file's test count and stays green (635 + 11 + the 2 baseline tests
from Task 1 that pass today), and no new ruff findings.

- [ ] **Step 5: Commit**

```bash
git add dryrun/cost/kernel/ops/__init__.py dryrun/cost/kernel/ops/base.py tests/test_kernel_ops_base.py
git commit -m "feat: add the KernelOp protocol and its registry

Op-specific knowledge is spread across ten sites, but the count is only the
symptom. database._sol and holdout._make_sol_fn are two independent
implementations of the same SOL binding, and drift between them would silently
change utility on one side of the holdout round trip: the quantity the
accuracy gate measures. sol() on the op is that binding, once.

Signature differences between families are absorbed by each op unpacking its
own slice_key rather than by a shared **kwargs, so GEMM keeps taking a dtype
and attention keeps taking five head and dtype parameters.

Registering a duplicate name raises: it would make op_for's answer depend on
import order. A subprocess test asserts the registry imports without torch,
since a simulator run prices kernels from parquet and must not pay a 4.2 s
GPU-stack import.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `GemmKernelOp`

The simplest op, so it is where the `sol()` contract gets pinned down. Note the class is named
`GemmKernelOp`, not `GemmOp`: `dryrun/cost/kernel/model_ops.py:30` already defines `GemmOp`
meaning "one GEMM in the per-step op list", a different concept read by two existing test
files. Do not rename that one.

**Files:**
- Create: `dryrun/cost/kernel/ops/gemm.py`
- Test: `tests/test_kernel_ops_gemm.py`

**Interfaces:**
- Consumes: `KernelOp`, `register_op` (Task 2); `dryrun.cost.kernel.sol.gemm_sol`.
- Produces: `GemmKernelOp` with `name = "gemm"`, `family = "gemm"`,
  `slice_columns = ("gemm_dtype", "n", "k")`, `axis_columns = ("m",)`, `fits_floor = False`,
  and `GEMM_OP: GemmKernelOp` registered at import. Also `GemmSweep` re-exported from
  `dryrun.profiler.kernels.config` is NOT used here: `parse_sweep` imports it lazily so the
  cost path stays free of the profiler package.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kernel_ops_gemm.py`:

```python
"""GemmKernelOp: its schema, its roofline, and its lack of a floor."""

from __future__ import annotations

import pytest

from dryrun.cost.kernel.hardware import load_gpu_spec
from dryrun.cost.kernel.ops.base import op_for
from dryrun.cost.kernel.ops.gemm import GEMM_OP
from dryrun.cost.kernel.sol import gemm_sol


@pytest.fixture(scope="module")
def spec():
    return load_gpu_spec("h20")


def test_the_op_is_registered_under_its_table_name():
    assert op_for("gemm") is GEMM_OP


def test_the_schema_matches_the_measured_parquet_columns():
    """These names are what PerfTableWriter writes and KernelTable reads."""
    assert GEMM_OP.slice_columns == ("gemm_dtype", "n", "k")
    assert GEMM_OP.axis_columns == ("m",)
    assert GEMM_OP.family == "gemm"


def test_latency_is_the_last_table_column():
    assert GEMM_OP.table_columns() == ("gemm_dtype", "n", "k", "m", "latency_s")


def test_gemm_takes_no_fitted_floor():
    """GEMM is compute- or bandwidth-limited, not dispatch-limited; its MRE is 3.91%."""
    assert GEMM_OP.fits_floor is False


def test_sol_reproduces_the_free_function_it_delegates_to(spec):
    """The op must not re-derive the roofline; it binds the existing formula to a slice."""
    slice_key = ("bfloat16", 24576, 4096)
    coords = (128.0,)
    expected = gemm_sol(spec, m=128, n=24576, k=4096, dtype="bfloat16")
    got = GEMM_OP.sol(spec, slice_key=slice_key, coords=coords)
    assert got.math_s == pytest.approx(expected.math_s, rel=1e-12)
    assert got.memory_s == pytest.approx(expected.memory_s, rel=1e-12)
    assert got.latency_s == pytest.approx(expected.latency_s, rel=1e-12)


def test_a_floor_argument_is_ignored_rather_than_applied(spec):
    """fits_floor is False, so passing one must not change the answer."""
    slice_key = ("bfloat16", 4096, 4096)
    coords = (1.0,)
    without = GEMM_OP.sol(spec, slice_key=slice_key, coords=coords).latency_s
    with_floor = GEMM_OP.sol(spec, slice_key=slice_key, coords=coords, floor_s=1.0).latency_s
    assert with_floor == pytest.approx(without, rel=1e-12)


def test_sol_reads_n_and_k_from_the_slice_not_the_coords(spec):
    """m is the only axis; n and k are categorical, so they must come from slice_key."""
    coords = (64.0,)
    small = GEMM_OP.sol(spec, slice_key=("bfloat16", 4096, 4096), coords=coords).latency_s
    large = GEMM_OP.sol(spec, slice_key=("bfloat16", 24576, 4096), coords=coords).latency_s
    assert large > small


def test_a_non_positive_m_is_rejected(spec):
    with pytest.raises(ValueError, match="m"):
        GEMM_OP.sol(spec, slice_key=("bfloat16", 4096, 4096), coords=(0.0,))


def test_parse_sweep_builds_the_gemm_grid():
    sweep = GEMM_OP.parse_sweep({"token_counts": [1, 16, 4096]})
    assert sweep.token_counts == (1, 16, 4096)
    assert sweep.logits_request_counts == (1, 2, 4, 8, 16, 32, 48, 64)


def test_parse_sweep_rejects_an_empty_grid():
    from dryrun.profiler.schema import ProfileValidationError

    with pytest.raises(ProfileValidationError, match="token_counts"):
        GEMM_OP.parse_sweep({"token_counts": []})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_kernel_ops_gemm.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dryrun.cost.kernel.ops.gemm'`.

- [ ] **Step 3: Write the implementation**

Create `dryrun/cost/kernel/ops/gemm.py`:

```python
"""
The unified GEMM op: every linear layer in the model, sliced by `(dtype, n, k)`.

One table serves forward, dgrad, and wgrad, because a GEMM's cost depends only on `(m, n, k)`
and the backward passes are the same arithmetic with permuted axes. `shapes.gemm_wgrad_shapes`
emits the permuted triples into this same table, so backward needs no separate op.

`m` is the only interpolation axis. `n` and `k` are categorical: blending across output or
input widths would mix kernels with different tiling, and a `(4096, 4096)` GEMM tells you
little about a `(151936, 4096)` one.

No fitted floor. GEMM's holdout MRE is 3.91% without one, and the regime is compute- or
bandwidth-limited rather than dispatch-limited, so a floor would have nothing to correct.
"""

from __future__ import annotations

from typing import Any

from ..hardware import GpuSpec
from ..sol import SolComponents, gemm_sol
from .base import register_op


class GemmKernelOp:
    """The unified GEMM perf table and its roofline.

    Named `GemmKernelOp` rather than `GemmOp` because `model_ops.GemmOp` already means
    "one GEMM in the per-step op list", which is a different concept.
    """

    name = "gemm"
    family = "gemm"
    slice_columns = ("gemm_dtype", "n", "k")
    axis_columns = ("m",)
    fits_floor = False

    def sol(
        self,
        spec: GpuSpec,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """
        Return the roofline of one `(m, k) @ (k, n)` GEMM.

        Args:
            spec (GpuSpec): Hardware the roofline is evaluated against.
            slice_key (tuple[Any, ...]): `(gemm_dtype, n, k)`.
            coords (tuple[float, ...]): `(m,)`.
            floor_s (float): Ignored; `fits_floor` is False. Accepted so every op shares
                one call signature.

        Returns:
            SolComponents: Compute and memory halves in seconds.

        Raises:
            ValueError: If `m`, `n`, or `k` is not positive.
        """
        dtype, n, k = slice_key[0], int(slice_key[1]), int(slice_key[2])
        return gemm_sol(spec, m=int(coords[0]), n=n, k=k, dtype=dtype)

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the `gemm:` sweep subtree into a `GemmSweep`."""
        # Imported lazily: `dryrun.profiler.kernels.config` reaches the profiler package, and
        # a simulator run that only prices kernels must not load it. Four dots because this
        # module is `dryrun.cost.kernel.ops.gemm`; Step 4 verifies the depth resolves.
        from ....profiler.kernels.config import GemmSweep

        return GemmSweep.from_dict(value)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)


GEMM_OP = GemmKernelOp()
register_op(GEMM_OP)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_kernel_ops_gemm.py -q
python -m pytest tests/test_kernel_ops_base.py -q
python -m pytest tests/ -q
ruff check dryrun/ tests/
python -c "
import sys
import dryrun.cost.kernel.ops.gemm
print('torch imported:', 'torch' in sys.modules)"
```

Expected: **10 passed**, base still **11 passed**, the suite grows by exactly the new file's test count and stays green, no new ruff findings,
and the last command prints **False**: `parse_sweep`'s import is lazy, so merely importing the
op must not reach the profiler package.

- [ ] **Step 5: Commit**

```bash
git add dryrun/cost/kernel/ops/gemm.py tests/test_kernel_ops_gemm.py
git commit -m "feat: add GemmKernelOp

The simplest op, so it pins the sol() contract: read the categorical dimensions
from slice_key, the interpolation axis from coords, and delegate to the existing
free function rather than re-deriving the roofline.

Named GemmKernelOp because model_ops.GemmOp already means 'one GEMM in the
per-step op list', a different concept that two test files read.

fits_floor is False and a passed floor is ignored: GEMM scores 3.91% without
one and its regime is compute- or bandwidth-limited, so a floor would have
nothing to correct. A test asserts passing one changes nothing.

parse_sweep imports GemmSweep lazily so importing the op does not drag the
profiler package into a simulator run; a test asserts torch stays absent.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: The two attention ops

Both phases share one family directory, one set of slice columns, and one floor policy, so they
belong in one module. They differ only in their third axis (`isl` vs `step`) and which SOL
formula they call.

**Files:**
- Create: `dryrun/cost/kernel/ops/attention.py`
- Test: `tests/test_kernel_ops_attention.py`

**Interfaces:**
- Consumes: `KernelOp`, `register_op` (Task 2); `context_attention_sol`, `generation_attention_sol` from `dryrun.cost.kernel.sol`.
- Produces: `ContextAttentionKernelOp` (`name = "context_attention"`, `axis_columns = ("num_heads", "batch_size", "isl")`), `GenerationAttentionKernelOp` (`name = "generation_attention"`, `axis_columns = ("num_heads", "batch_size", "step")`), both with `family = "attention"`, `fits_floor = True`, and `slice_columns = ("num_key_value_heads", "head_dim", "attn_dtype", "kv_cache_dtype", "block_size", "window_size")`. Registered as `CONTEXT_ATTENTION_OP` and `GENERATION_ATTENTION_OP`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kernel_ops_attention.py`:

```python
"""The two attention ops: shared slicing, distinct third axis, fitted floor."""

from __future__ import annotations

import pytest

from dryrun.cost.kernel.hardware import load_gpu_spec
from dryrun.cost.kernel.ops.attention import CONTEXT_ATTENTION_OP, GENERATION_ATTENTION_OP
from dryrun.cost.kernel.ops.base import op_for
from dryrun.cost.kernel.sol import context_attention_sol, generation_attention_sol

# (num_kv_heads, head_dim, attn_dtype, kv_cache_dtype, block_size, window_size)
_SLICE = (8, 128, "bfloat16", "bfloat16", 16, 0)


@pytest.fixture(scope="module")
def spec():
    return load_gpu_spec("h20")


def test_both_ops_are_registered_under_their_table_names():
    assert op_for("context_attention") is CONTEXT_ATTENTION_OP
    assert op_for("generation_attention") is GENERATION_ATTENTION_OP


def test_both_phases_share_one_family_and_one_slice_schema():
    """They live in one directory and one floors.json, so their slicing must agree."""
    assert CONTEXT_ATTENTION_OP.family == GENERATION_ATTENTION_OP.family == "attention"
    assert CONTEXT_ATTENTION_OP.slice_columns == GENERATION_ATTENTION_OP.slice_columns
    assert CONTEXT_ATTENTION_OP.slice_columns == (
        "num_key_value_heads",
        "head_dim",
        "attn_dtype",
        "kv_cache_dtype",
        "block_size",
        "window_size",
    )


def test_the_phases_differ_only_in_their_third_axis():
    assert CONTEXT_ATTENTION_OP.axis_columns == ("num_heads", "batch_size", "isl")
    assert GENERATION_ATTENTION_OP.axis_columns == ("num_heads", "batch_size", "step")


def test_both_phases_take_a_fitted_floor():
    """Decode latency is flat across a 64x KV increase at batch 1, so dispatch dominates."""
    assert CONTEXT_ATTENTION_OP.fits_floor is True
    assert GENERATION_ATTENTION_OP.fits_floor is True


def test_context_sol_reproduces_the_free_function(spec):
    coords = (32.0, 4.0, 1024.0)
    expected = context_attention_sol(
        spec,
        batch_size=4,
        seq_len=1024,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        window_size=0,
        floor_s=32e-6,
    )
    got = CONTEXT_ATTENTION_OP.sol(spec, slice_key=_SLICE, coords=coords, floor_s=32e-6)
    assert got.latency_s == pytest.approx(expected.latency_s, rel=1e-12)
    assert got.floor_s == pytest.approx(32e-6, rel=1e-12)


def test_generation_sol_reproduces_the_free_function(spec):
    coords = (32.0, 16.0, 2048.0)
    expected = generation_attention_sol(
        spec,
        batch_size=16,
        step=2048,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        window_size=0,
        floor_s=22e-6,
    )
    got = GENERATION_ATTENTION_OP.sol(spec, slice_key=_SLICE, coords=coords, floor_s=22e-6)
    assert got.latency_s == pytest.approx(expected.latency_s, rel=1e-12)


def test_the_floor_binds_when_the_work_is_small(spec):
    """A floor inside max() is a lower bound; at batch 1 step 1 it must be what binds."""
    tiny = GENERATION_ATTENTION_OP.sol(spec, slice_key=_SLICE, coords=(32.0, 1.0, 1.0), floor_s=22e-6)
    assert tiny.latency_s == pytest.approx(22e-6, rel=1e-12)
    assert tiny.math_s < 22e-6
    assert tiny.memory_s < 22e-6


def test_the_floor_does_not_bind_when_the_work_is_large(spec):
    """It is max(floor, work), not floor + work; deep KV must not pay the floor twice."""
    big = GENERATION_ATTENTION_OP.sol(spec, slice_key=_SLICE, coords=(32.0, 64.0, 32768.0), floor_s=22e-6)
    assert big.latency_s > 22e-6
    assert big.latency_s == pytest.approx(max(big.math_s, big.memory_s), rel=1e-12)


def test_both_ops_put_latency_last_in_their_columns():
    for op, axis in ((CONTEXT_ATTENTION_OP, "isl"), (GENERATION_ATTENTION_OP, "step")):
        assert op.table_columns() == (
            "num_key_value_heads",
            "head_dim",
            "attn_dtype",
            "kv_cache_dtype",
            "block_size",
            "window_size",
            "num_heads",
            "batch_size",
            axis,
            "latency_s",
        )


def test_parse_sweep_builds_the_attention_grid():
    sweep = CONTEXT_ATTENTION_OP.parse_sweep(
        {"batch_sizes": [1, 16], "context_seq_lens": [128, 1024], "generation_steps": [0, 2048]}
    )
    assert sweep.batch_sizes == (1, 16)
    assert sweep.context_seq_lens == (128, 1024)
    assert sweep.generation_steps == (0, 2048)


def test_both_phases_parse_the_same_sweep_subtree():
    """One `attention:` block configures both phases, so both must accept it."""
    payload = {"batch_sizes": [1], "context_seq_lens": [128], "generation_steps": [0]}
    assert CONTEXT_ATTENTION_OP.parse_sweep(payload) == GENERATION_ATTENTION_OP.parse_sweep(payload)


def test_a_negative_generation_step_is_rejected(spec):
    with pytest.raises(ValueError, match="step"):
        GENERATION_ATTENTION_OP.sol(spec, slice_key=_SLICE, coords=(32.0, 1.0, -1.0), floor_s=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_kernel_ops_attention.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dryrun.cost.kernel.ops.attention'`.

- [ ] **Step 3: Write the implementation**

Create `dryrun/cost/kernel/ops/attention.py`:

```python
"""
The two attention phases, which share a table directory but not an axis.

Prefill and decode are different regimes: prefill attention is quadratic in sequence length
while decode is linear in KV history, so they are separate ops with separate tables. They
share one family directory, one set of slice columns, and one `floors.json`, because they run
through the same backend and pay the same dispatch path.

Both take a fitted floor. Real H20 data shows decode latency flat at ~70 us across a 64x
increase in KV work at batch 1, which is conclusive that the cost is `max(floor, work)` rather
than `floor + work`. Placing the floor inside the max lifted the holdout gate from >20% MRE to
under 10%. The floor is a property of the backend's dispatch path rather than the silicon, so
its value is fitted per `(backend, version)` and stored beside the table; see `floor.py`.
"""

from __future__ import annotations

from typing import Any

from ..hardware import GpuSpec
from ..sol import SolComponents, context_attention_sol, generation_attention_sol
from .base import register_op

_SLICE_COLUMNS = (
    "num_key_value_heads",
    "head_dim",
    "attn_dtype",
    "kv_cache_dtype",
    "block_size",
    "window_size",
)


def _unpack(slice_key: tuple[Any, ...], coords: tuple[float, ...]) -> dict[str, Any]:
    """Return the SOL keyword arguments both phases share."""
    num_kv_heads, head_dim, attn_dtype, kv_cache_dtype, _block_size, window_size = slice_key
    return dict(
        num_heads=int(coords[0]),
        batch_size=int(coords[1]),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        attn_dtype=attn_dtype,
        kv_cache_dtype=kv_cache_dtype,
        window_size=int(window_size),
    )


def _parse_attention_sweep(value: dict[str, Any]) -> Any:
    """Parse the shared `attention:` sweep subtree."""
    # Lazy for the same reason as GemmKernelOp.parse_sweep: the cost path must not import the
    # profiler package.
    from ....profiler.kernels.config import AttentionSweep

    return AttentionSweep.from_dict(value)


class ContextAttentionKernelOp:
    """Causal prefill attention, interpolated over prompt length."""

    name = "context_attention"
    family = "attention"
    slice_columns = _SLICE_COLUMNS
    axis_columns = ("num_heads", "batch_size", "isl")
    fits_floor = True

    def sol(
        self,
        spec: GpuSpec,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """
        Return the roofline of one causal prefill attention call.

        Args:
            spec (GpuSpec): Hardware the roofline is evaluated against.
            slice_key (tuple[Any, ...]): This op's `slice_columns` values.
            coords (tuple[float, ...]): `(num_heads, batch_size, isl)`.
            floor_s (float): Fitted dispatch floor in seconds, applied inside the `max`.

        Returns:
            SolComponents: Compute, memory, and floor components in seconds.
        """
        return context_attention_sol(
            spec, seq_len=int(coords[2]), floor_s=floor_s, **_unpack(slice_key, coords)
        )

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the shared `attention:` sweep subtree."""
        return _parse_attention_sweep(value)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)


class GenerationAttentionKernelOp:
    """Paged decode attention, interpolated over KV history depth."""

    name = "generation_attention"
    family = "attention"
    slice_columns = _SLICE_COLUMNS
    axis_columns = ("num_heads", "batch_size", "step")
    fits_floor = True

    def sol(
        self,
        spec: GpuSpec,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """
        Return the roofline of one paged decode attention call.

        Args:
            spec (GpuSpec): Hardware the roofline is evaluated against.
            slice_key (tuple[Any, ...]): This op's `slice_columns` values.
            coords (tuple[float, ...]): `(num_heads, batch_size, step)`.
            floor_s (float): Fitted dispatch floor in seconds, applied inside the `max`.

        Returns:
            SolComponents: Compute, memory, and floor components in seconds.
        """
        return generation_attention_sol(
            spec, step=int(coords[2]), floor_s=floor_s, **_unpack(slice_key, coords)
        )

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the shared `attention:` sweep subtree."""
        return _parse_attention_sweep(value)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)


CONTEXT_ATTENTION_OP = ContextAttentionKernelOp()
GENERATION_ATTENTION_OP = GenerationAttentionKernelOp()
register_op(CONTEXT_ATTENTION_OP)
register_op(GENERATION_ATTENTION_OP)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_kernel_ops_attention.py -q
python -m pytest tests/ -q
ruff check dryrun/ tests/
```

Expected: **12 passed**, the suite grows by exactly the new file's test count and stays green, no new ruff findings.

- [ ] **Step 5: Commit**

```bash
git add dryrun/cost/kernel/ops/attention.py tests/test_kernel_ops_attention.py
git commit -m "feat: add the two attention kernel ops

Prefill and decode are different regimes: one quadratic in sequence length,
one linear in KV history, so they stay separate ops with separate tables. They
share a family directory, slice columns, and floors.json, because they run
through one backend and pay one dispatch path, so they live in one module with
a shared slice unpacker.

Both take a fitted floor applied inside max() rather than added. Tests assert
both halves of that: the floor binds at batch 1 step 1, and at deep KV the
result equals max(math, memory) with no floor contribution, so a deep-KV step
does not pay dispatch twice.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Retire `TableSchema` in favour of the op

`KernelTable` currently takes a `TableSchema`. This task makes it take a `KernelOp` and deletes
`TableSchema`, `schema_for`, and the three `*_SCHEMA` constants. Six test files reference those
names; per the spec, mechanical renames are the only permitted test edit and **no assertion may
be weakened or deleted**.

**Files:**
- Modify: `dryrun/cost/kernel/table.py`
- Modify: `dryrun/profiler/kernels/measurements.py:22,76,124` (import and the two `schema_for` calls)
- Modify: `tests/test_kernel_table.py`, `tests/test_kernel_holdout.py`, `tests/test_kernel_floor.py`, `tests/test_kernel_measurements.py`, `tests/test_kernel_surface_wiring.py`, `tests/test_kernel_attention_vllm.py`, `tests/test_kernel_attention_flashinfer.py`, `tests/test_kernel_gemm.py` (mechanical rename only)
- Modify: `dryrun/cost/kernel/holdout.py:23`, `dryrun/cost/kernel/floor.py:34`, `dryrun/cost/kernel/database.py:26` (imports only; their bodies change in Tasks 6-8)

**Interfaces:**
- Consumes: `op_for`, `KernelOp` (Task 2); `GEMM_OP` (Task 3); the attention ops (Task 4).
- Produces:
  - `KernelTable.load(path, op: KernelOp) -> KernelTable`: second parameter is now an op.
  - `KernelTable.op -> KernelOp`: replaces the `schema` property.
  - `KernelTable.grid(slice_key, sol_fn, *, surface="shepard")`: unchanged signature.
  - `dryrun.cost.kernel.table` no longer exports `TableSchema`, `schema_for`, `GEMM_SCHEMA`, `CONTEXT_ATTENTION_SCHEMA`, or `GENERATION_ATTENTION_SCHEMA`.

- [ ] **Step 1: Record the exact rename sites before editing**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/DryRun-RL
grep -rn "TableSchema\|schema_for\|GEMM_SCHEMA\|CONTEXT_ATTENTION_SCHEMA\|GENERATION_ATTENTION_SCHEMA\|\.schema\b" \
    dryrun/ tests/ --include=*.py | tee .superpowers/sdd/task5-sites.txt | wc -l
```

Expected: a non-empty list. Keep the file; Step 5 diffs against it to prove every site moved.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_kernel_ops_base.py`:

```python
def test_the_table_module_no_longer_exports_a_schema_type():
    """TableSchema is absorbed into KernelOp; leaving an alias invites two sources of truth."""
    import dryrun.cost.kernel.table as table_module

    for name in (
        "TableSchema",
        "schema_for",
        "GEMM_SCHEMA",
        "CONTEXT_ATTENTION_SCHEMA",
        "GENERATION_ATTENTION_SCHEMA",
    ):
        assert not hasattr(table_module, name), f"{name} should be gone from table.py"


def test_a_table_loads_against_an_op_and_exposes_it():
    """KernelTable's second parameter is the op, and the op is readable back off the table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from dryrun.cost.kernel.ops.gemm import GEMM_OP
    from dryrun.cost.kernel.table import KernelTable

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gemm_perf.parquet"
        pq.write_table(
            pa.table(
                {
                    "framework": ["vllm"],
                    "version": ["0.0.0"],
                    "device": ["NVIDIA H20"],
                    "op_name": ["gemm"],
                    "kernel_source": ["torch.nn.functional.linear"],
                    "gemm_dtype": ["bfloat16"],
                    "m": [64],
                    "n": [4096],
                    "k": [4096],
                    "latency_s": [1.0e-4],
                }
            ),
            path,
        )
        table = KernelTable.load(path, GEMM_OP)
        assert table.op is GEMM_OP
        assert table.row_count == 1
```

- [ ] **Step 3: Run test to verify it fails**

```bash
python -m pytest tests/test_kernel_ops_base.py -q \
    -k "no_longer_exports_a_schema_type or loads_against_an_op"
```

Expected: FAIL: the first because `TableSchema` still exists, the second with
`AttributeError: 'GemmKernelOp' object has no attribute 'required_columns'` or a `TypeError`
from `load`.

- [ ] **Step 4: Rewrite `table.py`**

In `dryrun/cost/kernel/table.py`, replace the import block, delete the schema definitions, and
rewrite the class to hold an op.

Replace lines 13-26 (imports through `_LATENCY_COLUMN`) with:

```python
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .interp import KernelCoverageError, UtilitySample
from .ops.base import KernelOp
from .surface import SURFACE_REGISTRY, UtilitySurface

_PROVENANCE_COLUMNS = ("framework", "version", "device", "op_name", "kernel_source")
_LATENCY_COLUMN = "latency_s"


def required_columns(op: KernelOp) -> tuple[str, ...]:
    """Return every column a conforming table for `op` must carry."""
    return _PROVENANCE_COLUMNS + op.table_columns()
```

Delete the `TableSchema` dataclass, `_ATTENTION_SLICE_COLUMNS`, `GEMM_SCHEMA`,
`CONTEXT_ATTENTION_SCHEMA`, `GENERATION_ATTENTION_SCHEMA`, and the `schema_for` function
entirely. Their content now lives on the ops.

In `KernelTable.__init__`, rename the first parameter and attribute:

```python
    def __init__(
        self,
        op: KernelOp,
        slices: dict[tuple[Any, ...], dict[tuple[float, ...], float]],
        kernel_sources: tuple[str, ...],
        devices: tuple[str, ...],
    ) -> None:
        self._op = op
        self._slices = slices
        self._kernel_sources = kernel_sources
        self._devices = devices
        self._grids: dict[tuple[tuple[Any, ...], Callable[[tuple[float, ...]], float], str], UtilitySurface] = {}
```

In `load`, take an op and read the column names off it:

```python
    @classmethod
    def load(cls, path: str | Path, op: KernelOp) -> KernelTable:
        """
        Read a parquet perf table.

        Args:
            path (str | Path): Parquet file written by `PerfTableWriter`.
            op (KernelOp): The op whose column contract the file must satisfy.

        Returns:
            KernelTable: Grouped measurements ready for grid construction.

        Raises:
            KernelCoverageError: If the file is absent, a column is missing, or a
                latency is not positive.
        """
        source = Path(path)
        if not source.is_file():
            raise KernelCoverageError(f"Perf table {source} does not exist.")
        table = pq.read_table(source)
        needed = required_columns(op)
        missing = [name for name in needed if name not in table.column_names]
        if missing:
            raise KernelCoverageError(
                f"Perf table {source} is missing columns: {', '.join(missing)}."
            )
        columns = {name: table.column(name).to_pylist() for name in needed}

        slices: dict[tuple[Any, ...], dict[tuple[float, ...], float]] = {}
        for index, latency in enumerate(columns[_LATENCY_COLUMN]):
            value = float(latency)
            if value <= 0.0:
                raise KernelCoverageError(
                    f"Perf table {source} row {index} has non-positive latency {value!r}."
                )
            slice_key = tuple(columns[name][index] for name in op.slice_columns)
            coords = tuple(float(columns[name][index]) for name in op.axis_columns)
            bucket = slices.setdefault(slice_key, {})
            previous = bucket.get(coords)
            bucket[coords] = value if previous is None else min(previous, value)

        sources = tuple(sorted({str(item) for item in columns["kernel_source"]}))
        device_values = tuple(sorted({str(item) for item in columns["device"]}))
        return cls(op, slices, sources, device_values)
```

Replace the `schema` property with:

```python
    @property
    def op(self) -> KernelOp:
        """Return the op this table was loaded against."""
        return self._op
```

In `rows`, change the error message's `self._schema.op_name` to `self._op.name`.

- [ ] **Step 5: Repoint every caller and test**

In `dryrun/profiler/kernels/measurements.py`, replace the `schema_for` import with
`from dryrun.cost.kernel.ops.base import op_for` and change `table_path`:

```python
        op = op_for(op_name)
        return (
            Path(self.output_root)
            / gpu_slug(self.device)
            / op.family
            / self.framework
            / self.version
            / f"{op_name}_perf.parquet"
        )
```

and in `write_attention`, change the guard to `if op_for(op_name).family != "attention":`.

In `dryrun/cost/kernel/holdout.py`, `floor.py`, and `database.py`, change
`from .table import KernelTable, schema_for` to `from .table import KernelTable` and add
`from .ops.base import op_for` where `schema_for` was used, replacing each
`schema_for(x)` with `op_for(x)` and each `schema.family` with `op.family`. Their remaining
bodies change in Tasks 6-8; this step only keeps them importable.

In the eight test files, apply the mechanical rename: `schema_for("gemm")` -> `op_for("gemm")`,
`GEMM_SCHEMA` -> `GEMM_OP` (importing from `dryrun.cost.kernel.ops.gemm`),
`CONTEXT_ATTENTION_SCHEMA` -> `CONTEXT_ATTENTION_OP` and `GENERATION_ATTENTION_SCHEMA` ->
`GENERATION_ATTENTION_OP` (from `dryrun.cost.kernel.ops.attention`), `table.schema` ->
`table.op`, and `schema.op_name` -> `op.name`. Change **no expected values**: if a test's
asserted number moves, the refactor was not behaviour-neutral and the code is what must change.

Then prove every site moved:

```bash
grep -rn "TableSchema\|schema_for\|GEMM_SCHEMA\|CONTEXT_ATTENTION_SCHEMA\|GENERATION_ATTENTION_SCHEMA" \
    dryrun/ tests/ --include=*.py || echo "ALL SCHEMA REFERENCES GONE"
```

Expected: `ALL SCHEMA REFERENCES GONE`.

- [ ] **Step 6: Run the tests and the baseline gate**

```bash
python -m pytest tests/test_kernel_ops_base.py tests/test_kernel_table.py \
    tests/test_kernel_measurements.py tests/test_kernel_surface_wiring.py -q
python -m pytest tests/test_kernel_baseline_regression.py -q \
    -k "holdout_accuracy or composed_model_accuracy"
python -m pytest tests/ -q
ruff check dryrun/ tests/
git diff --stat tests/
```

Expected: the targeted files pass, the **2 baseline tests still pass** (this is the gate: the
accuracy numbers must not have moved), the suite grows by exactly the new file's test count and stays green, no new ruff findings. The
`git diff --stat tests/` output must show only the eight files this task names; a ninth means
something behavioural changed.

- [ ] **Step 7: Commit**

```bash
git add dryrun/cost/kernel/table.py dryrun/profiler/kernels/measurements.py \
        dryrun/cost/kernel/holdout.py dryrun/cost/kernel/floor.py dryrun/cost/kernel/database.py \
        tests/test_kernel_ops_base.py tests/test_kernel_table.py tests/test_kernel_holdout.py \
        tests/test_kernel_floor.py tests/test_kernel_measurements.py \
        tests/test_kernel_surface_wiring.py tests/test_kernel_attention_vllm.py \
        tests/test_kernel_attention_flashinfer.py tests/test_kernel_gemm.py
git commit -m "refactor: replace TableSchema with the op it duplicated

TableSchema carried op_name, family, slice_columns, and axis_columns: exactly
what a KernelOp already declares. Keeping both would have left two sources of
truth for a table's shape, so KernelTable now takes an op and the schema type,
its three constants, and schema_for are deleted rather than aliased.

Test edits are mechanical renames only, and no expected value changed: the
baseline accuracy tests pass unmodified, which is the evidence this moved
lookups rather than arithmetic.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Route `database._sol` through the op

Half of the duplication this plan exists to remove.

**Files:**
- Modify: `dryrun/cost/kernel/database.py:298-328` (`_sol`) and `359-373` (`_grid`)
- Test: `tests/test_kernel_database_op_binding.py`

**Interfaces:**
- Consumes: `op_for` (Task 2), `KernelTable.op` (Task 5).
- Produces: `KernelDatabase._sol(spec, query, coords, backend=None) -> float` delegating to
  `op_for(query.op_name).sol(...)`. No public signature changes; `query_gemm`,
  `query_attention`, and `utility` keep their behaviour.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kernel_database_op_binding.py`:

```python
"""The database must obtain SOL from the op, not from its own branch on query type."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TABLES = _REPO / "outputs" / "profile" / "qwen3_8b" / "kernel_tp1" / "kernel"

pytestmark = pytest.mark.skipif(
    not _TABLES.is_dir(), reason="Measured perf tables are machine-local."
)


def test_the_database_has_no_isinstance_branch_on_query_type():
    """A type switch here is the duplication that let database and holdout drift apart."""
    import inspect

    from dryrun.cost.kernel.database import KernelDatabase

    source = inspect.getsource(KernelDatabase._sol)
    assert "isinstance" not in source, "_sol still branches on query type instead of asking the op"
    assert "GemmQuery" not in source, "_sol still names a concrete query class"


def test_a_gemm_query_prices_through_the_gemm_op():
    from dryrun.cost.kernel.database import GemmQuery, KernelDatabase
    from dryrun.cost.kernel.ops.gemm import GEMM_OP

    database = KernelDatabase.from_root(_TABLES, gpu="h20")
    query = GemmQuery(m=128, n=24576, k=4096)
    expected_sol = GEMM_OP.sol(
        database.gpu_spec, slice_key=query.slice_key, coords=query.coords
    ).latency_s
    utility = database.utility(query)
    assert database.query_gemm(query) == pytest.approx(expected_sol / utility, rel=1e-12)


def test_an_attention_query_prices_through_its_phase_op():
    from dryrun.cost.kernel.database import AttentionQuery, KernelDatabase
    from dryrun.cost.kernel.ops.attention import GENERATION_ATTENTION_OP

    database = KernelDatabase.from_root(_TABLES, gpu="h20")
    query = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=8, head_dim=128, step=2048
    )
    floors = database._floors_for_backend("vllm")
    expected_sol = GENERATION_ATTENTION_OP.sol(
        database.gpu_spec,
        slice_key=query.slice_key,
        coords=query.coords,
        floor_s=floors["generation_attention"],
    ).latency_s
    utility = database.utility(query)
    assert database.query_attention(query) == pytest.approx(expected_sol / utility, rel=1e-12)


def test_a_measured_shape_still_prices_to_its_measurement():
    """The round trip SOL/utility must return the stored latency at a measured point."""
    from dryrun.cost.kernel.database import GemmQuery, KernelDatabase
    from dryrun.cost.kernel.ops.base import op_for
    from dryrun.cost.kernel.table import KernelTable

    parquet = next((_TABLES / "h20" / "gemm" / "vllm").glob("*/gemm_perf.parquet"))
    table = KernelTable.load(parquet, op_for("gemm"))
    slice_key = ("bfloat16", 24576, 4096)
    coords, latency = table.rows(slice_key)[3]
    database = KernelDatabase.from_root(_TABLES, gpu="h20")
    query = GemmQuery(m=int(coords[0]), n=24576, k=4096)
    assert database.query_gemm(query) == pytest.approx(latency, rel=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_kernel_database_op_binding.py -q
```

Expected: FAIL. `test_the_database_has_no_isinstance_branch_on_query_type` fails because `_sol`
still contains `isinstance(query, GemmQuery)`; the attention test fails with `KeyError:
'generation_attention'` because floors are still keyed by phase.

- [ ] **Step 3: Rewrite `_sol`**

In `dryrun/cost/kernel/database.py`, replace the whole `_sol` method with:

```python
    def _sol(
        self,
        spec: GpuSpec,
        query: GemmQuery | AttentionQuery,
        coords: tuple[float, ...],
        backend: str | None = None,
    ) -> float:
        """Return the speed-of-light latency of one query, in seconds."""
        # The op owns the formula. Before this, database and holdout each held their own copy
        # of this binding, and drift between them would have changed utility on one side of the
        # holdout round trip: silently, since both sides would still look self-consistent.
        op = op_for(query.op_name)
        floor_s = 0.0
        if op.fits_floor:
            resolved_backend = backend or self._default_backend
            floor_s = self._floors_for_backend(resolved_backend).get(op.name, 0.0)
        return op.sol(spec, slice_key=query.slice_key, coords=coords, floor_s=floor_s).latency_s
```

Add `from .ops.base import op_for` to the imports, and drop the now-unused
`context_attention_sol`, `gemm_sol`, and `generation_attention_sol` from the `.sol` import,
keeping `ATTENTION_FLOOR_S` (renamed in Task 7).

`_grid` needs no change: it already closes over `self._sol`.

- [ ] **Step 4: Run test to verify it passes**

The attention test still needs Task 7's re-keying, so run only what this task can satisfy:

```bash
python -m pytest tests/test_kernel_database_op_binding.py -q \
    -k "no_isinstance_branch or gemm_query_prices or measured_shape_still_prices"
python -m pytest tests/test_kernel_database.py tests/test_kernel_surface_wiring.py -q
python -m pytest tests/test_kernel_baseline_regression.py -q -k "composed_model_accuracy"
ruff check dryrun/cost/kernel/database.py
```

Expected: **3 passed** in the new file, the two existing files pass untouched, the composed
baseline test passes (the accuracy gate), no ruff findings. The attention binding test stays
red until Task 7: note that in the commit message rather than weakening it.

- [ ] **Step 5: Commit**

```bash
git add dryrun/cost/kernel/database.py tests/test_kernel_database_op_binding.py
git commit -m "refactor: obtain SOL from the op inside KernelDatabase

_sol branched on isinstance(query, GemmQuery) and then on query.phase to pick a
formula, which was one of the two independent implementations of this binding.
It now asks op_for(query.op_name) and a test asserts the type switch is gone.

Floors are read only when op.fits_floor, so GEMM stops consulting a floor map it
never had an entry in.

test_an_attention_query_prices_through_its_phase_op is expected to fail until
floors are re-keyed by op name in the next task; it is left asserting the target
behaviour rather than the current one.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Re-key floors by op name

This is the format change. It deletes four phase-to-op translations and is what lets a new op
receive a fitted floor with no edit to `floor.py`.

**Files:**
- Modify: `dryrun/cost/kernel/sol.py:44-60` (rename and re-key the fallback map)
- Modify: `dryrun/cost/kernel/floor.py` (delete `_ATTENTION_OPS`, loop over `op.fits_floor`, reject the old format)
- Modify: `dryrun/cost/kernel/holdout.py:218-229` (drop the phase translation)
- Test: `tests/test_kernel_floor_op_keys.py`

**Interfaces:**
- Consumes: `op_for`, `ops_in` (Task 2), `op.fits_floor` (Tasks 3-4).
- Produces:
  - `dryrun.cost.kernel.sol.FALLBACK_FLOORS_S: dict[str, dict[str, float]]`: keyed `{backend: {op_name: floor_s}}`. `ATTENTION_FLOOR_S` is deleted.
  - `load_floors(version_dir, *, backend, warn_missing=True) -> dict[str, float] | None`: returns `{op_name: floor_s}`; raises `KernelCoverageError` on an old-format file.
  - `fit_floors(version_dir, *, backend, gpu, gpu_spec=None, candidates_us=CANDIDATE_FLOORS_US, seed=0) -> dict[str, Any]`: writes op-name keys.
  - `evaluate_holdout(...)`'s `floor_override` keeps its meaning.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kernel_floor_op_keys.py`:

```python
"""Floors are keyed by op name, and an old phase-keyed file is rejected, not misread."""

from __future__ import annotations

import json

import pytest

from dryrun.cost.kernel.floor import load_floors, write_floors
from dryrun.cost.kernel.interp import KernelCoverageError


def test_floors_round_trip_under_op_name_keys(tmp_path):
    data = {
        "context_attention": 32e-6,
        "generation_attention": 22e-6,
        "fitted_mre_percent": {"context_attention": 5.7432, "generation_attention": 6.6293},
        "candidates_us": [0, 22, 32],
        "gpu": "h20",
        "backend": "vllm",
    }
    write_floors(tmp_path, data)
    loaded = load_floors(tmp_path, backend="vllm", warn_missing=False)
    assert loaded == {"context_attention": 32e-6, "generation_attention": 22e-6}


def test_an_old_phase_keyed_file_is_rejected_naming_the_fit_command(tmp_path):
    """Reading it leniently would apply a zero floor: 34.89% MRE against 6.63% fitted."""
    (tmp_path / "floors.json").write_text(
        json.dumps({"context": 32e-6, "generation": 22e-6, "gpu": "h20", "backend": "vllm"}),
        encoding="utf-8",
    )
    with pytest.raises(KernelCoverageError, match="fit-floor"):
        load_floors(tmp_path, backend="vllm", warn_missing=False)


def test_a_missing_file_falls_back_to_the_hardcoded_map_under_op_keys(tmp_path):
    from dryrun.cost.kernel.sol import FALLBACK_FLOORS_S

    loaded = load_floors(tmp_path, backend="vllm", warn_missing=True)
    assert loaded == FALLBACK_FLOORS_S["vllm"]
    assert set(loaded) == {"context_attention", "generation_attention"}


def test_a_missing_file_returns_none_when_not_warning(tmp_path):
    assert load_floors(tmp_path, backend="vllm", warn_missing=False) is None


def test_the_fallback_map_is_keyed_by_op_name_not_phase():
    from dryrun.cost.kernel.sol import FALLBACK_FLOORS_S

    assert FALLBACK_FLOORS_S["vllm"] == {
        "context_attention": 32e-6,
        "generation_attention": 22e-6,
    }
    assert FALLBACK_FLOORS_S["flashinfer"] == {
        "context_attention": 4e-6,
        "generation_attention": 6e-6,
    }


def test_the_old_attention_floor_name_is_gone():
    """Leaving both names would let a caller read the phase-keyed map by accident."""
    import dryrun.cost.kernel.sol as sol_module

    assert not hasattr(sol_module, "ATTENTION_FLOOR_S")


def test_floor_fitting_skips_ops_that_declare_no_floor():
    """GEMM is excluded by fits_floor=False, not by absence from a hardcoded map."""
    import inspect

    from dryrun.cost.kernel import floor as floor_module

    source = inspect.getsource(floor_module)
    assert "_ATTENTION_OPS" not in source, "the phase map is still there"
    assert "fits_floor" in source, "fitting does not consult the op's floor policy"


def test_an_unregistered_op_key_in_floors_json_is_ignored(tmp_path):
    """A stale key from a removed op must not crash a load."""
    write_floors(
        tmp_path,
        {"context_attention": 32e-6, "retired_op": 9e-6, "gpu": "h20", "backend": "vllm"},
    )
    loaded = load_floors(tmp_path, backend="vllm", warn_missing=False)
    assert loaded == {"context_attention": 32e-6}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_kernel_floor_op_keys.py -q
```

Expected: FAIL: `ImportError: cannot import name 'FALLBACK_FLOORS_S'`.

- [ ] **Step 3: Re-key the fallback map in `sol.py`**

Replace the `ATTENTION_FLOOR_S` block (lines 44-60) with:

```python
# Per-backend, per-op dispatch floor in seconds, used only when a fitted floors.json is absent.
# Fitted for H20 + Qwen3-8B TP=1 (2026-08-24): these are the MRE-minimising floors for that
# hardware and those backend versions, not general truths. Run `dryrun-profile kernels fit-floor`
# to generate floors.json for the actual deployment; it is loaded in place of these values.
#
# Keyed by op name rather than by phase. The phase spelling forced four separate translations
# between `generation` and `generation_attention` (in floor fitting, in two holdout sites, and
# in the database), each of which was a place a new floor-taking op had to be threaded through.
# See `dryrun/cost/kernel/floor.py` for the fitting procedure.
FALLBACK_FLOORS_S: dict[str, dict[str, float]] = {
    "vllm": {
        "context_attention": 32e-6,
        "generation_attention": 22e-6,
    },
    "flashinfer": {
        "context_attention": 4e-6,
        "generation_attention": 6e-6,
    },
}
```

Update the module docstring's paragraph that names `ATTENTION_FLOOR_S` to name
`FALLBACK_FLOORS_S`, and change its sentence "A module-level mapping keyed by backend name is
honest" to "A module-level mapping keyed by backend and op name is honest".

- [ ] **Step 4: Rewrite the floor fitting and loading**

In `dryrun/cost/kernel/floor.py`:

Replace the imports of `ATTENTION_FLOOR_S` and `schema_for` with:

```python
from .interp import KernelCoverageError
from .ops.base import op_for, ops_in
from .sol import FALLBACK_FLOORS_S
from .table import KernelTable
```

Delete the `_ATTENTION_OPS` dict entirely and replace the `fit_floors` loop header:

```python
    for op in ops_in(version_dir):
        if not op.fits_floor:
            # GEMM declares fits_floor = False: its regime is compute- or bandwidth-limited,
            # so there is no dispatch cost for a floor to represent.
            continue
        parquet_path = version_dir / f"{op.name}_perf.parquet"
        try:
            table = KernelTable.load(parquet_path, op)
        except Exception as exc:
            dryrun_logger.warning(
                f"Skipping {op.name} floor fit for {backend!r}: could not load table: {exc}."
            )
            continue
```

Inside that loop, replace every `op_name=op_name` with `op_name=op.name`, every `phase` in a
log message with `op.name`, and the two result assignments:

```python
        fitted_floors[op.name] = best_floor_s
        fitted_mre[op.name] = round(best_mre, 4)
        dryrun_logger.info(
            f"Fitted {backend}/{op.name}: floor {best_floor_s * 1e6:.0f} us, MRE {best_mre:.2f}%."
        )
```

Rewrite `load_floors` to reject the old format and filter to registered ops:

```python
_LEGACY_PHASE_KEYS = frozenset({"context", "generation"})
_METADATA_KEYS = frozenset({"fitted_mre_percent", "candidates_us", "gpu", "backend"})


def load_floors(
    version_dir: Path,
    *,
    backend: str,
    warn_missing: bool = True,
) -> dict[str, float] | None:
    """
    Load fitted floors from `{version_dir}/floors.json`, keyed by op name.

    Args:
        version_dir (Path): The version directory that should contain `floors.json`.
        backend (str): Backend label; used in messages and to select the fallback.
        warn_missing (bool): When True, a missing file logs a warning and returns the
            hardcoded fallback. When False it returns None.

    Returns:
        dict[str, float] | None: `{op_name: floor_s}`, or None when the file is absent and
            `warn_missing` is False.

    Raises:
        KernelCoverageError: If the file carries the retired phase keys. Reading it leniently
            would apply a zero floor, which measures 34.89% MRE against 6.63% at the fitted
            22 us: a silent five-fold accuracy loss.
    """
    floors_path = version_dir / "floors.json"
    if not floors_path.is_file():
        if warn_missing:
            dryrun_logger.warning(
                f"No floors.json in {version_dir}; falling back to hardcoded FALLBACK_FLOORS_S "
                f"for backend {backend!r} (fitted for H20, may be wrong for other hardware or "
                f"backend versions). Run dryrun-profile kernels fit-floor to generate it."
            )
            return dict(FALLBACK_FLOORS_S.get(backend, {}))
        return None
    try:
        raw = json.loads(floors_path.read_text(encoding="utf-8"))
    except Exception as exc:
        dryrun_logger.warning(
            f"Could not read {floors_path}: {exc}. "
            f"Falling back to hardcoded FALLBACK_FLOORS_S for backend {backend!r}."
        )
        return dict(FALLBACK_FLOORS_S.get(backend, {}))
    if _LEGACY_PHASE_KEYS & set(raw):
        raise KernelCoverageError(
            f"{floors_path} uses the retired phase keys {sorted(_LEGACY_PHASE_KEYS & set(raw))}. "
            f"Floors are now keyed by op name. Regenerate it with "
            f"dryrun-profile kernels fit-floor."
        )
    result: dict[str, float] = {}
    for key, value in raw.items():
        if key in _METADATA_KEYS:
            continue
        try:
            op_for(key)
        except KernelCoverageError:
            # A stale key from a removed op: ignore it rather than failing a whole report.
            continue
        result[key] = float(value)
    return result
```

Finally, in `fit_floors_report`, replace the `has_attention` guard with a floor-aware one so the
walk stops assuming the family is called `attention`:

```python
            if not any(op.fits_floor for op in ops_in(version_dir)):
                continue
```

and change `attention_dir = root / slug / "attention"` to walk every family:

```python
    written: list[Path] = []
    for family_dir in sorted((root / slug).iterdir()) if (root / slug).is_dir() else []:
        if not family_dir.is_dir():
            continue
        for backend_dir in sorted(family_dir.iterdir()):
            if not backend_dir.is_dir():
                continue
            backend = backend_dir.name
            for version_dir in sorted(backend_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                if not any(op.fits_floor for op in ops_in(version_dir)):
                    continue
                dryrun_logger.info(f"Fitting floors for {backend!r} version {version_dir.name!r}...")
                data = fit_floors(
                    version_dir,
                    backend=backend,
                    gpu=slug,
                    gpu_spec=spec,
                    candidates_us=candidates_us,
                    seed=seed,
                )
                path = write_floors(version_dir, data)
                written.append(path)
                dryrun_logger.info(f"Wrote fitted floors to {path}.")
    return written
```

Replace the early `if not attention_dir.is_dir()` guard with:

```python
    gpu_dir = root / slug
    if not gpu_dir.is_dir():
        dryrun_logger.warning(f"No perf tables found at {gpu_dir}.")
        return []
```

- [ ] **Step 5: Drop the phase translation in `holdout.py`**

Replace lines 218-229 (the `floor_override` / `ATTENTION_FLOOR_S` block) with:

```python
    # Resolve the floor once for this (op, backend) pair so grid construction and the
    # predicted-SOL call use identical values: the round trip must be consistent.
    op = op_for(op_name)
    if not op.fits_floor:
        floor_s = 0.0
    elif floor_override is not None:
        floor_s = floor_override
    else:
        floor_s = FALLBACK_FLOORS_S.get(backend, {}).get(op_name, 0.0)
```

Replace the three `_sol_for_*` helpers and `_make_sol_fn` with one function:

```python
def _make_sol_fn(op_name: str, spec: GpuSpec, slice_key: tuple[Any, ...], floor_s: float = 0.0) -> Any:
    """Return the SOL callable for one op and slice."""
    op = op_for(op_name)

    def sol_fn(coords: tuple[float, ...]) -> float:
        return op.sol(spec, slice_key=slice_key, coords=coords, floor_s=floor_s).latency_s

    return sol_fn
```

In `holdout_report`, replace the attention special-casing with op-driven discovery:

```python
            for version_dir in sorted(backend_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                version_floors: dict[str, float] | None = None
                if any(op.fits_floor for op in ops_in(version_dir)):
                    version_floors = load_floors(version_dir, backend=backend, warn_missing=True)
                for op in ops_in(version_dir):
                    try:
                        table = KernelTable.load(version_dir / f"{op.name}_perf.parquet", op)
                    except KernelCoverageError:
                        continue
                    floor_override = (
                        version_floors.get(op.name) if version_floors is not None else None
                    )
                    result = evaluate_holdout(
                        table,
                        op_name=op.name,
                        backend=backend,
                        gpu_spec=spec,
                        fraction=fraction,
                        seed=seed,
                        floor_override=floor_override,
                    )
                    results.append(result)
```

Update `holdout.py`'s imports: drop `context_attention_sol`, `gemm_sol`,
`generation_attention_sol`, and `ATTENTION_FLOOR_S`; add `from .ops.base import op_for, ops_in`
and `from .sol import FALLBACK_FLOORS_S`.

- [ ] **Step 6: Regenerate the real `floors.json` files**

The committed files still carry phase keys, so every real load now raises until they are
regenerated. That is the intended breaking change.

```bash
dryrun-profile kernels fit-floor --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml
python - <<'PY'
import glob, json
for path in sorted(glob.glob("outputs/profile/qwen3_8b/kernel_tp1/kernel/h20/attention/*/*/floors.json")):
    data = json.load(open(path))
    print(path)
    print("  keys:", sorted(k for k in data if not isinstance(data[k], (dict, list))))
    assert "context" not in data and "generation" not in data, "still phase-keyed"
    assert "context_attention" in data and "generation_attention" in data, data
    print(f"  context_attention={data['context_attention']*1e6:.0f} us "
          f"generation_attention={data['generation_attention']*1e6:.0f} us")
PY
```

Expected: vllm reports **context_attention=32 us, generation_attention=22 us**; flashinfer
reports **4 us and 6 us**. Any other value means the fitting changed behaviour: STOP and
report rather than accepting the new number.

- [ ] **Step 7: Run the tests and the full baseline gate**

```bash
python -m pytest tests/test_kernel_floor_op_keys.py tests/test_kernel_floor.py -q
python -m pytest tests/test_kernel_database_op_binding.py -q
python -m pytest tests/test_kernel_baseline_regression.py -q
python -m pytest tests/ -q
ruff check dryrun/ tests/
```

Expected: **8 passed** in the new file, the existing floor tests pass, all **4 database binding
tests now pass** (including the attention one that Task 6 left red), and **all 3 baseline tests
pass**: including `test_fitted_floors_have_not_moved`, which was red since Task 1 and is what
proves the re-keying preserved the values. Suite **680 passed**, no new ruff findings.

- [ ] **Step 8: Commit**

```bash
git add dryrun/cost/kernel/sol.py dryrun/cost/kernel/floor.py dryrun/cost/kernel/holdout.py \
        tests/test_kernel_floor_op_keys.py tests/test_kernel_floor.py
git commit -m "refactor: key dispatch floors by op name instead of phase

floors.json stored {context, generation} while everything else spoke in op
names, so four sites translated between the two: the fitting loop's
_ATTENTION_OPS map, two branches in holdout, and the database. Each was a place
a new floor-taking op had to be threaded through.

Keyed by op, floor fitting becomes op-agnostic: it loops over ops_in() and
skips those declaring fits_floor = False, so GEMM is excluded by policy rather
than by absence from a map, and a new op gets a fitted floor with no edit here.
holdout's three per-op SOL closures collapse into one that asks the op, removing
the second copy of the SOL binding.

This is a breaking format change. An old phase-keyed file raises and names the
fit-floor command rather than being read leniently: a silent zero floor measures
34.89% MRE against 6.63% at the fitted 22 us.

Regenerated floors reproduce every fitted value exactly: vllm 32/22 us,
flashinfer 4/6 us, which the baseline test now pins.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Collapse the writer and move sweeps onto the op

The profiler side of the abstraction. After this, adding an op needs no new writer method and no
new config field.

**Files:**
- Modify: `dryrun/profiler/kernels/measurements.py` (replace `write_gemm`/`write_attention` with `write`; delete `GemmRow`/`AttentionRow`)
- Modify: `dryrun/profiler/kernels/config.py` (replace the `gemm`/`attention` fields with `sweeps` + `sweep(op_name)`)
- Modify: `dryrun/profiler/kernels/gemm.py`, `dryrun/profiler/kernels/_vllm_attn.py`, `dryrun/profiler/kernels/_flashinfer_attn.py` (call `write`)
- Modify: `tests/test_kernel_measurements.py`, `tests/test_kernel_holdout.py`, `tests/test_kernel_table.py`, `tests/test_kernel_database.py`, `tests/test_profiling_cli.py` (mechanical rewrite to `write`)
- Test: `tests/test_kernel_writer_columns.py`

**Interfaces:**
- Consumes: `op_for` (Task 2), `op.table_columns()` (Tasks 3-4).
- Produces:
  - `PerfTableWriter.write(op_name: str, rows: Sequence[Mapping[str, Any]]) -> Path`: validates columns against the op and raises `ValueError` on a missing or extra one.
  - `KernelProfileConfig.sweep(op_name: str) -> Any`: the parsed sweep for one op, raising `ProfileValidationError` when the config has no subtree for its family.
  - `KernelProfileConfig.gemm` and `.attention` are deleted.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kernel_writer_columns.py`:

```python
"""One write method, validated against the op's own column declaration."""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from dryrun.profiler.kernels.measurements import PerfTableWriter


@pytest.fixture
def writer(tmp_path):
    return PerfTableWriter(
        output_root=tmp_path,
        device="NVIDIA H20",
        framework="vllm",
        version="0.0.0",
        kernel_source="torch.nn.functional.linear",
    )


def _gemm_row(m: int) -> dict:
    return {"m": m, "n": 4096, "k": 4096, "gemm_dtype": "bfloat16", "latency_s": 1.0e-4}


def test_a_gemm_table_is_written_with_provenance_and_payload(writer):
    path = writer.write("gemm", [_gemm_row(1), _gemm_row(16)])
    table = pq.read_table(path)
    assert set(table.column_names) >= {
        "framework", "version", "device", "op_name", "kernel_source",
        "gemm_dtype", "n", "k", "m", "latency_s",
    }
    assert table.num_rows == 2
    assert set(table.column("op_name").to_pylist()) == {"gemm"}


def test_the_written_table_loads_back_through_its_op(writer):
    """A table the writer produced must satisfy the reader's contract by construction."""
    from dryrun.cost.kernel.ops.base import op_for
    from dryrun.cost.kernel.table import KernelTable

    path = writer.write("gemm", [_gemm_row(1), _gemm_row(16)])
    table = KernelTable.load(path, op_for("gemm"))
    assert table.row_count == 2


def test_a_missing_column_is_rejected_at_write_time(writer):
    """write_attention trusted its caller for eleven columns; a typo surfaced only on read."""
    incomplete = dict(_gemm_row(1))
    del incomplete["k"]
    with pytest.raises(ValueError, match="k"):
        writer.write("gemm", [incomplete])


def test_an_extra_column_is_rejected_at_write_time(writer):
    """An unexpected column means the caller and the op disagree about the schema."""
    extra = dict(_gemm_row(1))
    extra["batch_size"] = 8
    with pytest.raises(ValueError, match="batch_size"):
        writer.write("gemm", [extra])


def test_an_empty_row_list_is_rejected(writer):
    with pytest.raises(ValueError, match="no rows"):
        writer.write("gemm", [])


def test_an_unknown_op_is_rejected(writer):
    from dryrun.cost.kernel.interp import KernelCoverageError

    with pytest.raises(KernelCoverageError, match="wodget"):
        writer.write("wodget", [{"latency_s": 1.0}])


def test_a_ragged_row_set_is_rejected(writer):
    """Every row must carry the same keys, or the parquet columns misalign silently."""
    with pytest.raises(ValueError):
        writer.write("gemm", [_gemm_row(1), {"m": 2, "latency_s": 1.0e-4}])


def test_the_path_uses_the_ops_family_directory(writer, tmp_path):
    path = writer.write("gemm", [_gemm_row(1)])
    assert path == tmp_path / "h20" / "gemm" / "vllm" / "0.0.0" / "gemm_perf.parquet"


def test_an_attention_table_uses_the_attention_family_directory(writer, tmp_path):
    row = {
        "batch_size": 4, "num_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "isl": 1024, "attn_dtype": "bfloat16", "kv_cache_dtype": "bfloat16",
        "block_size": 16, "window_size": 0, "latency_s": 5.0e-4,
    }
    path = writer.write("context_attention", [row])
    assert path == tmp_path / "h20" / "attention" / "vllm" / "0.0.0" / "context_attention_perf.parquet"


def test_the_per_family_write_methods_are_gone():
    """Keeping them would leave two ways to write a table, one of them unvalidated."""
    assert not hasattr(PerfTableWriter, "write_gemm")
    assert not hasattr(PerfTableWriter, "write_attention")
```

Also append to `tests/test_kernel_ops_base.py`:

```python
def test_the_config_serves_a_sweep_per_op_name():
    """Adding an op must not require adding a named field to KernelProfileConfig."""
    from pathlib import Path as _Path

    from dryrun.profiler.kernels.config import KernelProfileConfig

    config_path = _Path(__file__).resolve().parents[1] / "examples" / "profile" / "kernel" / "qwen3_8b_h20_tp1.yaml"
    if not config_path.is_file():
        pytest.skip("the example kernel config is machine-local")
    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = KernelProfileConfig.from_dict(payload, base_dir=config_path.parent)
    assert config.sweep("gemm").token_counts[0] == 1
    assert config.sweep("context_attention").batch_sizes[0] == 1
    # Both attention phases read one `attention:` subtree.
    assert config.sweep("context_attention") == config.sweep("generation_attention")
    assert not hasattr(config, "gemm")
    assert not hasattr(config, "attention")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_kernel_writer_columns.py -q
```

Expected: FAIL with `AttributeError: 'PerfTableWriter' object has no attribute 'write'`.

- [ ] **Step 3: Collapse the writer**

In `dryrun/profiler/kernels/measurements.py`, delete the `GemmRow` and `AttentionRow`
dataclasses and both `write_*` methods, and replace them with one `write`:

```python
    def write(self, op_name: str, rows: Sequence[Mapping[str, Any]]) -> Path:
        """
        Write one op's perf table.

        Args:
            op_name (str): A registered op name; selects the family directory and the
                expected columns.
            rows (Sequence[Mapping[str, Any]]): One mapping per measured shape, carrying
                exactly the op's slice and axis columns plus `latency_s`.

        Returns:
            Path: The written parquet path.

        Raises:
            ValueError: If `rows` is empty, or any row's keys do not match the op's columns
                exactly.
            KernelCoverageError: If `op_name` is not registered.
        """
        op = op_for(op_name)
        if not rows:
            raise ValueError(f"Refusing to write a {op_name} perf table with no rows.")
        expected = set(op.table_columns())
        # Validate here rather than trusting the caller. The per-family methods this replaces
        # populated up to eleven columns by hand, so a misspelled key produced a table that
        # only failed when something later tried to read it: possibly hours after the sweep.
        for index, row in enumerate(rows):
            keys = set(row)
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            if missing or extra:
                parts = []
                if missing:
                    parts.append(f"missing {', '.join(missing)}")
                if extra:
                    parts.append(f"unexpected {', '.join(extra)}")
                raise ValueError(
                    f"Row {index} of a {op_name} perf table has {'; '.join(parts)}. "
                    f"Expected exactly: {', '.join(op.table_columns())}."
                )
        columns = {name: [row[name] for row in rows] for name in op.table_columns()}
        return self._write(op_name, len(rows), columns)
```

Change `table_path` to use `op_for(op_name).family` (already done in Task 5) and update the
imports:

```python
from collections.abc import Mapping, Sequence
from typing import Any

from dryrun.cost.kernel.hardware import gpu_slug
from dryrun.cost.kernel.ops.base import op_for
```

Update the module docstring's last paragraph:

```
Column order and naming come from the op itself, so the reader and writer cannot drift apart,
and a row whose keys do not match the op's declaration is rejected at write time rather than
at read time.
```

- [ ] **Step 4: Move sweeps onto the config**

In `dryrun/profiler/kernels/config.py`, delete the `gemm: GemmSweep` and
`attention: AttentionSweep` fields from `KernelProfileConfig` and add a parsed-sweep mapping:

```python
@dataclass(frozen=True)
class KernelProfileConfig:
    """Configuration required to profile kernels, with one sweep per op."""

    model: ModelDescriptor
    hardware: HardwareDescriptor
    output_dir: Path
    experiment_name: str
    tensor_parallel: int
    measurement: MeasurementConfig
    sweeps: dict[str, Any]

    def sweep(self, op_name: str) -> Any:
        """
        Return the parsed sweep grid for one op.

        Args:
            op_name (str): A registered op name.

        Returns:
            Any: The op's own sweep type, e.g. `GemmSweep` or `AttentionSweep`.

        Raises:
            ProfileValidationError: If the config carries no sweep for this op's family.
        """
        try:
            return self.sweeps[op_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.sweeps)) or "none"
            raise ProfileValidationError(
                f"No sweep configured for op {op_name!r}. Configured ops: {available}. "
                f"Add a `{op_for(op_name).family}:` block to the kernel profile config."
            ) from exc
```

and build `sweeps` in `from_dict` by asking each registered op to parse its family's subtree:

```python
    @classmethod
    def from_dict(cls, value: dict[str, Any], *, base_dir: Path | None = None) -> KernelProfileConfig:
        root = base_dir or Path.cwd()
        output_dir = Path(value.get("output_dir", "profile_output"))
        if not output_dir.is_absolute():
            output_dir = (root / output_dir).resolve()
        parallelism = value.get("parallelism", {})
        # Each op parses its own family's subtree, so adding an op adds no field here. A family
        # absent from the config yields no entry, and `sweep()` raises naming the missing block
        # only if something actually asks for it: profiling one family must not require
        # configuring them all.
        sweeps: dict[str, Any] = {}
        for op in OP_REGISTRY.values():
            subtree = value.get(op.family)
            if subtree is None:
                continue
            sweeps[op.name] = op.parse_sweep(subtree)
        return cls(
            model=model_descriptor_from_yaml(value["model"], base_dir=root),
            hardware=HardwareDescriptor.from_dict(value["hardware"]),
            output_dir=output_dir,
            experiment_name=str(value.get("experiment_name", "kernel")),
            tensor_parallel=int(parallelism.get("tensor_parallel", 1)),
            measurement=MeasurementConfig.from_dict(value.get("measurement", {})),
            sweeps=sweeps,
        )
```

Add only the registry lookup import:

```python
from ...cost.kernel.ops.base import OP_REGISTRY, op_for
```

**Do NOT add per-op module imports here.** Task 5 moved registration into
`dryrun/cost/kernel/ops/__init__.py`, which is the correct layer: `op_for` is called from
`database.py`, `floor.py`, and the perf-table writer, none of which should depend on the
profiler package having been imported first. Importing `ops.base` pulls in `ops/__init__.py`
and therefore every op. An earlier draft of this plan deferred registration to this file, which
was both too late and the wrong direction: verified by reproducing
`PerfTableWriter.table_path("gemm")` raising `Registered ops: none` while every unit test
passed, because each test imports its own op module.

Keep `GemmSweep` and `AttentionSweep` defined in this module: the ops import them lazily, so
moving them would create a cycle.

- [ ] **Step 5: Repoint the three profiler call sites**

In `dryrun/profiler/kernels/gemm.py`, replace the `GemmRow` construction and the write:

```python
    rows: list[dict[str, object]] = []
    for m, n, k in gemm_all_shapes(model, config.sweep("gemm"), tensor_parallel=config.tensor_parallel):
        activation = torch.randn(m, k, dtype=dtype, device=device)
        weight = torch.randn(n, k, dtype=dtype, device=device)
        latency_s = time_cuda_call(
            lambda a=activation, w=weight: torch.nn.functional.linear(a, w),
            warmup_iterations=measurement.warmup_iterations,
            active_iterations=measurement.active_iterations,
        )
        rows.append(
            {"m": m, "n": n, "k": k, "gemm_dtype": measurement.dtype, "latency_s": latency_s}
        )
        del activation, weight
        torch.cuda.empty_cache()
    return writer.write("gemm", rows)
```

and drop `GemmRow` from its imports.

In `_vllm_attn.py` and `_flashinfer_attn.py`, replace each `AttentionRow(...)` with the
equivalent dict and each `writer.write_attention(op_name, rows)` with
`writer.write(op_name, rows)`. Read the surrounding code first and match its existing variable
names; the ten keys are `batch_size`, `num_heads`, `num_key_value_heads`, `head_dim`, `isl`,
`attn_dtype`, `kv_cache_dtype`, `block_size`, `window_size`, `latency_s` for context, with
`step` replacing `isl` for generation. Also replace every `config.attention` with
`config.sweep(op_name)` and every `config.gemm` with `config.sweep("gemm")`.

Note the context and generation tables have **different** axis columns, so a context row must
not carry `step` and a generation row must not carry `isl`: the writer's extra-column check
now enforces that, where the old shared `AttentionRow` carried both.

- [ ] **Step 6: Run the tests and the baseline gate**

```bash
python -m pytest tests/test_kernel_writer_columns.py tests/test_kernel_ops_base.py -q
python -m pytest tests/test_kernel_measurements.py tests/test_kernel_gemm.py \
    tests/test_kernel_attention_sweep.py tests/test_profiling_cli.py -q
python -m pytest tests/test_kernel_baseline_regression.py -q
python -m pytest tests/ -q
ruff check dryrun/ tests/
grep -rn "write_gemm\|write_attention\|GemmRow\|AttentionRow\|config\.gemm\b\|config\.attention\b" \
    dryrun/ tests/ --include=*.py || echo "OLD WRITER API GONE"
```

Expected: **10 passed** in the writer file, the profiler tests pass, all **3 baseline tests
pass**, the suite grows by exactly the new file's test count and stays green, no new ruff findings, and `OLD WRITER API GONE`.

- [ ] **Step 7: Commit**

```bash
git add dryrun/profiler/kernels/measurements.py dryrun/profiler/kernels/config.py \
        dryrun/profiler/kernels/gemm.py dryrun/profiler/kernels/_vllm_attn.py \
        dryrun/profiler/kernels/_flashinfer_attn.py \
        tests/test_kernel_writer_columns.py tests/test_kernel_ops_base.py \
        tests/test_kernel_measurements.py tests/test_kernel_holdout.py \
        tests/test_kernel_table.py tests/test_kernel_database.py tests/test_profiling_cli.py
git commit -m "refactor: one write method and one sweep lookup, both driven by the op

write_gemm and write_attention differed only in their columns dict, and that
dict is exactly op.slice_columns + op.axis_columns + latency_s. Collapsed into
write(op_name, rows), which validates each row against the op's declaration.
That is strictly safer than what it replaces: write_attention populated eleven
columns by hand, so a misspelled key produced a table that only failed when
something later read it.

The check also catches a real hazard the shared AttentionRow hid: context and
generation have different third axes, and one row type carrying both isl and
step let either table be written with a column it does not declare.

KernelProfileConfig loses its gemm and attention fields for sweeps + sweep(),
with each op parsing its own family subtree. A family absent from the config
yields no entry and only raises if asked for, so profiling one family does not
require configuring them all.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Derive the CLI from the registry, and pin op coverage

The last hardcoded op list.

**Files:**
- Modify: `dryrun/cli/profile.py:207-248` (`cmd_kernels_run`) and `492-501` (the `--family` parser)
- Test: `tests/test_kernel_op_coverage.py`
- Test: `tests/test_profiling_cli.py` (add one parser test)

**Interfaces:**
- Consumes: `families`, `ops_for_family`, `OP_REGISTRY` (Task 2); `PerfTableWriter.write` (Task 8).
- Produces: `--family` accepting `"all"` plus every registered family; `cmd_kernels_run` dispatching by family without a per-family `if`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kernel_op_coverage.py`:

```python
"""Every measured table must have a registered op, and the CLI must not hardcode a list."""

from __future__ import annotations

from pathlib import Path

import pytest

from dryrun.cost.kernel.ops.base import OP_REGISTRY, families

_REPO = Path(__file__).resolve().parents[1]
_TABLES = _REPO / "outputs" / "profile" / "qwen3_8b" / "kernel_tp1" / "kernel"


def test_the_registry_covers_the_three_original_ops():
    assert {"gemm", "context_attention", "generation_attention"} <= set(OP_REGISTRY)


def test_the_cli_family_choices_come_from_the_registry():
    """A hardcoded choices list is the last place adding an op means editing the CLI."""
    from dryrun.cli.profile import build_parser

    parser = build_parser()
    args = parser.parse_args(["kernels", "run", "--config", "c.yaml"])
    assert args.family == "all"
    for family in families():
        parsed = parser.parse_args(["kernels", "run", "--config", "c.yaml", "--family", family])
        assert parsed.family == family


def test_an_unknown_family_is_rejected():
    from dryrun.cli.profile import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["kernels", "run", "--config", "c.yaml", "--family", "widgets"])


def test_the_family_choices_are_not_a_hardcoded_literal():
    """Derived choices are what make a new op need zero CLI edits.

    This is asserted separately from the parse test above because the literal it replaces
    happens to list exactly today's families, so a parse-only test would pass before the
    change and prove nothing.
    """
    import inspect

    from dryrun.cli import profile as profile_module

    source = inspect.getsource(profile_module.build_parser)
    assert '"gemm", "attention"' not in source, "family choices are still a hardcoded literal"


@pytest.mark.skipif(not _TABLES.is_dir(), reason="Measured perf tables are machine-local.")
def test_every_measured_parquet_has_a_registered_op():
    """An unregistered table is silently skipped by every report; that must be a failure."""
    stems = set()
    for parquet in (_TABLES / "h20").glob("*/*/*/*_perf.parquet"):
        stem = parquet.stem
        stems.add(stem[: stem.rfind("_perf")])
    assert stems, f"no perf tables found under {_TABLES / 'h20'}"
    unregistered = sorted(stems - set(OP_REGISTRY))
    assert not unregistered, f"perf tables with no registered op: {unregistered}"


@pytest.mark.skipif(not _TABLES.is_dir(), reason="Measured perf tables are machine-local.")
def test_every_measured_table_sits_in_its_ops_family_directory():
    """A table under the wrong family would be invisible to ops_for_family dispatch."""
    for parquet in (_TABLES / "h20").glob("*/*/*/*_perf.parquet"):
        stem = parquet.stem
        name = stem[: stem.rfind("_perf")]
        op = OP_REGISTRY.get(name)
        if op is None:
            continue
        family_dir = parquet.parents[2].name
        assert family_dir == op.family, f"{parquet} sits under {family_dir}, op declares {op.family}"
```

Append to `tests/test_profiling_cli.py`, matching that file's existing style:

```python
def test_kernels_run_accepts_every_registered_family():
    from dryrun.cost.kernel.ops.base import families

    parser = build_parser()
    for family in families():
        args = parser.parse_args(["kernels", "run", "--config", "c.yaml", "--family", family])
        assert args.family == family
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_kernel_op_coverage.py -q
```

Expected: FAIL on `test_the_family_choices_are_not_a_hardcoded_literal`, which reads
`build_parser`'s source and finds the literal still there.

Note why that test exists alongside `test_the_cli_family_choices_come_from_the_registry`: the
current literal `["all", "gemm", "attention"]` happens to cover today's families exactly, so a
test that only parses today's family names would pass before the change. Asserting the literal
is gone is what actually fails now.

- [ ] **Step 3: Rewrite the CLI dispatch**

In `dryrun/cli/profile.py`, replace `cmd_kernels_run`'s body with the following. Note it calls
`measure` once per **family**, not once per op: both attention phases come out of a single
backend sweep, so a per-op loop would measure the grid twice and could select a different
backend for each phase.

```python
def cmd_kernels_run(args: argparse.Namespace) -> None:
    """Profile every registered kernel op, or one family of them, directly on the GPU."""
    import torch

    from ..cost.kernel.ops.base import families, ops_for_family
    from ..profiler.kernels.measurements import PerfTableWriter

    config = _kernel_config(args.config)
    device = torch.cuda.get_device_name(0)
    table_root = config.table_root()
    selected = families() if args.family == "all" else (args.family,)

    def writer_factory(framework: str, version: str, kernel_source: str) -> PerfTableWriter:
        return PerfTableWriter(
            output_root=table_root,
            device=device,
            framework=framework,
            version=version,
            kernel_source=kernel_source,
        )

    written: list[Path] = []
    for family in selected:
        ops = ops_for_family(family)
        if not ops:
            raise SystemExit(f"No registered kernel ops in family {family!r}.")
        # One call per family, not per op. Both attention phases come out of a single backend
        # sweep that builds one KV cache and times both phases against it, so calling per op
        # would measure the grid twice and could pick a different backend for each phase.
        written.extend(ops[0].measure(config.model, config, writer_factory=writer_factory))

    for path in written:
        dryrun_logger.info(f"Wrote kernel perf table {path}.")
```

`ops[0].measure` measuring the whole family is a real constraint of the attention sweep, so it
must be stated in the `KernelOp` protocol docstring. In `dryrun/cost/kernel/ops/base.py`, add
`measure` to the protocol:

```python
    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """
        Measure this op's whole family on the GPU and write its perf tables.

        Called once per family, not once per op: both attention phases come out of a single
        sweep that allocates one KV cache and times both against it, so a per-op call would
        measure the grid twice and could select a different backend for each phase. An op
        whose family holds only itself simply writes one table.

        Args:
            model (Any): `ModelDescriptor` for the architecture being profiled.
            config (Any): `KernelProfileConfig` carrying the sweep and measurement settings.
            writer_factory (Any): Builds a `PerfTableWriter` from
                `(framework, version, kernel_source)`. The kernel source is known only after
                backend selection, so a writer cannot be passed in directly.

        Returns:
            tuple[Any, ...]: The written parquet paths.
        """
```

Then add `measure` to `GemmKernelOp` (in `ops/gemm.py`):

```python
    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Time every swept GEMM shape and write the unified table."""
        # Imported here so the cost path never loads torch.
        from ....profiler.kernels.gemm import profile_gemm

        writer = writer_factory("vllm", _torch_version(), "torch.nn.functional.linear")
        return (profile_gemm(model, config, writer=writer),)
```

with a module-level helper in `ops/gemm.py`:

```python
def _torch_version() -> str:
    """Return the torch version string used as the GEMM table's provenance version."""
    import torch

    return str(torch.__version__)
```

and to both attention ops (in `ops/attention.py`), sharing one implementation:

```python
def _measure_attention(model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
    """Measure both phases through both backends and write four tables."""
    from ....profiler.kernels.attention import (
        profile_flashinfer_attention,
        profile_vllm_attention,
    )

    written = list(profile_vllm_attention(model, config, writer_factory=writer_factory))
    written.extend(profile_flashinfer_attention(model, config, writer_factory=writer_factory))
    return tuple(written)
```

referenced from each attention op as:

```python
    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Measure both attention phases through both backends."""
        return _measure_attention(model, config, writer_factory=writer_factory)
```

Add `from typing import Any` where needed, and replace the `--family` parser argument:

```python
    kernels_run = kernels_workflows.add_parser("run")
    _add_config_argument(kernels_run)
    kernels_run.add_argument(
        "--family",
        choices=["all", *_kernel_families()],
        default="all",
        help="Kernel family to measure. Defaults to every registered family.",
    )
```

with a module-level helper in `dryrun/cli/profile.py`:

```python
def _kernel_families() -> tuple[str, ...]:
    """Return every registered kernel family, for the --family choices."""
    from ..cost.kernel.ops.base import families
    from ..profiler.kernels import config as _kernel_config_module  # noqa: F401  (registers ops)

    return families()
```

The `noqa`'d import is what populates the registry: `config.py` imports every op module for its
own sweep parsing (Task 8), so importing it is what makes `families()` non-empty. Verify that in
Step 4 rather than assuming it.

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_kernel_op_coverage.py tests/test_profiling_cli.py -q
python -m pytest tests/test_kernel_baseline_regression.py -q
python -m pytest tests/ -q
ruff check dryrun/ tests/
python -c "
from dryrun.cli.profile import build_parser
p = build_parser()
a = p.parse_args(['kernels','run','--config','c.yaml','--family','attention'])
print('parsed family:', a.family)"
```

Expected: **7 passed** in the coverage file, `test_profiling_cli.py` green, all 3 baseline tests
pass, the suite grows by exactly the new file's test count and stays green, no new ruff findings, and the last command prints
`parsed family: attention`. If `--family attention` is rejected, `_kernel_families()` returned
empty and the registry-populating import is wrong.

- [ ] **Step 5: Verify the real sweep still runs**

The dispatch rewrite changes how the profiler is entered, so it must be exercised on the GPU
before the stage-1 gate can be called clean.

```bash
dryrun-profile kernels run --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml --family gemm
dryrun-profile kernels fit-floor --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml
dryrun-profile kernels holdout --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml \
    --fraction 1.0 --gate 100 --output /tmp/holdout-after-task9.json
python - <<'PY'
import json
before = {(r["op_name"], r["backend"]): r for r in json.load(open(".superpowers/sdd/baseline/holdout.json"))}
after = {(r["op_name"], r["backend"]): r for r in json.load(open("/tmp/holdout-after-task9.json"))}
assert set(before) == set(after), (sorted(before), sorted(after))
for key in before:
    b, a = before[key], after[key]
    assert b["point_count"] == a["point_count"], f"{key} point count moved"
    assert abs(b["mre_percent"] - a["mre_percent"]) < 0.01, f"{key} MRE moved: {b['mre_percent']} -> {a['mre_percent']}"
print("HOLDOUT IDENTICAL AFTER RE-SWEEP")
PY
```

**This check as written is INVALID and must not be used.** Its premise: that re-measuring
GEMM reproduces the same table: is false. `latency_s` is a wall-clock GPU measurement: two
runs of *identical* code produced holdout MREs of 3.896% and 3.828%, 0.068 percentage points
apart, about seven times the 0.01pp tolerance this step demands. Measured directly, six repeats
of one GEMM kernel span 0.351% of their median. No byte-identical re-sweep gate is
constructible, so this step could not have passed for any implementation, correct or not.

Every parquet column *except* `latency_s` does reproduce bit-identically, so a sweep-level
check must exclude `latency_s`.

Prove dispatch neutrality without timings instead. Both of these are timing-independent:

1. **Call-argument equivalence.** Monkeypatch the profiler entry points and the writer to
   record `(name, args, kwargs)` rather than execute, run the old and new dispatch for each
   `--family` value, and compare the recorded multisets. `gemm` and `attention` must match
   call-for-call; `all` may differ in order only, because `families()` sorts.
2. **Real-run table placement.** Run `--family all` on the GPU once and assert all five tables
   land in their ops' family directories and each loads under its own op's column contract.

If you do run a real sweep, back up the pinned tables first and restore them afterwards: the
ten baseline numbers are pinned to those exact files.

- [ ] **Step 6: Commit**

```bash
git add dryrun/cli/profile.py dryrun/cost/kernel/ops/base.py dryrun/cost/kernel/ops/gemm.py \
        dryrun/cost/kernel/ops/attention.py tests/test_kernel_op_coverage.py tests/test_profiling_cli.py
git commit -m "refactor: derive kernel CLI dispatch from the op registry

--family carried the literal choices [all, gemm, attention] and the command body
had one if-block per family, so adding an op meant editing the CLI. Both now come
from the registry, and a test asserts the literal is gone rather than only that
today's families parse.

measure() is called once per family rather than once per op, which the protocol
docstring now states as a constraint: both attention phases come out of a single
sweep that allocates one KV cache and times both against it, so a per-op call
would measure the grid twice and could pick a different backend per phase.

Re-swept GEMM on the same machine and confirmed the holdout report is identical,
which is the evidence the writer collapse did not change what gets measured.

Also pins op coverage: a perf table with no registered op, or one sitting under
the wrong family directory, is now a test failure rather than a silent skip in
every report.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Stage 1 gate

**Do not start Task 10 until this passes.** The refactor is complete; this is the proof it was
behaviour-neutral. A new op measured before this point would make the comparison meaningless.

- [ ] **Run the full gate**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/DryRun-RL

python -m pytest tests/ -q --ignore=tests/test_fsdp_cost.py
ruff check dryrun/ tests/

# Do NOT re-sweep before this. The gate reads the pinned tables under outputs/; re-measuring
# them moves latency_s by ~0.35% per kernel and would fail the comparison for reasons that have
# nothing to do with the refactor.
dryrun-profile kernels holdout --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml \
    --fraction 1.0 --gate 100 --output /tmp/gate-holdout.json
for surface in pchip shepard; do
  dryrun-profile kernels validate \
      --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml \
      --baseline outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
      --model-path /apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B \
      --surface "${surface}" --min-samples 1 --gate 100 \
      --output "/tmp/gate-validate-${surface}.json"
done

python - <<'PY'
import json
ok = True
b = {(r["op_name"], r["backend"]): r for r in json.load(open(".superpowers/sdd/baseline/holdout.json"))}
a = {(r["op_name"], r["backend"]): r for r in json.load(open("/tmp/gate-holdout.json"))}
for key in sorted(set(b) | set(a)):
    if key not in b or key not in a:
        print(f"FAIL {key} appeared or disappeared"); ok = False; continue
    if b[key]["point_count"] != a[key]["point_count"] or abs(b[key]["mre_percent"] - a[key]["mre_percent"]) >= 0.01:
        print(f"FAIL holdout {key}: {b[key]['mre_percent']:.2f}% / {b[key]['point_count']} -> "
              f"{a[key]['mre_percent']:.2f}% / {a[key]['point_count']}"); ok = False
for surface in ("pchip", "shepard"):
    b = {r["phase"]: r for r in json.load(open(f".superpowers/sdd/baseline/validate-{surface}.json"))}
    a = {r["phase"]: r for r in json.load(open(f"/tmp/gate-validate-{surface}.json"))}
    for phase in sorted(set(b) | set(a)):
        if phase not in b or phase not in a:
            print(f"FAIL validate {surface} {phase} appeared or disappeared"); ok = False; continue
        if b[phase]["bucket_count"] != a[phase]["bucket_count"] or abs(b[phase]["mre_percent"] - a[phase]["mre_percent"]) >= 0.01:
            print(f"FAIL validate {surface} {phase}: {b[phase]['mre_percent']:.2f}% -> {a[phase]['mre_percent']:.2f}%")
            ok = False
print("STAGE 1 GATE PASSED" if ok else "STAGE 1 GATE FAILED")
PY
```

Expected: suite green, 7 pre-existing ruff findings only, and **STAGE 1 GATE PASSED**. If any
line reports FAIL, the refactor changed behaviour: find the cause and fix the code. Do not
update the baseline.

---

## Task 10: The `elementwise` op

The third op family, which is what proves the abstraction generalizes rather than merely
accommodating the two families that shaped it. **It is deliberately not wired into
`ComposedRollout.predict()`**: see the spec's elementwise section and the test in Step 6.

**Files:**
- Create: `dryrun/cost/kernel/ops/elementwise.py`
- Create: `dryrun/profiler/kernels/elementwise.py`
- Modify: `dryrun/profiler/kernels/config.py` (import the new op so it registers)
- Modify: `examples/profile/kernel/qwen3_8b_h20_tp1.yaml` (add the `elementwise:` block)
- Test: `tests/test_kernel_ops_elementwise.py`

**Interfaces:**
- Consumes: `KernelOp`, `register_op` (Task 2); `PerfTableWriter.write` (Task 8); `MeasurementConfig` from `dryrun.profiler.kernels.config`.
- Produces:
  - `ElementwiseSweep(token_counts: tuple[int, ...])`: frozen, in `dryrun/profiler/kernels/config.py`.
  - `ElementwiseKernelOp` with `name = "elementwise"`, `family = "elementwise"`, `slice_columns = ("dtype", "op_kind")`, `axis_columns = ("n_elements",)`, `fits_floor = True`, registered as `ELEMENTWISE_OP`.
  - `ELEMENTWISE_KINDS: tuple[str, ...] = ("rms_norm", "rotary_embedding", "silu_and_mul", "qk_norm")`.
  - `dryrun.profiler.kernels.elementwise.profile_elementwise(model, config, *, writer_factory) -> tuple[Path, ...]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kernel_ops_elementwise.py`:

```python
"""The elementwise op: bandwidth-only SOL, a fitted floor, and no wiring into predict()."""

from __future__ import annotations

import pytest

from dryrun.cost.kernel.hardware import load_gpu_spec
from dryrun.cost.kernel.ops.base import op_for
from dryrun.cost.kernel.ops.elementwise import ELEMENTWISE_KINDS, ELEMENTWISE_OP


@pytest.fixture(scope="module")
def spec():
    return load_gpu_spec("h20")


def test_the_op_is_registered_in_its_own_family():
    assert op_for("elementwise") is ELEMENTWISE_OP
    assert ELEMENTWISE_OP.family == "elementwise"


def test_the_kinds_share_one_schema_because_they_share_one_sol_shape():
    """All four are pure bandwidth; only the bytes touched differ, which n_elements carries."""
    assert ELEMENTWISE_OP.slice_columns == ("dtype", "op_kind")
    assert ELEMENTWISE_OP.axis_columns == ("n_elements",)
    assert set(ELEMENTWISE_KINDS) == {"rms_norm", "rotary_embedding", "silu_and_mul", "qk_norm"}


def test_the_op_takes_a_fitted_floor():
    """Measured: all four kinds sit at 4-5 us flat below 128 tokens, so dispatch dominates."""
    assert ELEMENTWISE_OP.fits_floor is True


def test_sol_has_no_compute_term(spec):
    """An elementwise kernel does no matmul, so a math-bound roofline would be meaningless."""
    got = ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rms_norm"), coords=(4096.0 * 128,))
    assert got.math_s == 0.0
    assert got.memory_s > 0.0


def test_sol_scales_linearly_with_element_count(spec):
    """Bandwidth-bound means doubling the elements doubles the time, exactly."""
    one = ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rms_norm"), coords=(1.0e6,)).memory_s
    two = ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rms_norm"), coords=(2.0e6,)).memory_s
    assert two == pytest.approx(2.0 * one, rel=1e-12)


def test_a_narrower_dtype_moves_fewer_bytes(spec):
    wide = ELEMENTWISE_OP.sol(spec, slice_key=("float32", "rms_norm"), coords=(1.0e6,)).memory_s
    narrow = ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rms_norm"), coords=(1.0e6,)).memory_s
    assert wide == pytest.approx(2.0 * narrow, rel=1e-12)


def test_the_floor_binds_at_a_tiny_element_count(spec):
    got = ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rms_norm"), coords=(64.0,), floor_s=4.5e-6)
    assert got.latency_s == pytest.approx(4.5e-6, rel=1e-12)


def test_the_floor_does_not_bind_at_a_large_element_count(spec):
    got = ELEMENTWISE_OP.sol(
        spec, slice_key=("bfloat16", "rms_norm"), coords=(1.0e9,), floor_s=4.5e-6
    )
    assert got.latency_s == pytest.approx(got.memory_s, rel=1e-12)
    assert got.latency_s > 4.5e-6


def test_a_non_positive_element_count_is_rejected(spec):
    with pytest.raises(ValueError, match="n_elements"):
        ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rms_norm"), coords=(0.0,))


def test_an_unknown_kind_is_rejected(spec):
    """A typo'd kind would silently create a new slice with one point in it."""
    with pytest.raises(ValueError, match="rmsnorm"):
        ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rmsnorm"), coords=(1.0e6,))


def test_latency_is_the_last_table_column():
    assert ELEMENTWISE_OP.table_columns() == ("dtype", "op_kind", "n_elements", "latency_s")


def test_parse_sweep_builds_the_token_grid():
    sweep = ELEMENTWISE_OP.parse_sweep({"token_counts": [1, 128, 4096]})
    assert sweep.token_counts == (1, 128, 4096)


def test_parse_sweep_rejects_an_empty_grid():
    from dryrun.profiler.schema import ProfileValidationError

    with pytest.raises(ProfileValidationError, match="token_counts"):
        ELEMENTWISE_OP.parse_sweep({"token_counts": []})


def test_the_composed_model_does_not_add_an_elementwise_term():
    """Measured: adding it takes decode 4.84% -> 11.18% because utility already absorbs it.

    Utility is SOL/measured, and the measured GEMM and attention latencies came from a server
    already running these kernels between them, so a separate term double-counts. If this test
    fails, someone wired elementwise into predict(); read the spec's elementwise section before
    changing this assertion.
    """
    import inspect

    from dryrun.cost.rollout.composed import ComposedRollout

    source = inspect.getsource(ComposedRollout)
    assert "elementwise" not in source.lower() or "query_elementwise" not in source, (
        "ComposedRollout appears to price elementwise ops; that double-counts work already "
        "inside the fitted GEMM and attention utilities."
    )
    assert not hasattr(ComposedRollout, "_elementwise_seconds")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_kernel_ops_elementwise.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dryrun.cost.kernel.ops.elementwise'`.

- [ ] **Step 3: Add the sweep type**

In `dryrun/profiler/kernels/config.py`, add beside `GemmSweep`:

```python
@dataclass(frozen=True)
class ElementwiseSweep:
    """Token counts to measure every elementwise kernel at."""

    token_counts: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ElementwiseSweep:
        return cls(
            token_counts=_positive_int_tuple(
                value["token_counts"], field="elementwise.token_counts"
            )
        )
```

- [ ] **Step 4: Write the cost-side op**

Create `dryrun/cost/kernel/ops/elementwise.py`:

```python
"""
Elementwise kernels: rmsnorm, rope, activation, and Qwen3's qk_norm.

All four are pure bandwidth (they read a tensor, write a tensor, and do no matmul) so they
share one schema and one roofline shape. Only the bytes touched differ, and `n_elements`
carries that, which is why `op_kind` is a slice column rather than a separate op per kernel:
one utility surface per kind, but one formula for all of them.

A fitted floor is essential here rather than a refinement. Measured on H20, all four kinds sit
at 4-5 us regardless of size below about 128 tokens, so in the decode regime they are entirely
dispatch-bound and the bandwidth term is meaningless. Their *share* of a step therefore grows as
the step shrinks, which is the opposite of how a bandwidth-bound term would behave.

**This table is deliberately not consumed by `ComposedRollout`.** Utility is
`SOL / measured_latency`, and the measured GEMM and attention latencies were collected from a
server whose elementwise kernels were running between them, so the elementwise cost is already
inside those fitted utilities. Adding a separate term double-counts it: measured over the 250
validate buckets it takes decode from 4.84% to 11.18% MRE, prefill 3.72% to 5.84%, and mixed
0.48% to 1.89%, with every median ratio moving away from 1.0. The table exists for the FSDP
training backend, which runs without CUDA graphs and needs an explicit term, and as the third
op family that proves this abstraction generalizes. Wiring it into a rollout prediction requires
first refitting GEMM and attention utility against kernel-only latency.
"""

from __future__ import annotations

from typing import Any

from ..hardware import GpuSpec, dtype_bytes
from ..sol import SolComponents
from .base import register_op

# The kernels vLLM runs between the GEMMs of one transformer block. `qk_norm` is separate from
# `rms_norm` because it normalizes over `head_dim` rather than `hidden_size`, so it lands on a
# different point of the bandwidth curve at the same token count.
ELEMENTWISE_KINDS: tuple[str, ...] = ("rms_norm", "rotary_embedding", "silu_and_mul", "qk_norm")


class ElementwiseKernelOp:
    """Bandwidth-bound pointwise kernels, sliced by dtype and kind."""

    name = "elementwise"
    family = "elementwise"
    slice_columns = ("dtype", "op_kind")
    axis_columns = ("n_elements",)
    fits_floor = True

    def sol(
        self,
        spec: GpuSpec,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """
        Return the roofline of one elementwise kernel call.

        `n_elements` counts the elements *read*; the write is assumed equal, so the byte
        traffic is `2 * n_elements * dtype_bytes`. That holds for all four kinds: rmsnorm and
        qk_norm read and write one tensor, rope reads and writes q and k, and silu_and_mul
        reads `2n` and writes `n` for a total of `3n` against the assumed `2n`: a 1.5x
        underestimate that the fitted utility absorbs, in the same way it absorbs every other
        SOL simplification.

        Args:
            spec (GpuSpec): Hardware the roofline is evaluated against.
            slice_key (tuple[Any, ...]): `(dtype, op_kind)`.
            coords (tuple[float, ...]): `(n_elements,)`.
            floor_s (float): Fitted dispatch floor in seconds, applied inside the `max`.

        Returns:
            SolComponents: A zero compute term, the bandwidth term, and the floor.

        Raises:
            ValueError: If `n_elements` is not positive or `op_kind` is unknown.
        """
        dtype, op_kind = slice_key[0], slice_key[1]
        if op_kind not in ELEMENTWISE_KINDS:
            known = ", ".join(ELEMENTWISE_KINDS)
            raise ValueError(f"Unknown elementwise op_kind {op_kind!r}. Known kinds: {known}.")
        n_elements = float(coords[0])
        if n_elements <= 0.0:
            raise ValueError(f"n_elements must be positive, got {n_elements!r}.")
        memory_s = 2.0 * n_elements * dtype_bytes(dtype) / spec.memory_bandwidth_bytes_per_s
        return SolComponents(math_s=0.0, memory_s=memory_s, floor_s=floor_s)

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the `elementwise:` sweep subtree into an `ElementwiseSweep`."""
        from ....profiler.kernels.config import ElementwiseSweep

        return ElementwiseSweep.from_dict(value)

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Time every elementwise kind across the token grid and write one table."""
        from ....profiler.kernels.elementwise import profile_elementwise

        return profile_elementwise(model, config, writer_factory=writer_factory)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)


ELEMENTWISE_OP = ElementwiseKernelOp()
register_op(ELEMENTWISE_OP)
```

Add `from ...cost.kernel.ops import elementwise as _elementwise_ops  # noqa: F401  (registers the op)`
to `dryrun/profiler/kernels/config.py`'s import block.

- [ ] **Step 5: Write the profiler side**

Create `dryrun/profiler/kernels/elementwise.py`:

```python
"""
Time vLLM's elementwise kernels: rmsnorm, rope, activation, and qk_norm.

Two traps make this module's shape non-obvious, both measured on H20 rather than assumed.

**The raw fused op must be timed, not the module that wraps it.** Calling `RMSNorm(...)` as a
module costs 60.33 us at 16 tokens against 4.57 us for `vllm._custom_ops.rms_norm`: 13.2x.
The gap is Python dispatch through `CustomOp.__call__`, which the server never pays because it
replays these kernels inside a CUDA graph. Timing the module would inflate the table by an
order of magnitude and the utility surface would absorb the error silently.

**A `CustomOp` needs an active `CompilationConfig(custom_ops=["all"])`.** Without it vLLM logs
`Priority not set for op rms_norm, using native implementation` and dispatches its native
PyTorch fallback, which is a different kernel from the fused one the server runs.

`silu_and_mul` is not exposed on `vllm._custom_ops`, so it is reached as `SiluAndMul().op`
after the module is constructed inside the config context; that attribute is what the module's
own `forward_cuda` calls.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..schema import ModelDescriptor, ProfileValidationError
from ._timing import time_cuda_call, torch_dtype
from .config import KernelProfileConfig, model_head_dim, model_intermediate_size
from .measurements import PerfTableWriter

_MAX_POSITION = 40960
_ROPE_THETA = 1.0e6


def profile_elementwise(
    model: ModelDescriptor,
    config: KernelProfileConfig,
    *,
    writer_factory: Callable[[str, str, str], PerfTableWriter],
) -> tuple[Path, ...]:
    """
    Time every elementwise kind across the configured token grid.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        config (KernelProfileConfig): Sweep grid, dtype, and iteration counts.
        writer_factory (Callable[[str, str, str], PerfTableWriter]): Builds a writer from
            `(framework, version, kernel_source)`.

    Returns:
        tuple[Path, ...]: The single written parquet path.

    Raises:
        ProfileValidationError: If the architecture is missing `hidden_size`.
    """
    import torch
    import vllm
    import vllm._custom_ops as ops
    from vllm.config import CompilationConfig, ModelConfig, VllmConfig, set_current_vllm_config

    if model.hidden_size is None:
        raise ProfileValidationError("Model is missing hidden_size.")
    if model.num_attention_heads is None:
        raise ProfileValidationError("Model is missing num_attention_heads.")

    sweep = config.sweep("elementwise")
    measurement = config.measurement
    dtype_name = measurement.dtype
    dtype = torch_dtype(dtype_name)
    hidden = int(model.hidden_size)
    head_dim = model_head_dim(model)
    intermediate = model_intermediate_size(model)
    heads = int(model.num_attention_heads)
    kv_heads = int(model.num_query_groups or heads)

    # `custom_ops=["all"]` is load-bearing: without it these modules dispatch vLLM's native
    # PyTorch fallback rather than the fused CUDA kernel the server runs.
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(model.name),
            tokenizer=str(model.name),
            trust_remote_code=False,
            dtype=dtype_name,
            seed=0,
            max_model_len=_MAX_POSITION,
        ),
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )

    rows: list[dict[str, object]] = []
    with set_current_vllm_config(vllm_config):
        from vllm.model_executor.layers.activation import SiluAndMul
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.rotary_embedding import get_rope

        hidden_norm = RMSNorm(hidden, dtype=dtype).cuda()
        head_norm = RMSNorm(head_dim, dtype=dtype).cuda()
        silu = SiluAndMul().cuda()
        rope = get_rope(
            head_size=head_dim,
            max_position=_MAX_POSITION,
            rope_parameters={"rope_type": "default", "rope_theta": _ROPE_THETA},
            dtype=dtype,
        ).cuda()

        for tokens in sweep.token_counts:
            positions = torch.arange(tokens, device="cuda")

            hidden_in = torch.randn(tokens, hidden, dtype=dtype, device="cuda")
            hidden_out = torch.empty_like(hidden_in)
            rows.append(
                _row(
                    dtype_name,
                    "rms_norm",
                    tokens * hidden,
                    time_cuda_call(
                        lambda: ops.rms_norm(hidden_out, hidden_in, hidden_norm.weight.data, 1e-6),
                        warmup_iterations=measurement.warmup_iterations,
                        active_iterations=measurement.active_iterations,
                    ),
                )
            )

            query = torch.randn(tokens, heads * head_dim, dtype=dtype, device="cuda")
            key = torch.randn(tokens, kv_heads * head_dim, dtype=dtype, device="cuda")
            rows.append(
                _row(
                    dtype_name,
                    "rotary_embedding",
                    tokens * (heads + kv_heads) * head_dim,
                    time_cuda_call(
                        lambda: ops.rotary_embedding(
                            positions, query, key, head_dim, rope.cos_sin_cache, rope.is_neox_style
                        ),
                        warmup_iterations=measurement.warmup_iterations,
                        active_iterations=measurement.active_iterations,
                    ),
                )
            )

            gate_up = torch.randn(tokens, 2 * intermediate, dtype=dtype, device="cuda")
            gate_out = torch.empty(tokens, intermediate, dtype=dtype, device="cuda")
            rows.append(
                _row(
                    dtype_name,
                    "silu_and_mul",
                    tokens * intermediate,
                    time_cuda_call(
                        lambda: silu.op(gate_out, gate_up),
                        warmup_iterations=measurement.warmup_iterations,
                        active_iterations=measurement.active_iterations,
                    ),
                )
            )

            # qk_norm normalizes each head independently, so its rows are token*heads long.
            head_in = torch.randn(tokens * heads, head_dim, dtype=dtype, device="cuda")
            head_out = torch.empty_like(head_in)
            rows.append(
                _row(
                    dtype_name,
                    "qk_norm",
                    tokens * heads * head_dim,
                    time_cuda_call(
                        lambda: ops.rms_norm(head_out, head_in, head_norm.weight.data, 1e-6),
                        warmup_iterations=measurement.warmup_iterations,
                        active_iterations=measurement.active_iterations,
                    ),
                )
            )

            del hidden_in, hidden_out, query, key, gate_up, gate_out, head_in, head_out
            torch.cuda.empty_cache()

    writer = writer_factory("vllm", str(vllm.__version__), "vllm._custom_ops")
    return (writer.write("elementwise", rows),)


def _row(dtype_name: str, op_kind: str, n_elements: int, latency_s: float) -> dict[str, object]:
    """Build one perf-table row for an elementwise measurement."""
    return {
        "dtype": dtype_name,
        "op_kind": op_kind,
        "n_elements": int(n_elements),
        "latency_s": latency_s,
    }
```

Add the sweep block to `examples/profile/kernel/qwen3_8b_h20_tp1.yaml`, matching the file's
existing two-space indentation:

```yaml
elementwise:
  token_counts: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
```

- [ ] **Step 6: Run the tests**

```bash
python -m pytest tests/test_kernel_ops_elementwise.py -q
python -m pytest tests/test_kernel_baseline_regression.py -q
python -m pytest tests/ -q
ruff check dryrun/ tests/
```

Expected: **14 passed** in the new file, all 3 baseline tests pass (the new op must not touch
tp=1 accuracy, since nothing consumes it), the suite grows by exactly the new file's test count and stays green, no new ruff findings.

- [ ] **Step 7: Measure it and gate its interpolation**

```bash
dryrun-profile kernels run --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml --family elementwise
dryrun-profile kernels fit-floor --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml
dryrun-profile kernels holdout --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml \
    --fraction 1.0 --gate 100 --output /tmp/holdout-elementwise.json
python - <<'PY'
import json
rows = {(r["op_name"], r["backend"]): r for r in json.load(open("/tmp/holdout-elementwise.json"))}
key = ("elementwise", "vllm")
assert key in rows, sorted(rows)
r = rows[key]
print(f"elementwise: {r['point_count']} points, MRE {r['mre_percent']:.2f}%, "
      f"p90 {r['p90_percent']:.2f}%, utility {r['utility_min']:.3f}-{r['utility_max']:.3f}")
assert r["point_count"] > 0, "zero scored points is a coverage failure, not an accuracy result"
# The other five must not have moved.
before = {(x["op_name"], x["backend"]): x for x in json.load(open(".superpowers/sdd/baseline/holdout.json"))}
for k, b in before.items():
    a = rows[k]
    assert a["point_count"] == b["point_count"] and abs(a["mre_percent"] - b["mre_percent"]) < 0.01, \
        f"{k} moved: {b['mre_percent']:.2f}% -> {a['mre_percent']:.2f}%"
print("EXISTING OPS UNCHANGED")
PY
```

Expected: a real MRE with `point_count > 0`, and `EXISTING OPS UNCHANGED`. Record the measured
MRE and the fitted elementwise floor: Task 12 documents them. A very high MRE here is a finding
to report, not to hide: the utility of a dispatch-bound kernel is small and the surface may be
poorly conditioned.

- [ ] **Step 8: Commit**

```bash
git add dryrun/cost/kernel/ops/elementwise.py dryrun/profiler/kernels/elementwise.py \
        dryrun/profiler/kernels/config.py examples/profile/kernel/qwen3_8b_h20_tp1.yaml \
        tests/test_kernel_ops_elementwise.py
git commit -m "feat: add the elementwise op, measured but not priced into a rollout

The third op family, which is what proves the abstraction generalizes past the
two that shaped it: a new family needed no edit to floor fitting, holdout
discovery, the writer, or the CLI.

Deliberately not wired into ComposedRollout.predict(). Utility is SOL/measured
and the measured GEMM and attention latencies came from a server already running
these kernels between them, so a separate term double-counts: measured over the
250 validate buckets it takes decode from 4.84% to 11.18%, prefill 3.72% to
5.84%, mixed 0.48% to 1.89%, with every median ratio moving away from 1.0. A
test asserts predict() has no elementwise term so a future change confronts
that reasoning.

Two measurement traps are handled and documented. Timing the vLLM CustomOp
module instead of the raw op measures Python dispatch: 60.33 us against 4.57 us
for rms_norm at 16 tokens, 13.2x, because the server replays these inside a CUDA
graph. And a CustomOp built without CompilationConfig(custom_ops=['all'])
silently dispatches the native PyTorch fallback rather than the fused kernel.

qk_norm is a separate kind from rms_norm because it normalizes over head_dim
rather than hidden_size, landing on a different point of the bandwidth curve at
the same token count.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: The `all_reduce` op

The fourth family and the one with a genuinely different measurement model: a multi-process NCCL
spawn rather than a single kernel call. That is what `measure()`'s coarse signature exists for.

**Files:**
- Create: `dryrun/cost/kernel/ops/collective.py`
- Modify: `dryrun/profiler/kernels/collective.py` (write a perf table; delete the JSON dataclasses)
- Modify: `dryrun/profiler/kernels/config.py` (import the op; add `CollectiveSweep`)
- Modify: `examples/profile/kernel/qwen3_8b_h20_tp1.yaml`; create `examples/profile/kernel/qwen3_8b_h20_tp2.yaml`
- Test: `tests/test_kernel_ops_collective.py`

**Interfaces:**
- Consumes: `KernelOp`, `register_op` (Task 2); `PerfTableWriter.write` (Task 8).
- Produces:
  - `CollectiveSweep(token_counts: tuple[int, ...])`: frozen, in `dryrun/profiler/kernels/config.py`.
  - `AllReduceKernelOp` with `name = "all_reduce"`, `family = "collective"`, `slice_columns = ("dtype", "world_size")`, `axis_columns = ("message_bytes",)`, `fits_floor = True`, registered as `ALL_REDUCE_OP`.
  - `CommQuery(world_size: int, message_bytes: int, dtype: str = "bfloat16")`: frozen, in `dryrun/cost/kernel/database.py`, with `op_name`, `slice_key`, `coords` matching the other queries.
  - `KernelDatabase.query_comm(query: CommQuery) -> float`: replaces the raising stub.
  - `dryrun.profiler.kernels.collective.profile_collective(model, config, *, writer_factory) -> tuple[Path, ...]`; `CollectiveProfile` and `CollectiveMeasurement` are deleted.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kernel_ops_collective.py`:

```python
"""The all_reduce op: ring-cost SOL, world_size in the slice, and a real query_comm."""

from __future__ import annotations

import pytest

from dryrun.cost.kernel.hardware import load_gpu_spec
from dryrun.cost.kernel.ops.base import op_for
from dryrun.cost.kernel.ops.collective import ALL_REDUCE_OP


@pytest.fixture(scope="module")
def spec():
    return load_gpu_spec("h20")


def test_the_op_is_registered_in_the_collective_family():
    assert op_for("all_reduce") is ALL_REDUCE_OP
    assert ALL_REDUCE_OP.family == "collective"


def test_world_size_is_categorical_not_an_axis():
    """Interpolating between 2 and 8 ranks would blend different ring topologies."""
    assert ALL_REDUCE_OP.slice_columns == ("dtype", "world_size")
    assert ALL_REDUCE_OP.axis_columns == ("message_bytes",)


def test_the_op_takes_a_fitted_floor():
    """NCCL has a well-known per-call launch and handshake latency."""
    assert ALL_REDUCE_OP.fits_floor is True


def test_sol_follows_the_ring_all_reduce_cost(spec):
    """A ring all-reduce moves 2*(N-1)/N * bytes per rank, not N*bytes."""
    message_bytes = 1 << 24
    for world_size in (2, 4, 8):
        got = ALL_REDUCE_OP.sol(
            spec, slice_key=("bfloat16", world_size), coords=(float(message_bytes),)
        )
        factor = 2.0 * (world_size - 1) / world_size
        expected = factor * message_bytes / spec.link_bandwidth_bytes_per_s
        assert got.memory_s == pytest.approx(expected, rel=1e-12)
        assert got.math_s == 0.0


def test_the_ring_factor_grows_with_world_size_but_stays_below_two(spec):
    """2*(N-1)/N is 1.0 at N=2 and approaches 2 as N grows; it must never exceed it."""
    message_bytes = 1 << 20
    previous = 0.0
    for world_size in (2, 4, 8, 16):
        got = ALL_REDUCE_OP.sol(
            spec, slice_key=("bfloat16", world_size), coords=(float(message_bytes),)
        ).memory_s
        assert got > previous
        previous = got
    limit = 2.0 * message_bytes / spec.link_bandwidth_bytes_per_s
    assert previous < limit


def test_a_single_rank_all_reduce_costs_no_transfer(spec):
    """At world_size 1 there is no collective; only the floor can bind."""
    got = ALL_REDUCE_OP.sol(spec, slice_key=("bfloat16", 1), coords=(1.0e6,), floor_s=5.0e-6)
    assert got.memory_s == 0.0
    assert got.latency_s == pytest.approx(5.0e-6, rel=1e-12)


def test_a_non_positive_world_size_is_rejected(spec):
    with pytest.raises(ValueError, match="world_size"):
        ALL_REDUCE_OP.sol(spec, slice_key=("bfloat16", 0), coords=(1.0e6,))


def test_a_non_positive_message_size_is_rejected(spec):
    with pytest.raises(ValueError, match="message_bytes"):
        ALL_REDUCE_OP.sol(spec, slice_key=("bfloat16", 4), coords=(0.0,))


def test_latency_is_the_last_table_column():
    assert ALL_REDUCE_OP.table_columns() == ("dtype", "world_size", "message_bytes", "latency_s")


def test_parse_sweep_builds_the_token_grid():
    sweep = ALL_REDUCE_OP.parse_sweep({"token_counts": [16, 4096]})
    assert sweep.token_counts == (16, 4096)


def test_a_comm_query_exposes_the_ops_slice_and_coords():
    from dryrun.cost.kernel.database import CommQuery

    query = CommQuery(world_size=4, message_bytes=1 << 20)
    assert query.op_name == "all_reduce"
    assert query.slice_key == ("bfloat16", 4)
    assert query.coords == (float(1 << 20),)


def test_a_comm_query_rejects_a_non_positive_world_size():
    from dryrun.cost.kernel.database import CommQuery
    from dryrun.cost.kernel.interp import KernelCoverageError

    with pytest.raises(KernelCoverageError, match="world_size"):
        CommQuery(world_size=0, message_bytes=1 << 20)


def test_query_comm_no_longer_raises_unconditionally():
    """It raised on every call as a placeholder; now it must reach the table machinery."""
    import inspect

    from dryrun.cost.kernel.database import KernelDatabase

    source = inspect.getsource(KernelDatabase.query_comm)
    assert "phase 1b" not in source, "query_comm is still the placeholder stub"
    assert "_grid" in source or "_sol" in source, "query_comm does not price through the table"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_kernel_ops_collective.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dryrun.cost.kernel.ops.collective'`.

- [ ] **Step 3: Add the sweep type and the cost-side op**

In `dryrun/profiler/kernels/config.py`, add beside `ElementwiseSweep`:

```python
@dataclass(frozen=True)
class CollectiveSweep:
    """Token counts to measure all-reduce message sizes at."""

    token_counts: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CollectiveSweep:
        return cls(
            token_counts=_positive_int_tuple(
                value["token_counts"], field="collective.token_counts"
            )
        )
```

Create `dryrun/cost/kernel/ops/collective.py`:

```python
"""
Tensor-parallel all-reduce, the one op whose measurement is not a single kernel call.

vLLM issues an all-reduce over per-token hidden activations after each attention block and each
MLP block, so a `tp>1` step pays two per layer. Without this table a composition at `tp>1` would
silently omit every one of them, which is why `ComposedRollout` rejected `tp>1` outright before
this op existed.

`world_size` is a slice column rather than an interpolation axis. A 2-rank all-reduce and an
8-rank one run different ring schedules over different link topologies, so blending them would
produce a number describing neither.

The SOL is the standard ring-all-reduce lower bound: each rank sends and receives
`(N-1)/N * bytes` in the reduce-scatter phase and the same again in the all-gather phase, for
`2*(N-1)/N * bytes` of traffic per rank. That factor is 1.0 at two ranks and approaches 2 as the
world grows, never reaching it. NCCL may choose a tree or a different algorithm at small sizes,
which the fitted utility absorbs in the same way it absorbs every other SOL simplification.

A fitted floor is essential: NCCL pays a per-call launch and handshake cost that dominates
entirely at small message sizes, which is the regime a decode step lives in.
"""

from __future__ import annotations

from typing import Any

from ..hardware import GpuSpec
from ..sol import SolComponents
from .base import register_op


class AllReduceKernelOp:
    """Ring all-reduce over per-token hidden activations."""

    name = "all_reduce"
    family = "collective"
    slice_columns = ("dtype", "world_size")
    axis_columns = ("message_bytes",)
    fits_floor = True

    def sol(
        self,
        spec: GpuSpec,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """
        Return the roofline of one all-reduce.

        Args:
            spec (GpuSpec): Hardware; `link_bandwidth_bytes_per_s` carries the interconnect.
            slice_key (tuple[Any, ...]): `(dtype, world_size)`.
            coords (tuple[float, ...]): `(message_bytes,)`, the per-rank tensor size in bytes.
            floor_s (float): Fitted NCCL launch floor in seconds, applied inside the `max`.

        Returns:
            SolComponents: A zero compute term, the ring transfer term, and the floor. At
                `world_size` 1 the transfer term is zero, because no collective runs.

        Raises:
            ValueError: If `world_size` or `message_bytes` is not positive.
        """
        world_size = int(slice_key[1])
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size!r}.")
        message_bytes = float(coords[0])
        if message_bytes <= 0.0:
            raise ValueError(f"message_bytes must be positive, got {message_bytes!r}.")
        # Ring all-reduce: reduce-scatter then all-gather, each moving (N-1)/N per rank.
        factor = 2.0 * (world_size - 1) / world_size
        memory_s = factor * message_bytes / spec.link_bandwidth_bytes_per_s
        return SolComponents(math_s=0.0, memory_s=memory_s, floor_s=floor_s)

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the `collective:` sweep subtree into a `CollectiveSweep`."""
        from ....profiler.kernels.config import CollectiveSweep

        return CollectiveSweep.from_dict(value)

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Spawn `tensor_parallel` ranks and time an all-reduce at each message size."""
        from ....profiler.kernels.collective import profile_collective

        return profile_collective(model, config, writer_factory=writer_factory)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)


ALL_REDUCE_OP = AllReduceKernelOp()
register_op(ALL_REDUCE_OP)
```

Add `from ...cost.kernel.ops import collective as _collective_ops  # noqa: F401  (registers the op)`
to `dryrun/profiler/kernels/config.py`.

- [ ] **Step 4: Add `CommQuery` and implement `query_comm`**

In `dryrun/cost/kernel/database.py`, add beside `AttentionQuery`:

```python
@dataclass(frozen=True)
class CommQuery:
    """One tensor-parallel all-reduce to price."""

    world_size: int
    message_bytes: int
    dtype: str = "bfloat16"
    backend: str | None = None

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise KernelCoverageError(f"Comm world_size must be positive, got {self.world_size!r}.")
        if self.message_bytes <= 0:
            raise KernelCoverageError(
                f"Comm message_bytes must be positive, got {self.message_bytes!r}."
            )

    @property
    def op_name(self) -> str:
        """Return the parquet op name this query reads."""
        return "all_reduce"

    @property
    def slice_key(self) -> tuple[Any, ...]:
        """Return the categorical key in the all-reduce op's slice order."""
        return (canonical_dtype(self.dtype), self.world_size)

    @property
    def coords(self) -> tuple[float, ...]:
        """Return interpolation coordinates in the all-reduce op's axis order."""
        return (float(self.message_bytes),)
```

Replace the `query_comm` stub with a real implementation:

```python
    def query_comm(self, query: CommQuery) -> float:
        """
        Return the predicted latency in seconds of one all-reduce.

        Args:
            query (CommQuery): Per-rank message size and world size.

        Returns:
            float: `SOL(query) / interpolated utility`, in seconds.

        Raises:
            KernelCoverageError: If no table or slice covers the query.
        """
        grid = self._grid(query)
        return self._sol(self._gpu_spec, query, query.coords) / grid.utility(query.coords)
```

Widen the `_sol`, `_grid`, and `utility` annotations from `GemmQuery | AttentionQuery` to
`GemmQuery | AttentionQuery | CommQuery`. No body changes are needed: `_sol` already dispatches
through `op_for(query.op_name)` after Task 6.

Delete `query_elementwise` entirely. Nothing consumes an elementwise price (Task 10), so a
method that only raises is dead code, and the project contract says to remove obsolete paths
rather than keep them.

- [ ] **Step 5: Rewrite the collective profiler to write a table**

In `dryrun/profiler/kernels/collective.py`, delete `CollectiveMeasurement` and
`CollectiveProfile`, and rewrite `profile_collective`:

```python
def profile_collective(
    model: ModelDescriptor,
    config: KernelProfileConfig,
    *,
    writer_factory: Callable[[str, str, str], PerfTableWriter],
) -> tuple[Path, ...]:
    """
    Time an all-reduce of per-token hidden activations at `config.tensor_parallel` ranks.

    Args:
        model (ModelDescriptor): Provides `hidden_size` for the message size.
        config (KernelProfileConfig): `tensor_parallel` sets the NCCL world size;
            `collective.token_counts` sets the message sizes.
        writer_factory (Callable[[str, str, str], PerfTableWriter]): Builds a writer from
            `(framework, version, kernel_source)`.

    Returns:
        tuple[Path, ...]: The written parquet path, or empty at `tensor_parallel <= 1`
            because no TP all-reduce runs at that world size.

    Raises:
        ProfileValidationError: If the architecture is missing `hidden_size`.
    """
    import torch
    import torch.multiprocessing as mp

    if config.tensor_parallel <= 1:
        dryrun_logger.warning(
            f"tensor_parallel is {config.tensor_parallel}; no TP all-reduce runs at that world "
            f"size, so no collective table is written."
        )
        return ()
    if model.hidden_size is None:
        raise ProfileValidationError("Model is missing hidden_size.")
    token_counts = config.sweep("all_reduce").token_counts
    port = _free_port()
    manager = mp.Manager()
    results = manager.list([None] * len(token_counts))
    mp.spawn(
        _collective_worker,
        args=(config.tensor_parallel, port, model.hidden_size, token_counts, config.measurement, results),
        nprocs=config.tensor_parallel,
        join=True,
    )
    element_bytes = torch.empty((), dtype=torch_dtype(config.measurement.dtype)).element_size()
    rows = [
        {
            "dtype": config.measurement.dtype,
            "world_size": config.tensor_parallel,
            "message_bytes": int(token_count) * int(model.hidden_size) * element_bytes,
            "latency_s": float(latency_s),
        }
        for token_count, latency_s in zip(token_counts, results, strict=True)
    ]
    writer = writer_factory("nccl", str(torch.version.cuda or "unknown"), "torch.distributed.all_reduce")
    return (writer.write("all_reduce", rows),)
```

Update the imports: add `from collections.abc import Callable`, `from pathlib import Path`,
`import logging`, `from .measurements import PerfTableWriter`, and a module logger
`dryrun_logger = logging.getLogger(__file__)`. Move `import torch`, `import torch.distributed as
dist`, and `import torch.multiprocessing as mp` inside the functions that use them so importing
this module stays cheap; `_collective_worker` needs `dist` and `torch` at its top.

Add to the module docstring:

```
Measurements land in the same parquet perf-table tree as every other op, so an all-reduce is
priced by the same `SOL / utility` split and gated by the same holdout instrument. It used to
return a bespoke JSON structure that nothing consumed.
```

Add the sweep block to both configs. In `examples/profile/kernel/qwen3_8b_h20_tp1.yaml`:

```yaml
collective:
  token_counts: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
```

Create `examples/profile/kernel/qwen3_8b_h20_tp2.yaml` by copying the tp1 file, changing
`tensor_parallel: 2`, `gpu_count: 2`, and `output_dir` to
`../../../outputs/profile/qwen3_8b/kernel_tp2`.

- [ ] **Step 6: Wire the comm term into `ModelOps` and `ComposedRollout`**

In `dryrun/cost/kernel/model_ops.py`, add `comm_queries` after `gemm_queries`:

```python
    def comm_queries(self, token_count: int) -> tuple[tuple[CommQuery, int], ...]:
        """
        Return the all-reduce queries of one step, each with its multiplier.

        vLLM issues one all-reduce after the attention block's output projection and one after
        the MLP's down projection, so a tensor-parallel step pays two per layer. The count is
        returned as data with an explicit multiplier for the same reason `logits_op` carries
        one: written inline it sits one indentation level away from a plausible-looking
        layer-count error.

        Args:
            token_count (int): Tokens the all-reduce covers. In decode this equals the request
                count; in prefill it is the total prompt token count.

        Returns:
            tuple[tuple[CommQuery, int], ...]: `(query, multiplier)` pairs, empty at
                `tensor_parallel == 1` because no collective runs.
        """
        if self._tensor_parallel <= 1:
            return ()
        message_bytes = token_count * self._hidden_size * self._element_bytes
        query = CommQuery(
            world_size=self._tensor_parallel, message_bytes=message_bytes, dtype=self._dtype
        )
        return ((query, 2 * self._num_layers),)
```

`ModelOps.__init__` must record the four new attributes it reads. Add after `self._head_dim`:

```python
        self._tensor_parallel = int(tensor_parallel)
        if model.hidden_size is None:
            raise ProfileValidationError("Model is missing hidden_size.")
        # The all-reduce carries the full hidden size: TP shards the projections, then reduces
        # across ranks to reassemble the unsharded activation.
        self._hidden_size = int(model.hidden_size)
        self._dtype = dtype
        self._element_bytes = dtype_bytes(dtype)
```

with `dtype: str = "bfloat16"` added as a keyword parameter to both `__init__` and
`from_checkpoint` (threading it through), plus `from .database import CommQuery` and
`from .hardware import dtype_bytes` in the imports. Expose `tensor_parallel` as a property so
`ComposedRollout` need not re-derive it:

```python
    @property
    def tensor_parallel(self) -> int:
        """Return the parallelism degree these ops were sharded for."""
        return self._tensor_parallel
```

In `dryrun/cost/rollout/composed.py`, delete the `tensor_parallel != 1` rejection block entirely
and add the comm term to `predict`:

```python
        for query, multiplier in self._ops.comm_queries(token_count):
            total += multiplier * self._database.query_comm(query)
```

placed immediately after the GEMM loop and before the attention term. Replace the `Raises:`
entry for `tensor_parallel` in `__init__`'s docstring with a note that `stride` must be
positive, and update the class docstring's known-limitation paragraph to state that `tp>1` is
now supported but **end-to-end unvalidated**, naming the absence of a vLLM `tp=2` baseline.

- [ ] **Step 7: Run the tests**

```bash
python -m pytest tests/test_kernel_ops_collective.py -q
python -m pytest tests/test_kernel_model_ops.py tests/test_rollout_composed.py \
    tests/test_rollout_composed_config.py -q
python -m pytest tests/test_kernel_baseline_regression.py -q
python -m pytest tests/ -q
ruff check dryrun/ tests/
grep -rn "CollectiveProfile\|CollectiveMeasurement\|query_elementwise" dryrun/ tests/ --include=*.py \
    || echo "RETIRED APIS GONE"
```

Expected: **13 passed** in the new file, all 3 baseline tests pass (tp=1 numbers must be
untouched, since `comm_queries` returns empty there), the suite grows by exactly the new file's test count and stays green, no new ruff findings,
and `RETIRED APIS GONE`. `tests/test_rollout_composed.py`'s
`test_tensor_parallel_above_one_is_rejected_naming_the_missing_tables` now contradicts the
feature: replace it with a test that `tp=2` constructs, and say so in the commit message.

- [ ] **Step 8: Measure at tp=2 and gate it**

```bash
dryrun-profile kernels run --config examples/profile/kernel/qwen3_8b_h20_tp2.yaml --family collective
dryrun-profile kernels fit-floor --config examples/profile/kernel/qwen3_8b_h20_tp2.yaml
dryrun-profile kernels holdout --config examples/profile/kernel/qwen3_8b_h20_tp2.yaml \
    --fraction 1.0 --gate 100 --output /tmp/holdout-collective.json
python - <<'PY'
import json
rows = {(r["op_name"], r["backend"]): r for r in json.load(open("/tmp/holdout-collective.json"))}
key = ("all_reduce", "nccl")
assert key in rows, sorted(rows)
r = rows[key]
print(f"all_reduce: {r['point_count']} points, MRE {r['mre_percent']:.2f}%, "
      f"p90 {r['p90_percent']:.2f}%, utility {r['utility_min']:.3f}-{r['utility_max']:.3f}")
assert r["point_count"] > 0, "zero scored points is a coverage failure, not an accuracy result"
PY
```

Expected: a real MRE with `point_count > 0`. Record it for Task 12. Note this measures **one**
world size, so the slice has a single `world_size` value and interpolation happens only along
`message_bytes`: that is correct and is why `world_size` is categorical.

- [ ] **Step 9: Commit**

```bash
git add dryrun/cost/kernel/ops/collective.py dryrun/cost/kernel/database.py \
        dryrun/cost/kernel/model_ops.py dryrun/cost/rollout/composed.py \
        dryrun/profiler/kernels/collective.py dryrun/profiler/kernels/config.py \
        examples/profile/kernel/qwen3_8b_h20_tp1.yaml examples/profile/kernel/qwen3_8b_h20_tp2.yaml \
        tests/test_kernel_ops_collective.py tests/test_rollout_composed.py
git commit -m "feat: add the all_reduce op and unblock tensor parallelism

The fourth family, and the one that justifies measure()'s coarse signature: a
collective is timed by spawning world_size processes with torch.multiprocessing
where only rank 0 keeps timings, so there is no per-shape call for a
fine-grained interface to hook.

query_comm stops raising and prices through the same SOL/utility split as every
other op, so ComposedRollout's tensor_parallel != 1 rejection is removed.
ModelOps.comm_queries returns two all-reduces per layer: after o_proj and
after down_proj: with the multiplier as data for the same reason logits_op
carries one.

world_size is a slice column rather than an axis: a 2-rank and an 8-rank
all-reduce run different ring schedules over different topologies, so blending
them would describe neither.

collective.py wrote a bespoke JSON structure that nothing consumed; it now
writes a parquet table into the shared tree, so an all-reduce is gated by the
same holdout instrument as every other op. query_elementwise is deleted rather
than left raising, since nothing prices elementwise work.

tp>1 is now constructible but its end-to-end accuracy is UNVALIDATED: no vLLM
tp=2 baseline exists, so kernels validate cannot score it. The class docstring
says so.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 12: Document what was measured and what is still unknown

**Files:**
- Modify: `docs/composed-rollout.md`
- Modify: `examples/profile/kernel/README.md`
- Modify: `.claude/codebase-map.md`
- Create: `docs/kernel-ops.md`

**Interfaces:**
- Consumes: the measured figures from Tasks 9-11.
- Produces: no code.

- [ ] **Step 1: Collect every figure fresh**

Do not copy numbers from this plan. Re-run and record what the branch actually produces.

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/DryRun-RL

dryrun-profile kernels holdout --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml \
    --fraction 1.0 --gate 100 --output /tmp/final-holdout-tp1.json
dryrun-profile kernels holdout --config examples/profile/kernel/qwen3_8b_h20_tp2.yaml \
    --fraction 1.0 --gate 100 --output /tmp/final-holdout-tp2.json
for surface in pchip shepard; do
  dryrun-profile kernels validate \
      --config examples/profile/kernel/qwen3_8b_h20_tp1.yaml \
      --baseline outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
      --model-path /apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B \
      --surface "${surface}" --min-samples 1 --gate 100 \
      --output "/tmp/final-validate-${surface}.json"
done
python - <<'PY'
import glob, json
for path in sorted(glob.glob("/tmp/final-holdout-tp*.json")):
    print(f"== {path}")
    for r in json.load(open(path)):
        print(f"  {r['op_name']:22s} {r['backend']:12s} n={r['point_count']:4d} "
              f"MRE={r['mre_percent']:6.2f}% p90={r['p90_percent']:6.2f}%")
for surface in ("pchip", "shepard"):
    print(f"== validate {surface}")
    for r in json.load(open(f"/tmp/final-validate-{surface}.json")):
        print(f"  {r['phase']:8s} n={r['bucket_count']:4d} MRE={r['mre_percent']:6.2f}% "
              f"median_ratio={r['median_ratio']:.4f}")
for path in sorted(glob.glob("outputs/profile/qwen3_8b/kernel_tp*/kernel/h20/*/*/*/floors.json")):
    data = json.load(open(path))
    floors = {k: v for k, v in data.items() if isinstance(v, float)}
    print(f"== floors {path}\n  " + ", ".join(f"{k}={v*1e6:.1f}us" for k, v in sorted(floors.items())))
PY
```

- [ ] **Step 2: Write `docs/kernel-ops.md`**

Cover, in this order:

1. **What a `KernelOp` is and why it exists.** One paragraph: op knowledge was spread across ten
   sites, but the defect was two independent SOL bindings that could drift and silently change
   utility on one side of the holdout round trip.
2. **How to add an op.** The concrete checklist: write the class, register it, add a sweep block
   to the config, and add `measure`. State explicitly that floor fitting, holdout discovery, the
   writer, and the CLI need **no** edits, and point at `elementwise` as the worked example that
   proved it.
3. **The four lookups** and the rule that nothing indexes `OP_REGISTRY` directly.
4. **The registered ops**, as a table: name, family, slice columns, axis columns, `fits_floor`,
   and the Step-1 holdout MRE for each.
5. **Why `measure()` is per-family, not per-op**: one attention sweep allocates one KV cache and
   times both phases against it.
6. **The `floors.json` format**, keyed by op name, with the note that an old phase-keyed file is
   rejected and the measured reason (floor 0 scores 34.89% MRE against 6.63% fitted).

- [ ] **Step 3: Update `docs/composed-rollout.md`**

Three changes:

1. Replace the "partly absorbed into the fitted utility" wording with the measured result:
   adding elementwise takes decode 4.84% -> 11.18%, prefill 3.72% -> 5.84%, mixed 0.48% ->
   1.89%, every median ratio moving away from 1.0, so the absorption is **complete** and a
   separate term double-counts. Name the test that pins it.
2. Add a `tp>1` section: constructible now, comm term is two all-reduces per layer, kernel-level
   holdout MRE from Step 1, and **end-to-end unvalidated** because no vLLM tp=2 baseline exists.
3. Add the small-batch finding: decode `median_ratio` is 1.1312 over 66 buckets at reqs 1-4
   against ~1.007-1.014 elsewhere, so the aggregate 1.19% is an averaging artifact. List the
   three ruled-out causes with their numbers: CUDA graph capture (eager/graph agree 1-12%),
   invalid kernel summation (4.644 ms summed vs 4.721 ms as one 36-layer graph replay, 1.7%), and
   `m=1` interpolation (`m=1` is measured; table equals query bit-for-bit). State that kernel
   overlap in the real server is the remaining candidate and that diagnosing it is future work.

- [ ] **Step 4: Update `examples/profile/kernel/README.md`**

Read it first and match its structure. Add `--family elementwise` and `--family collective` to
the run documentation, note that `--family` choices come from the registry, and record the two
elementwise measurement traps (raw op vs module, `custom_ops=["all"]`) since anyone re-running
the sweep needs them. Add the tp=2 config to the examples.

- [ ] **Step 5: Update `.claude/codebase-map.md`**

Read it first and match its existing format. Add the five new `cost/kernel/ops/` modules and
`profiler/kernels/elementwise.py`; record that `TableSchema` and `schema_for` are gone from
`table.py`; note that `dryrun/cost/kernel/ops/*` imports `dryrun/profiler/kernels/config` lazily
inside `parse_sweep` and `measure`, and that `dryrun/profiler/kernels/config` imports the op
modules at module scope to populate the registry.

- [ ] **Step 6: Verify every claim traces to a command**

```bash
python -m pytest tests/ -q
ruff check dryrun/ tests/
grep -rn "partly absorbed\|phase 1b" docs/ examples/ || echo "STALE CLAIMS REMOVED"
grep -rn "0.03%" docs/ || echo "NO STALE ESTIMATE"
```

Expected: suite green, 7 pre-existing ruff findings only, `STALE CLAIMS REMOVED`, and
`NO STALE ESTIMATE`. Every figure in the new docs must trace to a Step 1 command's output: if
you cannot point at the run that produced a number, remove the number.

- [ ] **Step 7: Commit**

```bash
git add docs/kernel-ops.md docs/composed-rollout.md examples/profile/kernel/README.md \
        .claude/codebase-map.md
git commit -m "docs: record the op abstraction and what measuring the new ops revealed

Documents how to add an op and states the payoff concretely: floor fitting,
holdout discovery, the writer, and the CLI needed no edits for elementwise or
all_reduce, which is the abstraction's only real proof.

Replaces composed-rollout.md's 'partly absorbed into the fitted utility' with
the measured result. Adding elementwise takes decode from 4.84% to 11.18% and
moves every median ratio away from 1.0, so the absorption is complete rather
than partial and a separate term double-counts.

Records the small-batch finding: decode over-prediction is 13% at reqs 1-4 and
near zero elsewhere, so the aggregate 1.19% was an averaging artifact. Lists the
three causes ruled out with their numbers, and names kernel overlap as the
remaining candidate.

States plainly that tp>1 is constructible but end-to-end unvalidated, since no
vLLM tp=2 baseline exists to score it against.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes

Recorded so the executor knows which risks were considered rather than missed.

**Task ordering is dependency-driven.** Task 1 must precede everything because a baseline
captured after an edit proves nothing. Task 5 (retire `TableSchema`) precedes Tasks 6-8 because
they all consume `KernelTable.op`. Task 7 (re-key floors) must follow Task 6, because Task 6's
attention test asserts the op-keyed lookup and is deliberately left red in between. The stage-1
gate blocks Tasks 10-11 so a new op's error can never be confused with refactor drift.

**The riskiest tasks are 5 and 7.** Task 5 edits eight test files; its gate is that only
mechanical renames appear in `git diff --stat tests/` and that the two behavioural baseline tests
still pass. Task 7 changes a persisted file format; its gate is that regenerated floors reproduce
32/22 and 4/6 us exactly, and it deliberately breaks every committed `floors.json` rather than
reading them leniently, because a silent zero floor costs five-fold accuracy.

**Three plan claims were corrected by measurement while writing**, and the corrections are
load-bearing. Elementwise is **1.20% of a decode step, not 0.03%**, because these kernels are
flat at 4-5 us rather than bandwidth-scaling. Wiring it in makes decode **2.3x worse**, so it is
built but not consumed: an executor who "finishes the job" by adding it to `predict()` will
regress the model. And timing the vLLM module instead of the raw op inflates the measurement
**13.2x**, which would have poisoned the table invisibly.

**The `GemmOp` name collision is resolved in the plan, not left to the implementer.**
`model_ops.GemmOp` already exists with a different meaning and is read by two test files; every
new op class therefore carries the `KernelOp` suffix.

**Suite counts are checked as deltas, not as memorised totals.** The baseline is a measured
635 at commit `72f00f5` and the plan adds 90 collected tests, ending at 725. Each task asserts
only that the suite is green and grew by its own new-test count, because a chain of hardcoded
intermediate totals silently goes wrong the moment one task's count is off by one, and then
every later task appears to fail. Reconcile a mismatched delta rather than ignoring it: too
small usually means a parametrization was dropped.

**Deliberately out of scope**, each a real gap rather than an oversight: diagnosing the 13%
small-batch over-prediction; refitting GEMM and attention utility against kernel-only latency so
elementwise could be priced without double-counting; a vLLM `tp=2` baseline to validate the comm
term end to end; measuring mixed batches directly; denser `context_seq_lens` below 1024 to fix
`context_attention/flashinfer` at 9.87%; and cross-hardware transfer, which stays unvalidated
with one GPU type available.
