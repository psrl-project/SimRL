# Testing

The suite mixes two kinds of test, and telling them apart matters: some are gates that must
pass anywhere, and some are instruments that only work on a machine holding measured data or a
local checkpoint. The second kind **skips** rather than fails on a fresh clone.

---

## Commands

```bash
# Run the full test suite.
pytest

# Check style and import hygiene (6 pre-existing findings, see below).
ruff check .

# Syntax-check all example shell scripts without executing them.
bash -n examples/profile/vllm/profile_qwen3_8b.sh
bash -n examples/profile/megatron/profile_qwen3_8b.sh
bash -n examples/simulate/simulate.sh

# Smoke-test every CLI entry point.
dryrun --help
dryrun-profile --help
dryrun-regress --help
```

The suite collects 946 tests. With `MODEL_PATH` set to a local checkpoint, all 946 pass. Without
it, 944 pass and 2 skip. The 6 `ruff` findings are pre-existing (5 import-ordering, 1 unused
variable in `tests/test_training_cost.py`) and are not introduced by any recent changes.

Some tests additionally need measured tables on disk and skip cleanly without them —
`test_rollout_composed_tp2.py` and the `tp=2` cases in `test_rollout_composed_config.py` need the
`tp=2` collective sweep, which `outputs/` being gitignored means a fresh clone does not have.

---

## CI gates vs local instruments

Not every test is a CI gate. The measured performance tables live in `outputs/`, which is
gitignored. On a fresh clone, `outputs/` is empty. Tests that depend on those tables — or on a
local model checkpoint — skip rather than fail. A skip is not a pass and it is not a failure; it
means the test cannot be evaluated in the current environment.

The distinction matters because some tests are *local instruments*: they are useful to the person
running a sweep on real hardware, but they cannot be part of a CI check on a machine that has no
GPU and no profiling output.

### Tests that skip without `MODEL_PATH`

Two tests require `MODEL_PATH` to point at a real HuggingFace checkpoint with a `config.json`:

- **`tests/test_kernel_ops_base.py`** — the `test_the_config_serves_a_sweep_per_op_name` test
  loads the example kernel config, which reads the checkpoint path from `${oc.env:MODEL_PATH,...}`.
  Without a local checkpoint the config loader raises before the assertions under test can run,
  so the test skips rather than passing vacuously.

- **`tests/test_kernel_validate.py`** — the `test_a_phase_dropped_by_min_samples_is_reported_not_silent`
  test calls `cmd_kernels_validate`, which also resolves the checkpoint before it reaches the
  coverage-warning logic under test. Same skip condition.

Both tests call `pytest.skip("MODEL_PATH is not set to a local checkpoint")` after checking
`os.environ.get("MODEL_PATH", "")` and verifying that `config.json` exists at that path.

### `tests/test_kernel_baseline_regression.py`

This test pins ten accuracy numbers from a completed kernel sweep. It is a local instrument: it
skips on a fresh clone because `outputs/` is empty. Its purpose is to catch unintentional changes
to the interpolation pipeline on a machine where the sweep has already been run.

---

## Re-running a sweep invalidates pinned numbers

`latency_s` is a wall-clock measurement. Two runs of identical code moved the GEMM holdout MRE
from 3.896% to 3.828%, and six repeats of a single GEMM kernel span 0.351% of their median. This
is expected: GPU kernel timing is not bit-reproducible.

The consequence for `tests/test_kernel_baseline_regression.py` is that its pinned figures shift
between identical runs for reasons entirely unrelated to code changes. Only `latency_s` fails to
reproduce; every other parquet column — shapes, derived utility, slice keys — reproduces
bit-identically. A sweep-level comparison must therefore exclude `latency_s`.

**Re-pinning procedure.** Before re-measuring, back up the current parquet files if the old
accuracy numbers are needed for reference. After the new sweep completes, run:

```bash
pytest tests/test_kernel_baseline_regression.py --update-snapshots 2>/dev/null || \
    python tests/test_kernel_baseline_regression.py --regenerate
```

Check which flag the test accepts by reading its source; the mechanism varies. Alternatively,
re-run `dryrun-profile kernels holdout --fraction 1.0` and copy the resulting MRE values into the
test's expected-values table manually. Whatever the mechanism, the numbers in the test and the
numbers in `holdout_report.json` must come from the same sweep.

---

## Import hygiene tests

Two tests exist specifically to ensure that the cost path and shape-derivation path never pull in
the GPU stack at import time:

**`tests/test_kernel_shapes_import.py`** runs a subprocess probe that imports
`dryrun.profiler.kernels.shapes` in a fresh interpreter and asserts that neither `torch` nor
`vllm` appear in `sys.modules` afterward. A 4.2-second GPU-stack import has no place on the cost
path that reads parquet and prices kernels.

**`tests/test_kernel_package_imports.py`** tests four related isolation properties:
`dryrun.profiler.kernels.config` imports without loading torch;
`dryrun.profiler.kernels.attention` imports without loading torch;
`dryrun.profiler.kernels.profile_gemm` is callable and *does* load torch (confirming the lazy-load
mechanism works in both directions);
`dryrun.profiler.kernels.PerfTableWriter` is accessible without torch.

What these tests protect: if any torch-importing module is pulled into a package `__init__` at the
wrong level, the cost path starts paying the GPU import cost on every simulator invocation. Because
that cost is paid silently (no exception, just slow start), it could go unnoticed for a long time.
The subprocess probe ensures the check runs in a genuinely fresh interpreter rather than in the
same process where torch is already loaded.
