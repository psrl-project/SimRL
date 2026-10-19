# Testing

The suite under `tests/` is flat. File names carry the grouping, and the grouping decides what a
test needs to run:

| Prefix | Covers |
|---|---|
| `test_kernel_*` | Kernel ops, sweeps, grids, perf tables, SOL, floors, holdout, the composed validation, and the import boundary |
| `test_engine_cpp_*` | The C++ engine and its agreement with the Python engine |
| `test_rollout_*`, `test_advance.py`, `test_composed_term_memo.py` | Rollout cost models, segment integration and the closed form |
| `test_training_*`, `test_fsdp_*` | Training cost models, the shared latency form, FSDP inventory and the FSDP profiler |
| `test_megatron_launch.py`, `test_profile_adapters.py`, `test_profiling_cli.py`, `test_hf_model.py` | Profiler adapters and the profiling CLI |
| `test_forecast_*`, `test_staleness_report.py` | The drain and pipeline forecasts, and the staleness report |
| `test_kv_*`, `test_eviction_strategies.py`, `test_router*.py`, `test_length_aware.py`, `test_inventory_deadline_strategy.py` | The engine's KV pool and the router |
| `test_policy_*`, `test_scg.py`, `test_simulator_runtime.py`, `test_dynamic_parallel_fleet.py` | Policies, the simulator driver and the runtime facade |
| `test_workload_*`, `test_prefix_probe.py` | Distribution sampling, group traces, sharding and collection |
| `test_telemetry.py`, `test_jsonl_store.py`, `test_psrl_log.py`, `test_plots.py` | Telemetry streams, the store, PSRL conversion and plotting |
| `test_e2e.py`, `test_config.py`, `test_logging_setup.py`, `test_undo.py` | End-to-end runs, the Hydra config, logging and the undo log |

Tests live in one directory because the package's layers are importable independently: a kernel
test does not need the simulator, and a policy test does not need a GPU.

---

## Commands

```bash
# Everything. Needs no GPU and no checkpoint for the majority of the suite.
pytest

# The same, in parallel, using the dev extra's pytest-xdist.
pytest -n auto

# One group.
pytest tests -k kernel
pytest tests/test_engine_cpp_undo.py -q

# Style and import hygiene.
ruff check .

# A file's tests without collecting the rest of the suite.
pytest tests/test_forecast_plan.py -q
```

`pyproject.toml` sets `testpaths = ["tests"]`. The dev extra installs `pytest`,
`pytest-xdist` and `ruff`, with ruff configured for `E`, `F`, `W` and `I` at a line length of 119.

Each of the seven console scripts can be smoke-tested with `--help`, which builds its parser
without running a workflow:

```bash
srl --help
srl-profile --help
srl-profile kernels --help
srl-regress --help
srl-plot --help
srl-psrl-plot --help
srl-compare --help
srl-workload --help
```

`bash -n` syntax-checks the example shell scripts without executing them:

```bash
for script in examples/*/*.sh; do bash -n "$script" || echo "FAILED: $script"; done
```

---

## What a skip means

Some tests need data that is not in the repository. They skip rather than fail, and a skip is
reported separately from a pass.

**`MODEL_PATH`.** The example profiling configs read the checkpoint from
`${oc.env:MODEL_PATH,...}`, so a test that loads one skips unless `MODEL_PATH` names a directory
holding a `config.json`. The cases are in `tests/test_kernel_ops_base.py` and
`tests/test_kernel_validate.py`: without the checkpoint the loader fails before the assertions
under test can run.

```bash
export MODEL_PATH=/path/to/Qwen3-8B
```

**Machine-local checkpoints and tables.** Most composed-model and baseline tests check for a
specific local checkpoint or a directory under `outputs/` and skip with a reason such as
"Measured perf tables are machine-local." when it is absent. `outputs/` is gitignored, so a fresh
clone has none of it. Typical gates:

| Gate | Representative test files |
|---|---|
| `outputs/profile/<model>/kernel_tp1/kernel` | `test_kernel_baseline_regression.py`, `test_kernel_op_coverage.py`, `test_rollout_composed.py` |
| `outputs/profile/<model>/kernel_tp2/kernel` and the TP=2 baseline | `test_collective_baseline_regression.py`, `test_rollout_composed_tp2.py`, `test_training_composed_accuracy.py` |
| `outputs/profile/<model>/vllm_tp1/profiles/vllm.jsonl` | `test_kernel_baseline_regression.py` |
| A local HuggingFace checkpoint | `test_rollout_composed.py`, `test_hf_model.py`, `test_training_composed_accuracy.py` |
| `srl._engine_cpp` | every `test_engine_cpp_*` and the C++ cases in `test_forecast_drain.py` |
| CUDA available | `test_kernel_attention_vllm.py`, `test_kernel_timing.py`, and the CUDA-marked cases in `test_kernel_gemm.py` |
| FlashInfer installed | `test_kernel_attention_flashinfer.py` |

These tests are local instruments. They are useful precisely where the data exists, and on a
fresh clone they say so instead of passing vacuously or failing for a missing file.

---

## Pinned figures

`tests/test_kernel_baseline_regression.py`, `tests/test_collective_baseline_regression.py`,
`test_training_composed_accuracy.py` and `test_rollout_composed_tp2.py` pin accuracy figures with
explicit tolerances. They exist to catch a change to the pricing pipeline, not to measure accuracy
again. The figures and their counts are on {doc}`../reference/validation`.

A re-sweep invalidates them. `latency_s` is wall-clock time, so two runs of identical code move a
holdout MRE in the third decimal and can exceed a tight tolerance with no code change. After
re-running `srl-profile kernels run`:

1. Back up the old tables if the old figures are still wanted for comparison.
2. Re-run `srl-profile kernels holdout --fraction 1.0` and, where a baseline exists,
   `srl-profile kernels validate`.
3. Update the expected values in the test that pins them. The comments in those tests record why
   each value is what it is, so read them before overwriting a number.

A tolerance that has to widen is a signal. The baseline tests use a tight band for compute
families and a wider one for tables that legitimately moved after a corrected re-sweep, and the
comments say which is which.

---

## Import-boundary tests

Two tests protect the cost path from the GPU stack. Both work by importing a module in a fresh
subprocess, because in-process `sys.modules` already holds `torch` from whichever test ran first.

`tests/test_kernel_package_imports.py`
: Discovers every module under `srl/profiler/kernel/sweep/` and `grid/`, plus `config.py`,
  `utils/model_dims.py` and `utils/validators.py`, and imports each without `torch` or vLLM
  appearing in `sys.modules`. It also imports the package root, which must stay cheap because it
  re-exports `KernelProfileConfig`. A new grid or sweep is covered the moment it is added to the
  directory.

`tests/test_kernel_ops_base.py`
: Probes `srl.cost.kernel.ops.base` and the registry in a fresh interpreter, and checks that
  importing only `ops.base` populates `OP_REGISTRY`.

`tests/test_kernel_op_coverage.py`
: Compares `OP_REGISTRY` against the modules under `ops/`, in a fresh interpreter and in process. A
  module that exists but is missing from `ops/__init__.py` fails here instead of failing at runtime
  in a fresh process.

These tests exist because a torch import on the cost path does not produce a wrong number. It
produces a slow start, which nobody notices until they profile the simulator.

---

## Test doubles

`tests/fixtures/` holds small records used instead of a real run:

- `hf/` holds two `config.json` and safetensors index pairs, for the model descriptor and shape
  tests.
- `vllm/` holds two scheduler run files, for the vLLM log parser.
- `megatron_bridge/` holds a `ConfigContainer`, a metrics sidecar, and success and failure status
  documents, including an OOM marker.
- `fsdp/` holds one metrics sidecar.

A test that can use a fixture uses one. A test whose subject is the fitted accuracy of real data
cannot, and skips when that data is absent.
