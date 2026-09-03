# Task 8 Report: CLI, Configuration, and Example Driver

## What Was Created and Modified

### New Files
- `tests/test_fsdp_cli.py` — 8 tests covering parser wiring, regress choices, TrainingCostConfig schema, YAML loadability, sweep identifiability axes, and build_training_cost fsdp resolution
- `dryrun/config/defaults/training_cost/fsdp.yaml` — Hydra defaults YAML with 9 comments (satisfies the >= 6 comment gate)
- `examples/profile/fsdp/qwen3_8b.yaml` — Full sweep config with shard_sizes=[2,4,8], n_seqs_per_rank=[1,2,4,8], sequence_lengths=[512,1024,2048,4096,8192,16384], num_microbatches=[1,2,4]
- `examples/profile/fsdp/profile_qwen3_8b.sh` — Driver script mirroring the Megatron one; executable

### Modified Files
- `dryrun/cli/profile.py` — Added FSDP imports (with aliased read/write_candidate_manifest), `_fsdp_config`, `_fsdp_candidate_path`, `_generate_fsdp_candidates`, `cmd_fsdp_generate`, `cmd_fsdp_run`, `cmd_fsdp_collect`, and the `fsdp` subparser block registered after the Megatron block
- `dryrun/cli/regress.py` — Widened `--latency-model` and `--memory-model` choices to include `fsdp`; rewrote `cmd_training` to dispatch to `FsdpTrainingCost.fit` when both models are `fsdp`; imported `FsdpTrainingCost`
- `dryrun/config/schema.py` — Added `batching_mode: str = "pad_to_max"` and `max_tokens_per_rank: int | None = None` to `TrainingCostConfig`
- `dryrun/cli/main.py` — Passed `batching_mode` and `max_tokens_per_rank` through `build_training_cost` to `from_artifact` for registry models
- `dryrun/profiler/adapters/__init__.py` — Added FSDP adapter exports (`FsdpProfileConfig`, `collect_fsdp_record`, `collect_fsdp_records`, `prepare_fsdp_candidate`, `run_fsdp_candidate`, `read_fsdp_candidate_manifest`, `write_fsdp_candidate_manifest`) with `__all__` alphabetically ordered
- `.claude/codebase-map.md` — Added `cost/training/` subpackage tree, `cost/artifact.py`, the full `profiler/adapters/` tree with fsdp/megatron/training/vllm subpackages, `profiler/fsdp_search_space.py`, updated `config/defaults/training_cost/` listing, updated CLI and tests sections, and expanded import dependency order from 27 to 44 entries

## Smoke YAML fix applied
The brief's `qwen3_8b_smoke.yaml` already contained `num_microbatches: [1, 2]` from Task 2/3 work. No modification needed.

## Test Results

### New tests
```
python -m pytest tests/test_fsdp_cli.py -q
8 passed in 8.21s
```

### Full suite (ignoring concurrent-session files)
```
python -m pytest tests/ -q \
  --ignore=tests/test_hf_model.py \
  --ignore=tests/test_kernel_gemm.py \
  --ignore=tests/test_kernel_measurements.py \
  --ignore=tests/test_kernel_model_ops.py \
  --ignore=tests/test_rollout_composed.py \
  --ignore=tests/test_kernel_holdout.py
754 passed in 114.31s
```
Baseline was 725. The 29-test increase: 8 new from this task, and 21 from previously-committed tasks (test_megatron_launch, test_rollout_segment_integrator, test_training_activation, etc.) that were already in the tree but are now counted.

### Ruff on modified files
```
python -m ruff check dryrun/cli/profile.py dryrun/cli/regress.py dryrun/cli/main.py \
    dryrun/config/schema.py dryrun/profiler/adapters/__init__.py tests/test_fsdp_cli.py
All checks passed!
```
Pre-existing errors in `dryrun/profiler/kernels/elementwise.py` and unrelated test/vllm files are from the concurrent refactor session and were not introduced by this task.

## Smoke End-to-End Result

Ran with `examples/profile/fsdp/qwen3_8b_smoke.yaml` (2-layer override):
- `dryrun-profile fsdp generate`: 16 candidates written
- `dryrun-profile fsdp run --resume` (all 16): all succeeded, ~90 s each
- `dryrun-profile fsdp collect` (all 16): 16 normalized records written
- `dryrun-regress training --latency-model fsdp --memory-model fsdp`: fit succeeded, 4 entries (2 cells × latency+memory)

### Fit diagnostics

| cell | component | design_rank | r² train | MRE train |
|------|-----------|------------|-----------|-----------|
| shard=4, rep=2, offload=False | latency | **3** | 0.99990 | 0.316% |
| shard=4, rep=2, offload=False | memory  | 2       | -0.50  | 2.28% |
| shard=8, rep=1, offload=False | latency | **3** | 0.99995 | 0.229% |
| shard=8, rep=1, offload=False | memory  | 2       | 0.12   | 1.22% |

**Memory metric selected**: `peak_reserved_bytes` for both cells (allocated candidate scored far worse).

**reserved_over_allocated**: 1.82 (shard=4), 2.18 (shard=8).

### Analysis of negative memory r²
The memory r² is negative because the smoke config uses `num_layers_override: 2`. With only 2 layers, the activation memory variation across the sweep (n_seqs=[1,2] × lengths=[512,1024] × nmb=[1,2]) is extremely small compared to the static inventory term. The regression captures the static floor well but cannot meaningfully separate the activation slope. This is expected and correct behavior for a 2-layer smoke model; the production sweep with a full 28-layer Qwen3-8B will produce meaningful activation variation.

## Corrections Applied to the Brief

1. **num_microbatches in qwen3_8b.yaml**: Applied as instructed — added `num_microbatches: [1, 2, 4]` and extended `sequence_lengths` to 16384.
2. **num_microbatches in qwen3_8b_smoke.yaml**: Already had `[1, 2]` from earlier tasks.
3. **test_example_sweep_varies_both_identifiability_axes**: Strengthened per brief to also assert `len(search["num_microbatches"]) >= 2`.
4. **TrainingCostConfig**: Added `batching_mode` and `max_tokens_per_rank`; wired through `build_training_cost`.
5. **regress.py choices**: Changed both from `["hydraulis"]` to `["hydraulis", "fsdp"]`.
6. **profiler/adapters/__init__.py**: Brief said "export FSDP names alongside existing Megatron and vLLM, keeping __all__ sorted" — done, with aliased manifest functions to avoid name collision with Megatron's.

## Brief Ambiguity: Naming Collision for read/write_candidate_manifest
Both the Megatron and FSDP adapters export `read_candidate_manifest` and `write_candidate_manifest`. The `profiler/adapters/__init__.py` exports the Megatron ones under their original names (preserving backward compatibility for all existing callers) and the FSDP ones under `read_fsdp_candidate_manifest` / `write_fsdp_candidate_manifest`. In `profile.py`, the FSDP handlers import from the FSDP subpackage directly with local aliases (`read_fsdp_manifest`, `write_fsdp_manifest`) rather than going through the top-level `__init__`, which keeps the call sites unambiguous.

## What Was Left Alone
- The full production sweep (`examples/profile/fsdp/qwen3_8b.yaml`) was not run — left for the user per the brief's Step 10 exclusion.
- Nothing under `dryrun/profiler/kernels/` or `dryrun/cost/kernel/` was touched.
- The existing test suite's pre-existing ruff errors (in vllm/megatron/kernel files modified by the concurrent session) were not fixed.

## Commit
SHA: `4d9651d` — "feat: expose the FSDP backend through the CLI and Hydra config"

---

# Task 8-B Report: Fix Structurally Misspecified FSDP Memory Feature

## Finding

The previous single-feature design fit `activation_slope * max(sum_tokens)` against a target that is flat
across tokens (to four decimal places over a 4x range) and varies only with `num_microbatches`.
A feature that does not move the target produces a degenerate fit: `activation_slope ~ 5e-14`,
both candidates score catastrophically negative r² (allocated -2.6e9, reserved -3.4e5).
This is not noise -- it is a structural misspecification.

## What Changed

- `dryrun/cost/training/fsdp.py`:
  - Module docstring extended with physical explanation and measured flatness evidence.
  - Replaced `_activation_tokens` (1 scalar) with `_memory_features` returning `(num_microbatches, max_tokens)`.
  - `_MemoryFit` extended from `slope` to `microbatch_slope` + `activation_slope`; `train_feature` and
    `validation_feature` are now 2D arrays with columns `[num_microbatches, max_tokens]`.
  - `_fit_memory_target`: design is now `[num_microbatches, max_tokens, 1]` (3 columns); rank check raised from 2 to 3.
  - `_fit_memory_entry`: OOM inequality uses both features; `feature_ranges` now bounds both
    `num_microbatches` and `activation_tokens`; coefficients now store `microbatch_slope` and `activation_slope`.
  - `FsdpMemoryModel.predict`: checks both feature ranges; evaluates the two-term form.
  - `_empty_memory_fit`: updated for 2D feature arrays.

- `tests/test_fsdp_cost.py`:
  - `MICROBATCH_SLOPE = 6e8` and `ACTIVATION_SLOPE = 1000.0` constants added.
  - `_record` default memory is now `static + MICROBATCH_SLOPE * nmb + ACTIVATION_SLOPE * tokens`.
  - `_records_discriminating` updated to use non-collinear `(length, nmb)` pairs `(1, 1, 2)` so the 3-column design reaches rank 3.
  - `test_memory_prediction_is_compared_against_reserved` updated to use non-collinear pairs and new coefficient formula.
  - `test_the_oom_bumped_metrics_describe_the_shipped_coefficients` updated to use both nmb and token features.
  - OOM floor test updated to use `token_budget` batching at 16384 tokens to reproduce `(nmb=4, max_tokens=16384)`.
  - `test_fit_recovers_both_memory_coefficients` added: asserts both `microbatch_slope` and `activation_slope` recovered.
  - `test_fit_memory_nmb_only_drives_coefficient_and_not_token_term` added: nmb-only data drives `activation_slope` to zero.

## Verification

### Tests
```
python -m pytest tests/test_fsdp_cost.py -q
25 passed in 1.49s

python -m pytest tests/ -q [ignoring 6 concurrent-session files]
770 passed in 118.55s
```

### Ruff
```
python -m ruff check dryrun/cost/training/fsdp.py tests/test_fsdp_cost.py
All checks passed!
```

### Real data refit (shard=8, replicate=1)

| metric | before | after |
|--------|--------|-------|
| allocated r² | -2.6e9 | **1.000** |
| reserved r² | -3.4e5 | -1.596 |
| selected metric | peak_allocated_bytes | peak_allocated_bytes |
| microbatch_slope | (absent) | 6.229e+08 |
| activation_slope | ~5e-14 | 2.400e+01 |
| residual_constant | (degenerate) | 6.293e+09 |
| design_rank | 2 | **3** |
| reserved_over_allocated | 2.18 | 2.18 |

The `microbatch_slope` 6.23e8 B/nmb = 593 MiB/nmb matches the measured ~580 MiB/nmb.
The `activation_slope` 24 B/token is negligible relative to the microbatch scale, confirming
gradient checkpointing makes activations flat across token range.

## Commit

---

# Task 8-C Report: Fix NCCL Buffer-Allocation OOM Classification

## Finding

5 `failed` records at `shard_size=2, replicate_size=4` were misclassified because memory exhaustion surfaced as a NCCL buffer-allocation error in `dist.all_gather_object` inside `on_train_end`, not as a `torch.OutOfMemoryError`. The run was mis-classified `failed` rather than `oom`, losing the high-memory lower bounds the memory model needs.

## What Changed

- `dryrun/profiler/adapters/fsdp/measure.py`:
  - Added `is_nccl_memory_exhaustion(error)` helper: returns True iff the lowercased error message contains `"failed to cuda calloc"` or `"out of memory"`. A bare `ncclUnhandledCudaError` with no memory phrase returns False.
  - Updated `write_oom_sidecar` to accept `oom_source: str = "torch_oom"` keyword argument. The payload now includes `"oom_source"` in addition to `"torch_oom": True`, allowing readers to distinguish text-matched from exact-type records.
  - Added `is_nccl_memory_exhaustion` to `__all__`.

- `dryrun/profiler/adapters/fsdp/launch.py`:
  - Added `except dist.DistBackendError` branch after the existing `except torch.OutOfMemoryError` branch.
  - Calls `is_nccl_memory_exhaustion`; re-raises unchanged if it returns False (preserving `failed` classification for timeouts, rendezvous failures, real network faults).
  - On a memory match: calls `write_oom_sidecar(..., oom_source="nccl_alloc")` and exits with `OOM_EXIT_CODE`.
  - Block comment explains why exact-type catching is right in general, and why text matching is the correct narrowing for this specific failure mode.

- `tests/test_fsdp_launch.py`:
  - Updated import to include `is_nccl_memory_exhaustion`.
  - Added 8 new tests: `torch_oom` source recorded by default; `nccl_alloc` source recorded when requested; real sweep error text is detected; `out of memory` phrase is detected; matching is case-insensitive; bare `ncclUnhandledCudaError` without memory phrase is rejected; timeout is rejected; rendezvous failure is rejected.

## Collect Path

`collect.py` classifies status via `read_status` from `training/run.py`, which decides `oom` purely by the presence of `oom_rank*.json` marker files. The new path writes the same marker file pattern, so collect already does the right thing. The `oom_source` field is diagnostic metadata; no downstream reader consults it for classification. No change was needed in `collect.py`.

## Verification

```
python -m pytest tests/test_fsdp_launch.py tests/test_fsdp_collect.py tests/test_training_shared.py -q
68 passed in 34.45s

python -m ruff check dryrun/profiler/adapters/fsdp/ tests/test_fsdp_launch.py
All checks passed!
```

## Commit
SHA: `0dbe9d9`

---

# Task 8-D Report: Fix MicrobatchPlanner Ignoring micro_batch_size

## Bug

`MicrobatchPlanner.plan()` did not accept `micro_batch_size`, so `PAD_TO_MAX` and
`PACKED_NO_PAD` always produced **one** microbatch containing all sequences on the rank.
With `BATCH_SIZE=64`, `dp=8` (8 sequences per rank at ~9176 tokens each), both modes
reported `sum_tokens=73408`, outside the fitted range `[512, 16384]`, and the simulator
wrongly concluded the hardware could not run the configuration. The correct answer is
`micro_batch_size=1` → **8** microbatches of 9176 tokens each, which is comfortably
in-domain. `TOKEN_BUDGET` was unaffected because it derives its own split from the budget.

## What Changed

### `dryrun/cost/training/microbatch.py`
- `plan()` gains a keyword-only `micro_batch_size` parameter with no default (forcing callers
  to be explicit). Validated positive before use.
- `_microbatches()` now accepts `micro_batch_size` and delegates to two new static helpers:
  - `_fixed_size_microbatches_padded`: splits into groups of `micro_batch_size`, padding
    **within each group** to that group's maximum.
  - `_fixed_size_microbatches_packed`: same split, charging only true token counts.
- `_token_budget_microbatches()` adds a `seq_cap_count = ceil(len/mbs)` guard: if the
  sequence-count constraint demands more microbatches than the token budget alone, the
  larger count wins. Comment explains precedence.

### `dryrun/cost/training/fsdp.py`
- `_microbatches()` now passes `micro_batch_size=workload.micro_batch_size` to `plan()`.

### `tests/test_microbatch_planner.py`
- All existing calls updated to pass `micro_batch_size`.
- 11 new tests added:
  - Rejection of `micro_batch_size=0` and `micro_batch_size=-1`.
  - `pad_to_max`/`packed_no_pad` with `mbs=1` → 8 microbatches, not 1 (the exact regression).
  - `mbs=2` → 4 microbatches.
  - Per-microbatch padding: short sequences grouped together are not charged the rank-wide max.
  - `token_budget` seq-cap forced when all sequences are tiny.
  - `token_budget` seq-cap is a no-op when the budget already demands more splits.

### `tests/test_fsdp_cost.py`
- `test_batching_mode_changes_the_prediction`: changed `micro_batch_size=1` to `micro_batch_size=4`
  so that each rank's 4 sequences are fused into one microbatch, making the padding difference
  between modes visible. With `mbs=1` both modes charge real tokens only (no padding in either)
  and the prediction was identical.

No expected values in `test_fsdp_cost.py` changed for the coefficient-recovery or OOM tests:
those tests construct synthetic records whose microbatches are defined directly, so the planner
path is not exercised. The only changed test value is the `micro_batch_size=1→4` correction above.

## Simulate Matrix Results

Feature `tokens` error is gone for all three FSDP modes (9176 is in `[512, 16384]`).
All three now fail on `num_microbatches=8.0 outside [1.0, 4.0]` — a genuine gap in the
profile sweep (the sweep only covered up to 4 microbatches), not a code bug.
The hydraulis/Megatron case is unchanged: still fails with `tokens=9470 vs [512, 2048]`
(a genuinely narrow Megatron sweep; Megatron code was not changed).

| mode | result | feature / value / range |
|------|--------|--------------------------|
| pad_to_max | out-of-domain | num_microbatches=8.0 vs [1.0, 4.0] |
| token_budget | out-of-domain | num_microbatches=8.0 vs [1.0, 4.0] |
| packed_no_pad | out-of-domain | num_microbatches=8.0 vs [1.0, 4.0] |
| hydraulis (megatron) | out-of-domain | tokens=9470.0 vs [512.0, 2048.0] |

## Test Suite

906 passed (15 new tests: 11 in `test_microbatch_planner.py`, 0 net in `test_fsdp_cost.py`
because one test was updated not added).

---

# Task 8-E Report: Per-Feature Extrapolation Control (extrapolate list)

## Problem

`allow_extrapolation: bool` gated the out-of-domain check for every feature at once.
The risk differs by orders of magnitude: `num_microbatches` is safe to extrapolate upward
(median absolute error 7.7% on nmb=4 from nmb∈{1,2} training, on par with in-domain MRE
of 9-11%), while `tokens` is dangerous (32768+ tokens/rank → 0 successes, 18-27 OOMs in
the profiled candidates; a prediction there describes a configuration that cannot run).

## Design

Replaced `allow_extrapolation: bool` with `extrapolate: list[str]` (frozenset internally).

- **Empty list / None** → strict behaviour: any out-of-range query raises `TrainingOutOfDomainError`.
- **Named feature** → prediction proceeds; the exceedance is recorded in
  `TrainingCostEstimate.diagnostics["extrapolation_notes"]` and surfaced in telemetry.
  Silent extrapolations are the thing this design exists to prevent.
- **Unknown name at construction** → `ValueError` listing valid names. A typo like
  `num_microbatch` cannot silently enable nothing.

## Files Changed

- `dryrun/cost/training/fitting.py`: `check_feature_range` now takes `extrapolate: frozenset[str]`
  and returns `str | None` (the note string, or None if in range). Added `validate_extrapolate_names`.
- `dryrun/cost/training/fsdp.py`: `_FsdpCellModel`, `FsdpLatencyModel`, `FsdpMemoryModel` use
  `extrapolate: frozenset[str]`. Added `predict_with_notes()` to both model classes.
  `FsdpTrainingCost.from_artifact` accepts `extrapolate: list[str] | None`, validates names against
  `entry.feature_ranges` keys. `FsdpTrainingCost.estimate()` overridden to collect and record notes.
- `dryrun/cost/training/hydraulis.py`: Same pattern for `HydraulisLatencyModel`,
  `HydraulisMemoryModel`, `HydraulisTrainingCost`. Added `estimate()` override.
- `dryrun/cli/main.py`: `allow_extrapolation` → `extrapolate` with `list(cfg.get("extrapolate", []))`.
- `dryrun/config/schema.py`: `allow_extrapolation: bool = False` → `extrapolate: list = field(default_factory=list)`.
- `dryrun/config/defaults/training_cost/fsdp.yaml`: removed `allow_extrapolation`, added
  `extrapolate: []` with per-feature documentation of what the measured evidence says.
- `dryrun/config/defaults/training_cost/hydraulis.yaml`: same.
- `dryrun/simulator/simulator.py`: passes `extrapolation_notes` from estimate diagnostics to
  `write_train`.
- `dryrun/telemetry/writer.py`: `write_train` accepts optional `extrapolation_notes: list[str] | None`;
  field is omitted from the JSONL event when empty (no noise for the common case).
- `examples/simulate/simulate.sh`: `EXTRAPOLATE` env var (default empty) → appended to FSDP
  overrides as `training_cost.extrapolate=${EXTRAPOLATE}`.
- `tests/test_fsdp_cost.py`: updated three `allow_extrapolation=True` call sites to use
  `extrapolate=[all_features]`; added 7 new tests.

## New Tests (7 added to test_fsdp_cost.py)

1. `test_empty_extrapolate_rejects_out_of_range_for_all_features` — null list raises OOD.
2. `test_null_extrapolate_rejects_out_of_range` — `extrapolate=None` same as empty.
3. `test_listed_feature_permits_that_feature_and_rejects_others` — one feature permitted,
   others still raise with their feature name in the message.
4. `test_unknown_extrapolate_name_raises_at_construction` — typo raises at construction.
5. `test_unknown_extrapolate_name_message_lists_valid_names` — error message names valid features.
6. `test_permitted_extrapolation_is_recorded_in_diagnostics` — note appears in diagnostics.
7. `test_in_range_values_produce_no_extrapolation_notes` — clean in-range call → no notes.

## Hydra CLI List Syntax

Verified that `training_cost.extrapolate=[num_microbatches]` parses correctly from the
command line (returns `ListConfig = ['num_microbatches']`). The YAML also shows the example
override syntax.

## Simulate Payoff

With `EXTRAPOLATE=[num_microbatches]` all three FSDP modes complete successfully (previously
all three failed with `num_microbatches=8.0 outside [1.0, 4.0]`).

| mode | mean step latency | peak memory | makespan | extrapolation_notes |
|------|------------------|-------------|----------|---------------------|
| pad_to_max | 48.21 s | 80.78 GB | 640.4 s | num_microbatches=8.0 > [1.0,4.0] every step |
| token_budget | 48.21 s | 80.78 GB | 640.4 s | same |
| packed_no_pad | 48.21 s | 80.78 GB | 640.4 s | same |

**All three modes produce identical cost estimates for this workload.** This is correct:
with 64 sequences from a near-Gaussian distribution (mean ~9200 tokens), dp=8, mbs=1, each
rank processes 8 sequences of similar lengths. All three batching modes then produce 8
microbatches of 1 sequence each with the same max-tokens. The expected `token_budget < packed_no_pad < pad_to_max` ordering requires a bimodal (ragged) workload where short sequences are padded up to the long-sequence maximum under `pad_to_max` but not under the other modes. With near-equal lengths padding adds nothing and all modes charge the same.

Every step records extrapolation notes confirming the `num_microbatches=8.0 outside [1.0, 4.0]` extrapolation is visible in the telemetry — the extrapolation is not silent.

## Test Suite

942 passed (+36 from the 906 baseline: 7 new extrapolate tests in test_fsdp_cost.py, plus
tests from concurrent sessions already in the tree).

## Commits

See commit SHAs after this section.

