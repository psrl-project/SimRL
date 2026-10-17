# FSDP2/HSDP Training Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FSDP2/HSDP as a third profiling and cost-model backend beside vLLM (rollout) and Megatron (training), end to end from profile through cost model to simulation.

**Architecture:** Profiling runs only uniform sequence shapes through a self-built `torch.distributed` + transformers launcher (no verl dependency), so every measured record satisfies the existing `TrainingPayload` invariants. Sequence packing lives entirely on the simulate side as a workload-to-microbatch-plan function with three selectable modes. Genuinely shared subprocess and sidecar handling is extracted from the Megatron adapter into `adapters/training/` and consumed by both.

**Tech Stack:** Python 3.12, PyTorch 2.11.0+cu128 (`torch.distributed.fsdp.fully_shard`, `init_device_mesh`, `MixedPrecisionPolicy`, `CPUOffloadPolicy`), transformers 5.5.3, flash-attn 2.8.3.post1, numpy, scipy, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-26-fsdp-training-backend-design.md`. Read it before starting.
- Coding style: `.claude/coding-style.md`. §1 is critical: **identifier strings never get a trailing period or case change**; human-readable messages are capitalized and end in `.`. Line length 119 (`[tool.ruff] line-length = 119`).
- Project contract: `AGENTS.md`. **No backward compatibility layers.** Remove obsolete paths rather than adding fallbacks or migrations.
- Activate the environment before running anything: `source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh`.
- Test baseline before any change: **602 passed**. It must never regress. Run `python -m pytest tests/ -q` to confirm.
- Comments explain **why**, never what. Every non-obvious constant or guard needs its rationale, matching the density of the surrounding Megatron adapter code.
- Do not use `Date.now()`-style nondeterminism in fixtures; `candidate_id` values must be stable across runs.
- `schedule` identifier value for FSDP is exactly `"fsdp"` (lowercase, no period).
- `sp > 1` (Ulysses sequence parallelism), FSDP1, tensor parallelism, and pipeline parallelism under FSDP are **out of scope** and must fail fast with an explicit not-implemented message.
- Never scan a log for the phrase "out of memory". OOM is decided **only** by per-rank marker files. A false positive can only raise the fitted memory floor, silently making the memory model more conservative.

## File Structure

**Created:**

| path | responsibility |
|---|---|
| `srl/profiler/adapters/training/__init__.py` | public exports of the shared training-adapter layer |
| `srl/profiler/adapters/training/run.py` | subprocess orchestration, OOM markers, status classification |
| `srl/profiler/adapters/training/sidecar.py` | sidecar reading, warmup slicing, peak aggregation, candidate-dir iteration |
| `srl/profiler/fsdp_search_space.py` | `FsdpCandidate`, `enumerate_fsdp_candidates` |
| `srl/profiler/adapters/fsdp/__init__.py` | public exports; deliberately excludes torch-importing modules |
| `srl/profiler/adapters/fsdp/config.py` | `FsdpProfileConfig` and YAML deserialization |
| `srl/profiler/adapters/fsdp/prepare.py` | launch spec, manifest, torchrun command |
| `srl/profiler/adapters/fsdp/launch.py` | torchrun child: mesh, `fully_shard`, train loop |
| `srl/profiler/adapters/fsdp/measure.py` | step times, per-rank peaks, comm/compute split, OOM sidecar |
| `srl/profiler/adapters/fsdp/execute.py` | thin delegation to the shared runner |
| `srl/profiler/adapters/fsdp/collect.py` | raw directories to `ProfileRecord` |
| `srl/cost/training/microbatch.py` | `BatchingMode`, `MicrobatchPlanner` |
| `srl/cost/training/fsdp_inventory.py` | FSDP static memory layout (`16p/ws`) |
| `srl/cost/training/fsdp.py` | `FsdpLatencyModel`, `FsdpMemoryModel`, `FsdpTrainingCost` |
| `srl/config/defaults/training_cost/fsdp.yaml` | Hydra defaults for the FSDP cost model |
| `examples/profile/fsdp/qwen3_8b.yaml` | full sweep config |
| `examples/profile/fsdp/qwen3_8b_smoke.yaml` | two-layer smoke config |
| `examples/profile/fsdp/profile_qwen3_8b.sh` | generate/run/collect driver |

**Modified:**

| path | change |
|---|---|
| `srl/profiler/schema.py:374` | accept `schedule == "fsdp"` in `TrainingPayload` |
| `srl/cost/training/base.py:66` | accept `schedule == "fsdp"` in `TrainingWorkload` |
| `srl/profiler/adapters/megatron/execute.py` | delegate to `training/run.py` |
| `srl/profiler/adapters/megatron/collect.py` | delegate to `training/sidecar.py` |
| `srl/profiler/adapters/__init__.py` | export the FSDP adapter |
| `srl/cost/training/__init__.py` | export FSDP models; register `"fsdp"` |
| `srl/cli/profile.py` | `fsdp` backend subcommands |
| `srl/cli/regress.py:93-94` | add `"fsdp"` to both model choices |
| `srl/config/schema.py` | `batching_mode`, `max_tokens_per_rank` on `TrainingCostConfig` |
| `srl/cli/main.py` | wire the new config fields into `build_training_cost` |
| `.claude/codebase-map.md` | record the new packages |

**Tests created:** `tests/test_training_shared.py`, `tests/test_fsdp_search_space.py`, `tests/test_fsdp_launch.py`, `tests/test_fsdp_collect.py`, `tests/test_microbatch_planner.py`, `tests/test_fsdp_inventory.py`, `tests/test_fsdp_cost.py`, plus fixtures under `tests/fixtures/fsdp/`.

---

### Task 1: Extract the shared training-adapter layer

Megatron's `execute.py` and `collect.py` contain roughly 135 lines that are genuinely backend-agnostic. Extract them first so the FSDP adapter consumes them rather than duplicating them. Megatron's behaviour must not change: its tests are the regression gate.

**Files:**
- Create: `srl/profiler/adapters/training/__init__.py`
- Create: `srl/profiler/adapters/training/run.py`
- Create: `srl/profiler/adapters/training/sidecar.py`
- Modify: `srl/profiler/adapters/megatron/execute.py` (replace body with delegation)
- Modify: `srl/profiler/adapters/megatron/collect.py:25-33` and `:78-101` (replace helpers with imports)
- Test: `tests/test_training_shared.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `OOM_EXIT_CODE: int` (value `42`)
  - `oom_marker_paths(run_dir: Path) -> list[Path]`
  - `read_status(run_dir: Path) -> tuple[str, float | None, str | None]`
  - `run_candidate(*, run_dir: Path, command: list[str], environment: dict[str, str], dry_run: bool = False, resume: bool = False) -> Path`
  - `load_sidecar_metrics(run_dir: Path) -> dict[str, Any] | None`
  - `slice_warmup(step_times: tuple[float, ...], warmup_steps: int) -> tuple[float, ...]`
  - `rank_memory_peaks(rank_memory: tuple[RankMemory, ...]) -> tuple[int | None, int | None]`
  - `iter_candidate_dirs(raw_root: Path, candidate_ids: list[str], *, allow_incomplete: bool) -> Iterator[tuple[str, Path]]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_training_shared.py`:

```python
"""The backend-agnostic training-adapter layer shared by Megatron and FSDP."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from srl.profiler.adapters.training import (
    OOM_EXIT_CODE,
    iter_candidate_dirs,
    load_sidecar_metrics,
    oom_marker_paths,
    rank_memory_peaks,
    read_status,
    run_candidate,
    slice_warmup,
)
from srl.profiler.schema import RankMemory


def test_oom_exit_code_is_forty_two():
    assert OOM_EXIT_CODE == 42


def test_read_status_reports_missing_status_document(tmp_path: Path):
    status, duration, error = read_status(tmp_path)
    assert status == "failed"
    assert duration is None
    assert "status.json" in error


def test_read_status_returns_success(tmp_path: Path):
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "success", "duration_s": 12.5}), encoding="utf-8"
    )
    assert read_status(tmp_path) == ("success", 12.5, None)


def test_read_status_prefers_oom_markers_over_generic_failure(tmp_path: Path):
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "failed", "duration_s": 3.0}), encoding="utf-8"
    )
    (tmp_path / "oom_rank2.json").write_text(json.dumps({"rank": 2}), encoding="utf-8")
    status, duration, error = read_status(tmp_path)
    assert status == "oom"
    assert duration == 3.0
    assert "rank(s) 2" in error


def test_read_status_never_infers_oom_from_a_log_phrase(tmp_path: Path):
    """A log mentioning the phrase must stay `failed`; only markers are authoritative.

    A false OOM adds an inequality constraint that can only raise the fitted memory
    floor, so it silently makes the memory model more conservative.
    """
    (tmp_path / "status.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    (tmp_path / "run.log").write_text("CUDA out of memory. Tried to allocate", encoding="utf-8")
    status, _, _ = read_status(tmp_path)
    assert status == "failed"


def test_oom_marker_paths_are_sorted(tmp_path: Path):
    for rank in (3, 1, 0):
        (tmp_path / f"oom_rank{rank}.json").write_text("{}", encoding="utf-8")
    assert [p.name for p in oom_marker_paths(tmp_path)] == [
        "oom_rank0.json",
        "oom_rank1.json",
        "oom_rank3.json",
    ]


def test_run_candidate_short_circuits_on_resume(tmp_path: Path):
    (tmp_path / "status.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    # `false` would exit non-zero, so reaching it would raise.
    result = run_candidate(
        run_dir=tmp_path, command=["false"], environment={}, resume=True
    )
    assert result == tmp_path


def test_run_candidate_dry_run_does_not_execute(tmp_path: Path):
    run_candidate(run_dir=tmp_path, command=["false"], environment={}, dry_run=True)
    assert not (tmp_path / "status.json").exists()


def test_run_candidate_records_success_and_clears_stale_markers(tmp_path: Path):
    stale = tmp_path / "oom_rank0.json"
    stale.write_text(json.dumps({"rank": 0}), encoding="utf-8")
    run_candidate(run_dir=tmp_path, command=["true"], environment={})
    assert not stale.exists()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "success"
    assert status["returncode"] == 0
    assert status["duration_s"] >= 0


def test_run_candidate_raises_and_preserves_status_on_failure(tmp_path: Path):
    import subprocess

    with pytest.raises(subprocess.CalledProcessError):
        run_candidate(run_dir=tmp_path, command=["false"], environment={})
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"


def test_load_sidecar_metrics_returns_none_when_absent(tmp_path: Path):
    assert load_sidecar_metrics(tmp_path) is None


def test_load_sidecar_metrics_rejects_a_non_object_document(tmp_path: Path):
    from srl.profiler.schema import ProfileValidationError

    (tmp_path / "srl_metrics.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ProfileValidationError):
        load_sidecar_metrics(tmp_path)


def test_slice_warmup_drops_leading_steps():
    assert slice_warmup((1.0, 2.0, 3.0, 4.0), 2) == (3.0, 4.0)


def test_slice_warmup_keeps_everything_when_it_would_empty_the_series():
    """Dropping every sample would turn a measured candidate into an unmeasured one."""
    assert slice_warmup((1.0, 2.0), 2) == (1.0, 2.0)
    assert slice_warmup((1.0, 2.0), 5) == (1.0, 2.0)


def test_rank_memory_peaks_takes_the_maximum_across_ranks():
    memory = (
        RankMemory(rank=0, allocated_bytes=10, reserved_bytes=20),
        RankMemory(rank=1, allocated_bytes=30, reserved_bytes=40),
    )
    assert rank_memory_peaks(memory) == (30, 40)


def test_rank_memory_peaks_of_nothing_is_none():
    assert rank_memory_peaks(()) == (None, None)


def test_iter_candidate_dirs_skips_missing_when_allowed(tmp_path: Path):
    present = tmp_path / "aaa"
    present.mkdir()
    (present / "manifest.json").write_text("{}", encoding="utf-8")
    found = list(iter_candidate_dirs(tmp_path, ["aaa", "bbb"], allow_incomplete=True))
    assert found == [("aaa", present)]


def test_iter_candidate_dirs_raises_on_missing_when_not_allowed(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        list(iter_candidate_dirs(tmp_path, ["bbb"], allow_incomplete=False))
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh
python -m pytest tests/test_training_shared.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'srl.profiler.adapters.training'`.

- [ ] **Step 3: Create `run.py`**

Create `srl/profiler/adapters/training/run.py`:

```python
"""Subprocess orchestration shared by every training backend.

Megatron and FSDP both launch one torchrun child per candidate, classify the
result, and record a status document. Only the command differs, so the command is
an argument rather than something this module knows how to build.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

# Exit code a launcher uses for a caught `torch.OutOfMemoryError`. torchrun does not
# forward child exit codes verbatim, so this is a hint only. The authoritative OOM
# signal is the per-rank marker file checked by `oom_marker_paths`.
OOM_EXIT_CODE = 42

_OOM_MARKER_GLOB = "oom_rank*.json"


def oom_marker_paths(run_dir: Path) -> list[Path]:
    """Return the per-rank OOM markers a launcher wrote, if any."""
    return sorted(run_dir.glob(_OOM_MARKER_GLOB))


def read_status(run_dir: Path) -> tuple[str, float | None, str | None]:
    """
    Classify one run from its status document and OOM markers.

    OOM is decided by the marker files a launcher writes on a caught
    `torch.OutOfMemoryError`, never by scanning the log. A log that merely mentions
    the phrase must stay `failed`: an OOM record adds an inequality constraint that
    can only raise the fitted memory floor, so a false positive silently makes the
    memory model more conservative.
    """
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return "failed", None, "status.json is missing."
    with status_path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    status = value.get("status")
    duration_s = value.get("duration_s")
    if status == "success":
        return "success", duration_s, None
    markers = oom_marker_paths(run_dir)
    if markers:
        ranks = ", ".join(str(json.loads(path.read_text())["rank"]) for path in markers)
        return "oom", duration_s, f"CUDA out of memory on rank(s) {ranks}."
    return "failed", duration_s, f"Process exited with status {status!r}."


def run_candidate(
    *,
    run_dir: Path,
    command: list[str],
    environment: dict[str, str],
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    """Execute one prepared candidate in an independent subprocess."""
    status_path = run_dir / "status.json"
    if resume and status_path.exists():
        with status_path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("status") == "success":
            return run_dir
    if dry_run:
        return run_dir

    # Stale markers from a previous attempt would misclassify this run.
    for marker in oom_marker_paths(run_dir):
        marker.unlink()

    merged = os.environ.copy()
    merged.update(environment)
    started = time.time()
    with (run_dir / "run.log").open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            command,
            check=False,
            env=merged,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode == 0:
        status = "success"
    elif oom_marker_paths(run_dir):
        status = "oom"
    else:
        status = "failed"
    status_payload = {
        "status": status,
        "returncode": result.returncode,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "duration_s": time.time() - started,
        "command": command,
    }
    status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return run_dir
```

- [ ] **Step 4: Create `sidecar.py`**

Create `srl/profiler/adapters/training/sidecar.py`:

```python
"""Sidecar reading and measurement normalization shared by training backends.

A launcher writes one `srl_metrics.json` per candidate and the collector turns it
into a payload. The warmup policy lives here, not in the launcher, so the launcher
stays a pure measurer and the retained window can change without re-profiling.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ...schema import ProfileValidationError, RankMemory

SIDECAR_NAME = "srl_metrics.json"
MANIFEST_NAME = "manifest.json"


def load_sidecar_metrics(run_dir: Path) -> dict[str, Any] | None:
    """Read one candidate's measurement sidecar, or None when it was never written."""
    path = run_dir / SIDECAR_NAME
    if not path.exists():
        return None
    import json

    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ProfileValidationError(f"Expected a JSON object in {path}.")
    return value


def slice_warmup(step_times: tuple[float, ...], warmup_steps: int) -> tuple[float, ...]:
    """
    Drop the leading warmup samples, unless doing so would leave nothing.

    Discarding every sample would turn a candidate that was measured into one that
    looks unmeasured, which the collector then rejects for a reason that has nothing
    to do with the run. Returning the full series keeps that failure visible as a
    dispersion diagnostic instead.
    """
    if warmup_steps <= 0 or len(step_times) <= warmup_steps:
        return step_times
    return step_times[warmup_steps:]


def rank_memory_peaks(
    rank_memory: tuple[RankMemory, ...],
) -> tuple[int | None, int | None]:
    """Return the worst rank's allocated and reserved peaks, or None when unmeasured."""
    if not rank_memory:
        return None, None
    allocated = max(item.allocated_bytes for item in rank_memory)
    reserved = max(item.reserved_bytes for item in rank_memory)
    return allocated, reserved


def iter_candidate_dirs(
    raw_root: Path,
    candidate_ids: list[str],
    *,
    allow_incomplete: bool,
) -> Iterator[tuple[str, Path]]:
    """
    Yield the candidate directories that carry an adapter manifest.

    A missing manifest means the candidate was never prepared. That is an error by
    default; `allow_incomplete` makes it a skip so a partially finished sweep can
    still be collected.
    """
    for candidate_id in candidate_ids:
        run_dir = raw_root / candidate_id
        if not (run_dir / MANIFEST_NAME).exists():
            if allow_incomplete:
                continue
            raise FileNotFoundError(f"Candidate output is missing: {run_dir}.")
        yield candidate_id, run_dir
```

- [ ] **Step 5: Create the package `__init__.py`**

Create `srl/profiler/adapters/training/__init__.py`:

```python
"""Backend-agnostic pieces of the training profiling adapters.

Megatron and FSDP differ in how they build a model and a launch command, but they
agree on how a candidate is run, how its result is classified, and how its
measurements are normalized. Only that agreement lives here. Resolved-config
validation deliberately does not: Megatron reads a serialized `ConfigContainer`
while FSDP inspects a device mesh, so one signature would be a parameter bag
neither backend fills.
"""

from .run import OOM_EXIT_CODE, oom_marker_paths, read_status, run_candidate
from .sidecar import (
    MANIFEST_NAME,
    SIDECAR_NAME,
    iter_candidate_dirs,
    load_sidecar_metrics,
    rank_memory_peaks,
    slice_warmup,
)

__all__ = [
    "MANIFEST_NAME",
    "OOM_EXIT_CODE",
    "SIDECAR_NAME",
    "iter_candidate_dirs",
    "load_sidecar_metrics",
    "oom_marker_paths",
    "rank_memory_peaks",
    "read_status",
    "run_candidate",
    "slice_warmup",
]
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
python -m pytest tests/test_training_shared.py -q
```

Expected: PASS, 18 tests.

- [ ] **Step 7: Migrate Megatron's `execute.py` onto the shared runner**

Replace the whole body of `srl/profiler/adapters/megatron/execute.py` with:

```python
"""Run one Megatron training candidate under torchrun."""

from __future__ import annotations

from pathlib import Path

from ...search_space import TrainingCandidate
from ..training import OOM_EXIT_CODE, oom_marker_paths, run_candidate
from .config import MegatronBridgeConfig
from .prepare import prepare_bridge_candidate

__all__ = ["OOM_EXIT_CODE", "oom_marker_paths", "run_bridge_candidate"]


def run_bridge_candidate(
    config: MegatronBridgeConfig,
    candidate: TrainingCandidate,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    """Prepare one candidate's launch spec and execute it in its own torchrun process."""
    command, run_dir = prepare_bridge_candidate(config, candidate)
    return run_candidate(
        run_dir=run_dir,
        command=command,
        environment=config.environment,
        dry_run=dry_run,
        resume=resume,
    )
```

`OOM_EXIT_CODE` and `oom_marker_paths` are re-exported because `launch.py` imports
`OOM_EXIT_CODE` from here and `collect.py` imports `oom_marker_paths` from here.

- [ ] **Step 8: Migrate Megatron's `collect.py` onto the shared sidecar helpers**

In `srl/profiler/adapters/megatron/collect.py`:

1. Delete `_load_sidecar_metrics` (lines 25-33) and `_status` (lines 78-101).
2. Replace the `from .execute import oom_marker_paths` import with:

```python
from ..training import (
    iter_candidate_dirs,
    load_sidecar_metrics,
    rank_memory_peaks,
    read_status,
    slice_warmup,
)
```

3. In `collect_bridge_record`, replace the status and metrics lines:

```python
    status, duration_s, error = read_status(source_dir)
    metrics = load_sidecar_metrics(source_dir) or {}

    step_times = slice_warmup(
        tuple(float(item) for item in metrics.get("step_times_s", [])),
        config.warmup_steps,
    )
```

4. Replace the peak computation:

```python
    peak_allocated, peak_reserved = rank_memory_peaks(rank_memory)
```

5. Replace the body of `collect_bridge_records`:

```python
def collect_bridge_records(
    config: MegatronBridgeConfig,
    candidates: list[TrainingCandidate],
) -> list[ProfileRecord]:
    """Collect all candidate directories that have adapter manifests."""
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    return [
        collect_bridge_record(config, by_id[candidate_id], run_dir=run_dir)
        for candidate_id, run_dir in iter_candidate_dirs(
            config.raw_root,
            [candidate.candidate_id for candidate in candidates],
            allow_incomplete=config.allow_incomplete,
        )
    ]
```

Remove any now-unused imports (`json` may still be needed for the manifest read;
check with ruff).

- [ ] **Step 9: Verify Megatron's behaviour is unchanged**

```bash
python -m pytest tests/test_profile_adapters.py tests/test_megatron_launch.py tests/test_profiling_cli.py -q
python -m ruff check srl/profiler/adapters/
```

Expected: all pass; ruff reports no unused imports.

- [ ] **Step 10: Confirm the full baseline still holds**

```bash
python -m pytest tests/ -q
```

Expected: `620 passed` (602 baseline + 18 new).

- [ ] **Step 11: Commit**

```bash
git add srl/profiler/adapters/training/ srl/profiler/adapters/megatron/execute.py \
        srl/profiler/adapters/megatron/collect.py tests/test_training_shared.py
git commit -m "refactor: extract the backend-agnostic training adapter layer

Megatron's execute and collect stages carried roughly 135 lines that say
nothing about Megatron: running a torchrun child, classifying its result,
reading a measurement sidecar, and slicing warmup. FSDP needs all of it, so
it moves to adapters/training rather than being copied.

Resolved-config validation stays behind. Megatron reads a serialized
ConfigContainer while FSDP will inspect a device mesh; the purposes match but
the mechanisms do not, and one signature would be a parameter bag neither
backend fills.

The rule that OOM is decided only by marker files, never by scanning a log,
now lives in one place. A false positive can only raise the fitted memory
floor, so it is exactly the kind of invariant that must not be duplicated."
```

---

### Task 2: Accept the FSDP schedule and add `FsdpCandidate`

The schedule identifier `"1f1b"` is hard-rejected in four places. Two must accept `"fsdp"`, one is untouched, and one must stay restrictive so the two backends' records never land in one fit.

**Files:**
- Modify: `srl/profiler/schema.py:374`
- Modify: `srl/cost/training/base.py:66`
- Create: `srl/profiler/fsdp_search_space.py`
- Test: `tests/test_fsdp_search_space.py`

**Interfaces:**
- Consumes: `Parallelism`, `ProfileValidationError` from `srl/profiler/schema.py`.
- Produces:
  - `TRAINING_SCHEDULES: frozenset[str]` in `srl/profiler/schema.py`, value `frozenset({"1f1b", "fsdp"})`
  - `FsdpCandidate` frozen dataclass with fields `shard_size: int`, `replicate_size: int`, `n_seqs_per_rank: int`, `sequence_length: int`, `gradient_checkpointing: bool`, `optimizer_offload: bool`, `reshard_after_forward: bool`, `sequence_parallel: int = 1`, `repeat: int = 0`; properties `world_size: int`, `candidate_id: str`, `num_microbatches: int`, `micro_batch_size: int`, `global_batch_size: int`, `sequence_shape: tuple[int, ...]`, `parallelism: Parallelism`; methods `to_dict(*, include_id: bool = True) -> dict[str, Any]`, `from_dict(value) -> FsdpCandidate`
  - `enumerate_fsdp_candidates(*, world_size: int, shard_sizes: list[int], n_seqs_per_rank: list[int], sequence_lengths: list[int], gradient_checkpointing: list[bool], optimizer_offload: list[bool], reshard_after_forward: list[bool], sequence_parallel_sizes: list[int] | None = None, repeats: int = 1) -> list[FsdpCandidate]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fsdp_search_space.py`:

```python
"""FSDP candidate enumeration and its topology constraints."""

from __future__ import annotations

import pytest

from srl.profiler.fsdp_search_space import FsdpCandidate, enumerate_fsdp_candidates
from srl.profiler.schema import ProfileValidationError


def _candidate(**overrides) -> FsdpCandidate:
    defaults = {
        "shard_size": 8,
        "replicate_size": 1,
        "n_seqs_per_rank": 2,
        "sequence_length": 1024,
        "gradient_checkpointing": True,
        "optimizer_offload": False,
        "reshard_after_forward": True,
    }
    return FsdpCandidate(**{**defaults, **overrides})


def test_world_size_is_the_mesh_product():
    assert _candidate(shard_size=4, replicate_size=2).world_size == 8


def test_every_rank_receives_distinct_data_so_dp_equals_world_size():
    """HSDP replicates parameters, not data, so the profile topology is pure DP."""
    parallelism = _candidate(shard_size=4, replicate_size=2).parallelism
    assert (parallelism.tp, parallelism.pp, parallelism.cp) == (1, 1, 1)
    assert parallelism.dp == 8
    assert parallelism.world_size == 8


def test_one_microbatch_per_rank_at_profile_time():
    """Profiling runs uniform shapes; packing is a simulate-side concern."""
    candidate = _candidate(n_seqs_per_rank=4)
    assert candidate.num_microbatches == 1
    assert candidate.micro_batch_size == 4
    assert candidate.sequence_shape == (1024, 1024, 1024, 1024)


def test_global_batch_size_matches_the_payload_invariant():
    candidate = _candidate(shard_size=8, replicate_size=1, n_seqs_per_rank=2)
    expected = candidate.micro_batch_size * candidate.parallelism.dp * candidate.num_microbatches
    assert candidate.global_batch_size == expected


def test_candidate_id_is_stable_and_sensitive():
    assert _candidate().candidate_id == _candidate().candidate_id
    assert _candidate().candidate_id != _candidate(sequence_length=2048).candidate_id
    assert len(_candidate().candidate_id) == 16


def test_round_trips_through_a_dict():
    candidate = _candidate(replicate_size=2, shard_size=4, optimizer_offload=True)
    assert FsdpCandidate.from_dict(candidate.to_dict()) == candidate


def test_rejects_sequence_parallelism():
    """Ulysses needs an all-to-all attention wrapper that does not exist yet."""
    with pytest.raises(ProfileValidationError, match="sequence_parallel"):
        _candidate(sequence_parallel=2)


def test_rejects_non_positive_dimensions():
    with pytest.raises(ProfileValidationError):
        _candidate(shard_size=0)
    with pytest.raises(ProfileValidationError):
        _candidate(n_seqs_per_rank=0)
    with pytest.raises(ProfileValidationError):
        _candidate(sequence_length=0)


def test_enumerate_derives_replicate_from_world_size():
    candidates = enumerate_fsdp_candidates(
        world_size=8,
        shard_sizes=[2, 4, 8],
        n_seqs_per_rank=[1],
        sequence_lengths=[512],
        gradient_checkpointing=[True],
        optimizer_offload=[False],
        reshard_after_forward=[True],
    )
    assert sorted((c.replicate_size, c.shard_size) for c in candidates) == [
        (1, 8),
        (2, 4),
        (4, 2),
    ]


def test_enumerate_skips_shard_sizes_that_do_not_divide_the_world():
    candidates = enumerate_fsdp_candidates(
        world_size=8,
        shard_sizes=[3, 8],
        n_seqs_per_rank=[1],
        sequence_lengths=[512],
        gradient_checkpointing=[True],
        optimizer_offload=[False],
        reshard_after_forward=[True],
    )
    assert [c.shard_size for c in candidates] == [8]


def test_enumerate_varies_both_axes_so_the_design_can_reach_rank_three():
    """`token_cost` needs sequence length to move; the two overheads need slots to move."""
    candidates = enumerate_fsdp_candidates(
        world_size=8,
        shard_sizes=[8],
        n_seqs_per_rank=[1, 2, 4],
        sequence_lengths=[512, 1024],
        gradient_checkpointing=[True],
        optimizer_offload=[False],
        reshard_after_forward=[True],
    )
    assert len({c.sequence_length for c in candidates}) == 2
    assert len({c.n_seqs_per_rank for c in candidates}) == 3
    assert len(candidates) == 6


def test_enumerate_repeats_produce_distinct_candidates():
    candidates = enumerate_fsdp_candidates(
        world_size=8,
        shard_sizes=[8],
        n_seqs_per_rank=[1],
        sequence_lengths=[512],
        gradient_checkpointing=[True],
        optimizer_offload=[False],
        reshard_after_forward=[True],
        repeats=3,
    )
    assert [c.repeat for c in candidates] == [0, 1, 2]
    assert len({c.candidate_id for c in candidates}) == 3


def test_enumerate_rejects_non_positive_repeats():
    with pytest.raises(ProfileValidationError):
        enumerate_fsdp_candidates(
            world_size=8,
            shard_sizes=[8],
            n_seqs_per_rank=[1],
            sequence_lengths=[512],
            gradient_checkpointing=[True],
            optimizer_offload=[False],
            reshard_after_forward=[True],
            repeats=0,
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_fsdp_search_space.py -q
```

Expected: `ModuleNotFoundError: No module named 'srl.profiler.fsdp_search_space'`.

- [ ] **Step 3: Widen the schedule check in `schema.py`**

In `srl/profiler/schema.py`, add above `TrainingPayload` (near the other module
constants):

```python
# Pipeline schedule identifiers a training payload may carry. `1f1b` is Megatron's
# interleaved-free pipeline schedule; `fsdp` marks a backend with no pipeline at all,
# where the schedule axis degenerates rather than taking another value.
TRAINING_SCHEDULES = frozenset({"1f1b", "fsdp"})
```

Replace lines 374-375:

```python
        if self.schedule.lower() not in TRAINING_SCHEDULES:
            raise ProfileValidationError(
                f"Training schedule must be one of {sorted(TRAINING_SCHEDULES)}, "
                f"got {self.schedule!r}."
            )
```

- [ ] **Step 4: Widen the schedule check in `base.py`**

In `srl/cost/training/base.py`, extend the existing schema import:

```python
from srl.profiler.schema import TRAINING_SCHEDULES, Parallelism, SequenceStats
```

Replace lines 66-67:

```python
        if self.schedule.lower() not in TRAINING_SCHEDULES:
            raise TrainingOutOfDomainError(
                f"Training schedule must be one of {sorted(TRAINING_SCHEDULES)}, "
                f"got {self.schedule!r}."
            )
```

Leave `srl/profiler/search_space.py:44` alone: `TrainingCandidate` is Megatron's
own type and FSDP uses `FsdpCandidate`. Leave `srl/cost/training/hydraulis.py:78`
alone as well: that filter is what keeps FSDP records out of the Megatron fit.

- [ ] **Step 5: Create `fsdp_search_space.py`**

Create `srl/profiler/fsdp_search_space.py`:

```python
"""Candidate generation for FSDP2 and HSDP profiling sweeps.

An FSDP candidate is described by a device mesh of two integers rather than the
tensor, pipeline, and context degrees Megatron needs. Data parallelism is the whole
topology: HSDP replicates *parameters* across mesh rows, not data, so every rank
still consumes a distinct microbatch and the profile topology is pure DP.

Profiling runs one uniform microbatch per rank. Splitting a rank's work into several
microbatches is what the simulate-side planner does with real trajectory lengths, and
mixing that into the sweep would make the measured shapes ragged for no gain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

from .schema import Parallelism, ProfileValidationError


@dataclass(frozen=True)
class FsdpCandidate:
    """Describe one independently executable FSDP profiling candidate."""

    shard_size: int
    replicate_size: int
    n_seqs_per_rank: int
    sequence_length: int
    gradient_checkpointing: bool
    optimizer_offload: bool
    reshard_after_forward: bool
    sequence_parallel: int = 1
    repeat: int = 0

    def __post_init__(self) -> None:
        for name in ("shard_size", "replicate_size", "n_seqs_per_rank", "sequence_length"):
            if getattr(self, name) <= 0:
                raise ProfileValidationError(f"{name} must be positive.")
        if self.sequence_parallel != 1:
            raise ProfileValidationError(
                "FSDP profiling V1 requires sequence_parallel=1. Ulysses needs an "
                "all-to-all attention wrapper that is not implemented."
            )
        if self.repeat < 0:
            raise ProfileValidationError("repeat must be non-negative.")

    @property
    def world_size(self) -> int:
        return self.shard_size * self.replicate_size

    @property
    def num_microbatches(self) -> int:
        """One microbatch per rank: profiling measures uniform shapes only."""
        return 1

    @property
    def micro_batch_size(self) -> int:
        return self.n_seqs_per_rank

    @property
    def global_batch_size(self) -> int:
        return self.n_seqs_per_rank * self.world_size * self.num_microbatches

    @property
    def sequence_shape(self) -> tuple[int, ...]:
        return (self.sequence_length,) * self.n_seqs_per_rank

    @property
    def parallelism(self) -> Parallelism:
        """
        Return the topology as the shared schema expresses it.

        Sharding and replication are configuration, not data partitioning, so they
        are carried by the artifact selector rather than by this type. Every rank
        holds different data, which is what `dp` means here.
        """
        return Parallelism(tp=1, pp=1, dp=self.world_size, cp=1, world_size=self.world_size)

    @property
    def candidate_id(self) -> str:
        """Return a compact, deterministic identity without encoding a lookup protocol."""
        value = json.dumps(self.to_dict(include_id=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if include_id:
            value["candidate_id"] = self.candidate_id
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FsdpCandidate:
        return cls(
            shard_size=int(value["shard_size"]),
            replicate_size=int(value["replicate_size"]),
            n_seqs_per_rank=int(value["n_seqs_per_rank"]),
            sequence_length=int(value["sequence_length"]),
            gradient_checkpointing=bool(value["gradient_checkpointing"]),
            optimizer_offload=bool(value["optimizer_offload"]),
            reshard_after_forward=bool(value["reshard_after_forward"]),
            sequence_parallel=int(value.get("sequence_parallel", 1)),
            repeat=int(value.get("repeat", 0)),
        )


def enumerate_fsdp_candidates(
    *,
    world_size: int,
    shard_sizes: list[int],
    n_seqs_per_rank: list[int],
    sequence_lengths: list[int],
    gradient_checkpointing: list[bool],
    optimizer_offload: list[bool],
    reshard_after_forward: list[bool],
    sequence_parallel_sizes: list[int] | None = None,
    repeats: int = 1,
) -> list[FsdpCandidate]:
    """
    Enumerate the candidates a fixed world size admits.

    The replicate degree is derived rather than swept: the mesh must cover the world
    exactly, so a shard degree that does not divide it describes no runnable topology.
    """
    if repeats <= 0:
        raise ProfileValidationError("repeats must be positive.")
    if world_size <= 0:
        raise ProfileValidationError("world_size must be positive.")
    sp_values = sequence_parallel_sizes or [1]
    candidates: list[FsdpCandidate] = []
    dimensions = product(
        sorted(set(shard_sizes)),
        sorted(set(n_seqs_per_rank)),
        sorted(set(sequence_lengths)),
        sorted(set(gradient_checkpointing)),
        sorted(set(optimizer_offload)),
        sorted(set(reshard_after_forward)),
        sorted(set(sp_values)),
    )
    for shard, n_seqs, length, checkpointing, offload, reshard, sp in dimensions:
        if shard <= 0 or world_size % shard != 0 or sp != 1:
            continue
        for repeat in range(repeats):
            candidates.append(
                FsdpCandidate(
                    shard_size=shard,
                    replicate_size=world_size // shard,
                    n_seqs_per_rank=n_seqs,
                    sequence_length=length,
                    gradient_checkpointing=checkpointing,
                    optimizer_offload=offload,
                    reshard_after_forward=reshard,
                    sequence_parallel=sp,
                    repeat=repeat,
                )
            )
    return candidates
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
python -m pytest tests/test_fsdp_search_space.py -q
```

Expected: PASS, 13 tests.

- [ ] **Step 7: Verify the schedule widening broke nothing**

```bash
python -m pytest tests/ -q
```

Expected: `633 passed` (620 + 13).

- [ ] **Step 8: Commit**

```bash
git add srl/profiler/schema.py srl/cost/training/base.py \
        srl/profiler/fsdp_search_space.py tests/test_fsdp_search_space.py
git commit -m "feat: admit an FSDP schedule and enumerate FSDP candidates

The schedule identifier was checked against the literal 1f1b in four places.
Two of them describe what a payload or a query may contain and now accept a
frozenset that also holds fsdp, which marks a backend with no pipeline rather
than a different pipeline.

The other two stay as they were, deliberately. TrainingCandidate is Megatron's
own type and FSDP has its own. Hydraulis's record filter is what keeps FSDP
records from being fitted by the Megatron model, so restricting it is the
feature, not an oversight.

FsdpCandidate describes a mesh of two integers. It reports dp equal to the
world size because HSDP replicates parameters rather than data, so every rank
still consumes a distinct microbatch, and it fixes num_microbatches at one
because splitting a rank's work belongs to the simulate-side planner."
```

---

### Task 3: FSDP adapter config, prepare, and manifest

The three stages that run on the CLI import path and must never import torch.

**Files:**
- Create: `srl/profiler/adapters/fsdp/__init__.py`
- Create: `srl/profiler/adapters/fsdp/config.py`
- Create: `srl/profiler/adapters/fsdp/prepare.py`
- Create: `srl/profiler/adapters/fsdp/execute.py`
- Test: `tests/test_fsdp_prepare.py`

**Interfaces:**
- Consumes: `FsdpCandidate` (Task 2); `run_candidate`, `OOM_EXIT_CODE`, `oom_marker_paths` (Task 1); `model_descriptor_from_yaml` from `srl/profiler/hf.py`; `HardwareDescriptor` from `srl/profiler/schema.py`.
- Produces:
  - `FsdpProfileConfig` frozen dataclass: `output_dir: Path`, `model: ModelDescriptor`, `hardware: HardwareDescriptor`, `experiment_name: str = ""`, `compute_dtype: str = "bf16"`, `optimizer: str = "adam"`, `parameter_dtype: str = "bf16"`, `gradient_dtype: str = "fp32"`, `optimizer_state_dtype: str = "fp32"`, `max_steps: int = 12`, `warmup_steps: int = 4`, `num_layers_override: int | None = None`, `attn_implementation: str = "flash_attention_2"`, `environment: dict[str, str]`, `backend_version: str`, `allow_incomplete: bool = False`; classmethod `from_dict(value, *, base_dir=None)`; property `raw_root: Path`
  - `LAUNCH_MODULE: str`, `LAUNCH_SPEC_VERSION: int`
  - `write_candidate_manifest(path, candidates: list[FsdpCandidate]) -> Path`
  - `read_candidate_manifest(path) -> list[FsdpCandidate]`
  - `build_launch_spec(config, candidate, run_dir) -> dict[str, Any]`
  - `build_launch_command(candidate, spec_path, *, nnodes=1, node_rank=0, master_addr=None, master_port=None) -> list[str]`
  - `prepare_fsdp_candidate(config, candidate, *, nnodes=1, node_rank=0, master_addr=None, master_port=None) -> tuple[list[str], Path]`
  - `run_fsdp_candidate(config, candidate, *, dry_run=False, resume=False) -> Path`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fsdp_prepare.py`:

```python
"""FSDP profile configuration, launch specs, and manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from srl.profiler.adapters.fsdp import (
    FsdpProfileConfig,
    build_launch_command,
    build_launch_spec,
    prepare_fsdp_candidate,
    read_candidate_manifest,
    write_candidate_manifest,
)
from srl.profiler.fsdp_search_space import FsdpCandidate
from srl.profiler.schema import HardwareDescriptor, ModelDescriptor, ProfileValidationError


def _model() -> ModelDescriptor:
    return ModelDescriptor(
        name="Qwen3-8B",
        architecture="Qwen3ForCausalLM",
        parameter_count=8_190_735_360,
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        extra={"model_type": "qwen3", "vocab_size": 151936},
    )


def _hardware() -> HardwareDescriptor:
    return HardwareDescriptor(
        gpu_name="H20", gpu_count=8, gpu_memory_mib=97871, node_count=1, gpus_per_node=8
    )


def _config(tmp_path: Path, **overrides) -> FsdpProfileConfig:
    defaults = {
        "output_dir": tmp_path,
        "model": _model(),
        "hardware": _hardware(),
        "model_path": "/models/Qwen3-8B",
    }
    return FsdpProfileConfig(**{**defaults, **overrides})


def _candidate(**overrides) -> FsdpCandidate:
    defaults = {
        "shard_size": 4,
        "replicate_size": 2,
        "n_seqs_per_rank": 2,
        "sequence_length": 1024,
        "gradient_checkpointing": True,
        "optimizer_offload": False,
        "reshard_after_forward": True,
    }
    return FsdpCandidate(**{**defaults, **overrides})


def test_defaults_retain_eight_measured_steps():
    """Twelve steps discarding four: one warmup step does not reach steady state."""
    config = _config(Path("/tmp"))
    assert config.max_steps == 12
    assert config.warmup_steps == 4
    assert config.max_steps - config.warmup_steps == 8


def test_rejects_a_warmup_that_consumes_every_step(tmp_path: Path):
    with pytest.raises(ProfileValidationError):
        _config(tmp_path, max_steps=4, warmup_steps=4)


def test_backend_version_records_the_torch_version(tmp_path: Path):
    """The backend is torch itself, so reproducibility is pinned to its version."""
    import torch

    assert _config(tmp_path).backend_version == torch.__version__


def test_raw_root_separates_experiments(tmp_path: Path):
    plain = _config(tmp_path).raw_root
    named = _config(tmp_path, experiment_name="sweep_a").raw_root
    assert plain == tmp_path / "raw" / "fsdp"
    assert named == tmp_path / "raw" / "fsdp" / "sweep_a"


def test_from_dict_reads_the_yaml_shape(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "num_hidden_layers": 36,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 151936,
            }
        ),
        encoding="utf-8",
    )
    config = FsdpProfileConfig.from_dict(
        {
            "model": str(tmp_path),
            "hardware": {
                "gpu_name": "H20",
                "gpu_count": 8,
                "gpu_memory_mib": 97871,
                "node_count": 1,
                "gpus_per_node": 8,
            },
            "output_dir": "out",
            "training": {
                "max_steps": 12,
                "warmup_steps": 4,
                "num_layers_override": 2,
                "optimizer": "adam",
            },
            "profiling": {"allow_incomplete": True},
            "environment": {"HF_HUB_OFFLINE": "1"},
        },
        base_dir=tmp_path,
    )
    assert config.num_layers_override == 2
    assert config.allow_incomplete is True
    assert config.environment == {"HF_HUB_OFFLINE": "1"}
    assert config.output_dir == (tmp_path / "out").resolve()


def test_launch_spec_carries_the_mesh_not_tensor_degrees(tmp_path: Path):
    spec = build_launch_spec(_config(tmp_path), _candidate(), tmp_path)
    assert spec["mesh"] == {"replicate_size": 2, "shard_size": 4}
    assert "tp" not in spec
    assert "pp" not in spec


def test_launch_spec_is_versioned_and_self_contained(tmp_path: Path):
    spec = build_launch_spec(_config(tmp_path), _candidate(), tmp_path)
    assert spec["schema_version"] == 1
    for key in (
        "model_path",
        "sequence_length",
        "n_seqs_per_rank",
        "train_iters",
        "gradient_checkpointing",
        "optimizer_offload",
        "reshard_after_forward",
        "parameter_dtype",
        "gradient_dtype",
        "optimizer_state_dtype",
        "attn_implementation",
        "run_dir",
    ):
        assert key in spec, key
    # A spec must survive a round trip through a file, which is how the child reads it.
    assert json.loads(json.dumps(spec))["mesh"]["shard_size"] == 4


def test_launch_spec_propagates_the_layer_override(tmp_path: Path):
    spec = build_launch_spec(_config(tmp_path, num_layers_override=2), _candidate(), tmp_path)
    assert spec["num_layers_override"] == 2


def test_launch_command_requests_one_process_per_rank(tmp_path: Path):
    command = build_launch_command(_candidate(), tmp_path / "launch_spec.json")
    assert command[0] == "torchrun"
    assert "--nproc-per-node" in command
    assert command[command.index("--nproc-per-node") + 1] == "8"
    assert "srl.profiler.adapters.fsdp.launch" in command


def test_launch_command_supports_multiple_nodes(tmp_path: Path):
    command = build_launch_command(
        _candidate(),
        tmp_path / "launch_spec.json",
        nnodes=2,
        node_rank=1,
        master_addr="10.0.0.1",
        master_port=29500,
    )
    assert "--standalone" not in command
    assert command[command.index("--nnodes") + 1] == "2"
    assert command[command.index("--node-rank") + 1] == "1"
    assert command[command.index("--master-addr") + 1] == "10.0.0.1"
    # Per-node process count, not the world size.
    assert command[command.index("--nproc-per-node") + 1] == "4"


def test_prepare_writes_a_spec_and_an_immutable_manifest(tmp_path: Path):
    config = _config(tmp_path)
    candidate = _candidate()
    command, run_dir = prepare_fsdp_candidate(config, candidate)
    assert run_dir == config.raw_root / candidate.candidate_id
    spec = json.loads((run_dir / "launch_spec.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert spec["mesh"]["shard_size"] == 4
    assert manifest["adapter"] == "fsdp"
    assert manifest["candidate"]["candidate_id"] == candidate.candidate_id
    assert manifest["command"] == command
    assert manifest["backend_version"]


def test_manifest_round_trips_candidates(tmp_path: Path):
    candidates = [_candidate(), _candidate(sequence_length=2048)]
    path = write_candidate_manifest(tmp_path / "candidates.json", candidates)
    assert read_candidate_manifest(path) == candidates


def test_manifest_rejects_an_unknown_schema(tmp_path: Path):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"schema_version": 99, "candidates": []}), encoding="utf-8")
    with pytest.raises(ProfileValidationError):
        read_candidate_manifest(path)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_fsdp_prepare.py -q
```

Expected: `ModuleNotFoundError: No module named 'srl.profiler.adapters.fsdp'`.

- [ ] **Step 3: Create `config.py`**

Create `srl/profiler/adapters/fsdp/config.py`:

```python
"""Configuration for SimRL-owned FSDP2 training profiling.

There is no recipe inference here. Megatron needs it because its library recipes
carry tuned defaults keyed by model family and size; an FSDP run is fully described
by a HuggingFace checkpoint plus a device mesh, so the checkpoint is read directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ...hf import model_descriptor_from_yaml
from ...schema import HardwareDescriptor, ModelDescriptor, ProfileValidationError


@dataclass(frozen=True)
class FsdpProfileConfig:
    """Configuration for candidate execution and result normalization."""

    output_dir: Path
    model: ModelDescriptor
    hardware: HardwareDescriptor
    model_path: str
    experiment_name: str = ""
    compute_dtype: str = "bf16"
    optimizer: str = "adam"
    parameter_dtype: str = "bf16"
    gradient_dtype: str = "fp32"
    optimizer_state_dtype: str = "fp32"
    # Twelve steps discarding four. Under a single warmup step the first profiled step
    # was the slowest in 18 of 24 Megatron runs where chance alone gives about 8, so one
    # step does not reach steady state. Eight retained samples then support a median and
    # outlier rejection, which two samples cannot.
    max_steps: int = 12
    warmup_steps: int = 4
    num_layers_override: int | None = None
    # flash_attention_2 is required for the packed no-pad batching mode and is what verl
    # runs, so the profile and the simulated backend agree on the kernel family.
    attn_implementation: str = "flash_attention_2"
    environment: dict[str, str] = field(default_factory=dict)
    # The backend *is* torch, so its version is what pins reproducibility.
    backend_version: str = field(default_factory=lambda: torch.__version__)
    allow_incomplete: bool = False

    def __post_init__(self) -> None:
        if self.max_steps <= 0 or self.warmup_steps < 0:
            raise ProfileValidationError(
                "max_steps must be positive and warmup_steps must be non-negative."
            )
        if self.max_steps <= self.warmup_steps:
            raise ProfileValidationError(
                f"max_steps {self.max_steps} must exceed warmup_steps {self.warmup_steps} "
                "so at least one profiled step remains."
            )
        if self.num_layers_override is not None and self.num_layers_override <= 0:
            raise ProfileValidationError("num_layers_override must be positive.")
        if not self.model_path:
            raise ProfileValidationError("model_path must name a checkpoint directory.")

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> FsdpProfileConfig:
        root = base_dir or Path.cwd()

        def resolve(path_value: str) -> Path:
            path = Path(path_value)
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        training = value.get("training", {})
        profiling = value.get("profiling", {})
        model = model_descriptor_from_yaml(value["model"], base_dir=root)
        num_layers_override = training.get("num_layers_override")
        raw_model = value["model"]
        model_path = raw_model if isinstance(raw_model, str) else raw_model.get("path", model.name)
        return cls(
            output_dir=resolve(value.get("output_dir", "profile_output")),
            model=model,
            hardware=HardwareDescriptor.from_dict(value["hardware"]),
            model_path=str(model_path),
            experiment_name=str(value.get("experiment_name", "")),
            compute_dtype=str(training.get("compute_dtype", "bf16")),
            optimizer=str(training.get("optimizer", "adam")),
            parameter_dtype=str(training.get("parameter_dtype", "bf16")),
            gradient_dtype=str(training.get("gradient_dtype", "fp32")),
            optimizer_state_dtype=str(training.get("optimizer_state_dtype", "fp32")),
            max_steps=int(training.get("max_steps", 12)),
            warmup_steps=int(training.get("warmup_steps", 4)),
            num_layers_override=(
                None if num_layers_override is None else int(num_layers_override)
            ),
            attn_implementation=str(
                training.get("attn_implementation", "flash_attention_2")
            ),
            environment={
                str(key): str(item) for key, item in value.get("environment", {}).items()
            },
            allow_incomplete=bool(profiling.get("allow_incomplete", False)),
        )

    @property
    def raw_root(self) -> Path:
        if self.experiment_name:
            return self.output_dir / "raw" / "fsdp" / self.experiment_name
        return self.output_dir / "raw" / "fsdp"
```

Note: importing `torch` at module scope is acceptable here because `torch` is already
a hard dependency imported by `srl.cost.training.latency`; it is `megatron` and the
FSDP *training* APIs that must stay out of the CLI path.

- [ ] **Step 4: Create `prepare.py`**

Create `srl/profiler/adapters/fsdp/prepare.py`:

```python
"""Candidate manifests, launch specs, and torchrun commands for FSDP profiling."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ...fsdp_search_space import FsdpCandidate
from ...schema import ProfileValidationError
from .config import FsdpProfileConfig

# The torchrun child entry point. Referenced by module path rather than imported so
# that nothing on the CLI import path pulls in the FSDP training APIs.
LAUNCH_MODULE = "srl.profiler.adapters.fsdp.launch"

LAUNCH_SPEC_VERSION = 1


def write_candidate_manifest(path: str | Path, candidates: list[FsdpCandidate]) -> Path:
    """Write enumerated candidates in a stable JSON document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def read_candidate_manifest(path: str | Path) -> list[FsdpCandidate]:
    """Read candidates generated by SimRL."""
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema_version") != 1:
        raise ProfileValidationError("Unsupported candidate manifest schema.")
    return [FsdpCandidate.from_dict(item) for item in value["candidates"]]


def build_launch_spec(
    config: FsdpProfileConfig,
    candidate: FsdpCandidate,
    run_dir: Path,
) -> dict[str, Any]:
    """
    Describe one candidate completely enough that the child needs no other input.

    Everything the child reads is here. Passing these as command-line flags is what
    the Megatron path had to stop doing: a wrapper that parses known arguments and
    discards the rest drops settings silently, whereas a missing key in a spec is an
    immediate error.
    """
    vocab_size = config.model.extra.get("vocab_size")
    if vocab_size is None:
        raise ProfileValidationError(
            "The HuggingFace config must provide vocab_size for FSDP profiling."
        )
    return {
        "schema_version": LAUNCH_SPEC_VERSION,
        "model_path": config.model_path,
        "mesh": {
            "replicate_size": candidate.replicate_size,
            "shard_size": candidate.shard_size,
        },
        "sequence_length": candidate.sequence_length,
        "n_seqs_per_rank": candidate.n_seqs_per_rank,
        "num_microbatches": candidate.num_microbatches,
        "train_iters": config.max_steps,
        "warmup_steps": config.warmup_steps,
        "vocab_size": int(vocab_size),
        "gradient_checkpointing": candidate.gradient_checkpointing,
        "optimizer_offload": candidate.optimizer_offload,
        "reshard_after_forward": candidate.reshard_after_forward,
        "compute_dtype": config.compute_dtype,
        "parameter_dtype": config.parameter_dtype,
        "gradient_dtype": config.gradient_dtype,
        "optimizer_state_dtype": config.optimizer_state_dtype,
        "optimizer": config.optimizer,
        "attn_implementation": config.attn_implementation,
        "num_layers_override": config.num_layers_override,
        "run_dir": str(run_dir),
    }


def build_launch_command(
    candidate: FsdpCandidate,
    spec_path: Path,
    *,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str | None = None,
    master_port: int | None = None,
) -> list[str]:
    """Build the torchrun command that profiles one candidate."""
    if nnodes <= 0 or candidate.world_size % nnodes != 0:
        raise ProfileValidationError(
            f"World size {candidate.world_size} is not divisible across {nnodes} nodes."
        )
    per_node = candidate.world_size // nnodes
    command = ["torchrun"]
    if nnodes == 1:
        command.append("--standalone")
    else:
        command += ["--nnodes", str(nnodes), "--node-rank", str(node_rank)]
        if master_addr:
            command += ["--master-addr", master_addr]
        if master_port:
            command += ["--master-port", str(master_port)]
    command += ["--nproc-per-node", str(per_node), "-m", LAUNCH_MODULE, "--spec", str(spec_path)]
    return command


def prepare_fsdp_candidate(
    config: FsdpProfileConfig,
    candidate: FsdpCandidate,
    *,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str | None = None,
    master_port: int | None = None,
) -> tuple[list[str], Path]:
    """Write one candidate's launch spec and immutable manifest."""
    run_dir = config.raw_root / candidate.candidate_id
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = build_launch_spec(config, candidate, run_dir)
    spec_path = run_dir / "launch_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    command = build_launch_command(
        candidate,
        spec_path,
        nnodes=nnodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
    )
    manifest = {
        "adapter": "fsdp",
        "backend": "local",
        "launcher": {"nnodes": nnodes, "node_rank": node_rank, "python": sys.executable},
        "candidate": candidate.to_dict(),
        # Recorded so the collected record's architecture matches what actually ran: an
        # overridden layer count changes both the parameter count and the activation
        # footprint the memory model is fitted against.
        "num_layers_override": config.num_layers_override,
        "command": command,
        "launch_spec": str(spec_path),
        "backend_version": config.backend_version,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return command, run_dir
```

- [ ] **Step 5: Create `execute.py`**

Create `srl/profiler/adapters/fsdp/execute.py`:

```python
"""Run one FSDP training candidate under torchrun."""

from __future__ import annotations

from pathlib import Path

from ...fsdp_search_space import FsdpCandidate
from ..training import OOM_EXIT_CODE, oom_marker_paths, run_candidate
from .config import FsdpProfileConfig
from .prepare import prepare_fsdp_candidate

__all__ = ["OOM_EXIT_CODE", "oom_marker_paths", "run_fsdp_candidate"]


def run_fsdp_candidate(
    config: FsdpProfileConfig,
    candidate: FsdpCandidate,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    """Prepare one candidate's launch spec and execute it in its own torchrun process."""
    command, run_dir = prepare_fsdp_candidate(config, candidate)
    return run_candidate(
        run_dir=run_dir,
        command=command,
        environment=config.environment,
        dry_run=dry_run,
        resume=resume,
    )
```

- [ ] **Step 6: Create the package `__init__.py`**

Create `srl/profiler/adapters/fsdp/__init__.py`:

```python
"""SimRL-owned FSDP2 and HSDP training profiling.

`launch` and `measure` construct a model and take collectives, so they are absent
from this namespace: only the torchrun child imports them. Everything exported here
is safe on the CLI import path.
"""

from .collect import collect_fsdp_record, collect_fsdp_records
from .config import FsdpProfileConfig
from .execute import OOM_EXIT_CODE, oom_marker_paths, run_fsdp_candidate
from .prepare import (
    build_launch_command,
    build_launch_spec,
    prepare_fsdp_candidate,
    read_candidate_manifest,
    write_candidate_manifest,
)

__all__ = [
    "FsdpProfileConfig",
    "OOM_EXIT_CODE",
    "build_launch_command",
    "build_launch_spec",
    "collect_fsdp_record",
    "collect_fsdp_records",
    "oom_marker_paths",
    "prepare_fsdp_candidate",
    "read_candidate_manifest",
    "run_fsdp_candidate",
    "write_candidate_manifest",
]
```

This imports `collect`, written in Task 5. To keep this task independently testable,
create a temporary `collect.py` containing only:

```python
"""Normalize raw FSDP candidate output into versioned profile records."""

from __future__ import annotations


def collect_fsdp_record(*args, **kwargs):
    raise NotImplementedError("Implemented in Task 5.")


def collect_fsdp_records(*args, **kwargs):
    raise NotImplementedError("Implemented in Task 5.")
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
python -m pytest tests/test_fsdp_prepare.py -q
```

Expected: PASS, 13 tests.

- [ ] **Step 8: Verify nothing on the CLI path imports the FSDP training APIs**

```bash
python -c "
import sys
import srl.profiler.adapters.fsdp as m
print('imported', m.__name__)
assert 'srl.profiler.adapters.fsdp.launch' not in sys.modules, 'launch leaked onto the import path'
assert 'srl.profiler.adapters.fsdp.measure' not in sys.modules, 'measure leaked onto the import path'
print('launch and measure stayed out of the namespace')
"
python -m pytest tests/ -q
```

Expected: both assertions hold; `646 passed` (633 + 13).

- [ ] **Step 9: Commit**

```bash
git add srl/profiler/adapters/fsdp/ tests/test_fsdp_prepare.py
git commit -m "feat: add the FSDP profiling config, launch spec, and manifest

An FSDP candidate is fully described by a checkpoint and a mesh of two
integers, so there is no recipe inference to mirror from the Megatron adapter.
The spec carries the mesh rather than tensor and pipeline degrees, and the
child reads only the spec: a missing key is then an immediate error rather
than a silently dropped setting.

max_steps defaults to twelve with four discarded, inherited from the Megatron
measurement work. One warmup step left the first profiled step the slowest in
18 of 24 runs, and eight retained samples are what make a median and outlier
rejection possible.

backend_version records the torch version, because for this adapter torch is
the backend rather than a dependency of it."
```

---

### Task 4: The torchrun child and its measurement

The one place this project writes real distributed training code. It is gated by a
two-layer smoke run before any full sweep.

**Files:**
- Create: `srl/profiler/adapters/fsdp/launch.py`
- Create: `srl/profiler/adapters/fsdp/measure.py`
- Test: `tests/test_fsdp_launch.py`

**Interfaces:**
- Consumes: `OOM_EXIT_CODE` from `srl/profiler/adapters/fsdp/execute.py`; the launch spec dict from Task 3.
- Produces:
  - `build_mesh(spec: dict[str, Any]) -> DeviceMesh`
  - `build_model(spec: dict[str, Any]) -> torch.nn.Module`
  - `apply_fsdp(model, spec, mesh) -> torch.nn.Module`
  - `synthetic_batch(spec: dict[str, Any], device: str) -> dict[str, torch.Tensor]`
  - `read_spec(path: Path) -> dict[str, Any]`
  - `FsdpMeasurement` class with `on_train_start()`, `step_begin()`, `step_end()`, `on_train_end()`, `local_memory() -> dict[str, int]`, `sidecar(rank_memory) -> dict[str, Any]`
  - `write_oom_sidecar(run_dir: Path, spec: dict[str, Any], error: BaseException) -> None`
  - `SIDECAR_NAME: str` (re-exported from the shared layer)

- [ ] **Step 1: Write the failing test**

Create `tests/test_fsdp_launch.py`. These tests must run without a process group, so
they exercise the pure functions and the sidecar shape only.

```python
"""FSDP launch-spec application and measurement bookkeeping.

Anything needing a process group is covered by the smoke run in Task 4 Step 8, not
here: a unit test that spawns eight ranks is neither fast nor diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from srl.profiler.adapters.fsdp.launch import build_model, read_spec, synthetic_batch
from srl.profiler.adapters.fsdp.measure import FsdpMeasurement, write_oom_sidecar


def _spec(tmp_path: Path, **overrides) -> dict:
    defaults = {
        "schema_version": 1,
        "model_path": "/models/Qwen3-8B",
        "mesh": {"replicate_size": 1, "shard_size": 8},
        "sequence_length": 128,
        "n_seqs_per_rank": 2,
        "num_microbatches": 1,
        "train_iters": 12,
        "warmup_steps": 4,
        "vocab_size": 151936,
        "gradient_checkpointing": True,
        "optimizer_offload": False,
        "reshard_after_forward": True,
        "compute_dtype": "bf16",
        "parameter_dtype": "bf16",
        "gradient_dtype": "fp32",
        "optimizer_state_dtype": "fp32",
        "optimizer": "adam",
        "attn_implementation": "flash_attention_2",
        "num_layers_override": 2,
        "run_dir": str(tmp_path),
    }
    return {**defaults, **overrides}


def test_read_spec_rejects_an_unknown_schema(tmp_path: Path):
    path = tmp_path / "launch_spec.json"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported launch spec"):
        read_spec(path)


def test_read_spec_returns_the_document(tmp_path: Path):
    path = tmp_path / "launch_spec.json"
    path.write_text(json.dumps(_spec(tmp_path)), encoding="utf-8")
    assert read_spec(path)["sequence_length"] == 128


def test_synthetic_batch_has_the_requested_shape(tmp_path: Path):
    batch = synthetic_batch(_spec(tmp_path), device="cpu")
    assert batch["input_ids"].shape == (2, 128)
    assert batch["labels"].shape == (2, 128)
    assert batch["position_ids"].shape == (2, 128)
    assert batch["input_ids"].max() < 151936


def test_synthetic_batch_position_ids_do_not_restart(tmp_path: Path):
    """Profiling measures padded uniform shapes; restarting ids would signal packing."""
    batch = synthetic_batch(_spec(tmp_path), device="cpu")
    assert torch.equal(batch["position_ids"][0], torch.arange(128))


def test_build_model_honours_the_layer_override(tmp_path: Path, monkeypatch):
    """A two-layer smoke model must actually have two layers, not thirty-six."""
    from transformers import AutoConfig

    def fake_from_pretrained(path, **kwargs):
        config = AutoConfig.for_model(
            "qwen3",
            num_hidden_layers=36,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1024,
        )
        return config

    monkeypatch.setattr(AutoConfig, "from_pretrained", fake_from_pretrained)
    spec = _spec(tmp_path, num_layers_override=2, attn_implementation="sdpa")
    model = build_model(spec)
    assert model.config.num_hidden_layers == 2
    assert len(model.model.layers) == 2
    # The override must reach the built module, not just the config it was read from.
    assert model.config.use_cache is False


def test_measurement_reports_one_time_per_step(tmp_path: Path):
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=_spec(tmp_path))
    for _ in range(3):
        measurement.step_begin()
        measurement.step_end()
    sidecar = measurement.sidecar([{"rank": 0, "allocated_bytes": 10, "reserved_bytes": 20}])
    assert len(sidecar["step_times_s"]) == 3
    assert all(value > 0 for value in sidecar["step_times_s"])


def test_measurement_keeps_warmup_steps_for_the_collector(tmp_path: Path):
    """The launcher is a pure measurer: the collector owns the warmup policy."""
    spec = _spec(tmp_path, train_iters=5, warmup_steps=2)
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=spec)
    for _ in range(5):
        measurement.step_begin()
        measurement.step_end()
    assert len(measurement.sidecar([])["step_times_s"]) == 5


def test_step_end_without_a_begin_is_an_error(tmp_path: Path):
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=_spec(tmp_path))
    with pytest.raises(RuntimeError, match="without a matching start"):
        measurement.step_end()


def test_sidecar_records_the_microbatch_shape(tmp_path: Path):
    spec = _spec(tmp_path, n_seqs_per_rank=3, sequence_length=64)
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=spec)
    measurement.step_begin()
    measurement.step_end()
    assert measurement.sidecar([])["microbatch_sequence_lengths"] == [[64, 64, 64]]


def test_sidecar_sorts_rank_memory(tmp_path: Path):
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=_spec(tmp_path))
    sidecar = measurement.sidecar(
        [
            {"rank": 2, "allocated_bytes": 3, "reserved_bytes": 4},
            {"rank": 0, "allocated_bytes": 1, "reserved_bytes": 2},
        ]
    )
    assert [item["rank"] for item in sidecar["rank_memory"]] == [0, 2]


def test_sidecar_carries_the_observed_mesh_for_cross_checking(tmp_path: Path):
    """A silently wrong mesh must fail in collect, not produce plausible numbers."""
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=_spec(tmp_path))
    measurement.record_topology(replicate_size=1, shard_size=8, parameter_count=1234)
    sidecar = measurement.sidecar([])
    assert sidecar["observed_mesh"] == {"replicate_size": 1, "shard_size": 8}
    assert sidecar["parameter_count"] == 1234


def test_sidecar_records_the_comm_compute_split(tmp_path: Path):
    """
    Ground truth for a later kernel-summation model.

    The regression never reads these timers; they exist because measuring the
    all-gather and reduce-scatter share now is cheap and re-profiling later is not.
    """
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=_spec(tmp_path))
    measurement.step_begin()
    measurement.record_collective_time(all_gather_s=0.02, reduce_scatter_s=0.03)
    measurement.step_end()
    timers = measurement.sidecar([])["timers_s"]
    assert timers["all_gather"] == (0.02,)
    assert timers["reduce_scatter"] == (0.03,)


def test_sidecar_omits_collective_timers_when_uninstrumented(tmp_path: Path):
    """Only one step is instrumented, so an empty timer dict must stay empty."""
    measurement = FsdpMeasurement(run_dir=tmp_path, spec=_spec(tmp_path))
    measurement.step_begin()
    measurement.step_end()
    assert measurement.sidecar([])["timers_s"] == {}


def test_oom_sidecar_is_written_per_rank(tmp_path: Path):
    write_oom_sidecar(tmp_path, _spec(tmp_path), RuntimeError("CUDA out of memory"))
    markers = list(tmp_path.glob("oom_rank*.json"))
    assert len(markers) == 1
    payload = json.loads(markers[0].read_text())
    assert payload["torch_oom"] is True
    assert payload["sequence_length"] == 128
    assert payload["n_seqs_per_rank"] == 2
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_fsdp_launch.py -q
```

Expected: `ModuleNotFoundError: No module named 'srl.profiler.adapters.fsdp.launch'`.

- [ ] **Step 3: Create `measure.py`**

Create `srl/profiler/adapters/fsdp/measure.py`:

```python
"""Measure step latency and per-rank memory peaks inside the FSDP training loop.

Takes collectives and reads the CUDA allocator, so this module is only ever imported
by the torchrun child, never from the adapter package namespace.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

SIDECAR_NAME = "srl_metrics.json"


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Replace `path` in one step so a reader never sees a partial document."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class FsdpMeasurement:
    """
    Record per-step wall time and per-rank CUDA peaks for one candidate.

    Timing uses a synchronized wall clock rather than CUDA events because the fitted
    model has an explicit per-step overhead term: the measurement must include the
    host-side optimizer step and the gradient reduce-scatter, not just stream time.

    No barrier is taken on the timing path. A barrier would charge every rank for the
    slowest one, turning a per-rank measurement into a straggler measurement.
    """

    def __init__(self, *, run_dir: Path, spec: dict[str, Any]):
        self._run_dir = run_dir
        self._spec = spec
        self._sequence_length = int(spec["sequence_length"])
        self._n_seqs_per_rank = int(spec["n_seqs_per_rank"])
        self._num_microbatches = int(spec["num_microbatches"])
        self._step_times: list[float] = []
        self._started_at: float | None = None
        self._static_allocated_bytes: int | None = None
        self._observed_mesh: dict[str, int] | None = None
        self._parameter_count: int | None = None
        self._collective_timers: dict[str, list[float]] = {}

    def on_train_start(self) -> None:
        """
        Reset peak counters once the model and optimizer exist.

        Resetting here discards the construction spike from weight loading while
        keeping the static pool: resetting peak statistics does not free memory, so
        already-allocated parameter, gradient, and optimizer state is immediately
        re-counted. The memory model fits the peak *including* that inventory.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._static_allocated_bytes = int(torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()

    def record_topology(
        self,
        *,
        replicate_size: int,
        shard_size: int,
        parameter_count: int,
    ) -> None:
        """
        Record what the process group actually built.

        The collector cross-checks this against the candidate. A mesh that silently
        differs from the requested one would otherwise yield a record attributed to
        the wrong topology, which is worse than no record at all.
        """
        self._observed_mesh = {"replicate_size": replicate_size, "shard_size": shard_size}
        self._parameter_count = parameter_count

    def step_begin(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._started_at = time.perf_counter()

    def record_collective_time(
        self,
        *,
        all_gather_s: float,
        reduce_scatter_s: float,
    ) -> None:
        """
        Record one step's collective share.

        Not read by the regression. It is ground truth for the kernel-summation model
        that `latency.py` names as the principled next step, and measuring it now costs
        one instrumented step where re-profiling the whole sweep later would not.
        """
        self._collective_timers.setdefault("all_gather", []).append(all_gather_s)
        self._collective_timers.setdefault("reduce_scatter", []).append(reduce_scatter_s)

    def step_end(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._started_at is None:
            raise RuntimeError("step_end fired without a matching start.")
        self._step_times.append(time.perf_counter() - self._started_at)
        self._started_at = None

    def on_train_end(self) -> None:
        """Gather every rank's peak and let rank zero write one consistent document."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        local = self.local_memory()
        if dist.is_initialized():
            gathered: list[Any] = [None] * _world_size()
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if _rank() == 0:
            _write_json_atomic(self._run_dir / SIDECAR_NAME, self.sidecar(gathered))

    def local_memory(self) -> dict[str, int]:
        """Return this rank's peak allocated and reserved bytes."""
        if not torch.cuda.is_available():
            return {"rank": _rank(), "allocated_bytes": 0, "reserved_bytes": 0}
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
        return {
            "rank": _rank(),
            "allocated_bytes": allocated,
            # The schema requires reserved >= allocated. Both come from the same
            # allocator, so this only guards against a driver-level surprise.
            "reserved_bytes": max(reserved, allocated),
        }

    def sidecar(self, rank_memory: list[dict[str, int]]) -> dict[str, Any]:
        """
        Build the collector's sidecar document.

        `step_times_s` keeps its warmup entries. The collector owns the warmup policy
        and slices them off, so the launcher stays a pure measurer and the retained
        window can be revisited without re-profiling.
        """
        payload: dict[str, Any] = {
            "step_times_s": list(self._step_times),
            "timers_s": {
                name: tuple(values) for name, values in self._collective_timers.items()
            },
            "rank_memory": sorted(
                (item for item in rank_memory if item is not None),
                key=lambda item: item["rank"],
            ),
            "microbatch_sequence_lengths": [
                [self._sequence_length] * self._n_seqs_per_rank
            ]
            * self._num_microbatches,
        }
        if self._parameter_count:
            payload["parameter_count"] = int(self._parameter_count)
        if self._observed_mesh is not None:
            payload["observed_mesh"] = self._observed_mesh
        if self._static_allocated_bytes is not None:
            # Diagnostic only. Comparing this against the analytic FSDP inventory is
            # how the 16p/world_size layout is confirmed rather than assumed.
            payload["static_allocated_bytes_at_train_start"] = self._static_allocated_bytes
        return payload


def write_oom_sidecar(
    run_dir: Path,
    spec: dict[str, Any],
    error: BaseException,
) -> None:
    """
    Mark this rank as having run out of memory.

    Every rank writes its own marker and no collective is taken. An out-of-memory
    failure is usually local to one rank while the others sit in a collective, so any
    gather here would block until the communicator times out.

    Out-of-memory candidates are wanted data: they bound the memory model from below.
    The activation shape comes from the candidate itself, so the record stays useful
    without any measurement.
    """
    rank = _rank()
    try:
        payload: dict[str, Any] = {
            "rank": rank,
            "torch_oom": True,
            "sequence_length": int(spec["sequence_length"]),
            "n_seqs_per_rank": int(spec["n_seqs_per_rank"]),
            "message": str(error)[:2000],
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            payload["device_free_bytes"] = int(free_bytes)
            payload["device_total_bytes"] = int(total_bytes)
            payload["device_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        _write_json_atomic(run_dir / f"oom_rank{rank}.json", payload)
    except Exception as marker_error:  # noqa: BLE001 - must never mask the OOM
        print(f"[srl] rank {rank} failed to write its OOM marker: {marker_error!r}")
```

- [ ] **Step 4: Create `launch.py`**

Create `srl/profiler/adapters/fsdp/launch.py`:

```python
"""Torchrun entry point that profiles one FSDP2 training candidate.

verl is not a dependency: it is absent from this environment and depending on it
would pull Ray and its whole worker tree into a simulator repository while pinning
the measurement to a verl version. The FSDP semantics that matter for cost modelling
are the mesh, the wrapping granularity, the mixed-precision policy, and the offload
policy, and all four are torch APIs applied directly here.

Imports the FSDP training APIs at call time, so this module must never be imported
from the adapter package namespace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch.distributed.device_mesh import DeviceMesh

from .execute import OOM_EXIT_CODE

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def _torch_dtype(name: str):
    import torch

    resolved = _DTYPES.get(name.lower())
    if resolved is None:
        raise ValueError(f"Unsupported dtype {name!r}.")
    return getattr(torch, resolved)


def read_spec(path: Path) -> dict[str, Any]:
    """Read one launch spec, refusing a schema this module does not understand."""
    with path.open(encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("schema_version") != 1:
        raise ValueError(f"Unsupported launch spec schema in {path}.")
    return spec


def build_mesh(spec: dict[str, Any]) -> DeviceMesh:
    """
    Build the device mesh one candidate describes.

    A one-dimensional mesh is pure FSDP. A two-dimensional mesh is HSDP: parameters
    are sharded within a row and replicated across rows, which is why the dimension
    names are the ones `fully_shard` recognizes.
    """
    from torch.distributed.device_mesh import init_device_mesh

    replicate = int(spec["mesh"]["replicate_size"])
    shard = int(spec["mesh"]["shard_size"])
    if replicate == 1:
        return init_device_mesh("cuda", (shard,), mesh_dim_names=("dp_shard",))
    return init_device_mesh(
        "cuda", (replicate, shard), mesh_dim_names=("dp_replicate", "dp_shard")
    )


def build_model(spec: dict[str, Any]) -> torch.nn.Module:
    """
    Construct the model from its HuggingFace config, with no weights loaded.

    Weights are irrelevant to a timing and memory measurement, and loading eight
    billion parameters per candidate would dominate the sweep. The architecture is
    what determines both, so the config alone is faithful.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(spec["model_path"])
    override = spec.get("num_layers_override")
    if override is not None:
        config.num_hidden_layers = int(override)
    # Caching is a decode-time concern and would only add allocator noise here.
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(
        config,
        dtype=_torch_dtype(spec["compute_dtype"]),
        attn_implementation=spec["attn_implementation"],
    )
    if spec["gradient_checkpointing"]:
        # Non-reentrant checkpointing is what composes with FSDP2's per-layer hooks.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def apply_fsdp(model: torch.nn.Module, spec: dict[str, Any], mesh: DeviceMesh):
    """
    Shard each transformer block, then the root module.

    Wrapping per block rather than once at the root is what makes the all-gather
    granular enough to overlap with compute; a single root wrap would gather every
    parameter before the first layer runs.
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    policy = MixedPrecisionPolicy(
        param_dtype=_torch_dtype(spec["parameter_dtype"]),
        reduce_dtype=_torch_dtype(spec["gradient_dtype"]),
        cast_forward_inputs=True,
    )
    kwargs: dict[str, Any] = {
        "mesh": mesh,
        "mp_policy": policy,
        "reshard_after_forward": bool(spec["reshard_after_forward"]),
    }
    if spec["optimizer_offload"]:
        kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=True)
    for block in model.model.layers:
        fully_shard(block, **kwargs)
    fully_shard(model, **kwargs)
    return model


def synthetic_batch(spec: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    """
    Build one uniform padded microbatch.

    Profiling measures uniform shapes only; forming ragged microbatches from real
    trajectory lengths is the simulate-side planner's job. `position_ids` therefore
    runs straight through each row rather than restarting, which is what would signal
    a packed batch to the attention kernel.
    """
    import torch

    n_seqs = int(spec["n_seqs_per_rank"])
    length = int(spec["sequence_length"])
    generator = torch.Generator(device="cpu").manual_seed(1234)
    input_ids = torch.randint(
        0, int(spec["vocab_size"]), (n_seqs, length), generator=generator
    ).to(device)
    position_ids = torch.arange(length, device=device).unsqueeze(0).expand(n_seqs, length)
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "position_ids": position_ids.contiguous(),
    }


def _measure_collectives(model, batch, optimizer, measurement) -> None:
    """
    Run one step under the profiler and attribute its collective time.

    NCCL kernels carry the collective's name, verified on this host as
    `ncclDevKernel_AllGather_RING_LL(...)` and
    `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL(...)`, so matching the name after
    stripping underscores and case separates the all-gather that fetches parameters
    from the reduce-scatter that reduces gradients.

    `self_device_time_total` is microseconds. This runs one step out of twelve, so the
    profiler's per-op overhead never reaches the retained timing samples.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as profiler:
        model(**batch).loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    totals = {"all_gather": 0.0, "reduce_scatter": 0.0}
    for event in profiler.key_averages():
        name = event.key.lower().replace("_", "")
        for kind in totals:
            if kind.replace("_", "") in name:
                totals[kind] += float(event.self_device_time_total) / 1e6
    measurement.record_collective_time(
        all_gather_s=totals["all_gather"],
        reduce_scatter_s=totals["reduce_scatter"],
    )


def main() -> None:
    """Profile one candidate and write its measurements next to the spec."""
    parser = argparse.ArgumentParser(
        prog="srl-fsdp-launch",
        description="Profile one FSDP2 training candidate under torchrun.",
    )
    parser.add_argument("--spec", required=True, help="Path to launch_spec.json.")
    args = parser.parse_args()

    spec = read_spec(Path(args.spec))
    run_dir = Path(spec["run_dir"])

    import torch
    import torch.distributed as dist

    from .measure import FsdpMeasurement, write_oom_sidecar

    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    measurement = FsdpMeasurement(run_dir=run_dir, spec=spec)
    try:
        model = build_model(spec).cuda()
        parameter_count = sum(p.numel() for p in model.parameters())
        mesh = build_mesh(spec)
        model = apply_fsdp(model, spec, mesh)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        batch = synthetic_batch(spec, device="cuda")
        measurement.record_topology(
            replicate_size=int(spec["mesh"]["replicate_size"]),
            shard_size=int(spec["mesh"]["shard_size"]),
            parameter_count=parameter_count,
        )
        measurement.on_train_start()
        instrumented = int(spec["warmup_steps"])
        for step in range(int(spec["train_iters"])):
            measurement.step_begin()
            optimizer.zero_grad(set_to_none=True)
            if step == instrumented:
                # Exactly one step carries the collective breakdown. The profiler adds
                # per-op overhead, so instrumenting every step would perturb the very
                # latency being fitted; the first post-warmup step is representative.
                _measure_collectives(model, batch, optimizer, measurement)
            else:
                model(**batch).loss.backward()
                optimizer.step()
            measurement.step_end()
        measurement.on_train_end()
    except torch.OutOfMemoryError as error:
        # Caught by exact type. A driver-level CUDA failure or a NCCL error is a
        # different fault and must stay classified as a failure, because an
        # out-of-memory record can only raise the fitted memory floor.
        write_oom_sidecar(run_dir, spec, error)
        raise SystemExit(OOM_EXIT_CODE) from error
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest tests/test_fsdp_launch.py -q
```

Expected: PASS, 14 tests.

- [ ] **Step 6: Create the smoke config**

Create `examples/profile/fsdp/qwen3_8b_smoke.yaml`:

```yaml
# Two-layer smoke configuration. Proves the launcher, the mesh, and the sidecar work
# before a full sweep spends hours on candidates that would fail the same way.
model: /apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B

hardware:
  gpu_name: H20
  gpu_count: 8
  gpu_memory_mib: 97871
  node_count: 1
  gpus_per_node: 8
  interconnect: NVLink

output_dir: ../../../outputs/profile/qwen3_8b_smoke/fsdp

training:
  compute_dtype: bf16
  optimizer: adam
  parameter_dtype: bf16
  gradient_dtype: fp32
  optimizer_state_dtype: fp32
  # Two layers keep a candidate under a minute while still exercising per-layer
  # wrapping, which a single-layer model would not.
  num_layers_override: 2
  max_steps: 12
  warmup_steps: 4

profiling:
  allow_incomplete: false

environment:
  HF_HUB_OFFLINE: "1"

search_space:
  world_size: 8
  # Both a pure-FSDP and an HSDP mesh, so mesh handling is covered.
  shard_sizes: [4, 8]
  n_seqs_per_rank: [1, 2]
  sequence_lengths: [512, 1024]
  gradient_checkpointing: [true]
  optimizer_offload: [false]
  reshard_after_forward: [true]
```

- [ ] **Step 7: Commit the launcher**

```bash
git add srl/profiler/adapters/fsdp/launch.py srl/profiler/adapters/fsdp/measure.py \
        tests/test_fsdp_launch.py examples/profile/fsdp/qwen3_8b_smoke.yaml
git commit -m "feat: add the FSDP torchrun launcher and its measurement

The launcher owns its training loop rather than delegating to a framework.
verl is absent from this environment, and depending on it would pull Ray and
its worker tree into a simulator repository while pinning the measurement to a
verl version. The four things that matter for cost modelling: the mesh, the
wrapping granularity, the mixed-precision policy, and the offload policy --
are torch APIs applied directly.

Blocks are wrapped individually before the root so the all-gather is granular
enough to overlap with compute. Weights are never loaded: the architecture
determines both timing and memory, and loading eight billion parameters per
candidate would dominate the sweep.

The measurement records the mesh it actually built. A mesh that silently
differs from the requested one would otherwise produce a plausible record
attributed to the wrong topology, which is worse than no record."
```

- [ ] **Step 8: Run the smoke profile on 8 GPUs**

This is the gate. Do not proceed to Task 5 until it passes.

```bash
cd examples/profile/fsdp
# `generate` and `run` require Task 6's CLI; until then drive it directly:
python - <<'PY'
from pathlib import Path
import yaml
from srl.profiler.adapters.fsdp import FsdpProfileConfig, prepare_fsdp_candidate, run_fsdp_candidate
from srl.profiler.fsdp_search_space import enumerate_fsdp_candidates

raw = yaml.safe_load(Path("qwen3_8b_smoke.yaml").read_text())
config = FsdpProfileConfig.from_dict(raw, base_dir=Path.cwd())
search = raw["search_space"]
candidates = enumerate_fsdp_candidates(
    world_size=search["world_size"],
    shard_sizes=search["shard_sizes"],
    n_seqs_per_rank=search["n_seqs_per_rank"],
    sequence_lengths=search["sequence_lengths"],
    gradient_checkpointing=search["gradient_checkpointing"],
    optimizer_offload=search["optimizer_offload"],
    reshard_after_forward=search["reshard_after_forward"],
)
print(f"{len(candidates)} candidates")
run_fsdp_candidate(config, candidates[0])
print("first candidate finished")
PY
```

Expected: exit 0, and the run directory contains `srl_metrics.json` with 12
`step_times_s` entries, 8 `rank_memory` entries, an `observed_mesh` matching the
candidate, and a `parameter_count`.

Verify the measurement is usable before trusting it:

```bash
python - <<'PY'
import json, statistics
from pathlib import Path
metrics = json.loads(next(Path("../../../outputs/profile/qwen3_8b_smoke/fsdp/raw/fsdp").glob("*/srl_metrics.json")).read_text())
retained = metrics["step_times_s"][4:]
median = statistics.median(retained)
spread = statistics.median([abs(v - median) for v in retained]) / median
print(f"retained={len(retained)} median={median*1e3:.1f}ms dispersion={spread:.1%}")
print(f"ranks={len(metrics['rank_memory'])} mesh={metrics['observed_mesh']}")
assert len(retained) == 8, retained
assert spread < 0.05, f"dispersion {spread:.1%} means the measurement, not the model, is the limit"
print("OK")
PY
```

Expected: dispersion below 5%. If it is higher, raise `warmup_steps` before
proceeding: fitting a model to a noisy measurement makes the fit error
unattributable, which is the failure the Megatron work had to undo.

- [ ] **Step 9: Record the smoke result**

If dispersion exceeded 5% or the mesh did not match, stop and report. Otherwise
commit the observed numbers as a fixture for Task 5:

```bash
mkdir -p ../../../tests/fixtures/fsdp
cp $(ls -d ../../../outputs/profile/qwen3_8b_smoke/fsdp/raw/fsdp/*/ | head -1)srl_metrics.json \
   ../../../tests/fixtures/fsdp/srl_metrics.json
cd ../../..
git add tests/fixtures/fsdp/srl_metrics.json
git commit -m "test: capture a measured FSDP sidecar as a collect fixture

Taken from the two-layer smoke run on 8x H20 rather than hand-written, so the
collector is tested against the document the launcher actually produces."
```

---

### Task 5: Collect FSDP records

Turn raw candidate directories into validated `ProfileRecord` objects, and make a
wrong mesh a hard error rather than a plausible number.

**Files:**
- Modify: `srl/profiler/adapters/fsdp/collect.py` (replace the Task 3 placeholder)
- Test: `tests/test_fsdp_collect.py`

**Interfaces:**
- Consumes: `FsdpProfileConfig`, `FsdpCandidate`; `read_status`, `load_sidecar_metrics`, `slice_warmup`, `rank_memory_peaks`, `iter_candidate_dirs` from `adapters/training`.
- Produces:
  - `collect_fsdp_record(config, candidate, *, run_dir=None) -> ProfileRecord`
  - `collect_fsdp_records(config, candidates) -> list[ProfileRecord]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fsdp_collect.py`:

```python
"""Normalizing raw FSDP candidate output into profile records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from srl.profiler.adapters.fsdp import (
    FsdpProfileConfig,
    collect_fsdp_record,
    collect_fsdp_records,
    prepare_fsdp_candidate,
)
from srl.profiler.fsdp_search_space import FsdpCandidate
from srl.profiler.schema import (
    HardwareDescriptor,
    ModelDescriptor,
    ProfileValidationError,
    TrainingPayload,
)


def _model() -> ModelDescriptor:
    return ModelDescriptor(
        name="Qwen3-8B",
        architecture="Qwen3ForCausalLM",
        parameter_count=8_190_735_360,
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        extra={"model_type": "qwen3", "vocab_size": 151936},
    )


def _config(tmp_path: Path, **overrides) -> FsdpProfileConfig:
    defaults = {
        "output_dir": tmp_path,
        "model": _model(),
        "hardware": HardwareDescriptor(
            gpu_name="H20", gpu_count=8, gpu_memory_mib=97871, node_count=1, gpus_per_node=8
        ),
        "model_path": "/models/Qwen3-8B",
    }
    return FsdpProfileConfig(**{**defaults, **overrides})


def _candidate(**overrides) -> FsdpCandidate:
    defaults = {
        "shard_size": 8,
        "replicate_size": 1,
        "n_seqs_per_rank": 2,
        "sequence_length": 1024,
        "gradient_checkpointing": True,
        "optimizer_offload": False,
        "reshard_after_forward": True,
    }
    return FsdpCandidate(**{**defaults, **overrides})


def _sidecar(candidate: FsdpCandidate, **overrides) -> dict:
    defaults = {
        "step_times_s": [0.5, 0.4, 0.35] + [0.30, 0.31, 0.30, 0.29, 0.30, 0.31, 0.30, 0.29, 0.30],
        "timers_s": {},
        "rank_memory": [
            {"rank": rank, "allocated_bytes": 1_000_000 + rank, "reserved_bytes": 2_000_000 + rank}
            for rank in range(candidate.world_size)
        ],
        "microbatch_sequence_lengths": [
            [candidate.sequence_length] * candidate.n_seqs_per_rank
        ],
        "parameter_count": 8_190_735_360,
        "observed_mesh": {
            "replicate_size": candidate.replicate_size,
            "shard_size": candidate.shard_size,
        },
    }
    return {**defaults, **overrides}


def _stage(config, candidate, *, status="success", sidecar=None) -> Path:
    _, run_dir = prepare_fsdp_candidate(config, candidate)
    (run_dir / "status.json").write_text(
        json.dumps({"status": status, "duration_s": 30.0}), encoding="utf-8"
    )
    if sidecar is not None:
        (run_dir / "srl_metrics.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return run_dir


def test_collect_produces_a_training_record_with_the_fsdp_schedule(tmp_path: Path):
    config, candidate = _config(tmp_path), _candidate()
    _stage(config, candidate, sidecar=_sidecar(candidate))
    record = collect_fsdp_record(config, candidate)
    assert record.record_type == "training"
    assert isinstance(record.payload, TrainingPayload)
    assert record.payload.schedule == "fsdp"
    assert record.software.backend == "fsdp"
    assert record.run.status == "success"


def test_collect_slices_the_warmup_steps(tmp_path: Path):
    config, candidate = _config(tmp_path, warmup_steps=4), _candidate()
    _stage(config, candidate, sidecar=_sidecar(candidate))
    record = collect_fsdp_record(config, candidate)
    # Twelve measured steps minus four warmup leaves eight.
    assert len(record.payload.step_times_s) == 8
    assert record.payload.warmup_steps == 4


def test_collect_takes_the_worst_rank_peak(tmp_path: Path):
    config, candidate = _config(tmp_path), _candidate()
    _stage(config, candidate, sidecar=_sidecar(candidate))
    record = collect_fsdp_record(config, candidate)
    assert record.payload.peak_allocated_bytes == 1_000_007
    assert record.payload.peak_reserved_bytes == 2_000_007


def test_collect_reports_dp_equal_to_the_world_size(tmp_path: Path):
    config, candidate = _config(tmp_path), _candidate(shard_size=4, replicate_size=2)
    _stage(config, candidate, sidecar=_sidecar(candidate))
    record = collect_fsdp_record(config, candidate)
    assert record.parallelism.dp == 8
    assert record.hardware.gpu_count == 8


def test_collect_tags_the_mesh_so_the_cell_key_can_read_it(tmp_path: Path):
    config, candidate = _config(tmp_path), _candidate(shard_size=4, replicate_size=2)
    _stage(config, candidate, sidecar=_sidecar(candidate))
    record = collect_fsdp_record(config, candidate)
    assert record.tags["shard_size"] == "4"
    assert record.tags["replicate_size"] == "2"
    assert record.tags["optimizer_offload"] == "False"
    assert record.tags["reshard_after_forward"] == "True"


def test_collect_rejects_a_mesh_that_differs_from_the_candidate(tmp_path: Path):
    """A silently wrong mesh would attribute the measurement to the wrong topology."""
    config, candidate = _config(tmp_path), _candidate(shard_size=8, replicate_size=1)
    sidecar = _sidecar(candidate, observed_mesh={"replicate_size": 2, "shard_size": 4})
    _stage(config, candidate, sidecar=sidecar)
    with pytest.raises(ProfileValidationError, match="mesh"):
        collect_fsdp_record(config, candidate)


def test_collect_uses_the_measured_parameter_count(tmp_path: Path):
    config, candidate = _config(tmp_path), _candidate()
    _stage(config, candidate, sidecar=_sidecar(candidate, parameter_count=1234))
    record = collect_fsdp_record(config, candidate)
    assert record.payload.parameter_count == 1234


def test_collect_describes_an_overridden_layer_count(tmp_path: Path):
    """The memory model derives activation scale from the layer count."""
    config = _config(tmp_path, num_layers_override=2)
    candidate = _candidate()
    _stage(config, candidate, sidecar=_sidecar(candidate, parameter_count=500_000))
    record = collect_fsdp_record(config, candidate)
    assert record.model.num_layers == 2
    assert record.model.parameter_count == 500_000


def test_collect_requires_step_times_for_a_successful_run(tmp_path: Path):
    config, candidate = _config(tmp_path), _candidate()
    _stage(config, candidate, sidecar=_sidecar(candidate, step_times_s=[]))
    with pytest.raises(ProfileValidationError, match="step time"):
        collect_fsdp_record(config, candidate)


def test_collect_tolerates_a_missing_sidecar_when_allowed(tmp_path: Path):
    config = _config(tmp_path, allow_incomplete=True)
    candidate = _candidate()
    _stage(config, candidate)
    record = collect_fsdp_record(config, candidate)
    assert record.payload.step_times_s == ()


def test_collect_keeps_an_oom_candidate_as_a_memory_lower_bound(tmp_path: Path):
    config, candidate = _config(tmp_path), _candidate()
    _, run_dir = prepare_fsdp_candidate(config, candidate)
    (run_dir / "status.json").write_text(
        json.dumps({"status": "failed", "duration_s": 5.0}), encoding="utf-8"
    )
    (run_dir / "oom_rank3.json").write_text(json.dumps({"rank": 3}), encoding="utf-8")
    record = collect_fsdp_record(config, candidate)
    assert record.run.status == "oom"
    assert "rank(s) 3" in record.run.error
    # The activation shape comes from the candidate, so the record is usable unmeasured.
    assert record.payload.microbatches[0].count == candidate.n_seqs_per_rank


def test_collect_records_iterates_every_prepared_candidate(tmp_path: Path):
    config = _config(tmp_path)
    candidates = [_candidate(), _candidate(sequence_length=2048)]
    for candidate in candidates:
        _stage(config, candidate, sidecar=_sidecar(candidate))
    records = collect_fsdp_records(config, candidates)
    assert len(records) == 2
    assert {record.run.run_id for record in records} == {
        f"fsdp-{candidate.candidate_id}" for candidate in candidates
    }


def test_collect_records_raises_on_a_missing_candidate(tmp_path: Path):
    config = _config(tmp_path)
    with pytest.raises(FileNotFoundError):
        collect_fsdp_records(config, [_candidate()])


def test_collect_matches_the_measured_smoke_sidecar():
    """The collector must accept the document the launcher actually writes."""
    fixture = Path("tests/fixtures/fsdp/srl_metrics.json")
    metrics = json.loads(fixture.read_text())
    for key in ("step_times_s", "rank_memory", "microbatch_sequence_lengths", "observed_mesh"):
        assert key in metrics, key
    assert len(metrics["step_times_s"]) == 12
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_fsdp_collect.py -q
```

Expected: FAIL with `NotImplementedError: Implemented in Task 5.`

- [ ] **Step 3: Replace `collect.py`**

Replace `srl/profiler/adapters/fsdp/collect.py` entirely:

```python
"""Normalize raw FSDP candidate output into versioned profile records."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ...fsdp_search_space import FsdpCandidate
from ...schema import (
    ModelDescriptor,
    ProfileRecord,
    ProfileValidationError,
    RankMemory,
    RunDescriptor,
    SequenceStats,
    SoftwareDescriptor,
    TrainingPayload,
)
from ..training import (
    iter_candidate_dirs,
    load_sidecar_metrics,
    rank_memory_peaks,
    read_status,
    slice_warmup,
)
from .config import FsdpProfileConfig

# The schedule identifier for a backend with no pipeline. The schedule axis degenerates
# here rather than taking another value, and Hydraulis filters on it to keep Megatron
# and FSDP records from being fitted together.
FSDP_SCHEDULE = "fsdp"


def _validate_observed_mesh(metrics: dict[str, Any], candidate: FsdpCandidate) -> None:
    """
    Cross-check the mesh the launcher built against the one requested.

    The launcher derives its mesh from the spec, but a mismatch would produce a record
    attributed to the wrong topology, and the cell key is keyed on that topology. This
    is where such a mismatch becomes a hard error instead of a plausible number.
    """
    observed = metrics.get("observed_mesh")
    if observed is None:
        return
    expected = {
        "replicate_size": candidate.replicate_size,
        "shard_size": candidate.shard_size,
    }
    if {key: int(value) for key, value in observed.items()} != expected:
        raise ProfileValidationError(
            f"Launcher built mesh {observed!r} but the candidate requested {expected!r}."
        )


def _effective_model(
    config: FsdpProfileConfig,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> ModelDescriptor:
    """
    Return the architecture that actually trained, not the checkpoint's.

    A layer-count override changes both the parameter count and the activation
    footprint, and the memory model derives its activation term from the layer count.
    A two-layer smoke run described as thirty-six layers would fit an activation scale
    near one eighteenth of the truth.
    """
    override = manifest.get("num_layers_override")
    if override is None:
        return config.model
    measured = metrics.get("parameter_count")
    return replace(
        config.model,
        num_layers=int(override),
        parameter_count=int(measured) if measured else config.model.parameter_count,
    )


def collect_fsdp_record(
    config: FsdpProfileConfig,
    candidate: FsdpCandidate,
    *,
    run_dir: str | Path | None = None,
) -> ProfileRecord:
    """Normalize one candidate's manifest, measurements, and status."""
    source_dir = Path(run_dir) if run_dir is not None else config.raw_root / candidate.candidate_id
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"FSDP manifest does not exist: {manifest_path}.")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    status, duration_s, error = read_status(source_dir)
    metrics = load_sidecar_metrics(source_dir) or {}
    _validate_observed_mesh(metrics, candidate)

    step_times = slice_warmup(
        tuple(float(item) for item in metrics.get("step_times_s", [])),
        config.warmup_steps,
    )
    if status == "success" and not step_times and not config.allow_incomplete:
        raise ProfileValidationError(
            f"No measured step time was found for successful candidate "
            f"{candidate.candidate_id!r}. The launcher must write srl_metrics.json."
        )

    rank_memory = tuple(
        RankMemory.from_dict(item) for item in metrics.get("rank_memory", [])
    )
    if status == "success" and not rank_memory and not config.allow_incomplete:
        raise ProfileValidationError(
            f"No rank memory peak was found for successful candidate "
            f"{candidate.candidate_id!r}. The launcher must write srl_metrics.json."
        )
    peak_allocated, peak_reserved = rank_memory_peaks(rank_memory)

    shapes = metrics.get("microbatch_sequence_lengths") or [list(candidate.sequence_shape)]
    microbatches = tuple(
        SequenceStats.from_lengths([int(item) for item in shape]) for shape in shapes
    )
    if len(microbatches) != candidate.num_microbatches:
        raise ProfileValidationError(
            "Collected microbatch sequence shapes do not match num_microbatches."
        )

    payload = TrainingPayload(
        micro_batch_size=candidate.micro_batch_size,
        global_batch_size=candidate.global_batch_size,
        num_microbatches=candidate.num_microbatches,
        schedule=FSDP_SCHEDULE,
        recompute="full" if candidate.gradient_checkpointing else "none",
        optimizer=config.optimizer,
        microbatches=microbatches,
        step_times_s=step_times,
        timers_s={
            str(name): tuple(float(item) for item in values)
            for name, values in metrics.get("timers_s", {}).items()
        },
        rank_memory=rank_memory,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        parameter_dtype=config.parameter_dtype,
        gradient_dtype=config.gradient_dtype,
        optimizer_state_dtype=config.optimizer_state_dtype,
        # FSDP2 shards optimizer state unconditionally: there is no flag to record,
        # and unlike Megatron it also shards parameters and gradients.
        distributed_optimizer=True,
        parameter_count=(
            int(metrics.get("parameter_count", config.model.parameter_count or 0)) or None
        ),
        warmup_steps=config.warmup_steps,
    )
    return ProfileRecord(
        record_type="training",
        run=RunDescriptor(
            run_id=f"fsdp-{candidate.candidate_id}",
            repeat=candidate.repeat,
            status=status,
            duration_s=duration_s,
            raw_artifact=str(source_dir.relative_to(config.output_dir)),
            error=error,
        ),
        model=_effective_model(config, manifest, metrics),
        hardware=replace(config.hardware, gpu_count=candidate.world_size),
        software=SoftwareDescriptor(
            backend="fsdp",
            backend_version=config.backend_version,
            command=tuple(str(item) for item in manifest["command"]),
        ),
        parallelism=candidate.parallelism,
        payload=payload,
        # The mesh is configuration rather than data partitioning, so it travels as tags
        # and reaches the artifact through the cell selector.
        tags={
            "source": "srl-fsdp2-launcher",
            "shard_size": str(candidate.shard_size),
            "replicate_size": str(candidate.replicate_size),
            "optimizer_offload": str(candidate.optimizer_offload),
            "reshard_after_forward": str(candidate.reshard_after_forward),
            "sequence_parallel": str(candidate.sequence_parallel),
            "attn_implementation": config.attn_implementation,
        },
    )


def collect_fsdp_records(
    config: FsdpProfileConfig,
    candidates: list[FsdpCandidate],
) -> list[ProfileRecord]:
    """Collect all candidate directories that have adapter manifests."""
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    return [
        collect_fsdp_record(config, by_id[candidate_id], run_dir=run_dir)
        for candidate_id, run_dir in iter_candidate_dirs(
            config.raw_root,
            [candidate.candidate_id for candidate in candidates],
            allow_incomplete=config.allow_incomplete,
        )
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_fsdp_collect.py -q
python -m pytest tests/ -q
```

Expected: 14 new tests pass; `674 passed` (660 + 14).

- [ ] **Step 5: Commit**

```bash
git add srl/profiler/adapters/fsdp/collect.py tests/test_fsdp_collect.py
git commit -m "feat: collect FSDP candidate output into profile records

The record reports dp equal to the world size, because HSDP replicates
parameters rather than data and every rank still consumes its own microbatch.
The mesh itself travels as tags, since it is configuration that the artifact
cell selector reads rather than a data partitioning the schema models.

The mesh the launcher built is cross-checked against the mesh requested. That
is the FSDP counterpart of Megatron's resolved-config check: same purpose,
different mechanism, which is why neither could be shared. A mismatch is a
hard error because the cell key is keyed on topology, so a wrong mesh would
otherwise yield a plausible record fitted into the wrong cell.

distributed_optimizer is fixed true and not read from configuration. FSDP2
shards optimizer state unconditionally, and there is no flag that could
disable it."
```

---

### Task 6: The microbatch planner and the FSDP memory inventory

The simulate side. This is where packing lives, and where the measured `16p/ws`
layout becomes an analytic term.

**Files:**
- Create: `srl/cost/training/microbatch.py`
- Create: `srl/cost/training/fsdp_inventory.py`
- Test: `tests/test_microbatch_planner.py`
- Test: `tests/test_fsdp_inventory.py`

**Interfaces:**
- Consumes: `SequenceStats` from `srl/profiler/schema.py`; `TrainingParallelism`, `dtype_bytes` from `srl/cost/training/`.
- Produces:
  - `BatchingMode` enum with members `PAD_TO_MAX = "pad_to_max"`, `TOKEN_BUDGET = "token_budget"`, `PACKED_NO_PAD = "packed_no_pad"`
  - `MicrobatchPlanner(mode: BatchingMode, *, max_tokens_per_rank: int | None = None)` with `plan(sequence_lengths: tuple[int, ...], dp: int) -> tuple[SequenceStats, ...]`
  - `verl_workload(length: int) -> int` returning `24576 * length + length ** 2`
  - `FsdpInventory` frozen dataclass with `model_parameters: float`, `gradients: float`, `optimizer_moments: float`, `compute_element_bytes: int`, `offloaded_bytes: float`; property `total_bytes: float`; method `to_dict() -> dict[str, float]`. **There is no `master_parameters` field.** AdamW's moments inherit the module's dtype, so the launcher builds the module in the master dtype (`optimizer_state_dtype`) and there is exactly one resident parameter copy. `parameter_dtype` denotes the transient all-gathered compute copy, which `reshard_after_forward=True` frees per block; `compute_element_bytes` records its per-element width for diagnostics and is excluded from `total_bytes`, which is invariant to `parameter_dtype`. Verified: `16p/world_size` exactly, ratio 0.979 against a measured 3.101 GiB on 8xH20.
  - `fsdp_inventory(*, parameter_count: int, world_size: int, parameter_dtype: str, gradient_dtype: str, optimizer_state_dtype: str, optimizer_offload: bool) -> FsdpInventory`

- [ ] **Step 1: Write the failing planner test**

Create `tests/test_microbatch_planner.py`:

```python
"""Turning real trajectory lengths into a microbatch plan."""

from __future__ import annotations

import pytest

from srl.cost.training.microbatch import BatchingMode, MicrobatchPlanner, verl_workload


def test_verl_workload_matches_the_reference_proxy():
    """verl balances on `24576 * L + L^2`; diverging would skew every predicted count."""
    assert verl_workload(1000) == 24576 * 1000 + 1000**2


def test_pad_to_max_charges_for_the_padding():
    planner = MicrobatchPlanner(BatchingMode.PAD_TO_MAX)
    plan = planner.plan((100, 200, 300, 400), dp=2)
    # Two ranks, so each holds two sequences in one microbatch.
    assert len(plan) == 1
    # The longest sequence on the critical rank sets every slot's cost.
    assert plan[0].count == 2
    assert plan[0].sum_tokens == plan[0].max_tokens * plan[0].count


def test_packed_no_pad_charges_only_real_tokens():
    planner = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD)
    plan = planner.plan((100, 200, 300, 400), dp=2)
    assert sum(item.sum_tokens for item in plan) < 400 * 2
    # Karmarkar-Karp pairs the longest with the shortest, giving 400 + 100 on the
    # critical rank against 300 + 200 on the other. Both total 500; the critical rank
    # is the one with the greater quadratic workload.
    assert plan[0].sum_tokens == 500
    assert plan[0].count == 2


def test_packed_beats_padded_on_a_skewed_distribution():
    """This is the 1.17x-to-4.34x measured gap; the plan must express it."""
    lengths = (10240, 10240, 900, 1100, 950, 1050, 800, 1200)
    padded = MicrobatchPlanner(BatchingMode.PAD_TO_MAX).plan(lengths, dp=2)
    packed = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD).plan(lengths, dp=2)
    padded_tokens = sum(item.sum_tokens for item in padded)
    packed_tokens = sum(item.sum_tokens for item in packed)
    assert packed_tokens < padded_tokens
    assert padded_tokens / packed_tokens > 1.5


def test_token_budget_splits_when_the_cap_is_exceeded():
    planner = MicrobatchPlanner(BatchingMode.TOKEN_BUDGET, max_tokens_per_rank=1000)
    plan = planner.plan((400, 400, 400, 400), dp=1)
    assert len(plan) >= 2
    assert all(item.sum_tokens <= 1000 for item in plan)


def test_token_budget_keeps_one_microbatch_when_it_fits():
    planner = MicrobatchPlanner(BatchingMode.TOKEN_BUDGET, max_tokens_per_rank=10_000)
    plan = planner.plan((400, 400), dp=1)
    assert len(plan) == 1


def test_token_budget_is_deterministic():
    """A nondeterministic plan would make a prediction irreproducible."""
    planner = MicrobatchPlanner(BatchingMode.TOKEN_BUDGET, max_tokens_per_rank=2000)
    lengths = (900, 1100, 700, 1300, 500, 1500)
    first = planner.plan(lengths, dp=2)
    second = planner.plan(lengths, dp=2)
    assert [(i.count, i.sum_tokens) for i in first] == [(i.count, i.sum_tokens) for i in second]


def test_token_budget_balances_rather_than_filling_greedily():
    """Karmarkar-Karp balances workload across microbatches; a greedy fill does not."""
    planner = MicrobatchPlanner(BatchingMode.TOKEN_BUDGET, max_tokens_per_rank=2100)
    plan = planner.plan((2000, 100, 2000, 100), dp=1)
    # Balancing pairs each long sequence with a short one, so both groups hold two.
    # A greedy fill would put the two 2000s together and leave a nearly empty group.
    assert all(item.count == 2 for item in plan)
    # Padding to the group maximum is why this costs 2 * 2000 rather than 2100: the
    # budget selects the microbatch *count*, and padding then charges for the short
    # sequence as if it were long. Only packed_no_pad avoids that.
    assert sorted(item.sum_tokens for item in plan) == [4000, 4000]


def test_token_budget_requires_a_cap():
    with pytest.raises(ValueError, match="max_tokens_per_rank"):
        MicrobatchPlanner(BatchingMode.TOKEN_BUDGET)


def test_a_single_sequence_longer_than_the_cap_still_yields_one_microbatch():
    """A sequence cannot be split, so the cap is a target rather than a guarantee."""
    planner = MicrobatchPlanner(BatchingMode.TOKEN_BUDGET, max_tokens_per_rank=100)
    plan = planner.plan((5000,), dp=1)
    assert len(plan) == 1
    assert plan[0].sum_tokens == 5000


def test_plan_uses_the_busiest_rank():
    """The step ends when the slowest rank ends, so the plan describes that rank."""
    planner = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD)
    plan = planner.plan((1000, 10, 10, 10), dp=2)
    assert max(item.max_tokens for item in plan) == 1000


def test_plan_rejects_a_batch_smaller_than_the_dp_degree():
    planner = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD)
    with pytest.raises(ValueError, match="dp"):
        planner.plan((100,), dp=4)


def test_plan_rejects_empty_lengths():
    planner = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD)
    with pytest.raises(ValueError):
        planner.plan((), dp=1)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/test_microbatch_planner.py -q
```

Expected: `ModuleNotFoundError: No module named 'srl.cost.training.microbatch'`.

- [ ] **Step 3: Create `microbatch.py`**

Create `srl/cost/training/microbatch.py`:

```python
"""Turn one global batch's real trajectory lengths into a microbatch plan.

Profiling measures uniform shapes, so this is the only place that knows how a real
RL batch becomes microbatches. Keeping it here rather than in the launcher means the
profile grid stays clean and there is exactly one implementation to keep honest.

The three modes span a measured 1.17x to 4.34x latency range on DAPO-like bimodal
lengths, so the mode is a first-class input to the cost model rather than a detail.
"""

from __future__ import annotations

from enum import Enum

from srl.profiler.schema import SequenceStats

# verl's balancing proxy: attention grows with the square of the length while the rest
# of the layer grows linearly, and 24576 is the linear weight it uses. Reproduced
# exactly, because a different proxy would partition differently and every predicted
# microbatch count would then be systematically wrong.
_VERL_LINEAR_WEIGHT = 24576


def verl_workload(length: int) -> int:
    """Return verl's per-sequence workload proxy."""
    return _VERL_LINEAR_WEIGHT * length + length**2


class BatchingMode(Enum):
    """How a rank's sequences are grouped into microbatches."""

    # Every microbatch is padded to its longest sequence. Pad tokens cost full compute.
    PAD_TO_MAX = "pad_to_max"
    # Grouped to a token budget, still padded within a microbatch. verl's default.
    TOKEN_BUDGET = "token_budget"
    # Concatenated with no pad tokens, priced on true token counts.
    PACKED_NO_PAD = "packed_no_pad"


def _balanced_partitions(lengths: list[int], count: int) -> list[list[int]]:
    """
    Partition indices into `count` groups of similar workload.

    Karmarkar-Karp differencing, matching verl. A greedy fill would leave the last
    group short and make the critical microbatch depend on input order.
    """
    import heapq

    if count <= 1:
        return [list(range(len(lengths)))]
    # Largest first, so the heaviest sequences are placed while groups are still empty.
    order = sorted(range(len(lengths)), key=lambda index: -verl_workload(lengths[index]))
    heap = [(0, group, []) for group in range(count)]
    heapq.heapify(heap)
    for index in order:
        total, group, members = heapq.heappop(heap)
        members.append(index)
        heapq.heappush(heap, (total + verl_workload(lengths[index]), group, members))
    return [sorted(members) for _, _, members in sorted(heap, key=lambda item: item[1])]


class MicrobatchPlanner:
    """Build the microbatch plan for the data-parallel rank that ends last."""

    def __init__(self, mode: BatchingMode, *, max_tokens_per_rank: int | None = None):
        if mode is BatchingMode.TOKEN_BUDGET and not max_tokens_per_rank:
            raise ValueError("max_tokens_per_rank is required for the token_budget mode.")
        self.mode = mode
        self.max_tokens_per_rank = max_tokens_per_rank

    def plan(self, sequence_lengths: tuple[int, ...], dp: int) -> tuple[SequenceStats, ...]:
        """
        Return the critical rank's microbatches.

        A training step ends when its slowest rank ends, so the plan describes the rank
        carrying the most work rather than an average one.
        """
        if not sequence_lengths:
            raise ValueError("A microbatch plan needs at least one sequence.")
        if dp <= 0:
            raise ValueError("dp must be positive.")
        if len(sequence_lengths) < dp:
            raise ValueError(
                f"A batch of {len(sequence_lengths)} sequences cannot fill dp={dp} ranks."
            )
        ranks = _balanced_partitions(list(sequence_lengths), dp)
        rank_lengths = [
            [sequence_lengths[index] for index in members] for members in ranks
        ]
        critical = max(
            rank_lengths,
            key=lambda lengths: sum(verl_workload(length) for length in lengths),
        )
        return self._microbatches(critical)

    def _microbatches(self, lengths: list[int]) -> tuple[SequenceStats, ...]:
        if self.mode is BatchingMode.PAD_TO_MAX:
            # One microbatch, every sequence padded to the longest. Pad tokens are real
            # compute, so they are charged as if they carried data.
            padded = max(lengths)
            return (SequenceStats.from_lengths([padded] * len(lengths)),)
        if self.mode is BatchingMode.PACKED_NO_PAD:
            return (SequenceStats.from_lengths(lengths),)
        return self._token_budget_microbatches(lengths)

    def _token_budget_microbatches(self, lengths: list[int]) -> tuple[SequenceStats, ...]:
        """
        Group to the token budget, then pad within each group.

        The count comes from the true token total, but a sequence cannot be split, so a
        single sequence longer than the budget yields one oversized microbatch. That is
        verl's behaviour too: the budget is a target, not a guarantee.
        """
        budget = int(self.max_tokens_per_rank or 0)
        total = sum(lengths)
        count = max(1, -(-total // budget))
        count = min(count, len(lengths))
        groups = _balanced_partitions(lengths, count)
        microbatches = []
        for members in groups:
            group = [lengths[index] for index in members]
            if not group:
                continue
            padded = max(group)
            microbatches.append(SequenceStats.from_lengths([padded] * len(group)))
        return tuple(microbatches)
```

- [ ] **Step 4: Run the planner tests**

```bash
python -m pytest tests/test_microbatch_planner.py -q
```

Expected: PASS, 13 tests. If `test_token_budget_balances_rather_than_filling_greedily`
fails, the partitioner is filling greedily: check that the heap is keyed on
accumulated workload, not on group index.

- [ ] **Step 5: Write the failing inventory test**
> **SUPERSEDED: do not transcribe.** The code and tests below encode a `master_parameters`
> field and a narrow-params-plus-master layout. That is Megatron's mixed-precision shape, not
> FSDP2's. AdamW's moments inherit the module's dtype, so the launcher builds the module in
> the master dtype and there is exactly one resident parameter copy. Task 6 shipped the
> corrected version; read `srl/cost/training/fsdp_inventory.py` and
> `tests/test_fsdp_inventory.py` for the delivered interface. Retained here only as the
> historical record of what the plan originally asked for.


Create `tests/test_fsdp_inventory.py`:

```python
"""The FSDP2 static memory layout, measured rather than assumed."""

from __future__ import annotations

import pytest

from srl.cost.training.fsdp_inventory import fsdp_inventory


def test_everything_is_sharded_giving_sixteen_p_over_world_size():
    """Measured on 4 ranks with a 1.611B model: 6.078 GiB against 16p/ws = 6.000 GiB."""
    parameter_count = 1_611_000_000
    inventory = fsdp_inventory(
        parameter_count=parameter_count,
        world_size=4,
        parameter_dtype="fp32",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    )
    expected = 16 * parameter_count / 4
    assert inventory.total_bytes == pytest.approx(expected, rel=1e-9)


def test_gradients_are_sharded_unlike_megatron():
    """
    Megatron holds a full-size fp32 gradient buffer as the reduce-scatter source.

    FSDP2 reduce-scatters per layer into already-sharded gradients, so no full-size
    buffer exists. Reusing the Megatron inventory here would overstate memory by
    roughly twofold.
    """
    parameter_count = 1_000_000_000
    inventory = fsdp_inventory(
        parameter_count=parameter_count,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    )
    assert inventory.gradients == pytest.approx(4 * parameter_count / 8)


def test_mixed_precision_keeps_a_master_copy():
    parameter_count = 1_000_000_000
    inventory = fsdp_inventory(
        parameter_count=parameter_count,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    )
    assert inventory.model_parameters == pytest.approx(2 * parameter_count / 8)
    assert inventory.master_parameters == pytest.approx(4 * parameter_count / 8)
    # Adam keeps two fp32 moments.
    assert inventory.optimizer_moments == pytest.approx(8 * parameter_count / 8)


def test_pure_fp32_needs_no_separate_master_copy():
    inventory = fsdp_inventory(
        parameter_count=1_000_000_000,
        world_size=8,
        parameter_dtype="fp32",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    )
    assert inventory.master_parameters == 0


def test_offloading_moves_optimizer_state_off_device():
    parameter_count = 1_000_000_000
    resident = fsdp_inventory(
        parameter_count=parameter_count,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    )
    offloaded = fsdp_inventory(
        parameter_count=parameter_count,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=True,
    )
    assert offloaded.total_bytes < resident.total_bytes
    assert offloaded.optimizer_moments == 0
    assert offloaded.master_parameters == 0
    assert offloaded.offloaded_bytes == pytest.approx(
        resident.master_parameters + resident.optimizer_moments
    )


def test_larger_shard_degree_reduces_per_rank_bytes():
    def total(world_size: int) -> float:
        return fsdp_inventory(
            parameter_count=8_000_000_000,
            world_size=world_size,
            parameter_dtype="bf16",
            gradient_dtype="fp32",
            optimizer_state_dtype="fp32",
            optimizer_offload=False,
        ).total_bytes

    assert total(8) == pytest.approx(total(4) / 2)


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        fsdp_inventory(
            parameter_count=0,
            world_size=8,
            parameter_dtype="bf16",
            gradient_dtype="fp32",
            optimizer_state_dtype="fp32",
            optimizer_offload=False,
        )
    with pytest.raises(ValueError):
        fsdp_inventory(
            parameter_count=1000,
            world_size=0,
            parameter_dtype="bf16",
            gradient_dtype="fp32",
            optimizer_state_dtype="fp32",
            optimizer_offload=False,
        )
```

- [ ] **Step 6: Create `fsdp_inventory.py`**

Create `srl/cost/training/fsdp_inventory.py`:

```python
"""Static parameter, gradient, and optimizer state inventory for FSDP2.

Separate from `inventory.py` because the sharding rule genuinely differs, and the
difference is a factor of two rather than a detail.

Megatron's distributed optimizer shards optimizer *state* only: parameters are held
at full model-parallel-local size and the fp32 gradient buffer is the reduce-scatter
source, so every rank holds it whole. Measured at Qwen3-8B TP1/PP1/DP8 that is
`2p + 4p + 12p/DP`, about 59 GiB.

FSDP2 shards all three. Each `fully_shard` unit reduce-scatters into an
already-sharded gradient, so no full-size buffer exists anywhere. Measured on 4 ranks
with a 1.611B-parameter model under `MixedPrecisionPolicy(bf16, fp32)` and AdamW, the
allocator reported 6.078 GiB against `16p/world_size = 6.000 GiB`: parameters,
gradients, master weights, and both Adam moments all divided by the world size.

Applying the Megatron inventory to FSDP would therefore overstate memory roughly
twofold, which is exactly the kind of error a fitted residual absorbs silently.
"""

from __future__ import annotations

from dataclasses import dataclass

from .inventory import dtype_bytes


@dataclass(frozen=True)
class FsdpInventory:
    """Break down per-rank static state bytes under FSDP2."""

    model_parameters: float
    gradients: float
    master_parameters: float
    optimizer_moments: float
    # Bytes moved to host memory by CPUOffloadPolicy. Tracked rather than dropped so a
    # prediction can explain why an offloaded cell needs less device memory.
    offloaded_bytes: float = 0.0

    @property
    def total_bytes(self) -> float:
        """Return the device-resident total."""
        return (
            self.model_parameters
            + self.gradients
            + self.master_parameters
            + self.optimizer_moments
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "model_parameters": self.model_parameters,
            "gradients": self.gradients,
            "master_parameters": self.master_parameters,
            "optimizer_moments": self.optimizer_moments,
            "offloaded_bytes": self.offloaded_bytes,
            "total_bytes": self.total_bytes,
        }


def fsdp_inventory(
    *,
    parameter_count: int,
    world_size: int,
    parameter_dtype: str,
    gradient_dtype: str,
    optimizer_state_dtype: str,
    optimizer_offload: bool,
) -> FsdpInventory:
    """
    Compute per-rank static state bytes under FSDP2.

    Every term is divided by the world size. `world_size` is the whole mesh, both its
    shard and replicate extents: replication duplicates a shard across rows rather than
    widening it, so a rank's share depends only on how many ranks there are in total.
    """
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive.")
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    local = parameter_count / world_size
    parameter_bytes = dtype_bytes(parameter_dtype)
    gradient_bytes = dtype_bytes(gradient_dtype)
    optimizer_bytes = dtype_bytes(optimizer_state_dtype)
    # A separate master copy exists only when parameters are kept narrower than the
    # optimizer's arithmetic.
    master_bytes = optimizer_bytes if parameter_bytes < optimizer_bytes else 0
    # Adam keeps a first and a second moment.
    moment_bytes = 2 * optimizer_bytes
    master = local * master_bytes
    moments = local * moment_bytes
    if optimizer_offload:
        # CPUOffloadPolicy keeps the sharded optimizer state and its master copy in
        # pinned host memory; the parameter and gradient shards stay resident because
        # forward and backward need them.
        return FsdpInventory(
            model_parameters=local * parameter_bytes,
            gradients=local * gradient_bytes,
            master_parameters=0.0,
            optimizer_moments=0.0,
            offloaded_bytes=master + moments,
        )
    return FsdpInventory(
        model_parameters=local * parameter_bytes,
        gradients=local * gradient_bytes,
        master_parameters=master,
        optimizer_moments=moments,
    )
```

- [ ] **Step 7: Run both test files**

```bash
python -m pytest tests/test_microbatch_planner.py tests/test_fsdp_inventory.py -q
python -m pytest tests/ -q
```

Expected: 20 new tests pass; `694 passed` (674 + 20).

- [ ] **Step 8: Commit**

```bash
git add srl/cost/training/microbatch.py srl/cost/training/fsdp_inventory.py \
        tests/test_microbatch_planner.py tests/test_fsdp_inventory.py
git commit -m "feat: add the microbatch planner and the FSDP memory inventory

The planner is the only place that knows how a real RL batch becomes
microbatches, because profiling measures uniform shapes. Its three modes span
a measured 1.17x to 4.34x range on DAPO-like bimodal lengths, so the mode is
an input to the cost model rather than a detail. Balancing reproduces verl's
Karmarkar-Karp differencing on 24576*L + L^2 exactly: a different proxy would
partition differently and skew every predicted microbatch count.

The inventory is separate from Megatron's because the sharding rule differs by
a factor of two. Megatron's distributed optimizer shards state only, holding a
full-size fp32 gradient buffer as the reduce-scatter source. FSDP2 shards
parameters, gradients, and optimizer state alike: measured at 16p over the
world size, 6.078 GiB against 6.000 GiB predicted, so reusing the Megatron
term would overstate FSDP memory twofold and a fitted residual would hide it."
```

---

### Task 7: The FSDP cost model

**Files:**
- Create: `srl/cost/training/fsdp.py`
- Modify: `srl/cost/training/__init__.py`
- Test: `tests/test_fsdp_cost.py`

**Interfaces:**
- Consumes: `latency_features`, `pipeline_slots`, `robust_step_time`, `step_time_dispersion` from `srl/cost/training/latency.py`; `fsdp_inventory` (Task 6); `MicrobatchPlanner`, `BatchingMode` (Task 6); `CostArtifact`, `ArtifactEntry`, `FitMetrics` from `srl/cost/artifact.py`; `profile_digest` from `srl/profiler/io.py`.
- Produces:
  - `FsdpLatencyModel(artifact, *, shard_size, replicate_size, recompute, optimizer_offload, reshard_after_forward, parameter_dtype, gradient_dtype, optimizer_state_dtype, planner, allow_extrapolation=False)` implementing `TrainingLatencyModel`
  - `FsdpMemoryModel(...)` with the same constructor shape, implementing `TrainingMemoryModel`
  - `FsdpTrainingCost` with `model_type = "fsdp"`, `fit(records, *, validation_fraction=0.2, require_full_rank=True) -> CostArtifact`, and `from_artifact(artifact_or_path, *, batching_mode="pad_to_max", max_tokens_per_rank=None, allow_extrapolation=False, **selectors) -> FsdpTrainingCost`
  - `TRAINING_MODEL_REGISTRY["fsdp"] = FsdpTrainingCost`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fsdp_cost.py`:

```python
"""Fitting and evaluating the FSDP training cost model."""

from __future__ import annotations

from pathlib import Path

import pytest

from srl.cost.artifact import CostArtifact
from srl.cost.training import TRAINING_MODEL_REGISTRY
from srl.cost.training.base import (
    TrainingOutOfDomainError,
    TrainingParallelism,
    TrainingWorkload,
)
from srl.cost.training.fsdp import FsdpTrainingCost
from srl.cost.training.fsdp_inventory import fsdp_inventory
from srl.profiler.schema import (
    HardwareDescriptor,
    ModelDescriptor,
    Parallelism,
    ProfileRecord,
    RankMemory,
    RunDescriptor,
    SequenceStats,
    SoftwareDescriptor,
    TrainingPayload,
)

PARAMETER_COUNT = 8_190_735_360
# A synthetic ground truth so a fitted coefficient can be checked against a known value.
PER_STEP = 0.20
TOKEN_COST = 5e-6
MICROBATCH_OVERHEAD = 0.05


def _record(
    *,
    n_seqs: int,
    length: int,
    num_microbatches: int,
    shard_size: int = 8,
    replicate_size: int = 1,
    offload: bool = False,
    status: str = "success",
    peak_allocated: int | None = None,
    run_id: str | None = None,
) -> ProfileRecord:
    world_size = shard_size * replicate_size
    microbatches = tuple(
        SequenceStats.from_lengths([length] * n_seqs) for _ in range(num_microbatches)
    )
    tokens = float(max(item.sum_tokens for item in microbatches))
    slots = float(num_microbatches)
    step = PER_STEP + slots * (TOKEN_COST * tokens + MICROBATCH_OVERHEAD)
    static = fsdp_inventory(
        parameter_count=PARAMETER_COUNT,
        world_size=world_size,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=offload,
    ).total_bytes
    allocated = peak_allocated if peak_allocated is not None else int(static + 1000.0 * tokens)
    return ProfileRecord(
        record_type="training",
        run=RunDescriptor(
            run_id=run_id or f"fsdp-{n_seqs}-{length}-{num_microbatches}-{shard_size}-{offload}",
            repeat=0,
            status=status,
            duration_s=10.0,
        ),
        model=ModelDescriptor(
            name="Qwen3-8B",
            architecture="Qwen3ForCausalLM",
            parameter_count=PARAMETER_COUNT,
            num_layers=36,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,
        ),
        hardware=HardwareDescriptor(
            gpu_name="H20",
            gpu_count=world_size,
            gpu_memory_mib=97871,
            node_count=1,
            gpus_per_node=8,
        ),
        software=SoftwareDescriptor(backend="fsdp", backend_version="2.11.0"),
        parallelism=Parallelism(tp=1, pp=1, dp=world_size, cp=1, world_size=world_size),
        payload=TrainingPayload(
            micro_batch_size=n_seqs,
            global_batch_size=n_seqs * world_size * num_microbatches,
            num_microbatches=num_microbatches,
            schedule="fsdp",
            recompute="full",
            optimizer="adam",
            microbatches=microbatches,
            step_times_s=(step, step * 1.001, step * 0.999) if status == "success" else (),
            rank_memory=(
                (RankMemory(rank=0, allocated_bytes=allocated, reserved_bytes=allocated + 1),)
                if status == "success"
                else ()
            ),
            peak_allocated_bytes=allocated if status == "success" else None,
            peak_reserved_bytes=allocated + 1 if status == "success" else None,
            parameter_dtype="bf16",
            gradient_dtype="fp32",
            optimizer_state_dtype="fp32",
            parameter_count=PARAMETER_COUNT,
            warmup_steps=4,
        ),
        tags={
            "shard_size": str(shard_size),
            "replicate_size": str(replicate_size),
            "optimizer_offload": str(offload),
            "reshard_after_forward": "True",
            "sequence_parallel": "1",
        },
    )


def _records() -> list[ProfileRecord]:
    """Vary both axes so the design reaches rank three."""
    records = []
    for length in (512, 1024, 2048):
        for num_microbatches in (1, 2, 4):
            records.append(
                _record(n_seqs=1, length=length, num_microbatches=num_microbatches)
            )
    return records


def _cost(**overrides) -> FsdpTrainingCost:
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    defaults = {"batching_mode": "pad_to_max"}
    return FsdpTrainingCost.from_artifact(artifact, **{**defaults, **overrides})


def test_registered_under_the_fsdp_name():
    assert TRAINING_MODEL_REGISTRY["fsdp"] is FsdpTrainingCost


def test_fit_recovers_the_known_coefficients():
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "latency")
    assert entry.coefficients["per_step_overhead"] == pytest.approx(PER_STEP, rel=0.02)
    assert entry.coefficients["token_cost"] == pytest.approx(TOKEN_COST, rel=0.02)
    assert entry.coefficients["microbatch_overhead"] == pytest.approx(
        MICROBATCH_OVERHEAD, rel=0.02
    )


def test_fit_records_no_pipeline_term():
    """FSDP has no pipeline, so slots reduce to the microbatch count."""
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "latency")
    assert "c_pipeline" not in entry.coefficients
    assert entry.diagnostics["slots"] == "num_microbatches"


def test_artifact_round_trips(tmp_path: Path):
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    path = artifact.save(tmp_path / "fsdp.json")
    reloaded = CostArtifact.load(path)
    assert reloaded.kind == "training"
    assert reloaded.model_type == "fsdp"
    assert len(reloaded.entries) == len(artifact.entries)


def test_predicts_a_known_step_time():
    cost = _cost()
    # One sequence per rank on 8 ranks, so dp=8 and each rank holds one 1024 sequence.
    workload = TrainingWorkload(
        sequence_lengths=(1024,) * 8,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    estimate = cost.estimate(workload, TrainingParallelism(dp=8))
    expected = PER_STEP + 1.0 * (TOKEN_COST * 1024 + MICROBATCH_OVERHEAD)
    assert estimate.latency_s == pytest.approx(expected, rel=0.05)


def test_memory_uses_the_fsdp_inventory_not_megatron_s():
    cost = _cost()
    workload = TrainingWorkload(
        sequence_lengths=(1024,) * 8,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    estimate = cost.estimate(workload, TrainingParallelism(dp=8))
    static = fsdp_inventory(
        parameter_count=PARAMETER_COUNT,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    ).total_bytes
    assert estimate.peak_memory_bytes >= static
    # Megatron's inventory would be roughly twice this, so a factor-two error is caught.
    assert estimate.peak_memory_bytes < static * 1.6


def test_memory_target_selection_is_recorded():
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "memory")
    diagnostics = entry.diagnostics
    assert diagnostics["memory_metric"] in ("peak_allocated_bytes", "peak_reserved_bytes")
    # Both candidates must be reported so the choice is auditable.
    assert "peak_allocated_bytes" in diagnostics["candidate_r_squared"]
    assert "peak_reserved_bytes" in diagnostics["candidate_r_squared"]


def test_memory_prediction_is_compared_against_reserved():
    """`allocated` excludes fragmentation, so an OOM check needs the reserved ratio."""
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "memory")
    assert entry.diagnostics["reserved_over_allocated"] >= 1.0


def test_batching_mode_changes_the_prediction():
    lengths = (10240, 10240, 900, 1100, 950, 1050, 800, 1200)
    workload = TrainingWorkload(
        sequence_lengths=lengths,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    padded = _cost(batching_mode="pad_to_max", allow_extrapolation=True)
    packed = _cost(batching_mode="packed_no_pad", allow_extrapolation=True)
    parallelism = TrainingParallelism(dp=2)
    padded_latency = padded.estimate(workload, parallelism).latency_s
    packed_latency = packed.estimate(workload, parallelism).latency_s
    assert packed_latency < padded_latency


def test_rejects_a_query_outside_the_profiled_range():
    cost = _cost()
    workload = TrainingWorkload(
        sequence_lengths=(65536,) * 8,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    with pytest.raises(TrainingOutOfDomainError):
        cost.estimate(workload, TrainingParallelism(dp=8))


def test_rejects_a_rank_deficient_design():
    """Sequence length alone caps the design at rank two."""
    records = [_record(n_seqs=1, length=length, num_microbatches=1) for length in (512, 1024, 2048)]
    with pytest.raises(ValueError, match="rank"):
        FsdpTrainingCost.fit(records, validation_fraction=0.0)


def test_oom_records_raise_the_memory_floor():
    records = _records()
    oom = _record(
        n_seqs=8,
        length=2048,
        num_microbatches=4,
        status="oom",
        run_id="fsdp-oom",
    )
    artifact = FsdpTrainingCost.fit([*records, oom], validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "memory")
    assert entry.diagnostics["oom_constraints"] == 1


def test_rejects_megatron_records():
    """A 1F1B record must never be fitted by the FSDP model."""
    record = _record(n_seqs=1, length=512, num_microbatches=1)
    megatron = ProfileRecord(
        record_type="training",
        run=record.run,
        model=record.model,
        hardware=record.hardware,
        software=record.software,
        parallelism=record.parallelism,
        payload=TrainingPayload(
            micro_batch_size=1,
            global_batch_size=8,
            num_microbatches=1,
            schedule="1f1b",
            recompute="full",
            optimizer="adam",
            microbatches=record.payload.microbatches,
            step_times_s=(1.0,),
            parameter_count=PARAMETER_COUNT,
        ),
        tags=record.tags,
    )
    with pytest.raises(ValueError, match="fsdp"):
        FsdpTrainingCost.fit([megatron], validation_fraction=0.0)


def test_satisfies_the_contract_the_simulator_calls():
    """
    The simulator's only requirement is `estimate(workload, parallelism)`.

    It reads `latency_s` to advance the clock and `peak_memory_bytes` to flag OOM
    against configured capacity, so both must be positive and finite for a workload
    built the way `Simulator._train` builds one.
    """
    import math

    from srl.cost.training.base import TrainingCostEstimate

    cost = _cost()
    workload = TrainingWorkload(
        sequence_lengths=(1024,) * 8,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    estimate = cost.estimate(workload, TrainingParallelism(dp=8))
    assert isinstance(estimate, TrainingCostEstimate)
    assert math.isfinite(estimate.latency_s) and estimate.latency_s > 0
    assert isinstance(estimate.peak_memory_bytes, int)
    assert estimate.peak_memory_bytes > 0
    # The diagnostics the simulator writes to telemetry must name the mesh.
    assert estimate.diagnostics["dp"] == 8
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/test_fsdp_cost.py -q
```

Expected: `ModuleNotFoundError: No module named 'srl.cost.training.fsdp'`.

- [ ] **Step 3: Implement `fsdp.py`**

Create `srl/cost/training/fsdp.py`. Model it closely on
`srl/cost/training/hydraulis.py`, which is the working reference for cell grouping,
`_scaled_nonnegative_fit`, OOM inequality handling, and `_check_feature`. The
differences to implement:

1. **Cell key.** Group on `(shard_size, replicate_size, recompute, optimizer,
   parameter_dtype, gradient_dtype, optimizer_state_dtype, optimizer_offload,
   reshard_after_forward, parameter_count)`, reading the mesh and flags from
   `record.tags`. Reject any record whose `payload.schedule.lower() != "fsdp"` with a
   message naming `fsdp`, and any record with `sequence_parallel != "1"`.
2. **Latency design.** Use `latency_features(payload.microbatches,
   TrainingParallelism(dp=world_size))` and `LatencyFeatures.design_row()`, which
   yields `[1, slots * tokens, slots]`. Fit with `_scaled_nonnegative_fit`. Require
   `rank >= 3` when `require_full_rank`, with a message that says sequence length alone
   caps the design at rank two. One row per candidate via `robust_step_time`, and record
   `step_time_dispersion` in diagnostics. Coefficient names: `per_step_overhead`,
   `token_cost`, `microbatch_overhead`. Set
   `diagnostics["slots"] = "num_microbatches"`.
3. **Memory design.** Fit `target - fsdp_inventory(...).total_bytes` against
   `[activation_feature, 1]` where `activation_feature = max(sum_tokens)`: there is no
   `min(nmb, pp)` factor because there is no pipeline, so only one microbatch is in
   flight. Do this **twice**, once against `peak_allocated_bytes` and once against
   `peak_reserved_bytes`. Compute each fit's validation `r^2` via
   `FitMetrics.from_predictions`. Select the higher, and record:
   - `diagnostics["memory_metric"]`: the winner's identifier string
   - `diagnostics["candidate_r_squared"]`: a dict mapping both identifier strings to
     their `r^2`
   - `diagnostics["reserved_over_allocated"]`: `max(1.0, mean(reserved / allocated))`
     across successful records
   Store the winner's coefficients as `activation_slope` and `residual_constant`, plus
   `parameter_count` and `world_size`.
4. **Memory prediction.** `fsdp_inventory(...).total_bytes + activation_slope *
   feature + residual_constant`, then multiply by `reserved_over_allocated` when the
   selected metric is `peak_allocated_bytes`. The simulator compares the result against
   device capacity, and `allocated` excludes fragmentation, so predicting `allocated`
   directly would under-report OOM.
5. **Planner integration.** Both models hold a `MicrobatchPlanner`. `predict` calls
   `planner.plan(workload.sequence_lengths, parallelism.dp)` to obtain the critical
   rank's microbatches instead of `workload.critical_rank_microbatches`, because that
   method assumes uniform equal-count microbatches.
6. **`from_artifact`.** Accepts `batching_mode: str = "pad_to_max"` and
   `max_tokens_per_rank: int | None = None`, builds the planner from
   `BatchingMode(batching_mode)`, and infers the discrete selectors from the artifact
   when a single value is present, exactly as `HydraulisTrainingCost.from_artifact`
   does.

- [ ] **Step 4: Register the model**

In `srl/cost/training/__init__.py` add the imports, extend `__all__`, and register:

```python
from .fsdp import FsdpLatencyModel, FsdpMemoryModel, FsdpTrainingCost
from .fsdp_inventory import FsdpInventory, fsdp_inventory
from .microbatch import BatchingMode, MicrobatchPlanner, verl_workload

TRAINING_MODEL_REGISTRY = {
    "hydraulis": HydraulisTrainingCost,
    "fsdp": FsdpTrainingCost,
}
```

- [ ] **Step 5: Run the tests**

```bash
python -m pytest tests/test_fsdp_cost.py -q
python -m pytest tests/ -q
```

Expected: 14 new tests pass; `708 passed` (694 + 14).

- [ ] **Step 6: Commit**

```bash
git add srl/cost/training/fsdp.py srl/cost/training/__init__.py tests/test_fsdp_cost.py
git commit -m "feat: add the FSDP training cost model

The latency form is the structured one from latency.py rather than a new
polynomial: FSDP is the degenerate case that module documents, where
pipeline_slots collapses to the microbatch count. Reusing it means the
quadratic term that was measured and rejected there cannot creep back in here.

The memory fit runs against both peak metrics and selects on validation r
squared, recording both scores and the winner in diagnostics so the choice is
auditable. Because a fitted allocated prediction excludes fragmentation, the
artifact also stores the measured reserved-over-allocated ratio and applies it
at predict time: the regression keeps the well-behaved target while the OOM
comparison keeps the quantity that actually causes OOM.

Records are grouped on the mesh and the offload flags read from tags, and a
1F1B record is rejected outright, so Megatron and FSDP measurements can never
land in one cell."
```

---

### Task 8: CLI, configuration, and the example driver

**Files:**
- Modify: `srl/cli/profile.py`
- Modify: `srl/cli/regress.py:93-94`
- Modify: `srl/config/schema.py`
- Modify: `srl/cli/main.py`
- Modify: `srl/profiler/adapters/__init__.py`
- Create: `srl/config/defaults/training_cost/fsdp.yaml`
- Create: `examples/profile/fsdp/qwen3_8b.yaml`
- Create: `examples/profile/fsdp/profile_qwen3_8b.sh`
- Modify: `.claude/codebase-map.md`
- Test: `tests/test_fsdp_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 3-7.
- Produces: `srl-profile fsdp {generate,run,collect}`; `srl-regress training --latency-model fsdp`; `training_cost.name=fsdp` resolvable by `build_training_cost`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fsdp_cli.py`:

```python
"""FSDP CLI wiring and Hydra configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from srl.cli.profile import build_parser


def test_parser_exposes_the_three_fsdp_workflows():
    parser = build_parser()
    for workflow in ("generate", "run", "collect"):
        args = parser.parse_args(["fsdp", workflow, "--config", "c.yaml"])
        assert args.backend_name == "fsdp"
        assert args.workflow == workflow
        assert callable(args.func)


def test_fsdp_run_accepts_selection_and_resume():
    parser = build_parser()
    args = parser.parse_args(
        ["fsdp", "run", "--config", "c.yaml", "--index", "2", "--resume", "--dry-run"]
    )
    assert args.index == 2
    assert args.resume is True
    assert args.dry_run is True


def test_regress_accepts_the_fsdp_models():
    from srl.cli.regress import build_parser as regress_parser

    args = regress_parser().parse_args(
        [
            "training",
            "--latency-model",
            "fsdp",
            "--memory-model",
            "fsdp",
            "--input",
            "in.jsonl",
            "--output",
            "out.json",
        ]
    )
    assert args.latency_model == "fsdp"
    assert args.memory_model == "fsdp"


def test_training_cost_config_carries_the_batching_mode():
    from srl.config.schema import TrainingCostConfig

    config = TrainingCostConfig()
    assert config.batching_mode == "pad_to_max"
    assert config.max_tokens_per_rank is None


def test_fsdp_defaults_yaml_is_loadable_and_documented():
    path = Path("srl/config/defaults/training_cost/fsdp.yaml")
    value = yaml.safe_load(path.read_text())
    assert value["name"] == "fsdp"
    assert "batching_mode" in value
    assert value["batching_mode"] in ("pad_to_max", "token_budget", "packed_no_pad")
    # Every knob a reader must choose needs its rationale in the file.
    assert path.read_text().count("#") >= 6


def test_example_sweep_varies_both_identifiability_axes():
    value = yaml.safe_load(Path("examples/profile/fsdp/qwen3_8b.yaml").read_text())
    search = value["search_space"]
    assert len(search["sequence_lengths"]) >= 3
    assert len(search["n_seqs_per_rank"]) >= 2
    assert sorted(search["shard_sizes"]) == [2, 4, 8]


def test_example_sweep_retains_eight_steps():
    value = yaml.safe_load(Path("examples/profile/fsdp/qwen3_8b.yaml").read_text())
    training = value["training"]
    assert training["max_steps"] - training["warmup_steps"] == 8


def test_build_training_cost_resolves_the_fsdp_name(tmp_path: Path, monkeypatch):
    from omegaconf import OmegaConf

    from srl.cli.main import build_training_cost

    # A name with no artifact must fail loudly rather than silently falling back.
    cfg = OmegaConf.create({"name": "fsdp", "artifact_path": ""})
    with pytest.raises(Exception):
        build_training_cost(cfg)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/test_fsdp_cli.py -q
```

Expected: multiple failures: the `fsdp` subparser does not exist and
`TrainingCostConfig` has no `batching_mode`.

- [ ] **Step 3: Add the CLI handlers**

In `srl/cli/profile.py`, add the FSDP imports beside the Megatron ones:

```python
from srl.profiler.adapters.fsdp import (
    FsdpProfileConfig,
    collect_fsdp_records,
    prepare_fsdp_candidate,
    run_fsdp_candidate,
)
from srl.profiler.adapters.fsdp import (
    read_candidate_manifest as read_fsdp_manifest,
)
from srl.profiler.adapters.fsdp import (
    write_candidate_manifest as write_fsdp_manifest,
)
from srl.profiler.fsdp_search_space import enumerate_fsdp_candidates
```

Add a config loader mirroring `_bridge_config`, a candidate generator reading
`search_space`, and three handlers `cmd_fsdp_generate`, `cmd_fsdp_run`,
`cmd_fsdp_collect` that mirror the Megatron trio exactly, substituting the FSDP
functions and defaulting the collect output to
`config.output_dir / "profiles" / "fsdp.jsonl"`.

Register the subparser immediately after the Megatron block, before `return parser`:

```python
    fsdp = backends.add_parser("fsdp", help="Use the SimRL FSDP2 launcher.")
    fsdp_workflows = fsdp.add_subparsers(dest="workflow", required=True)
    fsdp_generate = fsdp_workflows.add_parser("generate")
    _add_config_argument(fsdp_generate)
    fsdp_generate.add_argument("--candidates")
    fsdp_generate.set_defaults(func=cmd_fsdp_generate)

    fsdp_run = fsdp_workflows.add_parser("run")
    _add_config_argument(fsdp_run)
    _add_candidate_arguments(fsdp_run)
    fsdp_run.add_argument("--dry-run", action="store_true")
    fsdp_run.add_argument("--resume", action="store_true")
    fsdp_run.set_defaults(func=cmd_fsdp_run)

    fsdp_collect = fsdp_workflows.add_parser("collect")
    _add_config_argument(fsdp_collect)
    _add_candidate_arguments(fsdp_collect)
    fsdp_collect.add_argument("--output")
    fsdp_collect.set_defaults(func=cmd_fsdp_collect)
```

- [ ] **Step 4: Widen the regress choices and the config schema**

In `srl/cli/regress.py` lines 93-94, change both `choices=["hydraulis"]` to
`choices=["hydraulis", "fsdp"]`, and dispatch to `FsdpTrainingCost.fit` when the model
is `fsdp`.

In `srl/config/schema.py`, add to `TrainingCostConfig`:

```python
    # How real trajectory lengths become microbatches. Profiling measures uniform
    # shapes, so this is a simulate-time choice, and the three modes span a measured
    # 1.17x to 4.34x latency range on DAPO-like bimodal lengths.
    batching_mode: str = "pad_to_max"
    # Token budget per rank for the token_budget mode. verl's default is 10240.
    max_tokens_per_rank: int | None = None
```

In `srl/cli/main.py`, pass both through to `from_artifact` for registry models.

In `srl/profiler/adapters/__init__.py`, export the FSDP names alongside the
existing Megatron and vLLM ones, keeping `__all__` sorted.

- [ ] **Step 5: Create the defaults YAML**

Create `srl/config/defaults/training_cost/fsdp.yaml`:

```yaml
# Fitted FSDP2/HSDP training cost. Requires a profile JSONL regression artifact.
name: fsdp

# Path to a CostArtifact produced by `srl-regress training --latency-model fsdp`.
artifact_path: ""

# Simulated mesh. Every rank consumes distinct data, so dp is the world size;
# sharding and replication are recorded as artifact selectors, not here.
parallelism:
  tp: 1
  pp: 1
  dp: 8
  cp: 1

# Sequences per microbatch on one rank when the batching mode pads.
micro_batch_size: 1

# FSDP has no pipeline schedule; the axis degenerates rather than taking a value.
schedule: fsdp

# How real trajectory lengths become microbatches. pad_to_max charges for pad tokens,
# token_budget reproduces verl's dynamic batching, packed_no_pad charges only real
# tokens. Measured spread between the extremes is 1.17x to 4.34x with the fill ratio.
batching_mode: pad_to_max

# Token budget per rank for the token_budget mode. verl's DAPO scripts use 10240.
max_tokens_per_rank: null

# Activation checkpoint policy. Must match a profiled discrete cell.
activation_recompute: full

# Optimizer name. Must match a profiled discrete cell.
optimizer: adam

# Dtypes used by the static FSDP inventory and cell lookup. FSDP2 shards parameters,
# gradients, and optimizer state alike, so there is no distributed_optimizer flag.
parameter_dtype: bf16
gradient_dtype: fp32
optimizer_state_dtype: fp32

# When false, numerical features outside the profiled range fail fast.
allow_extrapolation: false

# Per-GPU capacity in MiB used by the simulator to flag OOM. null disables the check.
gpu_memory_mib: null
```

- [ ] **Step 6: Create the full sweep config and driver**

Create `examples/profile/fsdp/qwen3_8b.yaml`:

```yaml
# HuggingFace checkpoint. Architecture fields are read from config.json / AutoConfig.
model: /apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B

# gpu_count is the fixed world size: shard_size * replicate_size must equal it.
hardware:
  gpu_name: H20
  gpu_count: 8
  gpu_memory_mib: 97871
  node_count: 1
  gpus_per_node: 8
  interconnect: NVLink

output_dir: ../../../outputs/profile/qwen3_8b/fsdp

training:
  compute_dtype: bf16
  optimizer: adam
  parameter_dtype: bf16
  gradient_dtype: fp32
  optimizer_state_dtype: fp32
  # Twelve steps discarding four, inherited from the Megatron measurement work: one
  # warmup step left the first profiled step the slowest in 18 of 24 runs, and eight
  # retained samples are what make a median and outlier rejection possible.
  max_steps: 12
  warmup_steps: 4
  # flash_attention_2 is what verl runs and what the packed no-pad mode requires.
  attn_implementation: flash_attention_2

profiling:
  allow_incomplete: false

environment:
  HF_HUB_OFFLINE: "1"

search_space:
  world_size: 8
  # Pure FSDP and both HSDP meshes. The replicate degree is derived from world_size.
  shard_sizes: [2, 4, 8]
  # Varied independently of sequence_lengths so the design row [1, slots*tokens, slots]
  # reaches rank three; sweeping length alone would cap it at rank two.
  n_seqs_per_rank: [1, 2, 4, 8]
  sequence_lengths: [512, 1024, 2048, 4096, 8192]
  # psrl enables gradient checkpointing, so the profiled cell matches what runs.
  gradient_checkpointing: [true]
  # Both, because offloading changes latency and memory by a first-order amount.
  optimizer_offload: [true, false]
  reshard_after_forward: [true]
```

Create `examples/profile/fsdp/profile_qwen3_8b.sh`, mirroring
`examples/profile/megatron/profile_qwen3_8b.sh` with `megatron` replaced by `fsdp` and
the config defaulting to `qwen3_8b.yaml`. Make it executable with `chmod +x`.

- [ ] **Step 7: Run the tests**

```bash
python -m pytest tests/test_fsdp_cli.py -q
python -m pytest tests/ -q
python -m ruff check srl/ tests/
```

Expected: 8 new tests pass; `716 passed` (708 + 8); ruff clean.

- [ ] **Step 8: Update the codebase map**

In `.claude/codebase-map.md`, add `adapters/training/`, `adapters/fsdp/`,
`fsdp_search_space.py`, `cost/training/{fsdp,fsdp_inventory,microbatch}.py` to the
directory tree, and add the FSDP entries to the import-dependency order.

- [ ] **Step 9: Commit**

```bash
git add srl/cli/ srl/config/ srl/profiler/adapters/__init__.py \
        examples/profile/fsdp/ tests/test_fsdp_cli.py .claude/codebase-map.md
git commit -m "feat: expose the FSDP backend through the CLI and Hydra config

The three workflows mirror the Megatron subcommands, so the sweep is driven
the same way regardless of backend.

TrainingCostConfig gains batching_mode and max_tokens_per_rank. Those are
simulate-time choices rather than profile parameters, because profiling
measures uniform shapes and only the planner turns real trajectory lengths
into microbatches. They belong in configuration because the three modes span
a measured 1.17x to 4.34x latency range.

The sweep varies sequence length and sequences-per-rank independently. That is
not thoroughness for its own sake: sweeping length alone caps the design matrix
at rank two and the two overhead coefficients stop being identifiable."
```

- [ ] **Step 10: Run the full sweep and report both memory targets**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh
cd examples/profile/fsdp
bash profile_qwen3_8b.sh generate
bash profile_qwen3_8b.sh run --resume
bash profile_qwen3_8b.sh collect
cd ../../..
srl-regress training --latency-model fsdp --memory-model fsdp \
    --input outputs/profile/qwen3_8b/fsdp/profiles/fsdp.jsonl \
    --output outputs/cost/qwen3_8b/training_fsdp.json
```

Then report, per cell, from the artifact diagnostics:
- latency `r^2`, MRE, and `design_rank` (must be 3)
- `step_time_dispersion` (above 0.05 means the measurement is the limiting factor)
- both memory candidates' `r^2`, the selected `memory_metric`, and
  `reserved_over_allocated`

Do not claim the model works without these numbers. If latency MRE exceeds roughly
10% or the memory `r^2` is below 0.9 for the selected target, report the numbers and
stop rather than tuning coefficients to fit.

---

## Notes for the implementer

**Why the launcher is the risk concentrate.** Everything else in this plan is
testable without a GPU. `launch.py` is the one module whose correctness depends on
eight real ranks agreeing, which is why Task 4 ends with a smoke gate and why the
collector cross-checks the mesh the launcher reports. If the smoke run's dispersion
exceeds 5%, fix the measurement before touching the model: the Megatron work had to
undo a model fitted to a noisy measurement, and the fit error could not be attributed
to the model rather than the data.

**Why packing is not a profile axis.** It was considered and rejected. Profiling
ragged shapes would violate `TrainingPayload`'s equal-count invariant and force either
a schema fork or a loosened type, and it would put a second copy of the packing
algorithm in the launcher where it could silently diverge from the simulator's. Uniform
profiling plus a simulate-side planner gives one implementation and no schema change.

**What to do if `fsdp_inventory` disagrees with the measurement.** The launcher writes
`static_allocated_bytes_at_train_start` for exactly this comparison. If the analytic
`16p/world_size` is off by more than a few percent, report the discrepancy rather than
adjusting the fitted residual to absorb it: that is how the Megatron static term stayed
wrong by a factor of two while its predictions still looked reasonable inside the
profiled range.
