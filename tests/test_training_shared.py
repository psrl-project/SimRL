"""The backend-agnostic training-adapter layer shared by Megatron and FSDP."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from srl.profiler.adapters.base import (
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
    """A log mentioning the phrase must stay `failed`. Only markers are authoritative.

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
