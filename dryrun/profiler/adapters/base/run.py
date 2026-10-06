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
