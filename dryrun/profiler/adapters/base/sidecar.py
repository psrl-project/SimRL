"""Sidecar reading and measurement normalization shared by training backends.

A launcher writes one `dryrun_metrics.json` per candidate and the collector turns it
into a payload. The warmup policy lives here, not in the launcher, so the launcher
stays a pure measurer and the retained window can change without re-profiling.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ...schema import ProfileValidationError, RankMemory

SIDECAR_NAME = "dryrun_metrics.json"
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
