"""Run one FSDP training candidate under torchrun."""

from __future__ import annotations

from pathlib import Path

from ...fsdp_search_space import FsdpCandidate
from ..base import OOM_EXIT_CODE, oom_marker_paths, run_candidate
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
