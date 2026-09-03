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
