"""Run one Megatron training candidate under torchrun."""

from __future__ import annotations

from pathlib import Path

from ...search_space import TrainingCandidate
from ..base import OOM_EXIT_CODE, oom_marker_paths, run_candidate
from .config import MegatronBridgeConfig
from .prepare import prepare_bridge_candidate

__all__ = ["OOM_EXIT_CODE", "oom_marker_paths", "run_bridge_candidate"]


def run_bridge_candidate(
    config: MegatronBridgeConfig,
    candidate: TrainingCandidate,
    *,
    dry_run: bool = False,
    resume: bool = False,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str | None = None,
    master_port: int | None = None,
) -> Path:
    """Prepare one candidate's launch spec and execute it in its own torchrun process.

    Args:
        config: Adapter configuration.
        candidate: The topology and batch to measure.
        dry_run: Write the spec and command without executing.
        resume: Skip a candidate whose run directory already holds a result.
        nnodes: Nodes the candidate's mesh spans. A mesh larger than one host's GPU count
            needs this: Qwen3.5-35B-A3B cannot fit a trainable mesh on eight GPUs, so its
            sweep runs at 16.
        node_rank: This host's index in `[0, nnodes)`. Every host runs the same command with
            its own rank.
        master_addr: Rendezvous host, required when `nnodes > 1`.
        master_port: Rendezvous port, required when `nnodes > 1`.
    """
    command, run_dir = prepare_bridge_candidate(
        config,
        candidate,
        nnodes=nnodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
    )
    return run_candidate(
        run_dir=run_dir,
        command=command,
        environment=config.environment,
        dry_run=dry_run,
        resume=resume,
    )
