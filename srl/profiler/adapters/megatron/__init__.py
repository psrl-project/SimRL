"""SimRL-owned Megatron training profiling.

The local path does not use the vendored `scripts/performance` CLI. That wrapper
is built for the Slurm and NeMo-Run container flow: it parses known arguments and
discards everything else, so profiling, recompute, and output-path settings never
reach the training config. Here `prepare` writes a launch spec, `launch` applies it
to a `ConfigContainer` directly in Python, and `measure` records step times and
per-rank memory peaks through the official callback API.

`launch` and `measure` import torch and megatron, so they are deliberately absent
from this namespace; only the torchrun child imports them.
"""

from .collect import collect_bridge_record, collect_bridge_records
from .config import MegatronBridgeConfig, infer_bridge_recipe
from .execute import OOM_EXIT_CODE, oom_marker_paths, run_bridge_candidate
from .prepare import (
    build_launch_command,
    build_launch_spec,
    prepare_bridge_candidate,
    read_candidate_manifest,
    write_candidate_manifest,
)

__all__ = [
    "MegatronBridgeConfig",
    "OOM_EXIT_CODE",
    "build_launch_command",
    "build_launch_spec",
    "collect_bridge_record",
    "collect_bridge_records",
    "infer_bridge_recipe",
    "oom_marker_paths",
    "prepare_bridge_candidate",
    "read_candidate_manifest",
    "run_bridge_candidate",
    "write_candidate_manifest",
]
