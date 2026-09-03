"""DryRun-owned FSDP2 and HSDP training profiling.

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
