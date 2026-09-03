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
