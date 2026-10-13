"""The shared base of the training profiling adapters, not a backend of its own.

Megatron and FSDP differ in how they build a model and a launch command, but they
agree on how a candidate is run, how its result is classified, and how its
measurements are normalized. Only that agreement lives here. Resolved-config
validation deliberately does not: Megatron reads a serialized `ConfigContainer`
while FSDP inspects a device mesh, so one signature would be a parameter bag
neither backend fills.

Named `base` rather than `training` because it sits beside `megatron/`, `fsdp/` and
`vllm/`, where a backend's own name is what a directory means. `training` read as a
fourth backend and needed this docstring to say otherwise; `base` matches how the
rest of the tree already names a shared foundation (`cost/kernel/ops/base.py`,
`cost/rollout/base.py`, `cost/training/base.py`).
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
