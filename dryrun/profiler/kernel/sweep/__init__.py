"""One sweep grid per op family: what shapes to measure, parsed from the config's own subtree.

Each `*Sweep` owns the parsing and validation of one `<family>:` YAML block, so adding an op
adds a file here rather than a field to a shared config class. `cost/kernel/ops/<op>.py`
reaches its own sweep through `parse_sweep`, which imports from this package lazily: the cost
path prices kernels from parquet and must not load the profiler to do it.

Every module here must stay free of `torch`. A sweep is pure data validation, and
`tests/test_kernel_package_imports.py` enforces it per module.
"""

from __future__ import annotations

from .attention import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_GENERATION_KV_TOKENS,
    AttentionSweep,
)
from .attention_training import TrainingAttentionSweep
from .collective import CollectiveSweep
from .elementwise import ElementwiseSweep
from .gdn import GdnSweep
from .gemm import GemmSweep
from .optimizer import OptimizerSweep

__all__ = [
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_MAX_GENERATION_KV_TOKENS",
    "AttentionSweep",
    "CollectiveSweep",
    "ElementwiseSweep",
    "GdnSweep",
    "GemmSweep",
    "OptimizerSweep",
    "TrainingAttentionSweep",
]
