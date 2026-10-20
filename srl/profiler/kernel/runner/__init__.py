"""One runner per op family: the measurement loop that turns a grid into a perf table.

The GPU side of the tree. Every module here may import `torch`, and the cost path must never
reach them. `cost/kernel/ops/<op>.py` imports its runner inside `measure()` for that reason.

`KernelRunner` in `base.py` carries the mechanism every loop shares (timing with this sweep's
iteration counts, building a row checked against the op's declared columns) but not the loop
itself. See its docstring for why a template method would need an escape hatch per family.

`collective.py` sits outside this package: it launches torchrun rather than timing in-process,
so it has no loop for a runner to own.
"""

from __future__ import annotations

from .attention import AttentionRunner
from .attention_training import TrainingAttentionRunner
from .base import KernelRunner, WriterFactory
from .elementwise import ElementwiseRunner
from .gemm import GemmRunner
from .optimizer import OptimizerRunner

__all__ = [
    "AttentionRunner",
    "ElementwiseRunner",
    "GemmRunner",
    "KernelRunner",
    "OptimizerRunner",
    "TrainingAttentionRunner",
    "WriterFactory",
]
