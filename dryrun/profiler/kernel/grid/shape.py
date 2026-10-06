"""The GEMM weight shape, split out so a family module can name it without a cycle.

`shapes.py` imports the families to delegate to them, and each family returns `GemmShape`, so
the type cannot live in `shapes.py` without closing a loop. It has no dependencies at all, which
is why it is its own module rather than a member of one side.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GemmShape:
    """One per-rank GEMM weight shape inside a transformer block."""

    name: str
    n: int
    k: int
