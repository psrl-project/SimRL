"""Direct kernel profiling: measure attention, GEMM, collectives, and more on the GPU.

Four layers, each with its own `__init__` explaining its boundary:

- `sweep/`: one grid declaration per op family, parsed from the config's own YAML subtree.
- `grid/`: which shapes those grids enumerate, derived from the architecture. Torch-free.
- `runner/`: the measurement loops. The GPU side. The cost path must not reach them.
- `utils/`: timing, model dimensions, and config validators shared across the three.

Only `KernelProfileConfig` is re-exported here. The package used to expose 28 names behind a
PEP 562 lazy-import table so that reaching `config` did not pull in torch. The layering now
does that structurally, and only `cli/profile.py` ever imported from the package root. Every
other caller names the module it wants, which is also what keeps the torch-free boundary
checkable per module in `tests/test_kernel_package_imports.py`.
"""

from __future__ import annotations

from .config import KernelProfileConfig

__all__ = ["KernelProfileConfig"]
