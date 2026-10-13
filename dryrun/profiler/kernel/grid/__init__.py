"""What shapes each op measures, derived from the architecture without touching a GPU.

One module per op family, each answering "which points does this sweep enumerate, and what are
their dimensions". Kept separate from `runner/` for two reasons:

- The cost path imports `grid/gemm.py` and `grid/attention.py` to build its *queries*, and must
  not pay a 4.2-second GPU-stack import to do it.
- The arithmetic becomes testable. `grid/elementwise.py`'s element counts are derived a second
  time in `cost/training/training_ops.py` on the query side; while they lived inside the timing
  loop the only way to compare the two was to run a sweep on a GPU.

Every module here must stay free of `torch` and `vllm`;
`tests/test_kernel_package_imports.py` enforces it per module, so a new grid that reaches for a
tensor helper fails immediately rather than slowing every simulator run.

The GEMM grid is the one with real dispatch behind it: which shapes a layer runs is a property
of the architecture, so `grid/gemm.py` delegates to a `ModelFamily` from `grid/families.py` and
an unregistered architecture is refused rather than priced with the dense arithmetic.
"""
