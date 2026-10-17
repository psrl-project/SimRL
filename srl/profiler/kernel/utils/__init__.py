"""Cross-cutting helpers for the kernel profiler: timing, model dimensions, config validation.

Every module here must stay importable without `torch` except `timing.py`, which is the one
place that owns the CUDA event loop. The grid and sweep layers depend on `model_dims` and
`validators`, and both are reached from the cost path, which prices kernels from parquet and
must not pay a 4.2-second GPU-stack import to do it.
"""
