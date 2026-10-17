"""Measure one transformer block's linear layers as a unified GEMM table.

Shapes are sharded by tensor parallelism in `grid/gemm.py`, at construction time, so the table
itself is TP-agnostic: it stores `(m, n, k)` and nothing else. That is what lets a later phase
price backward dgrad and wgrad from these same rows by permuting the axes.

Every row's `kernel_source` is `torch.nn.functional.linear`, which is what vLLM's unquantized
bfloat16 linear layers dispatch to. Unlike the attention and collective families, that is known
before the sweep starts rather than selected at runtime, so this runner calls `writer_factory`
once up front.
"""

from __future__ import annotations

from pathlib import Path

from ..grid.gemm import gemm_all_shapes
from .base import KernelRunner, WriterFactory


class GemmRunner(KernelRunner):
    """Times every swept `(m, n, k)` through `torch.nn.functional.linear`."""

    def run(self, *, writer_factory: WriterFactory) -> tuple[Path, ...]:
        """
        Time every swept GEMM shape and write the unified table.

        Args:
            writer_factory (WriterFactory): Builds a writer from
                `(framework, version, kernel_source)`.

        Returns:
            tuple[Path, ...]: The single written parquet path.
        """
        import torch

        from ..utils.timing import torch_dtype

        dtype_name = self.config.measurement.dtype
        dtype = torch_dtype(dtype_name)
        device = torch.device("cuda")
        rows: list[dict[str, object]] = []
        shapes = gemm_all_shapes(
            self.model,
            self.config.sweep("gemm"),
            tensor_parallel=self.config.tensor_parallel,
        )
        for m, n, k in shapes:
            activation = torch.randn(m, k, dtype=dtype, device=device)
            weight = torch.randn(n, k, dtype=dtype, device=device)
            latency_s = self.time(
                # Bound as default arguments rather than captured, so the lambda holds its
                # own reference and the `del` below can free the tensors.
                lambda a=activation, w=weight: torch.nn.functional.linear(a, w)
            )
            rows.append(
                self.row("gemm", m=m, n=n, k=k, gemm_dtype=dtype_name, latency_s=latency_s)
            )
            del activation, weight
            torch.cuda.empty_cache()
        writer = writer_factory(
            "vllm", str(torch.__version__), "torch.nn.functional.linear"
        )
        return (writer.write("gemm", rows),)
