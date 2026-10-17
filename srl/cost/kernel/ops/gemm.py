"""
The unified GEMM op: every linear layer in the model, sliced by `(dtype, n, k)`.

One table serves forward, dgrad, and wgrad, because a GEMM's cost depends only on `(m, n, k)`
and the backward passes are the same arithmetic with permuted axes. `grid.gemm.gemm_wgrad_shapes`
emits the permuted triples into this same table, so backward needs no separate op.

`m` is the only interpolation axis. `n` and `k` are categorical: blending across output or
input widths would mix kernels with different tiling, and a `(4096, 4096)` GEMM tells you
little about a `(151936, 4096)` one.

No fitted floor. GEMM's holdout MRE is 3.91% without one, and the regime is compute- or
bandwidth-limited rather than dispatch-limited, so a floor would have nothing to correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..hardware import GpuSpec
from ..sol import SolComponents, gemm_sol
from .base import register_op


@dataclass(frozen=True)
class GemmKernelOp:
    """The unified GEMM perf table and its roofline.

    Named `GemmKernelOp` rather than `GemmOp` because `model_ops.GemmOp` already means
    "one GEMM in the per-step op list", which is a different concept.

    Frozen because the instance below is a registry singleton shared by every caller, so a
    mutation anywhere would silently change which columns every table is read against.
    """

    name = "gemm"
    family = "gemm"
    slice_columns = ("gemm_dtype", "n", "k")
    axis_columns = ("m",)
    fits_floor = False
    # Shared by the rollout and training models, and the two disagree: a decode GEMM is
    # inside the FULL graph while a Megatron GEMM is not. Left event-timed because the
    # stake does not justify splitting the table -- the gap is 1.68 us per call at m<=8
    # (9.64 us event against 7.96 us amortised), about 0.47 ms across a 40-layer step,
    # and it vanishes above m=128 where the kernel is compute-bound. Contrast GDN decode,
    # where the same gap was 21.95 us per call.
    graph_replayed = False

    def sol(
        self,
        spec: GpuSpec,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """
        Return the roofline of one `(m, k) @ (k, n)` GEMM.

        Args:
            spec (GpuSpec): Hardware the roofline is evaluated against.
            slice_key (tuple[Any, ...]): `(gemm_dtype, n, k)`.
            coords (tuple[float, ...]): `(m,)`.
            floor_s (float): Ignored; `fits_floor` is False. Accepted so every op shares
                one call signature.

        Returns:
            SolComponents: Compute and memory halves in seconds.

        Raises:
            ValueError: If `m`, `n`, or `k` is not positive.
        """
        dtype, n, k = slice_key[0], int(slice_key[1]), int(slice_key[2])
        return gemm_sol(spec, m=int(coords[0]), n=n, k=k, dtype=dtype)

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the `gemm:` sweep subtree into a `GemmSweep`."""
        # Imported lazily: `srl.profiler.kernel.sweep` reaches the profiler package, and
        # a simulator run that only prices kernels must not load it. Four dots because this
        # module is `srl.cost.kernel.ops.gemm`; Step 4 verifies the depth resolves.
        from ....profiler.kernel.sweep import GemmSweep

        return GemmSweep.from_dict(value)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Time every swept GEMM shape and write the unified table."""
        # Imported here so the cost path never loads torch.
        from ....profiler.kernel.runner.gemm import GemmRunner

        return GemmRunner(model, config).run(writer_factory=writer_factory)


GEMM_OP = GemmKernelOp()
register_op(GEMM_OP)
