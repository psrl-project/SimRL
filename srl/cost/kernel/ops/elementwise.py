"""
Elementwise kernels: rmsnorm, rope, activation, and Qwen3's qk_norm.

All four are pure bandwidth (they read a tensor, write a tensor, and do no matmul) so they
share one schema and one roofline shape. Only the bytes touched differ, and `n_elements`
carries that, which is why `op_kind` is a slice column rather than a separate op per kernel:
one utility surface per kind, but one formula for all of them.

A fitted floor is essential here rather than a refinement. Measured on H20, all four kinds sit
at 4-5 us regardless of size below about 128 tokens, so in the decode regime they are entirely
dispatch-bound and the bandwidth term is meaningless. Their *share* of a step therefore grows as
the step shrinks, which is the opposite of how a bandwidth-bound term would behave.

**This table is deliberately not consumed by `ComposedRollout`.** Utility is
`SOL / measured_latency`, and the measured GEMM and attention latencies were collected from a
server whose elementwise kernels were running between them, so the elementwise cost is already
inside those fitted utilities. Adding a separate term double-counts it: measured over the 250
validate buckets it takes decode from 4.84% to 11.18% MRE, prefill 3.72% to 5.84%, and mixed
0.48% to 1.89%, with every median ratio moving away from 1.0. The table exists for the FSDP
training backend, which runs without CUDA graphs and needs an explicit term, and as the third
op family that proves this abstraction generalizes. Wiring it into a rollout prediction requires
first refitting GEMM and attention utility against kernel-only latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..hardware import GpuSpec, dtype_bytes
from ..sol import SolComponents
from .base import register_op

# The kernels vLLM runs between the GEMMs of one transformer block. `qk_norm` is separate from
# `rms_norm` because it normalizes over `head_dim` rather than `hidden_size`, so it lands on a
# different point of the bandwidth curve at the same token count.
ELEMENTWISE_KINDS: tuple[str, ...] = (
    "rms_norm",
    "rotary_embedding",
    "silu_and_mul",
    "qk_norm",
    "tensor_plumbing",
)
"""The pointwise kinds a step issues.

`tensor_plumbing` is not a model kernel and is measured for the training path only. Tracing
a real Megatron step found 158.7 device launches per layer per microbatch against the 7 the
model inventory named, and the population is `aten::copy_` (414 calls), `aten::cat` (193,
the QKV split), `aten::to` (246), `aten::mul` (289) and `aten::add_` (158) rather than
norms or activations. It is tensor staging: dtype conversions, contiguity fixes, and the
concatenations TE's fused modules need around their inputs.

It is one kind rather than five because the cost is **85% launch floor** at 4096 tokens
(8.74 us per launch against 2.5 ns per token), so what distinguishes the members is their
count, not their bandwidth curve. Measuring `copy_` separately from `mul` would fit two
nearly identical floors. See `docs/reference/limitations.md`.
"""


@dataclass(frozen=True)
class ElementwiseKernelOp:
    """Bandwidth-bound pointwise kernels, sliced by dtype and kind.

    Frozen because the instance below is a registry singleton shared by every caller, so a
    mutation anywhere would silently change which columns every table is read against.
    """

    name = "elementwise"
    family = "elementwise"
    slice_columns = ("dtype", "op_kind")
    axis_columns = ("n_elements",)
    fits_floor = True
    # Consumed only by the training model, and Megatron runs no CUDA graphs, so the launch
    # cost is genuinely on its critical path. `ComposedRollout` does not charge elementwise
    # at all precisely because graph replay absorbs it there. See `training_ops.py`.
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
        Return the roofline of one elementwise kernel call.

        `n_elements` counts the elements *read*. The write is assumed equal, so the byte
        traffic is `2 * n_elements * dtype_bytes`. That holds for all four kinds: rmsnorm and
        qk_norm read and write one tensor, rope reads and writes q and k, and silu_and_mul
        reads `2n` and writes `n` for a total of `3n` against the assumed `2n`: a 1.5x
        underestimate that the fitted utility absorbs, in the same way it absorbs every other
        SOL simplification.

        Args:
            spec (GpuSpec): Hardware the roofline is evaluated against.
            slice_key (tuple[Any, ...]): `(dtype, op_kind)`.
            coords (tuple[float, ...]): `(n_elements,)`.
            floor_s (float): Fitted dispatch floor in seconds, applied inside the `max`.

        Returns:
            SolComponents: A zero compute term, the bandwidth term, and the floor.

        Raises:
            ValueError: If `n_elements` is not positive or `op_kind` is unknown.
        """
        dtype, op_kind = slice_key[0], slice_key[1]
        if op_kind not in ELEMENTWISE_KINDS:
            known = ", ".join(ELEMENTWISE_KINDS)
            raise ValueError(f"Unknown elementwise op_kind {op_kind!r}. Known kinds: {known}.")
        n_elements = float(coords[0])
        if n_elements <= 0.0:
            raise ValueError(f"n_elements must be positive, got {n_elements!r}.")
        memory_s = 2.0 * n_elements * dtype_bytes(dtype) / spec.memory_bandwidth_bytes_per_s
        return SolComponents(math_s=0.0, memory_s=memory_s, floor_s=floor_s)

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the `elementwise:` sweep subtree into an `ElementwiseSweep`."""
        from ....profiler.kernel.sweep import ElementwiseSweep

        return ElementwiseSweep.from_dict(value)

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Time every elementwise kind across the token grid and write one table."""
        from ....profiler.kernel.runner.elementwise import ElementwiseRunner

        return ElementwiseRunner(model, config).run(writer_factory=writer_factory)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)


ELEMENTWISE_OP = ElementwiseKernelOp()
register_op(ELEMENTWISE_OP)
