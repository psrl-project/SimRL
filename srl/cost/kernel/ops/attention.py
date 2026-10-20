"""
The two attention phases, which share a table directory but not an axis.

Prefill and decode are different regimes. Prefill attention is quadratic in sequence length
while decode is linear in KV history, so they are separate ops with separate tables. They
share one family directory, one set of slice columns, and one `floors.json`, because they run
through the same backend and pay the same dispatch path, so they live in one module with
a shared slice unpacker.

Both take a fitted floor. Real H20 data shows decode latency flat at ~70 us across a 64x
increase in KV work at batch 1, which is conclusive that the cost is `max(floor, work)` rather
than `floor + work`. Placing the floor inside the max lifted the holdout gate from >20% MRE to
under 10%. The floor is a property of the backend's dispatch path rather than the silicon, so
its value is fitted per `(backend, version)` and stored beside the table. See `floor.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..hardware import GpuSpec
from ..sol import SolComponents, context_attention_sol, generation_attention_sol
from .base import register_op

_SLICE_COLUMNS = (
    "num_key_value_heads",
    "head_dim",
    "attn_dtype",
    "kv_cache_dtype",
    "block_size",
    "window_size",
)


def _unpack(slice_key: tuple[Any, ...], coords: tuple[float, ...]) -> dict[str, Any]:
    """Return the SOL keyword arguments both phases share."""
    num_kv_heads, head_dim, attn_dtype, kv_cache_dtype, _block_size, window_size = slice_key
    return dict(
        num_heads=int(coords[0]),
        batch_size=int(coords[1]),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        attn_dtype=attn_dtype,
        kv_cache_dtype=kv_cache_dtype,
        window_size=int(window_size),
    )


def _parse_attention_sweep(value: dict[str, Any]) -> Any:
    """Parse the shared `attention:` sweep subtree."""
    # Imported lazily: `srl.profiler.kernel.sweep` reaches the profiler package, and
    # a simulator run that only prices kernels must not load it. Four dots because this
    # module is `srl.cost.kernel.ops.attention`. The torch-free check verifies the depth.
    from ....profiler.kernel.sweep import AttentionSweep

    return AttentionSweep.from_dict(value)


def _measure_attention(model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
    """Measure both phases through both backends and write four tables."""
    from ....profiler.kernel.runner.attention import AttentionRunner

    return AttentionRunner(model, config).run(writer_factory=writer_factory)


@dataclass(frozen=True)
class ContextAttentionKernelOp:
    """Causal prefill attention, interpolated over prompt length.

    Frozen because the instance below is a registry singleton shared by every caller.
    """

    name = "context_attention"
    family = "attention"
    slice_columns = _SLICE_COLUMNS
    axis_columns = ("num_heads", "batch_size", "isl")
    fits_floor = True
    # Prefill batches are captured PIECEWISE, and `unified_attention_with_output` is a
    # splitting op, so this call runs outside the graph and pays its dispatch eagerly.
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
        Return the roofline of one causal prefill attention call.

        Args:
            spec (GpuSpec): Hardware the roofline is evaluated against.
            slice_key (tuple[Any, ...]): This op's `slice_columns` values.
            coords (tuple[float, ...]): `(num_heads, batch_size, isl)`.
            floor_s (float): Fitted dispatch floor in seconds, applied inside the `max`.

        Returns:
            SolComponents: Compute, memory, and floor components in seconds.
        """
        return context_attention_sol(
            spec, seq_len=int(coords[2]), floor_s=floor_s, **_unpack(slice_key, coords)
        )

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the shared `attention:` sweep subtree."""
        return _parse_attention_sweep(value)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Measure both attention phases through both backends."""
        return _measure_attention(model, config, writer_factory=writer_factory)


@dataclass(frozen=True)
class GenerationAttentionKernelOp:
    """Paged decode attention, interpolated over KV history depth.

    Frozen because the instance below is a registry singleton shared by every caller.
    """

    name = "generation_attention"
    family = "attention"
    slice_columns = _SLICE_COLUMNS
    axis_columns = ("num_heads", "batch_size", "step")
    fits_floor = True
    # A decode batch replays one FULL graph spanning the whole forward pass, attention
    # included, so the engine pays no per-call dispatch here. Measured 67 us event-timed
    # against 15 us replayed at batch 1, a 4.5x overcharge repeated once per layer.
    graph_replayed = True

    def sol(
        self,
        spec: GpuSpec,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """
        Return the roofline of one paged decode attention call.

        Args:
            spec (GpuSpec): Hardware the roofline is evaluated against.
            slice_key (tuple[Any, ...]): This op's `slice_columns` values.
            coords (tuple[float, ...]): `(num_heads, batch_size, step)`.
            floor_s (float): Fitted dispatch floor in seconds, applied inside the `max`.

        Returns:
            SolComponents: Compute, memory, and floor components in seconds.
        """
        return generation_attention_sol(
            spec, step=int(coords[2]), floor_s=floor_s, **_unpack(slice_key, coords)
        )

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the shared `attention:` sweep subtree."""
        return _parse_attention_sweep(value)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Measure both attention phases through both backends."""
        return _measure_attention(model, config, writer_factory=writer_factory)


CONTEXT_ATTENTION_OP = ContextAttentionKernelOp()
GENERATION_ATTENTION_OP = GenerationAttentionKernelOp()
register_op(CONTEXT_ATTENTION_OP)
register_op(GENERATION_ATTENTION_OP)
