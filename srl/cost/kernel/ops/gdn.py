"""
Gated delta net: the linear-attention layer a hybrid model uses for most of its depth.

A separate family from `attention`, not a slice of it, because the two are different kernels
with different cost functions. Qwen3.5-35B-A3B runs softmax attention on 10 of 40 layers and
this on the other 30, so a cost model that priced every layer as softmax attention would
overstate a long prompt by roughly `seq_len / chunk_size`. At 32k tokens the SOL difference
is 128x.

The two ops here are also shaped differently from the two attention ops, in a way that is
worth stating because it looks like an omission:

- `gdn_context` has ONE axis, `total_tokens`. The kernel is varlen, so a packed batch is one
  row behind `cu_seqlens` and there is no batch dimension to sweep.
- `gdn_decode` has ONE axis, `batch_size`. There is no `step`: the recurrent state is a fixed
  `num_key_heads x head_dim x head_dim` per request whatever the KV depth, so a decode step
  costs the same at depth 128 as at 32768.

Both take a fitted floor, for the same reason the attention ops do: at small shapes the kernel
is dispatch-limited and the roofline alone underpredicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..hardware import GpuSpec
from ..sol import SolComponents, gdn_context_sol, gdn_decode_sol
from .base import register_op

# `chunk_size` is a slice column rather than a constant folded into the SOL: it is a property
# of the kernel version, and a future `fla` changing it would change the intra-chunk term.
# Keeping it in the key means the two regimes cannot silently share a table.
_SLICE_COLUMNS = (
    "num_key_heads",
    "num_value_heads",
    "head_dim",
    "attn_dtype",
    "chunk_size",
)


def _unpack(slice_key: tuple[Any, ...]) -> dict[str, Any]:
    """Return the SOL keyword arguments both phases share."""
    num_key_heads, num_value_heads, head_dim, attn_dtype, _chunk_size = slice_key
    return dict(
        num_key_heads=int(num_key_heads),
        num_value_heads=int(num_value_heads),
        head_dim=int(head_dim),
        attn_dtype=attn_dtype,
    )


def _parse_gdn_sweep(value: dict[str, Any]) -> Any:
    """Parse the shared `gdn:` sweep subtree."""
    # Imported lazily: the cost path prices kernels from parquet and must not load the
    # profiler package to do it. Four dots because this module is
    # `srl.cost.kernel.ops.gdn`. `tests/test_kernel_package_imports.py` checks the depth.
    from ....profiler.kernel.sweep import GdnSweep

    return GdnSweep.from_dict(value)


def _measure_gdn(model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
    """Measure both phases and write two tables."""
    from ....profiler.kernel.runner.gdn import profile_gdn

    return profile_gdn(model, config, writer_factory=writer_factory)


@dataclass(frozen=True)
class GdnContextKernelOp:
    """Chunked gated-delta-net prefill, interpolated over the packed token total.

    Frozen because the instance below is a registry singleton shared by every caller.
    """

    name = "gdn_context"
    family = "gdn"
    slice_columns = _SLICE_COLUMNS
    axis_columns = ("total_tokens",)
    fits_floor = True
    # `vllm::qwen_gdn_attention_core` is a splitting op, so a piecewise-captured prefill
    # runs this eagerly and does pay FlashInfer's CUTLASS DSL marshaling.
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
        Return the roofline of one chunked GDN prefill call.

        Args:
            spec (GpuSpec): Hardware the roofline is evaluated against.
            slice_key (tuple[Any, ...]): This op's `slice_columns` values.
            coords (tuple[float, ...]): `(total_tokens,)`.
            floor_s (float): Fitted dispatch floor in seconds, applied inside the `max`.

        Returns:
            SolComponents: Compute, memory, and floor components in seconds.
        """
        return gdn_context_sol(
            spec,
            total_tokens=int(coords[0]),
            chunk_size=int(slice_key[4]),
            floor_s=floor_s,
            **_unpack(slice_key),
        )

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the shared `gdn:` sweep subtree."""
        return _parse_gdn_sweep(value)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Measure both GDN phases."""
        return _measure_gdn(model, config, writer_factory=writer_factory)


@dataclass(frozen=True)
class GdnDecodeKernelOp:
    """Recurrent gated-delta-net decode, interpolated over batch size only.

    Frozen because the instance below is a registry singleton shared by every caller.
    """

    name = "gdn_decode"
    family = "gdn"
    slice_columns = _SLICE_COLUMNS
    axis_columns = ("batch_size",)
    fits_floor = True
    # Inside the FULL decode graph. The overcharge here was the largest in the tree: 25.8 us
    # event-timed against 3.9 us replayed at batch 1, times 30 GDN layers.
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
        Return the roofline of one recurrent GDN decode step.

        Args:
            spec (GpuSpec): Hardware the roofline is evaluated against.
            slice_key (tuple[Any, ...]): This op's `slice_columns` values.
            coords (tuple[float, ...]): `(batch_size,)`. No history depth: the state is a
                fixed size, so a decode step does not grow with the KV length.
            floor_s (float): Fitted dispatch floor in seconds, applied inside the `max`.

        Returns:
            SolComponents: Compute, memory, and floor components in seconds.
        """
        return gdn_decode_sol(
            spec, batch_size=int(coords[0]), floor_s=floor_s, **_unpack(slice_key)
        )

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the shared `gdn:` sweep subtree."""
        return _parse_gdn_sweep(value)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Measure both GDN phases."""
        return _measure_gdn(model, config, writer_factory=writer_factory)


GDN_CONTEXT_OP = GdnContextKernelOp()
GDN_DECODE_OP = GdnDecodeKernelOp()

register_op(GDN_CONTEXT_OP)
register_op(GDN_DECODE_OP)
