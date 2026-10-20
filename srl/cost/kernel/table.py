"""Load measured kernel perf tables from parquet and slice them into utility grids.

A table's columns split in two. Slice columns are categorical and matched exactly:
mixing two dtypes or two KV-head counts into one interpolation would blend unrelated
kernels. Axis columns are numeric and interpolated in log2 space. Latency is stored,
never utility, because utility depends on the GPU spec being priced against and a
table must remain reusable for a different target (D2).

Duplicate shapes keep the fastest measurement. A repeated point is a re-measurement,
and the slower one carries whatever transient noise the machine had at the time.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .interp import KernelCoverageError, UtilitySample
from .ops.base import KernelOp
from .surface import SURFACE_REGISTRY, UtilitySurface

_PROVENANCE_COLUMNS = ("framework", "version", "device", "op_name", "kernel_source")
_LATENCY_COLUMN = "latency_s"


def required_columns(op: KernelOp) -> tuple[str, ...]:
    """Return every column a conforming table for `op` must carry."""
    return _PROVENANCE_COLUMNS + op.table_columns()


class KernelTable:
    """One measured perf table, grouped into interpolable slices."""

    def __init__(
        self,
        op: KernelOp,
        slices: dict[tuple[Any, ...], dict[tuple[float, ...], float]],
        kernel_sources: tuple[str, ...],
        devices: tuple[str, ...],
    ) -> None:
        self._op = op
        self._slices = slices
        self._kernel_sources = kernel_sources
        self._devices = devices
        self._grids: dict[tuple[tuple[Any, ...], Callable[[tuple[float, ...]], float], str], UtilitySurface] = {}

    @classmethod
    def load(cls, path: str | Path, op: KernelOp) -> KernelTable:
        """
        Read a parquet perf table.

        Args:
            path (str | Path): Parquet file written by `PerfTableWriter`.
            op (KernelOp): The op whose column contract the file must satisfy.

        Returns:
            KernelTable: Grouped measurements ready for grid construction.

        Raises:
            KernelCoverageError: If the file is absent, a column is missing, or a
                latency is not positive.
        """
        source = Path(path)
        if not source.is_file():
            raise KernelCoverageError(f"Perf table {source} does not exist.")
        table = pq.read_table(source)
        needed = required_columns(op)
        missing = [name for name in needed if name not in table.column_names]
        if missing:
            # A missing column means the table predates a schema change, so the message names
            # the remedy rather than only the symptom. This matters most for the collective
            # family's `nodes`: read leniently as 1, a cross-node measurement would be
            # attributed to NVLink and the ~9x error would surface much later as an
            # unexplained model defect. Refusing is the only safe reading.
            raise KernelCoverageError(
                f"Perf table {source} is missing columns: {', '.join(missing)}. The table "
                f"predates the current {op.name!r} schema. Re-measure it with "
                f"`srl-profile kernels run --family {op.family}` rather than reading it "
                f"with assumed values."
            )
        columns = {name: table.column(name).to_pylist() for name in needed}

        slices: dict[tuple[Any, ...], dict[tuple[float, ...], float]] = {}
        for index, latency in enumerate(columns[_LATENCY_COLUMN]):
            value = float(latency)
            if value <= 0.0:
                raise KernelCoverageError(
                    f"Perf table {source} row {index} has non-positive latency {value!r}."
                )
            slice_key = tuple(columns[name][index] for name in op.slice_columns)
            coords = tuple(float(columns[name][index]) for name in op.axis_columns)
            bucket = slices.setdefault(slice_key, {})
            previous = bucket.get(coords)
            bucket[coords] = value if previous is None else min(previous, value)

        sources = tuple(sorted({str(item) for item in columns["kernel_source"]}))
        device_values = tuple(sorted({str(item) for item in columns["device"]}))
        return cls(op, slices, sources, device_values)

    @property
    def op(self) -> KernelOp:
        """Return the op this table was loaded against."""
        return self._op

    @property
    def row_count(self) -> int:
        """Return the number of distinct measured shapes across all slices."""
        return sum(len(bucket) for bucket in self._slices.values())

    def kernel_sources(self) -> tuple[str, ...]:
        """Return the sorted distinct `kernel_source` values in this table."""
        return self._kernel_sources

    def devices(self) -> tuple[str, ...]:
        """Return the sorted distinct `device` values in this table."""
        return self._devices

    def slice_keys(self) -> tuple[tuple[Any, ...], ...]:
        """Return the sorted categorical slice keys present in this table."""
        try:
            return tuple(sorted(self._slices))
        except TypeError:
            return tuple(sorted(self._slices, key=repr))

    def rows(self, slice_key: tuple[Any, ...]) -> tuple[tuple[tuple[float, ...], float], ...]:
        """
        Return one slice's measurements.

        Args:
            slice_key (tuple[Any, ...]): Values in `op.slice_columns` order.

        Returns:
            tuple[tuple[tuple[float, ...], float], ...]: `(coords, latency_s)` pairs,
                sorted by coordinate.

        Raises:
            KernelCoverageError: If the slice was never measured.
        """
        bucket = self._slices.get(slice_key)
        if bucket is None:
            available = ", ".join(repr(key) for key in self.slice_keys())
            raise KernelCoverageError(
                f"No {self._op.name} rows for slice {slice_key!r}. Measured: {available}."
            )
        return tuple(sorted(bucket.items()))

    def grid(
        self,
        slice_key: tuple[Any, ...],
        sol_fn: Callable[[tuple[float, ...]], float],
        *,
        surface: str = "shepard",
    ) -> UtilitySurface:
        """
        Build (and memoize) the utility grid of one slice.

        Args:
            slice_key (tuple[Any, ...]): Values in `op.slice_columns` order.
            sol_fn (Callable[[tuple[float, ...]], float]): Maps axis coordinates to a
                speed-of-light latency in seconds. Memoization keys on the callable's
                identity, so calling with the same object returns the same `UtilitySurface`.
            surface (str): Interpolation strategy name from `SURFACE_REGISTRY`. Memoization
                keys on this too, so two strategies over one slice do not collide.

        Returns:
            UtilitySurface: Utilities `sol_fn(coords) / measured_latency`.

        Raises:
            KernelCoverageError: If the slice was never measured, if `sol_fn` returns
                a non-finite value for any coordinate in the slice, or if `surface` is
                not a known strategy name.
        """
        surface_type = SURFACE_REGISTRY.get(surface)
        if surface_type is None:
            known = ", ".join(sorted(SURFACE_REGISTRY))
            raise KernelCoverageError(f"Unknown utility surface {surface!r}. Known: {known}.")
        cache_key = (slice_key, sol_fn, surface)
        cached = self._grids.get(cache_key)
        if cached is not None:
            return cached
        samples = []
        for coords, latency in self.rows(slice_key):
            # Skip rows where any axis is non-positive: they cannot be placed in log2
            # coordinate space. Generation attention at step=0 is the canonical case.
            if any(v <= 0.0 for v in coords):
                continue
            utility = sol_fn(coords) / latency
            if not math.isfinite(utility):
                raise KernelCoverageError(
                    f"SOL function returned non-finite utility {utility!r} at coords {coords!r}."
                )
            samples.append(UtilitySample(coords=coords, utility=utility))
        grid = surface_type(samples)
        self._grids[cache_key] = grid
        return grid
