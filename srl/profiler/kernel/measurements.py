"""Write measured kernel latencies to parquet perf tables.

Writes are atomic: rows go to a sibling `.tmp` file which then replaces the target.
A multi-hour sweep that dies partway must not leave a truncated parquet behind, because
`KernelTable.load` cannot tell a truncated table from a small one and would silently
interpolate over a hole.

Column order and naming come from the op itself, so the reader and writer cannot drift apart,
and a row whose keys do not match the op's declaration is rejected at write time rather than
at read time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from srl.cost.kernel.hardware import gpu_slug
from srl.cost.kernel.ops.base import op_for


@dataclass(frozen=True)
class PerfTableWriter:
    """Write perf tables for one (device, framework, version, kernel_source)."""

    output_root: Path
    device: str
    framework: str
    version: str
    kernel_source: str

    def table_path(self, op_name: str) -> Path:
        """
        Return the parquet path of one op's table.

        Args:
            op_name (str): A registered op name.

        Returns:
            Path: `{output_root}/{gpu}/{family}/{framework}/{version}/{op_name}_perf.parquet`.

        Raises:
            KernelCoverageError: If the op is not registered.
        """
        op = op_for(op_name)
        return (
            Path(self.output_root)
            / gpu_slug(self.device)
            / op.family
            / self.framework
            / self.version
            / f"{op_name}_perf.parquet"
        )

    def write(self, op_name: str, rows: Sequence[Mapping[str, Any]]) -> Path:
        """
        Write one op's perf table.

        Args:
            op_name (str): A registered op name. Selects the family directory and the
                expected columns.
            rows (Sequence[Mapping[str, Any]]): One mapping per measured shape, carrying
                exactly the op's slice and axis columns plus `latency_s`.

        Returns:
            Path: The written parquet path.

        Raises:
            ValueError: If `rows` is empty, or any row's keys do not match the op's columns
                exactly.
            KernelCoverageError: If `op_name` is not registered.
        """
        op = op_for(op_name)
        if not rows:
            raise ValueError(f"Refusing to write a {op_name} perf table with no rows.")
        expected = set(op.table_columns())
        # Validate here rather than trusting the caller. The per-family methods this replaces
        # populated up to eleven columns by hand, so a misspelled key produced a table that
        # only failed when something later tried to read it: possibly hours after the sweep.
        for index, row in enumerate(rows):
            keys = set(row)
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            if missing or extra:
                parts = []
                if missing:
                    parts.append(f"missing {', '.join(missing)}")
                if extra:
                    parts.append(f"unexpected {', '.join(extra)}")
                raise ValueError(
                    f"Row {index} of a {op_name} perf table has {' and '.join(parts)}. "
                    f"Expected exactly: {', '.join(op.table_columns())}."
                )
        columns = {name: [row[name] for row in rows] for name in op.table_columns()}
        return self._write(op_name, len(rows), columns)

    def _write(self, op_name: str, count: int, columns: dict[str, list]) -> Path:
        payload = {
            "framework": [self.framework] * count,
            "version": [self.version] * count,
            "device": [self.device] * count,
            "op_name": [op_name] * count,
            "kernel_source": [self.kernel_source] * count,
            **columns,
        }
        destination = self.table_path(op_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.tmp")
        try:
            pq.write_table(pa.table(payload), temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
