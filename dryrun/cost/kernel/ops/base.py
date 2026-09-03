"""
The contract one measured kernel op must satisfy, and the registry of them.

Op-specific knowledge used to be spread across ten sites: a SOL function in `sol.py`, a
column-contract dataclass in `table.py`, a query dataclass and an `isinstance` branch in
`database.py`, a second SOL binding in `holdout.py`, a phase map in `floor.py`, a writer method
in `measurements.py`, a sweep field in `config.py`, a profiler entry point, and a CLI branch.
The count was the visible symptom; the defect was that `database._sol` and
`holdout._make_sol_fn` were two independent implementations of the same binding, and drift
between them would silently change utility on one side of the holdout round trip -- the exact
quantity the accuracy gate measures.

`sol()` on the op is that binding, once. Signature differences between families are absorbed
by each op unpacking its own `slice_key`, whose column order that same op declares, so no
shared `**kwargs` signature is needed.

This module must stay importable without `torch`: a simulator run prices kernels from parquet
and must not pay a 4.2-second GPU-stack import to do it. Ops that measure on a GPU import
their dependencies inside the measurement call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..interp import KernelCoverageError
from ..sol import SolComponents


@runtime_checkable
class KernelOp(Protocol):
    """One measured kernel family: its table shape, its roofline, and its floor policy."""

    name: str
    family: str
    slice_columns: tuple[str, ...]
    axis_columns: tuple[str, ...]
    fits_floor: bool

    def sol(
        self,
        spec: Any,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """Return the roofline components of one shape in this op's own coordinates."""

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse this op's YAML sweep subtree, raising on an invalid grid."""

    def table_columns(self) -> tuple[str, ...]:
        """Return the non-provenance columns of this op's perf table, latency last."""

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """
        Measure this op's whole family on the GPU and write its perf tables.

        Called once per family, not once per op: both attention phases come out of a single
        sweep that allocates one KV cache and times both against it, so a per-op call would
        measure the grid twice and could select a different backend for each phase. An op
        whose family holds only itself simply writes one table.

        Args:
            model (Any): `ModelDescriptor` for the architecture being profiled.
            config (Any): `KernelProfileConfig` carrying the sweep and measurement settings.
            writer_factory (Any): Builds a `PerfTableWriter` from
                `(framework, version, kernel_source)`. The kernel source is known only after
                backend selection, so a writer cannot be passed in directly.

        Returns:
            tuple[Any, ...]: The written parquet paths.
        """


OP_REGISTRY: dict[str, KernelOp] = {}


def register_op(op: KernelOp) -> None:
    """
    Add one op to the registry.

    Args:
        op (KernelOp): The op to register.

    Raises:
        KernelCoverageError: If an op with the same name is already registered.
    """
    if op.name in OP_REGISTRY:
        raise KernelCoverageError(
            f"Kernel op {op.name!r} is already registered. Two ops sharing a name would make "
            f"op_for's answer depend on import order."
        )
    OP_REGISTRY[op.name] = op


def op_for(name: str) -> KernelOp:
    """
    Return the op measured under one name.

    Args:
        name (str): Op name, e.g. `gemm` or `generation_attention`.

    Returns:
        KernelOp: The matching op.

    Raises:
        KernelCoverageError: If no op is registered under `name`.
    """
    try:
        return OP_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(OP_REGISTRY)) or "none"
        raise KernelCoverageError(f"Unknown kernel op {name!r}. Registered ops: {known}.") from exc


def ops_for_family(family: str) -> tuple[KernelOp, ...]:
    """
    Return every op writing into one family directory.

    Args:
        family (str): Family directory name, e.g. `attention`.

    Returns:
        tuple[KernelOp, ...]: Sorted by op name; empty when the family has no ops.
    """
    return tuple(sorted((op for op in OP_REGISTRY.values() if op.family == family), key=lambda op: op.name))


def families() -> tuple[str, ...]:
    """Return the sorted distinct family directory names across every registered op."""
    return tuple(sorted({op.family for op in OP_REGISTRY.values()}))


def ops_in(version_dir: str | Path) -> tuple[KernelOp, ...]:
    """
    Return the ops whose perf table exists in one version directory.

    Replaces parsing an op name out of a parquet stem at each discovery site. A parquet with
    no registered op is skipped rather than raising, because a stray file must not break a
    report; `tests/test_kernel_op_coverage.py` is what fails on a genuinely unregistered op.

    Args:
        version_dir (str | Path): A `{root}/{gpu}/{family}/{backend}/{version}/` directory.

    Returns:
        tuple[KernelOp, ...]: Sorted by op name; empty when the directory is absent.
    """
    directory = Path(version_dir)
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            (op for op in OP_REGISTRY.values() if (directory / f"{op.name}_perf.parquet").is_file()),
            key=lambda op: op.name,
        )
    )
