"""Every query method must resolve the backend identically on both sides of SOL/utility.

`_grid` bakes `query.backend or default_backend` into the closure it derives utility from, so a
query method that resolved the backend differently for the numerator would take its floor from
one backend's floors.json while the denominator's utility came from another. The result is a
wrong latency that still looks self-consistent from either side, which is exactly the drift the
`KernelOp` refactor exists to make impossible.

A synthetic floor-taking op carries the test rather than a real one: GEMM declares
`fits_floor = False` so the bug is latent for today's op set, and the real ops are frozen
dataclasses whose `fits_floor` cannot be monkeypatched. Registering an op with a floor that
binds at every measured point makes the misbinding a large, unambiguous number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from srl.cost.kernel.database import KernelDatabase
from srl.cost.kernel.ops.base import OP_REGISTRY, register_op
from srl.cost.kernel.sol import SolComponents

_VERSION = "0.0.1"
# The backend whose floor is fitted for the query below, and the one the database defaults to.
# They must differ, because a query method that drops the resolved backend silently falls back
# to the default and a single-backend fixture could not tell the two apart.
_QUERY_BACKEND = "probe_narrow"
_DEFAULT_BACKEND = "probe_wide"
_FLOORS_S = {_DEFAULT_BACKEND: 50e-6, _QUERY_BACKEND: 5e-6}
_MEASURED = {1024.0: 6.0e-5, 2048.0: 7.0e-5, 4096.0: 9.0e-5}


class _FloorProbeOp:
    """A floor-taking op whose floor binds at every measured point."""

    name = "floor_probe"
    family = "floor_probe"
    slice_columns = ("dtype",)
    axis_columns = ("n",)
    fits_floor = True

    def sol(
        self,
        spec: Any,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """Return a negligible bandwidth term so the floor is always the binding component."""
        return SolComponents(math_s=0.0, memory_s=coords[0] * 1e-15, floor_s=floor_s)

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Reject sweeps. This op is never measured on a GPU."""
        raise NotImplementedError("floor_probe is a test fixture and has no sweep.")

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Reject measurement. This op is never measured on a GPU."""
        raise NotImplementedError("floor_probe is a test fixture and is never measured.")


@dataclass(frozen=True)
class _ProbeQuery:
    """A query against `_FloorProbeOp`, shaped like every real query dataclass."""

    n: float
    backend: str | None

    @property
    def op_name(self) -> str:
        """Return the parquet op name this query reads."""
        return "floor_probe"

    @property
    def slice_key(self) -> tuple[Any, ...]:
        """Return the categorical key in this op's slice order."""
        return ("bfloat16",)

    @property
    def coords(self) -> tuple[float, ...]:
        """Return interpolation coordinates in this op's axis order."""
        return (self.n,)


@pytest.fixture
def probe_op():
    """Register `_FloorProbeOp` for one test and remove it afterwards."""
    op = _FloorProbeOp()
    register_op(op)
    try:
        yield op
    finally:
        OP_REGISTRY.pop(op.name, None)


def _write_probe_tables(root: Path) -> None:
    """Write one identical perf table plus a distinct fitted floor under each probe backend."""
    coords = sorted(_MEASURED)
    for backend, floor_s in _FLOORS_S.items():
        version_dir = root / "h20" / "floor_probe" / backend / _VERSION
        version_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "framework": [backend] * len(coords),
                    "version": [_VERSION] * len(coords),
                    "device": ["NVIDIA H20"] * len(coords),
                    "op_name": ["floor_probe"] * len(coords),
                    "kernel_source": ["probe"] * len(coords),
                    "dtype": ["bfloat16"] * len(coords),
                    "n": coords,
                    "latency_s": [_MEASURED[value] for value in coords],
                }
            ),
            version_dir / "floor_probe_perf.parquet",
        )
        (version_dir / "floors.json").write_text(
            json.dumps({"floor_probe": floor_s}), encoding="utf-8"
        )


@pytest.mark.parametrize("method_name", ["query_gemm", "query_attention", "query_comm"])
def test_a_floor_taking_query_takes_its_floor_from_the_query_backend(tmp_path, probe_op, method_name):
    """Every query method must price through the same backend its utility was derived under.

    At a measured point the round trip `SOL / (SOL / latency)` returns the stored latency
    exactly. That identity holds only when both halves resolve the same backend, so a query
    method that dropped the resolved backend would price 10x high here: the ratio of the two
    fitted floors. The three public methods are parametrized because they are three entry points
    into one implementation, and the bug this pins was a method that omitted the argument.
    """
    _write_probe_tables(tmp_path)
    database = KernelDatabase.from_root(tmp_path, gpu="h20", default_backend=_DEFAULT_BACKEND)
    query = _ProbeQuery(n=2048.0, backend=_QUERY_BACKEND)

    priced = getattr(database, method_name)(query)

    assert priced == pytest.approx(_MEASURED[2048.0], rel=1e-9), (
        f"{method_name} priced a measured point at {priced!r} instead of "
        f"{_MEASURED[2048.0]!r}, so its numerator and denominator disagree on the backend."
    )


def test_the_probe_floor_actually_binds(tmp_path, probe_op):
    """Guard the fixture: a floor that never binds would make the test above vacuous."""
    database = KernelDatabase.from_root(tmp_path, gpu="h20", default_backend=_DEFAULT_BACKEND)
    components = probe_op.sol(
        database.gpu_spec, slice_key=("bfloat16",), coords=(2048.0,), floor_s=_FLOORS_S[_QUERY_BACKEND]
    )
    assert components.latency_s == pytest.approx(_FLOORS_S[_QUERY_BACKEND])
    assert _FLOORS_S[_DEFAULT_BACKEND] != _FLOORS_S[_QUERY_BACKEND]


def test_the_query_methods_share_one_pricing_implementation(probe_op):
    """Three copies of three lines is the template a sixth floor-taking op would inherit from."""
    import inspect

    for method_name in ("query_gemm", "query_attention", "query_comm"):
        source = inspect.getsource(getattr(KernelDatabase, method_name))
        assert "_price(query)" in source, (
            f"{method_name} does not delegate to _price, so it can resolve the backend "
            f"differently from the utility surface it divides by."
        )
        assert "_sol(" not in source, f"{method_name} calls _sol directly instead of via _price."
