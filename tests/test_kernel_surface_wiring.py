"""The surface strategy must reach the table and the database without moving the default."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from srl.cost.kernel.database import GemmQuery, KernelDatabase
from srl.cost.kernel.interp import KernelCoverageError
from srl.cost.kernel.ops.gemm import GEMM_OP
from srl.cost.kernel.surface import PchipSurface, ShepardSurface
from srl.cost.kernel.table import KernelTable

_SLICE = ("bfloat16", 4096, 4096)
# Utility must VARY across m, or Shepard and PCHIP agree everywhere and the wiring test
# cannot tell them apart. These latencies are chosen so utility rises then flattens, the
# shape the real GEMM tables show.
_ROWS = (
    {"m": 1, "n": 4096, "k": 4096, "latency_s": 2.0e-5},
    {"m": 16, "n": 4096, "k": 4096, "latency_s": 2.6e-5},
    {"m": 64, "n": 4096, "k": 4096, "latency_s": 4.0e-5},
    {"m": 1024, "n": 4096, "k": 4096, "latency_s": 2.2e-4},
    {"m": 4096, "n": 4096, "k": 4096, "latency_s": 8.4e-4},
)


def _write_table(root):
    """Write a GEMM parquet into the `{gpu}/{family}/{backend}/{version}/` layout."""
    path = root / "h20" / "gemm" / "vllm" / "0.26.1.dev0" / "gemm_perf.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(_ROWS)
    pq.write_table(
        pa.table(
            {
                "framework": ["vllm"] * count,
                "version": ["0.26.1.dev0"] * count,
                "device": ["NVIDIA H20"] * count,
                "op_name": ["gemm"] * count,
                "kernel_source": ["torch.nn.functional.linear"] * count,
                "gemm_dtype": ["bfloat16"] * count,
                "m": [row["m"] for row in _ROWS],
                "n": [row["n"] for row in _ROWS],
                "k": [row["k"] for row in _ROWS],
                "latency_s": [row["latency_s"] for row in _ROWS],
            }
        ),
        path,
    )
    return path


@pytest.fixture
def table_root(tmp_path):
    _write_table(tmp_path)
    return tmp_path


@pytest.fixture
def table(table_root):
    return KernelTable.load(
        table_root / "h20" / "gemm" / "vllm" / "0.26.1.dev0" / "gemm_perf.parquet", GEMM_OP
    )


def _sol(coords):
    return 2.0e-4 * (coords[0] / 1024.0)


def test_the_default_surface_is_shepard(table):
    assert isinstance(table.grid(_SLICE, _sol), ShepardSurface)


def test_the_surface_can_be_selected_by_name(table):
    assert isinstance(table.grid(_SLICE, _sol, surface="pchip"), PchipSurface)


def test_the_two_surfaces_are_memoized_separately(table):
    """A shared cache key would return a Shepard grid for a pchip request."""
    shepard = table.grid(_SLICE, _sol, surface="shepard")
    pchip = table.grid(_SLICE, _sol, surface="pchip")
    assert shepard is table.grid(_SLICE, _sol, surface="shepard")
    assert pchip is table.grid(_SLICE, _sol, surface="pchip")
    assert shepard is not pchip


def test_an_unknown_surface_name_is_rejected(table):
    with pytest.raises(KernelCoverageError, match="pchp"):
        table.grid(_SLICE, _sol, surface="pchp")


def test_both_surfaces_reproduce_a_measured_row(table):
    """Whatever the strategy, a measured shape must price to its measurement."""
    coords, latency = table.rows(_SLICE)[1]
    for name in ("shepard", "pchip"):
        grid = table.grid(_SLICE, _sol, surface=name)
        assert _sol(coords) / grid.utility(coords) == pytest.approx(latency, rel=1e-9)


def test_the_database_passes_its_surface_choice_down(table_root):
    """Selecting pchip must change a prediction between measured points, not just the type."""
    shepard = KernelDatabase.from_root(table_root, gpu="h20", utility_surface="shepard")
    pchip = KernelDatabase.from_root(table_root, gpu="h20", utility_surface="pchip")
    query = GemmQuery(m=48, n=4096, k=4096)
    assert pchip.query_gemm(query) != pytest.approx(shepard.query_gemm(query), rel=1e-6)


def test_the_database_default_is_shepard(table_root):
    plain = KernelDatabase.from_root(table_root, gpu="h20")
    shepard = KernelDatabase.from_root(table_root, gpu="h20", utility_surface="shepard")
    query = GemmQuery(m=48, n=4096, k=4096)
    assert plain.query_gemm(query) == pytest.approx(shepard.query_gemm(query), rel=1e-12)
