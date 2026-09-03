"""The all_reduce op: ring-cost SOL, world_size in the slice, and a real query_comm."""

from __future__ import annotations

import pytest

from dryrun.cost.kernel.hardware import load_gpu_spec
from dryrun.cost.kernel.ops.base import op_for
from dryrun.cost.kernel.ops.collective import ALL_REDUCE_OP


@pytest.fixture(scope="module")
def spec():
    return load_gpu_spec("h20")


def test_the_op_is_registered_in_the_collective_family():
    assert op_for("all_reduce") is ALL_REDUCE_OP
    assert ALL_REDUCE_OP.family == "collective"


def test_world_size_is_categorical_not_an_axis():
    """Interpolating between 2 and 8 ranks would blend different ring topologies."""
    assert ALL_REDUCE_OP.slice_columns == ("dtype", "world_size")
    assert ALL_REDUCE_OP.axis_columns == ("message_bytes",)


def test_the_op_takes_a_fitted_floor():
    """NCCL has a well-known per-call launch and handshake latency."""
    assert ALL_REDUCE_OP.fits_floor is True


def test_sol_follows_the_ring_all_reduce_cost(spec):
    """A ring all-reduce moves 2*(N-1)/N * bytes per rank, not N*bytes."""
    message_bytes = 1 << 24
    for world_size in (2, 4, 8):
        got = ALL_REDUCE_OP.sol(
            spec, slice_key=("bfloat16", world_size), coords=(float(message_bytes),)
        )
        factor = 2.0 * (world_size - 1) / world_size
        expected = factor * message_bytes / spec.link_bandwidth_bytes_per_s
        assert got.memory_s == pytest.approx(expected, rel=1e-12)
        assert got.math_s == 0.0


def test_the_ring_factor_grows_with_world_size_but_stays_below_two(spec):
    """2*(N-1)/N is 1.0 at N=2 and approaches 2 as N grows; it must never exceed it."""
    message_bytes = 1 << 20
    previous = 0.0
    for world_size in (2, 4, 8, 16):
        got = ALL_REDUCE_OP.sol(
            spec, slice_key=("bfloat16", world_size), coords=(float(message_bytes),)
        ).memory_s
        assert got > previous
        previous = got
    limit = 2.0 * message_bytes / spec.link_bandwidth_bytes_per_s
    assert previous < limit


def test_a_single_rank_all_reduce_costs_no_transfer(spec):
    """At world_size 1 there is no collective; only the floor can bind."""
    got = ALL_REDUCE_OP.sol(spec, slice_key=("bfloat16", 1), coords=(1.0e6,), floor_s=5.0e-6)
    assert got.memory_s == 0.0
    assert got.latency_s == pytest.approx(5.0e-6, rel=1e-12)


def test_a_non_positive_world_size_is_rejected(spec):
    with pytest.raises(ValueError, match="world_size"):
        ALL_REDUCE_OP.sol(spec, slice_key=("bfloat16", 0), coords=(1.0e6,))


def test_a_non_positive_message_size_is_rejected(spec):
    with pytest.raises(ValueError, match="message_bytes"):
        ALL_REDUCE_OP.sol(spec, slice_key=("bfloat16", 4), coords=(0.0,))


def test_latency_is_the_last_table_column():
    assert ALL_REDUCE_OP.table_columns() == ("dtype", "world_size", "message_bytes", "latency_s")


def test_parse_sweep_builds_the_token_grid():
    sweep = ALL_REDUCE_OP.parse_sweep({"token_counts": [16, 4096]})
    assert sweep.token_counts == (16, 4096)


def test_a_comm_query_exposes_the_ops_slice_and_coords():
    from dryrun.cost.kernel.database import CommQuery

    query = CommQuery(world_size=4, message_bytes=1 << 20)
    assert query.op_name == "all_reduce"
    assert query.slice_key == ("bfloat16", 4)
    assert query.coords == (float(1 << 20),)


def test_a_comm_query_rejects_a_non_positive_world_size():
    from dryrun.cost.kernel.database import CommQuery
    from dryrun.cost.kernel.interp import KernelCoverageError

    with pytest.raises(KernelCoverageError, match="world_size"):
        CommQuery(world_size=0, message_bytes=1 << 20)


def test_query_comm_no_longer_raises_unconditionally(tmp_path):
    """It raised on every call as a placeholder; now it must reach the table machinery.

    Asserted behaviourally rather than by greping the method body: `query_comm` delegates to the
    shared `_price`, so the names `_grid` and `_sol` no longer appear in it. What proves it
    reaches the tables is the failure mode -- a missing collective table produces a coverage
    error naming the sweep to run, which a placeholder raise could not.
    """
    from dryrun.cost.kernel.database import CommQuery, KernelDatabase
    from dryrun.cost.kernel.interp import KernelCoverageError

    (tmp_path / "h20").mkdir()
    database = KernelDatabase.from_root(tmp_path, gpu="h20")
    with pytest.raises(KernelCoverageError, match="--family collective"):
        database.query_comm(CommQuery(world_size=2, message_bytes=1 << 20))


def test_a_comm_query_reads_the_vllm_table_by_default():
    """The dispatched kernel is the one a server runs, and it is not NCCL.

    Measured on H20 at world sizes 2, 4 and 8, `torch.distributed.all_reduce` is never what
    `CudaCommunicator.all_reduce` selects: the custom P2P kernel covers the whole decode range.
    A silent revert of this default to `nccl` would price a kernel no server invokes, and would
    be invisible -- both tables exist, so the wrong one still resolves and still interpolates.
    """
    from dryrun.cost.kernel.database import CommQuery

    assert CommQuery(world_size=2, message_bytes=1 << 20).backend == "vllm"


def test_the_nccl_control_table_stays_reachable():
    """Quantifying the old table's bias requires reading it, so the field stays overridable."""
    from dryrun.cost.kernel.database import CommQuery

    query = CommQuery(world_size=2, message_bytes=1 << 20, backend="nccl")
    assert query.backend == "nccl"
    # The backend picks a directory, not a slice, so it must not leak into the table key.
    assert query.slice_key == ("bfloat16", 2)


def test_every_measured_collective_backend_has_a_fallback_floor():
    """A floor-taking op with no fallback silently degrades to zero.

    For `all_reduce` a zero floor is a 220x SOL error at 4096 bytes, and the fitted utility then
    launders it into a plausible-looking MRE. Both the dispatched table and the control need an
    entry, because the backend selects which `FALLBACK_FLOORS_S` subtree is read.
    """
    from dryrun.cost.kernel.sol import FALLBACK_FLOORS_S

    assert ALL_REDUCE_OP.fits_floor is True
    for backend in ("vllm", "nccl"):
        assert "all_reduce" in FALLBACK_FLOORS_S[backend], backend

