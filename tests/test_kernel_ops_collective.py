"""The collective ops: ring-cost SOL, topology in the slice, and a real query_comm.

Every assertion from the single-op version of this file is preserved. The signature changes
are mechanical (`slice_key` gains `nodes`, `CommQuery` gains `op` and `nodes`). The added
tests cover the four new ops and the node-boundary term.
"""

from __future__ import annotations

import pytest

from srl.cost.kernel.hardware import load_gpu_spec
from srl.cost.kernel.ops.base import op_for, ops_for_family
from srl.cost.kernel.ops.collective import (
    ALL_GATHER_OP,
    ALL_REDUCE_OP,
    BROADCAST_OP,
    P2P_OP,
    REDUCE_SCATTER_OP,
)

_COLLECTIVES = (ALL_REDUCE_OP, REDUCE_SCATTER_OP, ALL_GATHER_OP, BROADCAST_OP, P2P_OP)


@pytest.fixture(scope="module")
def spec():
    return load_gpu_spec("h20")


@pytest.mark.parametrize("op", _COLLECTIVES, ids=lambda op: op.name)
def test_the_op_is_registered_in_the_collective_family(op):
    assert op_for(op.name) is op
    assert op.family == "collective"


def test_every_collective_the_two_backends_issue_is_registered():
    """A missing op cannot be priced, and Megatron issues four of these five.

    `megatron/core/tensor_parallel/layers.py` calls reduce-scatter and all-gather whenever
    sequence parallelism is on, which the Megatron profiler enables for every tp>1 run, so
    an all_reduce-only family could not price a tensor-parallel training step at all.
    """
    assert {op.name for op in ops_for_family("collective")} == {
        "all_reduce",
        "reduce_scatter",
        "all_gather",
        "broadcast",
        "p2p",
    }


@pytest.mark.parametrize("op", _COLLECTIVES, ids=lambda op: op.name)
def test_topology_is_categorical_not_an_axis(op):
    """Interpolating between 2 and 8 ranks, or 1 and 2 nodes, would blend different schedules."""
    assert op.slice_columns == ("dtype", "world_size", "nodes")
    assert op.axis_columns == ("message_bytes",)


@pytest.mark.parametrize("op", _COLLECTIVES, ids=lambda op: op.name)
def test_the_op_takes_a_fitted_floor(op):
    """Every collective pays a per-call launch and handshake latency."""
    assert op.fits_floor is True


def test_sol_follows_the_ring_all_reduce_cost(spec):
    """A ring all-reduce moves 2*(N-1)/N * bytes per rank, not N*bytes."""
    message_bytes = 1 << 24
    for world_size in (2, 4, 8):
        got = ALL_REDUCE_OP.sol(
            spec, slice_key=("bfloat16", world_size, 1), coords=(float(message_bytes),)
        )
        factor = 2.0 * (world_size - 1) / world_size
        expected = factor * message_bytes / spec.link_bandwidth_bytes_per_s
        assert got.memory_s == pytest.approx(expected, rel=1e-12)
        assert got.math_s == 0.0


def test_the_ring_factor_grows_with_world_size_but_stays_below_two(spec):
    """2*(N-1)/N is 1.0 at N=2 and approaches 2 as N grows. It must never exceed it."""
    message_bytes = 1 << 20
    previous = 0.0
    for world_size in (2, 4, 8, 16):
        got = ALL_REDUCE_OP.sol(
            spec, slice_key=("bfloat16", world_size, 1), coords=(float(message_bytes),)
        ).memory_s
        assert got > previous
        previous = got
    limit = 2.0 * message_bytes / spec.link_bandwidth_bytes_per_s
    assert previous < limit


def test_the_halves_of_an_all_reduce_each_cost_half(spec):
    """reduce_scatter and all_gather are the two phases of a ring all-reduce."""
    message_bytes = 1 << 22
    key = ("bfloat16", 8, 1)
    whole = ALL_REDUCE_OP.sol(spec, slice_key=key, coords=(float(message_bytes),)).memory_s
    for half in (REDUCE_SCATTER_OP, ALL_GATHER_OP):
        got = half.sol(spec, slice_key=key, coords=(float(message_bytes),)).memory_s
        assert got == pytest.approx(whole / 2.0, rel=1e-12)


@pytest.mark.parametrize("op", (BROADCAST_OP, P2P_OP), ids=lambda op: op.name)
def test_broadcast_and_p2p_do_not_scale_with_world_size(spec, op):
    """Neither is a ring reduction, so neither pays `(N-1)/N`.

    A pipelined broadcast delivers the payload once to each rank, and a p2p send is a single
    directed transfer. Giving either the reduce-scatter factor would understate it by up to
    2x at large N, so the flat factor is asserted rather than left implicit.
    """
    message_bytes = float(1 << 20)
    expected = message_bytes / spec.link_bandwidth_bytes_per_s
    for world_size in (2, 8, 24):
        got = op.sol(spec, slice_key=("bfloat16", world_size, 1), coords=(message_bytes,))
        assert got.memory_s == pytest.approx(expected, rel=1e-12)


def test_a_single_rank_all_reduce_costs_no_transfer(spec):
    """At world_size 1 there is no collective. Only the floor can bind."""
    got = ALL_REDUCE_OP.sol(spec, slice_key=("bfloat16", 1, 1), coords=(1.0e6,), floor_s=5.0e-6)
    assert got.memory_s == 0.0
    assert got.latency_s == pytest.approx(5.0e-6, rel=1e-12)


def test_one_node_reproduces_the_pre_topology_formula_exactly(spec):
    """The neutrality gate rests on this being an identity, not an approximation.

    `nodes == 1` is special-cased rather than derived, because `nodes / world_size` would
    read 1/8 at eight ranks on one node and charge an intra-node sweep for NIC traffic it
    never paid, perturbing the very numbers
    `tests/test_collective_baseline_regression.py` pins.
    """
    message_bytes = float(1 << 17)
    for op in _COLLECTIVES:
        for world_size in (2, 4, 8):
            got = op.sol(
                spec, slice_key=("bfloat16", world_size, 1), coords=(message_bytes,)
            ).memory_s
            legacy = (
                op.traffic_factor(world_size) * message_bytes / spec.link_bandwidth_bytes_per_s
            )
            assert got == legacy, f"{op.name} at world_size {world_size} is not bit-identical"


def test_crossing_a_node_boundary_costs_more_than_staying_inside_one(spec):
    """The NIC is ~9x slower than NVLink here, so topology cannot be a free dimension."""
    message_bytes = float(1 << 26)
    intra = ALL_REDUCE_OP.sol(
        spec, slice_key=("bfloat16", 16, 1), coords=(message_bytes,)
    ).memory_s
    for nodes in (2, 4):
        crossed = ALL_REDUCE_OP.sol(
            spec, slice_key=("bfloat16", 16, nodes), coords=(message_bytes,)
        ).memory_s
        assert crossed > intra, f"nodes={nodes} priced no higher than a single node"


def test_more_nodes_at_a_fixed_world_size_never_gets_cheaper(spec):
    """Spreading the same ranks over more nodes puts more of the ring on the slow link."""
    message_bytes = float(1 << 26)
    previous = 0.0
    for nodes in (1, 2, 4, 8):
        got = ALL_REDUCE_OP.sol(
            spec, slice_key=("bfloat16", 8, nodes), coords=(message_bytes,)
        ).memory_s
        assert got >= previous, f"nodes={nodes} is cheaper than nodes<{nodes}"
        previous = got


def test_a_non_positive_world_size_is_rejected(spec):
    with pytest.raises(ValueError, match="world_size"):
        ALL_REDUCE_OP.sol(spec, slice_key=("bfloat16", 0, 1), coords=(1.0e6,))


def test_a_non_positive_node_count_is_rejected(spec):
    with pytest.raises(ValueError, match="nodes"):
        ALL_REDUCE_OP.sol(spec, slice_key=("bfloat16", 4, 0), coords=(1.0e6,))


def test_more_nodes_than_ranks_is_rejected(spec):
    """Describes no placement: every node in the group must hold at least one rank."""
    with pytest.raises(ValueError, match="exceeds world_size"):
        ALL_REDUCE_OP.sol(spec, slice_key=("bfloat16", 2, 4), coords=(1.0e6,))


def test_a_non_positive_message_size_is_rejected(spec):
    with pytest.raises(ValueError, match="message_bytes"):
        ALL_REDUCE_OP.sol(spec, slice_key=("bfloat16", 4, 1), coords=(0.0,))


@pytest.mark.parametrize("op", _COLLECTIVES, ids=lambda op: op.name)
def test_latency_is_the_last_table_column(op):
    assert op.table_columns() == (
        "dtype",
        "world_size",
        "nodes",
        "message_bytes",
        "latency_s",
    )


def test_parse_sweep_builds_the_message_and_topology_grid():
    sweep = ALL_REDUCE_OP.parse_sweep(
        {"message_bytes": [4096, 1 << 20], "topologies": [[2, 1], [8, 1]]}
    )
    assert sweep.message_bytes == (4096, 1 << 20)
    assert sweep.topologies == ((2, 1), (8, 1))


def test_a_comm_query_exposes_the_ops_slice_and_coords():
    from srl.cost.kernel.database import CommQuery

    query = CommQuery(op="all_reduce", world_size=4, nodes=1, message_bytes=1 << 20)
    assert query.op_name == "all_reduce"
    assert query.slice_key == ("bfloat16", 4, 1)
    assert query.coords == (float(1 << 20),)


def test_a_comm_query_names_the_op_it_prices():
    """One query type for five ops, so the op must be explicit rather than implied."""
    from srl.cost.kernel.database import CommQuery

    for name in ("reduce_scatter", "all_gather", "broadcast", "p2p"):
        query = CommQuery(op=name, world_size=8, nodes=2, message_bytes=1 << 20)
        assert query.op_name == name
        assert query.slice_key == ("bfloat16", 8, 2)


def test_a_comm_query_rejects_a_non_collective_op():
    """Caught at construction so a typo is not mistaken for an unswept topology."""
    from srl.cost.kernel.database import CommQuery
    from srl.cost.kernel.interp import KernelCoverageError

    with pytest.raises(KernelCoverageError, match="not a collective"):
        CommQuery(op="gemm", world_size=2, nodes=1, message_bytes=1 << 20)


def test_a_comm_query_rejects_a_non_positive_world_size():
    from srl.cost.kernel.database import CommQuery
    from srl.cost.kernel.interp import KernelCoverageError

    with pytest.raises(KernelCoverageError, match="world_size"):
        CommQuery(op="all_reduce", world_size=0, nodes=1, message_bytes=1 << 20)


def test_a_comm_query_rejects_more_nodes_than_ranks():
    from srl.cost.kernel.database import CommQuery
    from srl.cost.kernel.interp import KernelCoverageError

    with pytest.raises(KernelCoverageError, match="exceeds world_size"):
        CommQuery(op="all_reduce", world_size=2, nodes=8, message_bytes=1 << 20)


def test_query_comm_no_longer_raises_unconditionally(tmp_path):
    """It raised on every call as a placeholder. Now it must reach the table machinery.

    Asserted behaviourally rather than by greping the method body: `query_comm` delegates to the
    shared `_price`, so the names `_grid` and `_sol` no longer appear in it. What proves it
    reaches the tables is the failure mode: a missing collective table produces a coverage
    error naming the sweep to run, which a placeholder raise could not.
    """
    from srl.cost.kernel.database import CommQuery, KernelDatabase
    from srl.cost.kernel.interp import KernelCoverageError

    (tmp_path / "h20").mkdir()
    database = KernelDatabase.from_root(tmp_path, gpu="h20")
    with pytest.raises(KernelCoverageError, match="--family collective"):
        database.query_comm(
            CommQuery(op="all_reduce", world_size=2, nodes=1, message_bytes=1 << 20)
        )


def test_a_comm_query_reads_the_vllm_table_by_default():
    """The dispatched kernel is the one a server runs, and it is not NCCL.

    Measured on H20 at world sizes 2, 4 and 8, `torch.distributed.all_reduce` is never what
    `CudaCommunicator.all_reduce` selects: the custom P2P kernel covers the whole decode range.
    A silent revert of this default to `nccl` would price a kernel no server invokes, and would
    be invisible: both tables exist, so the wrong one still resolves and still interpolates.
    """
    from srl.cost.kernel.database import CommQuery

    query = CommQuery(op="all_reduce", world_size=2, nodes=1, message_bytes=1 << 20)
    assert query.backend == "vllm"


def test_the_nccl_control_table_stays_reachable():
    """Quantifying the old table's bias requires reading it, so the field stays overridable."""
    from srl.cost.kernel.database import CommQuery

    query = CommQuery(
        op="all_reduce", world_size=2, nodes=1, message_bytes=1 << 20, backend="nccl"
    )
    assert query.backend == "nccl"
    # The backend picks a directory, not a slice, so it must not leak into the table key.
    assert query.slice_key == ("bfloat16", 2, 1)


def test_every_measured_collective_backend_has_a_fallback_floor():
    """A floor-taking op with no fallback silently degrades to zero.

    For `all_reduce` a zero floor is a 220x SOL error at 4096 bytes, and the fitted utility then
    launders it into a plausible-looking MRE. Both the dispatched table and the control need an
    entry, because the backend selects which `FALLBACK_FLOORS_S` subtree is read.
    """
    from srl.cost.kernel.sol import FALLBACK_FLOORS_S

    for op in _COLLECTIVES:
        assert op.fits_floor is True
        for backend in ("vllm", "nccl"):
            assert op.name in FALLBACK_FLOORS_S[backend], f"{op.name}/{backend}"
