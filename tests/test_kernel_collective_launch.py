"""The collective driver's command construction and refusals, without a process group.

Mirrors `tests/test_fsdp_launch.py`: the launch path is exercised by building the command
and the spec rather than by running them, so these stay fast and need no GPU. What they
protect is the set of mistakes that would produce *plausible numbers from the wrong
measurement*, which is worse than a crash because a smooth interpolation hides it.
"""

from __future__ import annotations

import pytest

from dryrun.profiler.kernel.collective import _fabric_exports, _kernel_source, _rows_for
from dryrun.profiler.kernel.collective_launch import _nccl_call


def test_the_remote_command_forwards_every_nccl_variable(monkeypatch):
    """Both ends of a cross-node ring must run one fabric configuration, not two.

    A non-interactive ssh shell on this cluster inherits ~/.bashrc, which sets seven NCCL
    variables, while the project env script sets twenty-one and orders NCCL_IB_HCA
    differently for rail alignment. If each node resolved its own, the two ends would run
    different schedules and the measurement would describe neither.
    """
    monkeypatch.setenv("NCCL_IB_HCA", "mlx5_bond_1,mlx5_bond_5")
    monkeypatch.setenv("NCCL_ALGO", "allgather:^tree")
    monkeypatch.setenv("NCCL_LL_THRESHOLD", "16384")
    monkeypatch.setenv("UNRELATED_VARIABLE", "must-not-be-forwarded")

    exports = _fabric_exports()
    joined = " ".join(exports)
    assert "NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5" in joined
    assert "NCCL_ALGO='allgather:^tree'" in joined, "a value with ^ must be shell-quoted"
    assert "NCCL_LL_THRESHOLD=16384" in joined
    assert "UNRELATED_VARIABLE" not in joined


def test_a_value_needing_quoting_is_quoted(monkeypatch):
    """`NCCL_ALGO=allgather:^tree` unquoted is a history expansion in some shells."""
    monkeypatch.setenv("NCCL_DANGEROUS", "a b;c")
    assert "NCCL_DANGEROUS='a b;c'" in " ".join(_fabric_exports())


@pytest.mark.parametrize(
    "op",
    ("all_reduce", "reduce_scatter", "all_gather", "broadcast", "p2p"),
)
def test_every_collective_has_a_torch_distributed_call(op):
    """Built on a CPU tensor: the callable is constructed, never invoked.

    A real tensor rather than a stub, because `reduce_scatter` and `all_gather` allocate
    their `world_size` side buffers eagerly via `torch.empty_like`. That eagerness is
    deliberate: allocating inside the timed callable would charge the measurement for
    `world_size` allocations per iteration.
    """
    import torch

    tensor = torch.zeros(1024)
    call = _nccl_call(op, tensor, rank=0, world_size=8)
    assert callable(call)


def test_the_sharded_ops_size_their_buffers_by_world_size():
    """message_bytes must mean the same thing for every op: bytes one rank contributes.

    A reduce_scatter's collective buffer is `world_size` times its per-rank shard, so
    handing every op an identically sized tensor would silently price an 8-rank
    reduce_scatter as if it moved an eighth of its real traffic.
    """
    import torch

    tensor = torch.zeros(256)
    captured: dict[str, int] = {}
    real_empty_like = torch.empty_like

    def counting_empty_like(value, *args, **kwargs):
        captured["count"] = captured.get("count", 0) + 1
        return real_empty_like(value, *args, **kwargs)

    torch.empty_like = counting_empty_like
    try:
        _nccl_call("reduce_scatter", tensor, rank=0, world_size=8)
    finally:
        torch.empty_like = real_empty_like
    assert captured["count"] == 8


def test_an_unknown_op_is_refused_rather_than_silently_skipped():
    with pytest.raises(ValueError, match="Unsupported collective op"):
        _nccl_call("all_to_all", object(), rank=0, world_size=8)


def test_rows_are_projected_per_op_and_unmeasured_points_dropped():
    """p2p has no vllm latency, and a None must not become a zero-latency row.

    A zero or missing latency silently entering the table would be read as an
    infinitely fast collective, and `KernelTable.load` rejects non-positive latency only
    if the value actually arrives.
    """
    raw = [
        {
            "op": "all_reduce",
            "dtype": "bfloat16",
            "world_size": 8,
            "nodes": 1,
            "message_bytes": 4096,
            "nccl_latency_s": 2.0e-5,
            "vllm_latency_s": 1.5e-5,
            "dispatched_backend": "custom_all_reduce",
        },
        {
            "op": "p2p",
            "dtype": "bfloat16",
            "world_size": 8,
            "nodes": 1,
            "message_bytes": 4096,
            "nccl_latency_s": 1.1e-5,
            "vllm_latency_s": None,
            "dispatched_backend": None,
        },
    ]
    assert len(_rows_for(raw, "all_reduce", "vllm_latency_s")) == 1
    assert _rows_for(raw, "p2p", "vllm_latency_s") == []
    p2p_nccl = _rows_for(raw, "p2p", "nccl_latency_s")
    assert len(p2p_nccl) == 1
    assert p2p_nccl[0]["nodes"] == 1
    assert p2p_nccl[0]["latency_s"] == pytest.approx(1.1e-5)


def test_every_projected_row_carries_the_table_columns_and_nothing_else():
    """`PerfTableWriter.write` rejects an extra column, so the projection must be exact."""
    from dryrun.cost.kernel.ops.base import op_for

    raw = [
        {
            "op": "all_gather",
            "dtype": "float32",
            "world_size": 16,
            "nodes": 2,
            "message_bytes": 1 << 20,
            "nccl_latency_s": 9.0e-4,
            "vllm_latency_s": 8.0e-4,
            "dispatched_backend": "dispatcher",
        }
    ]
    row = _rows_for(raw, "all_gather", "nccl_latency_s")[0]
    assert set(row) == set(op_for("all_gather").table_columns())


def test_a_sweep_spanning_backends_is_labelled_mixed():
    """A table whose rows disagree on the kernel cannot be priced as one surface."""
    raw = [
        {"op": "all_reduce", "dispatched_backend": "custom_all_reduce"},
        {"op": "all_reduce", "dispatched_backend": "torch_symm_mem"},
    ]
    assert _kernel_source(raw, "all_reduce") == "mixed"


def test_a_single_backend_sweep_records_that_backend():
    raw = [{"op": "broadcast", "dispatched_backend": "dispatcher"}]
    assert _kernel_source(raw, "broadcast") == "dispatcher"


def test_the_dispatcher_is_only_consulted_within_a_node():
    """vLLM refuses a group spanning nodes, so standing its state up buys nothing.

    `custom_all_reduce.py:90` disables the custom kernel when `in_the_same_node_as` fails,
    and symmetric memory refuses the rendezvous for the same reason, leaving pynccl, which
    the control table already measures. The first cross-node attempt paid ~60 s per
    topology of vLLM init to produce a duplicate of that table.

    Asserted by reading the module source rather than by running a second node, because the
    property is a branch in `main()` and a regression here is silent: it costs time and
    produces a relabelled NCCL row rather than failing.
    """
    import inspect

    from dryrun.profiler.kernel import collective_launch

    source = inspect.getsource(collective_launch.main)
    assert "dispatcher_available = nodes == 1" in source, (
        "main() no longer gates vLLM parallel state on a single-node topology"
    )


def test_a_cross_node_topology_under_ib_disable_is_refused(monkeypatch, tmp_path):
    """A TCP fallback would yield rows that look like a slow NIC rather than an absent one.

    That is the failure mode this whole file exists to prevent: the numbers would be
    wrong, self-consistent, and smoothly interpolable, so nothing downstream would flag
    them.
    """
    from dryrun.profiler.kernel.collective import _run_topology

    monkeypatch.setenv("NCCL_IB_DISABLE", "1")
    with pytest.raises(RuntimeError, match="NCCL_IB_DISABLE"):
        _run_topology(
            world_size=16,
            nodes=2,
            hostnames=("host-a", "host-b"),
            spec={"ops": ["all_reduce"], "dtypes": ["bfloat16"], "message_bytes": [4096]},
            work_dir=tmp_path,
        )
