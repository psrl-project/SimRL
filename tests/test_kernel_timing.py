"""The two timing modes, and the contract that keeps a host-bound row out of a table."""

from __future__ import annotations

import pytest
import torch

from srl.profiler.kernel.utils.timing import (
    time_cuda_call,
    time_cuda_graph_replay,
    torch_dtype,
)

_CUDA = torch.cuda.is_available()


def test_every_registered_op_declares_a_timing_mode() -> None:
    """A new op must choose, because neither default is safe.

    Defaulting to events overcharges a graph-replayed op by its python dispatch, which for
    GDN decode was 25.8 us against 3.9 us of device work, repeated across 30 layers.
    Defaulting to replay hides a launch cost that Megatron genuinely pays. Deriving the
    check off the registry makes the invariant self-maintaining rather than relying on the
    next op's author remembering.
    """
    from srl.cost.kernel.ops.base import OP_REGISTRY

    for name, op in sorted(OP_REGISTRY.items()):
        assert isinstance(getattr(op, "graph_replayed", None), bool), (
            f"Op {name!r} does not declare `graph_replayed`. Decide whether the engine "
            f"replays it from a captured CUDA graph; see cost/kernel/ops/base.py."
        )


def test_training_ops_are_never_graph_replayed() -> None:
    """Megatron runs no CUDA graphs, so a graph-timed training row would under-report.

    Pinned as an invariant rather than left to review: these tables feed the training cost
    model, where the launch cost is on the critical path, and `training_ops.py` already
    depends on that asymmetry when it charges elementwise for training but not for rollout.
    """
    from srl.cost.kernel.ops.base import OP_REGISTRY

    training_only = {"optimizer_step", "elementwise"}
    for name, op in sorted(OP_REGISTRY.items()):
        if name in training_only or op.family == "attention_training":
            assert not op.graph_replayed, (
                f"Op {name!r} feeds the training model, which runs eager; timing it by "
                f"graph replay would discard a launch cost Megatron really pays."
            )


def test_the_two_phases_of_a_split_family_disagree() -> None:
    """Decode is captured FULL while prefill runs piecewise, so one table cannot serve both.

    This is the whole point of making the flag per-op rather than per-family: a change that
    collapsed it to a family-level setting would silently pick one phase's answer for both.
    """
    from srl.cost.kernel.ops.base import op_for

    assert op_for("generation_attention").graph_replayed
    assert not op_for("context_attention").graph_replayed
    assert op_for("gdn_decode").graph_replayed
    assert not op_for("gdn_context").graph_replayed


def test_torch_dtype_resolves_aliases() -> None:
    assert torch_dtype("bf16") is torch.bfloat16
    assert torch_dtype("BFloat16") is torch.bfloat16
    assert torch_dtype("fp32") is torch.float32


def test_torch_dtype_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        torch_dtype("int4")


def test_graph_replay_requires_a_warmup() -> None:
    """Capturing a cold kernel records its autotuning path, so zero warmup is refused.

    Checked without a GPU because the guard is deliberately placed before any CUDA call:
    the failure must be a clear ValueError, not whatever capture does with an un-warmed
    kernel on the machine that happens to run the sweep.
    """
    with pytest.raises(ValueError, match="at least one warmup iteration"):
        time_cuda_graph_replay(lambda: None, warmup_iterations=0, active_iterations=4)


@pytest.mark.skipif(not _CUDA, reason="requires a GPU")
def test_both_modes_agree_for_a_device_bound_kernel() -> None:
    """A kernel whose device work dominates must time the same either way.

    This is what licenses leaving GEMM and elementwise on `time_cuda_call`: when the host
    runs ahead, the two modes measure the same thing, so the choice only matters for the
    host-bound families.
    """
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")

    def call() -> None:
        torch.matmul(a, b)

    event_s = time_cuda_call(call, warmup_iterations=5, active_iterations=20)
    graph_s = time_cuda_graph_replay(call, warmup_iterations=5, active_iterations=20)
    assert event_s == pytest.approx(graph_s, rel=0.25)


@pytest.mark.skipif(not _CUDA, reason="requires a GPU")
def test_graph_replay_excludes_host_dispatch() -> None:
    """A call dominated by python work must time faster under replay than under events.

    The graph records only the device work, so a deliberately interpreter-heavy callable
    separates the two modes. This is the property the GDN runner depends on.
    """
    xs = [torch.zeros(1, device="cuda") for _ in range(64)]
    one = torch.ones(1, device="cuda")

    def call() -> None:
        for x in xs:
            x.add_(one)

    event_s = time_cuda_call(call, warmup_iterations=5, active_iterations=20)
    graph_s = time_cuda_graph_replay(call, warmup_iterations=5, active_iterations=20)
    assert graph_s < event_s


@pytest.mark.skipif(not _CUDA, reason="requires a GPU")
def test_graph_replay_computes_the_same_result() -> None:
    """Replay must recompute, not reuse a frozen output.

    A graph that silently skipped its work would look fast and be worthless. Checked by
    changing the input between replays and requiring the output to follow.
    """
    src = torch.ones(256, device="cuda")
    dst = torch.zeros(256, device="cuda")

    def call() -> None:
        dst.copy_(src * 2.0)

    time_cuda_graph_replay(call, warmup_iterations=3, active_iterations=3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    src.fill_(5.0)
    graph.replay()
    torch.cuda.synchronize()
    assert dst.mean().item() == pytest.approx(10.0)


@pytest.mark.skipif(not _CUDA, reason="requires a GPU")
def test_graph_capture_failure_propagates() -> None:
    """An uncapturable callable must raise, never fall back to event timing.

    Falling back would write a host-bound row into a table whose other rows are
    device-bound, and nothing downstream could tell the two apart.
    """

    def call() -> None:
        torch.zeros(4, device="cuda").cpu()  # a sync, which capture forbids

    with pytest.raises(RuntimeError):
        time_cuda_graph_replay(call, warmup_iterations=2, active_iterations=4)
