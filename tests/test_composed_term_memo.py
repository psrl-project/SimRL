"""
Check that the composed model's term memos never change a single bit.

A last-bit difference in a step time reorders simulation events, so every comparison
here is `==`, never `approx`.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from srl.cost.rollout.base import RolloutWorkload

ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = ROOT / "outputs/profile/qwen3_8b/kernel_tp1/kernel"
MODEL = Path("/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B")

pytestmark = pytest.mark.skipif(
    not (KERNEL_ROOT.is_dir() and MODEL.is_dir()),
    reason="Needs the profiled Qwen3-8B kernel tables and checkpoint.",
)


@pytest.fixture(scope="module")
def model():
    from srl.cost.kernel.database import KernelDatabase
    from srl.cost.kernel.model_ops import ModelOps
    from srl.cost.rollout.composed import ComposedRollout

    db = KernelDatabase.from_root(str(KERNEL_ROOT), gpu="h20")
    ops = ModelOps.from_checkpoint(str(MODEL), tensor_parallel=1)
    return ComposedRollout(db, ops, tensor_parallel=1)


def _workloads(seed: int, n: int):
    """Return decode, prefill and mixed steps whose keys repeat the way a real run's do."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        n_d = rng.choice([1, 2, 8, 37, 160, 160, 160])
        ctx_per = rng.choice([200, 5000, 5001, 5002, 20000, 64000])
        ctxsum = float(n_d * ctx_per + rng.randint(0, n_d - 1))
        kind = rng.random()
        if kind < 0.6:
            out.append(RolloutWorkload.from_step(n_d, 0.0, ctxsum, n_d, 0.0))
        else:
            chunk = rng.choice([64, 512, 4096, 12000])
            extent = float(chunk + rng.choice([0, 300, 9000]))
            n_pre = rng.choice([1, 2, 3])
            t2 = n_pre * extent * extent
            prefill_ctx = n_pre * extent
            decode_ctx = ctxsum if kind < 0.85 else 0.0
            out.append(
                RolloutWorkload.from_step(
                    n_d + n_pre * chunk, t2, decode_ctx, n_d + n_pre, prefill_ctx
                )
            )
    return out


def test_memoized_predict_is_bit_identical(model):
    """Check that every workload, first or repeated, prices exactly as an uncached evaluation."""
    workloads = _workloads(seed=0, n=3000)
    reference = [model._predict_uncached(w) for w in workloads]
    model._clear_term_memo()
    # The second pass is served almost entirely from the memo.
    for _ in range(2):
        for w, expected in zip(workloads, reference):
            got = model.predict(w)
            assert got == expected, (
                f"Memoized predict differs for {w}: {got!r} != {expected!r}."
            )


def test_memo_is_actually_used(model):
    """Check that `predict` actually populates the memos rather than bypassing them."""
    model._clear_term_memo()
    workloads = _workloads(seed=1, n=500)
    for w in workloads:
        model.predict(w)
    gemm_keys = {
        (max(int(round(w.n_tokens)), 1), max(int(w.request_count), 1)) for w in workloads
    }
    assert len(model._gemm_comm_memo) == len(gemm_keys)
    assert 0 < len(model._attention_memo) <= len(workloads)


def _edge_steps():
    """
    Return step arguments on the edges of CPython's `//` and `round`.

    The C++ engine reproduces both itself, so these cover exact `.5` ties, quotients one
    ulp either side of an integer, and large values.
    """
    import math

    steps = []
    for tokens in (1.0, 2.5, 3.5, 4.5, 1000.5, 12800.0, 0.5, 1.4999999999999998):
        for extent in (1.0, 3.0, 7.0, 1e-3, 12345.678, 1e9):
            for ratio in (1.0, 2.0, 2.5, 3.0, 0.1 * 3, 7.0 / 3.0):
                t2 = extent * extent * ratio
                steps.append((tokens, t2, 0.0, 1, extent))
                steps.append((tokens, math.nextafter(t2, 0.0), 0.0, 3, extent))
                steps.append((tokens, math.nextafter(t2, math.inf), 5000.0, 7, extent))
    for n_reqs in (1, 2, 3, 7, 160, 999):
        for per in (1.0, 2.0, 5000.0, 64000.0, 1e12):
            ctxsum = per * n_reqs
            for c in (ctxsum, math.nextafter(ctxsum, 0.0), math.nextafter(ctxsum, math.inf)):
                steps.append((float(n_reqs), 0.0, c, n_reqs, 0.0))
    # prefill_ctx == 0 makes the extent fall back to n_tokens, like Python's `or`.
    steps.append((512.0, 512.0 * 512.0, 0.0, 1, 0.0))
    steps.append((0.0, 0.0, 0.0, 0, 0.0))
    return steps


def test_cpp_term_fast_path_is_bit_identical(model):
    """Check that the engine's own key derivation and summation give `predict`'s exact float."""
    cpp = pytest.importorskip("srl._engine_cpp")
    wrapped = cpp.PyWrappedCostModel(model)
    assert wrapped.term_fast_path, "ComposedRollout did not enable the C++ term path."
    steps = _edge_steps()
    for w in _workloads(seed=2, n=3000):
        steps.append((w.n_tokens, w.sum_l2, w.ctxsum, w.request_count, w.prefill_ctx))
    for _ in range(2):  # the second pass is served from the C++ memos
        for args in steps:
            expected = model._predict_uncached(RolloutWorkload.from_step(*args))
            got = wrapped.step_time(*args)
            assert got == expected, f"C++ step_time{args} = {got!r}, predict = {expected!r}."
