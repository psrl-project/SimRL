"""Verify CppInstance matches NativeInstance on identical inputs."""

from __future__ import annotations

import random

import pytest

try:
    from dryrun._engine_cpp import (
        CppInstance,
        CppRequest,
        UnifiedRooflineCostModel,
    )
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

from dryrun.component.rollout.engine.instance import NativeInstance
from dryrun.component.rollout.engine.request import Request
from dryrun.cost.rollout import UnifiedRoofline

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ engine not compiled")


def _py_cost():
    return UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)


def _cpp_cost():
    return UnifiedRooflineCostModel(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)


def _full_state_py(inst):
    """Extract comparable state from the Python NativeInstance."""
    return {
        "t_local": inst.t_local,
        "waiting": [r.rid for r in inst.waiting],
        "running": [r.rid for r in inst.running],
        "n_completed": len(inst.completed),
        "n_preemptions": inst.n_preemptions,
        "blocks_used": inst.blocks_used(),
        "free_blocks": inst.free_blocks(),
        "queries": inst.kv.queries,
        "hits": inst.kv.hits,
        "reqs": sorted(
            (r.rid, r.prefilled, r.generated, r.num_preemptions)
            for r in (*inst.waiting, *inst.running)
        ),
    }


def _full_state_cpp(inst):
    """Extract comparable state from the C++ CppInstance."""
    return {
        "t_local": inst.t_local,
        "waiting": [r.rid for r in inst.waiting],
        "running": [r.rid for r in inst.running],
        "n_completed": len(inst.completed),
        "n_preemptions": inst.n_preemptions,
        "blocks_used": inst.blocks_used(),
        "free_blocks": inst.free_blocks(),
        "queries": inst.kv_queries,
        "hits": inst.kv_hits,
        "reqs": sorted(
            (r.rid, r.prefilled, r.generated, r.num_preemptions)
            for r in (*inst.waiting, *inst.running)
        ),
    }


def _event_kind_name(ev):
    """Normalize event kind to a comparable string."""
    # Both Python and C++ enums have `.name` = "ITERATION_DONE", etc.
    return ev.kind.name


@pytest.mark.parametrize("kv_blocks", [200, 600, 2000])
@pytest.mark.parametrize("caching", [True, False])
def test_cpp_matches_python_event_by_event(kv_blocks, caching):
    rng = random.Random(kv_blocks)
    max_concurrency = rng.choice([8, 64, 1024])
    token_budget = rng.choice([2048, 4096, 12800])

    py_inst = NativeInstance(
        cost=_py_cost(),
        token_budget=token_budget,
        kv_blocks=kv_blocks,
        block_size=16,
        max_concurrency=max_concurrency,
        enable_prefix_caching=caching,
    )
    cpp_inst = CppInstance(
        cost=_cpp_cost(),
        token_budget=token_budget,
        kv_blocks=kv_blocks,
        block_size=16,
        max_concurrency=max_concurrency,
        enable_prefix_caching=caching,
    )

    budget = kv_blocks * 16 // 2
    for rid in range(24):
        prompt = min(rng.choice([140, 320, 1024]), budget // 2)
        target = max(64, min(rng.choice([200, 600, 1500]), budget - prompt))
        py_req = Request(
            rid=rid,
            prompt_len=prompt,
            target_len=target,
            v_traj=0,
            group_id=rid // 4,
            source_group_id=rid // 4,
        )
        cpp_req = CppRequest(
            rid=rid,
            prompt_len=prompt,
            target_len=target,
            v_traj=0,
            group_id=rid // 4,
            source_group_id=rid // 4,
        )
        py_inst.add_request(py_req, t=float(rid))
        cpp_inst.add_request(cpp_req, t=float(rid))

    for step in range(300):
        py_ev = py_inst.advance_one_event(version=step % 3)
        cpp_ev = cpp_inst.advance_one_event(version=step % 3)

        if py_ev is None:
            assert cpp_ev is None, (
                f"step {step}: Python returned None but C++ did not"
            )
            break
        assert cpp_ev is not None, (
            f"step {step}: C++ returned None but Python did not"
        )
        py_kind = _event_kind_name(py_ev)
        cpp_kind = _event_kind_name(cpp_ev)
        assert py_kind == cpp_kind, (
            f"step {step}: event kinds differ: Python={py_kind}, C++={cpp_kind}"
        )

        py_inst.commit_last_event()
        cpp_inst.commit_last_event()

        py_state = _full_state_py(py_inst)
        cpp_state = _full_state_cpp(cpp_inst)
        assert py_state == cpp_state, (
            f"state diverged at step {step}:\n"
            f"  Python: {py_state}\n"
            f"  C++:    {cpp_state}"
        )

        if not py_inst.running and not py_inst.waiting:
            break


@pytest.mark.parametrize("kv_blocks", [200, 600])
@pytest.mark.parametrize("caching", [False])
def test_cpp_undo_matches_python_undo(kv_blocks, caching):
    rng = random.Random(kv_blocks)

    py_inst = NativeInstance(
        cost=_py_cost(),
        token_budget=4096,
        kv_blocks=kv_blocks,
        block_size=16,
        max_concurrency=64,
        enable_prefix_caching=caching,
    )
    cpp_inst = CppInstance(
        cost=_cpp_cost(),
        token_budget=4096,
        kv_blocks=kv_blocks,
        block_size=16,
        max_concurrency=64,
        enable_prefix_caching=caching,
    )

    budget = kv_blocks * 16 // 2
    for rid in range(16):
        prompt = min(rng.choice([140, 320]), budget // 2)
        target = max(64, min(rng.choice([200, 600]), budget - prompt))
        py_req = Request(
            rid=rid,
            prompt_len=prompt,
            target_len=target,
            v_traj=0,
            group_id=rid // 4,
            source_group_id=rid // 4,
        )
        cpp_req = CppRequest(
            rid=rid,
            prompt_len=prompt,
            target_len=target,
            v_traj=0,
            group_id=rid // 4,
            source_group_id=rid // 4,
        )
        py_inst.add_request(py_req, t=float(rid))
        cpp_inst.add_request(cpp_req, t=float(rid))

    undone = 0
    for step in range(200):
        py_before = _full_state_py(py_inst)
        cpp_before = _full_state_cpp(cpp_inst)

        py_ev = py_inst.advance_one_event(version=0)
        cpp_ev = cpp_inst.advance_one_event(version=0)
        if py_ev is None:
            break

        py_inst.undo_last_event()
        cpp_inst.undo_last_event()
        undone += 1

        assert _full_state_py(py_inst) == py_before, (
            f"Python undo diverged at step {step}"
        )
        assert _full_state_cpp(cpp_inst) == cpp_before, (
            f"C++ undo diverged at step {step}"
        )

        py_inst.advance_one_event(version=0)
        cpp_inst.advance_one_event(version=0)
        py_inst.commit_last_event()
        cpp_inst.commit_last_event()

        if not py_inst.running and not py_inst.waiting:
            break
    assert undone > 2, f"Expected at least 2 undone events, got {undone}."


def test_cpp_cost_model_matches_python():
    """Verify that the C++ and Python cost models produce the same results."""
    py_cost = _py_cost()
    cpp_cost = _cpp_cost()

    test_cases = [
        (10, 0, 100, 10, 0.0),
        (100, 50000, 500, 20, 100.0),
        (1, 0, 10, 1, 0.0),
        (5000, 2000000, 10000, 50, 5000.0),
    ]
    for n_tokens, t2, ctxsum, n_reqs, prefill_ctx in test_cases:
        py_result = py_cost.step_time(n_tokens, t2, ctxsum, n_reqs, prefill_ctx)
        cpp_result = cpp_cost.step_time(n_tokens, t2, ctxsum, n_reqs, prefill_ctx)
        assert py_result == pytest.approx(cpp_result, rel=1e-12), (
            f"Cost model mismatch for n_tokens={n_tokens}, t2={t2}, "
            f"ctxsum={ctxsum}: Python={py_result}, C++={cpp_result}"
        )


def test_cpp_advance_module_matches_python():
    """Verify that the C++ advance functions match the Python ones."""
    from dryrun._engine_cpp import crossing_step, elapsed, steps_within, next_completion
    from dryrun.cost.rollout import advance

    test_cases = [
        (0.005, 0.001, 0.0001),
        (0.01, 0.005, 0.001),
        (0.001, 0.002, 0.0),
        (0.0, 0.001, 0.001),
    ]
    for F, alpha, beta in test_cases:
        assert crossing_step(F, alpha, beta) == advance.crossing_step(F, alpha, beta)
        for k in [0, 1, 5, 10, 100]:
            assert elapsed(F, alpha, beta, k) == pytest.approx(
                advance.elapsed(F, alpha, beta, k), rel=1e-12
            )
        for budget in [0.0, 0.01, 0.1, 1.0, 10.0]:
            assert steps_within(F, alpha, beta, budget) == advance.steps_within(
                F, alpha, beta, budget
            )

    remaining = [10, 5, 8, 5, 12]
    assert next_completion(remaining) == advance.next_completion(remaining)
    assert next_completion([]) == advance.next_completion([])
