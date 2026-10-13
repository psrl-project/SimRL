"""
Check that undo is invisible: undoing an event and replaying it changes nothing.

Two identical C++ instances run in lockstep. One advances, undoes and advances again
before committing every event, the other only advances and commits, and their events
and full state must agree at every step. The comparison runs to completion because a
slightly wrong restore of held blocks or free-list order leaves the counters correct
and only surfaces later as a different prefix hit or block. The pools are tight
enough that one event both allocates and frees the same request.
"""

from __future__ import annotations

import random

import pytest

try:
    from dryrun._engine_cpp import CppInstance, CppRequest, UnifiedRooflineCostModel

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ engine not compiled")


def _make(kv_blocks: int, caching: bool, token_budget: int, max_concurrency: int):
    cost = UnifiedRooflineCostModel(F=0.005, W=0.001, G=0.0001, A_p=1e-6, A_d=1e-7)
    return CppInstance(
        cost=cost,
        token_budget=token_budget,
        kv_blocks=kv_blocks,
        block_size=16,
        max_concurrency=max_concurrency,
        enable_prefix_caching=caching,
    )


def _requests(seed: int, kv_blocks: int, n: int):
    rng = random.Random(seed)
    budget = kv_blocks * 16 // 2
    shapes = []
    for rid in range(n):
        prompt = min(rng.choice([48, 140, 320, 700]), budget // 2)
        target = max(32, min(rng.choice([90, 300, 900, 1600]), budget - prompt))
        # Shared source groups give prefix caching something to hit.
        shapes.append((rid, prompt, target, rid // 3))
    return shapes


def _add(inst, shapes):
    for rid, prompt, target, group in shapes:
        req = CppRequest(
            rid=rid,
            prompt_len=prompt,
            target_len=target,
            v_traj=0,
            group_id=group,
            source_group_id=group,
        )
        inst.add_request(req, t=float(rid) * 0.01)


def _state(inst):
    reqs = sorted(
        (
            r.rid,
            r.prefilled,
            r.generated,
            r.status.name,
            r.num_preemptions,
            r.num_aborts,
            r.reprefill_tokens,
            r.prefix_hit_tokens,
            inst.peek_hit_tokens(r),
        )
        for r in (*inst.waiting, *inst.running)
    )
    return {
        "t_local": inst.t_local,
        "waiting": [r.rid for r in inst.waiting],
        "running": [r.rid for r in inst.running],
        "completed": [r.rid for r in inst.completed],
        "n_preemptions": inst.n_preemptions,
        "blocks_used": inst.blocks_used(),
        "free_blocks": inst.free_blocks(),
        "committed_blocks": inst.committed_blocks(),
        "queries": inst.kv_queries,
        "hits": inst.kv_hits,
        "n_step_stats": len(inst.step_stats),
        "reqs": reqs,
        # Block level, because a wrong restore can leave every counter above matching.
        "kv": inst.debug_kv_state(),
    }


def _event(ev):
    if ev is None:
        return None
    return (
        ev.kind.name,
        ev.t,
        sorted(r.rid for r in ev.completed),
        sorted(r.rid for r in ev.preempted),
    )


@pytest.mark.parametrize("kv_blocks", [96, 160, 400, 2000])
@pytest.mark.parametrize("caching", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_undo_then_replay_is_invisible(kv_blocks, caching, seed):
    rng = random.Random(seed * 1000 + kv_blocks)
    token_budget = rng.choice([256, 1024, 4096])
    max_concurrency = rng.choice([4, 16, 1024])
    shapes = _requests(seed, kv_blocks, n=30)

    replayed = _make(kv_blocks, caching, token_budget, max_concurrency)
    reference = _make(kv_blocks, caching, token_budget, max_concurrency)
    _add(replayed, shapes)
    _add(reference, shapes)

    preempting_events = 0
    for step in range(60000):
        before = _state(replayed)
        first = replayed.advance_one_event(version=step % 3)
        if first is None:
            assert reference.advance_one_event(version=step % 3) is None
            break
        replayed.undo_last_event()
        assert _state(replayed) == before, f"Undo did not restore state at step {step}."

        again = replayed.advance_one_event(version=step % 3)
        assert _event(again) == _event(first), f"Replay produced a different event at step {step}."
        replayed.commit_last_event()

        ref = reference.advance_one_event(version=step % 3)
        reference.commit_last_event()
        assert _event(ref) == _event(again), f"Replay diverged from the reference at step {step}."
        assert _state(replayed) == _state(reference), f"State diverged at step {step}."
        if ref is not None and len(ref.preempted) > 0:
            preempting_events += 1
    else:
        pytest.fail("Run did not finish within the step limit.")

    if kv_blocks <= 160:
        assert preempting_events > 0, "The pool never forced a preemption, so the test misses its target."
