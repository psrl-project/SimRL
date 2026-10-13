"""
Locate state that an undo fails to restore, on either engine.

Runs the first events of a configuration on the Python and the C++ engine, doing
advance, undo and advance again, and prints both events and any requests whose
progress differs after the replay.

Usage:
    python examples/analysis/debug_undo.py <seed> <caching 0|1> <kv_blocks>
"""

import random
import sys

from dryrun._engine_cpp import CppInstance, CppRequest, UnifiedRooflineCostModel
from dryrun.component.rollout.engine.instance import NativeInstance
from dryrun.component.rollout.engine.request import Request
from dryrun.cost.rollout import UnifiedRoofline

seed, caching, kv_blocks = int(sys.argv[1]), bool(int(sys.argv[2])), int(sys.argv[3])
rng = random.Random(seed * 1000 + kv_blocks)
token_budget = rng.choice([256, 1024, 4096])
max_concurrency = rng.choice([4, 16, 1024])
rng2 = random.Random(seed)
budget = kv_blocks * 16 // 2
shapes = []
for rid in range(30):
    prompt = min(rng2.choice([48, 140, 320, 700]), budget // 2)
    target = max(32, min(rng2.choice([90, 300, 900, 1600]), budget - prompt))
    shapes.append((rid, prompt, target, rid // 3))
print(f"token_budget={token_budget} max_concurrency={max_concurrency} caching={caching}")


def build(kind):
    if kind == "py":
        inst = NativeInstance(
            cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=1e-6, A_d=1e-7),
            token_budget=token_budget, kv_blocks=kv_blocks, block_size=16,
            max_concurrency=max_concurrency, enable_prefix_caching=caching,
        )
        make = Request
    else:
        inst = CppInstance(
            cost=UnifiedRooflineCostModel(F=0.005, W=0.001, G=0.0001, A_p=1e-6, A_d=1e-7),
            token_budget=token_budget, kv_blocks=kv_blocks, block_size=16,
            max_concurrency=max_concurrency, enable_prefix_caching=caching,
        )
        make = CppRequest
    for rid, prompt, target, group in shapes:
        inst.add_request(
            make(rid=rid, prompt_len=prompt, target_len=target, v_traj=0,
                 group_id=group, source_group_id=group),
            t=float(rid) * 0.01,
        )
    return inst


def ev(e):
    return None if e is None else (e.kind.name, round(e.t, 9),
                                   sorted(r.rid for r in e.completed),
                                   sorted(r.rid for r in e.preempted))


def reqs(inst):
    return sorted((r.rid, r.prefilled, r.generated, r.status.name)
                  for r in (*inst.waiting, *inst.running))


for kind in ("py", "cpp"):
    inst = build(kind)
    for step in range(6):
        before = reqs(inst)
        first = inst.advance_one_event(version=0)
        if first is None:
            break
        after_first = reqs(inst)
        inst.undo_last_event()
        restored = reqs(inst)
        again = inst.advance_one_event(version=0)
        after_again = reqs(inst)
        inst.commit_last_event()
        same = ev(first) == ev(again)
        print(f"[{kind}] step {step}: first={ev(first)} again={ev(again)} "
              f"{'OK' if same else 'DIFFERENT'}  restored_ok={restored == before}")
        if not same:
            moved = [(a, b) for a, b in zip(after_first, after_again) if a != b]
            print(f"    requests differing after replay (first vs again): {moved[:6]}")
            break
