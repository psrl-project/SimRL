"""
Check that the C++ engine reproduces the native one exactly.

Both backends run the same simulation in-process with every telemetry stream enabled,
and each stream is compared row by row. The engine is deterministic, so any difference
is a bug, and the first differing row names the stream, the time and the fields.

`--mode batching` instead turns off the `instance/*` stream, which lets the engine
commit runs of ITERATION_DONE itself, and compares both batched backends against native
event-by-event stepping.

The cases reach paths a benign run skips: preemption, chunked prefill, prefix caching
on and off, partial rollout aborts, concurrency caps, and several instances.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
from pathlib import Path

from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel
from srl.policy.psrl import PSRLPolicy
from srl.policy.verl import VerlPolicy
from srl.simulator import SimConfig, Simulator
from srl.telemetry.store import JsonlStore
from srl.workload.distributions import bimodal, uniform
from srl.workload.grouped import GroupedWorkload

_COST_CACHE: dict = {}


def build_cost(name: str):
    """Build a rollout cost model once and reuse it, since some load tables."""
    if name in _COST_CACHE:
        return _COST_CACHE[name]
    root = Path(__file__).resolve().parents[2]
    if name == "roofline":
        cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=1e-6, A_d=1e-7)
    elif name in ("psrl", "distserve"):
        from srl.cost.rollout import ROLLOUT_MODEL_REGISTRY
        from srl.profiler.schema import Parallelism

        cost = ROLLOUT_MODEL_REGISTRY[name].from_artifact(
            str(root / f"outputs/cost/qwen3_8b/rollout_{name}.json"), Parallelism()
        )
    elif name == "composed":
        from srl.cost.kernel.database import KernelDatabase
        from srl.cost.kernel.model_ops import ModelOps
        from srl.cost.rollout.composed import ComposedRollout

        db = KernelDatabase.from_root(
            str(root / "outputs/profile/qwen3_8b/kernel_tp1/kernel"), gpu="h20"
        )
        ops = ModelOps.from_checkpoint(
            "/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B", tensor_parallel=1
        )
        cost = ComposedRollout(db, ops, tensor_parallel=1)
    else:
        raise KeyError(name)
    _COST_CACHE[name] = cost
    return cost


def run(
    backend: str,
    cfg_kwargs: dict,
    lengths,
    prompts,
    policy_name: str,
    out: Path,
    cost_name: str = "roofline",
    exclude: tuple[str, ...] = (),
    batching: bool = True,
):
    """Run one simulation and write its telemetry under `out`."""
    cfg_kwargs = dict(cfg_kwargs)
    group_size = cfg_kwargs.pop("group_size", 1)
    pause_mode = cfg_kwargs.pop("pause_mode", "abort")
    if group_size == 1:
        workload = GroupedWorkload.from_lengths(prompts, lengths)
    else:
        from srl.workload.trace import RequestGroup

        groups = [
            RequestGroup(
                group_id=i,
                prompt_len=prompts[i],
                output_lens=tuple(lengths[(i * group_size + j) % len(lengths)]
                                  for j in range(group_size)),
            )
            for i in range(len(prompts))
        ]
        workload = GroupedWorkload.from_groups(groups)
    # `SimConfig` checks `group_size` against the workload at construction.
    cfg = SimConfig(group_size=group_size, workload=workload, **cfg_kwargs)
    policy = build(policy_name)
    policy.pause_mode = lambda st: pause_mode
    store = JsonlStore(out, overwrite=True, exclude=exclude)
    sim = Simulator(
        rollout_cost=build_cost(cost_name),
        training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
        policy=policy,
        cfg=cfg,
        store=store,
        backend=backend,
    )
    if not batching:
        sim._batches_iterations = lambda instance_id: False
    result = sim.run()
    store.close()
    return result


def first_difference(a_dir: Path, b_dir: Path):
    """Return the earliest differing row across all streams, or None if they all agree."""
    streams = sorted({p.relative_to(a_dir).as_posix() for p in a_dir.rglob("*.jsonl")}
                     | {p.relative_to(b_dir).as_posix() for p in b_dir.rglob("*.jsonl")})
    best = None
    for s in streams:
        pa, pb = a_dir / s, b_dir / s
        ra = [json.loads(l) for l in pa.open()] if pa.exists() else []
        rb = [json.loads(l) for l in pb.open()] if pb.exists() else []
        for i in range(max(len(ra), len(rb))):
            x = ra[i] if i < len(ra) else None
            y = rb[i] if i < len(rb) else None
            if x != y:
                t = float((x or y).get("t", float("inf")) or float("inf"))
                if best is None or t < best[1]:
                    fields = sorted(k for k in set(x or {}) | set(y or {})
                                    if (x or {}).get(k) != (y or {}).get(k))
                    best = (s, t, i, fields, x, y)
                break
    return best


def build(policy_name: str):
    """Construct a policy with the CLI's defaults, so the harness tests what users run."""
    from srl.policy.areal import ArealPolicy
    from srl.policy.roll import RollPolicy
    from srl.policy.slime import SlimePolicy

    return {
        "psrl": lambda: PSRLPolicy(),
        "verl": lambda: VerlPolicy(staleness_threshold=2.0),
        "areal": lambda: ArealPolicy(),
        "roll": lambda: RollPolicy(),
        "slime": lambda: SlimePolicy(),
    }[policy_name]()


CASES = {
    "tight_kv": dict(kv_blocks=120, token_budget=512),
    "chunked": dict(kv_blocks=4000, token_budget=64),
    "no_prefix": dict(kv_blocks=400, token_budget=256, enable_prefix_caching=False),
    "concurrency_cap": dict(kv_blocks=2000, token_budget=1024, max_concurrency=3),
    "multi_instance": dict(kv_blocks=300, token_budget=256, n_instances=3),
    "preempt_loopback": dict(kv_blocks=150, token_budget=256, n_instances=2,
                             max_waiting_after_preempt=1),
    "router_cap": dict(kv_blocks=2000, token_budget=512, n_instances=2,
                       max_concurrent_seqs_per_instance=2),
    "no_partial": dict(kv_blocks=300, token_budget=256, pause_mode="in_place"),
    # Paths the default configuration never reaches.
    "groups": dict(kv_blocks=300, token_budget=256, n_instances=2, group_size=4),
    "groups_sticky": dict(kv_blocks=300, token_budget=256, n_instances=2, group_size=4,
                          enable_group_sticky=True),
    "evict_longest": dict(kv_blocks=120, token_budget=512, eviction_strategy="longest_ctx"),
    "route_rr": dict(kv_blocks=300, token_budget=256, n_instances=3,
                     routing_strategy="round_robin"),
    "route_kv": dict(kv_blocks=300, token_budget=256, n_instances=3,
                     routing_strategy="kv_cache_aware"),
    "route_prefix": dict(kv_blocks=300, token_budget=256, n_instances=3,
                         routing_strategy="prefix_aware"),
    "route_tput": dict(kv_blocks=300, token_budget=256, n_instances=3,
                       routing_strategy="throughput_balance"),
    "sort_short": dict(kv_blocks=300, token_budget=256, n_instances=2,
                       router_sort_key="short_length"),
    "reject_waiting": dict(kv_blocks=300, token_budget=256, n_instances=2,
                           router_reject_on_waiting=True),
    "engine_reject": dict(kv_blocks=300, token_budget=256, reject_if_kv_full=True,
                          reject_if_waiting=True),
    "max_running": dict(kv_blocks=2000, token_budget=1024, max_concurrent=2),
    "weight_load": dict(kv_blocks=300, token_budget=256, weight_load_time=0.7, sync_time=0.3),
    "length_caps": dict(kv_blocks=300, token_budget=256, max_prompt_length=120,
                        max_output_length=200),
}


def check_batching(tag, base: Path, cfg_kwargs, lengths, prompts, policy, cost_name) -> int:
    """Check both batched backends against native event-by-event stepping."""
    exclude = ("instance",)
    ref_dir = base / "native_unbatched"
    try:
        ref = run("native", cfg_kwargs, lengths, prompts, policy, ref_dir, cost_name,
                  exclude=exclude, batching=False)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR  {tag}/reference: {type(exc).__name__}: {exc}")
        return 1
    failed = 0
    for backend in ("native", "cpp"):
        out = base / f"{backend}_batched"
        try:
            res = run(backend, cfg_kwargs, lengths, prompts, policy, out, cost_name,
                      exclude=exclude, batching=True)
        except Exception as exc:  # noqa: BLE001
            failed = 1
            print(f"ERROR  {tag}/{backend}: {type(exc).__name__}: {exc}")
            continue
        diff = first_difference(ref_dir, out)
        if diff is None:
            print(f"OK     {tag}/{backend}_batched: consumed={res.n_consumed} t={res.sim_time:.3f}")
            continue
        failed = 1
        stream, t, row, fields, x, y = diff
        print(f"DIFF   {tag}/{backend}_batched: stream={stream} row={row} t={t:.6f} "
              f"fields={fields[:8]}")
        print(f"         unbatched={json.dumps(x)[:300]}")
        print(f"         batched  ={json.dumps(y)[:300]}")
        print(f"         final: ref t={ref.sim_time:.3f} | batched t={res.sim_time:.3f}")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--policies", nargs="*", default=["psrl", "verl"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--costs", nargs="*", default=["roofline"],
                        help="roofline | psrl | distserve | composed")
    parser.add_argument("--n", type=int, default=64, help="Requests in the workload.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Multiply output lengths, to reach realistic depths.")
    parser.add_argument("--keep", type=Path, default=None)
    parser.add_argument("--mode", choices=["backends", "batching"], default="backends")
    parser.add_argument("--undoable", nargs="*", default=None,
                        help="Override undoable_events, e.g. request_complete.")
    args = parser.parse_args()

    failures = 0
    for case, policy, seed, cost_name in itertools.product(
        args.cases, args.policies, args.seeds, args.costs
    ):
        cfg_kwargs = dict(batch_size=4, group_size=1, max_staleness=1, n_versions=6)
        cfg_kwargs.update(CASES[case])
        if args.undoable is not None:
            cfg_kwargs["undoable_events"] = frozenset(args.undoable)
        n = args.n
        lengths = (bimodal(n, 40, 700, 0.3, seed=seed)
                   if seed % 2 else uniform(n, 30, 500, seed=seed))
        lengths = [max(1, int(x * args.scale)) for x in lengths]
        prompts = uniform(n, 16, 300, seed=seed + 100)
        base = args.keep or Path(tempfile.mkdtemp(prefix="diffbe_"))
        tag = f"{case}_{policy}_s{seed}_{cost_name}"
        if args.mode == "batching":
            failures += check_batching(tag, base / tag, cfg_kwargs, lengths, prompts,
                                       policy, cost_name)
            continue
        da, db = base / tag / "native", base / tag / "cpp"
        try:
            ra = run("native", cfg_kwargs, lengths, prompts, policy, da, cost_name)
            rb = run("cpp", cfg_kwargs, lengths, prompts, policy, db, cost_name)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR  {tag}: {type(exc).__name__}: {exc}")
            continue
        diff = first_difference(da, db)
        if diff is None:
            print(f"OK     {tag}: versions={ra.versions_done} consumed={ra.n_consumed} "
                  f"t={ra.sim_time:.3f}")
            continue
        failures += 1
        stream, t, row, fields, x, y = diff
        print(f"DIFF   {tag}: stream={stream} row={row} t={t:.6f} fields={fields[:8]}")
        print(f"         native={json.dumps(x)[:300]}")
        print(f"         cpp   ={json.dumps(y)[:300]}")
        print(f"         final: native t={ra.sim_time:.3f} consumed={ra.n_consumed} | "
              f"cpp t={rb.sim_time:.3f} consumed={rb.n_consumed}")
    print(f"\n{failures} failing case(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
