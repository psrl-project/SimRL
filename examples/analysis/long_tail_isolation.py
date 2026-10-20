#!/usr/bin/env python
"""Does `inventory_deadline` protect a rare long-tail group from short-request contamination?

`headroom_probe.py`'s points are all one lognormal per point (`cv` only reshapes a single mode).
None of them separate "most groups are short enough that concurrency is never KV-bound" from "a
rare group is long enough that an instance can host only a handful of them" -- the regime this
module's author hypothesized actually has routing headroom: once a long-tail group reserves an
entry several buffers ahead (via the staleness window), a scheduler that is blind to that
reservation can keep piling fresh short arrivals onto the same instance, each one slowing the
long-tail group's decode and pushing its buffer's READY time out with it.

This workload is bimodal at the group level (every member of a group shares its mode, matching
GRPO's per-prompt correlation): `long_frac` of groups are near `length_cap`, the rest are a tight
short mode. `kv_tokens_per_instance` is sized so only `max_long_per_instance` long groups' peak
context fits per instance while short concurrency is nowhere near KV-bound, by construction --
this is the scenario the design doc's `cv`/`mean` grid cannot reach.

Usage:
    python examples/analysis/long_tail_isolation.py smoke
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load_forecast_accuracy():
    spec = importlib.util.spec_from_file_location("forecast_accuracy", ROOT / "examples" / "analysis" / "forecast_accuracy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class TailPoint:
    """A bimodal-at-the-group-level regime: a rare long-tail mode plus a common short mode.

    `kv_tokens_per_instance` is picked, not measured, so `max_long_per_instance` is exact by
    construction: `long_len * max_long_per_instance <= kv_tokens_per_instance < long_len * (max_long_per_instance + 1)`.
    Short concurrency is bounded by `token_budget`/decode cost, not KV, as long as
    `short_mean * (cap or 9999) << kv_tokens_per_instance`.
    """

    name: str
    train_s: float
    long_len: int = 30000
    long_jitter: int = 1500
    long_frac: float = 0.08
    short_mean: int = 500
    short_sigma: float = 0.3
    max_long_per_instance: int = 8
    length_cap: int = 32768
    staleness: int = 1
    n_versions: int = 10
    batch: int = 32
    group_size: int = 8
    n_instances: int = 2
    weight_load: float = 0.0
    strategy: str = "request_num_balance"
    strategy_args: dict = field(default_factory=dict)
    cap: int = 0
    eviction_strategy: str = "vllm_lru"
    seed: int = 0


def long_tail_workload(point: TailPoint, mod):
    """Build the bimodal `GroupedWorkload`: `point.long_frac` of groups are long-tail, correlated
    across the group the way GRPO responses to one prompt are; the rest are a tight short mode.
    """
    n_groups = point.batch * (point.n_versions + point.staleness + 3)
    rng = np.random.default_rng(point.seed)
    is_long = rng.random(n_groups) < point.long_frac
    prompts = np.clip(np.rint(rng.lognormal(np.log(140), 0.3, n_groups)), 16, 1024).astype(np.int64)
    groups = []
    for i in range(n_groups):
        if is_long[i]:
            lens = np.clip(
                np.rint(rng.normal(point.long_len, point.long_jitter, point.group_size)), 1, point.length_cap
            ).astype(np.int64)
        else:
            lens = np.clip(
                np.rint(rng.lognormal(np.log(point.short_mean), point.short_sigma, point.group_size)),
                1,
                point.length_cap,
            ).astype(np.int64)
        groups.append(mod.RequestGroup(group_id=i, prompt_len=int(prompts[i]), output_lens=[int(x) for x in lens]))
    n_long = int(is_long.sum())
    return mod.GroupedWorkload.from_groups(groups), n_long, n_groups


def build(point: TailPoint, cost_model, mod):
    """Wire `long_tail_workload` into the same `_MediumSimulator`/`SimConfig` shape `_build_medium`
    uses, so results are comparable with `headroom_probe.py`'s arms on everything except the
    workload and KV sizing.
    """
    workload, n_long, n_groups = long_tail_workload(point, mod)
    kv_tokens = point.long_len * point.max_long_per_instance + point.long_len // 2
    kv_blocks = kv_tokens // mod.MEDIUM_BLOCK_SIZE
    cfg = mod.SimConfig(
        batch_size=point.batch,
        group_size=point.group_size,
        staleness_horizon=point.staleness,
        n_versions=point.n_versions,
        n_instances=point.n_instances,
        weight_load_time=point.weight_load,
        sync_time=0.5,
        max_concurrent_seqs_per_instance=point.cap,
        workload=workload,
        log_telemetry=False,
        token_budget=mod.MEDIUM_TOKEN_BUDGET,
        kv_blocks=kv_blocks,
        block_size=mod.MEDIUM_BLOCK_SIZE,
        max_prompt_length=1024,
        max_output_length=point.length_cap,
        routing_strategy=point.strategy,
        routing_strategy_args=point.strategy_args,
        eviction_strategy=point.eviction_strategy,
    )
    sim = mod._MediumSimulator(
        rollout_cost=cost_model,
        training_cost=mod.FixedTrainingCost(point.train_s),
        sync_cost=mod.FixedSyncCost(0.5),
        policy=mod._MediumObservingPolicy(),
        cfg=cfg,
        backend="native",
    )
    return sim, n_long, n_groups, kv_blocks


def _smoke() -> None:
    """Confirm the workload is well-formed and the sim completes. Reports no speedup -- only
    that `max_long_per_instance` is realized and the run finishes within the horizon."""
    fa = _load_forecast_accuracy()
    cost_model = fa._composed_cost_model()
    point = TailPoint(name="smoke", train_s=345.0)
    sim, n_long, n_groups, kv_blocks = build(point, cost_model, fa)
    print(f"n_groups={n_groups} n_long={n_long} ({n_long / n_groups:.1%}) kv_blocks={kv_blocks}")
    result = sim.run()
    print(f"span={result.sim_time:.1f} versions_done={result.versions_done} livelocked={result.livelocked}")
    long_ctx = point.long_len * point.max_long_per_instance
    print(f"kv_tokens={kv_blocks * fa.MEDIUM_BLOCK_SIZE} vs {point.max_long_per_instance}x long_len={long_ctx}")


def _sweep(values: list[int]) -> None:
    """Run the same `TailPoint` at each `n_instances` in `values`, reporting span and completion."""
    fa = _load_forecast_accuracy()
    cost_model = fa._composed_cost_model()
    for n in values:
        point = TailPoint(name=f"sweep_n{n}", train_s=345.0, n_instances=n)
        sim, n_long, n_groups, kv_blocks = build(point, cost_model, fa)
        result = sim.run()
        print(
            f"n_instances={n:<3} span={result.sim_time:.1f} "
            f"versions_done={result.versions_done} livelocked={result.livelocked} "
            f"n_long={n_long}/{n_groups}"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        _smoke()
    elif len(sys.argv) > 2 and sys.argv[1] == "sweep":
        _sweep([int(v) for v in sys.argv[2].split(",")])
    else:
        print(__doc__)
