#!/usr/bin/env python
"""Does `inventory_deadline`'s placement help at all once admission is clamped to a fixed cap?

`headroom_probe.py run best` showed `inventory_deadline`'s own admission is too loose everywhere
(mean concurrency 1.2-1.7x the deployable `cap=48` baseline, at all 6 points including the two it
"won"), which means its wins were routing/load-balancing compensating for loose admission where
training had slack, not evidence the placement rule itself is good. This isolates that: clamp
`inventory_deadline` to the same hard `cap=48` the static arms use (an engine-level
`max_concurrent_seqs_per_instance`, independent of strategy, see `forecast_accuracy.py`'s
`regime.cap` -> `max_concurrent_seqs_per_instance`), so all three arms admit at the same rate, and
only placement differs. If `inventory_deadline@cap48` beats `cap_lpt48` here, the placement logic
has a real contribution and reworking admission into a rate decision is worth doing. If it does not,
admission is not the (only) problem and the lexicographic-`E` objective itself needs reconsidering
before touching admission.

`tokens_per_group` is reported and checked across arms at every point: this project has been burned
before by comparing arms whose consumed *set* of groups silently differed (same count, different
identities), which makes the span comparison not apples-to-apples. A point where `cap`'s own FIFO
arm and `inventory_deadline@cap` disagree on `tokens_per_group` by more than 1% is flagged, not
trusted at face value.

Usage:
    python examples/analysis/routing_ablation.py
"""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAP = 48


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "headroom_probe", ROOT / "examples" / "analysis" / "headroom_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _worker(args):
    point_name, arm, strategy, strategy_args = args
    probe = _load_probe()
    fa = probe._load_forecast_accuracy()
    mdt = fa._load_dispersed_trace_module()
    cost_model = fa._composed_cost_model()
    point = next(p for p in probe.points("best") if p.name == point_name)
    result = probe.run_arm(
        fa, point, cost_model, mdt,
        arm=arm, strategy=strategy, strategy_args=strategy_args, cap=CAP,
    )
    print(
        f"{point_name:<18}{arm:<20}span={result.span:>8.0f}  conc={result.mean_concurrency:>5.1f}"
        f"  peak={result.peak_concurrency:>4}  tok/grp={result.tokens_per_group:>9.1f}"
        f"  groups={result.consumed_groups}",
        flush=True,
    )
    return {"point": point_name, "arm": arm, "result": result}


def main() -> None:
    probe = _load_probe()
    points = [p.name for p in probe.points("best")]
    arms = [
        ("lpt48", "length_aware", {"order": "lpt", "balance": "load", "window_groups": probe.WINDOW_GROUPS}),
        ("inventory_deadline48", "inventory_deadline", {}),
    ]
    jobs = [(name, arm, strategy, strategy_args) for name in points for arm, strategy, strategy_args in arms]

    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(_worker, jobs))

    by_point: dict[str, dict] = {}
    for row in rows:
        by_point.setdefault(row["point"], {})[row["arm"]] = row["result"]

    print()
    print(f"{'point':<18}{'lpt48':>9}{'ours@48':>10}{'ours/lpt48':>12}{'verdict':>10}{'tok/grp drift':>16}")
    wins = 0
    clean_wins = 0
    clean_points = 0
    for name in points:
        arms_here = by_point[name]
        lpt = arms_here["lpt48"].span
        ours = arms_here["inventory_deadline48"].span
        ratio = lpt / ours
        verdict = "WINS" if ours < lpt else "loses"
        wins += ours < lpt
        tpg_lpt = arms_here["lpt48"].tokens_per_group
        tpg_ours = arms_here["inventory_deadline48"].tokens_per_group
        drift = tpg_ours / tpg_lpt - 1.0
        flag = "" if abs(drift) <= 0.01 else "  <-- NOT comparable"
        if abs(drift) <= 0.01:
            clean_points += 1
            clean_wins += ours < lpt
        print(f"{name:<18}{lpt:>9.0f}{ours:>10.0f}{ratio:>12.4f}{verdict:>10}{drift:>+15.2%}{flag}")
    print(f"\nRouting-only (cap={CAP} on all arms): inventory_deadline beats lpt48 on {wins}/{len(points)} points "
          f"raw, {clean_wins}/{clean_points} on points with a comparable consumed set.")


if __name__ == "__main__":
    main()
