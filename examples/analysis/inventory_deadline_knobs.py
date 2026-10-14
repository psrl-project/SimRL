#!/usr/bin/env python
"""Which `inventory_deadline` knob, if any, closes its loss on the rollout-bound regimes?

`headroom_probe.py run best` established the boundary: the strategy wins where training and rollout
are balanced (`tau` near the fleet's saturation cycle, `cv0.5_tau600` +4.6% over the best static arm)
and loses where rollout is clearly the bottleneck (`tau150`/`tau345`, ~-6%), where the concurrency it
picks for itself (mean ~44-70) is well above what a tuned static cap holds (~37-48) and where
`psrl_default` already shows the fleet saturated (mean 90-114, peak ~260).

This sweeps the strategy's own knobs one at a time at each losing point, and re-checks a winning
point, so a knob that recovers the losses can be seen to not spend the win. A knob that moves nothing
is reported as such rather than assumed away: the point is to find out whether the loss is a
mis-set parameter or a property of the decision rule itself, before reaching for a model change.

Usage:
    python examples/analysis/inventory_deadline_knobs.py run best
    python examples/analysis/inventory_deadline_knobs.py report best
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs" / "simulate" / "inventory_deadline_knobs"

KNOBS: dict[str, dict] = {
    "shipped": {},
    "eps0": {"eps_frac": 0.0},
    "eps_small": {"eps_frac": 0.001},
    "eps_big": {"eps_frac": 0.10},
    "horizon1": {"horizon_buffers": 1},
    "horizon4": {"horizon_buffers": 4},
    "no_defer": {"defer": False},
    "horizon4_eps_big": {"horizon_buffers": 4, "eps_frac": 0.10},
}
"""One knob moved at a time from the shipped defaults, plus one combination."""


def _load_probe():
    """Reuse `headroom_probe.py`'s points, harness and concurrency reconstruction."""
    spec = importlib.util.spec_from_file_location(
        "headroom_probe", ROOT / "examples" / "analysis" / "headroom_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _worker(args_tuple):
    grid, point_name, knob = args_tuple
    probe = _load_probe()
    fa = probe._load_forecast_accuracy()
    mdt = fa._load_dispersed_trace_module()
    cost_model = fa._composed_cost_model()
    point = next(p for p in probe.points(grid) if p.name == point_name)
    result = probe.run_arm(
        fa, point, cost_model, mdt,
        arm=knob, strategy="inventory_deadline", strategy_args=KNOBS[knob], cap=0,
    )
    print(f"{point_name} {knob} -> span {result.span:.0f}", flush=True)
    return {"point": point_name, "knob": knob, "result": asdict(result)}


def _run(args: argparse.Namespace) -> None:
    probe = _load_probe()
    out_dir = OUT_ROOT / args.grid
    out_dir.mkdir(parents=True, exist_ok=True)

    names = args.points or [p.name for p in probe.points(args.grid)]
    knobs = args.knobs or list(KNOBS)
    unknown = [k for k in knobs if k not in KNOBS]
    if unknown:
        raise SystemExit(f"Unknown knobs {unknown}. Known: {sorted(KNOBS)}.")
    jobs = [(args.grid, name, knob) for name in names for knob in knobs]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_worker, jobs))
    path = out_dir / "results.json"
    path.write_text(json.dumps(rows, indent=1))
    print(f"Wrote {path}")


def _report(args: argparse.Namespace) -> None:
    probe = _load_probe()
    rows = json.loads((OUT_ROOT / args.grid / "results.json").read_text())
    arms = json.loads((probe.OUT_ROOT / args.grid / "results.json").read_text())
    reference = {row["point"]["name"]: row for row in arms}

    names = []
    for row in rows:
        if row["point"] not in names:
            names.append(row["point"])

    print(
        f"{'point':<20}{'knob':<18}{'span':>9}{'vs shipped':>12}{'vs best static':>16}"
        f"{'conc':>7}{'peak':>6}{'tok/grp ok':>12}"
    )
    for name in names:
        by_knob = {row["knob"]: row["result"] for row in rows if row["point"] == name}
        ref = reference[name]
        shipped = by_knob["shipped"]["span"]
        best_static = min(ref["arms"]["static_cap"]["span"], ref["arms"]["static_cap_lpt"]["span"])
        base_tpg = ref["arms"]["psrl_default"]["tokens_per_group"]
        for knob in KNOBS:
            if knob not in by_knob:
                continue
            r = by_knob[knob]
            ok = "yes" if abs(r["tokens_per_group"] / base_tpg - 1.0) <= 0.01 else "NO"
            print(
                f"{name:<20}{knob:<18}{r['span']:>9.0f}{shipped / r['span']:>12.4f}"
                f"{best_static / r['span']:>16.4f}{r['mean_concurrency']:>7.1f}"
                f"{r['peak_concurrency']:>6}{ok:>12}"
            )
        print()


GRIDS = {"best": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("grid", choices=sorted(GRIDS))
    parser.add_argument("--points", nargs="*", default=None, help="Point names; default all in the grid.")
    parser.add_argument("--knobs", nargs="*", default=None, help="Knob names; default all.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "run":
        _run(args)
    else:
        _report(args)


if __name__ == "__main__":
    main()
