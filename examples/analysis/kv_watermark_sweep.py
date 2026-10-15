#!/usr/bin/env python
"""Does a KV-peak admission watermark bound `inventory_deadline`'s concurrency where it matters?

`headroom_probe.py run best` (post `window_groups` fix) shows admission is loose independently of
routing: uncapped concurrency is still 1.2-1.6x the tuned static-cap baseline's at every point, and
the fleet loses to a deployable fixed cap=48 on 4/6 points (geomean 0.980x). The design's own
"admission candidates = instances whose forecast future KV peak fits" limb was never given a
fractional threshold -- only literal `kv_overflow`, which a single marginal admission almost never
trips. This sweeps `InventoryDeadline(kv_watermark=...)` (now wired through `PlanSession`) across
the same 6 points, to see whether a soft KV ceiling on *admission* (not routing) can do what a
configured concurrency cap does, without being a configured concurrency cap.

Usage:
    python examples/analysis/kv_watermark_sweep.py run
    python examples/analysis/kv_watermark_sweep.py report
"""

from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "outputs" / "simulate" / "kv_watermark_sweep" / "results.json"
WATERMARKS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "headroom_probe", ROOT / "examples" / "analysis" / "headroom_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _worker(args):
    point_name, watermark = args
    probe = _load_probe()
    fa = probe._load_forecast_accuracy()
    mdt = fa._load_dispersed_trace_module()
    cost_model = fa._composed_cost_model()
    point = next(p for p in probe.points("best") if p.name == point_name)
    result = probe.run_arm(
        fa, point, cost_model, mdt,
        arm=f"wm{watermark}", strategy="inventory_deadline",
        strategy_args={"kv_watermark": watermark}, cap=0,
    )
    print(
        f"{point_name:<18}wm={watermark:<5}span={result.span:>8.0f}  conc={result.mean_concurrency:>5.1f}"
        f"  peak={result.peak_concurrency:>4}  tok/grp={result.tokens_per_group:>9.1f}",
        flush=True,
    )
    return {"point": point_name, "watermark": watermark, "result": asdict(result)}


def _run() -> None:
    probe = _load_probe()
    points = [p.name for p in probe.points("best")]
    jobs = [(name, wm) for name in points for wm in WATERMARKS]
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(_worker, jobs))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, indent=1))
    print(f"Wrote {OUT_PATH}")


def _report() -> None:
    probe = _load_probe()
    rows = json.loads(OUT_PATH.read_text())
    base = json.loads((probe.OUT_ROOT / "best" / "results.json").read_text())
    reference = {r["point"]["name"]: r for r in base}

    by_point: dict[str, dict] = {}
    for row in rows:
        by_point.setdefault(row["point"], {})[row["watermark"]] = row["result"]

    print(
        f"{'point':<18}{'watermark':>10}{'span':>9}{'vs default':>11}{'vs best_static':>15}"
        f"{'vs cap48':>9}{'conc':>7}{'tok/grp drift':>15}"
    )
    for name in [p.name for p in probe.points("best")]:
        ref = reference[name]
        default_span = ref["arms"]["psrl_default"]["span"]
        base_tpg = ref["arms"]["psrl_default"]["tokens_per_group"]
        best_static = min(ref["arms"]["static_cap"]["span"], ref["arms"]["static_cap_lpt"]["span"])
        cap48_fifo = next(r["span"] for r in ref["fifo_grid"] if r["cap"] == 48)
        cap48_lpt = next(r["span"] for r in ref["lpt_grid"] if r["cap"] == 48)
        cap48 = min(cap48_fifo, cap48_lpt)
        for wm in WATERMARKS:
            r = by_point[name][wm]
            drift = r["tokens_per_group"] / base_tpg - 1.0
            flag = "" if abs(drift) <= 0.01 else "  <-- NOT comparable"
            print(
                f"{name:<18}{wm:>10}{r['span']:>9.0f}{default_span / r['span']:>11.4f}"
                f"{best_static / r['span']:>15.4f}{cap48 / r['span']:>9.4f}"
                f"{r['mean_concurrency']:>7.1f}{drift:>+14.2%}{flag}"
            )
        print()


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "report"):
        raise SystemExit(f"Usage: {sys.argv[0]} run|report")
    if sys.argv[1] == "run":
        _run()
    else:
        _report()


if __name__ == "__main__":
    main()
