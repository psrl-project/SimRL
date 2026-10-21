"""Smallest experiment that exercises both arms and every gate.

Copy this file, change the point, the arms, and the flags, then run it. It drives the in-process
harness, measures each arm with `collect.measure`, and prints the gate verdicts.

Usage:
    python workflow/validate/experiments/smoke.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workflow" / "validate" / "scripts"))

import collect  # noqa: E402
import gates  # noqa: E402

POINT = {
    "name": "smoke_cv0.5_tau345",
    "cv": 0.5,
    "train_s": 345.0,
    "staleness": 1,
    "mean": 8000,
    "length_cap": 32768,
    "n_versions": 12,
    "batch": 32,
    "group_size": 8,
    "n_instances": 2,
}
ARMS = {
    "control": {"strategy": "inventory_deadline", "strategy_args": {}},
    "treatment": {"strategy": "inventory_deadline", "strategy_args": {"kv_watermark": 0.6}},
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    probe = _load("headroom_probe", ROOT / "examples" / "analysis" / "headroom_probe.py")
    fa = probe._load_forecast_accuracy()
    mdt = fa._load_dispersed_trace_module()
    cost_model = fa._composed_cost_model()
    point = probe.Point(**POINT)

    rows = []
    for arm, spec in ARMS.items():
        regime = probe._regime(fa, point, strategy=spec["strategy"], strategy_args=spec["strategy_args"], cap=0)
        sim = fa._build_medium(regime, cost_model, mdt)
        result = sim.run()
        rows.append({"point": point.name, "arm": arm, **collect.measure(sim, result, point.n_instances)})

    rows = collect.compare_rows(rows, "control")
    control = next(row for row in rows if row["arm"] == "control")
    out_dir = ROOT / "outputs" / "simulate" / "validate_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    verdicts = []
    for row in rows:
        if row["arm"] == "control":
            continue
        found = gates.evaluate_point(
            row, control, n_versions=point.n_versions, batch_size=point.batch
        )
        verdicts.append(
            {
                "name": f"{row['point']}/{row['arm']}",
                "speedup": row["speedup"],
                "gates": [{"name": gate.name, "ok": gate.ok, "detail": gate.detail} for gate in found],
            }
        )
        print(f"{row['arm']}: span={row['span']:.0f} speedup={row['speedup']:.4f} "
              f"conc={row['mean_concurrency']:.1f} groups={row['consumed_groups']}")
        for gate in found:
            print(f"  [{'PASS' if gate.ok else 'FAIL'}] {gate.name}: {gate.detail}")

    summary = gates.summarize(verdicts)
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0 if summary["n_gated"] == len(verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
