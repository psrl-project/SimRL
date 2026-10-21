"""Smallest experiment that exercises both arms, every gate, and the replay contract.

Copy this file, change the point, the arms, and the flags, then run it. It drives the in-process
harness, measures each arm with `collect.measure`, attaches the replay field each row needs, and
prints the gate verdicts.

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
import replayspec  # noqa: E402

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


def build(spec: dict, output_dir: Path | None = None):
    """
    Rebuild one arm of this experiment, optionally with full telemetry.

    Args:
        spec (dict): `{"point": <POINT dict>, "strategy": str, "strategy_args": dict, "cap": int}`.
        output_dir (Path | None): When set, write every telemetry stream here.

    Returns:
        tuple: An unstarted `Simulator` and the instance count.
    """
    probe = _load("headroom_probe", ROOT / "examples" / "analysis" / "headroom_probe.py")
    fa = probe._load_forecast_accuracy()
    mdt = fa._load_dispersed_trace_module()
    cost_model = fa._composed_cost_model()
    point = probe.Point(**spec["point"])
    regime = probe._regime(
        fa,
        point,
        strategy=spec["strategy"],
        strategy_args=spec["strategy_args"],
        cap=spec.get("cap", 0),
    )
    sim = fa._build_medium(regime, cost_model, mdt)
    if output_dir is not None:
        from srl.telemetry.store import JsonlStore
        from srl.telemetry.writer import SimTelemetry

        sim.cfg.log_telemetry = True
        sim.telemetry = SimTelemetry(JsonlStore(output_dir, overwrite=True), enabled=True)
    return sim, point.n_instances


def main() -> int:
    rows = []
    specs = {}
    for arm, arm_spec in ARMS.items():
        spec = {"point": POINT, "cap": 0, **arm_spec}
        sim, n_instances = build(spec)
        result = sim.run()
        rows.append({"point": POINT["name"], "arm": arm, **collect.measure(sim, result, n_instances)})
        specs[replayspec.row_id(POINT["name"], arm)] = spec

    rows = collect.compare_rows(rows, "control")
    rows = replayspec.attach(rows, Path(__file__), specs, ROOT)
    control = next(row for row in rows if row["arm"] == "control")
    out_dir = ROOT / "outputs" / "simulate" / "validate_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    verdicts = []
    for row in rows:
        if row["arm"] == "control":
            continue
        found = gates.evaluate_point(
            row, control, n_versions=POINT["n_versions"], batch_size=POINT["batch"]
        )
        verdicts.append(
            {
                "name": row["row_id"],
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
