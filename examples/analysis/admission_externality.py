#!/usr/bin/env python
"""At admission, does the priced objective see the fleet-wide slowdown a new group causes?

`inventory_deadline` compares options by their `E_k` vectors alone. If admitting a group slows many
in-flight requests without moving the nearest buffers, `E_k` stays flat while total flow time
(`Pipeline.flow`) jumps, and the strategy over-admits. This records, per admission decision, each
option's `E_k0`, `E_k1`, flow and the fleet concurrency, plus the externality: the summed finish-time
increase of requests already in flight if the group is admitted on that instance, then reports for the decisions that
admitted: how much `E_k` moved against DEFER, how much flow moved, bucketed by concurrency.

Usage:
    python examples/analysis/admission_externality.py cv1.0_tau345 cv0.5_tau345
    python examples/analysis/admission_externality.py --report cv1.0_tau345   # re-summarise saved rows
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDING = "inventory_deadline_externality"
OUT_DIR = Path("/tmp/admission_externality")
BUCKETS = ((0, 40), (41, 55), (56, 10_000))
"""Fleet-active-request bands: at or under the tuned static knee, just over, well over."""


def _load_probe():
    spec = importlib.util.spec_from_file_location("headroom_probe", ROOT / "examples" / "analysis" / "headroom_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _register() -> None:
    from srl.component.rollout.router.strategies.inventory_deadline import InventoryDeadline
    from srl.component.rollout.router.strategy import STRATEGY_REGISTRY, register_strategy
    from srl.forecast.fleet import arrival_of, fold_member

    if RECORDING in STRATEGY_REGISTRY:
        return

    @register_strategy(RECORDING)
    class Recording(InventoryDeadline):
        name = RECORDING

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.rows: list[dict] = []

        def select_context(self, req, candidates, context):
            session = self._session
            row = None
            if session is not None and req.group_id in session.staged_groups:
                group = session.staged_groups[req.group_id]
                if req.rid == group[0].rid:
                    options = session.evaluate_admission(group)
                    base = {r: t for r, t in session.fleet.finish_times().items() if math.isfinite(t)}
                    remaining_of = self.predictor.remaining
                    ext = {}
                    for instance_id in session.fleet.instances:
                        fleet_eval = session.fleet.with_placement(instance_id, arrival_of(group[0], session.now, remaining_of))
                        for member in session._rest_sorted(group[1:]):
                            fleet_eval, _, _ = fold_member(fleet_eval, member, remaining_of, now=session.now)
                        after = fleet_eval.finish_times()
                        ext[str(instance_id)] = sum(after[r] - t for r, t in base.items() if math.isfinite(after.get(r, math.inf)))
                    row = {
                        "now": session.now,
                        "conc": sum(v.n_active for v in candidates),
                        "n_existing": len(base),
                        "ext": ext,
                        "eps": self._eps(session),
                        "options": {
                            ("defer" if k is None else str(k)): (
                                None
                                if p is None
                                else {"e": [_finite(x) for x in p.train_end], "flow": _finite(p.flow)}
                            )
                            for k, p in options.items()
                        },
                    }
            chosen = super().select_context(req, candidates, context)
            if row is not None:
                row["chosen"] = "defer" if chosen is None else str(candidates[chosen].instance_id)
                self.rows.append(row)
            return chosen


def _finite(x: float):
    return x if math.isfinite(x) else None


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def _median_finite(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return statistics.median(finite) if finite else math.nan


def _summarise(name: str, rows: list[dict]) -> None:
    admitted = []
    for row in rows:
        chosen, defer = row["options"].get(row["chosen"]), row["options"].get("defer")
        if row["chosen"] == "defer" or chosen is None or defer is None:
            continue
        if chosen["e"][0] is None or defer["e"][0] is None or chosen["flow"] is None or defer["flow"] is None:
            continue
        both_e1 = len(chosen["e"]) > 1 and len(defer["e"]) > 1 and None not in (chosen["e"][1], defer["e"][1])
        admitted.append(
            {
                "conc": row["conc"],
                "eps": row["eps"],
                "dE0": chosen["e"][0] - defer["e"][0],
                "dE1": (chosen["e"][1] - defer["e"][1]) if both_e1 else math.nan,
                "dFlow": chosen["flow"] - defer["flow"],
                "ext": row["ext"].get(row["chosen"], math.nan),
                "per_req": row["ext"].get(row["chosen"], math.nan) / max(1, row["n_existing"]),
            }
        )
    total = len(rows)
    print(f"\n{name}: {total} admission decisions, {len(admitted)} admitted with a comparable DEFER")
    print(f"{'conc band':<12}{'n':>5}{'dE0 p50':>10}{'dE0 p90':>10}{'eps':>7}{'ext p10':>9}{'ext p50':>9}{'ext p90':>9}{'s/req p50':>11}{'tied&s/req>eps':>16}")
    for lo, hi in BUCKETS:
        band = [a for a in admitted if lo <= a["conc"] <= hi]
        if not band:
            continue
        tied = [a for a in band if abs(a["dE0"]) <= a["eps"] and a["per_req"] > a["eps"]]
        print(
            f"{f'{lo}-{hi}':<12}{len(band):>5}"
            f"{statistics.median(a['dE0'] for a in band):>10.1f}{_pct([a['dE0'] for a in band], 0.9):>10.1f}"
            f"{statistics.median(a['eps'] for a in band):>7.1f}"
            f"{_pct([a['ext'] for a in band], 0.1):>9.0f}{statistics.median(a['ext'] for a in band):>9.0f}"
            f"{_pct([a['ext'] for a in band], 0.9):>9.0f}{statistics.median(a['per_req'] for a in band):>11.2f}"
            f"{len(tied):>12}/{len(band):<3}"
        )


def main() -> None:
    if sys.argv[1] == "--report":
        for name in sys.argv[2:]:
            _summarise(name, json.loads((OUT_DIR / f"{name}.json").read_text()))
        return
    _register()
    probe = _load_probe()
    fa = probe._load_forecast_accuracy()
    mdt = fa._load_dispersed_trace_module()
    cost_model = fa._composed_cost_model()
    OUT_DIR.mkdir(exist_ok=True)
    for name in sys.argv[1:]:
        point = next(p for grid in ("best", "heldout") for p in probe.points(grid) if p.name == name)
        regime = probe._regime(fa, point, strategy=RECORDING, strategy_args={}, cap=0)
        sim = fa._build_medium(regime, cost_model, mdt)
        result = sim.run()
        rows = sim.router.strategy.rows
        (OUT_DIR / f"{name}.json").write_text(json.dumps(rows))
        print(f"{name}: span={result.sim_time:.0f} groups={len({r.group_id for r in sim.consumed})}")
        _summarise(name, rows)


if __name__ == "__main__":
    main()
