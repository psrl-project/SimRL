#!/usr/bin/env python
"""At admission, which buffer does the slowdown land on, and could the trainer absorb it?

`inventory_deadline` compares options by their `E_k` vectors. Admitting a group slows requests
already in flight (the externality); that only matters if it pushes some buffer's READY time past
the moment the trainer is free to take it, because `E_k = max(F_k, E_{k-1}) + tau` absorbs any
delay up to that slack. This records, per admission decision, each option's `E_k`, `F_k` and
publish times plus the externality, then reports for the decisions that admitted: the nearest
buffer's READY shift against DEFER, the trainer's slack at that buffer, and whether the second
buffer's READY time is even finite (when it is not, `E_k1` carries no information).

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
                        "cycle": next(iter(session.train_cache.values())).cycle if session.train_cache else None,
                        "options": {
                            ("defer" if k is None else str(k)): (
                                None
                                if p is None
                                else {
                                    "e": [_finite(x) for x in p.train_end],
                                    "ready": [_finite(x) for x in p.ready],
                                    "publish": [_finite(x) for x in p.publish_times],
                                }
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


def _summarise(name: str, rows: list[dict]) -> None:
    admitted = []
    for row in rows:
        chosen, defer = row["options"].get(row["chosen"]), row["options"].get("defer")
        if row["chosen"] == "defer" or chosen is None or defer is None or row["cycle"] is None:
            continue
        if None in (chosen["e"][0], defer["e"][0], chosen["ready"][0], defer["ready"][0], defer["publish"][0]):
            continue
        slack0 = (defer["publish"][0] - row["cycle"]) - defer["ready"][0]
        admitted.append(
            {
                "conc": row["conc"],
                "eps": row["eps"],
                "dE0": chosen["e"][0] - defer["e"][0],
                "dF0": chosen["ready"][0] - defer["ready"][0],
                "slack0": slack0,
                "f1_finite": len(defer["ready"]) > 1 and defer["ready"][1] is not None,
                "per_req": row["ext"].get(row["chosen"], math.nan) / max(1, row["n_existing"]),
            }
        )
    print(f"\n{name}: {len(rows)} admission decisions, {len(admitted)} admitted with a comparable DEFER")
    print(
        f"{'conc band':<12}{'n':>5}{'dF0 p50':>9}{'dF0 p90':>9}{'slack0 p10':>11}{'slack0 p50':>11}"
        f"{'slack0>0':>10}{'dF0<=slack0':>13}{'F1 finite':>11}{'s/req p50':>11}"
    )
    for lo, hi in BUCKETS:
        band = [a for a in admitted if lo <= a["conc"] <= hi]
        if not band:
            continue
        absorbed = [a for a in band if a["dF0"] <= max(a["slack0"], 0.0) + a["eps"]]
        print(
            f"{f'{lo}-{hi}':<12}{len(band):>5}"
            f"{statistics.median(a['dF0'] for a in band):>9.1f}{_pct([a['dF0'] for a in band], 0.9):>9.1f}"
            f"{_pct([a['slack0'] for a in band], 0.1):>11.1f}{statistics.median(a['slack0'] for a in band):>11.1f}"
            f"{sum(a['slack0'] > 0 for a in band):>7}/{len(band):<3}"
            f"{len(absorbed):>10}/{len(band):<3}"
            f"{sum(a['f1_finite'] for a in band):>8}/{len(band):<3}"
            f"{statistics.median(a['per_req'] for a in band):>11.2f}"
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
