#!/usr/bin/env python
"""What does `inventory_deadline` actually decide, and does it ever prefer to wait?

`headroom_probe.py` says the strategy loses exactly where rollout is the bottleneck, and
`inventory_deadline_knobs.py` says no knob brings its concurrency down there. Both point at the
same question, which neither can answer from `sim_time` alone: at each placement, does the priced
delay to the training pipeline ever prefer DEFER, and how large is that delay relative to `eps`?

The design's whole claim is that the cap emerges from the model, so a losing regime should show the
model *seeing* the cost and choosing differently. A losing regime that shows the model pricing every
option as indistinguishable would instead mean the model cannot see the cost at all, a different
diagnosis with a different fix. This measures which one it is, by registering a recording subclass
of the strategy so the run goes through the ordinary construction path and every decision is
recorded without being altered.

Usage:
    python examples/analysis/inventory_deadline_decisions.py cv0.5_tau600 cv0.5_tau150
"""

from __future__ import annotations

import importlib.util
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RECORDING = "inventory_deadline_recording"
"""Registered under its own name so `build_strategy` constructs it like any other strategy."""


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "headroom_probe", ROOT / "examples" / "analysis" / "headroom_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _register_recording() -> None:
    """A `InventoryDeadline` that records its own priced options, without changing a decision."""
    from srl.component.rollout.router.strategies.inventory_deadline import InventoryDeadline
    from srl.component.rollout.router.strategy import STRATEGY_REGISTRY, register_strategy

    if RECORDING in STRATEGY_REGISTRY:
        return

    @register_strategy(RECORDING)
    class Recording(InventoryDeadline):
        name = RECORDING

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.waits = 0
            self.places = 0
            self.admission_decisions = 0
            self.admission_waits = 0
            """Admission decisions the strategy itself resolved to DEFER_GROUP."""
            self.admission_places = 0
            self.admission_forced = 0
            """Admission decisions where defer was not an option (defer_valid=False or no feasible instance)."""
            self.admission_defer_wins = 0
            """Admission decisions where DEFER had a lower E than the best instance (delta > eps)."""
            self.concurrency_at_admission: list[int] = []
            """Fleet-wide active requests at each admission decision, from the strategy's own views."""
            self.deltas: list[float] = []
            """Per admission decision: best instance's E minus DEFER's E, nearest buffer, seconds."""
            self.epss: list[float] = []

        def select_context(self, req, candidates, context):
            session = self._session
            is_admission = session is not None and req.group_id in session.staged_groups
            if is_admission:
                group = session.staged_groups[req.group_id]
                if req.rid == group[0].rid:
                    options = session.evaluate_admission(group)
                    feasible = {k: v for k, v in options.items() if v is not None}
                    defer = feasible.get(None)
                    instances = [v for k, v in feasible.items() if k is not None]
                    if defer is not None and instances:
                        best = min(instances, key=lambda p: p.train_end[0])
                        delta = best.train_end[0] - defer.train_end[0]
                        eps = self._eps(session)
                        if math.isfinite(delta):
                            self.deltas.append(delta)
                        else:
                            self.deltas.append(math.inf if best.train_end[0] > defer.train_end[0] else -math.inf)
                        self.epss.append(eps)
                        if delta > eps:
                            self.admission_defer_wins += 1
                    elif not instances:
                        self.admission_forced += 1  # no feasible instance, will DEFER anyway
                    elif defer is None:
                        self.admission_forced += 1  # defer_valid=False, forced to place
                    self.concurrency_at_admission.append(sum(v.n_active for v in candidates))
                    self.admission_decisions += 1
            chosen = super().select_context(req, candidates, context)
            if chosen is None:
                self.waits += 1
            else:
                self.places += 1
            if is_admission:
                if chosen is None:
                    self.admission_waits += 1
                else:
                    self.admission_places += 1
            return chosen


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def main() -> None:
    _register_recording()
    probe = _load_probe()
    fa = probe._load_forecast_accuracy()
    mdt = fa._load_dispersed_trace_module()
    cost_model = fa._composed_cost_model()

    for name in sys.argv[1:]:
        point = next(
            (p for grid in ("best", "heldout") for p in probe.points(grid) if p.name == name),
            None,
        )
        if point is None:
            print(f"Unknown point {name!r}. Known: {[p.name for g in ('best','heldout') for p in probe.points(g)]}")
            continue
        for label, strategy, cap in (
            ("static_cap(48)", "request_num_balance", 48),
            ("inventory_deadline", RECORDING, 0),
        ):
            regime = probe._regime(fa, point, strategy=strategy, strategy_args={}, cap=cap)
            sim = fa._build_medium(regime, cost_model, mdt)
            result = sim.run()
            mean_conc, peak = probe._concurrency(sim, point)
            line = f"{name:<16}{label:<20}span={result.sim_time:>7.0f}  conc={mean_conc:>5.1f}  peak={peak:>4}"
            if strategy != RECORDING:
                print(line)
                continue
            strat = sim.router.strategy
            print(f"{line}  admissions={strat.admission_decisions}  placed={strat.places}  waited={strat.waits}")
            print(
                f"{'':16}admission decisions chose DEFER on "
                f"{strat.admission_waits}/{strat.admission_decisions}, "
                f"forced (no DEFER option): {strat.admission_forced}, "
                f"defer wins vs eps: {strat.admission_defer_wins}/{len(strat.deltas)}, "
                f"fleet active at that moment: median={statistics.median(strat.concurrency_at_admission):.0f} "
                f"p90={_pct(strat.concurrency_at_admission, 0.90):.0f}"
            )
            if strat.deltas:
                positives = [d for d in strat.deltas if d > 0.0]
                mean_eps = statistics.mean(strat.epss)
                print(
                    f"{'':16}placing minus waiting, nearest-buffer E (s, >0 means waiting is "
                    f"cheaper): p10={_pct(strat.deltas, 0.10):+.3f} "
                    f"median={statistics.median(strat.deltas):+.3f} "
                    f"p90={_pct(strat.deltas, 0.90):+.3f}, "
                    f"waiting wins on {len(positives)}/{len(strat.deltas)} decisions, "
                    f"eps~{mean_eps:.3f}s"
                )
        print()


if __name__ == "__main__":
    main()
