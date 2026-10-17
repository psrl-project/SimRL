#!/usr/bin/env python
"""G1 pass 1: does `inventory_deadline` actually run faster, not just forecast accurately.

Everything verified so far (G0, and `forecast_accuracy.py`'s self-consistency re-check) is about
whether the forecaster predicts buffer timing correctly. None of it says whether the strategy's
decisions make the run finish sooner. This grid runs the design doc's go/no-go comparison (see
"## Phases and gates" step 4) at the cheap toy scale: `psrl_default` (no cap), `static_cap` (best
of a small per-instance cap grid), `static_cap_lpt` (the same cap grid with `length_aware`
longest-first placement), and `inventory_deadline` (no cap -- it has none to tune), on the exact
same ten regimes `forecast_accuracy.py --scale toy` already validated the forecaster's accuracy on.

Every regime runs a fixed number of training versions, so "faster" means lower `sim_time`
(equivalently higher `throughput`, tokens/sim_time) to finish the same amount of training work --
not a fixed-wall-time token count. `tokens_per_group` (consumed tokens / distinct consumed groups)
is the sanity check this project has been burned by skipping before: a win that moves it away from
`psrl_default`'s value is training on a different, not-comparable set of responses, not a real win.

Usage:
    python examples/analysis/inventory_sched_sweep.py run toy
    python examples/analysis/inventory_sched_sweep.py report toy
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from srl.simulator.config import SimConfig
from srl.simulator.simulator import Simulator

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs" / "simulate" / "inventory_sched"

CAP_GRID = (8, 12, 16, 24, 32, 48, 64, 96)
ARMS = ("psrl_default", "static_cap", "static_cap_lpt", "inventory_deadline")


def _load_forecast_accuracy():
    """Reuse `forecast_accuracy.py`'s toy regimes and cost models directly, so this grid's numbers
    sit on the exact configs G0's forecast-accuracy gate already validated."""
    spec = importlib.util.spec_from_file_location(
        "forecast_accuracy", ROOT / "examples" / "analysis" / "forecast_accuracy.py"
    )
    module = importlib.util.module_from_spec(spec)
    # `dataclass(frozen=True)` resolves annotations via `sys.modules[cls.__module__]`, so the
    # module must be registered before `exec_module` runs its class bodies.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class ArmResult:
    arm: str
    cap: int
    span: float
    throughput: float
    consumed_tokens: int
    consumed_groups: int
    tokens_per_group: float


def _run_one(fa, regime, *, strategy: str, strategy_args: dict, cap: int) -> ArmResult:
    cfg = SimConfig(
        batch_size=regime.batch,
        group_size=regime.group_size,
        staleness_horizon=regime.staleness,
        n_versions=regime.n_versions,
        n_instances=regime.n_instances,
        weight_load_time=regime.weight_load,
        sync_time=0.5,
        max_concurrent_seqs_per_instance=cap,
        workload=fa._workload(regime),
        log_telemetry=False,
        routing_strategy=strategy,
        routing_strategy_args=strategy_args,
    )
    sim = Simulator(
        rollout_cost=fa.UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
        training_cost=fa.LinearTrainingCost(regime.time_per_token),
        sync_cost=fa.FixedSyncCost(0.5),
        policy=fa.PSRLPolicy(),
        cfg=cfg,
    )
    result = sim.run()
    consumed_tokens = sum(r.generated for r in result.consumed)
    consumed_groups = len({r.group_id for r in result.consumed})
    return ArmResult(
        arm="",
        cap=cap,
        span=result.sim_time,
        throughput=result.throughput,
        consumed_tokens=consumed_tokens,
        consumed_groups=consumed_groups,
        tokens_per_group=consumed_tokens / consumed_groups if consumed_groups else float("nan"),
    )


def _best_of_cap_grid(fa, regime, *, strategy: str, strategy_args: dict) -> tuple[ArmResult, list[ArmResult]]:
    candidates = [_run_one(fa, regime, strategy=strategy, strategy_args=strategy_args, cap=cap) for cap in CAP_GRID]
    best = max(candidates, key=lambda r: r.throughput)
    return best, candidates


def run_regime(fa, regime) -> dict:
    psrl_default = _run_one(fa, regime, strategy="request_num_balance", strategy_args={}, cap=0)
    static_cap, cap_candidates = _best_of_cap_grid(fa, regime, strategy="request_num_balance", strategy_args={})
    static_cap_lpt, lpt_candidates = _best_of_cap_grid(
        fa, regime, strategy="length_aware", strategy_args={"order": "lpt", "balance": "load"}
    )
    inventory_deadline = _run_one(fa, regime, strategy="inventory_deadline", strategy_args={}, cap=0)

    arms = {
        "psrl_default": psrl_default,
        "static_cap": static_cap,
        "static_cap_lpt": static_cap_lpt,
        "inventory_deadline": inventory_deadline,
    }
    return {
        "regime": regime.name,
        "arms": {name: asdict(result) for name, result in arms.items()},
        "static_cap_grid": [asdict(r) for r in cap_candidates],
        "static_cap_lpt_grid": [asdict(r) for r in lpt_candidates],
    }


def _run(args: argparse.Namespace) -> None:
    fa = _load_forecast_accuracy()
    out_dir = OUT_ROOT / args.grid
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime in fa.default_regimes():
        row = run_regime(fa, regime)
        rows.append(row)
        print(f"{regime.name} done: " + ", ".join(f"{a}={row['arms'][a]['span']:.0f}s" for a in ARMS))
    path = out_dir / "results.json"
    path.write_text(json.dumps(rows, indent=1))
    print(f"Wrote {path}")


def _report(args: argparse.Namespace) -> None:
    path = OUT_ROOT / args.grid / "results.json"
    rows = json.loads(path.read_text())
    print(
        f"{'regime':<22}{'arm':<20}{'cap':>5}{'span':>8}{'tput':>9}{'vs base':>9}{'tok/grp':>9}"
        f"{'grp_ratio':>10}"
    )
    speedups: dict[str, list[float]] = {a: [] for a in ARMS}
    for row in rows:
        base = row["arms"]["psrl_default"]
        for arm in ARMS:
            a = row["arms"][arm]
            speedup = a["throughput"] / base["throughput"]
            grp_ratio = a["tokens_per_group"] / base["tokens_per_group"]
            speedups[arm].append(speedup)
            flag = "" if arm == "psrl_default" or 0.97 <= grp_ratio <= 1.03 else "  <-- consumed set shifted"
            print(
                f"{row['regime']:<22}{arm:<20}{a['cap']:>5}{a['span']:>8.0f}{a['throughput']:>9.1f}"
                f"{speedup:>8.3f}x{a['tokens_per_group']:>9.0f}{grp_ratio:>10.3f}{flag}"
            )
        print()
    print("Geomean speedup vs psrl_default, across regimes:")
    for arm in ARMS:
        values = speedups[arm]
        geomean = statistics.geometric_mean(values) if values else float("nan")
        print(f"  {arm:<20}{geomean:>8.3f}x  (min {min(values):.3f}x, max {max(values):.3f}x)")


GRIDS = {"toy": None}
"""Only one grid exists today; kept as a dict so a second one slots in without touching the CLI."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("grid", choices=sorted(GRIDS))
    args = parser.parse_args()
    if args.command == "run":
        _run(args)
    else:
        _report(args)


if __name__ == "__main__":
    main()
