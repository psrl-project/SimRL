#!/usr/bin/env python
"""Where does `inventory_deadline` beat a tuned static cap, and where does it not?

G1 pass 1 (`inventory_sched_sweep.py`) showed routing is clean after the tie-break fix, but its
result at the headroom-study scale was mixed: a win at one regime and a loss at another. That
comparison ran as ad hoc one-liners, and two of its inputs did not match the regimes
`psrl-scheduling-headroom.md` actually measured (`MediumRegime`'s `mean`/`length_cap` defaults, and
`length_aware`'s `window_groups`, which the doc bounds to one training batch and the ad hoc runs left
at 0). This script makes the comparison a first-class, repeatable artifact: the same probe over a
grid of points, with the static arms' caps swept fresh (never assumed) and `window_groups` pinned to
the doc's own convention.

Each point also reports the per-instance concurrency the run actually sustained and the number of
preemptions, which is what distinguishes these arms: `inventory_deadline` has no cap to configure, so
the question it answers at every point is whether the concurrency it picks for itself is the right
one, and if not, in which direction.

Points are `psrl-scheduling-headroom.md` §3.4's own strongest regimes plus controls, at the doc's own
workload shape (`mean=8000`, `length_cap=32768`, `B=32 x 8`, 2 instances, S=1).

Usage:
    python examples/analysis/headroom_probe.py run best
    python examples/analysis/headroom_probe.py report best
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs" / "simulate" / "headroom_probe"

CAP_GRID = (24, 32, 40, 48, 56, 64)
"""The sweep the static arms' cap is chosen from, fresh at every point."""

WINDOW_GROUPS = 32
"""One training batch. `psrl-scheduling-headroom.md`'s own LPT jobs all bound reordering this way;
unbounded reordering starves one end of the length distribution and is not a fair comparison."""


def _load_forecast_accuracy():
    """Reuse `forecast_accuracy.py`'s medium-scale builder and cost model, so this probe's numbers
    are comparable with the G0/self-consistency runs on the same harness."""
    spec = importlib.util.spec_from_file_location(
        "forecast_accuracy", ROOT / "examples" / "analysis" / "forecast_accuracy.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Point:
    """One system configuration, at the doc's own workload shape unless overridden."""

    name: str
    cv: float
    train_s: float
    staleness: int = 1
    mean: int = 8000
    length_cap: int = 32768
    n_versions: int = 12
    batch: int = 32
    group_size: int = 8
    n_instances: int = 2
    weight_load: float = 0.0
    rho: float = 0.6
    ours_args: dict = field(default_factory=dict)
    """Strategy args for the gated `inventory_deadline_wm` arm; empty skips that arm."""


@dataclass
class ArmResult:
    arm: str
    strategy: str
    strategy_args: dict
    cap: int
    span: float
    consumed_tokens: int
    consumed_groups: int
    tokens_per_group: float
    mean_concurrency: float
    peak_concurrency: int
    preemptions: int


def points(grid: str) -> list[Point]:
    """`best`: the doc's strongest named regimes, plus controls at both ends of the tau axis."""
    if grid == "best":
        return [
            Point("cv0.5_tau600", cv=0.5, train_s=600.0),
            Point("cv1.0_tau600", cv=1.0, train_s=600.0),
            Point("cv1.0_tau345", cv=1.0, train_s=345.0),
            Point("cv0.5_tau345", cv=0.5, train_s=345.0),
            Point("cv0.5_tau150", cv=0.5, train_s=150.0),
            Point("cv0.5_tau600_s2", cv=0.5, train_s=600.0, staleness=2),
        ]
    if grid == "heldout":
        """Points that played no role in choosing any knob, for validating a value picked on
        `best`. Two interpolations, one cv extrapolation, and three axes `best` never moved:
        mean length (the dimensionless-knob claim's strongest test), arrival pressure, S=3."""
        return [
            Point("cv0.75_tau475", cv=0.75, train_s=475.0, ours_args={"kv_watermark": 0.6}),
            Point("cv0.5_tau225", cv=0.5, train_s=225.0, ours_args={"kv_watermark": 0.6}),
            Point("cv1.5_tau345", cv=1.5, train_s=345.0, ours_args={"kv_watermark": 0.6}),
            Point("mean12k_tau345", cv=0.5, train_s=345.0, mean=12000, ours_args={"kv_watermark": 0.6}),
            Point("rho0.5_tau600", cv=0.5, train_s=600.0, rho=0.5, ours_args={"kv_watermark": 0.6}),
            Point("s3_tau600", cv=0.5, train_s=600.0, staleness=3, ours_args={"kv_watermark": 0.6}),
        ]
    raise KeyError(f"Unknown grid {grid!r}. Known: ['best', 'heldout'].")


def _regime(fa, point: Point, *, strategy: str, strategy_args: dict, cap: int):
    return fa.MediumRegime(
        name=point.name,
        cv=point.cv,
        train_s=point.train_s,
        staleness=point.staleness,
        cap=cap,
        weight_load=point.weight_load,
        mean=point.mean,
        rho=point.rho,
        length_cap=point.length_cap,
        n_versions=point.n_versions,
        batch=point.batch,
        group_size=point.group_size,
        n_instances=point.n_instances,
        strategy=strategy,
        strategy_args=strategy_args,
    )


def _concurrency(sim, point: Point) -> tuple[float, int]:
    """
    Mean and peak concurrent requests per instance, reconstructed from the consumed set.

    Each consumed request's `admit_time` to `complete_time` is the span it held a slot, so a sweep
    over the merged admit/complete events gives the concurrency curve without needing telemetry.
    """
    spans: dict[int, list[tuple[float, float]]] = {}
    for req in sim.consumed:
        if req.complete_time is None:
            continue
        spans.setdefault(req.instance_id, []).append((req.admit_time, req.complete_time))

    means: list[float] = []
    peaks: list[int] = []
    for instance_id in range(point.n_instances):
        events: list[tuple[float, int]] = []
        for start, end in spans.get(instance_id, ()):
            events.append((start, 1))
            events.append((end, -1))
        if not events:
            means.append(0.0)
            peaks.append(0)
            continue
        events.sort()
        current = peak = 0
        area = 0.0
        previous = events[0][0]
        for time, delta in events:
            area += current * (time - previous)
            current += delta
            peak = max(peak, current)
            previous = time
        span = events[-1][0] - events[0][0]
        means.append(area / span if span else 0.0)
        peaks.append(peak)
    return statistics.mean(means), max(peaks)


def run_arm(fa, point: Point, cost_model, mdt, *, arm: str, strategy: str, strategy_args: dict, cap: int) -> ArmResult:
    regime = _regime(fa, point, strategy=strategy, strategy_args=strategy_args, cap=cap)
    sim = fa._build_medium(regime, cost_model, mdt)
    result = sim.run()
    tokens = sum(req.generated for req in sim.consumed)
    groups = len({req.group_id for req in sim.consumed})
    mean_conc, peak_conc = _concurrency(sim, point)
    return ArmResult(
        arm=arm,
        strategy=strategy,
        strategy_args=strategy_args,
        cap=cap,
        span=result.sim_time,
        consumed_tokens=tokens,
        consumed_groups=groups,
        tokens_per_group=tokens / groups if groups else float("nan"),
        mean_concurrency=mean_conc,
        peak_concurrency=peak_conc,
        preemptions=sum(req.num_preemptions for req in sim.consumed),
    )


def run_point(fa, point: Point, cost_model, mdt) -> dict:
    """Every arm this point is compared on, with the static arms' caps swept fresh."""
    arms: list[ArmResult] = [
        run_arm(
            fa, point, cost_model, mdt,
            arm="psrl_default", strategy="request_num_balance", strategy_args={}, cap=0,
        )
    ]

    fifo_grid = [
        run_arm(
            fa, point, cost_model, mdt,
            arm="fifo", strategy="request_num_balance", strategy_args={}, cap=cap,
        )
        for cap in CAP_GRID
    ]
    arms.append(dataclasses.replace(max(fifo_grid, key=lambda r: -r.span), arm="static_cap"))

    lpt_args: dict[str, Any] = {"order": "lpt", "balance": "load", "window_groups": WINDOW_GROUPS}
    lpt_grid = [
        run_arm(fa, point, cost_model, mdt, arm="lpt", strategy="length_aware", strategy_args=lpt_args, cap=cap)
        for cap in CAP_GRID
    ]
    arms.append(dataclasses.replace(max(lpt_grid, key=lambda r: -r.span), arm="static_cap_lpt"))

    arms.append(
        run_arm(
            fa, point, cost_model, mdt,
            arm="inventory_deadline", strategy="inventory_deadline", strategy_args={}, cap=0,
        )
    )
    if point.ours_args:
        arms.append(
            run_arm(
                fa, point, cost_model, mdt,
                arm="inventory_deadline_wm", strategy="inventory_deadline",
                strategy_args=dict(point.ours_args), cap=0,
            )
        )

    return {
        "point": asdict(point),
        "arms": {result.arm: asdict(result) for result in arms},
        "fifo_grid": [asdict(result) for result in fifo_grid],
        "lpt_grid": [asdict(result) for result in lpt_grid],
    }


def _run_point_worker(grid: str, point: Point) -> dict:
    """Run one point in its own process. Each is independent, so the grid parallelizes across them."""
    fa = _load_forecast_accuracy()
    mdt = fa._load_dispersed_trace_module()
    cost_model = fa._composed_cost_model()
    row = run_point(fa, point, cost_model, mdt)
    base = row["arms"]["psrl_default"]["span"]
    summary = ", ".join(
        f"{arm}={row['arms'][arm]['span'] / base:.4f}"
        for arm in ("static_cap", "static_cap_lpt", "inventory_deadline")
    )
    print(f"{point.name} done (span ratio vs default): {summary}", flush=True)
    return row


def _run(args: argparse.Namespace) -> None:
    out_dir = OUT_ROOT / args.grid
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = points(args.grid)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_run_point_worker, [args.grid] * len(todo), todo))

    path = out_dir / "results.json"
    path.write_text(json.dumps(rows, indent=1))
    print(f"Wrote {path}")


def _report(args: argparse.Namespace) -> None:
    rows = json.loads((OUT_ROOT / args.grid / "results.json").read_text())
    header = (
        f"{'point':<20}{'arm':<20}{'cap':>5}{'span':>9}{'speedup':>9}{'tok/grp':>9}"
        f"{'conc':>7}{'peak':>6}{'preempt':>9}"
    )
    print(header)
    for row in rows:
        base = row["arms"]["psrl_default"]
        arm_names = [a for a in ("psrl_default", "static_cap", "static_cap_lpt", "inventory_deadline", "inventory_deadline_wm") if a in row["arms"]]
        for arm in arm_names:
            a = row["arms"][arm]
            speedup = base["span"] / a["span"]
            ratio = a["tokens_per_group"] / base["tokens_per_group"]
            flag = "" if arm == "psrl_default" or 0.99 <= ratio <= 1.01 else "  <-- consumed set"
            print(
                f"{row['point']['name']:<20}{arm:<20}{a['cap']:>5}{a['span']:>9.0f}{speedup:>9.3f}"
                f"{a['tokens_per_group']:>9.0f}{a['mean_concurrency']:>7.1f}{a['peak_concurrency']:>6}"
                f"{a['preemptions']:>9}{flag}"
            )
        print()
    print("ours vs the better static arm, per point:")
    for row in rows:
        for arm in ("inventory_deadline", "inventory_deadline_wm"):
            if arm not in row["arms"]:
                continue
            ours = row["arms"][arm]["span"]
            best_static = min(row["arms"]["static_cap"]["span"], row["arms"]["static_cap_lpt"]["span"])
            cap48 = min(
                next(r["span"] for r in row["fifo_grid"] if r["cap"] == 48),
                next(r["span"] for r in row["lpt_grid"] if r["cap"] == 48),
            )
            verdict = "WINS" if ours < best_static else "loses"
            print(f"  {row['point']['name']:<20}{arm:<20}vs best static {best_static / ours:>8.4f}x  vs cap48 {cap48 / ours:>8.4f}x  {verdict}")


GRIDS = {"best": None, "heldout": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("grid", choices=sorted(GRIDS))
    parser.add_argument("--workers", type=int, default=6, help="Points to run concurrently.")
    args = parser.parse_args()
    if args.command == "run":
        _run(args)
    else:
        _report(args)


if __name__ == "__main__":
    main()
