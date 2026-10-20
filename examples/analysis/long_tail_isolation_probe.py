#!/usr/bin/env python
"""Does `inventory_deadline` protect a rare long-tail group from short-request contamination, on
`long_tail_isolation.py`'s bimodal workload?

Three questions, in order:

1. Mechanism: under a baseline that mixes short and long groups indiscriminately
   (`request_num_balance`, no cap), does a long-tail group's host instance visibly admit short
   groups while it is in flight, and is that associated with the long-tail group's own
   admit-to-complete time inflating?
2. Placement ablation: with the admission gate off (`throughput_margin_frac=0`), does
   `horizon_buffers` alone (1..4) change anything?
3. Admission gate: does `throughput_margin_frac` (0, 0.01, 0.02, 0.03, 0.05) recover a win, and if
   so is it because placement started avoiding contamination or because the gate is just capping
   concurrency again?

Also scores forecast accuracy (`sample.ready` vs the real buffer READY time) separately for
buffers whose consumed set includes a long-tail group versus buffers that are short-only, using
the `MediumSample`/`true_ready` data `_MediumSimulator` (which `long_tail_isolation.build` already
instantiates) records on every run, for every arm -- no separate harness needed.

Usage:
    python examples/analysis/long_tail_isolation_probe.py run baseline
    python examples/analysis/long_tail_isolation_probe.py run placement
    python examples/analysis/long_tail_isolation_probe.py run gate
    python examples/analysis/long_tail_isolation_probe.py report <grid>
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = Path("/tmp/long_tail_isolation_probe")

SEEDS = (0, 1, 2)
CAP_GRID = (24, 32, 40, 48, 56, 64)
"""Same grid `headroom_probe.py` sweeps, swept fresh here too."""

LONG_THRESHOLD = 15000
"""Separates the short mode (max draw well under 2000) from the long mode (~30000 +- jitter)."""


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_lti():
    return _load("long_tail_isolation", ROOT / "examples" / "analysis" / "long_tail_isolation.py")


def _classify_long(groups) -> dict[int, bool]:
    return {g.group_id: max(g.output_lens) > LONG_THRESHOLD for g in groups}


def _avg_active(intervals: list[tuple[float, float]], start: float, end: float) -> float:
    """Time-weighted average number of `intervals` overlapping `[start, end]`, clipped to it."""
    if end <= start:
        return 0.0
    events: list[tuple[float, int]] = []
    for s, e in intervals:
        cs, ce = max(s, start), min(e, end)
        if ce <= cs:
            continue
        events.append((cs, 1))
        events.append((ce, -1))
    if not events:
        return 0.0
    events.sort()
    current = 0
    area = 0.0
    previous = start
    for t, delta in events:
        area += current * (t - previous)
        current += delta
        previous = t
    area += current * (end - previous)
    return area / (end - start)


def _mechanism_rows(sim, is_long: dict[int, bool]) -> list[dict]:
    """Per long-tail group: host instance(s), admit-to-complete window, realized throughput, the
    number of *distinct short groups* whose first member was admitted onto a host instance while
    the window was open, and the time-averaged *external* concurrency (every other request's
    presence on the host instance during this group's own residency, net of the group's own
    members) -- the crowding the hypothesis says inflates a long-tail group's own decode time.

    External concurrency is a rate, not a count, so it is not mechanically inflated by a longer
    window the way a raw admission count in `[start, end]` would be.
    """
    spans_by_instance: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for req in sim.consumed:
        if req.complete_time is None:
            continue
        spans_by_instance[req.instance_id].append((req.admit_time, req.complete_time))

    long_members: dict[int, list[tuple[float, float, int, int]]] = defaultdict(list)
    short_first_admit: dict[int, tuple[float, int]] = {}
    for req in sim.consumed:
        if req.complete_time is None:
            continue
        if is_long.get(req.source_group_id, False):
            long_members[req.source_group_id].append(
                (req.admit_time, req.complete_time, req.instance_id, req.generated)
            )
        else:
            prior = short_first_admit.get(req.source_group_id)
            if prior is None or req.admit_time < prior[0]:
                short_first_admit[req.source_group_id] = (req.admit_time, req.instance_id)

    rows = []
    for gid, members in long_members.items():
        start = min(m[0] for m in members)
        end = max(m[1] for m in members)
        tokens = sum(m[3] for m in members)
        hosts = sorted({m[2] for m in members})
        dur = end - start

        ext_weighted, weight_total = 0.0, 0.0
        for admit, complete, instance_id, _generated in members:
            member_dur = complete - admit
            if member_dur <= 0:
                continue
            ext = max(0.0, _avg_active(spans_by_instance[instance_id], admit, complete) - 1.0)
            ext_weighted += ext * member_dur
            weight_total += member_dur
        ext_concurrency = ext_weighted / weight_total if weight_total > 0 else float("nan")

        contam_groups = sum(
            1 for (t, inst) in short_first_admit.values() if inst in hosts and start <= t <= end
        )
        rows.append(
            {
                "group_id": gid,
                "dur": dur,
                "tokens": tokens,
                "throughput": tokens / dur if dur > 0 else float("nan"),
                "n_members": len(members),
                "n_hosts": len(hosts),
                "contam_groups": contam_groups,
                "ext_concurrency": ext_concurrency,
            }
        )
    return rows


def _buffer_is_long(sim, is_long: dict[int, bool]) -> dict[int, bool]:
    """Which training buffer (= consume_version) ever consumed a long-tail group."""
    by_version: dict[int, set[int]] = defaultdict(set)
    for req in sim.consumed:
        if req.consume_version is None:
            continue
        by_version[req.consume_version].add(req.source_group_id)
    return {version: any(is_long.get(gid, False) for gid in gids) for version, gids in by_version.items()}


def _forecast_rows(sim, result, buffer_is_long: dict[int, bool]) -> list[dict]:
    """Signed relative forecast error on `sample.ready`, bucketed by whether the target buffer
    ever consumed a long-tail group. Mirrors `forecast_accuracy.score_medium`'s scorability rule."""
    true_ready = sim.policy.true_ready
    true_end = {index: end for index, (_, end) in enumerate(result.train_timestamps)}
    rows = []
    for sample in sim.samples:
        if sample.b0 is None:
            continue
        for buffer_id in (sample.b0, sample.b0 + 1):
            if buffer_id not in true_ready or buffer_id not in true_end:
                continue
            actual_f = true_ready[buffer_id]
            if sample.now >= actual_f or actual_f <= 0:
                continue
            predicted = sample.ready.get(buffer_id, math.inf)
            if not math.isfinite(predicted):
                continue
            rows.append(
                {
                    "buffer_id": buffer_id,
                    "is_long": buffer_is_long.get(buffer_id, False),
                    "signed_err": (predicted - actual_f) / actual_f,
                }
            )
    return rows


def _run_one(seed: int, arm: str, strategy: str, strategy_args: dict, cap: int, train_s: float) -> dict:
    lti = _load_lti()
    fa = lti._load_forecast_accuracy()
    sys.path.insert(0, str(ROOT / "workflow" / "validate_performance" / "scripts"))
    import collect  # noqa: E402

    cost_model = fa._composed_cost_model()
    point = lti.TailPoint(
        name=f"{arm}_seed{seed}",
        train_s=train_s,
        seed=seed,
        strategy=strategy,
        strategy_args=strategy_args,
        cap=cap,
    )
    sim, n_long, n_groups, kv_blocks = lti.build(point, cost_model, fa)
    result = sim.run()
    measured = collect.measure(sim, result, point.n_instances)
    is_long = _classify_long(sim.cfg.workload.groups)
    mechanism = _mechanism_rows(sim, is_long)
    buf_long = _buffer_is_long(sim, is_long)
    forecast = _forecast_rows(sim, result, buf_long)
    row = {
        "seed": seed,
        "arm": arm,
        "strategy": strategy,
        "strategy_args": strategy_args,
        "cap": cap,
        "measure": measured,
        "mechanism": mechanism,
        "forecast": forecast,
        "n_long_pool": n_long,
        "n_groups_pool": n_groups,
    }
    print(
        f"seed={seed} {arm:<22} span={measured['span']:.0f} groups={measured['consumed_groups']} "
        f"tok/grp={measured['tokens_per_group']:.0f} n_long_consumed={len(mechanism)}",
        flush=True,
    )
    return row


TRAIN_S = 345.0
"""One fixed training time for every arm; the hypothesis is about placement, not the tau axis."""


def _baseline_jobs() -> list[tuple]:
    jobs = []
    for seed in SEEDS:
        jobs.append((seed, "psrl_default", "request_num_balance", {}, 0, TRAIN_S))
        for cap in CAP_GRID:
            jobs.append((seed, f"fifo_cap{cap}", "request_num_balance", {}, cap, TRAIN_S))
        for cap in CAP_GRID:
            lpt_args = {"order": "lpt", "balance": "load", "window_groups": 32}
            jobs.append((seed, f"lpt_cap{cap}", "length_aware", lpt_args, cap, TRAIN_S))
    return jobs


def _placement_jobs() -> list[tuple]:
    jobs = []
    for seed in SEEDS:
        for h in (1, 2, 3, 4):
            args = {"throughput_margin_frac": 0.0, "horizon_buffers": h}
            jobs.append((seed, f"inv_h{h}_tm0", "inventory_deadline", args, 0, TRAIN_S))
    return jobs


def _gate_jobs(horizon: int) -> list[tuple]:
    jobs = []
    for seed in SEEDS:
        for tm in (0.0, 0.01, 0.02, 0.03, 0.05):
            args = {"throughput_margin_frac": tm, "horizon_buffers": horizon}
            jobs.append((seed, f"inv_h{horizon}_tm{tm:g}", "inventory_deadline", args, 0, TRAIN_S))
    return jobs


GRIDS = {
    "baseline": _baseline_jobs,
    "placement": _placement_jobs,
    "gate": lambda: _gate_jobs(2),
}


def _run(args: argparse.Namespace) -> None:
    out_dir = OUT_ROOT / args.grid
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = GRIDS[args.grid]()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_run_one, *zip(*jobs)))
    path = out_dir / "results.json"
    path.write_text(json.dumps(rows, indent=1))
    print(f"Wrote {path} ({len(rows)} rows)")


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(set(xs)) < 2:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _report_baseline(rows: list[dict]) -> None:
    by_seed_arm: dict[tuple[int, str], dict] = {(r["seed"], r["arm"]): r for r in rows}
    print(f"{'seed':>5}{'arm':<16}{'span':>9}{'tok/grp':>10}{'n_long':>8}")
    for (seed, arm), r in sorted(by_seed_arm.items()):
        m = r["measure"]
        print(f"{seed:>5}{arm:<16}{m['span']:>9.0f}{m['tokens_per_group']:>10.0f}{len(r['mechanism']):>8}")

    print("\nBest static per seed vs psrl_default:")
    for seed in SEEDS:
        base = by_seed_arm[(seed, "psrl_default")]["measure"]["span"]
        fifo_best = min(by_seed_arm[(seed, f"fifo_cap{c}")]["measure"]["span"] for c in CAP_GRID)
        lpt_best = min(by_seed_arm[(seed, f"lpt_cap{c}")]["measure"]["span"] for c in CAP_GRID)
        print(
            f"  seed={seed} psrl_default={base:.0f} best_fifo={fifo_best:.0f} "
            f"best_lpt={lpt_best:.0f} (speedup {base / min(fifo_best, lpt_best):.4f}x)"
        )

    print("\nMechanism check, pooled across seeds, psrl_default (indiscriminate mixing):")
    pooled = [row for arm in ("psrl_default",) for seed in SEEDS for row in by_seed_arm[(seed, arm)]["mechanism"]]
    _print_mechanism(pooled)
    print("\nMechanism check, pooled across seeds, fifo_cap48 (a representative static cap):")
    pooled48 = [row for seed in SEEDS for row in by_seed_arm[(seed, "fifo_cap48")]["mechanism"]]
    _print_mechanism(pooled48)


def _print_mechanism(pooled: list[dict]) -> None:
    if not pooled:
        print("  no long-tail groups consumed")
        return
    contams = [r["contam_groups"] for r in pooled]
    ext = [r["ext_concurrency"] for r in pooled]
    throughput = [r["throughput"] for r in pooled]
    corr_ext = _pearson(ext, throughput)
    n_contaminated = sum(1 for c in contams if c > 0)
    low = [r["throughput"] for r in pooled if r["ext_concurrency"] <= statistics.median(ext)]
    high = [r["throughput"] for r in pooled if r["ext_concurrency"] > statistics.median(ext)]
    print(
        f"  n_long_groups={len(pooled)} ever_shared_instance_with_a_new_short_group={n_contaminated} "
        f"({n_contaminated / len(pooled):.1%}) mean_contam_groups={statistics.mean(contams):.1f} "
        f"mean_ext_concurrency={statistics.mean(ext):.2f} corr(ext_concurrency, throughput)={corr_ext:.3f}"
    )
    print(f"  median throughput tok/s: low ext_concurrency half -> {_med(low):.2f} | high half -> {_med(high):.2f}")
    multi_host = sum(1 for r in pooled if r["n_hosts"] > 1)
    print(f"  groups split across >1 instance: {multi_host}/{len(pooled)}")


def _med(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _report_forecast(rows: list[dict], label: str) -> None:
    long_err = [abs(f["signed_err"]) for r in rows for f in r["forecast"] if f["is_long"]]
    short_err = [abs(f["signed_err"]) for r in rows for f in r["forecast"] if not f["is_long"]]
    long_signed = [f["signed_err"] for r in rows for f in r["forecast"] if f["is_long"]]
    print(f"\nForecast accuracy, {label} (pooled seeds):")
    print(
        f"  long-tail-bound buffers: n={len(long_err)} F_median={_med(long_err):.3%} "
        f"signed_median={_med(long_signed):.3%}"
    )
    print(f"  short-only buffers:      n={len(short_err)} F_median={_med(short_err):.3%}")


def _report_placement_or_gate(args: argparse.Namespace, rows: list[dict], baseline_rows: list[dict]) -> None:
    base_by_seed = {r["seed"]: r["measure"]["span"] for r in baseline_rows if r["arm"] == "psrl_default"}
    static_by_seed = {}
    for seed in SEEDS:
        fifo_best = min(
            r["measure"]["span"] for r in baseline_rows if r["seed"] == seed and r["arm"].startswith("fifo_cap")
        )
        lpt_best = min(
            r["measure"]["span"] for r in baseline_rows if r["seed"] == seed and r["arm"].startswith("lpt_cap")
        )
        static_by_seed[seed] = min(fifo_best, lpt_best)

    print(f"{'seed':>5}{'arm':<16}{'span':>9}{'vs psrl_default':>18}{'vs best static':>16}{'tok/grp ok':>12}")
    base_tpg = {r["seed"]: r["measure"]["tokens_per_group"] for r in baseline_rows if r["arm"] == "psrl_default"}
    for r in sorted(rows, key=lambda r: (r["seed"], r["arm"])):
        seed, m = r["seed"], r["measure"]
        ok = "yes" if abs(m["tokens_per_group"] / base_tpg[seed] - 1.0) <= 0.01 else "NO"
        print(
            f"{seed:>5}{r['arm']:<16}{m['span']:>9.0f}"
            f"{base_by_seed[seed] / m['span']:>18.4f}{static_by_seed[seed] / m['span']:>16.4f}{ok:>12}"
        )
        if ok == "NO":
            print(f"    <-- consumed-set gate failed, dropped from any mean")

    print("\nGeomean vs best static, per arm, over seeds that passed the consumed-set gate:")
    by_arm: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        seed, m = r["seed"], r["measure"]
        if abs(m["tokens_per_group"] / base_tpg[seed] - 1.0) <= 0.01:
            by_arm[r["arm"]].append(static_by_seed[seed] / m["span"])
    for arm, vals in sorted(by_arm.items()):
        gm = math.exp(statistics.mean(math.log(v) for v in vals)) if vals else float("nan")
        print(f"  {arm:<16} n={len(vals)} geomean_speedup_vs_best_static={gm:.4f}")


def _report(args: argparse.Namespace) -> None:
    rows = json.loads((OUT_ROOT / args.grid / "results.json").read_text())
    if args.grid == "baseline":
        _report_baseline(rows)
        for arm in ("psrl_default",):
            pooled = [r for r in rows if r["arm"] == arm]
            _report_forecast(pooled, arm)
    else:
        baseline_path = OUT_ROOT / "baseline" / "results.json"
        if not baseline_path.exists():
            raise SystemExit("Run the baseline grid first: `run baseline`.")
        baseline_rows = json.loads(baseline_path.read_text())
        _report_placement_or_gate(args, rows, baseline_rows)
        best_arm = max(
            {r["arm"] for r in rows},
            key=lambda arm: statistics.mean(
                r["measure"]["span"] for r in rows if r["arm"] == arm
            ) * -1,
        )
        for arm in sorted({r["arm"] for r in rows}):
            pooled = [r for r in rows if r["arm"] == arm]
            if pooled:
                _report_forecast(pooled, arm)
        best_rows = [r for r in rows if r["arm"] == best_arm]
        print(f"\nMechanism check for {best_arm} (pooled seeds):")
        pooled_mech = [m for r in best_rows for m in r["mechanism"]]
        _print_mechanism(pooled_mech)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("grid", choices=sorted(GRIDS))
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.command == "run":
        _run(args)
    else:
        _report(args)


if __name__ == "__main__":
    main()
