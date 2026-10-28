"""Standalone CLI: draw a staleness-throughput sweep from finished runs.

Never re-simulates. Takes a TSV manifest of `panel<TAB>label<TAB>run_dir` rows,
reduces each run's `request` stream, enforces that every point finished the same
training work, and writes one figure. The manifest exists because the sweep
runner is bash, which writes columns far more cleanly than it writes JSON.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .._logging import setup_logging
from ..diagnosis.staleness import check_same_consumption, staleness_breakdown
from ..telemetry.store import JsonlStore
from ..visualizer.staleness_sweep import Panel, plot_staleness_throughput


def load_manifest(path: Path, logs_subdir: str = "logs") -> list[Panel]:
    """Read a point manifest into panels, preserving row order.

    Args:
        path (Path): TSV with one `panel`, `label`, `run_dir` row per bar. An
            empty panel column puts every row in one unnamed panel.
        logs_subdir (str): Name of the log directory under each `run_dir`.

    Returns:
        list[Panel]: Panels in first-appearance order, bars in row order.

    Raises:
        SystemExit: If a row is malformed, a run directory has no log stream,
            or a run's own telemetry cannot be reduced (for example a
            livelocked point with no consumed trajectories). This is operator
            error in a generated file, so it reports the offending line
            rather than raising something a caller would catch.
    """
    grouped: dict[str, list] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise SystemExit(f"Malformed manifest {path!s} line {number}: expected 3 tab-separated fields.")
        panel, label, run_dir = (field.strip() for field in fields)
        logs = Path(run_dir) / logs_subdir
        if not (logs / "request.jsonl").is_file():
            raise SystemExit(f"No request stream under {run_dir!s} (manifest line {number}).")
        try:
            breakdown = staleness_breakdown(JsonlStore(logs))
        except ValueError as exc:
            raise SystemExit(f"{str(exc).rstrip('.')} (manifest line {number}).") from exc
        grouped.setdefault(panel, []).append((label, breakdown))
    if not grouped:
        raise SystemExit(f"Manifest {path!s} holds no points.")
    return [(panel, points) for panel, points in grouped.items()]


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="srl-staleness-plot",
        description="Draw a staleness-throughput sweep figure from finished simulation runs.",
    )
    parser.add_argument("manifest", type=Path, help="TSV of panel/label/run_dir rows, one per bar.")
    parser.add_argument("--primary-label", required=True, help="X axis label, naming the knob and its unit.")
    parser.add_argument("--out", type=Path, required=True, help="Destination PNG.")
    parser.add_argument(
        "--logs-subdir", default="logs", help="Name of the logs subdirectory under each run_dir."
    )
    parser.add_argument(
        "--allow-unequal-consumption",
        action="store_true",
        help="Draw even when points finished different amounts of training work. Off by default: "
        "throughput is a rate, so a point that ended early scores arbitrarily well.",
    )
    args = parser.parse_args()

    panels = load_manifest(args.manifest, logs_subdir=args.logs_subdir)
    every_point = [point for _, points in panels for point in points]
    if not args.allow_unequal_consumption:
        try:
            check_same_consumption(every_point)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    path = plot_staleness_throughput(
        panels,
        primary_label=args.primary_label,
        out_path=args.out,
    )

    header = f"  {'panel':<12}{'point':>8}{'span':>10}{'tok/s':>9}{'waste%':>8}  staleness"
    print(header)
    for panel, points in panels:
        for label, bd in points:
            share = 100.0 * bd.total_wasted_tokens / max(1, bd.total_consumed_tokens + bd.total_wasted_tokens)
            spread = " ".join(f"S{s}:{n / 1e6:.1f}M" for s, n in sorted(bd.consumed_tokens.items()))
            print(f"  {panel:<12}{label:>8}{bd.span:>10.0f}{bd.throughput:>9.0f}{share:>8.1f}  {spread}")
    counts = {bd.n_consumed for _, bd in every_point}
    if len(counts) == 1:
        print(f"\n  Every point consumed {every_point[0][1].n_consumed} trajectories.")
    else:
        print(f"\n  Points consumed different trajectory counts: {sorted(counts)}.")
    print(f"  {path}")


if __name__ == "__main__":
    main()
