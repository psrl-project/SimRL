"""Standalone CLI: render plots from a previous simulation run's logs.

Unlike `srl.cli.main`, this never re-simulates: it just points a
`SimPlotter` at an existing `logs/` directory (written by a prior `srl`
run) and regenerates the `plots/` directory from it. Useful for iterating on
plot styling, or for re-rendering after upgrading `srl.visualizer`, without
re-running an expensive simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .._logging import setup_logging
from ..telemetry.store import JsonlStore
from ..visualizer.plots import SimPlotter


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="srl-plot",
        description="Render SimRL plots from an existing run's logs, without re-simulating.",
    )
    parser.add_argument("run_dir", help="Run output directory (the --output_dir of a prior `srl` run).")
    parser.add_argument(
        "--logs-subdir", default="logs", help="Name of the logs subdirectory under run_dir."
    )
    parser.add_argument(
        "--plots-subdir", default="plots", help="Name of the plots subdirectory to write under run_dir."
    )
    parser.add_argument(
        "--only",
        choices=(
            "instance_load",
            "rollout_step",
            "staleness_distribution",
            "request_timeline",
            "throughput",
            "scg_curve",
        ),
        default=None,
        help="Render just one plot instead of all six.",
    )
    parser.add_argument(
        "--time-label",
        default="Simulation Time (s)",
        help="X-axis label. Match the real run's label when producing a figure to compare against it.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    logs_dir = run_dir / args.logs_subdir
    if not logs_dir.is_dir():
        raise SystemExit(
            f"No logs directory found at {logs_dir!s}. Did the run finish with log_telemetry enabled?"
        )

    store = JsonlStore(logs_dir)
    plotter = SimPlotter(store, run_dir / args.plots_subdir, time_label=args.time_label)
    paths = [getattr(plotter, f"plot_{args.only}")()] if args.only else plotter.plot_all()

    print(f"Plots saved to: {run_dir / args.plots_subdir}")
    for p in paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
