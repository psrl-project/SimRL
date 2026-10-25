"""Standalone CLI: render the instance-load plot from a real verl
separate_async run's telemetry directory.

This is the verl counterpart to `srl.cli.psrl_plot`: it converts a verl
telemetry directory into the same streams a simulation writes, then renders
the figure with the same `SimPlotter`, so a real run and a simulated one can
be compared without either side having its own drawing code.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from .._logging import setup_logging
from ..telemetry.store import JsonlStore
from ..telemetry.verl_log import VerlLogLayout, convert_verl_logs
from ..visualizer.plots import SimPlotter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="srl-verl-plot",
        description="Render the SimRL instance-load plot from a real verl separate_async run's telemetry.",
    )
    parser.add_argument("log_dir", help="A verl run's telemetry directory (VERL_TELEMETRY_LOG_DIR). Read only.")
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write the converted logs and the plot into."
    )
    parser.add_argument("--time-label", default="Wall-clock Time (s)", help="X-axis label for the rendered figure.")
    parser.add_argument(
        "--logs-subdir", default="logs", help="Name of the logs subdirectory to write under output_dir."
    )
    parser.add_argument(
        "--plots-subdir", default="plots", help="Name of the plots subdirectory to write under output_dir."
    )

    layout_group = parser.add_argument_group("log layout overrides")
    for spec in fields(VerlLogLayout):
        layout_group.add_argument(
            f"--{spec.name.replace('_', '-')}", default=spec.default, help=f"Default: {spec.default!r}"
        )
    return parser


def main() -> None:
    setup_logging()

    args = _build_parser().parse_args()
    layout = VerlLogLayout(**{spec.name: getattr(args, spec.name) for spec in fields(VerlLogLayout)})

    output_dir = Path(args.output_dir)
    store = JsonlStore(output_dir / args.logs_subdir, overwrite=True)
    try:
        summary = convert_verl_logs(args.log_dir, store, layout=layout)
    finally:
        store.close()

    plotter = SimPlotter(
        JsonlStore(output_dir / args.logs_subdir),
        output_dir / args.plots_subdir,
        time_label=args.time_label,
    )
    path = plotter.plot_instance_load()

    print(f"Converted {args.log_dir}")
    print(f"  instances plotted : {summary.instance_ids}")
    print(f"  occupancy samples : {summary.n_instance_rows}")
    print(f"  completed requests: {summary.n_request_rows}")
    print(f"  time span         : {summary.duration_s:.1f} s")
    print(f"Logs written to: {output_dir / args.logs_subdir}")
    print(f"Plot written to: {path}")


if __name__ == "__main__":
    main()
