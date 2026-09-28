"""Standalone CLI: render the instance-load plot from a real PSRL run's logs.

This is the measured counterpart to `dryrun.cli.plot`. It converts a PSRL log
directory into the same telemetry streams a simulation writes, then renders the
figure with the same `SimPlotter`, so a real run and a simulated one can be
compared without either side having its own drawing code.

The converted streams are written to `<output_dir>/logs` in the standard run
layout, so the result is inspectable as JSONL and can be re-rendered later with
`dryrun-plot <output_dir>`.

Only `instance_load` is rendered: the other plots need per-request and
per-engine-step streams that a PSRL run does not record, and would come out as
empty placeholders.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from .._logging import setup_logging
from ..telemetry.psrl_log import PsrlLogLayout, convert_psrl_logs
from ..telemetry.store import JsonlStore
from ..visualizer.plots import SimPlotter


def _parse_instances(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part) for part in value.replace(",", " ").split()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dryrun-psrl-plot",
        description="Render the DryRun-RL instance-load plot from a real PSRL run's logs.",
    )
    parser.add_argument("log_dir", help="A PSRL run's log directory. Read only, never modified.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the converted logs and the plot into.",
    )
    parser.add_argument(
        "--role",
        default=PsrlLogLayout.role,
        help="Replica role to plot. Validation replicas write the same stats files as rollout ones.",
    )
    parser.add_argument(
        "--instances",
        default=None,
        help="Replica indices to plot, e.g. '0,2'. Overrides role-based selection.",
    )
    parser.add_argument(
        "--utc-offset-hours",
        type=float,
        default=None,
        help="Offset the text logs were written in. Inferred from the logs when omitted.",
    )
    parser.add_argument(
        "--time-label",
        default="Wall-clock Time (s)",
        help="X-axis label for the rendered figure.",
    )
    parser.add_argument(
        "--logs-subdir", default="logs", help="Name of the logs subdirectory to write under output_dir."
    )
    parser.add_argument(
        "--plots-subdir", default="plots", help="Name of the plots subdirectory to write under output_dir."
    )

    # Every field of the layout is exposed, so a run that names its files
    # differently is retargeted from the command line rather than by editing
    # the converter.
    layout_group = parser.add_argument_group("log layout overrides")
    skip = {"role"}
    for spec in fields(PsrlLogLayout):
        if spec.name in skip:
            continue
        layout_group.add_argument(
            f"--{spec.name.replace('_', '-')}",
            default=spec.default,
            help=f"Default: {spec.default!r}",
        )
    return parser


def main() -> None:
    setup_logging()

    args = _build_parser().parse_args()
    layout = PsrlLogLayout(**{spec.name: getattr(args, spec.name) for spec in fields(PsrlLogLayout)})

    output_dir = Path(args.output_dir)
    store = JsonlStore(output_dir / args.logs_subdir, overwrite=True)
    try:
        summary = convert_psrl_logs(
            args.log_dir,
            store,
            layout=layout,
            instance_ids=_parse_instances(args.instances),
            utc_offset_hours=args.utc_offset_hours,
        )
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
    print(f"  instances dropped : {summary.dropped_instance_ids} (role != {layout.role!r})")
    print(f"  occupancy samples : {summary.n_instance_rows}")
    print(f"  router samples    : {summary.n_gateway_rows}")
    print(f"  training steps    : {summary.n_train_steps}")
    print(f"  time span         : {summary.duration_s:.1f} s")
    print(f"  utc offset        : {summary.utc_offset_hours:+.2f} h")
    print(
        f"  mean per step     : recompute {summary.mean_recompute_s:.2f} s, "
        f"train {summary.mean_train_s:.2f} s, sync {summary.mean_sync_s:.2f} s"
    )
    print(f"Logs written to: {output_dir / args.logs_subdir}")
    print(f"Plot written to: {path}")


if __name__ == "__main__":
    main()
