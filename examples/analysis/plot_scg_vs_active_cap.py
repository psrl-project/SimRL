"""
Plot SCG(S) against the router's per-instance active cap.

Reads the request telemetry of finished `psrl_aligned.sh` runs that differ only in
`rollout.router.max_concurrent_seqs_per_instance`, and draws one curve per staleness
budget S. SCG is recomputed from the `consume` rows exactly as `SimResult.scg` defines
it, because `SimResult` is not persisted. Run it through `scg_vs_active_cap.sh`.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class RunSummary:
    """Hold one finished simulation, reduced to what the SCG curve needs."""

    cap: int
    span: float
    n_consumed: int
    tokens_by_staleness: dict[int, int]

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens_by_staleness.values())

    @property
    def throughput(self) -> float:
        """Return tokens per simulated second, with no staleness filter applied."""
        return self.total_tokens / self.span

    def scg(self, s: int) -> float:
        """
        Return SCG(S) in tokens per second.

        Defined for every S. SCG saturates at `throughput` once nothing is excluded,
        so an S above the run's observed maximum equals the point below it.

        Args:
            s (int): Staleness budget, in weight updates.

        Returns:
            float: Tokens per simulated second from trajectories with `staleness_true <= s`.
        """
        tokens = sum(
            count for staleness, count in self.tokens_by_staleness.items() if staleness <= s
        )
        return tokens / self.span


def load_run(run_dir: Path, cap: int) -> RunSummary:
    """
    Reduce one run's request telemetry to a `RunSummary`.

    Args:
        run_dir (Path): A simulation `output_dir`, holding `logs/request.jsonl`.
        cap (int): The active cap this run was made with.

    Returns:
        RunSummary: The run's span, consumed count, and tokens grouped by true staleness.

    Raises:
        FileNotFoundError: If the run has no request stream.
        ValueError: If the stream has no consume rows or a non-positive span.
    """
    stream = run_dir / "logs" / "request.jsonl"
    if not stream.is_file():
        raise FileNotFoundError(
            f"No request telemetry at {stream}. The run either did not finish or was "
            f"made with log_telemetry=False."
        )

    span = 0.0
    tokens_by_staleness: dict[int, int] = {}
    n_consumed = 0
    with stream.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            # The span ends at the last row of any kind, matching `SimResult.sim_time`.
            span = max(span, float(row.get("t", 0.0)))
            if row.get("event") != "consume" or row.get("staleness_true") is None:
                continue
            n_consumed += 1
            staleness = int(row["staleness_true"])
            tokens_by_staleness[staleness] = tokens_by_staleness.get(staleness, 0) + int(
                row.get("generated", 0)
            )

    if not n_consumed:
        raise ValueError(f"{stream} holds no consume rows, so SCG is undefined.")
    if span <= 0.0:
        raise ValueError(f"{stream} has a non-positive span, so a rate cannot be formed.")
    return RunSummary(
        cap=cap,
        span=span,
        n_consumed=n_consumed,
        tokens_by_staleness=tokens_by_staleness,
    )


def check_comparable(runs: list[RunSummary]) -> None:
    """
    Refuse to plot runs that did not consume the same amount of work.

    SCG is a rate, so a run that ended early can score arbitrarily well.

    Args:
        runs (list[RunSummary]): Every run destined for the figure.

    Raises:
        ValueError: If the consumed trajectory counts disagree.
    """
    counts = {run.n_consumed for run in runs}
    if len(counts) > 1:
        detail = ", ".join(f"cap={run.cap}: {run.n_consumed}" for run in runs)
        raise ValueError(
            f"The runs consumed different trajectory counts, so their SCG rates are "
            f"not comparable: {detail}. Every cap must reach the same n_versions with "
            f"nothing dropped before the curves mean anything."
        )


def write_csv(runs: list[RunSummary], s_values: list[int], path: Path) -> None:
    """Write the plotted numbers, so the figure can be audited without re-running."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["active_cap", "span_s", "n_consumed", "throughput_tok_s"]
            + [f"scg_s{s}_tok_s" for s in s_values]
        )
        for run in runs:
            writer.writerow(
                [
                    run.cap,
                    f"{run.span:.2f}",
                    run.n_consumed,
                    f"{run.throughput:.1f}",
                ]
                + [f"{run.scg(s):.1f}" for s in s_values]
            )


def plot(runs: list[RunSummary], s_values: list[int], path: Path, policy: str) -> None:
    """Draw one line per staleness budget over the active-cap axis."""
    caps = [run.cap for run in runs]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.viridis([i / max(len(s_values) - 1, 1) * 0.8 for i in range(len(s_values))])

    for s, color in zip(s_values, colors):
        ax.plot(
            caps,
            [run.scg(s) for run in runs],
            marker="o",
            markersize=5,
            linewidth=2,
            color=color,
            label=f"S = {s}",
        )

    ax.set_xlabel("Router active cap  (max resident requests per instance)")
    ax.set_ylabel("SCG(S)  [tokens / s]")
    ax.set_title(f"Staleness-Constrained Goodput vs the per-instance active cap  ({policy})")
    ax.set_xticks(caps)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Staleness budget")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-base",
        type=Path,
        required=True,
        help="Directory holding one cap<N> subdirectory per simulated cap.",
    )
    parser.add_argument(
        "--caps",
        type=int,
        nargs="+",
        required=True,
        help="Active cap values to read, in the order they should be plotted.",
    )
    parser.add_argument(
        "--max-staleness",
        type=int,
        required=True,
        help="Highest S to draw. SCG saturates at throughput above a run's own maximum.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where to write the CSV and PNG.",
    )
    parser.add_argument(
        "--out-prefix",
        default="scg_vs_active_cap",
        help="Basename for the CSV and PNG, so two policies do not overwrite each other.",
    )
    parser.add_argument(
        "--policy",
        default="",
        help="Staleness policy the runs were made with, recorded in the figure title.",
    )
    args = parser.parse_args()

    runs = [load_run(args.run_base / f"cap{cap}", cap) for cap in args.caps]
    check_comparable(runs)

    s_values = list(range(args.max_staleness + 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.out_prefix}.csv"
    png_path = args.out_dir / f"{args.out_prefix}.png"
    write_csv(runs, s_values, csv_path)
    plot(runs, s_values, png_path, args.policy)

    header = "  cap " + "".join(f"{f'S={s}':>12}" for s in s_values) + f"{'span (s)':>12}"
    print(header)
    for run in runs:
        row = "".join(f"{run.scg(s):>12.1f}" for s in s_values)
        print(f"  {run.cap:<4}{row}{run.span:>12.1f}")
    print()
    print(f"  All {len(runs)} runs consumed {runs[0].n_consumed} trajectories.")
    print(f"  Table:  {csv_path}")
    print(f"  Figure: {png_path}")


if __name__ == "__main__":
    main()
