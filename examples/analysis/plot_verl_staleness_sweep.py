"""Plot SCG(S) against verl's `staleness_threshold`, from `verl_staleness_sweep.sh`.

Reuses the cap sweep's reduction of `request.jsonl`, so SCG here is computed the
same way as in every other sweep figure: tokens carried by trajectories whose
`staleness_true <= S`, over the simulated span.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot_policy_cap_sweep import RunSummary, check_comparable, load_run  # noqa: E402


def plot_scg(
    runs: list[tuple[float, RunSummary]],
    s_values: list[int],
    title_suffix: str,
    out_dir: Path,
) -> Path:
    """SCG(S) for every requested S over the threshold axis."""
    thresholds = [thr for thr, _ in runs]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    shades = plt.cm.viridis([i / max(len(s_values) - 1, 1) * 0.8 for i in range(len(s_values))])
    for s, color in zip(s_values, shades):
        ax.plot(
            thresholds,
            [r.scg(s) for _, r in runs],
            marker="o",
            markersize=5,
            linewidth=2,
            color=color,
            label=f"S = {s}",
        )
    ax.set_xlabel("verl staleness_threshold  (sample-overage ratio)")
    ax.set_ylabel("SCG(S)  [tokens / s]")
    ax.set_title(f"Staleness-Constrained Goodput vs staleness_threshold  ({title_suffix})")
    ax.set_xticks(thresholds)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Staleness budget")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = out_dir / "scg_vs_threshold.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_span_and_staleness(
    runs: list[tuple[float, RunSummary]],
    title_suffix: str,
    out_dir: Path,
) -> Path:
    """The two quantities SCG folds together: wall-clock cost and how stale the data got."""
    thresholds = [thr for thr, _ in runs]
    fig, ax_span = plt.subplots(figsize=(9, 5.5))
    ax_span.plot(
        thresholds,
        [r.span for _, r in runs],
        marker="o",
        markersize=5,
        linewidth=2,
        color="#2E6F95",
        label="span",
    )
    ax_span.set_xlabel("verl staleness_threshold  (sample-overage ratio)")
    ax_span.set_ylabel("Simulated span  [s]", color="#2E6F95")
    ax_span.set_xticks(thresholds)
    ax_span.grid(True, alpha=0.3)

    ax_s = ax_span.twinx()
    ax_s.plot(
        thresholds,
        [r.max_staleness_seen for _, r in runs],
        marker="s",
        markersize=5,
        linewidth=1.5,
        linestyle="--",
        color="#B23A48",
        label="highest staleness consumed",
    )
    ax_s.set_ylabel("Highest staleness consumed", color="#B23A48")
    ax_s.set_ylim(bottom=0)
    ax_s.yaxis.get_major_locator().set_params(integer=True)

    ax_span.set_title(f"Span and staleness vs staleness_threshold  ({title_suffix})")
    fig.tight_layout()
    path = out_dir / "span_and_staleness_vs_threshold.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", required=True)
    parser.add_argument("--s-values", type=int, nargs="+", required=True)
    parser.add_argument("--title-suffix", default="verl")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    # Directory names keep the threshold as printed by the sweep, so read them
    # back verbatim rather than reformatting a float.
    runs = [
        (float(thr), load_run(args.run_base / f"thr{thr}", "verl", cap=0))
        for thr in args.thresholds
    ]
    check_comparable([r for _, r in runs])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_scg(runs, args.s_values, args.title_suffix, args.out_dir),
        plot_span_and_staleness(runs, args.title_suffix, args.out_dir),
    ]

    print("  thr     span" + "".join(f"{f'S={s}':>8}" for s in args.s_values) + "  maxS")
    for thr, r in runs:
        cells = "".join(f"{r.scg(s):>8.0f}" for s in args.s_values)
        print(f"  {thr:<4.1f}{r.span:>9.0f}{cells}{r.max_staleness_seen:>6d}")
    print()
    print(f"  All runs consumed {runs[0][1].n_consumed} trajectories.")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
