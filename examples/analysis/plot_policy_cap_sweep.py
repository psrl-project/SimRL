"""
Plot SCG(S) against the router active cap, one figure per policy plus overlays.

Reads the telemetry of finished `policy_cap_sweep.sh` runs and writes PNGs only. SCG
is recomputed from the `consume` rows exactly as `SimResult.scg` defines it, because
`SimResult` is not persisted. `span` comes from `srl.diagnosis.staleness`,
which reconciles the request and train streams the same way `SimResult.sim_time`
does.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from srl.diagnosis.staleness import StalenessBreakdown, check_same_consumption, staleness_breakdown
from srl.telemetry.store import JsonlStore

# One colour per policy, shared by every figure so the overlays and panels agree.
POLICY_COLOR = {"psrl": "#B23A48", "verl": "#2E6F95", "roll": "#4C9A2A"}


@dataclass(frozen=True)
class RunSummary:
    """One finished simulation, labelled by the sweep axis it came from.

    The staleness arithmetic lives in `srl.diagnosis.staleness` so this figure
    and the per-preset sweep figures cannot disagree about what SCG means.
    """

    policy: str
    cap: int
    breakdown: StalenessBreakdown

    @property
    def span(self) -> float:
        return self.breakdown.span

    @property
    def n_consumed(self) -> int:
        return self.breakdown.n_consumed

    @property
    def max_staleness_seen(self) -> int:
        return self.breakdown.max_staleness

    @property
    def throughput(self) -> float:
        return self.breakdown.throughput

    def scg(self, s: int) -> float:
        return self.breakdown.scg(s)


def load_run(run_dir: Path, policy: str, cap: int) -> RunSummary:
    """Reduce one run's request telemetry to a `RunSummary`.

    Raises:
        FileNotFoundError: If the run has no request stream.
        ValueError: If it has no consume rows or a non-positive span.
    """
    logs = run_dir / "logs"
    if not (logs / "request.jsonl").is_file():
        raise FileNotFoundError(f"No telemetry at {logs / 'request.jsonl'}.")
    return RunSummary(policy=policy, cap=cap, breakdown=staleness_breakdown(JsonlStore(logs)))


def check_comparable(runs: list[RunSummary]) -> None:
    """Refuse to plot runs that did not consume the same amount of work."""
    check_same_consumption([(f"{r.policy}/cap{r.cap}", r.breakdown) for r in runs])


def plot_per_policy(
    runs: dict[str, list[RunSummary]],
    baselines: dict[str, RunSummary],
    s_values: list[int],
    out_dir: Path,
) -> list[Path]:
    """
    Draw one panel per policy, with every staleness budget over the cap axis.

    The no-cap run is a dashed line in the colour of its S.
    """
    paths = []
    for policy, series in runs.items():
        caps = [r.cap for r in series]
        baseline = baselines[policy]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        shades = plt.cm.viridis(
            [i / max(len(s_values) - 1, 1) * 0.8 for i in range(len(s_values))]
        )
        for s, color in zip(s_values, shades):
            ax.plot(
                caps,
                [r.scg(s) for r in series],
                marker="o",
                markersize=5,
                linewidth=2,
                color=color,
                label=f"S = {s}",
            )
            ax.axhline(baseline.scg(s), color=color, linestyle="--", linewidth=1.5, alpha=0.8)
        # One legend entry stands for every dashed line, since colour already keys S.
        ax.plot([], [], color="gray", linestyle="--", linewidth=1.5, label=f"no cap ({baseline.cap})")
        ax.set_xlabel("Router active cap  (max resident requests per instance)")
        ax.set_ylabel("SCG(S)  [tokens / s]")
        ax.set_title(f"Staleness-Constrained Goodput vs active cap  ({policy})")
        ax.set_xticks(caps)
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Staleness budget")
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        path = out_dir / f"scg_vs_cap_{policy}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_overlay_per_s(
    runs: dict[str, list[RunSummary]],
    baselines: dict[str, RunSummary],
    s_values: list[int],
    out_dir: Path,
) -> list[Path]:
    """Draw one panel per staleness budget, with every policy on the same axes."""
    paths = []
    for s in s_values:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for policy, series in runs.items():
            color = POLICY_COLOR.get(policy, None)
            ax.plot(
                [r.cap for r in series],
                [r.scg(s) for r in series],
                marker="o",
                markersize=5,
                linewidth=2,
                color=color,
                label=policy,
            )
            ax.axhline(
                baselines[policy].scg(s),
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label=f"{policy}, no cap ({baselines[policy].cap})",
            )
        caps = [r.cap for r in next(iter(runs.values()))]
        ax.set_xlabel("Router active cap  (max resident requests per instance)")
        ax.set_ylabel(f"SCG({s})  [tokens / s]")
        ax.set_title(f"Policies at staleness budget S = {s}")
        ax.set_xticks(caps)
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Policy")
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        path = out_dir / f"scg_overlay_s{s}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_span_and_violation(
    runs: dict[str, list[RunSummary]],
    baselines: dict[str, RunSummary],
    configured_max_staleness: int,
    out_dir: Path,
) -> list[Path]:
    """
    Draw the wall-clock span and the highest staleness each policy consumed.

    The violation panel shows whether a policy exceeded the configured
    `job.max_staleness`, which would mean training on data the configuration excluded.
    """
    caps = [r.cap for r in next(iter(runs.values()))]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for policy, series in runs.items():
        color = POLICY_COLOR.get(policy, None)
        ax.plot(
            caps,
            [r.span for r in series],
            marker="o",
            markersize=5,
            linewidth=2,
            color=color,
            label=policy,
        )
        ax.axhline(
            baselines[policy].span,
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"{policy}, no cap ({baselines[policy].cap})",
        )
    ax.set_xlabel("Router active cap  (max resident requests per instance)")
    ax.set_ylabel("Simulated span  [s]")
    ax.set_title("Wall-clock cost of the same consumed work")
    ax.set_xticks(caps)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Policy")
    fig.tight_layout()
    span_path = out_dir / "span_vs_cap.png"
    fig.savefig(span_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for policy, series in runs.items():
        ax.plot(
            caps,
            [r.max_staleness_seen for r in series],
            marker="o",
            markersize=6,
            linewidth=2,
            color=POLICY_COLOR.get(policy, None),
            label=policy,
        )
    ax.axhline(
        configured_max_staleness,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"configured bound ({configured_max_staleness})",
    )
    ax.set_xlabel("Router active cap  (max resident requests per instance)")
    ax.set_ylabel("Highest staleness actually consumed")
    ax.set_title("Does the policy honour its own staleness bound?")
    ax.set_xticks(caps)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    viol_path = out_dir / "staleness_violation.png"
    fig.savefig(viol_path, dpi=150)
    plt.close(fig)

    return [span_path, viol_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--caps", type=int, nargs="+", required=True)
    parser.add_argument("--baseline-cap", type=int, required=True)
    parser.add_argument("--max-staleness", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    runs: dict[str, list[RunSummary]] = {}
    for policy in args.policies:
        runs[policy] = [
            load_run(args.run_base / policy / f"cap{cap}", policy, cap) for cap in args.caps
        ]
    baselines = {
        policy: load_run(args.run_base / policy / f"cap{args.baseline_cap}", policy, args.baseline_cap)
        for policy in args.policies
    }
    every_run = [r for series in runs.values() for r in series] + list(baselines.values())
    check_comparable(every_run)

    # The axis spans every S any run reached, so an overshoot is visible rather than cropped.
    highest = max(r.max_staleness_seen for r in every_run)
    s_values = list(range(max(args.max_staleness, highest) + 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = plot_per_policy(runs, baselines, s_values, args.out_dir)
    paths += plot_overlay_per_s(runs, baselines, s_values, args.out_dir)
    paths += plot_span_and_violation(runs, baselines, args.max_staleness, args.out_dir)

    header = "  cap  " + "".join(f"{p:>22}" for p in args.policies)
    print(header)
    print("       " + "".join(f"{'span   S=0    S=1    maxS':>22}" for _ in args.policies))
    rows = [(cap, [runs[p][i] for p in args.policies]) for i, cap in enumerate(args.caps)]
    rows.append((args.baseline_cap, [baselines[p] for p in args.policies]))
    for cap, cells_runs in rows:
        cells = "".join(
            f"{r.span:>7.0f}{r.scg(0):>7.0f}{r.scg(1):>7.0f}{r.max_staleness_seen:>5d}"
            for r in cells_runs
        )
        print(f"  {cap:<5}{cells}")
    print()
    n = next(iter(runs.values()))[0].n_consumed
    print(f"  All runs consumed {n} trajectories.")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
