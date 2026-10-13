"""
Plot SCG(S) against the router active cap, one figure per policy plus overlays.

Reads the telemetry of finished `policy_cap_sweep.sh` runs and writes PNGs only. SCG
is recomputed from the `consume` rows exactly as `SimResult.scg` defines it, because
`SimResult` is not persisted. The span is `max(t)` over all rows, matching
`SimResult.sim_time`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# One colour per policy, shared by every figure so the overlays and panels agree.
POLICY_COLOR = {"psrl": "#B23A48", "verl": "#2E6F95", "roll": "#4C9A2A"}


@dataclass(frozen=True)
class RunSummary:
    """Hold one finished simulation, reduced to what the SCG curves need."""

    policy: str
    cap: int
    span: float
    n_consumed: int
    tokens_by_staleness: dict[int, int]
    max_staleness_seen: int

    @property
    def throughput(self) -> float:
        return sum(self.tokens_by_staleness.values()) / self.span

    def scg(self, s: int) -> float:
        """
        Return SCG(S) in tokens per second.

        Defined for every S. SCG saturates at `throughput` once nothing is excluded,
        so an S above the run's own maximum equals the value below it.
        """
        tokens = sum(n for staleness, n in self.tokens_by_staleness.items() if staleness <= s)
        return tokens / self.span


def load_run(run_dir: Path, policy: str, cap: int) -> RunSummary:
    """
    Reduce one run's request telemetry to a `RunSummary`.

    Raises:
        FileNotFoundError: If the run has no request stream.
        ValueError: If it has no consume rows or a non-positive span.
    """
    stream = run_dir / "logs" / "request.jsonl"
    if not stream.is_file():
        raise FileNotFoundError(f"No telemetry at {stream}.")

    span = 0.0
    tokens: dict[int, int] = {}
    n_consumed = 0
    with stream.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            span = max(span, float(row.get("t", 0.0)))
            if row.get("event") != "consume" or row.get("staleness_true") is None:
                continue
            n_consumed += 1
            s = int(row["staleness_true"])
            tokens[s] = tokens.get(s, 0) + int(row.get("generated", 0))

    if not n_consumed:
        raise ValueError(f"{stream} holds no consume rows.")
    if span <= 0.0:
        raise ValueError(f"{stream} has a non-positive span.")
    return RunSummary(
        policy=policy,
        cap=cap,
        span=span,
        n_consumed=n_consumed,
        tokens_by_staleness=tokens,
        max_staleness_seen=max(tokens),
    )


def check_comparable(runs: list[RunSummary]) -> None:
    """
    Refuse to plot runs that did not consume the same amount of work.

    SCG is a rate, so a run that ended early can score arbitrarily well.

    Raises:
        ValueError: If the consumed trajectory counts disagree.
    """
    counts = {r.n_consumed for r in runs}
    if len(counts) > 1:
        detail = ", ".join(f"{r.policy}/cap{r.cap}: {r.n_consumed}" for r in runs)
        raise ValueError(f"Runs consumed different trajectory counts: {detail}.")


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
