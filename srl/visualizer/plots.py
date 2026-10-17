"""Offline matplotlib visualization, driven entirely by a `JsonlStore`.

`SimPlotter` never touches `SimResult` or holds simulation history itself:
every plot streams the records it needs straight out of the store's JSONL
files. This means visualization can run standalone against a previous run's
logs (see `srl.cli.plot`) without re-simulating, and adding a new plot is
just a new `plot_*` method plus `store.iter("some_stream")`: no changes to
the simulator or the store are required.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ..telemetry.store import JsonlStore  # noqa: E402

_INSTANCE_PREFIX = "instance/"

# The training block, in the order the phases actually run. `Simulator._train`
# only sums the three durations and never orders them, so the order here is the
# one a real PSRL run exhibits: log-prob recomputation, then the optimizer step,
# then the weight push. Lightest to darkest, so the block reads left to right.
_TRAIN_PHASES = (
    ("recompute_time", "recompute", "lightsteelblue"),
    ("train_time", "train", "steelblue"),
    ("sync_time", "sync", "midnightblue"),
)

# Applied only when a fleet is heterogeneous. Darker with higher degree, so the
# figure reads as a shape without checking the text.
_TP_COLOURS = {
    1: "#1f77b4",
    2: "#d62728",
    4: "#7f2704",
    8: "#3f007d",
}


class SimPlotter:
    """Generates offline plots by reading a `JsonlStore`'s log streams.

    Args:
        store: The store holding the run's telemetry streams.
        output_dir: Directory the PNGs are written to.
        time_label: X-axis label. The default suits a simulation; a store
            converted from a real run should say so, since its axis is
            measured wall-clock rather than simulated time.
    """

    def __init__(
        self,
        store: JsonlStore,
        output_dir: str | Path = "outputs/plots",
        time_label: str = "Simulation Time (s)",
    ):
        self.store = store
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.time_label = time_label

    def _instance_ids(self) -> list[int]:
        ids = []
        for name in self.store.list_streams():
            if not name.startswith(_INSTANCE_PREFIX):
                continue
            try:
                ids.append(int(name[len(_INSTANCE_PREFIX) :]))
            except ValueError:
                continue
        return sorted(set(ids))

    def _train_intervals(self) -> list[tuple[float, float]]:
        rows = list(self.store.iter("train"))
        rows.sort(key=lambda r: r["start"])
        return [(r["start"], r["end"]) for r in rows]

    def _train_start_events(self) -> list[tuple[float, int]]:
        """Return `(start_time, published_version)` for each training step."""
        rows = list(self.store.iter("train"))
        rows.sort(key=lambda r: r["start"])
        return [
            (float(row["start"]), int(row.get("version", index + 1)))
            for index, row in enumerate(rows)
        ]

    def _train_phase_spans(self) -> list[list[tuple[float, float, str, str]]]:
        """Split each training block into its `(start, end, label, colour)` phases.

        A phase with zero duration yields no span, which is what keeps an
        unmodelled cost visible as an absence rather than silently widening its
        neighbours. Any time in the block not accounted for by the three named
        phases is attributed to `train`, so the drawn band always covers the
        whole block.
        """
        rows = list(self.store.iter("train"))
        rows.sort(key=lambda r: r["start"])

        blocks = []
        for row in rows:
            durations = {key: float(row.get(key, 0.0) or 0.0) for key, _, _ in _TRAIN_PHASES}
            unattributed = (row["end"] - row["start"]) - sum(durations.values())
            durations["train_time"] += unattributed

            spans, cursor = [], row["start"]
            for key, label, colour in _TRAIN_PHASES:
                width = durations[key]
                if width <= 0.0:
                    continue
                spans.append((cursor, cursor + width, label, colour))
                cursor += width
            blocks.append(spans)
        return blocks

    def _gateway_rows(self) -> list[dict]:
        return list(self.store.iter("gateway"))

    def _consume_rows(self) -> list[dict]:
        return [r for r in self.store.iter("request") if r.get("event") == "consume"]

    def _empty_placeholder(self, path: Path, title: str) -> Path:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, title, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_instance_load(self) -> Path:
        """Train phase band on top, one running/total/KV subplot per instance below.

        The same figure serves a simulation and a real PSRL run: both write the
        `instance/*` and `train` streams this reads, so there is one drawing
        implementation rather than two that drift apart. A real run additionally
        writes `gateway`, which adds a router queue-depth row above the band.
        """
        path = self.output_dir / "instance_load.png"
        instance_ids = self._instance_ids()
        if not instance_ids:
            return self._empty_placeholder(path, "No instance telemetry logged")

        intervals = self._train_intervals()
        train_start_events = self._train_start_events()
        phase_blocks = self._train_phase_spans()
        gateway_rows = self._gateway_rows()

        n = len(instance_ids)
        heights = ([0.5] if gateway_rows else []) + [0.35] + [1] * n
        fig, axes = plt.subplots(
            len(heights),
            1,
            figsize=(12, 2 + 2.6 * n + (1.2 if gateway_rows else 0.0)),
            sharex=True,
            gridspec_kw={"height_ratios": heights},
        )
        axes = list(axes)
        gateway_ax = axes.pop(0) if gateway_rows else None
        train_ax = axes[0]
        inst_axes = axes[1:]

        # Light shading spans the whole block on every row, so the training
        # block still reads as one region on the subplots below even though the
        # band itself is split into phases.
        for ax in [*([gateway_ax] if gateway_ax is not None else []), *axes]:
            for start, end in intervals:
                ax.axvspan(start, end, color="navajowhite", alpha=0.35, zorder=0)

        if gateway_ax is not None:
            ts = [r["t"] for r in gateway_rows]
            queue = [r["queue_len"] for r in gateway_rows]
            gateway_ax.step(ts, queue, where="post", color="darkorange", zorder=2)
            gateway_ax.fill_between(ts, queue, step="post", color="darkorange", alpha=0.25, zorder=1)
            gateway_ax.set_ylabel("router\nqueue")
            gateway_ax.grid(True, alpha=0.3)

        drawn_phases: dict[str, str] = {}
        for spans in phase_blocks:
            for start, end, label, colour in spans:
                train_ax.axvspan(start, end, color=colour, alpha=0.9, zorder=1)
                drawn_phases[label] = colour
        train_ax.set_yticks([])
        train_ax.set_ylabel("train")

        # Shading shows duration; these markers identify the exact event that
        # starts each training step. Labels live on one row only to keep the
        # instance panels readable, while the vertical line crosses every row.
        marker_axes = [*([gateway_ax] if gateway_ax is not None else []), *axes]
        for start, _ in train_start_events:
            for ax in marker_axes:
                ax.axvline(
                    start,
                    color="saddlebrown",
                    linestyle=":",
                    linewidth=0.7,
                    alpha=0.55,
                    zorder=3,
                )
        label_ax = gateway_ax if gateway_ax is not None else train_ax
        for start, _ in train_start_events:
            label_ax.text(
                start,
                0.96,
                f"{start:.1f}s",
                rotation=45,
                rotation_mode="anchor",
                ha="right",
                va="top",
                fontsize=6,
                color="saddlebrown",
                transform=label_ax.get_xaxis_transform(),
                zorder=4,
            )

        # The phase key goes on the figure rather than inside the band, whose
        # full width is occupied by the blocks it would otherwise cover.
        fig.suptitle("Instance Load")
        if drawn_phases:
            fig.legend(
                handles=[Patch(facecolor=c, label=name) for name, c in drawn_phases.items()],
                loc="upper right",
                bbox_to_anchor=(0.99, 0.995),
                ncol=len(drawn_phases),
                fontsize=8,
                frameon=False,
            )

        # Two identical occupancy curves mean different things if one instance
        # holds twice the GPUs.
        tp_by_instance = self._tp_by_instance(instance_ids)
        distinct_tp = {tp for tp in tp_by_instance.values() if tp is not None}
        heterogeneous = len(distinct_tp) > 1

        for ax, iid in zip(inst_axes, instance_ids):
            rows = list(self.store.iter(f"instance/{iid}"))
            if rows:
                ts = [r["t"] for r in rows]
                running = [r["num_running_reqs"] for r in rows]
                total = [r["num_running_reqs"] + r["num_waiting_reqs"] for r in rows]
                kv = [r.get("kv_cache_usage", 0.0) for r in rows]

                ax.step(ts, total, where="post", color="darkorange", label="request_num", zorder=2)
                ax.step(ts, running, where="post", color="seagreen", label="running_request_num", zorder=2)
                ax.set_ylabel("# requests")
                ax.legend(loc="upper left", fontsize=8)

                # KV is interpolated, not stepped: a committed decode segment
                # covers many steps with no event in between, and every running
                # request gains one token per step, so occupancy climbs linearly
                # across the gap. Stepping it would hold the segment's opening
                # value flat and understate KV for the whole segment. Genuine
                # discontinuities (abort on weight update) are logged as two
                # samples at the same `t`, so they still render vertical.
                kv_ax = ax.twinx()
                kv_ax.fill_between(ts, kv, color="lightskyblue", alpha=0.4, zorder=1)
                kv_ax.set_ylim(0, 1)
                kv_ax.set_ylabel("KV util")
            tp = tp_by_instance.get(iid)
            if tp is None:
                ax.set_title(f"instance {iid}", fontsize=10)
            else:
                # Coloured only when heterogeneous, so the homogeneous case does
                # not look busier than it is.
                ax.set_title(
                    f"instance {iid}  —  TP{tp} ({tp} GPU{'s' if tp > 1 else ''})",
                    fontsize=10,
                    color=_TP_COLOURS.get(tp, "black") if heterogeneous else "black",
                    fontweight="bold" if heterogeneous else "normal",
                )
            ax.grid(True, alpha=0.3)

        if heterogeneous:
            # The per-row titles say what each instance is. This says how the
            # GPUs were split overall.
            shape = "  ".join(
                f"{sum(1 for v in tp_by_instance.values() if v == tp)}xTP{tp}"
                for tp in sorted(distinct_tp)
            )
            total_gpus = sum(tp for tp in tp_by_instance.values() if tp is not None)
            fig.suptitle(
                f"Instance Load — heterogeneous fleet: {shape}  ({total_gpus} GPUs)"
            )

        inst_axes[-1].set_xlabel(self.time_label)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _tp_by_instance(self, instance_ids: Sequence[int]) -> dict[int, int | None]:
        """Return each instance's tensor-parallel degree, or None when unlogged.

        Reads the first sample of each stream: the degree is written on every
        sample so a re-shaped fleet is representable, but the opening value is
        what labels the row. None rather than a default of 1, so a run recorded
        before the field existed is drawn without a degree instead of being
        labelled with a wrong one.
        """
        degrees: dict[int, int | None] = {}
        for iid in instance_ids:
            degrees[iid] = None
            for row in self.store.iter(f"instance/{iid}"):
                tp = row.get("tp")
                if tp is not None:
                    degrees[iid] = int(tp)
                break
        return degrees

    def plot_rollout_step(self) -> Path:
        """Per-instance scatter of engine step wall-clock cost: prefill dt vs. decode dt."""
        path = self.output_dir / "rollout_step.png"
        rows = list(self.store.iter("rollout_step"))
        instance_ids = self._instance_ids()
        if not rows or not instance_ids:
            return self._empty_placeholder(path, "No rollout_step telemetry logged")

        intervals = self._train_intervals()
        by_instance: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            by_instance[r["instance_id"]].append(r)

        n = len(instance_ids)
        fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, iid in zip(axes, instance_ids):
            for start, end in intervals:
                ax.axvspan(start, end, color="navajowhite", alpha=0.35, zorder=0)
            inst_rows = by_instance.get(iid, [])
            mixed = [r for r in inst_rows if r["phase"] == "mixed"]
            decode = [r for r in inst_rows if r["phase"] == "decode"]

            ax.scatter(
                [r["t"] for r in mixed], [r["dt"] for r in mixed],
                s=8, alpha=0.6, color="crimson", label="engine step dt", zorder=2,
            )
            ax.scatter(
                [r["t"] for r in decode], [r["dt"] for r in decode],
                s=8, alpha=0.6, color="steelblue", label="decode segment dt", zorder=2,
            )
            ax.set_ylabel("step dt (s)")
            ax.set_title(f"instance {iid}", fontsize=10)
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel(self.time_label)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_staleness_distribution(self) -> Path:
        """Grouped integer bars: true vs reported staleness of consumed requests.

        True = consume_version - v_min (oldest weight that produced any token).
        Reported = consume_version - v_max (newest weight that produced any token).
        They differ only when a request's tokens span multiple weight versions
        (partial rollout that actually kept generating after an abort).
        """
        path = self.output_dir / "staleness_distribution.png"
        consume_rows = self._consume_rows()
        if not consume_rows:
            return self._empty_placeholder(path, "No consumed requests logged")

        true_vals = [
            r["staleness_true"] if r.get("staleness_true") is not None else r["consume_version"] - r["v_min"]
            for r in consume_rows
        ]
        reported_vals = [
            r["staleness_reported"]
            if r.get("staleness_reported") is not None
            else r["consume_version"] - r["v_max"]
            for r in consume_rows
        ]

        lo = min(min(true_vals), min(reported_vals))
        hi = max(max(true_vals), max(reported_vals))
        xs = list(range(lo, hi + 1))
        true_counts = [true_vals.count(x) for x in xs]
        reported_counts = [reported_vals.count(x) for x in xs]

        fig, ax = plt.subplots(figsize=(8, 5))
        width = 0.35
        ax.bar([x - width / 2 for x in xs], true_counts, width, color="coral", label="True = consume_v − v_min")
        ax.bar(
            [x + width / 2 for x in xs],
            reported_counts,
            width,
            color="steelblue",
            label="Reported = consume_v − v_max",
        )
        ax.set_xticks(xs)
        ax.set_xlabel("Staleness (integer versions)")
        ax.set_ylabel("Count")
        ax.set_title("Staleness of consumed requests")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_request_timeline(self) -> Path:
        path = self.output_dir / "request_timeline.png"
        consume_rows = self._consume_rows()
        if not consume_rows:
            return self._empty_placeholder(path, "No consumed requests logged")

        fig, ax = plt.subplots(figsize=(12, 6))
        admit_times = [r["admit_time"] for r in consume_rows]
        complete_times = [
            r["complete_time"] if r["complete_time"] is not None else r["admit_time"] for r in consume_rows
        ]
        durations = [c - a for a, c in zip(admit_times, complete_times)]
        lengths = [r["generated"] for r in consume_rows]

        sc = ax.scatter(admit_times, durations, c=lengths, s=10, alpha=0.6, cmap="viridis")
        ax.set_xlabel("Admission Time (s)")
        ax.set_ylabel("Generation Duration (s)")
        ax.set_title("Request Durations (color = output length)")
        fig.colorbar(sc, ax=ax, label="Output Tokens")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_throughput(self, window_size: float = 5.0) -> Path:
        path = self.output_dir / "throughput.png"
        consume_rows = self._consume_rows()
        times = sorted(r["complete_time"] for r in consume_rows if r.get("complete_time") is not None)
        if not times:
            return self._empty_placeholder(path, "No completed requests logged")

        fig, ax = plt.subplots(figsize=(10, 5))
        t_max = max(times)
        bins = np.arange(0, t_max + window_size, window_size)
        counts, edges = np.histogram(times, bins=bins)
        rates = counts / window_size

        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, rates, "b-", linewidth=1.5)
        ax.fill_between(centers, rates, alpha=0.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Completions / {window_size}s")
        ax.set_title("Throughput Over Time")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    @staticmethod
    def _scg_curve_data(
        staleness_values: list[int],
        span: float,
        token_values: list[int] | None = None,
    ) -> tuple[list[int], list[float]]:
        """Compute the SCG step curve from per-trajectory staleness values.

        SCG(S) = sum(tokens where staleness_true <= S) / span, for S in
        [0, max(staleness_values)].

        Factored out so tests can assert on exact numeric values without parsing a PNG.

        Args:
            staleness_values: One integer staleness per consumed trajectory.
            span: Total simulation time in seconds (the rate denominator).
            token_values: Generated tokens per consumed trajectory, positionally
                aligned with `staleness_values`. None falls back to counting
                trajectories, which is what a run logged before `generated` was
                recorded can support.

        Returns:
            xs: S values 0..max inclusive.
            ys: Corresponding SCG(S) rates, in tokens/s (or trajectories/s when
                `token_values` is None).
        """
        if not staleness_values or span <= 0:
            return [], []
        max_s = max(staleness_values)
        xs = list(range(max_s + 1))
        if token_values is None:
            ys = [sum(1 for v in staleness_values if v <= s) / span for s in xs]
        else:
            ys = [
                sum(t for v, t in zip(staleness_values, token_values) if v <= s) / span
                for s in xs
            ]
        return xs, ys

    def plot_scg_curve(self) -> Path:
        """Plot SCG(S): tokens from trajectories with staleness ≤ S, per second.

        SCG(S) is a non-decreasing step function of the integer staleness budget S.
        The denominator is ``max(complete_time)`` over all consume rows, the same
        time axis ``plot_throughput`` uses, so the rates are directly comparable
        across the two plots -- and the curve's last point equals the run's
        throughput by construction, since no trajectory is excluded there.

        Returns:
            Path to the saved PNG.
        """
        path = self.output_dir / "scg_curve.png"
        rows = self._consume_rows()

        # Rows from runs predating the staleness_true field are skipped rather
        # than falling back to v_min arithmetic, so the curve only reflects data
        # we can trust. If none survive the filter we fall through to placeholder.
        usable = [r for r in rows if r.get("staleness_true") is not None]
        staleness_values = [int(r["staleness_true"]) for r in usable]
        if not staleness_values:
            return self._empty_placeholder(path, "No staleness data to plot SCG curve")

        # Tokens are the goodput unit: a dispersed workload's trajectories differ
        # in length by more than an order of magnitude, so counting them weights a
        # short response the same as one 40x longer. A run logged before
        # `generated` existed falls back to the count form rather than plotting
        # zeros.
        token_values: list[int] | None = [int(r.get("generated", 0)) for r in usable]
        if token_values is not None and not any(token_values):
            token_values = None

        complete_times = [
            float(r["complete_time"])
            for r in rows
            if r.get("complete_time") is not None
        ]
        # Fall back to the event timestamp when complete_time is absent everywhere.
        span = max(complete_times) if complete_times else max(float(r["t"]) for r in rows)
        if span <= 0:
            span = 1.0  # guard: avoid division by zero on pathological input

        xs, ys = self._scg_curve_data(staleness_values, span, token_values)

        fig, ax = plt.subplots(figsize=(8, 5))
        # step(where='post') is the correct representation: SCG(S) holds constant
        # from integer S until the next integer, which is what 'post' draws.
        # Scatter markers make each defined S value explicit.
        ax.step(xs, ys, where="post", color="steelblue", linewidth=2)
        ax.scatter(xs, ys, color="steelblue", s=40, zorder=3)
        ax.set_xlabel("Staleness budget S (weight updates)")
        unit = "trajectories / s" if token_values is None else "tokens / s"
        ax.set_ylabel(f"SCG(S)  [{unit}]")
        ax.set_title("Staleness-Constrained Goodput")
        ax.set_xticks(xs)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_all(self) -> list[Path]:
        return [
            self.plot_instance_load(),
            self.plot_rollout_step(),
            self.plot_staleness_distribution(),
            self.plot_request_timeline(),
            self.plot_throughput(),
            self.plot_scg_curve(),
        ]
