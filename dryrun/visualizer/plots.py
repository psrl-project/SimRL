"""Offline matplotlib visualization, driven entirely by a `JsonlStore`.

`SimPlotter` never touches `SimResult` or holds simulation history itself:
every plot streams the records it needs straight out of the store's JSONL
files. This means visualization can run standalone against a previous run's
logs (see `dryrun.cli.plot`) without re-simulating, and adding a new plot is
just a new `plot_*` method plus `store.iter("some_stream")` -- no changes to
the simulator or the store are required.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..telemetry.store import JsonlStore  # noqa: E402

_INSTANCE_PREFIX = "instance/"


class SimPlotter:
    """Generates offline plots by reading a `JsonlStore`'s log streams."""

    def __init__(self, store: JsonlStore, output_dir: str | Path = "outputs/plots"):
        self.store = store
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        """Train interval band on top, one running/total/KV subplot per instance below."""
        path = self.output_dir / "instance_load.png"
        instance_ids = self._instance_ids()
        if not instance_ids:
            return self._empty_placeholder(path, "No instance telemetry logged")

        intervals = self._train_intervals()
        n = len(instance_ids)
        fig, axes = plt.subplots(
            1 + n,
            1,
            figsize=(12, 2 + 2.6 * n),
            sharex=True,
            gridspec_kw={"height_ratios": [0.35] + [1] * n},
        )
        train_ax = axes[0]
        inst_axes = axes[1:]

        # Light shading on every row keeps train boundaries visible even on
        # the per-instance subplots below, since all axes share the x-axis.
        for ax in axes:
            for start, end in intervals:
                ax.axvspan(start, end, color="navajowhite", alpha=0.35, zorder=0)
        for start, end in intervals:
            train_ax.axvspan(start, end, color="steelblue", alpha=0.9, zorder=1)
        train_ax.set_yticks([])
        train_ax.set_ylabel("train")
        train_ax.set_title("Instance Load")

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
            ax.set_title(f"instance {iid}", fontsize=10)
            ax.grid(True, alpha=0.3)

        inst_axes[-1].set_xlabel("Simulation Time (s)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

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
            prefill = [r for r in inst_rows if r["phase"] == "prefill"]
            decode = [r for r in inst_rows if r["phase"] == "decode"]

            ax.scatter(
                [r["t"] for r in prefill], [r["dt"] for r in prefill],
                s=8, alpha=0.6, color="crimson", label="prefill step dt", zorder=2,
            )
            ax.scatter(
                [r["t"] for r in decode], [r["dt"] for r in decode],
                s=8, alpha=0.6, color="steelblue", label="decode segment dt", zorder=2,
            )
            ax.set_ylabel("step dt (s)")
            ax.set_title(f"instance {iid}", fontsize=10)
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Simulation Time (s)")
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

    def plot_all(self) -> list[Path]:
        return [
            self.plot_instance_load(),
            self.plot_rollout_step(),
            self.plot_staleness_distribution(),
            self.plot_request_timeline(),
            self.plot_throughput(),
        ]
