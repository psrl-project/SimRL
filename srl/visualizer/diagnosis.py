"""Figures for a post-hoc diagnosis.

Every figure is derived from `srl/diagnosis` outputs, which are themselves
derived from the JSONL streams, so a figure adds no information and can be
regenerated or deleted freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..diagnosis.critical_path import StepWait  # noqa: E402
from ..diagnosis.fleet import IdleWindow, InstanceImbalance  # noqa: E402
from ..diagnosis.timeline import RequestTimeline  # noqa: E402

_PHASE_COLORS = {
    "decode": "#4878d0",
    "prefill": "#6acc64",
    "reprefill": "#d65f5f",
    "engine_wait": "#ee854a",
    "router_queue": "#956cb4",
    "weight_load": "#8c613c",
}


@dataclass
class DiagnosisPlotter:
    """Render the four diagnosis figures into `out_dir`."""

    store: object
    timelines: dict[int, RequestTimeline]
    steps_: list[StepWait]
    windows: list[IdleWindow]
    imbalance: list[InstanceImbalance]
    out_dir: Path
    figsize: tuple[float, float] = (12.0, 6.0)
    _written: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name: str) -> Path:
        path = self.out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def plot_critical_path(self) -> Path:
        """One stacked bar per training step: compute plus idle split by binding phase."""
        fig, ax = plt.subplots(figsize=self.figsize)
        versions = [step.version for step in self.steps_]
        ax.bar(versions, [step.train_seconds for step in self.steps_], label="train", color="#4878d0")
        bottoms = [step.train_seconds for step in self.steps_]
        for kind, color in _PHASE_COLORS.items():
            heights = []
            for step in self.steps_:
                line = self.timelines.get(step.binding_rid)
                share = line.seconds_in(kind) if line is not None else 0.0
                total = sum(line.seconds_in(k) for k in _PHASE_COLORS) if line is not None else 0.0
                heights.append(step.idle * (share / total) if total > 0 else 0.0)
            ax.bar(versions, heights, bottom=bottoms, label=f"idle: {kind}", color=color)
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        ax.set_xlabel("training version")
        ax.set_ylabel("seconds")
        ax.set_title("Per-step compute and idle, idle split by the binding request's phases")
        ax.legend(fontsize=8, ncol=2)
        return self._save(fig, "critical_path.png")

    def plot_binding_requests(self) -> Path:
        """A gantt of each step's binding request and its group siblings."""
        fig, ax = plt.subplots(figsize=self.figsize)
        row = 0
        labels = []
        for step in self.steps_:
            if step.binding_group_id is None:
                continue
            siblings = [
                line for line in self.timelines.values() if line.group_id == step.binding_group_id
            ]
            for line in sorted(siblings, key=lambda item: item.rid):
                for phase in line.phases:
                    ax.barh(
                        row,
                        phase.duration,
                        left=phase.t0,
                        height=0.8,
                        color=_PHASE_COLORS.get(phase.kind, "#999999"),
                    )
                labels.append(f"v{step.version} rid{line.rid}")
                row += 1
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("simulated seconds")
        ax.set_title("Binding requests and their group siblings, by phase")
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in _PHASE_COLORS.values()]
        ax.legend(handles, list(_PHASE_COLORS), fontsize=8, ncol=3)
        return self._save(fig, "binding_requests.png")

    def plot_buffers(self) -> Path:
        """Occupied and reserved entries per buffer over time, with train starts marked."""
        fig, ax = plt.subplots(figsize=self.figsize)
        series: dict[int, list[tuple[float, int, int]]] = {}
        for rec in self.store.iter("buffer"):
            for view in rec["buffers"]:
                series.setdefault(view["buffer_id"], []).append(
                    (rec["t"], view["n_occupied"], view["n_reserved"])
                )
        if not series:
            ax.set_title("No buffer stream: the policy exposes no inventory")
        for buffer_id, points in sorted(series.items()):
            times = [point[0] for point in points]
            ax.step(times, [point[1] for point in points], where="post", label=f"buffer {buffer_id} occupied")
            ax.step(
                times,
                [point[2] for point in points],
                where="post",
                linestyle="--",
                label=f"buffer {buffer_id} reserved",
            )
        for step in self.steps_:
            ax.axvline(step.start, color="#333333", linewidth=0.6, alpha=0.5)
        ax.set_xlabel("simulated seconds")
        ax.set_ylabel("entries")
        ax.set_title("Buffer occupancy and reservations, vertical lines are train starts")
        ax.legend(fontsize=7, ncol=2)
        return self._save(fig, "buffers.png")

    def plot_idle_windows(self) -> Path:
        """Per-instance busy fraction and router queue depth inside each idle window."""
        fig, (top, bottom) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        versions = [window.version for window in self.windows]
        instance_ids = sorted({iid for window in self.windows for iid in window.busy_fraction})
        for instance_id in instance_ids:
            top.plot(
                versions,
                [window.busy_fraction.get(instance_id, 0.0) for window in self.windows],
                marker="o",
                label=f"instance {instance_id}",
            )
        top.set_ylabel("busy fraction")
        top.set_ylim(0.0, 1.05)
        top.legend(fontsize=8)
        top.set_title("Fleet state inside each training idle window")
        bottom.plot(
            versions,
            [window.mean_queue_depth for window in self.windows],
            marker="s",
            color="#956cb4",
        )
        bottom.set_xlabel("training version")
        bottom.set_ylabel("mean router queue depth")
        return self._save(fig, "idle_windows.png")

    def plot_all(self) -> list[Path]:
        """Render every figure, returning the paths written."""
        return [
            self.plot_critical_path(),
            self.plot_binding_requests(),
            self.plot_buffers(),
            self.plot_idle_windows(),
        ]
