"""Standalone CLI: quantify the error between a simulation and a real run.

Two figures side by side show that a simulation is wrong but not by how much or
where, and eyeballing a KV curve cannot distinguish a cost-model error from a
workload mismatch. This reports the handful of scalars that localise the
disagreement, so a change can be judged by whether a number moved.

Both sides are read through `JsonlStore`, using the streams the simulator
writes and `dryrun.telemetry.psrl_log` reproduces, so no plotting is involved.

Metrics, grouped by what they would implicate:

- **Training phases** (`train_time`, `recompute_time`, `sync_time` per step):
  the training cost model and the recompute fraction.
- **Rollout span and step cadence**: the rollout cost model.
- **Occupancy and KV** (peak and mean running requests, peak and mean KV
  utilisation): admission control, the KV denominator, and the workload.
- **Idle fraction**: the share of wall clock with an instance at zero running
  requests. Time-weighted, not sample-counted, so it reflects the loop rather
  than the two sides' differing logging cadences.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .._logging import setup_logging
from ..telemetry.store import JsonlStore


@dataclass
class Series:
    """One run's comparable summary, whatever its provenance."""

    label: str
    n_instances: int
    span_s: float
    n_train_steps: int
    mean_train_s: float
    mean_recompute_s: float
    mean_sync_s: float
    peak_running: float
    mean_running_busy: float
    peak_kv: float
    mean_kv_busy: float
    idle_fraction: float

    @classmethod
    def from_store(cls, store: JsonlStore, label: str) -> Series:
        instance_ids = sorted(
            int(name.split("/", 1)[1])
            for name in store.list_streams()
            if name.startswith("instance/") and name.split("/", 1)[1].isdigit()
        )
        if not instance_ids:
            raise ValueError(f"{label}: no instance streams found.")

        train = sorted(store.iter("train"), key=lambda r: r["start"])

        def phase(key: str) -> float:
            values = [float(r.get(key, 0.0) or 0.0) for r in train]
            return sum(values) / len(values) if values else 0.0

        # Every occupancy average here is weighted by how long a sample was in
        # effect, never by sample count. The two sides log on different triggers:
        # a real run samples at a fixed 1 Hz while the simulator writes only on a
        # committed state change, so an idle stretch produces one simulated
        # record against a hundred real ones. Counting records measured the
        # logging cadence rather than the behaviour: it reported 21% idle against
        # 59% for two runs whose instances both drained to exactly zero for the
        # whole of every training block.
        peak_running = 0.0
        peak_kv = 0.0
        busy_time = 0.0
        running_integral = 0.0
        kv_integral = 0.0
        idle_time = 0.0
        span = 0.0
        for iid in instance_ids:
            rows = list(store.iter(f"instance/{iid}"))
            if not rows:
                continue
            span = max(span, rows[-1]["t"])
            for current, following in zip(rows, rows[1:]):
                width = following["t"] - current["t"]
                n_running = float(current["num_running_reqs"])
                usage = float(current.get("kv_cache_usage", 0.0))
                peak_running = max(peak_running, n_running)
                peak_kv = max(peak_kv, usage)
                if width <= 0:
                    continue
                if n_running > 0:
                    busy_time += width
                    running_integral += n_running * width
                    kv_integral += usage * width
                else:
                    idle_time += width
            last = rows[-1]
            peak_running = max(peak_running, float(last["num_running_reqs"]))
            peak_kv = max(peak_kv, float(last.get("kv_cache_usage", 0.0)))

        return cls(
            label=label,
            n_instances=len(instance_ids),
            span_s=span,
            n_train_steps=len(train),
            mean_train_s=phase("train_time"),
            mean_recompute_s=phase("recompute_time"),
            mean_sync_s=phase("sync_time"),
            peak_running=peak_running,
            mean_running_busy=running_integral / busy_time if busy_time else 0.0,
            peak_kv=peak_kv,
            mean_kv_busy=kv_integral / busy_time if busy_time else 0.0,
            idle_fraction=idle_time / (idle_time + busy_time) if (idle_time + busy_time) else 0.0,
        )


_ROWS = (
    ("instances", "n_instances", "{:.0f}", False),
    ("span (s)", "span_s", "{:.1f}", True),
    ("train steps", "n_train_steps", "{:.0f}", False),
    ("train / step (s)", "mean_train_s", "{:.2f}", True),
    ("recompute / step (s)", "mean_recompute_s", "{:.2f}", True),
    ("sync / step (s)", "mean_sync_s", "{:.2f}", True),
    ("peak running reqs", "peak_running", "{:.0f}", True),
    ("mean running (busy)", "mean_running_busy", "{:.1f}", True),
    ("peak KV util", "peak_kv", "{:.3f}", True),
    ("mean KV util (busy)", "mean_kv_busy", "{:.3f}", True),
    ("idle fraction", "idle_fraction", "{:.3f}", False),
)


def render(simulated: Series, real: Series) -> str:
    """Format the two summaries as a table with a relative error column."""
    lines = [
        f"{'metric':<22} {'simulated':>12} {'real':>12} {'error':>10}",
        "-" * 60,
    ]
    for title, attribute, fmt, relative in _ROWS:
        sim_value = float(getattr(simulated, attribute))
        real_value = float(getattr(real, attribute))
        if relative and real_value:
            error = f"{(sim_value - real_value) / real_value * 100:+.1f}%"
        elif real_value or sim_value:
            error = f"{sim_value - real_value:+.3f}"
        else:
            error = "-"
        lines.append(
            f"{title:<22} {fmt.format(sim_value):>12} {fmt.format(real_value):>12} {error:>10}"
        )
    return "\n".join(lines)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="dryrun-compare",
        description="Quantify where a simulation disagrees with a real run.",
    )
    parser.add_argument("simulated", help="A simulation run directory (holding logs/).")
    parser.add_argument("real", help="A converted PSRL run directory (see dryrun-psrl-plot).")
    parser.add_argument("--logs-subdir", default="logs")
    args = parser.parse_args()

    stores = {}
    for role, root in (("simulated", args.simulated), ("real", args.real)):
        logs = Path(root) / args.logs_subdir
        if not logs.is_dir():
            raise SystemExit(f"No logs directory at {logs!s}.")
        stores[role] = JsonlStore(logs)

    simulated = Series.from_store(stores["simulated"], "simulated")
    real = Series.from_store(stores["real"], "real")
    print(render(simulated, real))


if __name__ == "__main__":
    main()
