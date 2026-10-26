"""Where an async run's wall time goes, and how little of it is unavoidable.

Two questions, both answered from a finished run rather than from a model. `cadence_report`
splits the span into training, trainer idle, and fleet idle, and reports the step-interval
series with its first two autocorrelations, which is what decides whether a run oscillates
with a period at all. `span_contention_free` removes every capacity limit and leaves only the
two constraints no scheduler can relax: a request cannot finish faster than it decodes alone,
and a training step cannot start before psrl's own reserve and occupy rules fill its buffer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from psrl.workers.ps.staleness import InventorySnapshot, forecast_ready_times


@dataclass(frozen=True, slots=True)
class CadenceReport:
    """One run's wall-time split and its step cadence."""

    span: float
    train_busy: float
    trainer_idle: float
    """Span the trainer spent waiting for a buffer, including the wait before the first step."""
    fleet_idle_frac: float
    """One minus the mean per-instance busy fraction over the span."""
    intervals: tuple[float, ...]
    """Gaps between consecutive training-step end times."""
    acf1: float
    acf2: float
    """Lag-1 and lag-2 autocorrelation of `intervals`, `nan` when the series cannot support one."""


def cadence_report(log_dir: Path, n_instances: int) -> CadenceReport:
    """
    Build the cadence report from one run's telemetry.

    Args:
        log_dir (Path): The run's `logs/` directory, holding `train.jsonl` and
            `rollout_step.jsonl`.
        n_instances (int): Instances the run used, the denominator of the busy fraction.

    Raises:
        ValueError: If `train.jsonl` holds no step, or `n_instances` is below 1.
    """
    if n_instances < 1:
        raise ValueError(f"n_instances must be at least 1, got {n_instances!r}.")
    steps = [json.loads(line) for line in (log_dir / "train.jsonl").read_text().splitlines() if line]
    if not steps:
        raise ValueError(f"{log_dir / 'train.jsonl'} holds no training step.")
    steps.sort(key=lambda row: row["start"])

    span = steps[-1]["end"]
    train_busy = sum(row["end"] - row["start"] for row in steps)
    idle = 0.0
    previous = 0.0
    for row in steps:
        idle += max(0.0, row["start"] - previous)
        previous = row["end"]

    busy: dict[int, float] = {}
    rollout_path = log_dir / "rollout_step.jsonl"
    if rollout_path.exists():
        with rollout_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                busy[row["instance_id"]] = busy.get(row["instance_id"], 0.0) + row["dt"]
    mean_busy = sum(busy.values()) / n_instances / span if span > 0 else 0.0

    ends = [row["end"] for row in steps]
    intervals = tuple(ends[i + 1] - ends[i] for i in range(len(ends) - 1))
    return CadenceReport(
        span=span,
        train_busy=train_busy,
        trainer_idle=idle,
        fleet_idle_frac=1.0 - mean_busy,
        intervals=intervals,
        acf1=_autocorrelation(intervals, 1),
        acf2=_autocorrelation(intervals, 2),
    )


def _autocorrelation(series: Sequence[float], lag: int) -> float:
    """Sample autocorrelation at `lag`, `nan` when the series is too short or has no variance."""
    n = len(series)
    if lag < 1 or n <= lag + 1:
        return math.nan
    mean = sum(series) / n
    variance = sum((x - mean) ** 2 for x in series)
    if variance <= 0.0:
        return math.nan
    covariance = sum((series[i] - mean) * (series[i + lag] - mean) for i in range(n - lag))
    return covariance / variance


def span_contention_free(
    groups: Sequence[tuple[int, Sequence[int]]],
    *,
    staleness: int,
    batch_size: int,
    n_versions: int,
    solo_latency: Callable[[int, int], float],
    train_latency: Callable[[Sequence[int]], float],
    weight_load: float = 0.0,
) -> float:
    """
    Lower-bound the span of `n_versions` steps when no request ever shares an instance.

    Groups are admitted in rounds, one round per training step. A round admits new groups,
    all labeled with its own step index, until the number in flight (admitted but not yet
    consumed) reaches the staleness window's full capacity, mirroring that psrl refuses a
    reservation once every buffer in `[version, version + staleness]` is full. This is what
    lets a straggler slip past: a wide round makes many groups simultaneously eligible for the
    oldest open buffer, so psrl's own reserve and occupy rules let the fastest among them fill
    it while the straggler is pushed into a later one. Each round's membership (and so its
    `train_latency` lengths) is then the earliest-finishing groups not yet claimed by an
    earlier round, in ascending ETA order. The bound ignores fleet capacity entirely: large
    headroom against it is necessary for a scheduler to win and never sufficient.

    Args:
        groups (Sequence[tuple[int, Sequence[int]]]): `(prompt_len, output_lens)` per group, in
            the order the workload stages them.
        staleness (int): psrl's staleness tolerance.
        batch_size (int): Groups consumed per training step.
        n_versions (int): Steps to bound.
        solo_latency (Callable[[int, int], float]): Seconds for one request of
            `(prompt_len, output_len)` alone on an instance.
        train_latency (Callable[[Sequence[int]], float]): Seconds to train on the given
            per-response sequence lengths.
        weight_load (float): Delay between a publish and the fleet resuming.

    Returns:
        float: The earliest the `n_versions`-th step could end.

    Raises:
        ValueError: If there are fewer groups than `n_versions * batch_size`, or a buffer can
            never fill from the groups admitted to it.
    """
    needed = n_versions * batch_size
    if len(groups) < needed:
        raise ValueError(f"The bound needs at least {needed} groups for {n_versions} steps, got {len(groups)}.")

    new_groups: list[tuple[int, float]] = []
    members: list[int] = []
    next_group = 0
    consumed: set[int] = set()
    train_end: float | None = None

    for version in range(n_versions):
        resume = 0.0 if train_end is None else train_end + weight_load
        window_capacity = (staleness + 1) * batch_size
        while next_group < len(groups) and len(new_groups) - len(consumed) < window_capacity:
            prompt_len, output_lens = groups[next_group]
            eta = resume + max(solo_latency(prompt_len, length) for length in output_lens)
            new_groups.append((version, eta))
            members.append(next_group)
            next_group += 1

        snapshot = InventorySnapshot(staleness=staleness, num_entries=batch_size, next_buffer_id=0, buffers=())
        ready = forecast_ready_times(snapshot, {}, new_groups)
        f_k = ready.get(version, math.inf)
        if not math.isfinite(f_k):
            raise ValueError(f"Buffer {version} can never fill from the groups admitted to it.")

        pending = sorted(
            (index for index, group_id in enumerate(members) if group_id not in consumed),
            key=lambda index: new_groups[index][1],
        )
        lengths: list[int] = []
        for index in pending[:batch_size]:
            group_id = members[index]
            prompt_len, output_lens = groups[group_id]
            lengths.extend(prompt_len + length for length in output_lens)
            consumed.add(group_id)

        start = max(f_k, resume)
        train_end = start + train_latency(lengths)

    return float(train_end)
