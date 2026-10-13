"""Predict when each pending training buffer becomes ready and finishes training.

`PipelineForecast` replays psrl's own reserve/occupy rules (`forecast_ready_times`) over predicted
group finish times to get each buffer's READY time `F_k`, then chains buffers through the training
loop to get each one's training-finish time, `E_k = max(F_k, E_{k-1}) + tau`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from psrl.workers.ps.inventory_forecast import InventorySnapshot, forecast_ready_times


@dataclass(frozen=True, slots=True)
class TrainState:
    """What the training loop is doing right now, as the pipeline forecast needs it."""

    training_end: float | None
    """Absolute time the current train plus sync plus recompute finishes and publishes.

    None when the trainer is idle, waiting for a buffer.
    """
    now: float
    cycle: float
    """Predicted train plus sync plus recompute duration of the next buffer."""
    weight_load: float
    """Delay between a publish and the rollout fleet resuming on the new weights."""
    reconfig: float = 0.0
    """Extra delay from a fleet shape switch, charged alongside `weight_load`."""


@dataclass(frozen=True, slots=True)
class Pipeline:
    """One forecast of the training pipeline, one entry per pending buffer, ascending id."""

    buffers: tuple[int, ...]
    ready: tuple[float, ...]
    """`F_k`, each buffer's predicted READY time."""
    train_end: tuple[float, ...]
    """`E_k`, each buffer's predicted training-finish time."""
    publish_times: tuple[float, ...]
    """When each buffer's training publishes and aborts rollout, before its weight load."""


def group_etas(
    member_finish: Mapping[int, float],
    members_of: Mapping[int, Sequence[int]],
    finished: Mapping[int, float],
) -> dict[int, float]:
    """
    Return each group's ETA as the latest of its members' finish times.

    Args:
        member_finish (Mapping[int, float]): Predicted finish time of a placed member, by request id.
        members_of (Mapping[int, Sequence[int]]): Group id to its member request ids.
        finished (Mapping[int, float]): Known completion time of an already-finished member, by
            request id.

    Returns:
        dict[int, float]: Group id to its ETA. A member missing from both maps makes the group `inf`.
    """
    etas: dict[int, float] = {}
    for group_id, members in members_of.items():
        worst = -math.inf
        for member in members:
            if member in member_finish:
                finish = member_finish[member]
            elif member in finished:
                finish = finished[member]
            else:
                finish = math.inf
            worst = max(worst, finish)
        etas[group_id] = worst
    return etas


class PipelineForecast:
    """Chain one inventory snapshot's buffers through the training pipeline."""

    def __init__(self, snapshot: InventorySnapshot, train: TrainState) -> None:
        self.snapshot = snapshot
        self.train = train

    def evaluate(
        self,
        group_eta: Mapping[int, float],
        new_groups: Sequence[tuple[int, float]] = (),
    ) -> Pipeline:
        """
        Predict the READY and training-finish time of every pending buffer.

        Args:
            group_eta (Mapping[int, float]): Predicted finish time of each reserved group, by
                prompt id. Passed straight to `forecast_ready_times`.
            new_groups (Sequence[tuple[int, float]]): Hypothetical admissions as `(version, eta)`,
                reserved in order as `reserve_data` would at the snapshot time.

        Returns:
            Pipeline: One entry per buffer the snapshot and `new_groups` touch, ascending id. A
                buffer whose `F_k` is `inf` yields `E_k = inf`, and every later buffer follows it.
        """
        ready = forecast_ready_times(self.snapshot, group_eta, new_groups)
        buffer_ids = sorted(ready)
        train = self.train
        # An in-progress cycle still owes its weight load before the next one can start training.
        e_prev = train.now if train.training_end is None else train.training_end + train.weight_load
        readies: list[float] = []
        publishes: list[float] = []
        ends: list[float] = []
        for buffer_id in buffer_ids:
            f_k = ready[buffer_id]
            start = max(f_k, e_prev)
            publish_k = start + train.cycle
            e_k = publish_k + train.weight_load + train.reconfig
            readies.append(f_k)
            publishes.append(publish_k)
            ends.append(e_k)
            e_prev = e_k
        return Pipeline(
            buffers=tuple(buffer_ids),
            ready=tuple(readies),
            train_end=tuple(ends),
            publish_times=tuple(publishes),
        )


def compare(a: Pipeline, b: Pipeline, eps: float) -> int:
    """
    Lexicographically compare two pipelines' `train_end` vectors with tolerance `eps`.

    Components are compared left to right, shorter vector first: if one pipeline opened fewer
    hypothetical buffers than the other, only their shared prefix is compared. The first pair
    differing by more than `eps` decides the result. Two infinite components of the same sign are
    equal and comparison continues to the next pair. This tolerance is not transitive: `a` may tie
    `b` and `b` tie `c` while `a` and `c` differ by up to `2 * eps`.

    Args:
        a (Pipeline): One pipeline.
        b (Pipeline): The other pipeline.
        eps (float): Absolute tolerance below which two components are considered equal.

    Returns:
        int: -1 if `a` is lexicographically smaller, 1 if `b` is, 0 if every compared component ties.
    """
    for x, y in zip(a.train_end, b.train_end):
        if math.isinf(x) or math.isinf(y):
            if x == y:
                continue
            return -1 if x < y else 1
        if abs(x - y) <= eps:
            continue
        return -1 if x < y else 1
    return 0
