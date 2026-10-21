"""What the fleet was doing while the trainer waited.

An idle window where every instance is busy means the fleet is saturated and
the straggler is simply long. A window where instances sit idle with an empty
router queue means the work does not exist yet, which no placement rule can
fix. The distinction decides whether a scheduling mechanism has anything to
work with, so it is measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .critical_path import StepWait
from .timeline import RequestTimeline


@dataclass(frozen=True)
class IdleWindow:
    """One stretch between training steps, with the fleet state inside it."""

    version: int
    t0: float
    t1: float
    busy_fraction: dict[int, float]
    mean_queue_depth: float
    binding_group_id: int | None
    binding_reserved_in_buffer: int | None
    binding_at_staleness_limit: bool | None
    n_occupied_later: int


@dataclass(frozen=True)
class InstanceImbalance:
    """Whole-run totals for one instance."""

    instance_id: int
    busy_seconds: float
    tokens: int
    last_active: float


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _rollout_segments(store) -> dict[int, list[tuple[float, float, int]]]:
    """Return `(t0, t1, n_tokens)` per instance. A record's `t` is its end."""
    out: dict[int, list[tuple[float, float, int]]] = {}
    for rec in store.iter("rollout_step"):
        end = rec["t"]
        out.setdefault(rec["instance_id"], []).append((end - rec["dt"], end, rec["n_tokens"]))
    for segments in out.values():
        segments.sort()
    return out


def _v_traj_of_group(store) -> dict[int, int]:
    """Return the rollout version each group was admitted at.

    Every member of a group is admitted with the same `v_traj`, so scanning
    all `admit` events in the `request` stream and letting the last one win
    per `group_id` is harmless, and avoids relying on `RequestTimeline`,
    which does not carry `v_traj`.
    """
    out: dict[int, int] = {}
    for rec in store.iter("request"):
        if rec["event"] == "admit":
            out[rec["group_id"]] = rec["v_traj"]
    return out


def idle_windows(
    store,
    steps_: list[StepWait],
    timelines: dict[int, RequestTimeline],
    staleness_horizon: int,
) -> list[IdleWindow]:
    """
    Describe the fleet inside every nonzero wait between training steps.

    Args:
        store (JsonlStore): A finished run's log store.
        steps_ (list[StepWait]): Training steps in version order.
        timelines (dict[int, RequestTimeline]): Reconstructed timelines, accepted for
            interface symmetry with the other diagnosis entry points but not read here,
            since the binding group's `v_traj` comes straight from the `request` stream.
        staleness_horizon (int): The policy's declared horizon, used to decide
            whether a reservation sits at its limit and cannot move later.

    Returns:
        list[IdleWindow]: One entry per step whose `idle` is positive.
    """
    segments = _rollout_segments(store)
    gateway = sorted((rec["t"], rec["queue_len"]) for rec in store.iter("gateway"))
    # Sort key excludes the per-record `buffers` payload: two snapshots can share a
    # timestamp, and a list of dicts has no ordering, so comparing it would raise.
    buffers = sorted(
        ((rec["t"], rec["buffers"]) for rec in store.iter("buffer")),
        key=lambda pair: pair[0],
    )
    v_traj_of_group = _v_traj_of_group(store)

    out: list[IdleWindow] = []
    previous_end = 0.0
    for step in steps_:
        t0, t1 = previous_end, step.start
        previous_end = step.end
        if t1 - t0 <= 1e-9:
            continue
        width = t1 - t0
        busy = {
            instance_id: sum(_overlap(s0, s1, t0, t1) for s0, s1, _ in runs) / width
            for instance_id, runs in segments.items()
        }
        depths = [depth for t, depth in gateway if t0 <= t <= t1]
        reserved_in = at_limit = None
        occupied_later = 0
        # Strict `<`: the binding group's own completion is what ends this window, and a
        # buffer snapshot at exactly `t1` already reflects that reserved-to-occupied move.
        state = [views for t, views in buffers if t < t1]
        if state and step.binding_group_id is not None:
            latest = state[-1]
            for view in latest:
                if step.binding_group_id in view["reserved_group_ids"]:
                    reserved_in = view["buffer_id"]
                # The batch this step trains on was tagged `step.version - 1`, so a
                # buffer at or past `step.version` holds work beyond that batch.
                if view["buffer_id"] >= step.version:
                    occupied_later += view["n_occupied"]
            v_traj = v_traj_of_group.get(step.binding_group_id)
            if reserved_in is not None and v_traj is not None:
                at_limit = reserved_in - v_traj >= staleness_horizon
        out.append(
            IdleWindow(
                version=step.version,
                t0=t0,
                t1=t1,
                busy_fraction=busy,
                mean_queue_depth=sum(depths) / len(depths) if depths else 0.0,
                binding_group_id=step.binding_group_id,
                binding_reserved_in_buffer=reserved_in,
                binding_at_staleness_limit=at_limit,
                n_occupied_later=occupied_later,
            )
        )
    return out


def instance_imbalance(store) -> list[InstanceImbalance]:
    """
    Return per-instance busy seconds, tokens and last activity.

    Args:
        store (JsonlStore): A finished run's log store.

    Returns:
        list[InstanceImbalance]: One entry per instance that ran a step,
            ordered by instance id. A fleet whose entries differ widely was
            unevenly loaded, whatever the mean concurrency reported.
    """
    rows = []
    for instance_id, segments in sorted(_rollout_segments(store).items()):
        rows.append(
            InstanceImbalance(
                instance_id=instance_id,
                busy_seconds=sum(t1 - t0 for t0, t1, _ in segments),
                tokens=sum(tokens for _, _, tokens in segments),
                last_active=max(t1 for _, t1, _ in segments),
            )
        )
    return rows
