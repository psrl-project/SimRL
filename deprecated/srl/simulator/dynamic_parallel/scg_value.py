"""Score fleet plans by version-capped cohort makespan.

The assignment mirrors `VersionCappedTp`. Stale trajectories use only the
highest TP tier. Current trajectories use the lowest tier while stale work
still exceeds the high-tier admission cap, then balance across the whole fleet.
The cap also limits each modeled resident wave.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...cost.rollout.base import RolloutCostModel


@dataclass(frozen=True)
class TrajSnapshot:
    """Observable state of one in-flight trajectory at decision time."""

    v_traj: int
    """Weight version this trajectory was admitted at."""
    ctx: int
    """Current context depth (prompt + generated so far)."""
    remaining: int
    """Tokens left to generate, positive.

    Exact in the simulator. A real scheduler would supply an estimate (the
    hazard-model mean of §6.2), which is why this is a snapshot type rather than
    a direct read of `Request`.
    """


def _drain_seconds(
    model: RolloutCostModel,
    trajs: list[TrajSnapshot],
    max_concurrency: int | None,
    active_cap: int | None,
) -> float:
    """Seconds for one instance to drain `trajs`, and the cost of sharing.

    Models continuous batching the way the engine runs it: all resident requests
    decode together, so each one pays a step cost set by the *whole* batch's
    summed context. The instance is done when its longest member is done.

    Requests beyond `max_concurrency` queue behind the resident set rather than
    joining the batch, so they are charged the resident drain first. That matters
    for scoring a concentrated plan honestly: piling the fleet onto one instance
    does not make its engine wider than the engine is.

    Args:
        model: Cost model for this instance's TP degree.
        trajs: Requests assigned to this instance.
        max_concurrency: Engine cap on simultaneously-decoding requests.

    Returns:
        Seconds until the last assigned request finishes. 0.0 when none.
    """
    if not trajs:
        return 0.0

    resident = len(trajs)
    if max_concurrency is not None:
        resident = min(resident, max_concurrency)
    if active_cap is not None:
        resident = min(resident, active_cap)

    # Average context across the batch sets the shared ctxsum each step pays.
    mean_ctx = sum(t.ctx for t in trajs) / len(trajs)
    ctxsum = resident * mean_ctx

    # The batch is alive until its longest member drains. Pricing the longest
    # member against the shared batch cost is what makes isolating a tail pay:
    # the same token count costs far less per step in a batch of 14 than of 64.
    longest = max(t.remaining for t in trajs)
    drain = model.decode_segment_time(resident, ctxsum, int(longest))

    # Work that could not fit in the resident window runs after it, in waves.
    waves = (len(trajs) + resident - 1) // resident
    if waves > 1:
        drain *= waves

    return drain


def plan_makespan(
    tp_per_instance: tuple[int, ...],
    models: dict[int, RolloutCostModel],
    assignments: list[list[TrajSnapshot]],
    max_concurrency: int | None = None,
    active_cap: int | None = None,
) -> float:
    """Seconds until every assigned request finishes, across the whole fleet.

    Args:
        tp_per_instance: TP degree of each instance.
        models: TP -> cost model.
        assignments: Requests per instance, parallel to `tp_per_instance`.
        max_concurrency: Engine cap on resident requests per instance.

    Returns:
        The fleet makespan in seconds, i.e. the slowest instance's drain.
    """
    worst = 0.0
    for tp, trajs in zip(tp_per_instance, assignments):
        worst = max(
            worst,
            _drain_seconds(models[tp], trajs, max_concurrency, active_cap),
        )
    return worst


def assign_version_capped(
    tp_per_instance: tuple[int, ...],
    old_trajs: list[TrajSnapshot],
    new_trajs: list[TrajSnapshot],
    active_cap: int | None,
) -> tuple[list[list[TrajSnapshot]], list[list[TrajSnapshot]]]:
    """Bucket stale and current trajectories by the router's version tiers."""
    n = len(tp_per_instance)
    old_buckets: list[list[TrajSnapshot]] = [[] for _ in range(n)]
    new_buckets: list[list[TrajSnapshot]] = [[] for _ in range(n)]
    if n == 0:
        return old_buckets, new_buckets

    max_tp = max(tp_per_instance)
    min_tp = min(tp_per_instance)
    high = [i for i, tp in enumerate(tp_per_instance) if tp == max_tp]
    low = [i for i, tp in enumerate(tp_per_instance) if tp == min_tp]
    _assign_balanced(old_buckets, old_trajs, high)

    high_capacity = None if active_cap is None else active_cap * len(high)
    stale_backlog = high_capacity is not None and len(old_trajs) > high_capacity
    current_targets = low if stale_backlog and min_tp < max_tp else list(range(n))
    initial_load = [len(bucket) for bucket in old_buckets]
    _assign_balanced(new_buckets, new_trajs, current_targets, initial_load)
    return old_buckets, new_buckets


def _assign_balanced(
    buckets: list[list[TrajSnapshot]],
    trajs: list[TrajSnapshot],
    targets: list[int],
    initial_load: list[int] | None = None,
) -> None:
    """Assign trajectories to the least-loaded target in version order."""
    if not targets:
        return
    load = list(initial_load) if initial_load is not None else [0] * len(buckets)
    for traj in sorted(trajs, key=lambda item: (item.v_traj, -item.remaining)):
        target = min(targets, key=lambda index: (load[index], index))
        buckets[target].append(traj)
        load[target] += 1


def score_plan_scg(
    tp_per_instance: tuple[int, ...],
    models: dict[int, RolloutCostModel],
    old_trajs: list[TrajSnapshot],
    new_trajs: list[TrajSnapshot],
    T_tr: float,
    max_concurrency: int | None = None,
    active_cap: int | None = None,
    future_discount: float = 1.0,
    drain_weight: float = 1.0,
) -> float:
    """Score a candidate fleet plan. Higher is better.

    The score is the negated fleet makespan over the in-flight set, in units of
    training cycles rather than seconds so it reads as "weight updates this plan
    lets the cohort span" -- which is staleness, and therefore SCG:

        score = -(makespan / T_tr)

    `drain_weight` biases the tail's contribution: the makespan is recomputed
    with only the deep requests when it exceeds 1.0, which expresses "spend
    capacity on the stragglers even if the bulk slows a little".

    Args:
        tp_per_instance: TP degree of each instance.
        models: TP -> cost model.
        old_trajs: Trajectories from before the current rollout version.
        new_trajs: Trajectories at the current rollout version.
        T_tr: Expected training-cycle duration in seconds, the staleness quantum.
        max_concurrency: Engine cap on resident requests per instance.
        future_discount: Weight on the new-version population's makespan, in
            [0, 1]. Below 1.0 the solver cares less about fresh work, which has a
            full staleness budget and is in less danger of being dropped.
        drain_weight: Weight on the old-version (straggler) makespan term.

    Returns:
        Negated, cycle-normalised makespan. Higher means the cohort drains in
        fewer weight updates, which is a higher SCG.
    """
    if not tp_per_instance:
        return float("-inf")
    quantum = max(T_tr, 1e-9)
    old_assignments, new_assignments = assign_version_capped(
        tp_per_instance,
        old_trajs,
        new_trajs,
        active_cap,
    )

    # Old-version work is what is at risk of ageing out, so it is scored on its
    # own makespan rather than folded into a fleet-wide average that a large
    # healthy population would dominate.
    old_span = 0.0
    if old_trajs:
        old_span = plan_makespan(
            tp_per_instance,
            models,
            old_assignments,
            max_concurrency,
            active_cap,
        )

    new_span = 0.0
    if new_trajs and future_discount > 0.0:
        new_span = plan_makespan(
            tp_per_instance,
            models,
            new_assignments,
            max_concurrency,
            active_cap,
        )

    return -(drain_weight * old_span + future_discount * new_span) / quantum
