"""Optimistic span bounds, to say what any scheduler could still win.

Both bounds hold the consumed set and the training durations fixed and remove
only contention, so they ignore capacity entirely: no fleet could actually run
every request alone. That makes each one an upper bound on the gain available
to routing, admission or fleet-shape changes. Large headroom is necessary for
a mechanism to win, never sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass

from .critical_path import StepWait, decompose, group_by_group_id
from .pricer import SoloPricer
from .timeline import RequestTimeline


@dataclass(frozen=True)
class Bound:
    """One counterfactual span, with the assumptions that produced it."""

    name: str
    span: float
    headroom: float
    assumptions: tuple[str, ...]


def _observed_span(steps_: list[StepWait]) -> float:
    return max((step.end for step in steps_), default=0.0)


def _solo_lifetime(
    line: RequestTimeline,
    pricer: SoloPricer,
    tp_by_instance: dict[int, int],
    charge_reprefill: bool,
) -> float:
    """Return how long this request would take alone, from its own admission."""
    parts = decompose(line, pricer, tp_by_instance)
    solo = parts.solo_seconds + parts.weight_load_seconds
    if charge_reprefill:
        solo += parts.reprefill_seconds
    return solo


def span_solo_all(
    timelines: dict[int, RequestTimeline],
    steps_: list[StepWait],
    pricer: SoloPricer,
    tp_by_instance: dict[int, int],
    *,
    charge_reprefill: bool,
) -> Bound:
    """
    Price a run in which every request decodes alone from the moment it is admitted.

    Each training step still waits for its own batch, and each step's own
    duration is unchanged. Only contention is removed, and the chain is
    rebuilt forward so an earlier step start lets the next batch start earlier.

    Args:
        timelines (dict[int, RequestTimeline]): Reconstructed timelines.
        steps_ (list[StepWait]): Training steps in version order.
        pricer (SoloPricer): Prices a phase at a batch of one.
        tp_by_instance (dict[int, int]): Tensor-parallel degree per instance id.
        charge_reprefill (bool): Whether a solo run still pays its observed
            re-prefill. False is the strictly more optimistic variant, since a
            request that never waited might never have been interrupted.

    Returns:
        Bound: The counterfactual span and `observed / bound`.
    """
    by_group = group_by_group_id(timelines)

    clock = 0.0
    for step in steps_:
        members = [line for gid in step.group_ids for line in by_group.get(gid, ())]
        if members:
            # A request cannot start before the publish that stamped it, which
            # in this chain is the previous step's end.
            ready = max(
                clock + _solo_lifetime(line, pricer, tp_by_instance, charge_reprefill)
                for line in members
            )
        else:
            ready = clock
        clock = max(clock, ready) + step.train_seconds
    observed = _observed_span(steps_)
    span = min(clock, observed)
    return Bound(
        name="span_solo_all",
        span=span,
        headroom=observed / span if span > 0 else 1.0,
        assumptions=(
            "every request decodes at a batch of one, so fleet capacity is ignored",
            "the consumed set and every training step duration are unchanged",
            (
                "re-prefill is charged at its observed cost"
                if charge_reprefill
                else "re-prefill is removed entirely"
            ),
        ),
    )


def span_tail_only(
    timelines: dict[int, RequestTimeline],
    steps_: list[StepWait],
    pricer: SoloPricer,
    tp_by_instance: dict[int, int],
) -> Bound:
    """
    Price a run in which only each step's binding request runs alone.

    Everything else keeps its observed timing, and the next-latest member takes
    over as binding once the binding one is sped up, so the bound cannot claim
    more than the second-slowest member allows. This is the regime a long-tail
    isolation mechanism targets: protect the straggler, leave the rest alone.

    Args:
        timelines (dict[int, RequestTimeline]): Reconstructed timelines.
        steps_ (list[StepWait]): Training steps in version order.
        pricer (SoloPricer): Prices a phase at a batch of one.
        tp_by_instance (dict[int, int]): Tensor-parallel degree per instance id.

    Returns:
        Bound: The counterfactual span and `observed / bound`.
    """
    by_group = group_by_group_id(timelines)

    clock = 0.0
    for step in steps_:
        members = [line for gid in step.group_ids for line in by_group.get(gid, ())]
        if members:
            finishes = []
            for line in members:
                if line.rid == step.binding_rid:
                    finishes.append(clock + _solo_lifetime(line, pricer, tp_by_instance, True))
                else:
                    finishes.append(line.complete_time)
            ready = max(finishes)
        else:
            ready = clock
        clock = max(clock, ready) + step.train_seconds
    observed = _observed_span(steps_)
    span = min(clock, observed)
    return Bound(
        name="span_tail_only",
        span=span,
        headroom=observed / span if span > 0 else 1.0,
        assumptions=(
            "only each step's slowest member runs alone, every other member keeps its observed time",
            "the next-slowest member becomes binding once the slowest is accelerated",
            "the consumed set and every training step duration are unchanged",
        ),
    )
