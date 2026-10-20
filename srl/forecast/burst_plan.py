"""Price a dispatch burst's requests jointly, instead of one at a time.

See `docs/superpowers/specs/2026-10-20-burst-joint-placement-design.md`. A burst is the set of
already-admitted members a `PlanSession` would otherwise place one at a time, in its own
longest-remaining-first `tail_order`. `plan_burst` tries a small, fixed grid of isolation fractions
-- how many of the longest members get concentrated onto the currently least-loaded instance(s) as
a block -- and prices each whole candidate with `PlanSession.price_assignment`, picking the
cheapest with the existing `compare()`. The `0.0` fraction (no isolation, plain earliest-finish
placement via `fold_member`) is always included, so this can never price worse than that.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..component.rollout.engine import Request, TrajView, traj_view
from .fleet import FleetForecast, arrival_of, fold_member
from .pipeline import Pipeline, compare
from .plan import PlanSession

ISOLATION_FRACTIONS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
"""Fraction of the burst, by count (longest first), concentrated onto the least-loaded instance(s)."""


def _isolation_sizes(n: int) -> list[int]:
    """One size per fraction, deduplicated, ascending, always including 0."""
    return sorted({round(frac * n) for frac in ISOLATION_FRACTIONS})


def _assign_isolated_block(
    block: Sequence[Request],
    pool: Sequence[int],
    fleet: FleetForecast,
    remaining_of: Callable[[TrajView], float],
    now: float,
    assignment: dict[int, int],
) -> FleetForecast:
    """
    Fill `pool` (already least-loaded first) with `block` (already longest-first).

    Spills to the next pool instance once the current one's `max_concurrency` is exceeded by its
    residents plus how many of `block` have already landed there. `max_concurrency=None` never
    spills, so a single-instance pool always takes the whole block.
    """
    fleet_now = fleet
    if not block or not pool:
        return fleet_now
    filled = {i: len(fleet_now.instances[i].state.running) for i in pool}
    cursor = 0
    for req in block:
        cap = fleet_now.instances[pool[cursor]].state.max_concurrency
        while cap is not None and filled[pool[cursor]] >= cap and cursor + 1 < len(pool):
            cursor += 1
        target = pool[cursor]
        assignment[req.rid] = target
        filled[target] += 1
        fleet_now = fleet_now.with_placement(target, arrival_of(req, now, remaining_of))
    return fleet_now


def generate_candidates(session: PlanSession, burst: Sequence[Request]) -> list[dict[int, int]]:
    """
    Build one candidate `rid -> instance_id` assignment per isolation size.

    Args:
        session (PlanSession): Supplies the fleet (stripped of `burst`'s own tentative placements),
            the length predictor, and the current time.
        burst (Sequence[Request]): The already-admitted members this dispatch is placing.

    Returns:
        list[dict[int, int]]: One full assignment per isolation size that could place every request
            in `burst` without KV overflow during construction (overflow against the *rest* of the
            fleet is still checked later, by `PlanSession.price_assignment`). Empty when `burst` is
            empty or every isolation size failed to place someone.
    """
    if not burst:
        return []
    remaining_of = session.length_predictor.remaining
    ordered = sorted(burst, key=lambda r: -remaining_of(traj_view(r)))
    base = session.strip(ordered)
    instance_ids = sorted(base.instances)
    loaded_order = sorted(instance_ids, key=lambda i: len(base.instances[i].state.running))
    candidates: list[dict[int, int]] = []
    for k in _isolation_sizes(len(ordered)):
        isolated, bulk = ordered[:k], ordered[k:]
        assignment: dict[int, int] = {}
        working = _assign_isolated_block(isolated, loaded_order, base, remaining_of, session.now, assignment)
        for req in bulk:
            working, chosen, _ = fold_member(working, req, remaining_of, now=session.now)
            if chosen is not None:
                assignment[req.rid] = chosen
        if len(assignment) == len(ordered):
            candidates.append(assignment)
    return candidates


def plan_burst(session: PlanSession, burst: Sequence[Request], eps: float) -> dict[int, int] | None:
    """
    Price every isolation candidate for `burst` and return the cheapest one's assignment.

    Args:
        session (PlanSession): Prices each candidate via `session.price_assignment`.
        burst (Sequence[Request]): The already-admitted members this dispatch is placing.
        eps (float): Tolerance `compare()` uses to rank candidate pipelines.

    Returns:
        dict[int, int] | None: `rid -> instance_id` for every request in `burst`, `{}` if `burst`
            is empty, or `None` if every candidate -- including the always-included `0.0` one --
            overflowed some instance's KV pool. Callers should fall back to per-request placement
            in the `None` case.
    """
    if not burst:
        return {}
    best: dict[int, int] | None = None
    best_pipeline: Pipeline | None = None
    for candidate in generate_candidates(session, burst):
        pipeline = session.price_assignment(burst, candidate)
        if pipeline is None:
            continue
        if best_pipeline is None or compare(pipeline, best_pipeline, eps) < 0:
            best, best_pipeline = candidate, pipeline
    return best
