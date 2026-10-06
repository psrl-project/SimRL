"""The routing strategies.

Each is the smallest rule that captures one placement idea. Ties always resolve
to the lowest index, which is instance id order, so a run is reproducible and
`request_num_balance` on one instance reduces to "the only instance".
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ..engine import InstanceView, Request
from .strategy import register_strategy

if TYPE_CHECKING:
    from dryrun.simulator.water_filling import FleetPlan


@register_strategy("request_num_balance")
class RequestNumBalance:
    """Fewest in-flight requests wins.

    What the matched PSRL run uses, and what the simulator's former
    `_select_instance` did (`argmin` over `n_waiting + n_running`), so this is
    the behaviour-preserving default rather than a new option.
    """

    name = "request_num_balance"

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        return min(range(len(candidates)), key=lambda i: candidates[i].n_active)


@register_strategy("round_robin")
class RoundRobin:
    """Cycle through candidates.

    The counter advances per placement, not per candidate list, so a shrinking
    candidate set shifts the cycle rather than restarting it. That is SMG's
    behaviour and the reason this is not simply `rid % n`.
    """

    name = "round_robin"

    def __init__(self) -> None:
        self._counter = 0

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        index = self._counter % len(candidates)
        self._counter += 1
        return index


@register_strategy("random")
class RandomChoice:
    """Uniform over candidates, from a seeded generator.

    The generator is owned rather than taken from the `random` module, so a
    simulation's routing cannot be perturbed by unrelated code drawing from the
    global stream.
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        return self._rng.randrange(len(candidates))


@register_strategy("kv_cache_aware")
class KvCacheAware:
    """Most free KV blocks wins, counting what is committed but not yet held.

    Balances the resource that actually binds. `n_active` treats a request that
    has generated 10k tokens as equal to one that has generated none, while
    those two differ by three orders of magnitude in blocks held.

    `free_blocks` alone is not enough, and the failure is silent rather than
    noisy. A request sitting in an instance's waiting queue has allocated
    nothing yet, so two instances holding wildly different commitments report
    identical free blocks, `max` keeps returning the first of them, and an
    entire batch lands on one instance while the rest idle. Subtracting the
    committed context is what makes the signal appear before the blocks do.
    It is the same quantity `add_request` reserves on, so the strategy and the
    gate agree about what "full" means.
    """

    name = "kv_cache_aware"

    def _headroom(self, view: InstanceView) -> int:
        return view.free_blocks - view.committed_blocks

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        return max(range(len(candidates)), key=lambda i: self._headroom(candidates[i]))


@register_strategy("prefix_aware")
class PrefixAware:
    """Route to whoever already holds this request's prefix, unless skewed.

    Two guards, both from SMG's `cache_aware`:

    - **Imbalance first.** Affinity is abandoned when the load spread exceeds
      `balance_abs_threshold` *and* the ratio exceeds `balance_rel_threshold`.
      Both, not either: an absolute gap of 2 between 2 and 4 requests is a
      ratio of 2.0 but means nothing, while a gap of 64 between 500 and 564 is
      a ratio of 1.1 and means plenty.
    - **A floor on the hit.** Below `cache_threshold` of the context, chasing a
      hit costs more in imbalance than it saves in prefill.

    Falls back to `request_num_balance` in both cases, so the degenerate
    configuration behaves exactly like the default rather than unpredictably.
    """

    name = "prefix_aware"

    def __init__(
        self,
        cache_threshold: float = 0.3,
        balance_abs_threshold: int = 64,
        balance_rel_threshold: float = 1.5,
    ) -> None:
        self.cache_threshold = cache_threshold
        self.balance_abs_threshold = balance_abs_threshold
        self.balance_rel_threshold = balance_rel_threshold
        self._balance = RequestNumBalance()

    def _imbalanced(self, candidates: list[InstanceView]) -> bool:
        loads = [view.n_active for view in candidates]
        low, high = min(loads), max(loads)
        return (
            high - low > self.balance_abs_threshold
            and high > low * self.balance_rel_threshold
        )

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        if self._imbalanced(candidates):
            return self._balance.select(req, candidates)
        context = max(1, req.prompt_len + req.generated)
        best = max(range(len(candidates)), key=lambda i: candidates[i].prefix_hit_tokens(req))
        if candidates[best].prefix_hit_tokens(req) / context <= self.cache_threshold:
            return self._balance.select(req, candidates)
        return best


@register_strategy("throughput_balance")
class ThroughputBalance:
    """Smallest predicted step-time increase wins.

    The only strategy that consults the cost model. No run has been matched
    against it, so it is offered rather than recommended. See
    `docs/simulation/routing.md`.
    """

    name = "throughput_balance"

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        return min(range(len(candidates)), key=lambda i: candidates[i].step_time_delta(req))


@register_strategy("cohort_affinity")
class CohortAffinity:
    """Route old-version requests to the high-TP cohort; new to the low-TP cohort.

    Cohort assignment is determined by comparing v_traj against each cohort's
    max_version predicate from the FleetPlan.  Falls back to request_num_balance
    across all candidates when the preferred cohort has no eligible instances,
    so the fleet split never strands a request.

    Requires a FleetPlan to be set via `update_plan` before the first dispatch.
    HeterogeneousFleet calls this immediately after rebuilding instances.
    """

    name = "cohort_affinity"

    def __init__(self, plan: "FleetPlan | None" = None) -> None:
        self._plan: FleetPlan | None = plan
        self._fallback = RequestNumBalance()
        self.n_cohort_leaks: int = 0
        """Count of requests that leaked to a non-preferred cohort due to capacity."""

    def update_plan(self, plan: "FleetPlan") -> None:
        """Replace the cohort layout after a fleet rebuild.

        Args:
            plan: The new fleet layout. Called by HeterogeneousFleet immediately
                after it rebuilds its instance list so strategy and fleet agree.
        """
        self._plan = plan

    def _preferred_cohort_id(self, req: "Request") -> int:
        """Return the cohort index this request belongs to per the current plan.

        Walks cohorts in order and returns the first whose max_version predicate
        accepts the request's v_traj.  A cohort with max_version=None is a
        catch-all and always accepts.  When no plan has been set yet (before the
        first fleet build) every request is treated as cohort 0, which is the
        only safe default when cohort counts and version bounds are unknown.

        Args:
            req: The request being routed.

        Returns:
            The index of the cohort that should serve this request.
        """
        if self._plan is None:
            return 0
        for i, cohort in enumerate(self._plan.cohorts):
            if cohort.max_version is None or req.v_traj <= cohort.max_version:
                return i
        return len(self._plan.cohorts) - 1  # catch-all

    def select(self, req: "Request", candidates: list[InstanceView]) -> int | None:
        """Choose an instance, preferring the cohort that matches req.v_traj.

        Narrows the candidate list to instances in the preferred cohort and
        applies request_num_balance within that subset.  When no preferred-cohort
        instance is present — because all are at capacity and were filtered by the
        admission gate — the strategy falls back to request_num_balance over the
        whole candidate list and increments n_cohort_leaks.  Stranding the request
        instead would livelock the simulator: if the fleet is heterogeneous and
        old-version requests can only go to one instance, a single busy instance
        would block all old-version work indefinitely.

        Args:
            req: The request to place.
            candidates: Instances that passed the admission gate, in instance
                id order.

        Returns:
            An index into candidates, or None to leave the request queued.
        """
        preferred = self._preferred_cohort_id(req)
        narrowed = [i for i, v in enumerate(candidates) if v.cohort_id == preferred]
        if narrowed:
            # Within the preferred cohort, use request_num_balance so ties
            # still resolve deterministically (lowest index wins).
            sub = [candidates[i] for i in narrowed]
            local_idx = self._fallback.select(req, sub)
            if local_idx is not None:
                return narrowed[local_idx]
        # Preferred cohort full — fall back to the whole candidate list.  A high
        # n_cohort_leaks rate in the results means the measured cohort effect is
        # attenuated: old-version requests ran on new-version instances, which
        # reduces any TP-mismatch signal the experiment was designed to observe.
        self.n_cohort_leaks += 1
        return self._fallback.select(req, candidates)
