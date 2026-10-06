"""Choosing a fleet layout: how many instances, each at what TP degree.

The solver enumerates every legal way to spend the GPU budget and scores each by
predicted aggregate token rate, reading the same measured kernel tables the
simulation itself prices steps with.

**Scoring models what the router will do, not a partition.** Placement is a soft
preference (`VersionAwareTp`): old-version trajectories prefer high-TP instances,
and everything else balances by request count across the whole fleet. So a
candidate is scored by walking that same preference over the snapshot and asking
what batch each instance ends up holding. Scoring a partition the router does
not implement would optimise for a system that does not exist.

The solver never sees ``target_len`` or ``remaining``. It takes a
`RequestSnapshot` carrying only what a real scheduler can observe at decision
time, so a result cannot depend on knowing the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

from .plan import FleetPlan

if TYPE_CHECKING:
    from ...cost.rollout.base import RolloutCostModel

# Interpolating a utility surface costs real time, and the enumeration prices every
# candidate. A fleet this wide is far past anything the kernel tables cover, so the cap
# bounds the search rather than constraining a realistic configuration.
_MAX_INSTANCES = 64


@dataclass(frozen=True)
class RequestSnapshot:
    """The subset of `Request` fields the solver is allowed to observe.

    Omits ``target_len`` and ``remaining`` by construction: a solver that could
    read true remaining length would be an oracle, and every number measured
    against it would be an unreachable upper bound rather than a prediction.
    """

    v_traj: int
    """Weight version this trajectory was admitted at."""
    ctx: int
    """Context depth so far, ``prompt_len + generated``."""


class ParallelismSolver:
    """Picks the TP degree of every rollout instance.

    Two strategies:

    - ``search`` enumerates every legal multiset of TP degrees summing to the GPU
      budget and takes the argmax of predicted token rate.
    - ``fixed`` returns a configured plan verbatim, ignoring the snapshot. This is
      how a static homogeneous baseline runs through the identical code path, so a
      measured difference cannot be an artefact of the dynamic plumbing.
    """

    def __init__(
        self,
        cost_models: dict[int, "RolloutCostModel"],
        n_gpus: int,
        tp_candidates: tuple[int, ...],
        strategy: str = "search",
        fixed_tp_per_instance: tuple[int, ...] = (),
        max_concurrency: int | None = None,
        high_tp_bias: float = 1.0,
    ) -> None:
        """
        Bind the cost models and the budget the solver must spend.

        Args:
            cost_models: TP degree -> cost model. Must cover every entry in
                `tp_candidates`; a model is sharded for one degree at construction,
                so a TP=2 instance cannot be priced by a TP=1 model.
            n_gpus: GPU budget every plan must spend exactly.
            tp_candidates: TP degrees the solver may use.
            strategy: ``search`` or ``fixed``.
            fixed_tp_per_instance: The plan to return under ``fixed``.
            max_concurrency: Engine cap on resident requests per instance, applied
                when estimating the batch an instance would hold.
            high_tp_bias: How strongly the router is assumed to steer old-version
                work toward high-TP instances, in ``[0, 1]``. Must match the
                router's own bias, or the solver optimises for a placement that
                will not happen. See `VersionAwareTp`.

        Raises:
            ValueError: If the strategy is unknown, the candidates are empty, a
                cost model is missing, or a ``fixed`` plan is absent or misspends
                the budget.
        """
        if strategy not in ("search", "fixed"):
            raise ValueError(
                f"Unknown parallelism strategy {strategy!r}; expected 'search' or 'fixed'."
            )
        if not tp_candidates:
            raise ValueError("tp_candidates must not be empty.")
        missing = sorted(set(tp_candidates) - set(cost_models))
        if missing:
            raise ValueError(
                f"No cost model for TP {missing}. Each TP degree needs its own model: "
                f"a model shards its ops at construction, so one built for TP=1 cannot "
                f"price a TP=2 instance."
            )
        if not 0.0 <= high_tp_bias <= 1.0:
            raise ValueError(f"high_tp_bias must lie in [0, 1], got {high_tp_bias!r}.")
        self._models = cost_models
        self._n_gpus = int(n_gpus)
        # Descending, so the enumeration reaches concentrated plans first and the
        # tie-break below keeps the simplest plan among equals.
        self._tp_candidates = tuple(sorted({int(tp) for tp in tp_candidates}, reverse=True))
        self._strategy = strategy
        self._max_concurrency = max_concurrency
        self._high_tp_bias = float(high_tp_bias)
        self._fixed = tuple(int(tp) for tp in fixed_tp_per_instance)
        if strategy == "fixed":
            # Validated here rather than at the first solve: a misspent fixed plan is a
            # configuration error, and failing at construction names it before a
            # simulation has run for minutes.
            FleetPlan(tp_per_instance=self._fixed, n_gpus=self._n_gpus)
        self._legal_plans = self._enumerate_plans()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self, snapshot: list[RequestSnapshot], rollout_version: int
    ) -> FleetPlan:
        """
        Return the fleet layout for the next rollout interval.

        Args:
            snapshot: In-flight requests, as observable projections.
            rollout_version: The version just published. Requests below it are the
                old-version tail.

        Returns:
            A plan spending the GPU budget exactly.
        """
        if self._strategy == "fixed":
            return FleetPlan(tp_per_instance=self._fixed, n_gpus=self._n_gpus)
        return self._search(snapshot, rollout_version)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enumerate_plans(self) -> tuple[tuple[int, ...], ...]:
        """Return every multiset of TP degrees that spends the budget exactly.

        Computed once at construction: the legal set depends only on the budget and
        the candidate degrees, never on the snapshot, so re-deriving it per weight
        update would repeat identical work.

        A multiset, not a permutation: two instances at the same degree are
        interchangeable, so ordering them would multiply the search by instance
        count without changing any score.
        """
        smallest = min(self._tp_candidates)
        max_instances = min(self._n_gpus // smallest, _MAX_INSTANCES)
        plans: list[tuple[int, ...]] = []
        for size in range(1, max_instances + 1):
            for combo in combinations_with_replacement(self._tp_candidates, size):
                if sum(combo) == self._n_gpus:
                    plans.append(combo)
        if not plans:
            raise ValueError(
                f"No fleet of TP degrees {self._tp_candidates} spends {self._n_gpus} "
                f"GPUs exactly."
            )
        return tuple(plans)

    def _search(
        self, snapshot: list[RequestSnapshot], rollout_version: int
    ) -> FleetPlan:
        """Score every legal plan and return the best."""
        old = [r for r in snapshot if r.v_traj < rollout_version]
        new = [r for r in snapshot if r.v_traj >= rollout_version]

        best_key: tuple[float, int] | None = None
        best: tuple[int, ...] = self._legal_plans[0]
        for tp_per_instance in self._legal_plans:
            score = self._score_plan(tp_per_instance, len(old), _mean_ctx(old),
                                     len(new), _mean_ctx(new))
            # Prefer fewer instances on a tie, which keeps the simplest fleet among
            # equals and stops the solver claiming credit for a split worth nothing.
            key = (score, -len(tp_per_instance))
            if best_key is None or key > best_key:
                best_key, best = key, tp_per_instance
        return FleetPlan(tp_per_instance=best, n_gpus=self._n_gpus)

    def _score_plan(
        self,
        tp_per_instance: tuple[int, ...],
        r_old: int,
        d_old: int,
        r_new: int,
        d_new: int,
    ) -> float:
        """
        Predict the aggregate token rate of one candidate fleet.

        Mirrors the router's placement rather than assuming a partition: the
        old-version tail is steered toward the highest TP degrees in proportion to
        `high_tp_bias`, and everything else spreads evenly by request count. The
        per-instance batch that falls out is what each instance is priced at.

        Args:
            tp_per_instance: The candidate plan.
            r_old: Old-version request count.
            d_old: Mean context depth of the old-version requests.
            r_new: New-version request count.
            d_new: Mean context depth of the new-version requests.

        Returns:
            Predicted tokens per second across the fleet.
        """
        assigned = self._expected_placement(tp_per_instance, r_old, r_new)
        total = 0.0
        for tp, (n_old_here, n_new_here) in zip(tp_per_instance, assigned):
            batch = n_old_here + n_new_here
            if batch <= 0:
                continue
            if self._max_concurrency is not None:
                batch = min(batch, self._max_concurrency)
            # Weighted mean depth. A naive mean would price a mix of a few very deep
            # trajectories and many shallow ones as if all were shallow, and attention
            # is quadratic in length, so the error is not second-order.
            depth = (
                (n_old_here * d_old + n_new_here * d_new) / (n_old_here + n_new_here)
            )
            step_s = self._models[tp].step_time(batch, 0, int(batch * depth), batch)
            total += batch / max(step_s, 1e-9)
        return total

    def _expected_placement(
        self, tp_per_instance: tuple[int, ...], r_old: int, r_new: int
    ) -> list[tuple[int, int]]:
        """
        Predict how the router will spread work over a candidate fleet.

        Returns:
            ``(old_count, new_count)`` per instance, in plan order.
        """
        n = len(tp_per_instance)
        # Old-version work prefers high TP. Rank instances by degree, then hand the
        # tail out in that order up to the bias; the remainder spreads evenly, which
        # is what the router's balance fallback does once the preference is satisfied.
        order = sorted(range(n), key=lambda i: -tp_per_instance[i])
        biased = int(round(r_old * self._high_tp_bias))
        old_counts = [0] * n
        for position, index in enumerate(order):
            share = biased // len(order) + (1 if position < biased % len(order) else 0)
            old_counts[index] = share
        # Whatever the bias did not steer spreads evenly over the whole fleet.
        _spread(old_counts, r_old - biased)
        new_counts = [0] * n
        _spread(new_counts, r_new)
        return list(zip(old_counts, new_counts))


def _spread(counts: list[int], total: int) -> None:
    """Add `total` requests evenly across `counts`, in place."""
    if total <= 0 or not counts:
        return
    n = len(counts)
    base, extra = divmod(total, n)
    for i in range(n):
        counts[i] += base + (1 if i < extra else 0)


def _mean_ctx(requests: list[RequestSnapshot]) -> int:
    """Return the mean context depth, or 1 when the set is empty."""
    if not requests:
        return 1
    return max(1, sum(r.ctx for r in requests) // len(requests))
