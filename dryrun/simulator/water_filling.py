"""
The water-filling solver: decides the cohort layout for the rollout fleet.

Given an in-flight request snapshot and a GPU budget, `WaterFillingSolver.solve`
returns a `FleetPlan` describing how many instances to allocate per TP degree.
The ``search`` strategy enumerates all legal (tp, n_instances) allocations and
picks the one that maximises predicted aggregate fleet token rate, querying the
cost models.  The ``fixed`` strategy returns a user-specified layout verbatim,
used by homogeneous-fleet baselines.

The solver receives a ``list[RequestSnapshot]`` rather than ``list[Request]`` so
that it structurally cannot read ``target_len`` or ``remaining``: a solver that
peeks at true future lengths would be an oracle and its throughput estimates
would not generalise to real deployments.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as cartesian_product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dryrun.cost.rollout.base import RolloutCostModel


@dataclass(frozen=True)
class RequestSnapshot:
    """
    The subset of ``Request`` fields the solver is allowed to observe.

    Deliberately omits ``target_len`` and ``remaining`` so the solver cannot
    act as an oracle.  All three fields are observable at decision time.
    """

    v_traj: int
    """Weight version at which this trajectory was admitted."""
    ctx: int
    """Current context depth: prompt length plus tokens generated so far."""
    instance_id: int | None
    """Instance most recently serving this request, or ``None`` if unplaced."""


@dataclass(frozen=True)
class CohortPlan:
    """One group of instances sharing a TP degree and a version predicate."""

    tp: int
    """Tensor-parallel degree for every instance in this cohort."""
    n_instances: int
    """Number of inference engine instances in this cohort."""
    max_version: int | None
    """
    Routing predicate.  A request is routed here when
    ``max_version is None or req.v_traj <= max_version``.
    ``None`` makes this cohort a catch-all for any version not claimed earlier.
    """


@dataclass(frozen=True)
class FleetPlan:
    """
    The complete cohort layout for one weight-update interval.

    Invariants enforced in ``__post_init__``:

    1. **GPU budget**: ``sum(c.tp * c.n_instances for c in cohorts)`` must equal
       ``n_gpus`` exactly.  The solver enumerates candidates arithmetically; this
       assertion catches an off-by-one in that enumeration before it can produce
       plausible-but-wrong numbers.

    2. **Catch-all uniqueness**: at most one cohort may have ``max_version=None``.
       A second catch-all is unreachable by the version-predicate router and
       wastes GPU budget on instances that will never receive requests.
    """

    cohorts: tuple[CohortPlan, ...]
    n_gpus: int

    def __post_init__(self) -> None:
        used = sum(c.tp * c.n_instances for c in self.cohorts)
        if used != self.n_gpus:
            raise ValueError(
                f"GPU budget violated: cohorts consume {used} GPU-ranks "
                f"but n_gpus={self.n_gpus}."
            )
        n_catchall = sum(1 for c in self.cohorts if c.max_version is None)
        if n_catchall > 1:
            raise ValueError(
                f"Routing invariant violated: {n_catchall} cohorts have "
                f"max_version=None (catch-all), but the version-predicate router "
                f"can reach at most one.  Every extra catch-all is unreachable "
                f"and idles its GPU-ranks."
            )


class WaterFillingSolver:
    """
    Decides the cohort layout at each weight-publish event.

    Two strategies are supported.

    ``search``:
        Enumerate all legal (tp_old, n_old, tp_new, n_new) tuples that
        partition ``n_gpus`` GPU-ranks exactly, score each by predicted
        aggregate fleet token rate using the provided cost models, and return
        the highest-scoring plan.  Ties are broken by preferring fewer cohorts
        so that a homogeneous fleet is chosen when splitting buys nothing.

    ``fixed``:
        Return the cohort layout from ``fixed_cohorts`` verbatim, ignoring the
        snapshot.  Used by the two homogeneous-TP baselines.
    """

    def __init__(
        self,
        cost_models: dict[int, "RolloutCostModel"],
        n_gpus: int,
        tp_candidates: tuple[int, ...],
        strategy: str,
        fixed_cohorts: list[dict],
        max_concurrency: int | None,
    ) -> None:
        """
        Args:
            cost_models (dict[int, RolloutCostModel]): Mapping from TP degree to
                its fitted cost model.  The ``search`` strategy queries
                ``model.step_time(n_tokens, t2, ctxsum, n_reqs)`` for each candidate.
            n_gpus (int): Total GPU-rank budget the plan must consume exactly.
            tp_candidates (tuple[int, ...]): TP degrees the solver may choose from.
                Every value must divide ``n_gpus``.
            strategy (str): Either ``"search"`` or ``"fixed"``.
            fixed_cohorts (list[dict]): Used only when ``strategy="fixed"``.
                Each dict must have keys ``"tp"`` and ``"n_instances"``; ``"max_version"``
                defaults to ``None``.
            max_concurrency (int | None): Cap on the per-instance batch size used
                in scoring.  ``None`` means no cap.

        Raises:
            ValueError: If ``strategy`` is not ``"search"`` or ``"fixed"``.
        """
        if strategy not in ("search", "fixed"):
            raise ValueError(
                f"Unknown water-filling strategy {strategy!r}; expected 'search' or 'fixed'."
            )
        self._models = cost_models
        self._n_gpus = n_gpus
        # Sort descending so higher-TP options are enumerated first; this is a
        # purely cosmetic preference for tie iteration order, not a correctness
        # requirement.
        self._tp_candidates = tuple(sorted(tp_candidates, reverse=True))
        self._strategy = strategy
        self._fixed_cohorts = fixed_cohorts
        self._max_concurrency = max_concurrency

    def solve(
        self,
        snapshot: list[RequestSnapshot],
        rollout_version: int,
    ) -> FleetPlan:
        """
        Return the fleet plan for the next rollout interval.

        Args:
            snapshot (list[RequestSnapshot]): In-flight requests at decision
                time.  Must use ``RequestSnapshot``, not ``Request``, to
                enforce the oracle firewall.
            rollout_version (int): The version that was just published.
                Requests with ``v_traj < rollout_version`` belong to the old
                cohort; all others belong to the new cohort.

        Returns:
            FleetPlan: A plan whose GPU-rank usage equals ``n_gpus``.
        """
        if self._strategy == "fixed":
            return self._fixed_plan()
        return self._search_plan(snapshot, rollout_version)

    # --- Fixed strategy ---

    def _fixed_plan(self) -> FleetPlan:
        cohorts = tuple(
            CohortPlan(
                tp=int(c["tp"]),
                n_instances=int(c["n_instances"]),
                max_version=c.get("max_version", None),
            )
            for c in self._fixed_cohorts
        )
        return FleetPlan(cohorts=cohorts, n_gpus=self._n_gpus)

    # --- Search strategy ---

    def _search_plan(
        self,
        snapshot: list[RequestSnapshot],
        rollout_version: int,
    ) -> FleetPlan:
        old = [r for r in snapshot if r.v_traj < rollout_version]
        new = [r for r in snapshot if r.v_traj >= rollout_version]
        r_old, r_new = len(old), len(new)

        # Mean context depth per partition; fall back to 1 when a partition is
        # empty so division by zero cannot occur.
        d_old = int(sum(r.ctx for r in old) / max(1, r_old))
        d_new = int(sum(r.ctx for r in new) / max(1, r_new))

        # The key is a single tuple (score, -n_cohorts) compared atomically.
        # Higher score wins; on a tie, fewer cohorts win (so a homogeneous
        # fleet is preferred when the split buys nothing).  Both components
        # are kept in ONE tuple; comparing a float against a tuple would raise
        # a TypeError on the very first iteration.
        best_key: tuple[float, int] | None = None
        best_plan: FleetPlan | None = None

        for tp_old, tp_new in cartesian_product(self._tp_candidates, repeat=2):
            # n_old may be 0, which means the old cohort is absent entirely.
            for n_old in range(self._n_gpus // tp_old + 1):
                remaining = self._n_gpus - n_old * tp_old
                if remaining < 0:
                    continue
                if remaining % tp_new != 0:
                    continue
                n_new = remaining // tp_new
                # Skip plans that leave new requests unservable.
                if n_new == 0 and r_new > 0:
                    continue

                score = self._score(r_old, d_old, tp_old, n_old) + self._score(
                    r_new, d_new, tp_new, n_new
                )
                # Count only cohorts that would actually be emitted (n > 0).
                n_cohorts = (1 if n_old > 0 else 0) + (1 if n_new > 0 else 0)
                key = (score, -n_cohorts)
                if best_key is None or key > best_key:
                    best_key = key
                    cohorts: list[CohortPlan] = []
                    if n_old > 0:
                        cohorts.append(CohortPlan(
                            tp=tp_old,
                            n_instances=n_old,
                            max_version=rollout_version - 1,
                        ))
                    if n_new > 0:
                        cohorts.append(CohortPlan(
                            tp=tp_new,
                            n_instances=n_new,
                            max_version=None,
                        ))
                    best_plan = FleetPlan(cohorts=tuple(cohorts), n_gpus=self._n_gpus)

        assert best_plan is not None, (
            "WaterFillingSolver found no valid plan.  Check that n_gpus and "
            "tp_candidates are consistent."
        )
        return best_plan

    def _score(self, r: int, d: int, tp: int, n_instances: int) -> float:
        """
        Predict aggregate fleet token rate for one cohort.

        Args:
            r (int): Number of requests assigned to this cohort.
            d (int): Mean context depth (tokens) per request.
            tp (int): TP degree; selects the cost model.
            n_instances (int): Number of instances in this cohort.

        Returns:
            float: Predicted tokens-per-second across all instances, or 0.0
                when the cohort is empty or has no cost model.
        """
        if n_instances == 0 or r == 0:
            return 0.0
        model = self._models.get(tp)
        if model is None:
            return 0.0
        # Distribute r requests evenly across instances; clip to max_concurrency
        # if set, and floor at 1 to avoid a zero-batch step_time call.
        batch = r // n_instances
        if self._max_concurrency is not None:
            batch = min(batch, self._max_concurrency)
        batch = max(1, batch)
        # Decode-only step: n_tokens=batch, t2=0, ctxsum=batch*d, n_reqs=batch.
        step_s = model.step_time(batch, 0, batch * d, batch)
        return n_instances * batch / max(step_s, 1e-9)
