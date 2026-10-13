"""Choosing a fleet layout: how many instances, each at what TP degree.

The solver enumerates every legal way to spend the GPU budget and scores each
candidate. Two scoring objectives are available:

- ``throughput``: predicted aggregate token rate. Always picks the all-TP=1
  replication plan, because more instances means more concurrency. Used as a
  baseline to reproduce the original behaviour.

- ``scg``: expected SCG completions over one training cycle (the default). Scores
  each plan by the expected number of trajectories that finish within T_tr seconds,
  decomposed into old-version drain (the main lever) plus a discounted throughput
  term for new-version admissions. This objective is what drives dynamic
  heterogeneous parallelism to be useful: concentration onto high-TP instances
  reduces per-step latency, so an old-version tail that would miss the staleness
  boundary on TP=1 can drain in time on TP=2.

  The ``scg`` objective requires a training-cycle estimate ``T_tr`` and exposes
  two knobs: ``future_discount`` (weight on new-version completions, default 1.0)
  and ``drain_weight`` (multiplier on old-version drain relative to new-version
  throughput, default 1.0). Setting ``drain_weight > 1.0`` shifts the solver
  toward aggressively draining the old-version tail even at the cost of some
  throughput for fresh work.

**Scoring models what the router will do.** Stale trajectories use the highest
TP tier. Current trajectories stay on the lowest tier while stale work exceeds
the high-tier active cap, then balance across the whole fleet.

**Two snapshot types.** `RequestSnapshot` omits ``remaining`` so the solver
cannot act as an oracle on future length in a general setting. `RichRequestSnapshot`
carries ``remaining`` so the SCG objective can compute exact completion times when
the simulator provides it. In a real system, pass an estimate in place of the true
remaining length.
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


@dataclass(frozen=True)
class RichRequestSnapshot:
    """Extended snapshot that includes the remaining token count.

    Used by the SCG scoring objective, which needs completion-time estimates.
    In the simulator this is exact (``req.remaining``). In a real system, pass
    a per-trajectory estimate such as the hazard-model mean or a fixed budget.
    """

    v_traj: int
    """Weight version this trajectory was admitted at."""
    ctx: int
    """Context depth so far, ``prompt_len + generated``."""
    remaining: int
    """Tokens still to generate. Must be positive."""
    rid: int = -1
    """Request id, so a planner can return its own per-request assignment."""


class ParallelismSolver:
    """Picks the TP degree of every rollout instance.

    Two strategies:

    - ``search`` enumerates every legal multiset of TP degrees summing to the GPU
      budget and takes the argmax of the configured objective.
    - ``fixed`` returns a configured plan verbatim, ignoring the snapshot. This is
      how a static homogeneous baseline runs through the identical code path, so a
      measured difference cannot be an artefact of the dynamic plumbing.

    Two objectives (``objective``):

    - ``scg`` (default): expected SCG completions over one training cycle. Values
      the old-version tail draining before the next weight update, which is what
      makes concentration onto high TP worth anything.
    - ``throughput``: predicted aggregate token rate. Reproduces the original
      behaviour and serves as the ablation that shows the objective is the lever.
    """

    def __init__(
        self,
        cost_models: dict[int, "RolloutCostModel"],
        n_gpus: int,
        tp_candidates: tuple[int, ...],
        strategy: str = "search",
        fixed_tp_per_instance: tuple[int, ...] = (),
        max_concurrency: int | None = None,
        active_cap: int | None = None,
        objective: str = "scg",
        training_cycle_s: float = 60.0,
        future_discount: float = 1.0,
        drain_weight: float = 1.0,
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
            active_cap: Router cap on accepted requests per instance.
            objective: ``scg`` or ``throughput``.
            training_cycle_s: Expected seconds per training cycle, the deadline the
                SCG objective drains the old-version tail against. Updated at
                runtime via `observe_training_cycle` once real step times are known.
            future_discount: Weight on new-version completions relative to
                old-version drain, in ``[0, 1]``. Only read by the SCG objective.
            drain_weight: Multiplier on the old-version drain term. Above 1.0 the
                solver trades fresh-work throughput for tail drain. Only read by
                the SCG objective.

        Raises:
            ValueError: If the strategy or objective is unknown, the candidates are
                empty, a cost model is missing, or a ``fixed`` plan is absent or
                misspends the budget.
        """
        if strategy not in ("search", "fixed"):
            raise ValueError(
                f"Unknown parallelism strategy {strategy!r}; expected 'search' or 'fixed'."
            )
        if objective not in ("scg", "throughput"):
            raise ValueError(
                f"Unknown parallelism objective {objective!r}; expected 'scg' or 'throughput'."
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
        if active_cap is not None and active_cap <= 0:
            raise ValueError(f"active_cap must be positive when set, got {active_cap!r}.")
        if not 0.0 <= future_discount <= 1.0:
            raise ValueError(
                f"future_discount must lie in [0, 1], got {future_discount!r}."
            )
        if drain_weight < 0.0:
            raise ValueError(f"drain_weight must be >= 0, got {drain_weight!r}.")
        if training_cycle_s <= 0.0:
            raise ValueError(
                f"training_cycle_s must be positive, got {training_cycle_s!r}."
            )
        self._models = cost_models
        self._n_gpus = int(n_gpus)
        # Descending, so the enumeration reaches concentrated plans first and the
        # tie-break below keeps the simplest plan among equals.
        self._tp_candidates = tuple(sorted({int(tp) for tp in tp_candidates}, reverse=True))
        self._strategy = strategy
        self._max_concurrency = max_concurrency
        self._active_cap = active_cap
        self._objective = objective
        self._training_cycle_s = float(training_cycle_s)
        self._future_discount = float(future_discount)
        self._drain_weight = float(drain_weight)
        self._fixed = tuple(int(tp) for tp in fixed_tp_per_instance)
        if strategy == "fixed":
            # Validated here rather than at the first solve: a misspent fixed plan is a
            # configuration error, and failing at construction names it before a
            # simulation has run for minutes.
            FleetPlan(tp_per_instance=self._fixed, n_gpus=self._n_gpus)
        self._legal_plans = self._enumerate_plans()
        self.last_scores: dict[tuple[int, ...], float] = {}
        """Score of every candidate at the most recent solve, for telemetry."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def observe_training_cycle(self, seconds: float) -> None:
        """Record the measured duration of the last training cycle.

        The SCG objective drains the old-version tail against this deadline, so a
        stale estimate mis-scores every candidate in the same direction. Called by
        the simulator after each training step rather than configured once,
        because the real cycle time depends on the batch that was actually taken.

        Args:
            seconds: Wall-clock seconds the last cycle took. Non-positive values
                are ignored, so a degenerate first step cannot zero the deadline.
        """
        if seconds > 0.0:
            self._training_cycle_s = float(seconds)

    def solve(
        self,
        snapshot: list[RequestSnapshot] | list[RichRequestSnapshot],
        rollout_version: int,
    ) -> FleetPlan:
        """
        Return the fleet layout for the next rollout interval.

        Args:
            snapshot: In-flight requests, as observable projections. The SCG
                objective needs `RichRequestSnapshot` (carrying ``remaining``);
                given plain `RequestSnapshot` it falls back to the throughput
                objective, because without a remaining length there is no
                completion time to compare against the deadline.
            rollout_version: The version just published. Requests below it are the
                old-version tail.

        Returns:
            A plan spending the GPU budget exactly.
        """
        if self._strategy == "fixed":
            return FleetPlan(tp_per_instance=self._fixed, n_gpus=self._n_gpus)
        use_scg = self._objective == "scg" and all(
            hasattr(r, "remaining") for r in snapshot
        )
        if use_scg:
            return self._search_scg(snapshot, rollout_version)
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

    def _search_scg(
        self, snapshot: list[RichRequestSnapshot], rollout_version: int
    ) -> FleetPlan:
        """Score every legal plan by expected SCG completions and return the best.

        This is the objective the paper's §9 argues for. The difference from
        `_search` is not a tweak to the weighting: aggregate token rate always
        prefers replication (more instances means more concurrency), while SCG
        rewards the old-version tail finishing inside the training cycle, which
        a lower per-step latency buys. The two objectives therefore disagree in a
        way no amount of parameter tuning on the throughput form could reproduce.

        Args:
            snapshot: In-flight requests, carrying `remaining`.
            rollout_version: The version just published.

        Returns:
            A plan spending the GPU budget exactly.
        """
        from .scg_value import TrajSnapshot, score_plan_scg

        old = [
            TrajSnapshot(v_traj=r.v_traj, ctx=r.ctx, remaining=max(1, r.remaining))
            for r in snapshot
            if r.v_traj < rollout_version
        ]
        new = [
            TrajSnapshot(v_traj=r.v_traj, ctx=r.ctx, remaining=max(1, r.remaining))
            for r in snapshot
            if r.v_traj >= rollout_version
        ]

        self.last_scores = {}
        best_key: tuple[float, int] | None = None
        best: tuple[int, ...] = self._legal_plans[0]
        for tp_per_instance in self._legal_plans:
            score = score_plan_scg(
                tp_per_instance=tp_per_instance,
                models=self._models,
                old_trajs=old,
                new_trajs=new,
                T_tr=self._training_cycle_s,
                max_concurrency=self._max_concurrency,
                active_cap=self._active_cap,
                future_discount=self._future_discount,
                drain_weight=self._drain_weight,
            )
            self.last_scores[tp_per_instance] = score
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

        Mirrors the router's version tiers and active cap. Stale work uses the
        highest TP tier. Current work stays on the lowest tier while stale work
        exceeds the high-tier cap, then balances across the fleet.

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
            if self._active_cap is not None:
                batch = min(batch, self._active_cap)
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
        max_tp = max(tp_per_instance)
        min_tp = min(tp_per_instance)
        high = [index for index, tp in enumerate(tp_per_instance) if tp == max_tp]
        low = [index for index, tp in enumerate(tp_per_instance) if tp == min_tp]
        old_counts = [0] * n
        _spread_to_targets(old_counts, r_old, high)

        high_capacity = None
        if self._active_cap is not None:
            high_capacity = self._active_cap * len(high)
        stale_backlog = high_capacity is not None and r_old > high_capacity
        current_targets = low if stale_backlog and min_tp < max_tp else list(range(n))
        new_counts = [0] * n
        load = list(old_counts)
        for _ in range(r_new):
            target = min(current_targets, key=lambda index: (load[index], index))
            new_counts[target] += 1
            load[target] += 1
        return list(zip(old_counts, new_counts))


def _spread_to_targets(
    counts: list[int],
    total: int,
    targets: list[int],
) -> None:
    """Add `total` requests evenly across selected count buckets."""
    if total <= 0 or not targets:
        return
    n = len(targets)
    base, extra = divmod(total, n)
    for position, index in enumerate(targets):
        counts[index] += base + (1 if position < extra else 0)


def _mean_ctx(requests: list[RequestSnapshot]) -> int:
    """Return the mean context depth, or 1 when the set is empty."""
    if not requests:
        return 1
    return max(1, sum(r.ctx for r in requests) // len(requests))
