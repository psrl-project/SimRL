"""Routes a request into one of two TP tiers by a fixed length threshold.

Where `length_aware.py` balances by load across a homogeneous fleet, this strategy
assumes the fleet itself is heterogeneous (some instances at `tail_tp`, the rest at
`bulk_tp`, e.g. a static `srl.simulator.dynamic_parallel.SlotFleet` awake set, or the
same fleet reshaping mid-round under a `ReshapePlanner`) and decides which tier a
request may land in. The tier membership is read off each candidate's own `tp` field,
not a separate instance-id list, so the strategy stays correct no matter which
concrete slots are currently awake at that degree.

The split is a request's own predicted total length against one threshold, not a
group's slowest member and not a replanned boundary: simplicity is deliberate here.
TierPlanner's three solvers (tested and deprecated, see
`deprecated/srl/simulator/dynamic_parallel/planner.py`) already explored boundary
search and replanning, and a deployable (non-oracle) length predictor lost in every
one of those -- a fixed threshold cannot do better with a bad predictor, but it also
cannot compound a boundary-search error on top of a prediction error.
"""

from __future__ import annotations

from .....forecast.length import build_length_predictor
from ...engine import InstanceView, Request, traj_view
from ..strategy import register_strategy


@register_strategy("tier_threshold")
class TierThreshold:
    """Fixed length-threshold placement across a heterogeneous (tail_tp, bulk_tp) fleet."""

    name = "tier_threshold"

    def __init__(
        self,
        tail_tp: int,
        bulk_tp: int,
        tail_threshold: float,
        length_model: str = "oracle",
        length_model_args: dict | None = None,
    ) -> None:
        """
        Args:
            tail_tp (int): TP degree of the tail pool's instances.
            bulk_tp (int): TP degree of the bulk pool's instances.
            tail_threshold (float): A request predicted to need at least this many
                total tokens is routed only to `tail_tp` instances; everything else
                is routed only to `bulk_tp` instances.
            length_model (str): Name in `LENGTH_PREDICTORS`.
            length_model_args (dict | None): Constructor keywords for that predictor.
        """
        if tail_tp == bulk_tp:
            raise ValueError(f"tail_tp and bulk_tp must differ, got {tail_tp} twice.")
        self.tail_tp = tail_tp
        self.bulk_tp = bulk_tp
        self.tail_threshold = float(tail_threshold)
        self.predictor = build_length_predictor(length_model, **(length_model_args or {}))

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        predicted_total = self.predictor.predict_total(traj_view(req))
        wanted_tp = self.tail_tp if predicted_total >= self.tail_threshold else self.bulk_tp
        indices = [i for i in range(len(candidates)) if candidates[i].tp == wanted_tp]
        if not indices:
            # The wanted tier is absent from this fleet (e.g. a homogeneous control
            # run reusing this strategy by mistake): fall back to every candidate
            # rather than refusing every request of that class outright.
            indices = list(range(len(candidates)))
        return min(indices, key=lambda i: (self._remaining_load(candidates[i]), candidates[i].n_active))

    def _remaining_load(self, view: InstanceView) -> float:
        """Predicted tokens the instance's residents still have to generate."""
        total = 0.0
        for traj in view.residents():
            total += max(0.0, self.predictor.predict_total(traj) - traj.generated)
        return total
