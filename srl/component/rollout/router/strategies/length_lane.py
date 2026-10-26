"""Confine long requests to a fixed set of instances behind a length threshold.

The static ablation for `slack_lane`. The fast set is the first `n_fast` instance ids, fixed
for the whole run, and membership is a request's own predicted total length against one
threshold. Nothing is replanned and nothing is priced, so a result matching `slack_lane`
means the planning and the slack signal bought nothing a fixed threshold does not.
"""

from __future__ import annotations

from .....forecast.length import build_length_predictor
from ...engine import InstanceView, Request, traj_view
from ..strategy import register_strategy


@register_strategy("length_lane")
class LengthLane:
    """Static fast instances for requests predicted to be long."""

    name = "length_lane"

    def __init__(
        self,
        n_fast: int,
        threshold: float,
        c_fast: int,
        length_model: str = "oracle",
        length_model_args: dict | None = None,
    ) -> None:
        """
        Args:
            n_fast (int): Instances reserved for long requests, taken as instance ids
                `0 .. n_fast - 1`.
            threshold (float): A request predicted to need at least this many total tokens
                prefers the fast instances.
            c_fast (int): Requests one fast instance may hold before it is treated as full.
            length_model (str): Name in `LENGTH_PREDICTORS`.
            length_model_args (dict | None): Constructor keywords for that predictor.

        Raises:
            ValueError: If `n_fast` is below 1, or `c_fast` is below 1.
        """
        if n_fast < 1:
            raise ValueError(
                f"n_fast must be at least 1, and must still leave at least one bulk instance, got {n_fast!r}."
            )
        if c_fast < 1:
            raise ValueError(f"c_fast must be at least 1, got {c_fast!r}.")
        self.n_fast = int(n_fast)
        self.threshold = float(threshold)
        self.c_fast = int(c_fast)
        self.predictor = build_length_predictor(length_model, **(length_model_args or {}))

    def narrow_candidates(self, req: Request, candidates: list[InstanceView], _context) -> list[int]:
        """Confine a predicted long request to its fast lanes, else keep it off them.

        Raises:
            ValueError: If `candidates` holds no fast instance or no bulk instance.
        """
        fast = [i for i, view in enumerate(candidates) if view.instance_id < self.n_fast]
        bulk = [i for i, view in enumerate(candidates) if view.instance_id >= self.n_fast]
        if not fast or not bulk:
            raise ValueError(
                f"length_lane needs at least one bulk instance beyond its {self.n_fast} fast ones, "
                f"and the fleet offered {len(candidates)} candidates."
            )
        if self.predictor.predict_total(traj_view(req)) < self.threshold:
            return bulk
        room = [i for i in fast if len(candidates[i].residents()) < self.c_fast]
        return room or bulk

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        """Take the candidate with the fewest predicted tokens left."""
        return min(
            range(len(candidates)),
            key=lambda i: (self._remaining_load(candidates[i]), candidates[i].n_active),
        )

    def _remaining_load(self, view: InstanceView) -> float:
        total = 0.0
        for traj in view.residents():
            total += max(0.0, self.predictor.predict_total(traj) - traj.generated)
        return total
