"""Keep short arrivals off an instance that already hosts a long-tail resident.

`inventory_deadline`'s buffer-pipeline forecast was checked and found structurally
incapable of ever choosing isolation: its comparison ranks candidates by whichever
buffer fills soonest, so sacrificing near-term throughput -- holding an instance
back from the far more numerous short arrivals -- to protect one long-tail group's
own, much later buffer is a trade that objective can never credit, no matter how
far ahead or how jointly it looks. This strategy skips that forecast machinery
entirely and encodes the trade directly: an instance currently hosting a resident
predicted to run past `tail_length` stays closed to new short arrivals as long as
that resident is still there, so the two kinds of work stop sharing a decode
step's token budget.

Which instances end up reserved, and how many, is never configured -- it falls out
of how many long-tail residents are concurrently active and the engine's own KV
admission, the same way `length_aware`'s load balance picks instances without a
hand-picked id. A long-tail arrival itself prefers an instance already closed to
short arrivals, so a run of them consolidates onto as few instances as fit rather
than opening a new one each time.
"""

from __future__ import annotations

from .....forecast.length import build_length_predictor
from ...engine import InstanceView, Request, traj_view
from ..strategy import register_strategy


@register_strategy("tail_reserve")
class TailReserve:
    """Load-balanced placement, with long-tail residents closing their instance to new
    short arrivals and consolidating onto an instance already closed."""

    name = "tail_reserve"

    def __init__(
        self,
        tail_length: float = 100_000.0,
        length_model: str = "oracle",
        length_model_args: dict | None = None,
    ) -> None:
        """
        Args:
            tail_length (float): Predicted-length threshold above which a resident counts
                as long-tail and closes its instance to new short arrivals.
            length_model (str): Name in `LENGTH_PREDICTORS`.
            length_model_args (dict | None): Constructor keywords for that predictor.
        """
        if tail_length <= 0.0:
            raise ValueError(f"tail_length must be > 0, got {tail_length!r}.")
        self.tail_length = float(tail_length)
        self.predictor = build_length_predictor(length_model, **(length_model_args or {}))

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        if self.predictor.predict_total(traj_view(req)) > self.tail_length:
            return min(
                range(len(candidates)),
                key=lambda i: (
                    not self._hosts_tail(candidates[i]),
                    self._remaining_load(candidates[i]),
                    candidates[i].n_active,
                ),
            )
        open_indices = [i for i in range(len(candidates)) if not self._hosts_tail(candidates[i])]
        pool = open_indices or range(len(candidates))
        return min(
            pool,
            key=lambda i: (self._remaining_load(candidates[i]), candidates[i].n_active),
        )

    def _hosts_tail(self, view: InstanceView) -> bool:
        return any(self.predictor.predict_total(traj) > self.tail_length for traj in view.residents())

    def _remaining_load(self, view: InstanceView) -> float:
        """Predicted tokens the instance's residents still have to generate."""
        total = 0.0
        for traj in view.residents():
            total += max(0.0, self.predictor.predict_total(traj) - traj.generated)
        return total
