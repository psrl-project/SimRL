"""What a scheduler is allowed to know about a trajectory, and how long it will run.

`cost/` predicts time. This package predicts trajectories, and `LengthModel` is the
only modelled quantity in it: everything else is arithmetic on top.

A trajectory that has generated `l` tokens and has not stopped is known to satisfy
`L > l`, so the remaining-length distribution is the conditional one:

    V = P(L <= l + B | L > l) = [F(l + B) - F(l)] / [1 - F(l)]

Conditioning is what makes the answer depend on how far along a request is. Without
the division a freshly admitted request and one nine thousand tokens in score the
same, and any ordering built on it is meaningless.

The per-turn hazard product of the design document is this same number. The product
telescopes: with `R_b` trajectories reaching bucket `b` and `N_b` stopping in it,
`prod (R_b - N_b)/R_b = R_last/R_first`, which is the ratio above. The hazard form is
a bookkeeping device for online estimation under per-bucket censoring, and a complete
single-turn trace needs none of that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TrajView:
    """A read-only projection of one trajectory, the unit every model reads.

    Strategies and length models receive this rather than a `Request` so they
    cannot mutate the engine, and so the same code runs against either backend.

    `target_len` is oracle information. An observable model must not read it, which
    `tests/test_value_length.py` asserts by varying it and requiring the answer to
    be unchanged.
    """

    generated: int
    prompt_len: int
    target_len: int
    prompt_key: int
    v_traj: int
    admit_time: float

    @property
    def remaining(self) -> int:
        return max(0, self.target_len - self.generated)

    @property
    def ctx(self) -> int:
        """Context this trajectory occupies, the depth a decode step pays for."""
        return self.prompt_len + self.generated


def traj_view(req) -> TrajView:
    """Project a `Request` or its C++ equivalent, which share these field names."""
    return TrajView(
        generated=req.generated,
        prompt_len=req.prompt_len,
        target_len=req.target_len,
        prompt_key=req.source_group_id,
        v_traj=req.v_traj,
        admit_time=req.admit_time,
    )


class LengthModel(Protocol):
    """Predicts whether a trajectory finishes inside a token budget."""

    name: str

    @property
    def ready(self) -> bool:
        """Whether this model has enough evidence to be worth consulting.

        A model that answers before it has learned anything does not degrade
        gracefully, it degrades silently. Callers admit freely while this is False,
        which is the design document's safety property: a failed predictor falls
        back to fair sharing rather than mis-ordering.
        """
        ...

    def hit_prob(self, traj: TrajView, budget: float) -> float:
        """Return `P(this trajectory finishes within budget tokens)`, in [0, 1]."""
        ...

    def expected_final_length(self, traj: TrajView) -> float:
        """Return the total length this trajectory is predicted to reach.

        What a KV footprint is sized on. A request holds only its prompt when it
        is admitted, so the blocks it will eventually need are a prediction, and
        comparing the present depth against pool capacity reports room that the
        batch is already committed to consuming.
        """
        ...

    def observe(self, traj: TrajView, final_length: int) -> None:
        """Record a finished trajectory."""

    def decay(self, factor: float) -> None:
        """Discount accumulated evidence, so the model tracks a drifting workload."""
