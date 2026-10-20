"""What a fleet layout is: one TP degree per instance, and nothing else.

`FleetPlan` deliberately carries no grouping and no routing predicate. It says
how many instances exist and what TP degree each runs at. Which request lands
where is the router's decision, re-made at every dispatch. See the package
docstring for why the earlier predicate-carrying design was dropped.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class FleetPlan:
    """The TP degree of every rollout instance, in instance-id order."""

    tp_per_instance: tuple[int, ...]
    """``tp_per_instance[i]`` is the TP degree of instance ``i``.

    A flat tuple rather than ``(tp, count)`` pairs because instance id is the
    only identity the router and the engine loop share: `Simulator` indexes
    `self.instances[iid]` directly, so position *is* identity here.
    """
    n_gpus: int
    """GPU budget the plan must spend exactly."""

    def __post_init__(self) -> None:
        if not self.tp_per_instance:
            raise ValueError("A fleet plan needs at least one instance.")
        if any(tp <= 0 for tp in self.tp_per_instance):
            raise ValueError(
                f"Every TP degree must be positive, got {self.tp_per_instance!r}."
            )
        used = sum(self.tp_per_instance)
        if used != self.n_gpus:
            # Raised rather than asserted: a plan that overspends or underspends is a
            # solver arithmetic error, and letting it through would silently simulate a
            # fleet of a different size than the operator configured.
            raise ValueError(
                f"GPU budget violated: {self.tp_per_instance} consumes {used} "
                f"GPU-ranks but n_gpus={self.n_gpus}."
            )

    @property
    def n_instances(self) -> int:
        """Return how many instances this plan describes."""
        return len(self.tp_per_instance)

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the plan mixes TP degrees.

        A plan of four TP=1 instances is homogeneous even though it has four
        instances. The distinction matters when reporting whether dynamic
        heterogeneity actually engaged, because a uniform plan is what a static
        configuration would have produced anyway.
        """
        return len(set(self.tp_per_instance)) > 1

    @property
    def tp_histogram(self) -> dict[int, int]:
        """Return ``{tp degree: instance count}``, for telemetry and logging."""
        return dict(sorted(Counter(self.tp_per_instance).items()))

    def describe(self) -> str:
        """Return a compact human-readable shape such as ``1xTP2 + 2xTP1``."""
        parts = [
            f"{count}xTP{tp}" for tp, count in sorted(self.tp_histogram.items(), reverse=True)
        ]
        return " + ".join(parts)

    @classmethod
    def uniform(cls, tp: int, n_gpus: int) -> FleetPlan:
        """Build the homogeneous plan that spends `n_gpus` at a single TP degree.

        Args:
            tp: TP degree for every instance.
            n_gpus: Total GPU budget. Must be divisible by `tp`.

        Returns:
            A plan of ``n_gpus // tp`` instances.

        Raises:
            ValueError: If `tp` does not divide `n_gpus`.
        """
        if tp <= 0:
            raise ValueError(f"TP degree must be positive, got {tp!r}.")
        if n_gpus % tp != 0:
            raise ValueError(
                f"A uniform TP={tp} fleet cannot spend {n_gpus} GPUs exactly."
            )
        return cls(tp_per_instance=(tp,) * (n_gpus // tp), n_gpus=n_gpus)
