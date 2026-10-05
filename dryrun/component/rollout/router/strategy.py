"""What a routing strategy is, and how one is looked up by name.

A strategy is pure selection: given one request and the instances that could
take it, say which. It does not queue, does not check capacity, and does not
mutate anything. The admission gate has already removed ineligible candidates
before a strategy sees them, so a strategy that returns an index is promising
only a preference, never an admission.

That split is why adding a strategy means adding one class here and one
registry entry, and touching no dispatch code. It is the same boundary SMG
draws between its `LoadBalancingPolicy` trait and its admission gate.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from ..engine import InstanceView, Request


@runtime_checkable
class RoutingStrategy(Protocol):
    """Picks one instance for one request."""

    name: str

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        """Choose among `candidates`.

        Args:
            req: The request to place.
            candidates: Instances that passed the admission gate, in instance
                id order. Never empty: the caller skips the strategy entirely
                when nothing is eligible.

        Returns:
            An index into `candidates`, or None to leave the request queued.
        """
        ...


STRATEGY_REGISTRY: dict[str, Callable[..., RoutingStrategy]] = {}
"""Strategy name -> constructor. Populated by `register_strategy`."""


def register_strategy(name: str) -> Callable[[type], type]:
    """Class decorator adding a strategy to the registry under `name`.

    Raises:
        ValueError: If `name` is already registered, so two strategies cannot
            silently shadow each other.
    """

    def wrap(cls: type) -> type:
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"Routing strategy {name!r} is already registered.")
        STRATEGY_REGISTRY[name] = cls
        cls.name = name
        return cls

    return wrap


def build_strategy(name: str, **kwargs) -> RoutingStrategy:
    """Construct the registered strategy called `name`.

    Raises:
        KeyError: If no strategy is registered under that name, listing the
            ones that are.
    """
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown routing strategy {name!r}. Available: {sorted(STRATEGY_REGISTRY)}."
        )
    return STRATEGY_REGISTRY[name](**kwargs)
