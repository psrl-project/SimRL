"""Router: where a request waits before an instance takes it, and who takes it.

The simulator admits into the router, not onto an instance, so a request no
instance can accept dwells here rather than being rolled back. Three concerns,
deliberately separate:

- `RouterQueue` orders what is waiting.
- `RoutingStrategy` picks among instances that could take a request.
- A strategy may restrict candidates and allow dispatch to bypass a blocked
  version queue.
- `Router` owns the queue, applies the `AdmissionGate`, and runs dispatch.

Importing this package registers every built-in strategy, which is the only
registration path.
"""

from . import strategies as _strategies  # noqa: F401  (populates STRATEGY_REGISTRY)
from .request_queue import RouterQueue, SortKey
from .router import AdmissionGate, Router
from .strategy import STRATEGY_REGISTRY, RoutingContext, RoutingStrategy, build_strategy, register_strategy

__all__ = [
    "STRATEGY_REGISTRY",
    "AdmissionGate",
    "Router",
    "RouterQueue",
    "RoutingContext",
    "RoutingStrategy",
    "SortKey",
    "build_strategy",
    "register_strategy",
]
