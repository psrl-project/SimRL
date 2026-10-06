"""Dynamic heterogeneous parallelism for the rollout fleet.

Opt-in and inert by default. Nothing in this package runs unless
``dynamic_parallel.enabled`` is set, in which case `Simulator` rebuilds its
instance list at every weight update instead of holding one fixed shape.

**The fleet is a flat list of instances, not a set of partitions.** A plan is
just a TP degree per instance; there is no grouping, no membership predicate and
no per-group quota. Placement is entirely the router's business, expressed as a
*soft preference* over instances (see
``component/rollout/router/strategies.py::VersionAwareTp``).

That split is deliberate. An earlier design made a plan carry version
predicates, so a high-TP group could only accept trajectories from before the
last weight update. It load-balances badly for exactly the reason the mechanism
exists: once the old-version tail drains to a handful of trajectories, the
high-TP instances have nothing left to admit and idle while the low-TP ones
queue. Keeping placement soft lets fresh work fill that capacity, and the router
already holds requests that no instance can take yet, so anything refused now is
simply re-offered at the next release point.

Three pieces:

- `FleetPlan` (`plan.py`) -- the TP degree of each instance, plus the GPU budget
  invariant.
- `ParallelismSolver` (`solver.py`) -- picks a plan. Scores candidates by
  modelling what the router will actually do with them, rather than assuming a
  partition the router does not implement.
- `DynamicParallelFleet` (`fleet.py`) -- applies a plan at a weight update,
  which is the one moment it is free: the update invalidates every KV block
  anyway, so rebuilding costs only scheduling metadata.
"""

from __future__ import annotations

from .fleet import DynamicParallelFleet
from .plan import FleetPlan
from .solver import ParallelismSolver, RequestSnapshot

__all__ = [
    "DynamicParallelFleet",
    "FleetPlan",
    "ParallelismSolver",
    "RequestSnapshot",
]
