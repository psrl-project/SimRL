"""Dynamic heterogeneous parallelism for the rollout fleet.

Opt-in and inert by default. Nothing in this package runs unless
``dynamic_parallel.enabled`` is set, in which case `Simulator` rebuilds its
instance list at every weight update instead of holding one fixed shape.

**The fleet is a flat list of instances.** A plan carries one TP degree per
instance. `VersionCappedTp` assigns stale work to the highest TP tier under a
shared active cap. Current work uses lower TP instances while stale work remains
queued, then fills any remaining capacity across the fleet.

Three pieces:

- `FleetPlan` (`plan.py`) records the TP degree of each instance and the GPU budget
  invariant.
- `ParallelismSolver` (`solver.py`) picks a plan. It scores candidates by
  modelling what the router will actually do with them, rather than assuming a
  partition the router does not implement.
- `DynamicParallelFleet` (`fleet.py`) applies a plan at a weight update, when no
  KV state can survive the new weights.
"""

from __future__ import annotations

from .fleet import DynamicParallelFleet
from .plan import FleetPlan
from .planner import PlannerSolver, TierPlan, TierPlanner
from .solver import ParallelismSolver, RequestSnapshot, RichRequestSnapshot

__all__ = [
    "DynamicParallelFleet",
    "FleetPlan",
    "PlannerSolver",
    "TierPlan",
    "TierPlanner",
    "ParallelismSolver",
    "RequestSnapshot",
    "RichRequestSnapshot",
]
