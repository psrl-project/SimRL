"""Dynamic heterogeneous parallelism for the rollout fleet.

Infrastructure only, nothing in `Simulator` constructs it yet. A fleet is a flat
list of instances and a plan carries one TP degree per instance.

- `FleetPlan` (`plan.py`) records the TP degree of each instance and the GPU budget
  invariant.
- `DynamicParallelFleet` (`fleet.py`) applies a plan at a weight update, when no
  KV state can survive the new weights.
"""

from __future__ import annotations

from .fleet import DynamicParallelFleet
from .plan import FleetPlan

__all__ = [
    "DynamicParallelFleet",
    "FleetPlan",
]
