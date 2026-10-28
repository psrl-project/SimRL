"""Dynamic heterogeneous parallelism for the rollout fleet.

- `SlotTable` (`plan.py`) is the fixed catalogue of every TP shape a run may use: each `Slot`
  reserves a GPU-disjoint window and never changes degree.
- `SlotFleet` (`fleet.py`) owns one instance per slot for the whole run and reshapes which
  GPU-disjoint subset is awake, mid-round, at a completion event -- not only at a weight update.
- `ReshapePlanner` / `ResidentThresholdPlanner` (`planner.py`) decide the target awake set from
  the fleet's current occupancy.

Wired into `Simulator` via `SimConfig.dynamic_parallel` (`native` backend only).
"""

from __future__ import annotations

from .fleet import SlotFleet
from .plan import DynamicParallelConfig, Slot, SlotTable
from .planner import ReshapePlanner, ReshapeState, ResidentThresholdPlanner, build_reshape_planner

__all__ = [
    "DynamicParallelConfig",
    "ReshapePlanner",
    "ReshapeState",
    "ResidentThresholdPlanner",
    "Slot",
    "SlotFleet",
    "SlotTable",
    "build_reshape_planner",
]
