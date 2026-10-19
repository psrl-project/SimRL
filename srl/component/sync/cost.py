"""Sync cost models.

After a training step produces new weights, they must be distributed to
all rollout instances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class SyncCostModel(ABC):
    """Weight synchronization latency model."""

    @abstractmethod
    def sync_time(self, n_instances: int) -> float:
        """Time to distribute updated weights to all rollout instances."""


class FixedSyncCost(SyncCostModel):
    """Fixed synchronization time."""

    def __init__(self, sync_time: float):
        self.sync_time_val = sync_time

    def sync_time(self, n_instances: int) -> float:
        return self.sync_time_val
