"""Sync cost models.

After a training step produces new weights, they must be distributed to
all rollout instances. The cost depends on model size, number of instances,
and available interconnect bandwidth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class SyncCostModel(ABC):
    """
    Weight synchronization latency model.

    After a training step produces new weights, they must be distributed to
    all rollout instances. The cost depends on model size, number of
    instances, and available interconnect bandwidth.
    """

    @abstractmethod
    def sync_time(self, model_size_bytes: int, n_instances: int, bandwidth_gbps: float) -> float:
        """
        Time to distribute updated weights to all rollout instances.

        Args:
            model_size_bytes: Size of the model parameters in bytes.
            n_instances: Number of rollout instances to update.
            bandwidth_gbps: Available network bandwidth in Gbps.
        """


class FixedSyncCost(SyncCostModel):
    """Fixed synchronization time."""

    def __init__(self, sync_time: float):
        self.sync_time_val = sync_time

    def sync_time(self, model_size_bytes: int, n_instances: int, bandwidth_gbps: float) -> float:
        return self.sync_time_val


class BandwidthSyncCost(SyncCostModel):
    """Bandwidth-based sync time: model_size / bandwidth."""

    def sync_time(self, model_size_bytes: int, n_instances: int, bandwidth_gbps: float) -> float:
        bandwidth_bytes_per_sec = bandwidth_gbps * 1e9 / 8
        return model_size_bytes / bandwidth_bytes_per_sec
