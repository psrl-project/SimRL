"""Constant training latency and memory models."""

from __future__ import annotations

from .base import (
    TrainingCostModel,
    TrainingLatencyModel,
    TrainingMemoryModel,
    TrainingParallelism,
    TrainingWorkload,
)


class FixedTrainingLatency(TrainingLatencyModel):
    """Return one fixed training step time."""

    def __init__(self, latency_s: float):
        if latency_s <= 0:
            raise ValueError("latency_s must be positive.")
        self.latency_s = latency_s

    def predict(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> float:
        return self.latency_s


class FixedTrainingMemory(TrainingMemoryModel):
    """Return one fixed peak memory value."""

    def __init__(self, peak_memory_bytes: int = 0):
        if peak_memory_bytes < 0:
            raise ValueError("peak_memory_bytes must be non-negative.")
        self.peak_memory_bytes = peak_memory_bytes

    def predict(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> int:
        return self.peak_memory_bytes


class FixedTrainingCost(TrainingCostModel):
    """Compose constant latency and memory models."""

    def __init__(self, latency_s: float, peak_memory_bytes: int = 0):
        super().__init__(
            FixedTrainingLatency(latency_s),
            FixedTrainingMemory(peak_memory_bytes),
        )
