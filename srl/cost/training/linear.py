"""Linear token-time and inverse-GPU memory training cost models."""

from __future__ import annotations

from .base import (
    TrainingCostModel,
    TrainingLatencyModel,
    TrainingMemoryModel,
    TrainingParallelism,
    TrainingWorkload,
)


class LinearTrainingLatency(TrainingLatencyModel):
    """Scale training step time linearly with the global token count."""

    def __init__(self, time_per_token: float):
        if time_per_token <= 0:
            raise ValueError("time_per_token must be positive.")
        self.time_per_token = time_per_token

    def predict(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> float:
        return self.time_per_token * workload.total_tokens


class LinearTrainingMemory(TrainingMemoryModel):
    """Scale peak memory inversely with allocated GPU count."""

    def __init__(self, base_memory_bytes: int = 0):
        if base_memory_bytes < 0:
            raise ValueError("base_memory_bytes must be non-negative.")
        self.base_memory_bytes = base_memory_bytes

    def predict(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> int:
        return int(self.base_memory_bytes / parallelism.world_size)


class LinearTrainingCost(TrainingCostModel):
    """Compose linear token latency and inverse-GPU memory models."""

    def __init__(self, time_per_token: float, base_memory_bytes: int = 0):
        super().__init__(
            LinearTrainingLatency(time_per_token),
            LinearTrainingMemory(base_memory_bytes),
        )
