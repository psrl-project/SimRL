"""Compute and bandwidth roofline training cost models."""

from __future__ import annotations

from .base import (
    TrainingCostModel,
    TrainingLatencyModel,
    TrainingMemoryModel,
    TrainingParallelism,
    TrainingWorkload,
)


class AnalyticalTrainingLatency(TrainingLatencyModel):
    """Use a simple compute and bandwidth roofline over all allocated GPUs."""

    def __init__(
        self,
        *,
        model_params: int,
        peak_flops: float,
        memory_bandwidth: float,
        model_bytes: int | None = None,
        overhead_factor: float = 1.1,
    ):
        if model_params <= 0:
            raise ValueError("model_params must be positive.")
        if peak_flops <= 0:
            raise ValueError("peak_flops must be positive.")
        if memory_bandwidth <= 0:
            raise ValueError("memory_bandwidth must be positive.")
        if overhead_factor <= 0:
            raise ValueError("overhead_factor must be positive.")
        self.model_params = model_params
        self.peak_flops = peak_flops
        self.memory_bandwidth = memory_bandwidth
        self.model_bytes = model_bytes or model_params * 2
        self.overhead_factor = overhead_factor

    def predict(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> float:
        compute_time = (
            6
            * self.model_params
            * workload.total_tokens
            / (parallelism.world_size * self.peak_flops)
        )
        memory_time = self.model_bytes / (
            parallelism.tp * parallelism.pp * self.memory_bandwidth
        )
        return max(compute_time, memory_time) * self.overhead_factor


class AnalyticalTrainingMemory(TrainingMemoryModel):
    """Estimate static model state memory without fitted activations."""

    def __init__(
        self,
        *,
        model_params: int,
        bytes_per_parameter: float = 16.0,
    ):
        if model_params <= 0:
            raise ValueError("model_params must be positive.")
        if bytes_per_parameter <= 0:
            raise ValueError("bytes_per_parameter must be positive.")
        self.model_params = model_params
        self.bytes_per_parameter = bytes_per_parameter

    def predict(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> int:
        shard = parallelism.tp * parallelism.pp
        return int(self.model_params * self.bytes_per_parameter / shard)


class AnalyticalTrainingCost(TrainingCostModel):
    """Compose analytical roofline latency and static-state memory models."""

    def __init__(
        self,
        *,
        model_params: int,
        peak_flops: float,
        memory_bandwidth: float,
        model_bytes: int | None = None,
        overhead_factor: float = 1.1,
        bytes_per_parameter: float = 16.0,
    ):
        super().__init__(
            AnalyticalTrainingLatency(
                model_params=model_params,
                peak_flops=peak_flops,
                memory_bandwidth=memory_bandwidth,
                model_bytes=model_bytes,
                overhead_factor=overhead_factor,
            ),
            AnalyticalTrainingMemory(
                model_params=model_params,
                bytes_per_parameter=bytes_per_parameter,
            ),
        )
