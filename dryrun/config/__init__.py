"""Hydra configuration for DryRun-RL."""

from hydra.core.config_store import ConfigStore

from .schema import (
    AdmissionControlConfig,
    DistributionConfig,
    JobConfig,
    JointTraceConfig,
    ParallelismConfig,
    PolicyConfig,
    RecomputeCostConfig,
    RolloutConfig,
    RolloutCostConfig,
    SimulateConfig,
    SyncCostConfig,
    TrainingCostConfig,
    WorkloadConfig,
)

__all__ = [
    "AdmissionControlConfig",
    "DistributionConfig",
    "JobConfig",
    "JointTraceConfig",
    "ParallelismConfig",
    "PolicyConfig",
    "RecomputeCostConfig",
    "RolloutConfig",
    "RolloutCostConfig",
    "SimulateConfig",
    "SyncCostConfig",
    "TrainingCostConfig",
    "WorkloadConfig",
    "register_configs",
]


def register_configs() -> None:
    """Register structured configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="simulate_schema", node=SimulateConfig)
    cs.store(group="rollout_cost", name="roofline_schema", node=RolloutCostConfig)
    cs.store(group="policy", name="psrl_schema", node=PolicyConfig)
    cs.store(group="workload", name="workload_schema", node=WorkloadConfig)
    cs.store(group="training_cost", name="fixed_schema", node=TrainingCostConfig)
    cs.store(group="sync_cost", name="fixed_schema", node=SyncCostConfig)
    cs.store(group="recompute_cost", name="fixed_schema", node=RecomputeCostConfig)
