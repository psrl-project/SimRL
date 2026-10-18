"""Hydra configuration for SimRL."""

from dataclasses import asdict

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .policies import (
    PRESETS,
    ArealPolicyConfig,
    PsrlPolicyConfig,
    RollPolicyConfig,
    SkyRLPolicyConfig,
    SlimePolicyConfig,
    VerlPolicyConfig,
    VerlV1PolicyConfig,
)
from .schema import (
    AdmissionControlConfig,
    DistributionConfig,
    JobConfig,
    JointTraceConfig,
    ParallelismConfig,
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
    "ArealPolicyConfig",
    "DistributionConfig",
    "JobConfig",
    "JointTraceConfig",
    "ParallelismConfig",
    "PsrlPolicyConfig",
    "RecomputeCostConfig",
    "RollPolicyConfig",
    "RolloutConfig",
    "RolloutCostConfig",
    "SimulateConfig",
    "SkyRLPolicyConfig",
    "SlimePolicyConfig",
    "SyncCostConfig",
    "TrainingCostConfig",
    "VerlPolicyConfig",
    "VerlV1PolicyConfig",
    "WorkloadConfig",
    "register_configs",
]


def _simulate_schema():
    """Return the primary schema with an untyped `policy`, so a group can replace it."""
    schema = OmegaConf.structured(SimulateConfig)
    schema.policy = OmegaConf.create(asdict(PsrlPolicyConfig()))
    return schema


def register_configs() -> None:
    """Register structured configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="simulate_schema", node=_simulate_schema())
    cs.store(group="rollout_cost", name="roofline_schema", node=RolloutCostConfig)
    for name, cls in PRESETS.items():
        cs.store(group="policy", name=f"{name}_schema", node=cls)
        cs.store(group="policy", name=name, node=cls)
    cs.store(group="workload", name="workload_schema", node=WorkloadConfig)
    cs.store(group="training_cost", name="fixed_schema", node=TrainingCostConfig)
    cs.store(group="sync_cost", name="fixed_schema", node=SyncCostConfig)
    cs.store(group="recompute_cost", name="fixed_schema", node=RecomputeCostConfig)
