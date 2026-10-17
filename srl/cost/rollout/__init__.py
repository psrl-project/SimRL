"""Rollout latency models and workload vocabulary."""

from .base import RolloutCostModel, RolloutWorkload
from .composed import ComposedRollout
from .distserve import DistServe
from .psrl import PSRLFitted
from .roofline import LinearLPS, UnifiedRoofline

ROLLOUT_MODEL_REGISTRY = {
    "roofline": UnifiedRoofline,
    "psrl": PSRLFitted,
    "distserve": DistServe,
    "composed": ComposedRollout,
}

__all__ = [
    "ComposedRollout",
    "DistServe",
    "LinearLPS",
    "PSRLFitted",
    "ROLLOUT_MODEL_REGISTRY",
    "RolloutCostModel",
    "RolloutWorkload",
    "UnifiedRoofline",
]
