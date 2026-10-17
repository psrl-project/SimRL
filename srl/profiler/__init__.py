"""Versioned profiling data and official benchmark adapters."""

from .hf import load_hf_model_descriptor, model_descriptor_from_yaml
from .io import iter_records, profile_digest, read_records, write_records
from .schema import (
    HardwareDescriptor,
    ModelDescriptor,
    Parallelism,
    ProfileRecord,
    ProfileValidationError,
    RankMemory,
    RolloutPayload,
    RunDescriptor,
    SequenceStats,
    SoftwareDescriptor,
    TrainingPayload,
)
from .search_space import (
    ModelConstraints,
    TrainingCandidate,
    enumerate_training_candidates,
)

__all__ = [
    "HardwareDescriptor",
    "ModelConstraints",
    "ModelDescriptor",
    "Parallelism",
    "ProfileRecord",
    "ProfileValidationError",
    "RankMemory",
    "RolloutPayload",
    "RunDescriptor",
    "SequenceStats",
    "SoftwareDescriptor",
    "TrainingCandidate",
    "TrainingPayload",
    "enumerate_training_candidates",
    "iter_records",
    "load_hf_model_descriptor",
    "model_descriptor_from_yaml",
    "profile_digest",
    "read_records",
    "write_records",
]
