"""Configuration for DryRun-owned FSDP2 training profiling.

There is no recipe inference here. Megatron needs it because its library recipes
carry tuned defaults keyed by model family and size; an FSDP run is fully described
by a HuggingFace checkpoint plus a device mesh, so the checkpoint is read directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ...hf import model_descriptor_from_yaml
from ...schema import HardwareDescriptor, ModelDescriptor, ProfileValidationError


@dataclass(frozen=True)
class FsdpProfileConfig:
    """Configuration for candidate execution and result normalization."""

    output_dir: Path
    model: ModelDescriptor
    hardware: HardwareDescriptor
    model_path: str
    experiment_name: str = ""
    compute_dtype: str = "bf16"
    optimizer: str = "adam"
    parameter_dtype: str = "bf16"
    gradient_dtype: str = "fp32"
    optimizer_state_dtype: str = "fp32"
    # Twelve steps discarding four. Under a single warmup step the first profiled step
    # was the slowest in 18 of 24 Megatron runs where chance alone gives about 8, so one
    # step does not reach steady state. Eight retained samples then support a median and
    # outlier rejection, which two samples cannot.
    max_steps: int = 12
    warmup_steps: int = 4
    num_layers_override: int | None = None
    # flash_attention_2 is required for the packed no-pad batching mode and is what verl
    # runs, so the profile and the simulated backend agree on the kernel family.
    attn_implementation: str = "flash_attention_2"
    environment: dict[str, str] = field(default_factory=dict)
    # The backend *is* torch, so its version is what pins reproducibility.
    backend_version: str = field(default_factory=lambda: torch.__version__)
    allow_incomplete: bool = False

    def __post_init__(self) -> None:
        if self.max_steps <= 0 or self.warmup_steps < 0:
            raise ProfileValidationError(
                "max_steps must be positive and warmup_steps must be non-negative."
            )
        if self.max_steps <= self.warmup_steps:
            raise ProfileValidationError(
                f"max_steps {self.max_steps} must exceed warmup_steps {self.warmup_steps} "
                "so at least one profiled step remains."
            )
        if self.num_layers_override is not None and self.num_layers_override <= 0:
            raise ProfileValidationError("num_layers_override must be positive.")
        if not self.model_path:
            raise ProfileValidationError("model_path must name a checkpoint directory.")

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> FsdpProfileConfig:
        root = base_dir or Path.cwd()

        def resolve(path_value: str) -> Path:
            path = Path(path_value)
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        training = value.get("training", {})
        profiling = value.get("profiling", {})
        model = model_descriptor_from_yaml(value["model"], base_dir=root)
        num_layers_override = training.get("num_layers_override")
        raw_model = value["model"]
        if isinstance(raw_model, str):
            model_path = raw_model
        else:
            # A dict entry without "path" would fall back to model.name, which is a
            # human-readable label, not a filesystem path. Fail here so the caller
            # sees a clear message rather than a torch.load error in the child.
            if "path" not in raw_model:
                raise ProfileValidationError(
                    "The model dict must include a 'path' key pointing to a checkpoint directory."
                )
            model_path = raw_model["path"]
        return cls(
            output_dir=resolve(value.get("output_dir", "profile_output")),
            model=model,
            hardware=HardwareDescriptor.from_dict(value["hardware"]),
            model_path=str(model_path),
            experiment_name=str(value.get("experiment_name", "")),
            compute_dtype=str(training.get("compute_dtype", "bf16")),
            optimizer=str(training.get("optimizer", "adam")),
            parameter_dtype=str(training.get("parameter_dtype", "bf16")),
            gradient_dtype=str(training.get("gradient_dtype", "fp32")),
            optimizer_state_dtype=str(training.get("optimizer_state_dtype", "fp32")),
            max_steps=int(training.get("max_steps", 12)),
            warmup_steps=int(training.get("warmup_steps", 4)),
            num_layers_override=(
                None if num_layers_override is None else int(num_layers_override)
            ),
            attn_implementation=str(
                training.get("attn_implementation", "flash_attention_2")
            ),
            environment={
                str(key): str(item) for key, item in value.get("environment", {}).items()
            },
            allow_incomplete=bool(profiling.get("allow_incomplete", False)),
        )

    @property
    def raw_root(self) -> Path:
        if self.experiment_name:
            return self.output_dir / "raw" / "fsdp" / self.experiment_name
        return self.output_dir / "raw" / "fsdp"
