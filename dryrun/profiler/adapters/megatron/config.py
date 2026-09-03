"""Configuration for DryRun-owned Megatron training profiling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...hf import model_descriptor_from_yaml
from ...schema import (
    HardwareDescriptor,
    ModelDescriptor,
    ProfileValidationError,
)

PINNED_BRIDGE_VERSION = "0.4.2"
PINNED_BRIDGE_COMMIT = "c810129341a84e58f4cbed3093f70668a088c028"

# The recompute method paired with `recompute: full`. Uniform with one layer
# recomputes every transformer layer individually, which keeps peak activation
# close to linear in tokens and therefore keeps the Hydraulis memory feature
# well conditioned. It is recorded in the manifest because the Hydraulis cell
# key only carries the opaque `recompute` label, so a future `block` variant
# must not land in the same cell unnoticed.
RECOMPUTE_METHOD = "uniform"
RECOMPUTE_NUM_LAYERS = 1


def infer_bridge_recipe(model: ModelDescriptor) -> tuple[str, str]:
    """Infer the official performance recipe from a HuggingFace model descriptor."""
    model_type = str(model.extra.get("model_type") or "").lower()
    architecture = str(model.architecture or "").lower()
    identity = f"{model_type} {architecture} {model.name.lower()}"
    params = model.parameter_count or 0
    size_b = max(1, int(round(params / 1e9))) if params else None
    if "qwen3" in identity:
        if size_b is None:
            raise ProfileValidationError("Qwen3 recipe inference requires parameter_count.")
        return "qwen", f"qwen3_{size_b}b"
    raise ProfileValidationError(
        f"Cannot infer Megatron-Bridge recipe from model {model.name!r}."
    )


@dataclass(frozen=True)
class MegatronBridgeConfig:
    """Configuration for candidate execution and result normalization."""

    bridge_dir: Path
    output_dir: Path
    model: ModelDescriptor
    hardware: HardwareDescriptor
    experiment_name: str = ""
    model_family_name: str = "qwen"
    model_recipe_name: str = "qwen3_8b"
    compute_dtype: str = "bf16"
    optimizer: str = "adam"
    parameter_dtype: str = "bf16"
    gradient_dtype: str = "fp32"
    optimizer_state_dtype: str = "fp32"
    distributed_optimizer: bool = True
    max_steps: int = 4
    warmup_steps: int = 1
    num_layers_override: int | None = None
    environment: dict[str, str] = field(default_factory=dict)
    backend_version: str = PINNED_BRIDGE_VERSION
    source_commit: str = PINNED_BRIDGE_COMMIT
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

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> MegatronBridgeConfig:
        root = base_dir or Path.cwd()

        def resolve(path_value: str) -> Path:
            path = Path(path_value)
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        training = value.get("training", {})
        profiling = value.get("profiling", {})
        model = model_descriptor_from_yaml(value["model"], base_dir=root)
        hardware = HardwareDescriptor.from_dict(value["hardware"])
        family, recipe = infer_bridge_recipe(model)
        num_layers_override = training.get("num_layers_override")
        return cls(
            bridge_dir=(
                resolve(value["bridge_dir"])
                if value.get("bridge_dir")
                else _default_bridge_dir()
            ),
            output_dir=resolve(value.get("output_dir", "profile_output")),
            model=model,
            hardware=hardware,
            experiment_name=str(value.get("experiment_name", "")),
            model_family_name=str(value.get("model_family_name", family)),
            model_recipe_name=str(value.get("model_recipe_name", recipe)),
            compute_dtype=str(training.get("compute_dtype", "bf16")),
            optimizer=str(training.get("optimizer", "adam")),
            parameter_dtype=str(training.get("parameter_dtype", "bf16")),
            gradient_dtype=str(training.get("gradient_dtype", "fp32")),
            optimizer_state_dtype=str(training.get("optimizer_state_dtype", "fp32")),
            distributed_optimizer=bool(training.get("distributed_optimizer", True)),
            max_steps=int(training.get("max_steps", 4)),
            warmup_steps=int(training.get("warmup_steps", 1)),
            num_layers_override=(
                None if num_layers_override is None else int(num_layers_override)
            ),
            environment={
                str(key): str(item) for key, item in value.get("environment", {}).items()
            },
            allow_incomplete=bool(profiling.get("allow_incomplete", False)),
        )

    @property
    def raw_root(self) -> Path:
        if self.experiment_name:
            return self.output_dir / "raw" / "megatron_bridge" / self.experiment_name
        return self.output_dir / "raw" / "megatron_bridge"


def _default_bridge_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "third_party" / "Megatron-Bridge"
