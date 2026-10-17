"""Configuration for SimRL-owned Megatron training profiling."""

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


# Library pretrain recipes this adapter can resolve, keyed by the HuggingFace `model_type`
# the checkpoint declares, and within that by the recipe's own parameter count.
#
# A table rather than `f"{family}_{round(params / 1e9)}b"`, because that arithmetic names
# recipes that do not exist. Megatron-Bridge spells the sub-billion sizes `600m` and `1p7b`,
# and it ships `qwen25_14b` but no `qwen2_14b`, so a 14.77B Qwen2-architecture checkpoint
# rounds to a `qwen2_15b` that was never written. `_resolve_recipe` would then raise inside
# the torchrun child, after the candidate had already paid for distributed setup.
#
# `qwen2` covers Qwen2 and Qwen2.5 alike: HuggingFace gives both the same `model_type`, and
# nothing in `config.json` separates them. Resolving by size is therefore the only available
# discriminator, and it is sufficient here, because the two generations' recipes differ
# only in `ddp.check_for_nan_in_grad` (a correctness assertion, not a performance knob) and
# in the hub id whose architecture `_local_checkpoint` replaces with the profiled checkpoint
# anyway. Where both generations declare a size, the Qwen2.5 recipe is the entry: it is the
# newer tuning and the one a 14B or 32B checkpoint can actually reach.
_BRIDGE_RECIPES: dict[str, tuple[tuple[int, str, str], ...]] = {
    "qwen3": (
        (600_000_000, "qwen", "qwen3_600m"),
        (1_700_000_000, "qwen", "qwen3_1p7b"),
        (4_000_000_000, "qwen", "qwen3_4b"),
        (8_190_000_000, "qwen", "qwen3_8b"),
        (14_800_000_000, "qwen", "qwen3_14b"),
        (32_800_000_000, "qwen", "qwen3_32b"),
    ),
    "qwen2": (
        (494_000_000, "qwen", "qwen25_500m"),
        (1_540_000_000, "qwen", "qwen25_1p5b"),
        (7_620_000_000, "qwen", "qwen25_7b"),
        (14_770_000_000, "qwen", "qwen25_14b"),
        (32_760_000_000, "qwen", "qwen25_32b"),
        (72_700_000_000, "qwen", "qwen25_72b"),
    ),
}

# How far a checkpoint's parameter count may sit from a recipe's before the match is refused.
# A distill or a width-trimmed variant drifts a little from the size it is named after, but a
# different size entirely is a different model: the 8B and 14B Qwen3 recipes are 1.8x apart,
# so this tolerance cannot confuse them.
_RECIPE_SIZE_TOLERANCE = 0.08


def infer_bridge_recipe(model: ModelDescriptor) -> tuple[str, str]:
    """
    Infer the official Megatron-Bridge pretrain recipe from a HuggingFace descriptor.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.

    Returns:
        tuple[str, str]: The recipe's `(family, name)`, as `launch.py` imports them.

    Raises:
        ProfileValidationError: If the checkpoint's `model_type` has no recipe table, if
            `parameter_count` is unknown, or if no tabled size is within
            `_RECIPE_SIZE_TOLERANCE` of the checkpoint. Set `model_family_name` and
            `model_recipe_name` in the YAML to name a recipe this table does not hold.
    """
    model_type = str(model.extra.get("model_type") or "").lower()
    architecture = str(model.architecture or "").lower()
    identity = f"{model_type} {architecture} {model.name.lower()}"
    # Matched against the whole identity, not just `model_type`: a Qwen3 checkpoint whose
    # config predates the `qwen3` model_type still names itself in its architecture.
    # Longest first, so `qwen3` is not swallowed by a `qwen2` substring test.
    for family in sorted(_BRIDGE_RECIPES, key=len, reverse=True):
        if family in identity:
            break
    else:
        raise ProfileValidationError(
            f"Cannot infer a Megatron-Bridge recipe from model {model.name!r}: no recipe "
            f"table for any of {sorted(_BRIDGE_RECIPES)}. Set model_family_name and "
            "model_recipe_name in the YAML to name one explicitly."
        )
    params = model.parameter_count
    if not params:
        raise ProfileValidationError(
            f"Recipe inference for {family!r} requires parameter_count, which is unknown "
            f"for model {model.name!r}."
        )
    size, recipe_family, recipe_name = min(
        _BRIDGE_RECIPES[family],
        key=lambda entry: abs(entry[0] - params),
    )
    if abs(size - params) > _RECIPE_SIZE_TOLERANCE * size:
        raise ProfileValidationError(
            f"Model {model.name!r} has {params} parameters, which is not within "
            f"{_RECIPE_SIZE_TOLERANCE:.0%} of any {family!r} recipe (closest is "
            f"{recipe_name!r} at {size}). Set model_family_name and model_recipe_name in "
            "the YAML to name a recipe explicitly."
        )
    return recipe_family, recipe_name


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
        # Inference only runs when the YAML leaves both fields unset. It raises for an
        # architecture the recipe table does not hold, and naming the recipe explicitly is
        # the documented way out of that -- which only works if the override is consulted
        # first. Both fields are required together: a family without a name, or the reverse,
        # would pair an override with an inferred half and resolve a recipe neither asked
        # for.
        stated_family = value.get("model_family_name")
        stated_recipe = value.get("model_recipe_name")
        if bool(stated_family) != bool(stated_recipe):
            raise ProfileValidationError(
                "model_family_name and model_recipe_name must be set together, or both "
                "left unset for inference from the checkpoint."
            )
        if stated_family and stated_recipe:
            family, recipe = str(stated_family), str(stated_recipe)
        else:
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
            model_family_name=family,
            model_recipe_name=recipe,
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
