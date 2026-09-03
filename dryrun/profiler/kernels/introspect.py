"""
Check declared kernel shapes against a real module graph.

`families_dense.py` declares that a dense layer fuses into four projections. That claim is
arithmetic over `config.json` and nothing in the fast path can falsify it, so this module is the
oracle: it builds the actual model and compares.

Two things make the comparison non-trivial.

The model is built on the **meta device**, so no weights are allocated and no GPU is touched.
Measured on Qwen3-8B: 3.09 seconds and 0 MiB of VRAM for all 254 GEMM modules. That is what makes
this cheap enough to run as a gate rather than a research exercise.

HuggingFace and an inference engine do not agree on how many GEMMs a layer has. HF keeps
`q_proj`, `k_proj`, `v_proj` separate and `gate_proj`/`up_proj` separate -- seven 2-D weights per
layer -- while vLLM fuses them into `QKVParallelLinear` and `MergedColumnParallelLinear`, giving
four. We model the engine, so the real modules are fused here before comparing. Summing HF's row
counts reproduces the declared `n` values exactly on both local checkpoints, which is the evidence
that the four fused shapes are the engine's truth rather than an approximation.

This module imports `torch` and `transformers`, so it must never be imported by the cost path.
It is reached only from the `kernels verify-shapes` CLI command.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schema import ModelDescriptor, ProfileValidationError
from .families import family_for

# How the engine fuses HuggingFace's separate projections. Each entry maps a declared shape name
# onto the HF module suffixes whose output rows concatenate into it. `o_proj` and `down_proj` are
# absent because they are not fused, and are compared one-to-one.
_FUSION_RULES: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
}

_PASSTHROUGH: dict[str, str] = {
    "o_proj": "self_attn.o_proj",
    "down_proj": "mlp.down_proj",
}

_LOGITS_MODULE = "lm_head"


@dataclass(frozen=True)
class ShapeComparison:
    """One declared shape beside the real fused shape it claims to describe."""

    name: str
    declared: tuple[int, int]
    observed: tuple[int, int] | None

    @property
    def matches(self) -> bool:
        """Return whether the declared shape equals what the module graph actually holds."""
        return self.observed is not None and self.declared == self.observed


@dataclass(frozen=True)
class VerificationReport:
    """The outcome of comparing one model's declared shapes against its modules."""

    architecture: str
    family: str
    layer_count: int
    comparisons: tuple[ShapeComparison, ...]
    heterogeneous_layers: tuple[int, ...]

    @property
    def ok(self) -> bool:
        """Return whether every shape matched and every layer had the same shape set."""
        return all(item.matches for item in self.comparisons) and not self.heterogeneous_layers


def _linear_shapes(model: Any) -> dict[str, tuple[int, int]]:
    """Return every 2-D weight in the module graph, keyed by qualified module name."""
    shapes: dict[str, tuple[int, int]] = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "dim", None) is None or weight.dim() != 2:
            continue
        shapes[name] = (int(weight.shape[0]), int(weight.shape[1]))
    return shapes


def _layer_index(name: str) -> int | None:
    """Return the decoder-layer index a module belongs to, or None if it sits outside the stack."""
    parts = name.split(".")
    for index, part in enumerate(parts):
        if part == "layers" and index + 1 < len(parts) and parts[index + 1].isdigit():
            return int(parts[index + 1])
    return None


def _suffix(name: str, layer: int) -> str:
    """Return a module's name relative to its decoder layer."""
    marker = f"layers.{layer}."
    return name.split(marker, 1)[1] if marker in name else name


def _fuse(layer_shapes: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """Apply the engine's fusion rules to one layer's HuggingFace modules."""
    fused: dict[str, tuple[int, int]] = {}
    for declared_name, suffixes in _FUSION_RULES.items():
        parts = [layer_shapes[suffix] for suffix in suffixes if suffix in layer_shapes]
        if len(parts) != len(suffixes):
            continue
        columns = {shape[1] for shape in parts}
        if len(columns) != 1:
            # A fused projection concatenates along rows, so every part must share its input
            # width. Differing widths mean these modules are not the ones the rule describes.
            continue
        fused[declared_name] = (sum(shape[0] for shape in parts), parts[0][1])
    for declared_name, suffix in _PASSTHROUGH.items():
        if suffix in layer_shapes:
            fused[declared_name] = layer_shapes[suffix]
    return fused


def verify_shapes(
    model_path: str | Path,
    descriptor: ModelDescriptor,
    *,
    tensor_parallel: int = 1,
) -> VerificationReport:
    """
    Compare a family's declared shapes against the checkpoint's real module graph.

    Args:
        model_path (str | Path): Checkpoint directory or hub id, used to build the module graph.
        descriptor (ModelDescriptor): The descriptor the declared shapes come from.
        tensor_parallel (int): Must be 1. The module graph is unsharded, so comparing it against
            sharded shapes would report a spurious mismatch on every entry.

    Returns:
        VerificationReport: Per-shape comparisons plus any layer whose shape set differs from
            layer 0's.

    Raises:
        ProfileValidationError: If `tensor_parallel` is not 1, if the checkpoint has no decoder
            layers, or if no family is registered for the architecture.
    """
    if tensor_parallel != 1:
        raise ProfileValidationError(
            f"verify_shapes compares against an unsharded module graph, so tensor_parallel must "
            f"be 1, got {tensor_parallel!r}."
        )
    # Imported here: this module is the only torch-dependent part of the profiler's shape code,
    # and the cost path must not pay a GPU-stack import to price a kernel.
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(model_path))
    with torch.device("meta"):
        module = AutoModelForCausalLM.from_config(config)

    shapes = _linear_shapes(module)
    per_layer: dict[int, dict[str, tuple[int, int]]] = {}
    outside: dict[str, tuple[int, int]] = {}
    for name, shape in shapes.items():
        layer = _layer_index(name)
        if layer is None:
            outside[name] = shape
        else:
            per_layer.setdefault(layer, {})[_suffix(name, layer)] = shape
    if not per_layer:
        raise ProfileValidationError(
            f"Found no decoder-layer linear modules in {model_path}, so there is nothing to "
            f"compare declared shapes against."
        )

    first = min(per_layer)
    reference = _fuse(per_layer[first])
    # Every layer must fuse to the same shape set. No config key reliably says whether a model is
    # layer-heterogeneous, so this is the only check that catches a model whose layers differ --
    # which the dense family would flatten into one shape times num_layers.
    heterogeneous = tuple(
        sorted(layer for layer, entries in per_layer.items() if _fuse(entries) != reference)
    )

    family = family_for(descriptor)
    declared = {
        shape.name: (shape.n, shape.k)
        for shape in family.gemm_shapes(descriptor, tensor_parallel=tensor_parallel)
    }
    logits = family.logits_shape(descriptor, tensor_parallel=tensor_parallel)
    declared[logits.name] = (logits.n, logits.k)
    observed = dict(reference)
    for name, shape in outside.items():
        if name.endswith(_LOGITS_MODULE):
            observed[logits.name] = shape

    comparisons = tuple(
        ShapeComparison(name=name, declared=value, observed=observed.get(name))
        for name, value in sorted(declared.items())
    )
    return VerificationReport(
        architecture=str(descriptor.architecture),
        family=family.name,
        layer_count=len(per_layer),
        comparisons=comparisons,
        heterogeneous_layers=heterogeneous,
    )
