"""
Which kernels a model's layer executes, dispatched by architecture.

The shapes a transformer layer runs are a property of its architecture, not of any single
formula. A dense Llama-style block fuses into four projections; a mixture-of-experts block adds
a router and per-expert GEMMs; a compressed-KV block does not have a
`(heads + 2 * kv_heads) * head_dim` qkv projection at all. One hardcoded arithmetic path can
only be right for one of those, and its failure mode on the others is a plausible-looking
number rather than an error.

So the arithmetic lives behind a `ModelFamily`, chosen by `ModelDescriptor.architecture`, and an
architecture with no registered family is **refused**. That refusal is the point of this module:
mispricing an unsupported architecture is worse than declining to price it, because a wrong
latency propagates silently into every downstream schedule while a raised error does not.

This module stays free of `torch` and of the profiler's measurement code. It is imported by the
cost path, which prices kernels from parquet and must not pay a GPU-stack import to do it.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..schema import ModelDescriptor, ProfileValidationError


@runtime_checkable
class ModelFamily(Protocol):
    """The per-layer kernel shapes of one architecture family."""

    name: str

    def gemm_shapes(self, model: ModelDescriptor, *, tensor_parallel: int) -> tuple[Any, ...]:
        """Return one layer's per-rank GEMM shapes, excluding the logits projection."""

    def logits_shape(self, model: ModelDescriptor, *, tensor_parallel: int) -> Any:
        """Return the per-rank logits projection, which runs once per step rather than per layer."""

    def head_counts(self, model: ModelDescriptor, *, tensor_parallel: int) -> tuple[int, int]:
        """Return this rank's `(num_heads, num_kv_heads)`."""


FAMILY_REGISTRY: dict[str, ModelFamily] = {}


def register_family(architecture: str, family: ModelFamily) -> None:
    """
    Map one HuggingFace architecture name onto the family that prices it.

    Args:
        architecture (str): The `architectures[0]` value from `config.json`.
        family (ModelFamily): The family whose shapes that architecture runs.

    Raises:
        ProfileValidationError: If the architecture is already registered.
    """
    existing = FAMILY_REGISTRY.get(architecture)
    if existing is not None:
        raise ProfileValidationError(
            f"Architecture {architecture!r} is already registered to family {existing.name!r}. "
            f"Two families claiming one architecture would make dispatch depend on import order."
        )
    FAMILY_REGISTRY[architecture] = family


def family_for(model: ModelDescriptor) -> ModelFamily:
    """
    Return the family that prices one model's layers.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.

    Returns:
        ModelFamily: The registered family.

    Raises:
        ProfileValidationError: If the descriptor carries no architecture, or if its
            architecture has no registered family. Both are refusals rather than fallbacks:
            pricing an unknown architecture with the dense arithmetic would produce a
            plausible number for kernels the model never runs.
    """
    architecture = model.architecture
    if not architecture:
        raise ProfileValidationError(
            "Model descriptor carries no architecture, so its kernel shapes cannot be derived. "
            "The architecture comes from config.json's `architectures` field."
        )
    family = FAMILY_REGISTRY.get(architecture)
    if family is None:
        known = ", ".join(sorted(FAMILY_REGISTRY)) or "none"
        raise ProfileValidationError(
            f"No kernel-shape family is registered for architecture {architecture!r}. "
            f"Registered: {known}. Add one in dryrun/profiler/kernels/families_dense.py "
            f"(or a new families_*.py) and register it, rather than pricing this model with "
            f"another family's shapes."
        )
    return family


# Registration happens here, at the bottom of the protocol module, so that importing `families`
# is sufficient to populate the registry. Leaving it to each caller to import the right
# families_*.py made `family_for` fail in a fresh interpreter while every test passed, because
# each test imported the family module itself -- the same trap the cost layer's ops/__init__.py
# documents. The import is at the end to avoid a cycle: families_dense imports this module.
from . import families_dense as _families_dense  # noqa: E402,F401  (registers dense architectures)
