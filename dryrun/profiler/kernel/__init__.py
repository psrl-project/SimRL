"""Direct kernel profiling: measure attention, GEMM, and collective ops on the GPU."""

from __future__ import annotations

# Eager imports: these modules are GPU-free and safe to pull in unconditionally.
from .attention import (
    AttentionPoint,
    attention_head_dim,
    context_sweep_points,
    generation_sweep_points,
    profile_flashinfer_attention,
    profile_vllm_attention,
    rank_head_counts,
)
from .config import (
    AttentionSweep,
    GemmSweep,
    KernelProfileConfig,
    MeasurementConfig,
)
from .measurements import PerfTableWriter

# Lazy mapping: GPU-touching names to their owning submodule.  The submodule is only
# imported when the name is first accessed, so loading this package (e.g. to reach
# `config` or `attention`) does not pull in torch, vLLM, or FlashInfer.
_LAZY: dict[str, str] = {
    # gemm shapes (pure arithmetic, no torch)
    "GemmShape": "shapes",
    "gemm_all_shapes": "shapes",
    "gemm_sweep_shapes": "shapes",
    "gemm_wgrad_shapes": "shapes",
    "logits_gemm_shape": "shapes",
    "logits_sweep_shapes": "shapes",
    "transformer_gemm_shapes": "shapes",
    # gemm measurement (requires torch)
    "profile_gemm": "gemm",
    # collective
    "profile_collective": "collective",
    # vllm attention backend
    "SelectedBackend": "_vllm_attn",
    "select_attention_backend": "_vllm_attn",
    "time_attention_point": "_vllm_attn",
    # flashinfer attention backend
    "flashinfer_kernel_source": "_flashinfer_attn",
    "time_flashinfer_context_point": "_flashinfer_attn",
    "time_flashinfer_generation_point": "_flashinfer_attn",
}

__all__ = [
    "AttentionPoint",
    "AttentionSweep",
    "GemmShape",
    "GemmSweep",
    "KernelProfileConfig",
    "MeasurementConfig",
    "PerfTableWriter",
    "SelectedBackend",
    "attention_head_dim",
    "context_sweep_points",
    "generation_sweep_points",
    "gemm_all_shapes",
    "gemm_sweep_shapes",
    "gemm_wgrad_shapes",
    "logits_gemm_shape",
    "logits_sweep_shapes",
    "profile_collective",
    "profile_flashinfer_attention",
    "profile_gemm",
    "profile_vllm_attention",
    "rank_head_counts",
    "select_attention_backend",
    "time_attention_point",
    "time_flashinfer_context_point",
    "time_flashinfer_generation_point",
    "transformer_gemm_shapes",
    "flashinfer_kernel_source",
]


def __getattr__(name: str) -> object:
    """Lazily import GPU-touching names on first access (PEP 562)."""
    submodule = _LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
    import importlib

    mod = importlib.import_module(f".{submodule}", package=__name__)
    value = getattr(mod, name)
    # Cache in module globals so subsequent accesses skip __getattr__.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose the full public surface to tab-completion and `dir()`."""
    return __all__
