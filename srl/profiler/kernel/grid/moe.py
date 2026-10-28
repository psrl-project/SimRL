"""Per-rank MoE grouped-GEMM dimensions, derived from the checkpoint.

Lives in `grid/` rather than beside the measurement loop because the COST path needs it:
`cost/kernel/model_ops.py` shards the MoE block the same way the sweep did, and that module
must stay importable without torch. `tests/test_kernel_package_imports.py` gates that per
module, which is why this is not simply left in `runner/moe.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...schema import ProfileValidationError


def _required(model: Any, key: str) -> int:
    """Return an architecture value, raising when the checkpoint does not carry it."""
    value = model.extra.get(key)
    if value is None:
        raise ProfileValidationError(
            f"MoE profiling needs {key!r} from the checkpoint config. This model does not "
            f"declare it, so it is probably not a routed mixture-of-experts architecture."
        )
    return int(value)


@dataclass(frozen=True)
class MoeDims:
    """This rank's per-layer MoE grouped-GEMM dimensions."""

    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int


def moe_dims(model: Any, *, tensor_parallel: int) -> MoeDims:
    """
    Return this rank's `(hidden_size, intermediate_size, num_experts, top_k)`.

    `hidden_size` is never sharded here: TP divides `moe_intermediate_size` (the expert
    width) and an all-reduce reassembles the unsharded activation afterward -- the same
    convention `families_moe.py`/`families_qwen3_moe.py` already use for
    `expert_gate_up`/`expert_down`.

    Args:
        model (Any): A `ModelDescriptor` read from the checkpoint.
        tensor_parallel (int): Ranks the block is sharded across.

    Returns:
        MoeDims: This rank's grouped-GEMM dimensions.

    Raises:
        ProfileValidationError: If the architecture lacks the MoE fields, or
            `moe_intermediate_size` is not divisible by `tensor_parallel`.
    """
    if model.hidden_size is None:
        raise ProfileValidationError("Model is missing hidden_size.")
    moe_inter = _required(model, "moe_intermediate_size")
    num_experts = _required(model, "num_experts")
    top_k = _required(model, "num_experts_per_tok")
    if moe_inter % tensor_parallel:
        raise ProfileValidationError(
            f"moe_intermediate_size ({moe_inter}) must be divisible by tensor_parallel "
            f"({tensor_parallel})."
        )
    return MoeDims(
        hidden_size=int(model.hidden_size),
        intermediate_size=moe_inter // tensor_parallel,
        num_experts=num_experts,
        top_k=top_k,
    )
