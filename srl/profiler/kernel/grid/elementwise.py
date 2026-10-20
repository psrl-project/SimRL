"""
Enumerate the elementwise kernels one token count issues, and how many elements each touches.

Extracted from the measurement loop so the `n_elements` arithmetic can be checked without a
GPU. That matters because `cost/training/training_ops.py` derives the *same* quantities on the
query side (`rms_norm` over `tokens * hidden_size`, `rotary_embedding` over
`tokens * heads * head_dim`), and a disagreement between the two means a query lands on a
point of the surface the sweep never measured. While the expressions lived inside the timing
loop, the only way to compare them was to run a sweep.

The five kinds are not symmetric in what sizes them: `rms_norm` and `tensor_plumbing` scale
with `hidden_size`, `silu_and_mul` with the MLP's `intermediate_size`, and `rotary_embedding`
and `qk_norm` with head counts. `rotary_embedding` is the only one that touches query *and*
key in one call, which is why its count sums both head groups.

`ELEMENTWISE_KINDS` is not redeclared here: `cost/kernel/ops/elementwise.py` owns it, and a
second list would let the two drift into measuring a kind nothing prices.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...schema import ModelDescriptor, ProfileValidationError
from ..sweep.elementwise import ElementwiseSweep
from ..utils.model_dims import model_head_dim, model_intermediate_size


@dataclass(frozen=True)
class ElementwisePoint:
    """One elementwise kernel call to measure, with the element count it touches."""

    op_kind: str
    token_count: int
    n_elements: int


@dataclass(frozen=True)
class ElementwiseDims:
    """The per-rank dimensions every elementwise kernel is sized from."""

    hidden_size: int
    head_dim: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int


def elementwise_dims(model: ModelDescriptor) -> ElementwiseDims:
    """
    Read the dimensions the elementwise kernels are sized from.

    Unsharded, matching the measurement: this sweep times one rank's kernels at the full
    head count, and the cost side applies its own tensor-parallel division when it queries.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.

    Returns:
        ElementwiseDims: Hidden size, head dim, intermediate size, and both head counts.

    Raises:
        ProfileValidationError: If the architecture is missing `hidden_size` or
            `num_attention_heads`.
    """
    if model.hidden_size is None:
        raise ProfileValidationError("Model is missing hidden_size.")
    if model.num_attention_heads is None:
        raise ProfileValidationError("Model is missing num_attention_heads.")
    heads = int(model.num_attention_heads)
    return ElementwiseDims(
        hidden_size=int(model.hidden_size),
        head_dim=model_head_dim(model),
        intermediate_size=model_intermediate_size(model),
        num_heads=heads,
        num_kv_heads=int(model.num_query_groups or heads),
    )


def elementwise_sweep_points(
    model: ModelDescriptor,
    sweep: ElementwiseSweep,
) -> tuple[ElementwisePoint, ...]:
    """
    Enumerate every `(op_kind, token_count)` pair with the element count it touches.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        sweep (ElementwiseSweep): Token counts to measure at.

    Returns:
        tuple[ElementwisePoint, ...]: One point per kind per token count, in the order the
            measurement loop issues them.

    Raises:
        ProfileValidationError: If the architecture is incomplete.
    """
    dims = elementwise_dims(model)
    points: list[ElementwisePoint] = []
    for token_count in sweep.token_counts:
        for op_kind, n_elements in (
            ("rms_norm", token_count * dims.hidden_size),
            # Rope rotates query and key in one call, so both head groups are read.
            (
                "rotary_embedding",
                token_count * (dims.num_heads + dims.num_kv_heads) * dims.head_dim,
            ),
            ("silu_and_mul", token_count * dims.intermediate_size),
            # qk_norm normalizes each head independently, so its rows are token*heads long.
            ("qk_norm", token_count * dims.num_heads * dims.head_dim),
            # `tensor_plumbing` is timed as a hidden-size copy: the modal member of the
            # traced population. See `cost/kernel/ops/elementwise.py` for why one
            # representative rather than five.
            ("tensor_plumbing", token_count * dims.hidden_size),
        ):
            points.append(
                ElementwisePoint(
                    op_kind=op_kind, token_count=token_count, n_elements=n_elements
                )
            )
    return tuple(points)
