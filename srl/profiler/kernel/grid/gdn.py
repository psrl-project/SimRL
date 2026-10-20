"""Per-rank gated-delta-net head counts, derived from the checkpoint.

Lives in `grid/` rather than beside the measurement loop because the COST path needs it:
`cost/kernel/model_ops.py` shards the GDN layer the same way the sweep did, and that module
must stay importable without torch. `tests/test_kernel_package_imports.py` gates that per
module, which is why this is not simply left in `runner/gdn.py`.
"""

from __future__ import annotations

from typing import Any

from ...schema import ProfileValidationError


def _require(model: Any, key: str) -> int:
    """Return an architecture value, raising when the checkpoint does not carry it."""
    value = model.extra.get(key)
    if value is None:
        raise ProfileValidationError(
            f"Gated-delta-net profiling needs {key!r} from the checkpoint config. This model "
            f"does not declare it, so it is probably not a hybrid-attention architecture."
        )
    return int(value)


def gdn_head_counts(model: Any, *, tensor_parallel: int) -> tuple[int, int, int]:
    """
    Return this rank's `(num_key_heads, num_value_heads, head_dim)` for the GDN layer.

    Sharded by `tensor_parallel` the same way the attention heads are, and refused rather
    than rounded when the split is uneven: a rank holding a fractional head is not a
    configuration the kernel can run, so a table keyed on it would never be queried.

    Args:
        model (Any): A `ModelDescriptor` read from the checkpoint.
        tensor_parallel (int): Ranks the layer is sharded across.

    Returns:
        tuple[int, int, int]: Key heads, value heads, and the shared head dimension.

    Raises:
        ProfileValidationError: If the architecture lacks the GDN fields, if the key and
            value head dimensions differ, or if a head count is not divisible by
            `tensor_parallel`.
    """
    key_heads = _require(model, "linear_num_key_heads")
    value_heads = _require(model, "linear_num_value_heads")
    head_dim = _require(model, "linear_key_head_dim")
    value_head_dim = int(model.extra.get("linear_value_head_dim") or head_dim)
    if value_head_dim != head_dim:
        raise ProfileValidationError(
            f"This op assumes one head dimension for keys and values, but the checkpoint has "
            f"linear_key_head_dim={head_dim} and linear_value_head_dim={value_head_dim}. "
            "The SOL formulas would need separate terms."
        )
    for name, value in (
        ("linear_num_key_heads", key_heads),
        ("linear_num_value_heads", value_heads),
    ):
        if value % tensor_parallel:
            raise ProfileValidationError(
                f"{name} ({value}) must be divisible by tensor_parallel ({tensor_parallel})."
            )
    return key_heads // tensor_parallel, value_heads // tensor_parallel, head_dim
