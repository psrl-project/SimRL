"""Token counts to measure the routed-MoE grouped GEMM at.

One axis, unlike GDN's two: there is no phase split for MoE (see `cost/kernel/ops/moe.py`), so
`num_tokens` alone spans both decode-shaped and prefill-shaped regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.validators import positive_int_tuple


@dataclass(frozen=True)
class MoeSweep:
    """Token counts to measure the routed-MoE grouped GEMM at."""

    num_tokens: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MoeSweep:
        """
        Parse the `moe:` subtree of a kernel profile config.

        Args:
            value (dict[str, Any]): The `moe:` block.

        Returns:
            MoeSweep: The parsed grid.

        Raises:
            ProfileValidationError: If the list is empty or holds a non-positive entry.
        """
        return cls(num_tokens=positive_int_tuple(value["num_tokens"], field="moe.num_tokens"))
