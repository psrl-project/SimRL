"""Token and batch counts to measure gated-delta-net (linear attention) at.

Two axes rather than the attention sweep's three, and that asymmetry is the architecture
rather than an omission. A gated delta net carries a fixed-size recurrent state instead of a
growing KV cache, so:

- Prefill is swept over `total_tokens` alone. The kernel
  (`fla.ops.gated_delta_rule.chunk_gated_delta_rule`) is varlen: a packed batch is one row
  behind `cu_seqlens`, so there is no batch dimension, and cost is linear in the token total
  rather than quadratic in any one sequence's length.
- Decode is swept over `batch_size` alone. There is no `step` axis because the state is the
  same size at KV depth 128 as at 32768, so a decode step costs the same either way.
  Interpolating over history depth would fit noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.validators import positive_int_tuple


@dataclass(frozen=True)
class GdnSweep:
    """Packed token counts and decode batch sizes to measure gated delta net at."""

    total_tokens: tuple[int, ...]
    batch_sizes: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GdnSweep:
        """
        Parse the `gdn:` subtree of a kernel profile config.

        Args:
            value (dict[str, Any]): The `gdn:` block.

        Returns:
            GdnSweep: The parsed grid.

        Raises:
            ProfileValidationError: If either list is empty or holds a non-positive entry.
        """
        return cls(
            total_tokens=positive_int_tuple(value["total_tokens"], field="gdn.total_tokens"),
            batch_sizes=positive_int_tuple(value["batch_sizes"], field="gdn.batch_sizes"),
        )
