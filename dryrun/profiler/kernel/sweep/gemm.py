"""Token counts to measure every per-rank GEMM shape at."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.validators import positive_int_tuple


@dataclass(frozen=True)
class GemmSweep:
    """Token counts to measure every per-rank GEMM shape at."""

    token_counts: tuple[int, ...]
    logits_request_counts: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GemmSweep:
        """
        Parse the `gemm:` subtree of a kernel profile config.

        Args:
            value (dict[str, Any]): The `gemm:` block.

        Returns:
            GemmSweep: The parsed grid.

        Raises:
            ProfileValidationError: If either list is empty or holds a non-positive entry.
        """
        return cls(
            token_counts=positive_int_tuple(value["token_counts"], field="gemm.token_counts"),
            logits_request_counts=positive_int_tuple(
                value.get("logits_request_counts", [1, 2, 4, 8, 16, 32, 48, 64]),
                field="gemm.logits_request_counts",
            ),
        )
