"""Token counts to measure every elementwise kernel at."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.validators import positive_int_tuple


@dataclass(frozen=True)
class ElementwiseSweep:
    """Token counts to measure every elementwise kernel at."""

    token_counts: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ElementwiseSweep:
        """
        Parse the `elementwise:` subtree of a kernel profile config.

        Args:
            value (dict[str, Any]): The `elementwise:` block.

        Returns:
            ElementwiseSweep: The parsed grid.

        Raises:
            ProfileValidationError: If `token_counts` is empty or holds a non-positive entry.
        """
        return cls(
            token_counts=positive_int_tuple(
                value["token_counts"], field="elementwise.token_counts"
            )
        )
