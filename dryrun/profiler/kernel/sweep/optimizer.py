"""Parameter-shard sizes to measure the fused Adam step at."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.validators import positive_int_tuple


@dataclass(frozen=True)
class OptimizerSweep:
    """Parameter-shard sizes to measure the fused Adam step at.

    Parameter counts rather than token counts: an Adam step's cost is independent of how
    many tokens produced the gradients. This is the only sweep in the tree with no token
    axis, and that is a property of the op rather than an omission.

    The grid should bracket `total_parameters / (tp * pp * dp)` for the configurations being
    modelled, since that is the shard one rank updates. Qwen3-8B at tp=2, pp=3 gives
    1.365B, and a dp=8 distributed optimizer would divide that again.
    """

    param_counts: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OptimizerSweep:
        """
        Parse the `optimizer:` subtree of a kernel profile config.

        Args:
            value (dict[str, Any]): The `optimizer:` block.

        Returns:
            OptimizerSweep: The parsed grid.

        Raises:
            ProfileValidationError: If `param_counts` is empty or holds a non-positive entry.
        """
        return cls(
            param_counts=positive_int_tuple(
                value["param_counts"], field="optimizer.param_counts"
            )
        )
