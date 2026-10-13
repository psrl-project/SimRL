"""
Enumerate the parameter-shard sizes to measure the fused Adam step at.

The thinnest grid in the tree, and deliberately so: there is no shape to derive. An Adam
step reads a parameter, a gradient and three state tensors and writes two of them back, so
its cost is a function of the parameter count alone. The sweep declares those counts
directly and this module passes them through.

**The absent token axis is a property of the op, not an omission here.** Every other grid
crosses something with a token count because every other kernel's work scales with how much
data flowed through it. An optimizer step runs once per training step over a fixed shard
regardless of whether that step consumed 4096 tokens or 40960, which is why
`OptimizerSweep` has no token field for this module to cross against.

The grid should bracket `total_parameters / (tp * pp * dp)`, since that is the shard one rank
updates. A point that does not fit in memory is refused by the runner rather than dropped
here: the refusal needs the device, and a shard that OOMs during state allocation produces a
very *fast* row rather than an error. See `runner/optimizer.py`.
"""

from __future__ import annotations

from ...schema import ProfileValidationError
from ..sweep.optimizer import OptimizerSweep


def optimizer_param_counts(sweep: OptimizerSweep) -> tuple[int, ...]:
    """
    Return the parameter-shard sizes to measure, sorted ascending.

    Sorted rather than left in declaration order so the sweep measures small shards first.
    A grid whose largest point exceeds device memory then fails after recording every point
    that fits, instead of failing first and recording none.

    Args:
        sweep (OptimizerSweep): Declares `param_counts`.

    Returns:
        tuple[int, ...]: Deduplicated counts, ascending.

    Raises:
        ProfileValidationError: If the grid is empty, which `OptimizerSweep.from_dict`
            already rejects but a directly-constructed sweep could carry.
    """
    counts = tuple(sorted(set(sweep.param_counts)))
    if not counts:
        raise ProfileValidationError(
            "optimizer.param_counts is empty, so there is no shard size to measure."
        )
    if counts[0] <= 0:
        raise ProfileValidationError(
            f"optimizer.param_counts holds a non-positive entry {counts[0]!r}."
        )
    return counts
