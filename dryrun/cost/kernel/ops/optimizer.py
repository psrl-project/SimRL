"""
The fused Adam step: one bandwidth-bound pass over a parameter shard.

Megatron does not run torch's `AdamW`. `megatron/core/optimizer/__init__.py:13` imports
`FusedAdam` from `transformer_engine.pytorch.optimizers`, falling back to apex and only then
to torch, so the kernel to measure is TE's `multi_tensor_adam`. Timing
`torch.optim.AdamW(fused=True)` would be the same class of error as timing
`torch.distributed.all_reduce` for a vLLM tensor-parallel step: a plausible kernel that this
configuration never invokes.

The op has **no token axis**, which makes it the only op in the tree whose cost is
independent of the workload. An Adam step reads and writes the optimizer state once per
step regardless of how many tokens produced the gradients, so `param_count` is the only
axis and the measured curve is a straight line through the origin above the launch floor.

**Measured magnitude, so a reader does not have to guess whether this term matters.** On
H20 with `master_weights=True` and fp32 moments, the step runs at 2703 GB/s for shards from
10M to 1.365B parameters: 81% of the card's 3.35 TB/s HBM peak, and flat across two orders
of magnitude, which is what a pure-bandwidth kernel looks like. A Qwen3-8B rank at tp=2,
pp=3 holds 1.365B parameters and pays **14.1 ms**. Against the 1.06-72 s training steps in
`outputs/profile/qwen3_8b/megatron_tp2_pp3_thd/` that is 0.02-1.3%, so this term is real but
it is not where a composed step's error lives. It is measured because an unmeasured term
cannot be ruled out, and ruling it out is the result.

Traffic per parameter under the mixed-precision path Megatron uses: read bf16 grad (2) +
fp32 master (4) + fp32 `exp_avg` (4) + fp32 `exp_avg_sq` (4), write fp32 master (4) + both
moments (8) + bf16 param (2) = **28 bytes**. The SOL uses that figure rather than a
dtype-derived one, because which tensors are fp32 is a property of the optimizer's
configuration and not of the parameter's own dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..hardware import GpuSpec
from ..sol import SolComponents
from .base import register_op

# Bytes of HBM traffic per parameter per step, derived above. Named rather than inlined
# because changing the optimizer's state dtypes changes this number and nothing else.
BYTES_PER_PARAMETER = 28

# The state layouts this table describes. `master_fp32` is Megatron's default: bf16
# parameters with an fp32 master copy and fp32 moments. A run with bf16 moments moves
# different traffic, so it is a separate slice rather than an interpolation.
OPTIMIZER_STATE_KINDS: tuple[str, ...] = ("master_fp32",)


@dataclass(frozen=True)
class OptimizerStepKernelOp:
    """The fused Adam update over one rank's parameter shard.

    Frozen for the same reason the other ops are: the registry holds one shared instance,
    so a mutation anywhere would change which columns every table is read against.
    """

    name = "optimizer_step"
    family = "optimizer"
    slice_columns = ("optimizer", "state_kind")
    axis_columns = ("param_count",)
    fits_floor = True
    # Training-only, eager, and bandwidth-bound at 2703 GB/s well before a launch cost
    # shows: a real shard is 10^8-10^9 parameters.
    graph_replayed = False

    def sol(
        self,
        spec: GpuSpec,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """
        Return the roofline of one optimizer step over `param_count` parameters.

        Pure bandwidth, no math term: Adam's arithmetic is a handful of FLOPs per parameter
        against 28 bytes of traffic, which puts it four orders of magnitude from the
        compute roof on any current GPU. Declaring `math_s=0.0` says that explicitly rather
        than computing a term that `max` would always discard.

        Args:
            spec (GpuSpec): Hardware the roofline is evaluated against.
            slice_key (tuple[Any, ...]): `(optimizer, state_kind)`.
            coords (tuple[float, ...]): `(param_count,)`.
            floor_s (float): Fitted launch floor in seconds, applied inside the `max`.

        Returns:
            SolComponents: A zero compute term, the bandwidth term, and the floor.

        Raises:
            ValueError: If `param_count` is not positive or `state_kind` is unknown.
        """
        state_kind = slice_key[1]
        if state_kind not in OPTIMIZER_STATE_KINDS:
            known = ", ".join(OPTIMIZER_STATE_KINDS)
            raise ValueError(
                f"Unknown optimizer state_kind {state_kind!r}. Known kinds: {known}."
            )
        param_count = float(coords[0])
        if param_count <= 0.0:
            raise ValueError(f"param_count must be positive, got {param_count!r}.")
        memory_s = param_count * BYTES_PER_PARAMETER / spec.memory_bandwidth_bytes_per_s
        return SolComponents(math_s=0.0, memory_s=memory_s, floor_s=floor_s)

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the `optimizer:` sweep subtree into an `OptimizerSweep`."""
        from ....profiler.kernel.sweep import OptimizerSweep

        return OptimizerSweep.from_dict(value)

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Time the fused Adam step across the parameter-count grid and write one table."""
        from ....profiler.kernel.runner.optimizer import OptimizerRunner

        return OptimizerRunner(model, config).run(writer_factory=writer_factory)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)


OPTIMIZER_STEP_OP = OptimizerStepKernelOp()
register_op(OPTIMIZER_STEP_OP)
