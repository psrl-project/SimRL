"""
Time Transformer Engine's fused Adam step over a parameter shard.

**The optimizer being timed must be the one Megatron imports.**
`megatron/core/optimizer/__init__.py:13` takes `FusedAdam` from
`transformer_engine.pytorch.optimizers`, and only falls back to apex and then to torch's
`AdamW`. Timing `torch.optim.AdamW(fused=True)` would measure a kernel this configuration
never runs, which is the error `docs/profiling/tensor-parallel.md` records for
`torch.distributed.all_reduce` and the Spec B design doc records for FlashAttention.

**The first `step()` call must not be timed.** `FusedAdam` allocates `exp_avg`,
`exp_avg_sq` and the fp32 master copy lazily, inside the first step, so that call includes
three allocations of 4 bytes per parameter. At 1.365B parameters that is 16 GiB of
allocation attributed to a kernel that moves 38 GB: the measured step would be roughly
double. One untimed `step()` precedes the warmup for that reason, and it is why this family
cannot share a timing loop with the others.

**One flat tensor, not many.** Megatron's `param_and_grad_buffer` coalesces parameters into
contiguous buckets, so the real kernel sees a handful of large tensors rather than hundreds
of small ones. `multi_tensor_adam` is launch-bound on a long list of small tensors and
bandwidth-bound on a short list of large ones, and the second is what a real step pays.
Measuring per-parameter tensors would report a cost that bucketing removes.

**A shard that does not fit must be refused, not recorded.** The first sweep of this op
included a 4.096B-parameter point, which needs ~115 GiB of parameter, gradient and
optimizer state against H20's 95 GiB. The allocating `step()` raised `OutOfMemoryError`
inside TE, the exception was swallowed, and the *timed* `step()` then iterated over an
empty state list and returned in 0.60 ms where 2.048B parameters took 21.56 ms. That row
implied 190 TB/s against the card's 4.0 TB/s and gave a utility of 47.6, and nothing in
the pipeline would have rejected it: a too-fast row is not a coverage error. The state
allocation is therefore verified after the first step, and a point that cannot be measured
raises with its own size rather than entering the table.
"""

from __future__ import annotations

from pathlib import Path

from ...schema import ProfileValidationError
from ..grid.optimizer import optimizer_param_counts
from .base import KernelRunner, WriterFactory

_LEARNING_RATE = 1.0e-4
# Adam keeps these three tensors per parameter. All three must exist and be the parameter's
# own length, or the timed step is iterating over a shorter list than the grid point claims.
_ADAM_STATE_TENSORS = ("exp_avg", "exp_avg_sq", "master_param")


def _require_allocated_state(optimizer: object, parameter: object, param_count: int) -> None:
    """Raise unless the optimizer allocated full-length state for `parameter`.

    The failure this catches is silent by construction: when the state allocation OOMs, TE
    leaves the state empty and the next `step()` returns almost immediately, so the sweep
    records a very *fast* row rather than an error. Nothing downstream rejects a fast row.
    A floor fit accepts it, and the holdout MRE improves because the point is easy to
    interpolate. Only the utility exceeding 1 gives it away, and that is a diagnostic a
    reader has to go looking for.
    """
    state = optimizer.state.get(parameter, {})  # type: ignore[attr-defined]
    missing = [name for name in _ADAM_STATE_TENSORS if name not in state]
    if missing:
        raise ProfileValidationError(
            f"Optimizer state {missing} was not allocated for a {param_count:,}-parameter "
            f"shard, so the timed step would measure an empty update. The shard needs "
            f"roughly {param_count * 18 / 2**30:.1f} GiB for parameters, gradients and "
            f"fp32 state. Drop this point from optimizer.param_counts or profile on a "
            f"larger card."
        )
    for name in _ADAM_STATE_TENSORS:
        allocated = state[name].numel()
        if allocated != param_count:
            raise ProfileValidationError(
                f"Optimizer state {name!r} holds {allocated:,} elements for a "
                f"{param_count:,}-parameter shard, so the timed step would measure a "
                f"different size than the grid point names."
            )


class OptimizerRunner(KernelRunner):
    """Times `FusedAdam.step()` across the parameter-count grid, state allocation verified."""

    def run(self, *, writer_factory: WriterFactory) -> tuple[Path, ...]:
        """
        Time the fused Adam step across the configured parameter-count grid.

        Args:
            writer_factory (WriterFactory): Builds a writer from
                `(framework, version, kernel_source)`.

        Returns:
            tuple[Path, ...]: The single written parquet path.

        Raises:
            ProfileValidationError: If a shard's optimizer state was not fully allocated,
                which means the timed step would measure an empty update.
        """
        import torch
        import transformer_engine
        from transformer_engine.pytorch.optimizers import FusedAdam

        from ..utils.timing import torch_dtype

        param_dtype = torch_dtype(self.config.measurement.dtype)
        rows: list[dict[str, object]] = []
        for param_count in optimizer_param_counts(self.config.sweep("optimizer_step")):
            latency_s = self._time_shard(param_count, param_dtype=param_dtype, adam=FusedAdam)
            torch.cuda.empty_cache()
            rows.append(
                self.row(
                    "optimizer_step",
                    optimizer="adam",
                    state_kind="master_fp32",
                    param_count=int(param_count),
                    latency_s=latency_s,
                )
            )

        writer = writer_factory(
            "megatron",
            str(transformer_engine.__version__),
            "transformer_engine.pytorch.optimizers.FusedAdam",
        )
        return (writer.write("optimizer_step", rows),)

    def _time_shard(self, param_count: int, *, param_dtype: object, adam: object) -> float:
        """
        Allocate one shard, prime its optimizer state, verify it, and time a step.

        The parameter and optimizer are function-locals so a shard worth tens of GiB is
        released when this returns, before the next and larger point allocates.

        Args:
            param_count (int): Parameters in this shard.
            param_dtype (object): The `torch.dtype` parameters and gradients are held in.
            adam (object): The `FusedAdam` class, passed in so the TE import stays in `run`.

        Returns:
            float: Mean per-step latency in seconds.

        Raises:
            ProfileValidationError: If the state allocation did not produce three
                full-length tensors.
        """
        import torch

        parameter = torch.nn.Parameter(
            torch.randn(param_count, dtype=param_dtype, device="cuda")
        )
        parameter.grad = torch.randn(param_count, dtype=param_dtype, device="cuda")
        optimizer = adam(  # type: ignore[operator]
            [parameter],
            lr=_LEARNING_RATE,
            master_weights=True,
            master_weight_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
        )
        # Untimed: this call allocates the fp32 master copy and both moments. Including it
        # would roughly double the reported cost at large shards.
        optimizer.step()
        torch.cuda.synchronize()
        _require_allocated_state(optimizer, parameter, param_count)
        return self.time(lambda o=optimizer: o.step())
