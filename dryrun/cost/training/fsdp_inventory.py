"""Static parameter, gradient, and optimizer state inventory for FSDP2.

Separate from `inventory.py` because the sharding rule genuinely differs, and the
difference is a factor of two rather than a detail.

Megatron's distributed optimizer shards optimizer *state* only: parameters are held
at full model-parallel-local size and the fp32 gradient buffer is the reduce-scatter
source, so every rank holds it whole. Measured at Qwen3-8B TP1/PP1/DP8 that is
`2p + 4p + 12p/DP`, about 59 GiB.

FSDP2 shards all three. Each `fully_shard` unit reduce-scatters into an
already-sharded gradient, so no full-size buffer exists anywhere. Measured on 8 ranks
with a 1.630B-parameter model under AdamW, the allocator reported 3.101 GiB against
`16p/world_size = 3.037 GiB` (ratio 0.979): parameters, gradients, and both Adam
moments all divided by the world size.

Applying the Megatron inventory to FSDP would therefore overstate memory roughly
twofold, which is exactly the kind of error a fitted residual absorbs silently.

Physical layout under FSDP2
----------------------------
The FSDP launcher builds the module in the **master dtype** (`optimizer_state_dtype`)
so that AdamW's `exp_avg`/`exp_avg_sq` buffers inherit the same dtype as the
parameters they track: a bf16 module cannot carry fp32 optimizer state.  That means
there is exactly **one** resident parameter copy on device, sized by the master dtype.

`MixedPrecisionPolicy(param_dtype=compute_dtype)` creates a narrow compute copy used
for the all-gather during forward and backward.  With `reshard_after_forward=True`
that copy is freed after each layer, so it is **transient** and is not part of the
static pool counted here.  Contrast this with Megatron's mixed-precision layout, where
the narrow parameters are resident at full local size and a separate fp32 master lives
alongside them as a distinct tensor.

`parameter_dtype` documents the compute dtype and is recorded for diagnostics but
contributes **no bytes** to `total_bytes`.

dtype conventions
-----------------
`optimizer_state_dtype` (the master dtype) sizes the resident per-rank parameter shard
(`model_parameters`) and both Adam moments.

`gradient_dtype` sizes the resident sharded gradient buffer.

`parameter_dtype` is the transient all-gathered compute dtype.  It is exposed via
`compute_element_bytes` for diagnostics but is NOT included in `total_bytes`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .inventory import dtype_bytes


@dataclass(frozen=True)
class FsdpInventory:
    """Break down per-rank static state bytes under FSDP2.

    The resident parameter shard is sized by the master dtype (`optimizer_state_dtype`),
    not by `parameter_dtype`.  There is no separate master-weight tensor because the
    module is built in the master dtype from the start: see module docstring for the
    physical reasoning.

    `compute_element_bytes` records the per-element byte width of the transient
    all-gathered compute copy.  It is a per-element width stored for diagnostics
    (e.g. peak activation accounting) so callers cannot mistake it for a resident
    contribution; it does not feed into `total_bytes`.  It is deliberately excluded
    from `total_bytes` because `reshard_after_forward=True` frees the all-gathered
    compute copy after each transformer block, making it transient rather than part
    of the static resident pool.
    """

    model_parameters: float
    gradients: float
    optimizer_moments: float
    # Per-element byte width of the transient compute dtype (e.g. 2 for bf16, 4 for fp32).
    # Stored for diagnostics (e.g. peak activation accounting) but is NOT part of total_bytes
    # because the compute copy is freed per layer under reshard_after_forward=True.
    compute_element_bytes: int
    # Bytes moved to host memory by CPUOffloadPolicy.  Tracked rather than dropped so
    # a prediction can explain why an offloaded cell needs less device memory.
    offloaded_bytes: float = 0.0

    @property
    def total_bytes(self) -> float:
        """Return the device-resident total.

        compute_element_bytes is intentionally excluded: the all-gathered compute copy is
        transient under reshard_after_forward=True and does not belong to the static pool.
        """
        return self.model_parameters + self.gradients + self.optimizer_moments

    def to_dict(self) -> dict[str, float]:
        return {
            "model_parameters": self.model_parameters,
            "gradients": self.gradients,
            "optimizer_moments": self.optimizer_moments,
            "compute_element_bytes": self.compute_element_bytes,
            "offloaded_bytes": self.offloaded_bytes,
            "total_bytes": self.total_bytes,
        }


def fsdp_inventory(
    *,
    parameter_count: int,
    world_size: int,
    parameter_dtype: str,
    gradient_dtype: str,
    optimizer_state_dtype: str,
    optimizer_offload: bool,
) -> FsdpInventory:
    """
    Compute per-rank static state bytes under FSDP2.

    Every term is divided by the world size.  `world_size` is the whole mesh, both its
    shard and replicate extents: replication duplicates a shard across rows rather than
    widening it, so a rank's share depends only on how many ranks there are in total.

    Physical layout
    ~~~~~~~~~~~~~~~
    The launcher builds the module in `optimizer_state_dtype` (the master dtype) because
    AdamW moments must share dtype with the tensors they track.  There is therefore one
    resident parameter copy, sized by the master dtype rather than by
    `parameter_dtype`.

    There is no second, separate master-weight tensor.  Megatron keeps one because its
    module resides in the narrow dtype and the master is a separate fp32 buffer.  FSDP2
    has no such buffer: the module itself is already the master.

    The transient all-gathered compute copy produced by
    `MixedPrecisionPolicy(param_dtype=parameter_dtype)` with `reshard_after_forward=True`
    is freed after each layer.  It is NOT counted here; `compute_element_bytes` is exposed
    for downstream activation accounting only.

    Consequently `total_bytes` is invariant to `parameter_dtype`: changing the compute
    dtype does not touch any resident tensor.

    Contrast with Megatron
    ~~~~~~~~~~~~~~~~~~~~~~
    Megatron holds a full-size fp32 gradient buffer as the reduce-scatter source, so
    every rank holds it whole.  FSDP2 reduce-scatters per layer into already-sharded
    gradients; no full-size buffer exists anywhere.  Reusing the Megatron inventory for
    FSDP would overstate memory by roughly twofold.

    `optimizer_offload`
    ~~~~~~~~~~~~~~~~~~~
    CPUOffloadPolicy keeps the sharded optimizer state (moments) in pinned host memory.
    The parameter and gradient shards stay resident because forward and backward need
    them on device.  `offloaded_bytes` reports exactly what left the device.
    """
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive.")
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    local = parameter_count / world_size
    # The module is built in the master dtype, so the resident parameter shard is
    # sized here.  parameter_dtype sizes only the transient compute copy.
    master_bytes = dtype_bytes(optimizer_state_dtype)
    gradient_bytes = dtype_bytes(gradient_dtype)
    param_bytes = dtype_bytes(parameter_dtype)
    # Adam keeps a first and a second moment, both in the master dtype.
    moments = local * 2 * master_bytes
    if optimizer_offload:
        # CPUOffloadPolicy offloads the optimizer moments to pinned host memory.
        # The parameter shard stays resident because forward/backward need it.
        return FsdpInventory(
            model_parameters=local * master_bytes,
            gradients=local * gradient_bytes,
            optimizer_moments=0.0,
            compute_element_bytes=param_bytes,
            offloaded_bytes=moments,
        )
    return FsdpInventory(
        model_parameters=local * master_bytes,
        gradients=local * gradient_bytes,
        optimizer_moments=moments,
        compute_element_bytes=param_bytes,
    )
