"""The FSDP2 static memory layout, measured rather than assumed."""

from __future__ import annotations

import pytest

from dryrun.cost.training.fsdp_inventory import fsdp_inventory


def test_fp32_master_dtype_gives_sixteen_p_over_world_size():
    """
    Ground truth: fp32 master dtype must yield exactly 16p/ws.

    Measured on 8×H20 with a 1.630B-parameter model: allocator reported 3.101 GiB
    against the analytic 16p/ws = 3.037 GiB, a ratio of 0.979.  The analytic figure
    must be exact to floating-point precision; the 2.1 % gap is allocator overhead.
    """
    parameter_count = 1_630_600_000
    world_size = 8
    inventory = fsdp_inventory(
        parameter_count=parameter_count,
        world_size=world_size,
        parameter_dtype="fp32",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    )
    expected = 16 * parameter_count / world_size
    assert inventory.total_bytes == pytest.approx(expected, rel=1e-9)
    # Also confirm the analytic value is within 2.1 % of the measurement.
    measured_gib = 3.101 * (1024 ** 3)
    assert inventory.total_bytes == pytest.approx(measured_gib, rel=0.021)


def test_total_bytes_invariant_to_parameter_dtype():
    """
    total_bytes must not change when only parameter_dtype changes.

    Under reshard_after_forward=True the all-gathered compute copy is freed after each
    layer and is not part of the static pool.  The module is built in the master dtype
    (optimizer_state_dtype), so the resident parameter shard is always sized by that
    dtype regardless of the compute dtype requested via parameter_dtype.
    """
    base_kwargs = dict(
        parameter_count=1_000_000_000,
        world_size=8,
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    )
    inv_fp32 = fsdp_inventory(**base_kwargs, parameter_dtype="fp32")
    inv_bf16 = fsdp_inventory(**base_kwargs, parameter_dtype="bf16")
    assert inv_fp32.total_bytes == pytest.approx(inv_bf16.total_bytes, rel=1e-9)


def test_compute_element_bytes_reflects_parameter_dtype():
    """compute_element_bytes is purely diagnostic and records the per-element compute dtype width."""
    inv_fp32 = fsdp_inventory(
        parameter_count=1_000_000_000,
        world_size=8,
        parameter_dtype="fp32",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    )
    inv_bf16 = fsdp_inventory(
        parameter_count=1_000_000_000,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    )
    assert inv_fp32.compute_element_bytes == 4
    assert inv_bf16.compute_element_bytes == 2


def test_gradients_are_sharded_unlike_megatron():
    """
    Megatron holds a full-size fp32 gradient buffer as the reduce-scatter source.

    FSDP2 reduce-scatters per layer into already-sharded gradients, so no full-size
    buffer exists.  Reusing the Megatron inventory here would overstate memory by
    roughly twofold.
    """
    parameter_count = 1_000_000_000
    inventory = fsdp_inventory(
        parameter_count=parameter_count,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    )
    assert inventory.gradients == pytest.approx(4 * parameter_count / 8)


def test_model_parameters_sized_by_master_dtype_not_compute_dtype():
    """
    The resident parameter shard is in the master dtype because the module is built
    in optimizer_state_dtype so AdamW moments inherit the right dtype.  There is no
    separate master-weight tensor; the module IS the master.
    """
    parameter_count = 1_000_000_000
    inventory = fsdp_inventory(
        parameter_count=parameter_count,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    )
    # model_parameters must be fp32 (4 bytes), not bf16 (2 bytes).
    assert inventory.model_parameters == pytest.approx(4 * parameter_count / 8)
    # Adam keeps two fp32 moments.
    assert inventory.optimizer_moments == pytest.approx(8 * parameter_count / 8)


def test_offloading_moves_optimizer_moments_off_device():
    parameter_count = 1_000_000_000
    resident = fsdp_inventory(
        parameter_count=parameter_count,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    )
    offloaded = fsdp_inventory(
        parameter_count=parameter_count,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=True,
    )
    assert offloaded.total_bytes < resident.total_bytes
    assert offloaded.optimizer_moments == 0
    assert offloaded.offloaded_bytes == pytest.approx(resident.optimizer_moments)
    assert offloaded.model_parameters == resident.model_parameters
    assert offloaded.gradients == resident.gradients


def test_larger_shard_degree_reduces_per_rank_bytes():
    def total(world_size: int) -> float:
        return fsdp_inventory(
            parameter_count=8_000_000_000,
            world_size=world_size,
            parameter_dtype="bf16",
            gradient_dtype="fp32",
            optimizer_state_dtype="fp32",
            optimizer_offload=False,
        ).total_bytes

    assert total(8) == pytest.approx(total(4) / 2)


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        fsdp_inventory(
            parameter_count=0,
            world_size=8,
            parameter_dtype="bf16",
            gradient_dtype="fp32",
            optimizer_state_dtype="fp32",
            optimizer_offload=False,
        )
    with pytest.raises(ValueError):
        fsdp_inventory(
            parameter_count=1000,
            world_size=0,
            parameter_dtype="bf16",
            gradient_dtype="fp32",
            optimizer_state_dtype="fp32",
            optimizer_offload=False,
        )
