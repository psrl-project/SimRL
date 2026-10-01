"""Torchrun entry point that profiles one FSDP2 training candidate.

verl is not a dependency: it is absent from this environment and depending on it
would pull Ray and its whole worker tree into a simulator repository while pinning
the measurement to a verl version. The FSDP semantics that matter for cost modelling
are the mesh, the wrapping granularity, the mixed-precision policy, and the offload
policy, and all four are torch APIs applied directly here.

Imports the FSDP training APIs at call time, so this module must never be imported
from the adapter package namespace.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch.distributed.device_mesh import DeviceMesh

from .execute import OOM_EXIT_CODE

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def _torch_dtype(name: str):
    import torch

    resolved = _DTYPES.get(name.lower())
    if resolved is None:
        raise ValueError(f"Unsupported dtype {name!r}.")
    return getattr(torch, resolved)


def read_spec(path: Path) -> dict[str, Any]:
    """Read one launch spec, refusing a schema this module does not understand."""
    with path.open(encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("schema_version") != 1:
        raise ValueError(f"Unsupported launch spec schema in {path}.")
    return spec


def build_mesh(spec: dict[str, Any]) -> DeviceMesh:
    """
    Build the device mesh one candidate describes.

    A one-dimensional mesh is pure FSDP. A two-dimensional mesh is HSDP: parameters
    are sharded within a row and replicated across rows, which is why the dimension
    names are the ones `fully_shard` recognizes.
    """
    from torch.distributed.device_mesh import init_device_mesh

    replicate = int(spec["mesh"]["replicate_size"])
    shard = int(spec["mesh"]["shard_size"])
    if replicate == 1:
        return init_device_mesh("cuda", (shard,), mesh_dim_names=("dp_shard",))
    return init_device_mesh(
        "cuda", (replicate, shard), mesh_dim_names=("dp_replicate", "dp_shard")
    )


def observed_mesh(mesh: DeviceMesh) -> dict[str, int]:
    """
    Read the mesh that was actually built, rather than echoing the request.

    `record_topology` exists so the collector can reject a record attributed to the
    wrong topology. Feeding it the spec it came from would make that check
    tautological, so the sizes are read back off the `DeviceMesh` itself. A
    one-dimensional mesh has no replicate dimension, which is a replicate size of one.
    """
    names = mesh.mesh_dim_names or ()
    return {
        "replicate_size": mesh["dp_replicate"].size() if "dp_replicate" in names else 1,
        "shard_size": mesh["dp_shard"].size(),
    }


def build_model(spec: dict[str, Any]) -> torch.nn.Module:
    """
    Construct the model from its HuggingFace config, with no weights loaded.

    Weights are irrelevant to a timing and memory measurement, and loading eight
    billion parameters per candidate would dominate the sweep. The architecture is
    what determines both, so the config alone is faithful.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(spec["model_path"])
    override = spec.get("num_layers_override")
    if override is not None:
        config.num_hidden_layers = int(override)
    # Caching is a decode-time concern and would only add allocator noise here.
    config.use_cache = False
    # The module dtype is the master-weight dtype: the storage the optimizer operates on
    # and from which the static inventory is sized. AdamW inherits its moment dtype from
    # the parameter dtype it is handed: verified experimentally: a bf16 module produces
    # bf16 moments, an fp32 module produces fp32 moments. `MixedPrecisionPolicy.param_dtype`
    # (`compute_dtype`) governs only the all-gathered parameters used during forward/backward;
    # those are transient and freed per block when `reshard_after_forward=True`, so they are
    # not part of the static pool at all. The analytic inventory (`16p/world_size`) was fitted
    # on a module built in fp32 (4p params + 4p grads + 8p Adam moments, all sharded), so
    # building in any other dtype would make the measured pool disagree with that formula.
    model = AutoModelForCausalLM.from_config(
        config,
        dtype=_torch_dtype(spec["optimizer_state_dtype"]),
        attn_implementation=spec["attn_implementation"],
    )
    if spec["gradient_checkpointing"]:
        # Non-reentrant checkpointing is what composes with FSDP2's per-layer hooks.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    # Train mode is pinned, not assumed. HuggingFace gates recompute on `self.training`,
    # so a model left in eval turns gradient checkpointing into a silent no-op while the
    # candidate still claims it: measured here as 2351.8 MB against 5573.1 MB peak, a
    # 2.4x memory error that looks entirely plausible in the fitted model.
    model.train()
    return model


def apply_fsdp(model: torch.nn.Module, spec: dict[str, Any], mesh: DeviceMesh):
    """
    Shard each transformer block, then the root module.

    Wrapping per block rather than once at the root is what makes the all-gather
    granular enough to overlap with compute; a single root wrap would gather every
    parameter before the first layer runs.
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    # `param_dtype` is the dtype the all-gathered parameters are *computed* in, which is
    # `compute_dtype`; `reduce_dtype` is what the gradient reduce-scatter accumulates in.
    # Passing `parameter_dtype` as `param_dtype` would conflate storage with compute and
    # silently make bf16-stored/fp32-computed candidates unmeasurable.
    policy = MixedPrecisionPolicy(
        param_dtype=_torch_dtype(spec["compute_dtype"]),
        reduce_dtype=_torch_dtype(spec["gradient_dtype"]),
        cast_forward_inputs=True,
    )
    kwargs: dict[str, Any] = {"mesh": mesh, "mp_policy": policy}
    if spec["optimizer_offload"]:
        kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=True)
    for block in model.model.layers:
        fully_shard(block, reshard_after_forward=bool(spec["reshard_after_forward"]), **kwargs)
    # The root is always `False`, whatever the candidate asks for. Torch's own guidance
    # is that resharding the root is wasted work: its parameters are needed immediately
    # when backward begins, so the reshard is undone by an all-gather microseconds later.
    # Measured here as 66.21 ms against 64.15 ms with one extra all-gather kernel and no
    # memory saving. `reshard_after_forward` is a swept axis, so letting the root inherit
    # it would add a fixed cost to one side of the very comparison the sweep isolates.
    fully_shard(model, reshard_after_forward=False, **kwargs)
    return model


def synthetic_batch(spec: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    """
    Build one uniform padded microbatch.

    Profiling measures uniform shapes only, and a candidate's `num_microbatches` are all
    this same shape, which is what gradient accumulation feeds a step. Forming *ragged*
    microbatches from real trajectory lengths is the simulate-side planner's job.
    `position_ids` therefore runs straight through each row rather than restarting, which
    is what would signal a packed batch to the attention kernel.
    """
    import torch

    n_seqs = int(spec["n_seqs_per_rank"])
    length = int(spec["sequence_length"])
    generator = torch.Generator(device="cpu").manual_seed(1234)
    input_ids = torch.randint(
        0, int(spec["vocab_size"]), (n_seqs, length), generator=generator
    ).to(device)
    position_ids = torch.arange(length, device=device).unsqueeze(0).expand(n_seqs, length)
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "position_ids": position_ids.contiguous(),
    }


def run_accumulated_step(model, batch, optimizer, num_microbatches: int) -> None:
    """
    Run one optimizer step over `num_microbatches` identically-shaped microbatches.

    This is gradient accumulation as real FSDP training performs it: forward and backward
    per microbatch, gradients summing into the same buffers, then a single
    `optimizer.step()`. The microbatch count is a swept axis because the latency design
    row is `[1, nmb*tokens, nmb]`, and at a fixed `nmb` the last column is collinear with
    the intercept, capping the design at rank two where the model needs three.

    The caller owns `zero_grad` and the timing bracket: the bracket must span the whole
    step, accumulation included, since `per_step_overhead` is defined as the cost paid
    once per step.

    Args:
        num_microbatches (int): How many microbatches accumulate into one step.
    """
    for _ in range(num_microbatches):
        model(**batch).loss.backward()
    optimizer.step()


def check_optimizer_state_dtype(optimizer, spec: dict[str, Any]) -> None:
    """
    Fail loudly when AdamW's moments disagree with the declared `optimizer_state_dtype`.

    The launcher does not yet honour the declared dtype: torch allocates the moments
    to match the parameters it was handed, but the analytic inventory sizes optimizer
    state *from that declaration*. A spec claiming fp32 while the moments are bf16 is a
    plausible-but-wrong measurement that nothing downstream can detect, so the mismatch
    is raised at profile time instead of becoming a wrong coefficient at fit time.

    Must be called after the first `optimizer.step()`: AdamW allocates `exp_avg` and
    `exp_avg_sq` lazily, so before it there is no state to inspect.
    """
    declared = _torch_dtype(spec["optimizer_state_dtype"])
    for state in optimizer.state.values():
        moment = state.get("exp_avg")
        if moment is None:
            continue
        if moment.dtype is not declared:
            raise ValueError(
                f"Optimizer state dtype mismatch: the launch spec declares "
                f"{spec['optimizer_state_dtype']!r} ({declared}) but the optimizer "
                f"allocated its moments as {moment.dtype}. The analytic inventory sizes "
                f"optimizer state from the declared dtype, so this would be measured "
                f"correctly and modelled wrongly."
            )
        return


def _measure_collectives(model, batch, optimizer, measurement, num_microbatches: int) -> None:
    """
    Run one step under the profiler and attribute its collective time.

    NCCL kernels carry the collective's name, verified on this host as
    `ncclDevKernel_AllGather_RING_LL(...)` and
    `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL(...)`, so matching the name after
    stripping underscores and case separates the all-gather that fetches parameters
    from the reduce-scatter that reduces gradients.

    `self_device_time_total` is microseconds, and it measures kernel *occupancy*, not
    exposed wall time. With `reshard_after_forward=True` most of the all-gather runs
    concurrently with compute on another stream, so these totals are an upper bound on
    what the collectives cost the step: a kernel-summation model that adds them to
    compute time would over-predict. They bound the communication term; they do not
    decompose the step.

    The profiled step accumulates the same `num_microbatches` the timed loop does, so the
    recorded occupancy describes a whole step rather than a fraction of one.

    Runs after the timed loop, never inside it. See `main`.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as profiler:
        run_accumulated_step(model, batch, optimizer, num_microbatches)
        torch.cuda.synchronize()
    totals = {"all_gather": 0.0, "reduce_scatter": 0.0}
    for event in profiler.key_averages():
        name = event.key.lower().replace("_", "")
        for kind in totals:
            if kind.replace("_", "") in name:
                totals[kind] += float(event.self_device_time_total) / 1e6
    measurement.record_collective_time(
        all_gather_s=totals["all_gather"],
        reduce_scatter_s=totals["reduce_scatter"],
    )


def main() -> None:
    """Profile one candidate and write its measurements next to the spec."""
    parser = argparse.ArgumentParser(
        prog="dryrun-fsdp-launch",
        description="Profile one FSDP2 training candidate under torchrun.",
    )
    parser.add_argument("--spec", required=True, help="Path to launch_spec.json.")
    args = parser.parse_args()

    spec = read_spec(Path(args.spec))
    run_dir = Path(spec["run_dir"])

    import torch
    import torch.distributed as dist

    from .measure import FsdpMeasurement, is_nccl_memory_exhaustion, write_oom_sidecar

    dist.init_process_group("nccl")
    # `LOCAL_RANK`, not the global rank modulo the device count. torchrun sets it per
    # node, and it already respects `CUDA_VISIBLE_DEVICES`; deriving the device from the
    # global rank would collide as soon as a node hosts an unequal share of the ranks.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    measurement = FsdpMeasurement(run_dir=run_dir, spec=spec)
    try:
        model = build_model(spec).cuda()
        parameter_count = sum(p.numel() for p in model.parameters())
        mesh = build_mesh(spec)
        model = apply_fsdp(model, spec, mesh)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        batch = synthetic_batch(spec, device="cuda")
        built = observed_mesh(mesh)
        measurement.record_topology(
            replicate_size=built["replicate_size"],
            shard_size=built["shard_size"],
            parameter_count=parameter_count,
        )
        measurement.on_train_start()
        num_microbatches = int(spec["num_microbatches"])
        for step in range(int(spec["train_iters"])):
            measurement.step_begin()
            optimizer.zero_grad(set_to_none=True)
            # The bracket spans the whole step, accumulation included, because
            # `per_step_overhead` is defined as the cost paid once per step.
            run_accumulated_step(model, batch, optimizer, num_microbatches)
            measurement.step_end()
            if step == 0:
                # AdamW allocates `exp_avg` and `exp_avg_sq` lazily on the first step, so
                # neither the static-pool diagnostic nor the dtype check can be taken
                # before this point: `len(optimizer.state)` is zero until now, and the
                # recorded pool would be a quarter to an eighth of the sharded inventory
                # Task 5 compares it against. Taken outside the timed region.
                measurement.record_static_pool()
                check_optimizer_state_dtype(optimizer, spec)
        # Collectives are profiled after the timed loop, never inside it. torch.profiler
        # inflates its own step by roughly an order of magnitude and leaves a ~2x tail on
        # the next one or two steps, and MAD dispersion is blind to a single outlier, so
        # an in-loop profiled step passes the smoke gate while poisoning retained[0].
        # Measured 2.73x against a 1.35x control on 8 ranks. Out of the loop the profiled
        # step needs no warmup of its own and can contaminate no retained sample.
        # The zero_grad matters: without it the final timed step's gradients accumulate
        # into this one and change the reduce-scatter's input state.
        optimizer.zero_grad(set_to_none=True)
        _measure_collectives(model, batch, optimizer, measurement, num_microbatches)
        measurement.on_train_end()
    except torch.OutOfMemoryError as error:
        # Caught by exact type. A driver-level CUDA failure or a NCCL error is a
        # different fault and must stay classified as a failure, because an
        # out-of-memory record can only raise the fitted memory floor.
        write_oom_sidecar(run_dir, spec, error, oom_source="torch_oom")
        raise SystemExit(OOM_EXIT_CODE) from error
    except dist.DistBackendError as error:
        # Exact-type catching is correct as a general rule because a false OOM record
        # raises the fitted memory floor, silently making the memory model conservative.
        # However, memory exhaustion can surface as a NCCL buffer-allocation failure:
        # training completes, then `dist.all_gather_object` inside `on_train_end` tries
        # to allocate a 10 MiB NCCL buffer and fails because the device is full. Measured
        # in the live sweep: 5 `failed` records at `shard_size=2, replicate_size=4`,
        # all dying with "Failed to CUDA calloc 10485760 bytes". Discarding those records
        # loses the high-memory lower bounds preferentially at narrow shard degrees,
        # which is exactly the end the memory model needs as a floor. Text-matching
        # narrows to genuine allocation failures rather than broadening to every
        # `DistBackendError` (which would admit timeouts, rendezvous failures, and real
        # network faults).
        if not is_nccl_memory_exhaustion(error):
            raise
        write_oom_sidecar(run_dir, spec, error, oom_source="nccl_alloc")
        raise SystemExit(OOM_EXIT_CODE) from error
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
