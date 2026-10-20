"""YAML-driven official vLLM and Megatron-Bridge profiling workflows."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .._logging import setup_logging, srl_logger
from ..profiler.adapters import (
    MegatronBridgeConfig,
    VLLMSweepConfig,
    collect_bridge_records,
    collect_vllm_records,
    format_vllm_job_command,
    prepare_bridge_candidate,
    prepare_vllm_jobs,
    read_candidate_manifest,
    run_bridge_candidate,
    run_vllm_profiles,
    write_candidate_manifest,
)
from ..profiler.adapters.fsdp import (
    FsdpProfileConfig,
    collect_fsdp_records,
    prepare_fsdp_candidate,
    run_fsdp_candidate,
)
from ..profiler.adapters.fsdp import (
    read_candidate_manifest as read_fsdp_manifest,
)
from ..profiler.adapters.fsdp import (
    write_candidate_manifest as write_fsdp_manifest,
)
from ..profiler.fsdp_search_space import enumerate_fsdp_candidates
from ..profiler.io import write_records
from ..profiler.kernel import KernelProfileConfig
from ..profiler.search_space import (
    ModelConstraints,
    TrainingCandidate,
    enumerate_training_candidates,
)


def _load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    value = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {config_path}.")
    return value, config_path.parent


def _phases(config: VLLMSweepConfig, requested: str) -> list[str]:
    if requested == "all":
        return list(config.phase_parameters)
    if requested not in config.phase_parameters:
        raise ValueError(f"Phase {requested!r} is not configured.")
    return [requested]


def _vllm_config(path: str | Path) -> VLLMSweepConfig:
    value, base_dir = _load_config(path)
    return VLLMSweepConfig.from_dict(value, base_dir=base_dir)


def _kernel_config(path: str | Path) -> KernelProfileConfig:
    value, base_dir = _load_config(path)
    return KernelProfileConfig.from_dict(value, base_dir=base_dir)


def _max_parallel_arg(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "max parallel must be 'auto' or a positive integer"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("max parallel must be positive")
    return parsed


def _bridge_config(path: str | Path) -> tuple[MegatronBridgeConfig, dict[str, Any]]:
    value, base_dir = _load_config(path)
    return MegatronBridgeConfig.from_dict(value, base_dir=base_dir), value


def _candidate_path(config: MegatronBridgeConfig, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return config.output_dir / "control" / "megatron_bridge" / "candidates.json"


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def _generate_bridge_candidates(
    config: MegatronBridgeConfig,
    raw_config: dict[str, Any],
) -> list[TrainingCandidate]:
    search = raw_config.get("search_space", {})
    if config.model.num_layers is None:
        raise ValueError("HuggingFace config must provide num_layers for candidate generation.")
    if config.model.num_attention_heads is None:
        raise ValueError("HuggingFace config must provide num_attention_heads for candidate generation.")
    if config.model.num_query_groups is None:
        raise ValueError("HuggingFace config must provide num_query_groups for candidate generation.")
    constraints = ModelConstraints(
        num_layers=int(config.model.num_layers),
        num_attention_heads=int(config.model.num_attention_heads),
        num_query_groups=int(config.model.num_query_groups),
    )
    world_size = int(search.get("world_size", config.hardware.gpu_count))
    micro_batch_size = int(search.get("micro_batch_size", 1))
    pipeline_parallel_sizes = _as_int_list(search["pipeline_parallel_sizes"])
    if "num_microbatches" in search:
        num_microbatches = _as_int_list(search["num_microbatches"])
    else:
        num_microbatches = sorted(set([1, *pipeline_parallel_sizes]))
    if "sequence_lengths" in search:
        sequence_shapes = [
            [int(length)] * micro_batch_size
            for length in search["sequence_lengths"]
        ]
    elif "sequence_shapes" in search:
        sequence_shapes = [
            [int(length) for length in shape]
            for shape in search["sequence_shapes"]
        ]
    else:
        raise ValueError("search_space.sequence_lengths is required.")
    recompute = search.get("recompute", search.get("recompute_modes", "full"))
    recompute_modes = [str(recompute)] if isinstance(recompute, str) else [str(item) for item in recompute]
    return enumerate_training_candidates(
        world_sizes=[world_size],
        tensor_parallel_sizes=_as_int_list(search["tensor_parallel_sizes"]),
        pipeline_parallel_sizes=pipeline_parallel_sizes,
        context_parallel_sizes=_as_int_list(search.get("context_parallel_sizes", [1])),
        micro_batch_sizes=[micro_batch_size],
        num_microbatches=num_microbatches,
        sequence_shapes=sequence_shapes,
        schedules=[str(item) for item in search.get("schedules", ["1f1b"])],
        recompute_modes=recompute_modes,
        constraints=constraints,
        repeats=1,
        # THD packs sequences behind cu_seqlens the way verl's Megatron engine does
        # with use_remove_padding=True. A cost model meant to predict verl steps has
        # to be fitted on THD measurements: BSHD charges attention at the padded
        # length, so its quadratic term is a different quantity.
        packing=str(search.get("packing", "bshd")),
        # Absent, DP absorbs the GPUs TP and PP leave over, which confounds any
        # sweep meant to isolate one of them. See `enumerate_training_candidates`.
        data_parallel_sizes=(
            _as_int_list(search["data_parallel_sizes"])
            if "data_parallel_sizes" in search
            else None
        ),
        # Expert parallelism for a MoE block. Absent means [1], every rank holding every
        # expert, which is the only valid value for a dense model.
        expert_parallel_sizes=(
            _as_int_list(search["expert_parallel_sizes"])
            if "expert_parallel_sizes" in search
            else None
        ),
    )


def _select_candidates(
    candidates: list[TrainingCandidate],
    *,
    candidate_id: str | None,
    index: int | None,
) -> list[TrainingCandidate]:
    if candidate_id is not None:
        selected = [candidate for candidate in candidates if candidate.candidate_id == candidate_id]
        if not selected:
            raise ValueError(f"Unknown candidate_id {candidate_id!r}.")
        return selected
    if index is not None:
        try:
            return [candidates[index]]
        except IndexError as exc:
            raise ValueError(f"Candidate index {index} is out of range.") from exc
    return candidates


def cmd_vllm_generate(args: argparse.Namespace) -> None:
    config = _vllm_config(args.config)
    jobs = prepare_vllm_jobs(
        config,
        _phases(config, args.phase),
    )
    for job in jobs:
        srl_logger.info(
            f"Prepared vLLM {job.phase!r} {job.measurement} sweep: "
            f"{format_vllm_job_command(job)}."
        )


def cmd_vllm_run(args: argparse.Namespace) -> None:
    config = _vllm_config(args.config)
    max_parallel = (
        config.max_parallel
        if args.max_parallel is None
        else args.max_parallel
    )
    run_vllm_profiles(
        config,
        _phases(config, args.phase),
        max_parallel=max_parallel,
        dry_run=args.dry_run,
        resume=args.resume,
    )


def cmd_vllm_collect(args: argparse.Namespace) -> None:
    config = _vllm_config(args.config)
    records = [
        record
        for phase in _phases(config, args.phase)
        for record in collect_vllm_records(config, phase)
    ]
    output = Path(args.output) if args.output else config.output_dir / "profiles" / "vllm.jsonl"
    write_records(output, records)
    srl_logger.info(f"Wrote {len(records)} normalized vLLM observations to {output}.")


def cmd_kernels_run(args: argparse.Namespace) -> None:
    """Profile every registered kernel op, or one family of them, directly on the GPU."""
    import torch

    from ..cost.kernel.ops.base import families, ops_for_family
    from ..profiler.kernel.measurements import PerfTableWriter

    config = _kernel_config(args.config)
    device = torch.cuda.get_device_name(0)
    table_root = config.table_root()
    if args.family == "all":
        selected = tuple(
            family
            for family in families()
            if any(op.name in config.sweeps for op in ops_for_family(family))
        )
        for family in sorted(set(families()) - set(selected)):
            srl_logger.info(f"Skipping family {family!r}: the config sweeps none of its ops.")
    else:
        selected = (args.family,)

    def writer_factory(framework: str, version: str, kernel_source: str) -> PerfTableWriter:
        return PerfTableWriter(
            output_root=table_root,
            device=device,
            framework=framework,
            version=version,
            kernel_source=kernel_source,
        )

    written: list[Path] = []
    for family in selected:
        ops = ops_for_family(family)
        if not ops:
            raise SystemExit(f"No registered kernel ops in family {family!r}.")
        # One call per family, not per op. Both attention phases come out of a single backend
        # sweep that builds one KV cache and times both phases against it, so calling per op
        # would measure the grid twice and could pick a different backend for each phase.
        written.extend(ops[0].measure(config.model, config, writer_factory=writer_factory))

    for path in written:
        srl_logger.info(f"Wrote kernel perf table {path}.")


def cmd_kernels_fit_floor(args: argparse.Namespace) -> None:
    """Scan candidate floors and write floors.json into every floor-taking version directory."""
    from ..cost.kernel.floor import CANDIDATE_FLOORS_US, fit_floors_report

    config = _kernel_config(args.config)
    candidates_us = tuple(float(v) for v in args.candidates_us) if args.candidates_us else CANDIDATE_FLOORS_US
    written = fit_floors_report(
        config.table_root(),
        gpu=config.gpu(),
        candidates_us=candidates_us,
    )
    if not written:
        raise SystemExit(
            f"No floor-taking perf tables found under {config.table_root()}. "
            "Run srl-profile kernels run first."
        )
    for path in written:
        srl_logger.info(f"Wrote {path}.")


def cmd_kernels_verify_shapes(args: argparse.Namespace) -> None:
    """Compare a family's declared kernel shapes against the checkpoint's real module graph."""
    from ..profiler.hf import load_hf_model_descriptor
    from ..profiler.kernel.introspect import verify_shapes

    if args.model_path:
        model_path = args.model_path
    elif args.config:
        model_path = _kernel_config(args.config).model.name
    else:
        raise SystemExit("Pass --model-path or --config so the checkpoint can be located.")

    descriptor = load_hf_model_descriptor(model_path)
    report = verify_shapes(model_path, descriptor)
    srl_logger.info(
        f"{report.architecture} -> family {report.family!r}, {report.layer_count} layers."
    )
    for item in report.comparisons:
        observed = "absent" if item.observed is None else str(item.observed)
        srl_logger.info(
            f"  {item.name}: declared {item.declared}, observed {observed} "
            f"{'OK' if item.matches else 'MISMATCH'}."
        )
    if report.heterogeneous_layers:
        srl_logger.error(
            f"Layers {list(report.heterogeneous_layers)} do not share layer "
            f"{min(report.heterogeneous_layers) - 1 if report.heterogeneous_layers else 0}'s shape "
            f"set. This family multiplies one layer's shapes by the layer count, so a "
            f"heterogeneous model would be priced as though every layer matched the first."
        )
    if not report.ok:
        raise SystemExit(
            f"Declared kernel shapes do not match {model_path}'s module graph. Either the family "
            f"is wrong for this architecture or the checkpoint is not what it claims."
        )
    srl_logger.info("Declared shapes match the module graph.")


def cmd_kernels_holdout(args: argparse.Namespace) -> None:
    """Report held-out interpolation accuracy per measured kernel family."""
    from ..cost.kernel.holdout import holdout_report

    config = _kernel_config(args.config)
    if args.fraction < 1.0:
        # A subset can make a bad surface look good by sampling only the well-fit points.
        # Reporting a sampled figure as the surface's accuracy is the mistake this guards against.
        srl_logger.warning(
            f"--fraction {args.fraction} scores only a subset of the bracketed points, so the "
            f"MRE below understates the surface's real error. Do not report it as the gate "
            f"result. Rerun with --fraction 1.0."
        )
    results = holdout_report(
        config.table_root(),
        gpu=config.gpu(),
        fraction=args.fraction,
        seed=args.seed,
    )
    if not results:
        raise SystemExit(
            f"No kernel perf tables under {config.table_root()}. Run srl-profile kernels run first."
        )
    failed = []
    for result in results:
        if result.point_count == 0:
            srl_logger.warning(
                f"{result.op_name} [{result.backend}]: 0 held-out points scored, "
                f"{result.skipped_slices} slices skipped, "
                f"utility {result.utility_min:.3f}-{result.utility_max:.3f}. "
                "This is a coverage problem (sweep grid too sparse or degenerate), not an accuracy problem."
            )
        else:
            srl_logger.info(
                f"{result.op_name} [{result.backend}]: {result.point_count} held-out points, "
                f"MRE {result.mre_percent:.2f}%, p90 {result.p90_percent:.2f}%, "
                f"max {result.max_percent:.2f}% at {result.worst_coords}, "
                f"utility {result.utility_min:.3f}-{result.utility_max:.3f}, "
                f"{result.skipped_slices} slices skipped."
            )
        if not result.passes(args.gate):
            failed.append(result)
    output = Path(args.output) if args.output else config.table_root() / "holdout_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    srl_logger.info(f"Wrote holdout report to {output}.")
    if failed:
        coverage_failed = [r for r in failed if r.point_count == 0]
        accuracy_failed = [r for r in failed if r.point_count > 0]
        parts = []
        if accuracy_failed:
            names = ", ".join(f"{r.op_name} [{r.backend}]" for r in accuracy_failed)
            parts.append(f"Held-out MRE exceeded the {args.gate:.1f}% gate for: {names}.")
        if coverage_failed:
            names = ", ".join(f"{r.op_name} [{r.backend}]" for r in coverage_failed)
            parts.append(
                f"No held-out point was scorable for: {names}. "
                "This is a coverage problem, not an accuracy problem."
            )
        raise SystemExit(" ".join(parts))


def cmd_kernels_validate(args: argparse.Namespace) -> None:
    """Score the composed cost model against a real vLLM baseline, per phase."""
    from ..cost.kernel.database import KernelDatabase
    from ..cost.kernel.model_ops import ModelOps
    from ..cost.kernel.validate import load_baseline, score_model
    from ..cost.rollout.composed import ComposedRollout

    config = _kernel_config(args.config)
    buckets = load_baseline(args.baseline, min_samples=args.min_samples)
    if not buckets:
        raise SystemExit(f"No usable iterations in {args.baseline}.")
    # Warn about a phase the baseline holds but `--min-samples` dropped entirely. The default
    # of 20 suppresses noisy one-off buckets, but on a decode-dominated baseline it can also
    # hide prefill and mixed completely, and a report naming only decode reads as "prefill was
    # never validated" rather than "prefill was filtered out".
    scored_phases = {bucket.phase for bucket in buckets}
    for phase in sorted({bucket.phase for bucket in load_baseline(args.baseline)} - scored_phases):
        srl_logger.warning(
            f"Phase {phase} appears in {args.baseline}, but no bucket holds "
            f"{args.min_samples} or more iterations, so it is not scored below. "
            f"Pass --min-samples 1 to score it."
        )
    database = KernelDatabase.from_root(
        config.table_root(), gpu=config.gpu(), utility_surface=args.surface
    )
    ops = ModelOps.from_checkpoint(
        args.model_path or config.model.name, tensor_parallel=config.tensor_parallel
    )
    model = ComposedRollout(database, ops, tensor_parallel=config.tensor_parallel)
    results = score_model(model, buckets)
    failed = []
    for result in results:
        srl_logger.info(
            f"{result.phase}: {result.bucket_count} buckets, "
            f"MRE {result.mre_percent:.2f}%, p90 {result.p90_percent:.2f}%, "
            f"max {result.max_percent:.2f}% at reqs/ctxsum {result.worst_bucket}, "
            f"median ratio {result.median_ratio:.4f}."
        )
        if not result.passes(args.gate):
            failed.append(result)
    output = Path(args.output) if args.output else config.table_root() / "validate_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    srl_logger.info(f"Wrote validation report to {output}.")
    if failed:
        names = ", ".join(f"{r.phase} ({r.mre_percent:.2f}%)" for r in failed)
        raise SystemExit(f"Composed-model MRE exceeded the {args.gate:.1f}% gate for: {names}.")


def cmd_megatron_generate(args: argparse.Namespace) -> None:
    config, raw_config = _bridge_config(args.config)
    candidates = _generate_bridge_candidates(config, raw_config)
    output = _candidate_path(config, args.candidates)
    write_candidate_manifest(output, candidates)
    srl_logger.info(f"Wrote {len(candidates)} constrained candidates to {output}.")


def cmd_megatron_run(args: argparse.Namespace) -> None:
    config, _ = _bridge_config(args.config)
    path = _candidate_path(config, args.candidates)
    candidates = _select_candidates(
        read_candidate_manifest(path),
        candidate_id=args.candidate_id,
        index=args.index,
    )
    failures = 0
    for candidate in candidates:
        if args.dry_run:
            command, _ = prepare_bridge_candidate(config, candidate)
            srl_logger.info(
                f"Prepared Megatron-Bridge candidate {candidate.candidate_id!r}: "
                f"{shlex.join(command)}."
            )
            continue
        try:
            run_bridge_candidate(
                config,
                candidate,
                resume=args.resume,
                nnodes=args.nnodes,
                node_rank=args.node_rank,
                master_addr=args.master_addr,
                master_port=args.master_port,
            )
        except subprocess.CalledProcessError as exc:
            failures += 1
            srl_logger.warning(
                f"Candidate {candidate.candidate_id!r} exited with code {exc.returncode}. "
                "Its status and raw log were preserved."
            )
    if failures:
        srl_logger.warning(f"{failures} Megatron-Bridge candidates failed or ran out of memory.")


def cmd_megatron_collect(args: argparse.Namespace) -> None:
    config, _ = _bridge_config(args.config)
    path = _candidate_path(config, args.candidates)
    candidates = _select_candidates(
        read_candidate_manifest(path),
        candidate_id=args.candidate_id,
        index=args.index,
    )
    records = collect_bridge_records(config, candidates)
    output = (
        Path(args.output)
        if args.output
        else config.output_dir / "profiles" / "megatron_bridge.jsonl"
    )
    write_records(output, records)
    srl_logger.info(
        f"Wrote {len(records)} normalized Megatron-Bridge observations to {output}."
    )


def _fsdp_config(path: str | Path) -> tuple[FsdpProfileConfig, dict[str, Any]]:
    value, base_dir = _load_config(path)
    return FsdpProfileConfig.from_dict(value, base_dir=base_dir), value


def _fsdp_candidate_path(config: FsdpProfileConfig, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return config.output_dir / "control" / "fsdp" / "candidates.json"


def _generate_fsdp_candidates(
    config: FsdpProfileConfig,
    raw_config: dict[str, Any],
) -> list:
    search = raw_config.get("search_space", {})
    world_size = int(search.get("world_size", config.hardware.gpu_count))
    return enumerate_fsdp_candidates(
        world_size=world_size,
        shard_sizes=_as_int_list(search["shard_sizes"]),
        n_seqs_per_rank=_as_int_list(search["n_seqs_per_rank"]),
        sequence_lengths=_as_int_list(search["sequence_lengths"]),
        gradient_checkpointing=[bool(item) for item in search.get("gradient_checkpointing", [True])],
        optimizer_offload=[bool(item) for item in search.get("optimizer_offload", [False])],
        reshard_after_forward=[bool(item) for item in search.get("reshard_after_forward", [True])],
        num_microbatches=_as_int_list(search["num_microbatches"]) if "num_microbatches" in search else None,
    )


def cmd_fsdp_generate(args: argparse.Namespace) -> None:
    config, raw_config = _fsdp_config(args.config)
    candidates = _generate_fsdp_candidates(config, raw_config)
    output = _fsdp_candidate_path(config, args.candidates)
    write_fsdp_manifest(output, candidates)
    srl_logger.info(f"Wrote {len(candidates)} FSDP candidates to {output}.")


def cmd_fsdp_run(args: argparse.Namespace) -> None:
    config, _ = _fsdp_config(args.config)
    path = _fsdp_candidate_path(config, args.candidates)
    all_candidates = read_fsdp_manifest(path)
    candidates = _select_candidates(
        all_candidates,
        candidate_id=args.candidate_id,
        index=args.index,
    )
    failures = 0
    for candidate in candidates:
        if args.dry_run:
            command, _ = prepare_fsdp_candidate(config, candidate)
            srl_logger.info(
                f"Prepared FSDP candidate {candidate.candidate_id!r}: "
                f"{shlex.join(command)}."
            )
            continue
        try:
            run_fsdp_candidate(config, candidate, resume=args.resume)
        except subprocess.CalledProcessError as exc:
            failures += 1
            srl_logger.warning(
                f"Candidate {candidate.candidate_id!r} exited with code {exc.returncode}. "
                "Its status and raw log were preserved."
            )
    if failures:
        srl_logger.warning(f"{failures} FSDP candidates failed or ran out of memory.")


def cmd_fsdp_collect(args: argparse.Namespace) -> None:
    config, _ = _fsdp_config(args.config)
    path = _fsdp_candidate_path(config, args.candidates)
    candidates = _select_candidates(
        read_fsdp_manifest(path),
        candidate_id=args.candidate_id,
        index=args.index,
    )
    records = collect_fsdp_records(config, candidates)
    output = (
        Path(args.output)
        if args.output
        else config.output_dir / "profiles" / "fsdp.jsonl"
    )
    write_records(output, records)
    srl_logger.info(f"Wrote {len(records)} normalized FSDP observations to {output}.")


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML workflow configuration.")


def _kernel_families() -> tuple[str, ...]:
    """Return every registered kernel family, for the --family choices."""
    # Importing `ops.base` runs `ops/__init__.py`, which imports every op module and so
    # populates the registry. Nothing under `srl.profiler` is touched, which keeps
    # building the parser free of the GPU stack.
    from ..cost.kernel.ops.base import families

    return families()


def _add_candidate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidates", help="Candidate manifest path.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--candidate-id", help="Run or collect one candidate ID.")
    selection.add_argument("--index", type=int, help="Run or collect one candidate index.")


def cmd_kv_probe(args: argparse.Namespace) -> None:
    """Measure and cache the KV capacity a real vLLM engine allocates."""
    from ..cost.rollout.kv_probe import DEFAULT_CACHE_PATH, probe_kv_capacity

    if args.refresh:
        # For when the answer changed under a stable key: a new driver, a rebuilt
        # vLLM. Clearing the file is simpler than recomputing one key, and the
        # dropped entries are cheap to re-measure.
        cache = Path(os.environ.get("SIMRL_KV_CACHE", str(DEFAULT_CACHE_PATH)))
        if cache.is_file():
            cache.unlink()
            srl_logger.info("Cleared KV cache at %s for --refresh.", cache)

    capacity = probe_kv_capacity(
        args.model,
        tensor_parallel=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        block_size=args.block_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_name=args.gpu_name,
    )
    print(
        f"tp={args.tp}: {capacity.num_gpu_blocks} blocks x {capacity.block_size} tokens "
        f"= {capacity.total_tokens:,} tokens  [{capacity.source}]"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the profiling command tree for tests and console entrypoints."""
    parser = argparse.ArgumentParser(
        prog="srl-profile",
        description="Run official profilers and normalize their raw artifacts.",
    )
    backends = parser.add_subparsers(dest="backend_name", required=True)

    kv = backends.add_parser(
        "kv-probe",
        help="Measure real KV capacity by starting vLLM once, and cache it.",
        description=(
            "Start a vLLM engine, read back the KV block count it allocates, and "
            "write it to the on-disk cache that `srl simulate` reads. Run this "
            "ahead of time so a simulation never has to wait on an engine start, "
            "or on a GPU box to populate a cache a GPU-less box will consume."
        ),
    )
    kv.add_argument("--model", required=True)
    kv.add_argument("--tp", type=int, default=1, help="Tensor-parallel degree.")
    kv.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    kv.add_argument("--max-model-len", type=int, default=32768)
    kv.add_argument("--dtype", default="bfloat16")
    kv.add_argument("--max-num-batched-tokens", type=int, default=0)
    kv.add_argument("--max-num-seqs", type=int, default=0)
    kv.add_argument("--gpu-name", default="unknown")
    kv.add_argument("--block-size", type=int, default=0)
    kv.add_argument(
        "--refresh",
        action="store_true",
        help="Re-probe and overwrite even when the key is already cached.",
    )
    kv.set_defaults(func=cmd_kv_probe)

    vllm = backends.add_parser("vllm", help="Use vLLM bench sweep serve.")
    vllm_workflows = vllm.add_subparsers(dest="workflow", required=True)
    for name, handler in (
        ("generate", cmd_vllm_generate),
        ("run", cmd_vllm_run),
        ("collect", cmd_vllm_collect),
    ):
        workflow = vllm_workflows.add_parser(name)
        _add_config_argument(workflow)
        workflow.add_argument("--phase", choices=["all", "prefill", "decode"], default="all")
        if name == "run":
            workflow.add_argument("--dry-run", action="store_true")
            workflow.add_argument("--resume", action="store_true")
            workflow.add_argument(
                "--max-parallel",
                type=_max_parallel_arg,
                default=None,
                metavar="auto|N",
                help=(
                    "Maximum concurrent vLLM servers. Defaults to run.max_parallel "
                    "or all available GPU slots."
                ),
            )
        if name == "collect":
            workflow.add_argument("--output")
        workflow.set_defaults(func=handler)

    kernels = backends.add_parser("kernels", help="Profile attention/GEMM/collective kernels directly.")
    kernels_workflows = kernels.add_subparsers(dest="workflow", required=True)

    run_kernels = kernels_workflows.add_parser("run")
    _add_config_argument(run_kernels)
    run_kernels.add_argument(
        "--family",
        choices=["all", *_kernel_families()],
        default="all",
        help="Kernel family to measure. Defaults to every registered family.",
    )
    run_kernels.set_defaults(func=cmd_kernels_run)

    fit_floor_kernels = kernels_workflows.add_parser("fit-floor")
    _add_config_argument(fit_floor_kernels)
    fit_floor_kernels.add_argument(
        "--candidates-us",
        type=float,
        nargs="+",
        metavar="US",
        default=None,
        help=(
            "Floor candidates in microseconds. Defaults to "
            "[0, 2, 4, 6, 8, 12, 16, 22, 32, 45, 60, 80]."
        ),
    )
    fit_floor_kernels.set_defaults(func=cmd_kernels_fit_floor)

    holdout_kernels = kernels_workflows.add_parser("holdout")
    _add_config_argument(holdout_kernels)
    holdout_kernels.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of bracketed points to score. Below 1.0 samples a subset and can "
        "flatter the surface: GEMM once measured 3.38%% over 13 sampled points against "
        "8.48%% over all 64. Lower it only for a quick smoke check.",
    )
    holdout_kernels.add_argument("--seed", type=int, default=0)
    holdout_kernels.add_argument("--gate", type=float, default=5.0)
    holdout_kernels.add_argument("--output", help="Holdout report JSON path.")
    holdout_kernels.set_defaults(func=cmd_kernels_holdout)

    verify_shapes_kernels = kernels_workflows.add_parser("verify-shapes")
    # --config is optional here, unlike every other kernels subcommand: this check needs only a
    # checkpoint, so it must be runnable on a model that has no profiling config yet.
    verify_shapes_kernels.add_argument("--config", help="YAML workflow configuration.")
    verify_shapes_kernels.add_argument(
        "--model-path",
        help="Checkpoint directory or hub id. Defaults to the config's model.",
    )
    verify_shapes_kernels.set_defaults(func=cmd_kernels_verify_shapes)

    validate_kernels = kernels_workflows.add_parser("validate")
    _add_config_argument(validate_kernels)
    validate_kernels.add_argument(
        "--baseline", required=True, help="vLLM profiling JSONL to score against."
    )
    validate_kernels.add_argument("--model-path", help="Checkpoint path. Defaults to the config's.")
    validate_kernels.add_argument("--surface", default="pchip", choices=["pchip", "shepard"])
    validate_kernels.add_argument("--min-samples", type=int, default=20)
    validate_kernels.add_argument("--gate", type=float, default=6.0)
    validate_kernels.add_argument("--output", help="Validation report JSON path.")
    validate_kernels.set_defaults(func=cmd_kernels_validate)

    megatron = backends.add_parser("megatron", help="Use Megatron-Bridge recipes.")
    megatron_workflows = megatron.add_subparsers(dest="workflow", required=True)
    generate = megatron_workflows.add_parser("generate")
    _add_config_argument(generate)
    generate.add_argument("--candidates")
    generate.set_defaults(func=cmd_megatron_generate)

    run = megatron_workflows.add_parser("run")
    _add_config_argument(run)
    _add_candidate_arguments(run)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume", action="store_true")
    # Multi-node. A mesh wider than one host's GPU count needs these: every host runs the
    # same command with its own --node-rank, against one shared --master-addr. Required for
    # Qwen3.5-35B-A3B, whose smallest trainable mesh is 16 ranks.
    run.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Nodes the candidate mesh spans (default 1, single-host torchrun --standalone).",
    )
    run.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="This host's index in [0, nnodes).",
    )
    run.add_argument("--master-addr", help="Rendezvous host, required when --nnodes > 1.")
    run.add_argument(
        "--master-port", type=int, help="Rendezvous port, required when --nnodes > 1."
    )
    run.set_defaults(func=cmd_megatron_run)

    collect = megatron_workflows.add_parser("collect")
    _add_config_argument(collect)
    _add_candidate_arguments(collect)
    collect.add_argument("--output")
    collect.set_defaults(func=cmd_megatron_collect)

    fsdp = backends.add_parser("fsdp", help="Use the SimRL FSDP2 launcher.")
    fsdp_workflows = fsdp.add_subparsers(dest="workflow", required=True)
    fsdp_generate = fsdp_workflows.add_parser("generate")
    _add_config_argument(fsdp_generate)
    fsdp_generate.add_argument("--candidates")
    fsdp_generate.set_defaults(func=cmd_fsdp_generate)

    fsdp_run = fsdp_workflows.add_parser("run")
    _add_config_argument(fsdp_run)
    _add_candidate_arguments(fsdp_run)
    fsdp_run.add_argument("--dry-run", action="store_true")
    fsdp_run.add_argument("--resume", action="store_true")
    fsdp_run.set_defaults(func=cmd_fsdp_run)

    fsdp_collect = fsdp_workflows.add_parser("collect")
    _add_config_argument(fsdp_collect)
    _add_candidate_arguments(fsdp_collect)
    fsdp_collect.add_argument("--output")
    fsdp_collect.set_defaults(func=cmd_fsdp_collect)
    return parser


def main() -> None:
    """Run a profiling workflow."""
    parser = build_parser()
    args = parser.parse_args()
    setup_logging()
    args.func(args)


if __name__ == "__main__":
    main()
