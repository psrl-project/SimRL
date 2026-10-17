"""Normalize vLLM timing sweep shards into `ProfileRecord` observations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from ...._logging import srl_logger
from ...schema import (
    Parallelism,
    ProfileRecord,
    ProfileValidationError,
    RunDescriptor,
    SoftwareDescriptor,
)
from .config import VLLMSweepConfig
from .parse import parse_sweep_log, payload_from_iteration, trusted_iterations

_INPUT_LEN_PATTERN = re.compile(r"random-input-len[=-](?P<length>\d+)")


def _value_from_keys(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return default


def _parallelism_from_result(value: dict[str, Any], config: VLLMSweepConfig) -> Parallelism:
    tp = int(
        _value_from_keys(
            value,
            "tensor-parallel-size",
            "tensor_parallel_size",
            default=config.serve_args.get("tensor_parallel_size", config.hardware.gpu_count),
        )
    )
    pp = int(
        _value_from_keys(
            value,
            "pipeline-parallel-size",
            "pipeline_parallel_size",
            default=config.serve_args.get("pipeline_parallel_size", 1),
        )
    )
    dp = int(
        _value_from_keys(
            value,
            "data-parallel-size",
            "data_parallel_size",
            default=config.serve_args.get("data_parallel_size", 1),
        )
    )
    return Parallelism(tp=tp, pp=pp, dp=dp, cp=1)


def _manifest_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _experiment_dirs(config: VLLMSweepConfig, phase: str) -> dict[str, Path]:
    legacy = config.raw_experiment_dir(phase, "timing")
    shard_pattern = re.compile(rf"{re.escape(legacy.name)}-(?P<shard>s\d+)")
    shards = {
        match.group("shard"): path
        for path in sorted(legacy.parent.glob(f"{legacy.name}-s*"))
        if path.is_dir()
        and (match := shard_pattern.fullmatch(path.name)) is not None
    }
    if shards:
        return shards
    return {"": legacy} if legacy.is_dir() else {}


def _input_len(combo_path: str, value: dict[str, Any]) -> int:
    match = _INPUT_LEN_PATTERN.search(combo_path)
    if match is not None:
        return int(match.group("length"))
    lengths = value.get("input_lens") or value.get("input_lengths")
    if lengths:
        unique = {int(item) for item in lengths}
        if len(unique) == 1:
            return next(iter(unique))
        raise ProfileValidationError(
            "Equal-length feature formulas require a single input length, "
            f"got {sorted(unique)} in {combo_path}."
        )
    raise ProfileValidationError(
        f"Cannot recover random-input-len for vLLM run {combo_path}."
    )


def _prompt_token_budget(combo_path: str, value: dict[str, Any]) -> int:
    """Total prompt tokens a run may schedule before context means recompute."""
    total = value.get("total_input_tokens")
    if total is None:
        lengths = value.get("input_lens") or value.get("input_lengths")
        if not lengths:
            raise ProfileValidationError(
                f"Cannot recover the prompt token total for vLLM run {combo_path}."
            )
        total = sum(int(item) for item in lengths)
    return int(total)


def collect_vllm_records(
    config: VLLMSweepConfig,
    phase: str,
) -> list[ProfileRecord]:
    """Normalize scheduler iteration logs into one record per step."""
    experiment_dirs = _experiment_dirs(config, phase)
    if not experiment_dirs:
        raise FileNotFoundError(
            f"vLLM timing experiment directories do not exist for phase {phase!r}."
        )
    records: list[ProfileRecord] = []
    for shard_id, timing_root in experiment_dirs.items():
        result_paths = sorted(
            timing_root.rglob("run=*.json"),
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
        )
        if not result_paths:
            raise ProfileValidationError(
                f"No vLLM timing run JSON files were found under {timing_root}."
            )
        iterations = parse_sweep_log(
            config.sweep_log_path(phase, "timing", shard_id or None)
        )
        for result_path in result_paths:
            details = iterations.get(result_path.resolve(), [])
            if not details:
                raise ProfileValidationError(
                    "Missing scheduler iteration timings for vLLM run: "
                    f"{result_path}. vLLM only emits Iteration(...) logs at INFO; "
                    "VLLM_LOGGING_LEVEL=WARN disables LoggingStatLogger."
                )
            with result_path.open(encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ProfileValidationError(f"Expected a JSON object in {result_path}.")
            combo_path = str(result_path.relative_to(timing_root))
            input_len = _input_len(combo_path, value)
            details, trust = trusted_iterations(
                details,
                input_len=input_len,
                prompt_token_budget=_prompt_token_budget(combo_path, value),
            )
            if trust.truncation_reason is not None:
                srl_logger.warning(
                    "Truncated vLLM run %s at iteration %s (%s): kept %d steps and "
                    "dropped %d after the KV cache filled and vLLM preempted requests. "
                    "Lower max-concurrency or random-output-len to profile this point "
                    "without preemption.",
                    combo_path,
                    trust.truncation_iteration,
                    trust.truncation_reason,
                    trust.kept,
                    trust.dropped_after_preemption,
                )
            if trust.dropped_unmeasured:
                srl_logger.warning(
                    "Dropped %d step(s) of vLLM run %s whose elapsed time is an "
                    "async-scheduling artifact rather than a forward pass.",
                    trust.dropped_unmeasured,
                    combo_path,
                )
            if not details:
                srl_logger.warning(
                    "Skipped vLLM run %s: no scheduler step carries a usable "
                    "measurement. Re-run the sweep so the server disables async "
                    "scheduling.",
                    combo_path,
                )
                continue
            parallelism = _parallelism_from_result(value, config)
            hardware = replace(config.hardware, gpu_count=int(parallelism.world_size))
            relative_path = _manifest_path(result_path, config.output_dir)
            identity = hashlib.sha256(
                f"{phase}:{combo_path}".encode("utf-8")
            ).hexdigest()[:20]
            run_number = int(
                value.get(
                    "run_number",
                    result_path.stem.removeprefix("run="),
                )
            )
            run = RunDescriptor(
                run_id=f"vllm-{identity}",
                repeat=run_number,
                status="success" if int(value.get("completed", 0)) > 0 else "failed",
                duration_s=(
                    float(value["duration"]) if value.get("duration") is not None else None
                ),
                raw_artifact=relative_path,
                error=None if int(value.get("completed", 0)) > 0 else "No requests completed.",
            )
            software = SoftwareDescriptor(
                backend="vllm",
                backend_version=config.backend_version,
                source_commit=config.source_commit,
                command=(config.executable, "bench", "sweep", "serve"),
            )
            # Age every request by the engine's own iteration index rather than
            # by a running counter, so dropping an untrusted step in the middle
            # of a run does not shift the ages of the steps after it.
            first_generation_index = next(
                (item.index for item in details if item.gen_requests > 0),
                None,
            )
            for detail in details:
                payload = payload_from_iteration(
                    detail,
                    input_len=input_len,
                    decode_ordinal=(
                        max(detail.index - first_generation_index, 0)
                        if first_generation_index is not None
                        else 0
                    ),
                    batch_step_index=detail.index,
                )
                if payload is None:
                    continue
                if phase == "prefill" and payload.phase == "decode":
                    continue
                if phase == "decode" and payload.phase == "prefill":
                    continue
                records.append(
                    ProfileRecord(
                        record_type="rollout",
                        run=run,
                        model=config.model,
                        hardware=hardware,
                        software=software,
                        parallelism=parallelism,
                        payload=payload,
                        tags={
                            "phase": phase,
                            "observed_phase": payload.phase,
                            "source": "vllm-scheduler-iteration-log",
                            "latency_source": "scheduler-iteration-elapsed-ms",
                            "feature_source": "equal-length-formula",
                            "timing_benchmark_artifact": relative_path,
                            "scheduler_iteration": detail.to_dict(),
                        },
                    )
                )
    return records
