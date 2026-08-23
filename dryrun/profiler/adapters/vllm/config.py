"""Configuration for the official ``vllm bench sweep serve`` workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

from ...hf import model_descriptor_from_yaml
from ...schema import HardwareDescriptor, ModelDescriptor, ProfileValidationError

PINNED_VLLM_VERSION = "0.26.0"
PINNED_VLLM_COMMIT = "568afb3a13806beb53bb2e6bd518269357b237c0"

# Grace margin added to `server_ready_timeout` when a job waits for a startup slot, so a
# server that is merely slow does not make the pool give up on the queue behind it.
DEFAULT_STARTUP_STAGGER_SECONDS = 30.0

# Concurrent vLLM startups contend for the shared filesystem holding torch, vLLM and the
# checkpoint. Eight simultaneous imports on CephFS took over 40 minutes and blew past
# `server_ready_timeout`; admitting one startup at a time keeps each server inside it.
# Raise this when the checkpoint and environment live on a fast local disk.
DEFAULT_MAX_CONCURRENT_STARTUPS = 1


def _expand_grid(value: dict[str, Any]) -> list[dict[str, Any]]:
    if "parameters" in value:
        parameters = value["parameters"]
        if not isinstance(parameters, list) or not all(isinstance(item, dict) for item in parameters):
            raise ProfileValidationError("A sweep parameters value must be a list of mappings.")
        return [dict(item) for item in parameters]
    grid = value.get("grid", {})
    if not isinstance(grid, dict):
        raise ProfileValidationError("A sweep grid must be a mapping.")
    if not grid:
        return [{}]
    keys = list(grid)
    choices = [items if isinstance(items, list) else [items] for items in grid.values()]
    return [dict(zip(keys, combination, strict=True)) for combination in product(*choices)]


def _normalized_parameters(parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {str(key).replace("_", "-"): value for key, value in parameter.items()}
        for parameter in parameters
    ]


def _positive_int(value: Any, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ProfileValidationError(
            f"{field} must be a positive integer, got {value!r}."
        ) from exc
    if parsed <= 0:
        raise ProfileValidationError(f"{field} must be positive.")
    return parsed


def _max_parallel_value(value: Any) -> int | None:
    if value is None or str(value).lower() == "auto":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ProfileValidationError(
            f"run.max_parallel must be 'auto' or a positive integer, got {value!r}."
        ) from exc
    if parsed <= 0:
        raise ProfileValidationError("run.max_parallel must be positive.")
    return parsed


@dataclass(frozen=True)
class VLLMSweepConfig:
    """Configuration required to generate, run, and collect a vLLM sweep."""

    model: ModelDescriptor
    hardware: HardwareDescriptor
    output_dir: Path
    experiment_name: str
    serve_args: dict[str, Any] = field(default_factory=dict)
    serve_parameters: tuple[dict[str, Any], ...] = ({},)
    bench_args: dict[str, Any] = field(default_factory=dict)
    phase_parameters: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    num_runs: int = 3
    server_ready_timeout: int = 900
    executable: str = "vllm"
    backend_version: str = PINNED_VLLM_VERSION
    source_commit: str = PINNED_VLLM_COMMIT
    environment: dict[str, str] = field(default_factory=dict)
    max_parallel: int | None = None
    startup_stagger_seconds: float = DEFAULT_STARTUP_STAGGER_SECONDS
    max_concurrent_startups: int = DEFAULT_MAX_CONCURRENT_STARTUPS

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ProfileValidationError("experiment_name must not be empty.")
        if self.num_runs <= 0:
            raise ProfileValidationError("num_runs must be positive.")
        if self.server_ready_timeout <= 0:
            raise ProfileValidationError("server_ready_timeout must be positive.")
        if self.max_parallel is not None and self.max_parallel <= 0:
            raise ProfileValidationError("max_parallel must be positive.")
        if self.startup_stagger_seconds < 0:
            raise ProfileValidationError("startup_stagger_seconds must not be negative.")
        if self.max_concurrent_startups <= 0:
            raise ProfileValidationError("max_concurrent_startups must be positive.")
        range_ratio = self.bench_args.get(
            "random-range-ratio",
            self.bench_args.get("random_range_ratio", 0.0),
        )
        if float(range_ratio) != 0.0:
            raise ProfileValidationError(
                "Equal-length feature formulas require bench random-range-ratio=0."
            )
        if any(
            key.replace("_", "-") in {"profiler", "profiler-config"}
            for key in self.serve_args
        ):
            raise ProfileValidationError(
                "serve.args must not enable a profiler; timing-only collection "
                "uses scheduler iteration logs."
            )
        if any(
            key.replace("_", "-") in {"async-scheduling", "no-async-scheduling"}
            for key in self.serve_args
        ):
            raise ProfileValidationError(
                "serve.args must not set async scheduling; timing collection "
                "always disables it so each iteration's elapsed time measures "
                "its own forward pass."
            )
        if not self.phase_parameters:
            raise ProfileValidationError("At least one vLLM phase must be configured.")
        for phase, parameters in self.phase_parameters.items():
            if phase not in {"prefill", "decode"}:
                raise ProfileValidationError(f"Unsupported vLLM phase {phase!r}.")
            for item in parameters:
                output_length = item.get("random-output-len", item.get("output-len"))
                if phase == "prefill" and output_length is not None and int(output_length) != 1:
                    raise ProfileValidationError("Prefill sweep points must request exactly one output token.")
                if phase == "decode" and output_length is not None and int(output_length) <= 1:
                    raise ProfileValidationError("Decode sweep points must request more than one output token.")

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, base_dir: Path | None = None) -> VLLMSweepConfig:
        root = base_dir or Path.cwd()
        phases = {
            phase: tuple(_normalized_parameters(_expand_grid(section)))
            for phase, section in value.get("phases", {}).items()
        }
        output_dir = Path(value.get("output_dir", "profile_output"))
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        output_dir = output_dir.resolve()
        serve = value.get("serve", {})
        bench = value.get("bench", {})
        run = value.get("run", {})
        if not isinstance(run, dict):
            raise ProfileValidationError("run must be a mapping.")
        return cls(
            model=model_descriptor_from_yaml(value["model"], base_dir=root),
            hardware=HardwareDescriptor.from_dict(value["hardware"]),
            output_dir=output_dir,
            experiment_name=str(value.get("experiment_name", "vllm")),
            serve_args=dict(serve.get("args", {})),
            serve_parameters=tuple(_normalized_parameters(_expand_grid(serve))),
            bench_args=dict(bench.get("args", {})),
            phase_parameters=phases,
            num_runs=int(value.get("num_runs", 3)),
            server_ready_timeout=int(value.get("server_ready_timeout", 900)),
            executable=str(value.get("executable", "vllm")),
            environment={str(key): str(item) for key, item in value.get("environment", {}).items()},
            max_parallel=_max_parallel_value(run.get("max_parallel")),
            startup_stagger_seconds=float(
                run.get("startup_stagger_seconds", DEFAULT_STARTUP_STAGGER_SECONDS)
            ),
            max_concurrent_startups=_positive_int(
                run.get("max_concurrent_startups", DEFAULT_MAX_CONCURRENT_STARTUPS),
                field="run.max_concurrent_startups",
            ),
        )

    def raw_experiment_dir(
        self,
        phase: str,
        measurement: str,
        shard_id: str | None = None,
    ) -> Path:
        suffix = f"-{shard_id}" if shard_id else ""
        return (
            self.output_dir
            / "raw"
            / f"vllm_{measurement}"
            / f"{self.experiment_name}-{phase}-{measurement}{suffix}"
        )

    def sweep_log_path(
        self,
        phase: str,
        measurement: str,
        shard_id: str | None = None,
    ) -> Path:
        suffix = f"_{shard_id}" if shard_id else ""
        return (
            self.output_dir
            / "control"
            / "vllm"
            / phase
            / f"{measurement}{suffix}_sweep.log"
        )
