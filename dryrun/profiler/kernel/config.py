"""The top-level kernel profile config: what model, what hardware, where to write, how to time.

The sweep grids themselves live in `sweep/`, one module per op family, and are reached through
each op's own `parse_sweep`. That is why this file carries no per-family fields: adding an op
adds a file under `sweep/` and nothing here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...cost.kernel.hardware import gpu_slug
from ...cost.kernel.ops.base import families, op_for, ops_for_family
from ..hf import model_descriptor_from_yaml
from ..schema import HardwareDescriptor, ModelDescriptor, ProfileValidationError


@dataclass(frozen=True)
class MeasurementConfig:
    """Warmup/active iteration counts and dtype shared by every kernel profiler."""

    warmup_iterations: int = 5
    active_iterations: int = 20
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.warmup_iterations < 0:
            raise ProfileValidationError("measurement.warmup_iterations must not be negative.")
        if self.active_iterations <= 0:
            raise ProfileValidationError("measurement.active_iterations must be positive.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MeasurementConfig:
        """Parse the `measurement:` subtree, falling back to the shared defaults."""
        return cls(
            warmup_iterations=int(value.get("warmup_iterations", 5)),
            active_iterations=int(value.get("active_iterations", 20)),
            dtype=str(value.get("dtype", "bfloat16")),
            kv_cache_dtype=str(value.get("kv_cache_dtype", "bfloat16")),
        )


@dataclass(frozen=True)
class KernelProfileConfig:
    """Configuration required to profile kernels, with one sweep per op."""

    model: ModelDescriptor
    hardware: HardwareDescriptor
    output_dir: Path
    experiment_name: str
    tensor_parallel: int
    measurement: MeasurementConfig
    sweeps: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ProfileValidationError("experiment_name must not be empty.")
        if self.tensor_parallel <= 0:
            raise ProfileValidationError("parallelism.tensor_parallel must be positive.")

    def sweep(self, op_name: str) -> Any:
        """
        Return the parsed sweep grid for one op.

        Args:
            op_name (str): A registered op name.

        Returns:
            Any: The op's own sweep type, e.g. `GemmSweep` or `AttentionSweep`.

        Raises:
            ProfileValidationError: If the config carries no sweep for this op's family.
        """
        try:
            return self.sweeps[op_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.sweeps)) or "none"
            raise ProfileValidationError(
                f"No sweep configured for op {op_name!r}. Configured ops: {available}. "
                f"Add a `{op_for(op_name).family}:` block to the kernel profile config."
            ) from exc

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, base_dir: Path | None = None) -> KernelProfileConfig:
        root = base_dir or Path.cwd()
        output_dir = Path(value.get("output_dir", "profile_output"))
        if not output_dir.is_absolute():
            output_dir = (root / output_dir).resolve()
        parallelism = value.get("parallelism", {})
        # Each op parses its own family's subtree, so adding an op adds no field here. A family
        # absent from the config yields no entry, and `sweep()` raises naming the missing block
        # only if something actually asks for it: profiling one family must not require
        # configuring them all.
        sweeps: dict[str, Any] = {}
        for family in families():
            subtree = value.get(family)
            if subtree is None:
                continue
            for op in ops_for_family(family):
                sweeps[op.name] = op.parse_sweep(subtree)
        return cls(
            model=model_descriptor_from_yaml(value["model"], base_dir=root),
            hardware=HardwareDescriptor.from_dict(value["hardware"]),
            output_dir=output_dir,
            experiment_name=str(value.get("experiment_name", "kernel")),
            tensor_parallel=int(parallelism.get("tensor_parallel", 1)),
            measurement=MeasurementConfig.from_dict(value.get("measurement", {})),
            sweeps=sweeps,
        )

    def table_root(self) -> Path:
        """Return the root of the parquet perf-table tree."""
        return self.output_dir / "kernel"

    def gpu(self) -> str:
        """Return the perf-table directory slug for this hardware."""
        return gpu_slug(self.hardware.gpu_name)
