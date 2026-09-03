"""GPU specifications backing speed-of-light latency and utility normalization.

Utility is `SOL(shape, spec) / measured_latency`, so every number here scales every
prediction made against that GPU. Values are vendor-published dense peaks; a kernel
reaching 60% of them is normal and a utility above ~1.2 means the SOL formula is
missing work rather than that the hardware exceeded spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_SPEC_PATH = Path(__file__).parent / "data" / "gpu_specs.yaml"

_DTYPE_ALIASES: dict[str, str] = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "tf32": "float32",
    "fp8": "fp8",
    "float8": "fp8",
    "int8": "int8",
}

_DTYPE_BYTES: dict[str, int] = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "fp8": 1,
    "int8": 1,
}


def canonical_dtype(name: str) -> str:
    """
    Fold a dtype alias onto the spelling used in perf tables and spec keys.

    Args:
        name (str): A dtype name such as `bf16` or `bfloat16`.

    Returns:
        str: The canonical spelling, e.g. `bfloat16`.

    Raises:
        ValueError: If the dtype is not recognized.
    """
    try:
        return _DTYPE_ALIASES[name.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(set(_DTYPE_ALIASES)))
        raise ValueError(f"Unsupported dtype {name!r}. Supported: {supported}.") from exc


def dtype_bytes(name: str) -> int:
    """
    Return the element size in bytes of a dtype.

    Args:
        name (str): A dtype name or alias.

    Returns:
        int: Bytes per element.

    Raises:
        ValueError: If the dtype is not recognized.
    """
    return _DTYPE_BYTES[canonical_dtype(name)]


def gpu_slug(name: str) -> str:
    """
    Normalize a device name into the directory slug used by perf tables.

    Args:
        name (str): A `torch.cuda.get_device_name()` string or an existing slug.

    Returns:
        str: Lowercase slug with the `NVIDIA` prefix dropped and separators
            collapsed to underscores, e.g. `NVIDIA H200 SXM` becomes `h200_sxm`.
    """
    cleaned = name.strip().lower()
    for prefix in ("nvidia ", "nvidia-"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    parts = [part for part in cleaned.replace("-", " ").replace("_", " ").split() if part]
    return "_".join(parts)


@dataclass(frozen=True)
class GpuSpec:
    """Peak throughput and capacity of one GPU model."""

    name: str
    slug: str
    sm_count: int
    memory_bytes: int
    memory_bandwidth_bytes_per_s: float
    tc_flops_per_s: dict[str, float]
    link_bandwidth_bytes_per_s: float

    def flops_for(self, dtype: str) -> float:
        """
        Return dense tensor-core throughput for one dtype.

        Args:
            dtype (str): A dtype name or alias.

        Returns:
            float: Peak FLOP/s.

        Raises:
            KeyError: If this GPU has no published figure for the dtype.
        """
        try:
            key = canonical_dtype(dtype)
        except ValueError:
            recognized = ", ".join(sorted(set(_DTYPE_ALIASES)))
            raise KeyError(
                f"Dtype {dtype!r} is not recognized. Recognized dtype names: {recognized}."
            )
        try:
            return self.tc_flops_per_s[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.tc_flops_per_s))
            raise KeyError(
                f"GPU {self.slug!r} has no {key!r} throughput. Available: {available}."
            ) from exc

    @classmethod
    def from_dict(cls, slug: str, value: dict[str, Any]) -> GpuSpec:
        """
        Build a spec from one entry of the specs YAML.

        Args:
            slug (str): The entry key, used as the directory slug.
            value (dict[str, Any]): The entry body.

        Returns:
            GpuSpec: The parsed spec.
        """
        return cls(
            name=str(value["name"]),
            slug=slug,
            sm_count=int(value["sm_count"]),
            memory_bytes=int(value["memory_bytes"]),
            memory_bandwidth_bytes_per_s=float(value["memory_bandwidth_bytes_per_s"]),
            tc_flops_per_s={
                canonical_dtype(key): float(item)
                for key, item in value["tc_flops_per_s"].items()
            },
            link_bandwidth_bytes_per_s=float(value["link_bandwidth_bytes_per_s"]),
        )


@lru_cache(maxsize=None)
def _load_specs(path: Path) -> dict[str, GpuSpec]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {slug: GpuSpec.from_dict(slug, entry) for slug, entry in value.items()}


def load_gpu_spec(slug: str, *, path: str | Path | None = None) -> GpuSpec:
    """
    Load one GPU spec by slug.

    Args:
        slug (str): Directory slug or raw device name; normalized via `gpu_slug`.
        path (str | Path | None): Alternate specs YAML. Defaults to the bundled file.

    Returns:
        GpuSpec: The matching spec.

    Raises:
        KeyError: If no entry matches.
    """
    resolved = Path(path) if path is not None else _DEFAULT_SPEC_PATH
    specs = _load_specs(resolved)
    key = gpu_slug(slug)
    try:
        return specs[key]
    except KeyError as exc:
        available = ", ".join(sorted(specs))
        raise KeyError(f"No GPU spec for {key!r} in {resolved}. Available: {available}.") from exc
