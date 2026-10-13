"""Estimate KV capacity arithmetically, as a fallback to measuring it.

`kv_probe.py` is the primary path: it asks a real vLLM engine. This module
reproduces vLLM's sizing arithmetic for the case where no GPU is available:

    page_size_bytes = 2 * block_size * num_kv_heads * head_dim * dtype_bytes
    num_gpu_blocks  = available_kv_memory // (page_size_bytes * num_layers)

It is only honest for a plain dense model. vLLM subtracts a peak activation,
non-torch allocation and CUDA graph reservation that it measures at startup, and
a hybrid attention/linear model pads its mamba pages to the attention page.
Neither is derivable from a config file, so `non_kv_overhead_bytes` is the only
lever and it has to come from somewhere else.

Imports no torch and no vllm, so it runs on a machine with no inference stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..._logging import dryrun_logger

_BYTES_PER_MIB = 1024 * 1024


def _flatten_text_config(config: dict) -> dict:
    """Merge nested ``text_config`` into the top-level dict for multimodal checkpoints.

    Mirrors the same helper in `dryrun.profiler.hf` so that models whose
    architecture fields live under ``text_config`` (e.g. Qwen3.5-MoE) resolve
    correctly without importing the profiler stack.
    """
    text_sub = config.get("text_config")
    if not isinstance(text_sub, dict):
        return config
    merged = dict(config)
    for key, value in text_sub.items():
        if key not in merged:
            merged[key] = value
    return merged


def _full_attention_layer_count(config: dict, total_layers: int) -> int:
    """Return the number of layers that carry a KV cache.

    Equals ``total_layers`` for a pure-attention model. A hybrid model declaring
    ``full_attention_interval`` holds a KV cache only in its full-attention
    layers; charging all of them overstates capacity by that interval.
    """
    interval = config.get("full_attention_interval")
    if interval is None:
        return total_layers
    interval = int(interval)
    if interval <= 0:
        return total_layers
    return total_layers // interval

# Bytes per element, keyed by the `torch_dtype` string a HuggingFace config uses.
_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "half": 2,
    "bf16": 2,
    "fp16": 2,
    "float32": 4,
    "fp32": 4,
    "float8_e4m3fn": 1,
    "fp8": 1,
}

def dtype_bytes(dtype: str) -> int:
    """Bytes per element for a HuggingFace `torch_dtype` string.

    Args:
        dtype: Dtype name as it appears in a model's `config.json`.

    Raises:
        ValueError: If the dtype is not one this module knows a width for.
    """
    try:
        return _DTYPE_BYTES[dtype.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dtype {dtype!r}. Known dtypes: {sorted(_DTYPE_BYTES)}."
        ) from exc


@dataclass(frozen=True)
class KvCacheGeometry:
    """The model fields that set the per-token cost of one KV cache entry.

    Args:
        num_layers: Attention layers that each hold a KV cache.
        num_kv_heads: Key/value heads *before* tensor-parallel sharding. For a
            GQA model this is `num_key_value_heads`, well below
            `num_attention_heads`, and it is the whole reason such models cache
            cheaply.
        head_dim: Width of one attention head.
        dtype_bytes: Bytes per stored element.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int

    @classmethod
    def from_hf_config(cls, model_path: str | Path) -> KvCacheGeometry:
        """Read the geometry from a HuggingFace checkpoint's `config.json`.

        Plain `json` rather than `transformers.AutoConfig`, so this stays
        importable without the inference stack. `head_dim` is explicit in recent
        configs and falls back to `hidden_size // num_attention_heads`, which is
        the same default `AutoConfig` applies.

        Args:
            model_path: Directory holding `config.json`, or the file itself.

        Raises:
            ValueError: If the config is missing or lacks a required field.
        """
        path = Path(model_path)
        if path.is_dir():
            path = path / "config.json"
        if not path.is_file():
            raise ValueError(f"No model config at {path!s}.")

        config = json.loads(path.read_text(encoding="utf-8"))
        config = _flatten_text_config(config)
        try:
            num_layers = int(config["num_hidden_layers"])
            num_attention_heads = int(config["num_attention_heads"])
        except KeyError as exc:
            raise ValueError(f"Model config {path!s} is missing {exc.args[0]!r}.") from exc

        # A model without grouped-query attention omits the key, in which case
        # every attention head carries its own KV.
        num_kv_heads = int(config.get("num_key_value_heads", num_attention_heads))
        head_dim = int(config.get("head_dim") or (int(config["hidden_size"]) // num_attention_heads))
        # Only full-attention layers carry a KV cache. The linear layers hold a
        # per-sequence mamba state instead.
        num_layers = _full_attention_layer_count(config, num_layers)
        return cls(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_bytes=dtype_bytes(str(config.get("torch_dtype", "bfloat16"))),
        )

    def page_size_bytes(self, block_size: int, tp_size: int = 1) -> int:
        """Bytes one KV block occupies in one layer.

        Mirrors vLLM's `AttentionSpec.real_page_size_bytes`. Tensor parallelism
        splits the KV heads across ranks, so a TP rank pays for its own shard
        only.

        Args:
            block_size: Tokens per KV block.
            tp_size: Tensor-parallel degree of the inference engine.

        Raises:
            ValueError: If `tp_size` does not divide `num_kv_heads`. vLLM
                replicates KV heads rather than splitting them in that case,
                which this function does not model.
        """
        if self.num_kv_heads % tp_size != 0:
            raise ValueError(
                f"Tensor-parallel degree {tp_size} does not divide {self.num_kv_heads} KV heads. "
                f"vLLM replicates KV heads in that case, which this estimate does not model."
            )
        heads_per_rank = self.num_kv_heads // tp_size
        return 2 * block_size * heads_per_rank * self.head_dim * self.dtype_bytes

    def bytes_per_block(self, block_size: int, tp_size: int = 1) -> int:
        """Bytes one KV block occupies across every layer."""
        return self.page_size_bytes(block_size, tp_size) * self.num_layers


def parameter_bytes_from_checkpoint(model_path: str | Path) -> int | None:
    """Return the checkpoint's on-disk weight size in bytes, or None if unknown.

    Read from the checkpoint rather than configured, because the weight footprint
    is the largest subtraction in the KV budget and a configured constant
    silently survives a model change. `metadata.total_size` is the materialized
    size in the checkpoint's own dtype, so it already covers MoE expert weights.
    None when there is no index file; the caller may still pass a count.

    Args:
        model_path: Directory holding the checkpoint, or its `config.json`.

    Returns:
        Total weight bytes, or None when no index file is present.
    """
    path = Path(model_path)
    if path.is_file():
        path = path.parent
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        return None
    total = json.loads(index.read_text(encoding="utf-8")).get("metadata", {}).get("total_size")
    return int(total) if total else None


def kv_blocks_from_memory(
    geometry: KvCacheGeometry,
    gpu_memory_mib: int,
    gpu_memory_utilization: float,
    parameter_count: int | None = None,
    *,
    weight_bytes_total: int | None = None,
    block_size: int = 16,
    tp_size: int = 1,
    parameter_dtype_bytes: int = 2,
    non_kv_overhead_bytes: int = 0,
) -> int:
    """Estimate how many KV blocks a vLLM engine would allocate.

    Follows vLLM's sizing: take the utilisation-scaled share of the device,
    subtract what is not KV cache, and floor-divide by the per-block cost.

    Args:
        geometry: Per-token KV cost of the model.
        gpu_memory_mib: Total device memory.
        gpu_memory_utilization: vLLM's fraction of the device it may use.
        parameter_count: Total parameters before TP sharding. Ignored when
            `weight_bytes_total` is given.
        weight_bytes_total: Materialized weight bytes before TP sharding.
            Preferred: `parameter_count` assumes a uniform dtype and is easy to
            leave pointing at another model.
        block_size: Tokens per KV block.
        tp_size: Tensor-parallel degree. Shards both weights and KV heads.
        parameter_dtype_bytes: Bytes per weight element.
        non_kv_overhead_bytes: Non-KV memory to subtract, covering the CUDA
            context, activation high-water mark and graph pool. vLLM measures
            this rather than deriving it, so at 0 the result overestimates.

    Returns:
        Block count, floored at 0.

    Raises:
        ValueError: If the utilisation is outside (0, 1], or if neither
            `parameter_count` nor `weight_bytes_total` is given.
    """
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError(f"Expected gpu_memory_utilization in (0, 1], got {gpu_memory_utilization!r}.")
    if weight_bytes_total is None and parameter_count is None:
        raise ValueError(
            "kv_blocks_from_memory needs either weight_bytes_total or parameter_count: "
            "the weight footprint is the largest subtraction in the KV budget and "
            "cannot be defaulted."
        )

    budget = int(gpu_memory_mib * _BYTES_PER_MIB * gpu_memory_utilization)
    if weight_bytes_total is not None:
        weight_bytes = int(weight_bytes_total) // tp_size
    else:
        weight_bytes = parameter_count * parameter_dtype_bytes // tp_size
    available = budget - weight_bytes - non_kv_overhead_bytes
    if available <= 0:
        dryrun_logger.warning(
            "Weights (%.2f GiB) plus overhead already exceed the %.2f GiB budget, so no KV cache fits.",
            weight_bytes / 1024**3,
            budget / 1024**3,
        )
        return 0

    blocks = available // geometry.bytes_per_block(block_size, tp_size)
    dryrun_logger.info(
        "Estimated %d KV blocks: %.2f GiB budget - %.2f GiB weights - %.2f GiB overhead, "
        "at %d B per block.",
        blocks,
        budget / 1024**3,
        weight_bytes / 1024**3,
        non_kv_overhead_bytes / 1024**3,
        geometry.bytes_per_block(block_size, tp_size),
    )
    return max(int(blocks), 0)
