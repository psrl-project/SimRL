"""KV cache capacity in blocks, derived from the model and the GPU budget.

`rollout.kv_blocks` is the denominator of every `kv_cache_usage` number the
simulator reports, so a wrong value rescales the whole KV curve without
changing anything else. The default of 100000 is a placeholder: a real vLLM
engine sizes its cache from whatever GPU memory is left after weights, and for
Qwen3-8B on an H20 at `gpu_memory_utilization=0.9` that is closer to 31000.
Comparing a simulated KV trace against a measured one is meaningless until the
two share a denominator.

This reproduces vLLM's own arithmetic rather than inventing one:

    page_size_bytes = 2 * block_size * num_kv_heads * head_dim * dtype_bytes
    num_gpu_blocks  = available_kv_memory // (page_size_bytes * num_layers)

(`vllm/v1/kv_cache_interface.py:AttentionSpec.real_page_size_bytes` and
`vllm/v1/core/kv_cache_utils.py`, where the same floor division appears.) The
leading 2 is K and V; `num_kv_heads` rather than `num_attention_heads` is what
makes a GQA model cheap to cache.

Nothing here imports torch or vllm. The inputs are a handful of integers read
from a HuggingFace `config.json`, so this runs on a machine with no GPU and no
inference stack installed.

**This is an estimate of a measurement.** vLLM subtracts its *observed* peak
non-KV memory, which includes the CUDA context, allocator fragmentation and the
activation high-water mark of a profile run. Those cannot be derived from a
config file. For Qwen3-8B on one H20 at utilisation 0.9 this function returns
32205 against a measured 31279, a 3.0% overestimate, all of it that unmodelled
overhead.

That 3% is the price of not needing a real run to have happened, which is the
whole point of a simulator: a configuration worth simulating is usually one
nobody has executed yet. Use `non_kv_overhead_bytes` to fold in a per-hardware
constant once one is known: on H20 at this utilisation it is about 2 GiB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..._logging import dryrun_logger

_BYTES_PER_MIB = 1024 * 1024

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
        try:
            num_layers = int(config["num_hidden_layers"])
            num_attention_heads = int(config["num_attention_heads"])
        except KeyError as exc:
            raise ValueError(f"Model config {path!s} is missing {exc.args[0]!r}.") from exc

        # A model without grouped-query attention omits the key, in which case
        # every attention head carries its own KV.
        num_kv_heads = int(config.get("num_key_value_heads", num_attention_heads))
        head_dim = int(config.get("head_dim") or (int(config["hidden_size"]) // num_attention_heads))
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


def kv_blocks_from_memory(
    geometry: KvCacheGeometry,
    gpu_memory_mib: int,
    gpu_memory_utilization: float,
    parameter_count: int,
    *,
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
        parameter_count: Total model parameters, before TP sharding.
        block_size: Tokens per KV block.
        tp_size: Tensor-parallel degree. Shards both weights and KV heads.
        parameter_dtype_bytes: Bytes per weight element.
        non_kv_overhead_bytes: Additional non-KV memory to subtract, such as the
            CUDA context and the activation high-water mark. Left at 0 the
            result overestimates by roughly 3%, because vLLM measures this
            rather than deriving it. About 2 GiB on an H20.

    Returns:
        Block count, floored at 0.

    Raises:
        ValueError: If the utilisation is outside (0, 1].
    """
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError(f"Expected gpu_memory_utilization in (0, 1], got {gpu_memory_utilization!r}.")

    budget = int(gpu_memory_mib * _BYTES_PER_MIB * gpu_memory_utilization)
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
