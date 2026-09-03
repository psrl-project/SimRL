"""Load profiling model identity from a HuggingFace checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import ModelDescriptor, ProfileValidationError

_DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float8": 1,
    "fp8": 1,
}

# Keys that identify a block shape the dense kernel-shape family does not describe. They are
# forwarded verbatim so `families_dense.py` can refuse the model instead of pricing a
# mixture-of-experts or compressed-KV block as though it were dense. Nothing reads them for
# arithmetic yet; they exist so the refusal is reachable from a real checkpoint.
_ARCHITECTURE_MARKER_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "kv_lora_rank",
    "q_lora_rank",
)


def model_descriptor_from_yaml(value: Any, *, base_dir: Path) -> ModelDescriptor:
    """
    Resolve a YAML `model` entry to a `ModelDescriptor`.

    The YAML value must be a HuggingFace directory, hub id, or a mapping with
    only `path` or `name`. Architecture fields are read from the checkpoint.
    """
    source = _yaml_model_source(value)
    path = Path(source).expanduser()
    if not path.is_absolute():
        resolved = (base_dir / path).resolve()
        if resolved.exists():
            return load_hf_model_descriptor(resolved)
    return load_hf_model_descriptor(source)


def load_hf_model_descriptor(source: str | Path) -> ModelDescriptor:
    """Build a `ModelDescriptor` from HuggingFace `AutoConfig` or `config.json`."""
    path = Path(str(source)).expanduser()
    if path.exists():
        path = path.resolve()
        source_name = str(path)
    else:
        source_name = str(source)
    config = _read_hf_config(path if path.exists() else source)
    architecture = _first(
        (config.get("architectures") or [None])[0],
        config.get("model_type"),
    )
    hidden_size = _required_int(config, "hidden_size", "n_embd", "dim")
    num_layers = _required_int(config, "num_hidden_layers", "num_layers", "n_layer")
    num_attention_heads = _required_int(config, "num_attention_heads", "n_head")
    num_query_groups = _optional_int(
        config,
        "num_key_value_heads",
        "num_query_groups",
    ) or num_attention_heads
    intermediate_size = _optional_int(config, "intermediate_size", "ffn_dim", "n_inner")
    vocab_size = _optional_int(config, "vocab_size")
    extra: dict[str, Any] = {}
    if intermediate_size is not None:
        extra["intermediate_size"] = intermediate_size
    if vocab_size is not None:
        extra["vocab_size"] = vocab_size
    if config.get("model_type"):
        extra["model_type"] = config["model_type"]
    if config.get("head_dim") is not None:
        extra["head_dim"] = int(config["head_dim"])
    for marker in _ARCHITECTURE_MARKER_KEYS:
        value = config.get(marker)
        if value is not None:
            extra[marker] = int(value)
    return ModelDescriptor(
        name=source_name,
        architecture=None if architecture is None else str(architecture),
        parameter_count=_parameter_count(path if path.exists() else None, config),
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        extra=extra,
    )


def _yaml_model_source(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        extra = set(value) - {"path", "name", "revision"}
        if extra:
            raise ProfileValidationError(
                "model YAML only accepts a HuggingFace path or hub id. "
                f"Architecture fields {sorted(extra)} are read from the checkpoint."
            )
        source = value.get("path") or value.get("name")
        if isinstance(source, str) and source:
            return source
    raise ProfileValidationError("model must be a HuggingFace path or hub id.")


def _read_hf_config(source: str | Path) -> dict[str, Any]:
    path = Path(source)
    config_file = path / "config.json" if path.is_dir() else path
    if config_file.is_file() and config_file.name == "config.json":
        loaded = json.loads(config_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ProfileValidationError(f"Expected a JSON object in {config_file}.")
        return loaded
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ProfileValidationError(
            f"Cannot read HuggingFace config from {str(source)!r}. "
            "Install transformers or pass a local directory with config.json."
        ) from exc
    return AutoConfig.from_pretrained(str(source)).to_dict()


def _parameter_count(model_dir: Path | None, config: dict[str, Any]) -> int | None:
    if model_dir is not None:
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            total_size = (index.get("metadata") or {}).get("total_size")
            if total_size:
                dtype = str(config.get("torch_dtype", "bfloat16")).lower()
                width = _DTYPE_BYTES.get(dtype, 2)
                return int(total_size) // width
    return _estimate_parameter_count(config)


def _estimate_parameter_count(config: dict[str, Any]) -> int | None:
    hidden = _optional_int(config, "hidden_size", "n_embd", "dim")
    layers = _optional_int(config, "num_hidden_layers", "num_layers", "n_layer")
    heads = _optional_int(config, "num_attention_heads", "n_head")
    if hidden is None or layers is None or heads is None:
        return None
    kv_heads = _optional_int(config, "num_key_value_heads", "num_query_groups") or heads
    head_dim = _optional_int(config, "head_dim") or hidden // heads
    intermediate = _optional_int(config, "intermediate_size", "ffn_dim", "n_inner")
    if intermediate is None:
        intermediate = 4 * hidden
    vocab = _optional_int(config, "vocab_size") or 0
    query = hidden * heads * head_dim
    key_value = hidden * kv_heads * head_dim * 2
    output = hidden * heads * head_dim
    mlp = 3 * hidden * intermediate
    norms = 2 * hidden
    qk_norm = 2 * head_dim if str(config.get("model_type", "")).startswith("qwen3") else 0
    per_layer = query + key_value + output + mlp + norms + qk_norm
    embeddings = vocab * hidden
    lm_head = 0 if config.get("tie_word_embeddings") else embeddings
    return int(embeddings + lm_head + layers * per_layer)


def _required_int(config: dict[str, Any], *names: str) -> int:
    value = _optional_int(config, *names)
    if value is None:
        raise ProfileValidationError(
            f"HuggingFace config is missing {' / '.join(names)}."
        )
    return value


def _optional_int(config: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = config.get(name)
        if value is not None:
            return int(value)
    return None


def _first(*values: Any) -> Any:
    for value in values:
        if value:
            return value
    return None
