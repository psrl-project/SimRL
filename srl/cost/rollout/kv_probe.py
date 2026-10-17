"""Measure KV capacity by asking vLLM, and cache the answer on disk.

vLLM sizes its KV cache from memory left after weights, its own measured peak
activation, non-torch allocations and CUDA graphs. The last three are observed
during a profile run and cannot be derived from `config.json`; a hybrid
attention/linear model additionally pads its mamba pages to the attention page,
which is not reconstructible either. So the capacity is read from a real engine
rather than estimated.

A probe loads the weights and runs one profile pass, so it costs minutes and the
GPUs the deployment would use. Results are cached on everything that changes the
answer: checkpoint identity, TP degree, memory budget, dtype, vLLM version and
GPU model. A key miss re-probes rather than interpolating, because these
quantities do not interpolate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from ..._logging import srl_logger

DEFAULT_CACHE_PATH = Path.home() / ".cache" / "srl" / "kv_capacity.json"

# Entries at another version are ignored rather than migrated: a capacity is
# cheap to re-measure and expensive to get silently wrong.
_CACHE_SCHEMA = 2


@dataclass(frozen=True)
class KvProbeKey:
    """Everything that changes the measured block count.

    Keyed on the checkpoint's fingerprint rather than its path: two paths may
    hold the same weights, and one path may be updated in place.
    """

    checkpoint_fingerprint: str
    tensor_parallel: int
    gpu_memory_utilization: float
    max_model_len: int
    dtype: str
    gpu_name: str
    vllm_version: str
    block_size: int
    max_num_batched_tokens: int
    """Bounds the activation high-water mark, so it moves the non-KV overhead."""
    max_num_seqs: int
    """Bounds the activation peak, and on a hybrid model the mamba state cache."""

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:24]


@dataclass(frozen=True)
class KvCapacity:
    """What a real engine reported, plus enough context to audit it."""

    num_gpu_blocks: int
    block_size: int
    total_tokens: int
    source: str
    """`"probe"` when measured here, `"cache"` when read back."""

    @property
    def tokens_per_block(self) -> int:
        return self.block_size


def checkpoint_fingerprint(model_path: str | Path) -> str:
    """Identify a checkpoint by its config and weight size, not its path.

    Moves when the weights or architecture change, which should invalidate a
    cached capacity; stable across a copy to a new path, which should not.
    """
    path = Path(model_path)
    if path.is_file():
        path = path.parent
    hasher = hashlib.sha256()
    config = path / "config.json"
    if config.is_file():
        hasher.update(config.read_bytes())
    index = path / "model.safetensors.index.json"
    if index.is_file():
        meta = json.loads(index.read_text(encoding="utf-8")).get("metadata", {})
        hasher.update(str(meta.get("total_size", "")).encode())
    else:
        # Single-shard checkpoint: fall back to the weight file's size.
        for shard in sorted(path.glob("*.safetensors")):
            hasher.update(str(shard.stat().st_size).encode())
    return hasher.hexdigest()[:16]


def _load_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.is_file():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        srl_logger.warning("KV capacity cache at %s is unreadable; ignoring it.", cache_path)
        return {}
    if raw.get("schema") != _CACHE_SCHEMA:
        return {}
    return raw.get("entries", {})


def _store_cache(cache_path: Path, entries: dict[str, dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so racing probes cannot leave a half-written cache.
    tmp = cache_path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps({"schema": _CACHE_SCHEMA, "entries": entries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(cache_path)


# Run in a subprocess: `vllm.LLM` initialises CUDA and NCCL without reliably
# releasing them, which would poison a run that probes several TP degrees.
_PROBE_SOURCE = """
import json, sys
model, tp, util, maxlen, dtype, block, batched, seqs = sys.argv[1:9]
from vllm import LLM
kwargs = dict(
    model=model,
    tensor_parallel_size=int(tp),
    gpu_memory_utilization=float(util),
    max_model_len=int(maxlen),
    dtype=dtype,
    trust_remote_code=True,
    # Random weights: the profile depends on tensor shapes, not values, and this
    # skips reading the checkpoint. Verified equal to a real load. CUDA graphs are
    # left enabled because a real server captures them before sizing the cache.
    load_format="dummy",
)
if int(block) > 0:
    kwargs["block_size"] = int(block)
if int(batched) > 0:
    kwargs["max_num_batched_tokens"] = int(batched)
if int(seqs) > 0:
    kwargs["max_num_seqs"] = int(seqs)
llm = LLM(**kwargs)
cache = llm.llm_engine.vllm_config.cache_config
print("SIMRL_KV_RESULT " + json.dumps({
    "num_gpu_blocks": int(cache.num_gpu_blocks),
    "block_size": int(cache.block_size),
}))
"""


def probe_kv_capacity(
    model_path: str | Path,
    *,
    tensor_parallel: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
    dtype: str = "bfloat16",
    block_size: int = 0,
    max_num_batched_tokens: int = 0,
    max_num_seqs: int = 0,
    gpu_name: str = "unknown",
    cache_path: Path | None = None,
    timeout_s: int = 1800,
) -> KvCapacity:
    """Return the KV capacity a real vLLM engine allocates for this deployment.

    Args:
        model_path: HuggingFace checkpoint directory.
        tensor_parallel: TP degree. Needs that many free GPUs when probing.
        gpu_memory_utilization: vLLM's `--gpu-memory-utilization`.
        max_model_len: vLLM's `--max-model-len`. Affects the activation peak, so
            it is part of the cache key.
        dtype: Weight dtype.
        block_size: Tokens per block, or 0 to let vLLM choose. A hybrid model
            overrides it to align its mamba pages.
        max_num_batched_tokens: vLLM's `--max-num-batched-tokens`, or 0 for its
            default. Sets the activation peak, so it must match the deployment.
        max_num_seqs: vLLM's `--max-num-seqs`, or 0 for its default. On a hybrid
            model this also bounds the mamba state cache.
        gpu_name: Device identity, for the cache key.
        cache_path: Override the default cache location.
        timeout_s: Probe timeout.

    Returns:
        KvCapacity, with `source` recording whether it was measured or cached.

    Raises:
        RuntimeError: If the probe fails. Never falls back to an estimate, which
            would rescale every occupancy number downstream without failing.
    """
    cache_path = cache_path or Path(
        os.environ.get("SIMRL_KV_CACHE", str(DEFAULT_CACHE_PATH))
    )
    key = KvProbeKey(
        checkpoint_fingerprint=checkpoint_fingerprint(model_path),
        tensor_parallel=int(tensor_parallel),
        gpu_memory_utilization=round(float(gpu_memory_utilization), 4),
        max_model_len=int(max_model_len),
        dtype=str(dtype),
        gpu_name=str(gpu_name),
        vllm_version=_vllm_version(),
        block_size=int(block_size),
        max_num_batched_tokens=int(max_num_batched_tokens),
        max_num_seqs=int(max_num_seqs),
    )
    digest = key.digest()

    entries = _load_cache(cache_path)
    hit = entries.get(digest)
    if hit is not None:
        return KvCapacity(
            num_gpu_blocks=int(hit["num_gpu_blocks"]),
            block_size=int(hit["block_size"]),
            total_tokens=int(hit["num_gpu_blocks"]) * int(hit["block_size"]),
            source="cache",
        )


    srl_logger.info(
        "Probing real KV capacity for %s at tp=%d (loads the checkpoint; ~2-4 min). "
        "Caching to %s.",
        model_path,
        tensor_parallel,
        cache_path,
    )
    measured = _run_probe(
        model_path,
        tensor_parallel=tensor_parallel,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        timeout_s=timeout_s,
    )

    entries[digest] = {**measured, "key": asdict(key)}
    _store_cache(cache_path, entries)
    srl_logger.info(
        "Measured %d KV blocks x %d tokens = %d tokens at tp=%d.",
        measured["num_gpu_blocks"],
        measured["block_size"],
        measured["num_gpu_blocks"] * measured["block_size"],
        tensor_parallel,
    )
    return KvCapacity(
        num_gpu_blocks=measured["num_gpu_blocks"],
        block_size=measured["block_size"],
        total_tokens=measured["num_gpu_blocks"] * measured["block_size"],
        source="probe",
    )


def _vllm_version() -> str:
    """
    Return vLLM's `__version__` without importing vLLM.

    The value is read from vLLM's generated `_version.py`, a file of plain constants that
    `__version__` is re-exported from. Package metadata is not equivalent: it carries a
    `.precompiled` suffix, and a changed cache key would silently re-run the GPU probe.
    """
    try:
        spec = importlib.util.find_spec("vllm")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return "absent"
    for location in spec.submodule_search_locations or ():
        version_file = Path(location) / "_version.py"
        if not version_file.is_file():
            continue
        namespace: dict = {}
        try:
            exec(compile(version_file.read_text(), str(version_file), "exec"), namespace)
        except Exception:  # noqa: BLE001 - fall back to the import below
            break
        version = namespace.get("__version__")
        if isinstance(version, str):
            return version
    try:
        import vllm

        return str(vllm.__version__)
    except Exception:  # noqa: BLE001 - absence is a valid key component
        return "absent"


def _run_probe(
    model_path: str | Path,
    *,
    tensor_parallel: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    block_size: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    timeout_s: int,
) -> dict[str, int]:
    """Launch vLLM in a subprocess and read back its allocated block count."""
    command = [
        sys.executable,
        "-c",
        _PROBE_SOURCE,
        str(model_path),
        str(tensor_parallel),
        str(gpu_memory_utilization),
        str(max_model_len),
        str(dtype),
        str(block_size),
        str(max_num_batched_tokens),
        str(max_num_seqs),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"KV probe for {model_path} at tp={tensor_parallel} exceeded {timeout_s}s."
        ) from exc

    for line in completed.stdout.splitlines():
        if line.startswith("SIMRL_KV_RESULT "):
            return json.loads(line[len("SIMRL_KV_RESULT ") :])

    tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-25:])
    raise RuntimeError(
        f"KV probe for {model_path} at tp={tensor_parallel} did not report a block "
        f"count (exit {completed.returncode}). Last output:\n{tail}"
    )
