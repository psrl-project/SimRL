"""KV capacity: the probe-and-cache path, and the guards on its fallback.

The capacity cannot be derived from a config file. vLLM measures its peak
activation and non-torch memory at startup, and a hybrid model pads its mamba
pages to the attention page. So it is read from a real engine and cached. What
is testable without a GPU is the cache keying, and the refusals that stop the
fallback inventing a number.
"""

from __future__ import annotations

import json

import pytest

from srl.cost.rollout.kv_cache import (
    KvCacheGeometry,
    kv_blocks_from_memory,
    parameter_bytes_from_checkpoint,
)
from srl.cost.rollout.kv_probe import (
    KvProbeKey,
    checkpoint_fingerprint,
    probe_kv_capacity,
)

H20_MIB = 97_871


def _key(**overrides) -> KvProbeKey:
    base = dict(
        checkpoint_fingerprint="abc123",
        tensor_parallel=1,
        gpu_memory_utilization=0.9,
        max_model_len=32768,
        dtype="bfloat16",
        gpu_name="H20",
        vllm_version="0.26.1",
        block_size=0,
        max_num_batched_tokens=65536,
        max_num_seqs=300,
    )
    base.update(overrides)
    return KvProbeKey(**base)


def _write_checkpoint(tmp_path, *, total_size: int, config: dict | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(config or {"num_hidden_layers": 40}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}})
    )
    return tmp_path


class TestCacheKey:
    """Every field that moves the measurement must move the key.

    A key that ignores one serves a stale capacity for a different deployment.
    """

    @pytest.mark.parametrize(
        "field,value",
        [
            ("tensor_parallel", 2),
            ("gpu_memory_utilization", 0.85),
            ("max_model_len", 16384),
            ("dtype", "float16"),
            ("gpu_name", "A100"),
            ("vllm_version", "0.27.0"),
            ("block_size", 32),
            ("max_num_batched_tokens", 8192),
            ("max_num_seqs", 1024),
            ("checkpoint_fingerprint", "def456"),
        ],
    )
    def test_each_field_changes_the_digest(self, field: str, value):
        assert _key().digest() != _key(**{field: value}).digest()

    def test_identical_keys_agree(self):
        assert _key().digest() == _key().digest()


class TestCheckpointFingerprint:
    def test_differs_when_weights_change(self, tmp_path):
        a = _write_checkpoint(tmp_path / "a", total_size=1000)
        b = _write_checkpoint(tmp_path / "b", total_size=2000)
        assert checkpoint_fingerprint(a) != checkpoint_fingerprint(b)

    def test_differs_when_architecture_changes(self, tmp_path):
        a = _write_checkpoint(
            tmp_path / "a", total_size=1000, config={"num_hidden_layers": 40}
        )
        b = _write_checkpoint(
            tmp_path / "b", total_size=1000, config={"num_hidden_layers": 32}
        )
        assert checkpoint_fingerprint(a) != checkpoint_fingerprint(b)

    def test_survives_a_copy_to_a_new_path(self, tmp_path):
        """Identity is the weights, not the path: a move must not re-probe."""
        a = _write_checkpoint(tmp_path / "a", total_size=1000)
        b = _write_checkpoint(tmp_path / "elsewhere", total_size=1000)
        assert checkpoint_fingerprint(a) == checkpoint_fingerprint(b)


class TestProbeCache:
    def test_a_cached_entry_is_returned_without_probing(self, tmp_path, monkeypatch):
        """The whole point of the cache: no engine start on a hit."""
        model = _write_checkpoint(tmp_path / "model", total_size=1000)
        cache = tmp_path / "kv.json"

        def explode(*_args, **_kwargs):
            raise AssertionError("probe ran despite a cache hit")

        monkeypatch.setattr("srl.cost.rollout.kv_probe._run_probe", explode)
        monkeypatch.setattr(
            "srl.cost.rollout.kv_probe._vllm_version", lambda: "test-version"
        )
        key = KvProbeKey(
            checkpoint_fingerprint=checkpoint_fingerprint(model),
            tensor_parallel=1,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
            dtype="bfloat16",
            gpu_name="H20",
            vllm_version="test-version",
            block_size=0,
            max_num_batched_tokens=0,
            max_num_seqs=0,
        )
        cache.write_text(
            json.dumps(
                {
                    "schema": 2,
                    "entries": {key.digest(): {"num_gpu_blocks": 409, "block_size": 1056}},
                }
            )
        )

        got = probe_kv_capacity(model, gpu_name="H20", cache_path=cache)
        assert got.source == "cache"
        assert got.num_gpu_blocks == 409
        assert got.total_tokens == 409 * 1056

    def test_a_miss_probes_and_writes_back(self, tmp_path, monkeypatch):
        model = _write_checkpoint(tmp_path / "model", total_size=1000)
        cache = tmp_path / "kv.json"
        monkeypatch.setattr(
            "srl.cost.rollout.kv_probe._run_probe",
            lambda *a, **k: {"num_gpu_blocks": 777, "block_size": 16},
        )
        first = probe_kv_capacity(model, cache_path=cache)
        assert first.source == "probe"

        monkeypatch.setattr(
            "srl.cost.rollout.kv_probe._run_probe",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-probed")),
        )
        second = probe_kv_capacity(model, cache_path=cache)
        assert second.source == "cache"
        assert second.num_gpu_blocks == first.num_gpu_blocks

    @pytest.mark.parametrize(
        "contents",
        [
            pytest.param('{"schema": 1, "entries": {"k": {}}}', id="stale-schema"),
            pytest.param("{ not json", id="corrupt"),
        ],
    )
    def test_an_unusable_cache_re_probes_rather_than_failing(
        self, tmp_path, monkeypatch, contents: str
    ):
        """A capacity is cheap to re-measure and expensive to get silently wrong."""
        model = _write_checkpoint(tmp_path / "model", total_size=1000)
        cache = tmp_path / "kv.json"
        cache.write_text(contents)
        monkeypatch.setattr(
            "srl.cost.rollout.kv_probe._run_probe",
            lambda *a, **k: {"num_gpu_blocks": 5, "block_size": 16},
        )
        assert probe_kv_capacity(model, cache_path=cache).source == "probe"

    def test_a_failed_probe_raises_rather_than_estimating(self, tmp_path, monkeypatch):
        """A wrong capacity rescales every occupancy number without failing."""
        model = _write_checkpoint(tmp_path / "model", total_size=1000)
        monkeypatch.setattr(
            "srl.cost.rollout.kv_probe._run_probe",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("engine died")),
        )
        with pytest.raises(RuntimeError, match="engine died"):
            probe_kv_capacity(model, cache_path=tmp_path / "kv.json")


class TestFallbackArithmetic:
    """The `probe: false` path. Still guarded, because it is still wrong-able."""

    def test_refuses_without_a_weight_footprint(self):
        """The largest subtraction in the budget must never be defaulted."""
        geometry = KvCacheGeometry(num_layers=10, num_kv_heads=2, head_dim=256, dtype_bytes=2)
        with pytest.raises(ValueError, match="weight_bytes_total or parameter_count"):
            kv_blocks_from_memory(
                geometry, gpu_memory_mib=H20_MIB, gpu_memory_utilization=0.9
            )

    def test_checkpoint_weights_win_over_a_configured_count(self):
        """A stale `parameter_count` from another model must not bite."""
        geometry = KvCacheGeometry(num_layers=10, num_kv_heads=2, head_dim=256, dtype_bytes=2)
        shared = dict(
            gpu_memory_mib=H20_MIB, gpu_memory_utilization=0.9, block_size=16, tp_size=1
        )
        weights = int(66.97 * 1024**3)
        assert kv_blocks_from_memory(
            geometry, weight_bytes_total=weights, **shared
        ) == kv_blocks_from_memory(
            geometry,
            parameter_count=8_190_735_360,  # Qwen3-8B: the original bug
            weight_bytes_total=weights,
            **shared,
        )

    def test_reads_weight_size_from_the_safetensors_index(self, tmp_path):
        model = _write_checkpoint(tmp_path / "m", total_size=71_900_000_000)
        assert parameter_bytes_from_checkpoint(model) == 71_900_000_000

    def test_returns_none_without_an_index(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert parameter_bytes_from_checkpoint(tmp_path) is None


class TestHybridLayerCount:
    """Only full-attention layers hold a KV cache; the rest hold mamba state."""

    def test_qwen35_counts_ten_of_forty(self, tmp_path):
        config = {
            "text_config": {
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "hidden_size": 2048,
                "full_attention_interval": 4,
            }
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        geometry = KvCacheGeometry.from_hf_config(tmp_path)
        assert geometry.num_layers == 10
        assert geometry.num_kv_heads == 2
        assert geometry.head_dim == 256

    def test_a_dense_model_counts_every_layer(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "num_hidden_layers": 32,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "hidden_size": 4096,
                }
            )
        )
        assert KvCacheGeometry.from_hf_config(tmp_path).num_layers == 32
