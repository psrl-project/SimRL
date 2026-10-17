"""Tests for KV cache capacity sizing, checked against vLLM's own arithmetic."""

from __future__ import annotations

import json

import pytest

from srl.cost.rollout.kv_cache import KvCacheGeometry, dtype_bytes, kv_blocks_from_memory

# Qwen3-8B, from its config.json.
_QWEN3_8B = {
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 4096,
    "torch_dtype": "bfloat16",
}
_QWEN3_8B_PARAMS = 8_190_735_360

# What the real engine allocated at gpu_memory_utilization=0.9 on one H20.
_MEASURED_BLOCKS = 31_279


def _write_config(tmp_path, **overrides):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**_QWEN3_8B, **overrides}), encoding="utf-8")
    return tmp_path


class TestGeometry:
    def test_reads_gqa_kv_heads_not_attention_heads(self, tmp_path):
        """The KV head count is what makes a GQA model cheap to cache."""
        geometry = KvCacheGeometry.from_hf_config(_write_config(tmp_path))
        assert geometry.num_kv_heads == 8
        assert geometry.num_layers == 36
        assert geometry.head_dim == 128
        assert geometry.dtype_bytes == 2

    def test_head_dim_falls_back_to_hidden_over_heads(self, tmp_path):
        geometry = KvCacheGeometry.from_hf_config(_write_config(tmp_path, head_dim=None))
        assert geometry.head_dim == 4096 // 32

    def test_absent_kv_heads_means_no_grouped_query_attention(self, tmp_path):
        path = tmp_path / "config.json"
        config = {k: v for k, v in _QWEN3_8B.items() if k != "num_key_value_heads"}
        path.write_text(json.dumps(config), encoding="utf-8")
        assert KvCacheGeometry.from_hf_config(tmp_path).num_kv_heads == 32

    def test_page_size_matches_vllm(self, tmp_path):
        """2 * block_size * num_kv_heads * head_dim * dtype_bytes, per layer."""
        geometry = KvCacheGeometry.from_hf_config(_write_config(tmp_path))
        assert geometry.page_size_bytes(block_size=16) == 2 * 16 * 8 * 128 * 2
        assert geometry.bytes_per_block(block_size=16) == 2 * 16 * 8 * 128 * 2 * 36

    def test_tensor_parallelism_shards_the_kv_heads(self, tmp_path):
        geometry = KvCacheGeometry.from_hf_config(_write_config(tmp_path))
        assert geometry.page_size_bytes(16, tp_size=2) * 2 == geometry.page_size_bytes(16)

    def test_indivisible_tp_raises_rather_than_guessing(self, tmp_path):
        """vLLM replicates KV heads there, which this estimate does not model."""
        geometry = KvCacheGeometry.from_hf_config(_write_config(tmp_path))
        with pytest.raises(ValueError, match="replicates KV heads"):
            geometry.page_size_bytes(16, tp_size=16)

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No model config"):
            KvCacheGeometry.from_hf_config(tmp_path / "absent")

    def test_missing_field_names_itself(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"num_hidden_layers": 36}), encoding="utf-8")
        with pytest.raises(ValueError, match="num_attention_heads"):
            KvCacheGeometry.from_hf_config(tmp_path)


class TestDtypeBytes:
    def test_known_dtypes(self):
        assert dtype_bytes("bfloat16") == 2
        assert dtype_bytes("float32") == 4
        assert dtype_bytes("BFloat16") == 2

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            dtype_bytes("int3")


class TestBlocksFromMemory:
    def test_within_three_percent_of_the_measured_engine(self, tmp_path):
        """The residual is vLLM's measured non-KV peak, which a config cannot reveal."""
        geometry = KvCacheGeometry.from_hf_config(_write_config(tmp_path))
        blocks = kv_blocks_from_memory(geometry, 97_871, 0.9, _QWEN3_8B_PARAMS)
        assert blocks == pytest.approx(_MEASURED_BLOCKS, rel=0.035)
        # It overestimates, because the unmodelled term is always a subtraction.
        assert blocks > _MEASURED_BLOCKS

    def test_overhead_closes_the_gap(self, tmp_path):
        geometry = KvCacheGeometry.from_hf_config(_write_config(tmp_path))
        bare = kv_blocks_from_memory(geometry, 97_871, 0.9, _QWEN3_8B_PARAMS)
        corrected = kv_blocks_from_memory(
            geometry,
            97_871,
            0.9,
            _QWEN3_8B_PARAMS,
            non_kv_overhead_bytes=2 * 1024**3,
        )
        assert corrected < bare
        assert abs(corrected - _MEASURED_BLOCKS) < abs(bare - _MEASURED_BLOCKS)

    def test_lower_utilization_gives_fewer_blocks(self, tmp_path):
        geometry = KvCacheGeometry.from_hf_config(_write_config(tmp_path))
        assert kv_blocks_from_memory(geometry, 97_871, 0.5, _QWEN3_8B_PARAMS) < kv_blocks_from_memory(
            geometry, 97_871, 0.9, _QWEN3_8B_PARAMS
        )

    def test_weights_exceeding_the_budget_yields_zero(self, tmp_path):
        geometry = KvCacheGeometry.from_hf_config(_write_config(tmp_path))
        assert kv_blocks_from_memory(geometry, 8_000, 0.9, _QWEN3_8B_PARAMS) == 0

    def test_out_of_range_utilization_raises(self, tmp_path):
        geometry = KvCacheGeometry.from_hf_config(_write_config(tmp_path))
        for bad in (0.0, 1.5, -0.1):
            with pytest.raises(ValueError, match="gpu_memory_utilization"):
                kv_blocks_from_memory(geometry, 97_871, bad, _QWEN3_8B_PARAMS)
