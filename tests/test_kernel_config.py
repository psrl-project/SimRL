"""Kernel profile config parsing, sweep grids, and path derivation."""

from __future__ import annotations

from pathlib import Path

import pytest

from srl.profiler.kernel.config import KernelProfileConfig
from srl.profiler.schema import ProfileValidationError

_FIXTURE_MODEL = Path("tests/fixtures/hf/qwen3_8b").resolve()


def _config_dict(**overrides) -> dict:
    value = {
        "model": str(_FIXTURE_MODEL),
        "hardware": {"gpu_name": "H20", "gpu_count": 1, "gpu_memory_mib": 97871},
        "output_dir": "outputs/profile/qwen3_8b/kernel_tp1",
        "experiment_name": "qwen3-8b",
        "parallelism": {"tensor_parallel": 1},
        "gemm": {"token_counts": [1, 64, 1024]},
        "attention": {
            "batch_sizes": [1, 4, 16],
            "context_seq_lens": [256, 1024],
            "generation_steps": [0, 1024],
            "block_size": 16,
            "max_context_tokens": 16384,
        },
        "measurement": {"warmup_iterations": 2, "active_iterations": 5},
    }
    value.update(overrides)
    return value


def test_from_dict_builds_both_sweeps(tmp_path):
    config = KernelProfileConfig.from_dict(_config_dict(), base_dir=tmp_path)
    assert config.sweep("gemm").token_counts == (1, 64, 1024)
    attention = config.sweep("context_attention")
    assert attention.batch_sizes == (1, 4, 16)
    assert attention.context_seq_lens == (256, 1024)
    assert attention.generation_steps == (0, 1024)
    assert attention.block_size == 16
    assert attention.max_context_tokens == 16384
    assert config.measurement.active_iterations == 5
    assert config.measurement.kv_cache_dtype == "bfloat16"


def test_table_root_and_gpu_are_derived(tmp_path):
    config = KernelProfileConfig.from_dict(_config_dict(), base_dir=tmp_path)
    assert config.table_root() == config.output_dir / "kernel"
    assert config.gpu() == "h20"


def test_generation_steps_may_include_zero(tmp_path):
    config = KernelProfileConfig.from_dict(_config_dict(), base_dir=tmp_path)
    assert 0 in config.sweep("generation_attention").generation_steps


def test_an_unconfigured_family_only_raises_when_its_sweep_is_asked_for(tmp_path):
    """Profiling one family must not require configuring every other one."""
    payload = _config_dict()
    del payload["attention"]
    config = KernelProfileConfig.from_dict(payload, base_dir=tmp_path)
    assert config.sweep("gemm").token_counts == (1, 64, 1024)
    with pytest.raises(ProfileValidationError, match="attention:"):
        config.sweep("context_attention")


def test_nonpositive_tensor_parallel_is_rejected(tmp_path):
    with pytest.raises(ProfileValidationError):
        KernelProfileConfig.from_dict(
            _config_dict(parallelism={"tensor_parallel": 0}), base_dir=tmp_path
        )


def test_empty_batch_sizes_is_rejected_by_name(tmp_path):
    payload = _config_dict()
    payload["attention"]["batch_sizes"] = []
    with pytest.raises(ProfileValidationError, match="attention.batch_sizes"):
        KernelProfileConfig.from_dict(payload, base_dir=tmp_path)


def test_negative_generation_step_is_rejected(tmp_path):
    payload = _config_dict()
    payload["attention"]["generation_steps"] = [-1, 1024]
    with pytest.raises(ProfileValidationError):
        KernelProfileConfig.from_dict(payload, base_dir=tmp_path)


def test_a_leftover_synthesis_key_is_ignored(tmp_path):
    payload = _config_dict()
    payload["synthesis"] = {"prefill": {"batch_sizes": [1], "seq_lens": [128]}}
    config = KernelProfileConfig.from_dict(payload, base_dir=tmp_path)
    assert not hasattr(config, "synthesis")
