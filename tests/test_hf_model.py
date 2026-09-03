"""HuggingFace checkpoint loading for profiling model identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dryrun.cli.profile import _generate_bridge_candidates
from dryrun.profiler.adapters.megatron import (
    MegatronBridgeConfig,
    prepare_bridge_candidate,
)
from dryrun.profiler.adapters.vllm import VLLMSweepConfig
from dryrun.profiler.hf import load_hf_model_descriptor, model_descriptor_from_yaml
from dryrun.profiler.schema import (
    HardwareDescriptor,
    ModelDescriptor,
    Parallelism,
    ProfileValidationError,
)
from dryrun.profiler.search_space import ModelConstraints, TrainingCandidate, enumerate_training_candidates

FIXTURES = Path(__file__).parent / "fixtures"
HF_QWEN = FIXTURES / "hf" / "qwen3_8b"
REAL_QWEN = Path("/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B")


def test_load_hf_model_descriptor_from_local_checkpoint():
    descriptor = load_hf_model_descriptor(HF_QWEN)
    assert descriptor.architecture == "Qwen3ForCausalLM"
    assert descriptor.num_layers == 36
    assert descriptor.hidden_size == 4096
    assert descriptor.num_attention_heads == 32
    assert descriptor.num_query_groups == 8
    assert descriptor.extra["intermediate_size"] == 12288
    assert descriptor.extra["vocab_size"] == 151936
    assert descriptor.parameter_count == 16381470720 // 2


def test_load_hf_model_descriptor_estimates_without_weight_index(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        (HF_QWEN / "config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    descriptor = load_hf_model_descriptor(tmp_path)
    assert descriptor.parameter_count is not None
    assert 8_000_000_000 < descriptor.parameter_count < 8_500_000_000


def test_yaml_model_rejects_inline_architecture_fields(tmp_path: Path):
    with pytest.raises(ProfileValidationError, match="HuggingFace path"):
        model_descriptor_from_yaml(
            {"name": "Qwen3-8B", "num_layers": 36, "hidden_size": 4096},
            base_dir=tmp_path,
        )


def test_vllm_yaml_model_is_hf_path(tmp_path: Path):
    config = VLLMSweepConfig.from_dict(
        {
            "model": str(HF_QWEN),
            "hardware": {"gpu_name": "H100", "gpu_count": 1},
            "output_dir": str(tmp_path),
            "experiment_name": "test",
            "run": {"max_parallel": 4},
            "phases": {"prefill": {"grid": {"random-output-len": [1]}}},
        },
        base_dir=tmp_path,
    )
    assert config.model.architecture == "Qwen3ForCausalLM"
    assert config.model.num_query_groups == 8
    assert config.backend_version == "0.26.0"
    assert config.source_commit == "568afb3a13806beb53bb2e6bd518269357b237c0"
    assert config.max_parallel == 4


def test_megatron_yaml_uses_hf_path_and_fixed_world_size(tmp_path: Path):
    raw = {
        "model": str(HF_QWEN),
        "hardware": {"gpu_name": "H100", "gpu_count": 8, "gpu_memory_mib": 81920},
        "output_dir": str(tmp_path),
        "experiment_name": "test",
        "search_space": {
            "tensor_parallel_sizes": [2],
            "pipeline_parallel_sizes": [2],
            "num_microbatches": [1, 4],
            "sequence_lengths": [512],
            "recompute": "full",
        },
    }
    config = MegatronBridgeConfig.from_dict(raw, base_dir=tmp_path)
    candidates = _generate_bridge_candidates(config, raw)
    assert config.model.num_layers == 36
    assert config.model_family_name == "qwen"
    assert config.model_recipe_name == "qwen3_8b"
    assert config.max_steps == 4
    assert config.warmup_steps == 1
    assert {candidate.parallelism.world_size for candidate in candidates} == {8}
    assert {candidate.parallelism.dp for candidate in candidates} == {2}
    assert {candidate.num_microbatches for candidate in candidates} == {1, 4}
    assert {candidate.global_batch_size for candidate in candidates} == {2, 8}
    assert {candidate.micro_batch_size for candidate in candidates} == {1}
    assert {candidate.recompute for candidate in candidates} == {"full"}
    assert {candidate.repeat for candidate in candidates} == {0}
    assert len(candidates) == 2


def test_fixed_world_size_assigns_remaining_ranks_to_dp():
    candidates = enumerate_training_candidates(
        world_sizes=[8],
        tensor_parallel_sizes=[2],
        pipeline_parallel_sizes=[2],
        micro_batch_sizes=[1],
        num_microbatches=[4],
        sequence_shapes=[[512]],
        constraints=ModelConstraints(
            num_layers=36,
            num_attention_heads=32,
            num_query_groups=8,
        ),
    )
    assert len(candidates) == 1
    assert candidates[0].parallelism.dp == 2
    assert candidates[0].num_microbatches == 4
    assert candidates[0].global_batch_size == 8
    assert candidates[0].recompute == "full"
    assert candidates[0].repeat == 0


def test_megatron_full_recompute_reaches_the_launch_spec(tmp_path: Path):
    """
    Recompute settings travel in the launch spec, not on a command line.

    The vendored performance CLI dropped unknown arguments, so the recompute
    overrides it was handed never reached the model. The spec is read directly by
    the launcher, which removes that whole class of silent loss.
    """
    config = MegatronBridgeConfig(
        bridge_dir=tmp_path / "Megatron-Bridge",
        output_dir=tmp_path,
        experiment_name="test",
        model=ModelDescriptor(
            name=str(HF_QWEN),
            extra={"vocab_size": 151936},
        ),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=8),
    )
    candidate = TrainingCandidate(
        parallelism=Parallelism(tp=2, pp=2, dp=2, cp=1),
        micro_batch_size=1,
        global_batch_size=8,
        num_microbatches=4,
        sequence_shape=(512,),
        recompute="full",
    )
    command, run_dir = prepare_bridge_candidate(config, candidate)
    spec = json.loads((run_dir / "launch_spec.json").read_text())
    assert spec["recompute"] == "full"
    assert spec["recompute_method"] == "uniform"
    assert spec["recompute_num_layers"] == 1
    assert spec["vocab_size"] == 151936
    assert spec["train_iters"] == 4
    # Nothing is marshalled through the vendored CLI any more.
    assert command[command.index("--spec") + 1] == str(run_dir / "launch_spec.json")
    assert not any("run_recipe.py" in item for item in command)


@pytest.mark.skipif(not REAL_QWEN.exists(), reason="Qwen3-8B checkpoint is not available.")
def test_load_real_qwen3_checkpoint():
    descriptor = load_hf_model_descriptor(REAL_QWEN)
    assert descriptor.architecture == "Qwen3ForCausalLM"
    assert descriptor.num_layers == 36
    assert descriptor.num_query_groups == 8
    assert descriptor.extra["intermediate_size"] == 12288
    assert descriptor.parameter_count is not None
    assert descriptor.parameter_count > 8_000_000_000


def test_megatron_config_carries_no_gpu_token(tmp_path: Path):
    """
    GPU identity lives only on the hardware descriptor.

    A second, differently spelled GPU field existed to feed the vendored CLI's
    `--gpu` choices, which rejected this cluster's accelerator outright. The
    launcher never needed it, so the duplicate source of truth is gone.
    """
    config = MegatronBridgeConfig.from_dict(
        {
            "model": str(HF_QWEN),
            "hardware": {"gpu_name": "Custom Accelerator", "gpu_count": 8},
            "output_dir": str(tmp_path),
        },
        base_dir=tmp_path,
    )

    assert not hasattr(config, "gpu_type")
    assert config.hardware.gpu_name == "Custom Accelerator"
