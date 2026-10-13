"""HuggingFace checkpoint loading for profiling model identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dryrun.cli.profile import _generate_bridge_candidates
from dryrun.profiler.adapters.megatron import (
    MegatronBridgeConfig,
    infer_bridge_recipe,
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
HF_DEEPSEEK_14B = FIXTURES / "hf" / "deepseek_r1_distill_qwen_14b"
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


def test_qk_layernorm_is_detected_for_qwen3_and_absent_elsewhere(tmp_path: Path):
    """Two extra RMSNorms per layer, and they are the costliest kind per element.

    Measured on H20 at 25.2M elements, `qk_norm` costs 122.9 us against `rms_norm`'s
    40.8 us: a 128-wide row cannot saturate the memory system. So `ComposedTraining` must
    know whether they are present, and defaulting either way would be a real error in one
    direction. HuggingFace Qwen3 configs do not carry the flag, so the architecture name is
    the fallback; `self_attn.q_norm.weight` in the weight index is the ground truth.
    """
    assert load_hf_model_descriptor(HF_QWEN).extra["qk_layernorm"] is True

    config = json.loads((HF_QWEN / "config.json").read_text(encoding="utf-8"))
    config["architectures"] = ["LlamaForCausalLM"]
    config["model_type"] = "llama"
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert load_hf_model_descriptor(tmp_path).extra["qk_layernorm"] is False


def test_an_explicit_qk_layernorm_flag_overrides_the_architecture_guess(tmp_path: Path):
    """Megatron's `TransformerConfig` spells it out, and a stated value is authoritative."""
    config = json.loads((HF_QWEN / "config.json").read_text(encoding="utf-8"))
    config["qk_layernorm"] = False
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert load_hf_model_descriptor(tmp_path).extra["qk_layernorm"] is False


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


def test_bridge_recipe_resolves_qwen2_architecture_by_size():
    """
    A Qwen2-architecture checkpoint resolves to the recipe matching its parameter count.

    The previous implementation knew only Qwen3 and computed the name arithmetically as
    `f"qwen3_{round(params / 1e9)}b"`. That spelling names recipes the library does not
    ship: Megatron-Bridge has `qwen25_14b` but no `qwen2_14b`, and a 14.77B checkpoint
    rounds to 15 regardless. The failure surfaced inside the torchrun child, after the
    candidate had already paid for distributed setup, so the table is checked here.
    """
    model = load_hf_model_descriptor(HF_DEEPSEEK_14B)

    assert model.architecture == "Qwen2ForCausalLM"
    assert infer_bridge_recipe(model) == ("qwen", "qwen25_14b")


def test_bridge_recipe_still_resolves_qwen3():
    """The table must not regress the architecture it replaced a one-branch rule for."""
    assert infer_bridge_recipe(load_hf_model_descriptor(HF_QWEN)) == ("qwen", "qwen3_8b")


def test_bridge_recipe_refuses_a_size_no_recipe_covers(tmp_path: Path):
    """
    A parameter count far from every tabled recipe raises instead of picking the nearest.

    Silently resolving a 20B checkpoint to the 14B recipe would profile a different model
    than the one asked for, and the step times would look entirely reasonable.
    """
    checkpoint = tmp_path / "qwen2_20b"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 6144,
                "intermediate_size": 16384,
                "num_hidden_layers": 60,
                "num_attention_heads": 48,
                "num_key_value_heads": 8,
                "vocab_size": 152064,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfileValidationError, match="not within"):
        infer_bridge_recipe(load_hf_model_descriptor(checkpoint))


def test_megatron_yaml_recipe_override_requires_both_fields(tmp_path: Path):
    """
    `model_family_name` and `model_recipe_name` are set together or not at all.

    Inference raises for an architecture the table does not hold, and naming the recipe
    explicitly is the documented way out. Accepting one field alone would pair an override
    with an inferred half and resolve a recipe neither the table nor the caller asked for.
    """
    raw = {
        "model": str(HF_DEEPSEEK_14B),
        "hardware": {"gpu_name": "H20", "gpu_count": 8},
        "output_dir": str(tmp_path),
        "model_family_name": "qwen",
    }

    with pytest.raises(ProfileValidationError, match="must be set together"):
        MegatronBridgeConfig.from_dict(raw, base_dir=tmp_path)

    config = MegatronBridgeConfig.from_dict(
        {**raw, "model_recipe_name": "qwen25_32b"},
        base_dir=tmp_path,
    )
    assert (config.model_family_name, config.model_recipe_name) == ("qwen", "qwen25_32b")
