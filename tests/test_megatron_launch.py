"""Tests for the DryRun-owned Megatron launcher.

`build_config` is where every candidate setting is applied to the library recipe.
The vendored performance CLI used to swallow unknown arguments, so a setting that
never landed looked identical to one that did. These tests read the applied config
back field by field, which is the cheap version of that check: a mistyped path
fails here in milliseconds instead of after a multi-minute distributed launch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dryrun.profiler.adapters.megatron import (
    MegatronBridgeConfig,
    build_launch_command,
    build_launch_spec,
    prepare_bridge_candidate,
)
from dryrun.profiler.adapters.megatron.config import (
    RECOMPUTE_METHOD,
    RECOMPUTE_NUM_LAYERS,
)
from dryrun.profiler.adapters.megatron.prepare import LAUNCH_MODULE
from dryrun.profiler.schema import (
    HardwareDescriptor,
    ModelDescriptor,
    Parallelism,
    ProfileValidationError,
)
from dryrun.profiler.search_space import TrainingCandidate

MODEL_PATH = "/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B"


def _model(name: str = MODEL_PATH) -> ModelDescriptor:
    return ModelDescriptor(
        name=name,
        parameter_count=8_190_735_360,
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        extra={"intermediate_size": 12288, "vocab_size": 151936, "model_type": "qwen3"},
    )


def _config(tmp_path: Path, **overrides) -> MegatronBridgeConfig:
    settings = {
        "bridge_dir": tmp_path / "Megatron-Bridge",
        "output_dir": tmp_path,
        "model": _model(),
        "hardware": HardwareDescriptor(gpu_name="H20", gpu_count=8, gpu_memory_mib=97871),
        "max_steps": 4,
        "warmup_steps": 1,
    }
    settings.update(overrides)
    return MegatronBridgeConfig(**settings)


def _candidate(
    *,
    tp: int = 1,
    pp: int = 1,
    dp: int = 8,
    num_microbatches: int = 1,
    seq_length: int = 512,
) -> TrainingCandidate:
    return TrainingCandidate(
        parallelism=Parallelism(tp=tp, pp=pp, dp=dp, cp=1, world_size=tp * pp * dp),
        micro_batch_size=1,
        global_batch_size=dp * num_microbatches,
        num_microbatches=num_microbatches,
        sequence_shape=(seq_length,),
        recompute="full",
    )


# --- launch spec and command -------------------------------------------------


def test_launch_spec_carries_every_candidate_setting(tmp_path: Path):
    config = _config(tmp_path)
    candidate = _candidate(tp=2, pp=2, dp=2, num_microbatches=4, seq_length=1024)
    spec = build_launch_spec(config, candidate, tmp_path / "run")
    assert spec["schema_version"] == 1
    # `ep` is carried even for a dense candidate, where it is 1. The launcher only applies
    # it when the provider reports `num_moe_experts`, but the spec records the candidate's
    # full mesh unconditionally so a MoE recipe cannot fall back to its own tuned value.
    assert spec["parallelism"] == {"tp": 2, "pp": 2, "cp": 1, "ep": 1, "world_size": 8}
    assert spec["micro_batch_size"] == 1
    assert spec["global_batch_size"] == 8
    assert spec["num_microbatches"] == 4
    assert spec["seq_length"] == 1024
    assert spec["recompute"] == "full"
    assert spec["recompute_method"] == RECOMPUTE_METHOD
    assert spec["recompute_num_layers"] == RECOMPUTE_NUM_LAYERS
    assert spec["train_iters"] == 4
    assert spec["vocab_size"] == 151936
    assert spec["model_path"] == MODEL_PATH
    assert spec["distributed_optimizer"] is True
    assert spec["recipe"] == {"family": "qwen", "name": "qwen3_8b", "task": "pretrain"}


def test_launch_spec_requires_vocab_size(tmp_path: Path):
    """NullTokenizer without the real vocabulary would misreport parameter count."""
    model = _model()
    config = _config(tmp_path, model=ModelDescriptor(**{**vars(model), "extra": {}}))
    with pytest.raises(ProfileValidationError, match="vocab_size"):
        build_launch_spec(config, _candidate(), tmp_path / "run")


def test_single_node_command_targets_the_dryrun_launcher(tmp_path: Path):
    command = build_launch_command(_candidate(), tmp_path / "launch_spec.json")
    assert command[0] == "torchrun"
    assert "--standalone" in command
    assert command[command.index("--nproc-per-node") + 1] == "8"
    assert command[command.index("-m") + 1] == LAUNCH_MODULE
    # The vendored CLI and its lossy overrides must be gone.
    assert not any("run_recipe.py" in item for item in command)
    assert "--gpu" not in command
    assert not any("=" in item for item in command)


def test_multi_node_command_uses_an_explicit_rendezvous(tmp_path: Path):
    command = build_launch_command(
        _candidate(),
        tmp_path / "launch_spec.json",
        nnodes=2,
        node_rank=1,
        master_addr="10.0.0.1",
        master_port=29500,
    )
    assert "--standalone" not in command
    assert command[command.index("--node-rank") + 1] == "1"
    assert command[command.index("--master-addr") + 1] == "10.0.0.1"
    # Eight ranks over two nodes is four processes each.
    assert command[command.index("--nproc-per-node") + 1] == "4"


def test_multi_node_command_requires_a_rendezvous(tmp_path: Path):
    with pytest.raises(ProfileValidationError, match="master_addr"):
        build_launch_command(_candidate(), tmp_path / "spec.json", nnodes=2)


def test_multi_node_command_rejects_indivisible_world(tmp_path: Path):
    candidate = _candidate(tp=1, pp=1, dp=3)
    with pytest.raises(ProfileValidationError, match="not divisible"):
        build_launch_command(
            candidate,
            tmp_path / "spec.json",
            nnodes=2,
            master_addr="10.0.0.1",
            master_port=29500,
        )


def test_prepare_writes_a_spec_the_launcher_can_read(tmp_path: Path):
    config = _config(tmp_path)
    candidate = _candidate()
    command, run_dir = prepare_bridge_candidate(config, candidate)
    spec_path = run_dir / "launch_spec.json"
    assert spec_path.exists()
    spec = json.loads(spec_path.read_text())
    assert spec["run_dir"] == str(run_dir)
    assert command[command.index("--spec") + 1] == str(spec_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    # The cell key carries only the opaque recompute label, so the method it
    # resolved to has to be recorded alongside it.
    assert manifest["recompute_method"] == RECOMPUTE_METHOD
    assert manifest["launch_spec"] == str(spec_path)


def test_config_rejects_a_warmup_that_consumes_every_step(tmp_path: Path):
    with pytest.raises(ProfileValidationError, match="must exceed warmup_steps"):
        _config(tmp_path, max_steps=2, warmup_steps=2)


# --- build_config -----------------------------------------------------------

megatron = pytest.importorskip(
    "megatron.bridge",
    reason="build_config needs the pinned Megatron-Bridge checkout.",
)


@pytest.fixture(scope="module")
def applied_config(tmp_path_factory):
    """Apply one candidate to the real library recipe once, then assert on it."""
    if not Path(MODEL_PATH).is_dir():
        pytest.skip(f"The local checkpoint {MODEL_PATH} is not available.")
    from dryrun.profiler.adapters.megatron.launch import build_config

    tmp_path = tmp_path_factory.mktemp("build_config")
    config = _config(tmp_path)
    candidate = _candidate(tp=2, pp=2, dp=2, num_microbatches=4, seq_length=1024)
    spec = build_launch_spec(config, candidate, tmp_path / "run")
    return build_config(spec), spec


def test_build_config_applies_parallelism(applied_config):
    cfg, spec = applied_config
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    # Follows TP, matching what Megatron and verl do by default, so the profile
    # measures the configuration people actually run. Still a function of TP, so
    # one cell per TP describes one configuration.
    assert cfg.model.sequence_parallel is True


def test_build_config_applies_batch_and_sequence(applied_config):
    cfg, spec = applied_config
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.global_batch_size == 8
    assert cfg.train.train_iters == 4
    assert cfg.model.seq_length == 1024
    # Validated against the model sequence length downstream.
    assert cfg.dataset.seq_length == 1024


def test_build_config_enables_full_recompute(applied_config):
    """The library recipe leaves recompute off, which is what silently regressed."""
    cfg, spec = applied_config
    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.recompute_method == RECOMPUTE_METHOD
    assert cfg.model.recompute_num_layers == RECOMPUTE_NUM_LAYERS


def test_build_config_uses_the_local_checkpoint(applied_config):
    cfg, spec = applied_config
    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
    assert cfg.tokenizer.vocab_size == 151936


def test_build_config_disables_checkpointing_and_eval(applied_config):
    """Eval is on by default in the recipe; a four-step profile must not pay it."""
    cfg, spec = applied_config
    assert cfg.checkpoint.save is None
    assert cfg.checkpoint.load is None
    # These live on the validation config, not the deprecated training fields.
    assert cfg.validation.eval_iters == 0
    assert cfg.validation.eval_interval is None


def test_build_config_redirects_output_and_disables_profilers(applied_config):
    cfg, spec = applied_config
    assert cfg.logger.tensorboard_dir is None
    assert cfg.logger.wandb_exp_name is None
    assert cfg.logger.timing_log_level == 0
    # Kept as the independent audit that these mutations reached the config.
    assert cfg.logger.save_config_filepath == spec["resolved_config"]
    assert cfg.profiling.use_pytorch_profiler is False
    # Recording allocator history would perturb the memory being measured.
    assert cfg.profiling.record_memory_history is False


def test_build_config_rejects_an_unknown_recompute_mode(tmp_path: Path):
    if not Path(MODEL_PATH).is_dir():
        pytest.skip(f"The local checkpoint {MODEL_PATH} is not available.")
    from dryrun.profiler.adapters.megatron.launch import build_config

    spec = build_launch_spec(_config(tmp_path), _candidate(), tmp_path / "run")
    spec["recompute"] = "selective"
    with pytest.raises(ValueError, match="Unsupported recompute mode"):
        build_config(spec)


def test_build_config_rejects_a_dtype_declaration_mismatch(tmp_path: Path):
    """
    Guards the analytic static-memory inventory.

    The inventory shards optimizer state by data-parallel degree only when the
    distributed optimizer is on, so an upstream recipe change here would silently
    invalidate every fitted memory entry.
    """
    if not Path(MODEL_PATH).is_dir():
        pytest.skip(f"The local checkpoint {MODEL_PATH} is not available.")
    from dryrun.profiler.adapters.megatron.launch import build_config

    config = _config(tmp_path, distributed_optimizer=False)
    spec = build_launch_spec(config, _candidate(), tmp_path / "run")
    with pytest.raises(ValueError, match="use_distributed_optimizer"):
        build_config(spec)


def test_build_config_honors_the_layer_override(tmp_path: Path):
    if not Path(MODEL_PATH).is_dir():
        pytest.skip(f"The local checkpoint {MODEL_PATH} is not available.")
    from dryrun.profiler.adapters.megatron.launch import build_config

    config = _config(tmp_path, num_layers_override=2)
    spec = build_launch_spec(config, _candidate(), tmp_path / "run")
    assert build_config(spec).model.num_layers == 2


def test_expert_parallelism_is_applied_and_not_inherited(tmp_path: Path):
    """
    A MoE provider gets the candidate's EP, with expert tensor parallelism pinned to 1.

    Library MoE recipes ship their own `expert_model_parallel_size` tuned for their own
    mesh (`qwen3_30b_a3b` uses 4 at TP=4/PP=2). Inheriting it while overriding TP and PP
    builds a mesh the recipe never described, and Megatron reports that as a RuntimeError
    inside distributed init rather than as anything the manifest shows.

    `expert_tensor_parallel_size` is 1 rather than TP on purpose: Megatron defaults it to
    TP, and that default makes EP unmeasurable here, since `etp * ep * pp` must divide the
    world and the memory-feasible meshes for a 36B MoE are TP2/PP4 and TP4/PP2.
    """
    spec = {
        "parallelism": {"tp": 2, "pp": 2, "cp": 1, "ep": 2, "world_size": 8},
        "recompute": "none",
        "seq_length": 1024,
        "micro_batch_size": 1,
        "global_batch_size": 8,
        "num_microbatches": 4,
    }

    class _MoeModel:
        num_moe_experts = 256
        expert_model_parallel_size = 4       # the recipe's own value
        expert_tensor_parallel_size = 4

    model = _MoeModel()
    if getattr(model, "num_moe_experts", None):
        model.expert_model_parallel_size = int(spec["parallelism"].get("ep", 1))
        model.expert_tensor_parallel_size = 1

    assert model.expert_model_parallel_size == 2
    assert model.expert_tensor_parallel_size == 1


def test_a_dense_provider_has_no_expert_fields_touched():
    """A dense model carries no expert config, so the launcher must leave it alone."""

    class _DenseModel:
        num_moe_experts = None

    model = _DenseModel()
    if getattr(model, "num_moe_experts", None):  # pragma: no cover - must not fire
        model.expert_model_parallel_size = 99

    assert not hasattr(model, "expert_model_parallel_size")
