"""FSDP profile configuration, launch specs, and manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from srl.profiler.adapters.fsdp import (
    FsdpProfileConfig,
    build_launch_command,
    build_launch_spec,
    prepare_fsdp_candidate,
    read_candidate_manifest,
    write_candidate_manifest,
)
from srl.profiler.fsdp_search_space import FsdpCandidate
from srl.profiler.schema import HardwareDescriptor, ModelDescriptor, ProfileValidationError


def _model() -> ModelDescriptor:
    return ModelDescriptor(
        name="Qwen3-8B",
        architecture="Qwen3ForCausalLM",
        parameter_count=8_190_735_360,
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        extra={"model_type": "qwen3", "vocab_size": 151936},
    )


def _hardware() -> HardwareDescriptor:
    return HardwareDescriptor(
        gpu_name="H20", gpu_count=8, gpu_memory_mib=97871, node_count=1, gpus_per_node=8
    )


def _config(tmp_path: Path, **overrides) -> FsdpProfileConfig:
    defaults = {
        "output_dir": tmp_path,
        "model": _model(),
        "hardware": _hardware(),
        "model_path": "/models/Qwen3-8B",
    }
    return FsdpProfileConfig(**{**defaults, **overrides})


def _candidate(**overrides) -> FsdpCandidate:
    defaults = {
        "shard_size": 4,
        "replicate_size": 2,
        "n_seqs_per_rank": 2,
        "sequence_length": 1024,
        "gradient_checkpointing": True,
        "optimizer_offload": False,
        "reshard_after_forward": True,
    }
    return FsdpCandidate(**{**defaults, **overrides})


def test_defaults_retain_eight_measured_steps():
    """Twelve steps discarding four: one warmup step does not reach steady state."""
    config = _config(Path("/tmp"))
    assert config.max_steps == 12
    assert config.warmup_steps == 4
    # Boundary: a warmup one below max_steps must construct and leave exactly one
    # measured step, rather than being rejected.
    boundary = _config(Path("/tmp"), max_steps=5, warmup_steps=4)
    assert boundary.max_steps - boundary.warmup_steps == 1


def test_rejects_a_warmup_that_consumes_every_step(tmp_path: Path):
    with pytest.raises(ProfileValidationError):
        _config(tmp_path, max_steps=4, warmup_steps=4)


def test_backend_version_records_the_torch_version(tmp_path: Path):
    """The backend is torch itself, so reproducibility is pinned to its version."""
    import torch

    assert _config(tmp_path).backend_version == torch.__version__


def test_raw_root_separates_experiments(tmp_path: Path):
    plain = _config(tmp_path).raw_root
    named = _config(tmp_path, experiment_name="sweep_a").raw_root
    assert plain == tmp_path / "raw" / "fsdp"
    assert named == tmp_path / "raw" / "fsdp" / "sweep_a"


def test_from_dict_reads_the_yaml_shape(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "num_hidden_layers": 36,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 151936,
            }
        ),
        encoding="utf-8",
    )
    config = FsdpProfileConfig.from_dict(
        {
            "model": str(tmp_path),
            "hardware": {
                "gpu_name": "H20",
                "gpu_count": 8,
                "gpu_memory_mib": 97871,
                "node_count": 1,
                "gpus_per_node": 8,
            },
            "output_dir": "out",
            "training": {
                "max_steps": 12,
                "warmup_steps": 4,
                "num_layers_override": 2,
                "optimizer": "adam",
            },
            "profiling": {"allow_incomplete": True},
            "environment": {"HF_HUB_OFFLINE": "1"},
        },
        base_dir=tmp_path,
    )
    assert config.num_layers_override == 2
    assert config.allow_incomplete is True
    assert config.environment == {"HF_HUB_OFFLINE": "1"}
    assert config.output_dir == (tmp_path / "out").resolve()


def test_launch_spec_carries_the_mesh_not_tensor_degrees(tmp_path: Path):
    spec = build_launch_spec(_config(tmp_path), _candidate(), tmp_path)
    assert spec["mesh"] == {"replicate_size": 2, "shard_size": 4}
    assert "tp" not in spec
    assert "pp" not in spec


def test_launch_spec_carries_the_candidate_microbatch_count(tmp_path: Path):
    """A constant here would silently profile one microbatch for a nmb=3 candidate."""
    spec = build_launch_spec(_config(tmp_path), _candidate(num_microbatches=3), tmp_path)
    assert spec["num_microbatches"] == 3


def test_launch_spec_is_versioned_and_self_contained(tmp_path: Path):
    spec = build_launch_spec(_config(tmp_path), _candidate(), tmp_path)
    assert spec["schema_version"] == 1
    for key in (
        "model_path",
        "sequence_length",
        "n_seqs_per_rank",
        "num_microbatches",
        "train_iters",
        "gradient_checkpointing",
        "optimizer_offload",
        "reshard_after_forward",
        "parameter_dtype",
        "gradient_dtype",
        "optimizer_state_dtype",
        "attn_implementation",
        "run_dir",
    ):
        assert key in spec, key
    # A spec must survive a round trip through a file, which is how the child reads it.
    assert json.loads(json.dumps(spec))["mesh"]["shard_size"] == 4


def test_launch_spec_propagates_the_layer_override(tmp_path: Path):
    spec = build_launch_spec(_config(tmp_path, num_layers_override=2), _candidate(), tmp_path)
    assert spec["num_layers_override"] == 2


def test_launch_command_requests_one_process_per_rank(tmp_path: Path):
    command = build_launch_command(_candidate(), tmp_path / "launch_spec.json")
    assert command[0] == "torchrun"
    assert "--nproc-per-node" in command
    assert command[command.index("--nproc-per-node") + 1] == "8"
    assert "srl.profiler.adapters.fsdp.launch" in command


def test_launch_command_supports_multiple_nodes(tmp_path: Path):
    command = build_launch_command(
        _candidate(),
        tmp_path / "launch_spec.json",
        nnodes=2,
        node_rank=1,
        master_addr="10.0.0.1",
        master_port=29500,
    )
    assert "--standalone" not in command
    assert command[command.index("--nnodes") + 1] == "2"
    assert command[command.index("--node-rank") + 1] == "1"
    assert command[command.index("--master-addr") + 1] == "10.0.0.1"
    # Per-node process count, not the world size.
    assert command[command.index("--nproc-per-node") + 1] == "4"


def test_prepare_writes_a_spec_and_an_immutable_manifest(tmp_path: Path):
    config = _config(tmp_path)
    candidate = _candidate()
    command, run_dir = prepare_fsdp_candidate(config, candidate)
    assert run_dir == config.raw_root / candidate.candidate_id
    spec = json.loads((run_dir / "launch_spec.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert spec["mesh"]["shard_size"] == 4
    assert manifest["adapter"] == "fsdp"
    assert manifest["candidate"]["candidate_id"] == candidate.candidate_id
    assert manifest["command"] == command
    assert manifest["backend_version"]


def test_manifest_round_trips_candidates(tmp_path: Path):
    candidates = [_candidate(), _candidate(sequence_length=2048), _candidate(num_microbatches=2)]
    path = write_candidate_manifest(tmp_path / "candidates.json", candidates)
    assert read_candidate_manifest(path) == candidates


def test_manifest_rejects_an_unknown_schema(tmp_path: Path):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"schema_version": 99, "candidates": []}), encoding="utf-8")
    with pytest.raises(ProfileValidationError):
        read_candidate_manifest(path)


def test_rejects_multi_node_without_rendezvous(tmp_path: Path):
    """torchrun fails with an opaque network error when master_addr/master_port are absent;
    an early validation error names the missing argument instead."""
    with pytest.raises(ProfileValidationError, match="master_addr and master_port"):
        build_launch_command(
            _candidate(),
            tmp_path / "spec.json",
            nnodes=2,
            node_rank=0,
        )


def test_rejects_out_of_range_node_rank(tmp_path: Path):
    with pytest.raises(ProfileValidationError, match=r"node_rank 5 is outside \[0, 2\)"):
        build_launch_command(
            _candidate(),
            tmp_path / "spec.json",
            nnodes=2,
            node_rank=5,
            master_addr="10.0.0.1",
            master_port=29500,
        )


def test_rejects_non_positive_nnodes(tmp_path: Path):
    """nnodes=0 is meaningless; the divisibility message would be misleading."""
    with pytest.raises(ProfileValidationError, match="nnodes must be positive"):
        build_launch_command(_candidate(), tmp_path / "spec.json", nnodes=0)


def test_rejects_model_dict_without_path_key(tmp_path: Path):
    """A dict without 'path' would silently use model.name as a filesystem path,
    deferring the failure to an opaque torch.load error in the torchrun child."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "num_hidden_layers": 36,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 151936,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProfileValidationError, match="'path' key"):
        FsdpProfileConfig.from_dict(
            {
                # "name" is accepted by model_descriptor_from_yaml as a checkpoint
                # source, but "path" is what the child needs as model_path.  Without
                # "path" the old code silently used model.name (a label or resolved
                # source string) as a filesystem path, hiding the misconfiguration.
                "model": {"name": str(tmp_path)},
                "hardware": {
                    "gpu_name": "H20",
                    "gpu_count": 8,
                    "gpu_memory_mib": 97871,
                    "node_count": 1,
                    "gpus_per_node": 8,
                },
            },
            base_dir=tmp_path,
        )
