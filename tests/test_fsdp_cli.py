"""FSDP CLI wiring and Hydra configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from srl.cli.profile import build_parser


def test_parser_exposes_the_three_fsdp_workflows():
    parser = build_parser()
    for workflow in ("generate", "run", "collect"):
        args = parser.parse_args(["fsdp", workflow, "--config", "c.yaml"])
        assert args.backend_name == "fsdp"
        assert args.workflow == workflow
        assert callable(args.func)


def test_fsdp_run_accepts_selection_and_resume():
    parser = build_parser()
    args = parser.parse_args(
        ["fsdp", "run", "--config", "c.yaml", "--index", "2", "--resume", "--dry-run"]
    )
    assert args.index == 2
    assert args.resume is True
    assert args.dry_run is True


def test_regress_accepts_the_fsdp_models():
    from srl.cli.regress import build_parser as regress_parser

    args = regress_parser().parse_args(
        [
            "training",
            "--latency-model",
            "fsdp",
            "--memory-model",
            "fsdp",
            "--input",
            "in.jsonl",
            "--output",
            "out.json",
        ]
    )
    assert args.latency_model == "fsdp"
    assert args.memory_model == "fsdp"


def test_training_cost_config_carries_the_batching_mode():
    from srl.config.schema import TrainingCostConfig

    config = TrainingCostConfig()
    assert config.batching_mode == "pad_to_max"
    assert config.max_tokens_per_rank is None


def test_fsdp_defaults_yaml_is_loadable_and_documented():
    path = Path("srl/config/defaults/training_cost/fsdp.yaml")
    value = yaml.safe_load(path.read_text())
    assert value["name"] == "fsdp"
    assert "batching_mode" in value
    assert value["batching_mode"] in ("pad_to_max", "token_budget", "packed_no_pad")
    # Every knob a reader must choose needs its rationale in the file.
    assert path.read_text().count("#") >= 6


def test_example_sweep_varies_both_identifiability_axes():
    value = yaml.safe_load(Path("examples/profile/fsdp/qwen3_8b.yaml").read_text())
    search = value["search_space"]
    assert len(search["sequence_lengths"]) >= 3
    assert len(search["n_seqs_per_rank"]) >= 2
    assert sorted(search["shard_sizes"]) == [2, 4, 8]
    # num_microbatches is a required identifiability axis: the design row is
    # [1, nmb*tokens, nmb], so with a fixed nmb the design caps at rank 2 while
    # the fit demands rank 3. At least two values make all three coefficients separable.
    assert len(search["num_microbatches"]) >= 2


def test_example_sweep_retains_eight_steps():
    value = yaml.safe_load(Path("examples/profile/fsdp/qwen3_8b.yaml").read_text())
    training = value["training"]
    assert training["max_steps"] - training["warmup_steps"] == 8


def test_build_training_cost_resolves_the_fsdp_name(tmp_path: Path, monkeypatch):
    from omegaconf import OmegaConf

    from srl.cli.main import build_training_cost

    # A name with no artifact must fail loudly rather than silently falling back.
    cfg = OmegaConf.create({"name": "fsdp", "artifact_path": ""})
    with pytest.raises(Exception):
        build_training_cost(cfg)
