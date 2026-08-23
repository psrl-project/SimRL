"""Tests for Hydra configuration system."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from dryrun.cli.main import (
    build_distribution,
    build_policy,
    build_recompute_cost,
    build_rollout_cost,
    build_sync_cost,
    build_training_cost,
    build_training_parallelism,
    build_workload,
)
from dryrun.component.recompute.cost import FixedRecomputeCost
from dryrun.component.sync.cost import FixedSyncCost
from dryrun.config.schema import (
    DistributionConfig,
    PolicyConfig,
    RolloutCostConfig,
    SimulateConfig,
    WorkloadConfig,
)
from dryrun.cost.rollout import DistServe, UnifiedRoofline
from dryrun.cost.training import (
    AnalyticalTrainingLatency,
    FixedTrainingLatency,
    LinearTrainingCost,
    LinearTrainingLatency,
    LinearTrainingMemory,
    TrainingParallelism,
    TrainingWorkload,
)
from dryrun.policy.areal import ArealPolicy
from dryrun.policy.psrl import PSRLPolicy
from dryrun.policy.roll import RollPolicy
from dryrun.policy.slime import SlimePolicy
from dryrun.policy.verl import VerlPolicy


class TestStructuredConfigs:
    """Test that dataclass configs have correct defaults."""

    def test_simulate_defaults(self):
        cfg = SimulateConfig()
        assert cfg.job.batch_size == 8
        assert cfg.job.max_staleness == 2
        assert cfg.job.n_versions == 20
        assert cfg.job.max_prompt_length is None
        assert cfg.job.max_output_length is None
        assert cfg.rollout.n_instances == 1
        assert cfg.rollout.partial_rollout is True
        assert cfg.rollout.admission_control.reject_if_kv_full is True
        assert cfg.rollout.admission_control.reject_if_waiting is False
        assert cfg.rollout.admission_control.reject_if_running_full is False
        assert cfg.rollout_cost.name == "roofline"
        assert cfg.training_cost.name == "fixed"
        assert cfg.policy.name == "psrl"
        assert cfg.workload.output.name == "uniform"
        assert cfg.workload.prompt.name == "fixed"
        assert cfg.workload.prompt.lo == 512

    def test_rollout_cost_defaults(self):
        cfg = RolloutCostConfig()
        assert cfg.name == "roofline"
        assert cfg.F == 0.005
        assert cfg.W == 0.001
        assert cfg.G == 0.0001
        assert cfg.A_d == 1e-7

    def test_policy_defaults(self):
        cfg = PolicyConfig()
        assert cfg.name == "psrl"

    def test_distribution_defaults(self):
        cfg = DistributionConfig()
        assert cfg.name == "uniform"
        assert cfg.n == 200
        assert cfg.seed == 42

    def test_workload_defaults(self):
        cfg = WorkloadConfig()
        assert cfg.output.name == "uniform"
        assert cfg.prompt.name == "fixed"
        assert cfg.prompt.lo == 512
        assert cfg.prompt.hi == 512


class TestFactoryFunctions:
    """Test that factory functions produce correct instances."""

    def test_build_roofline(self):
        cfg = OmegaConf.create({"name": "roofline", "F": 0.01, "W": 0.002, "G": 0.0002, "A_p": 0, "A_d": 2e-7})
        model = build_rollout_cost(cfg)
        assert isinstance(model, UnifiedRoofline)
        assert model.F == 0.01
        assert model.A_d == 2e-7

    def test_build_distserve(self):
        cfg = OmegaConf.create({
            "name": "distserve", "h": 4096, "m": 11008,
            "C1": 1e-13, "C2": 1e-13, "C3": 0.001, "C4": 0, "C5": 1e-13,
        })
        model = build_rollout_cost(cfg)
        assert isinstance(model, DistServe)

    def test_build_unknown_cost_model(self):
        cfg = OmegaConf.create({"name": "nonexistent"})
        with pytest.raises(ValueError, match="Unknown rollout cost model"):
            build_rollout_cost(cfg)

    def test_build_all_policies(self):
        for name, cls in [
            ("psrl", PSRLPolicy),
            ("areal", ArealPolicy),
            ("roll", RollPolicy),
            ("slime", SlimePolicy),
            ("verl", VerlPolicy),
        ]:
            cfg = OmegaConf.create({"name": name})
            policy = build_policy(cfg, max_staleness=2)
            assert isinstance(policy, cls), f"Expected {cls}, got {type(policy)}."
            assert policy.name == name

    def test_build_verl_uses_staleness_fallback(self):
        fallback = build_policy(
            OmegaConf.create({"name": "verl", "staleness_threshold": None}),
            max_staleness=2,
        )
        assert isinstance(fallback, VerlPolicy)
        assert fallback.threshold == 2.0

        explicit = build_policy(
            OmegaConf.create({
                "name": "verl",
                "staleness_threshold": 0.25,
            }),
            max_staleness=2,
        )
        assert isinstance(explicit, VerlPolicy)
        assert explicit.threshold == 0.25

    def test_build_unknown_policy(self):
        cfg = OmegaConf.create({"name": "nonexistent"})
        with pytest.raises(ValueError, match="Unknown policy"):
            build_policy(cfg, max_staleness=2)

    def test_build_all_distributions(self):
        for name in ["uniform", "bimodal", "lognormal", "normal", "powerlaw"]:
            cfg = OmegaConf.create({"name": name, "n": 50, "seed": 42})
            lengths = build_distribution(cfg)
            assert len(lengths) == 50
            assert all(isinstance(length, int) for length in lengths)

    def test_build_normal_distribution_optional_hi(self):
        cfg = OmegaConf.create({
            "name": "normal",
            "n": 40,
            "seed": 0,
            "mu": 100.0,
            "sigma": 50.0,
            "lo": 16,
            "hi": 64,
        })
        lengths = build_distribution(cfg)
        assert len(lengths) == 40
        assert all(16 <= length <= 64 for length in lengths)

    def test_build_fixed_distribution(self):
        cfg = OmegaConf.create({"name": "fixed", "n": 10, "lo": 512})
        lengths = build_distribution(cfg)
        assert lengths == [512] * 10

    def test_build_from_trace_distribution(self, tmp_path):
        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text('{"_meta": true}\n{"length": 10}\n{"length": 20}\n')
        cfg = OmegaConf.create({"name": "from_trace", "trace_path": str(trace_path), "n": 0})
        lengths = build_distribution(cfg)
        assert lengths == [10, 20]

    def test_build_unknown_distribution(self):
        cfg = OmegaConf.create({"name": "nonexistent"})
        with pytest.raises(ValueError, match="Unknown workload distribution"):
            build_distribution(cfg)

    def test_build_workload_prompt_and_output(self):
        cfg = OmegaConf.create({
            "output": {"name": "uniform", "n": 20, "seed": 1, "lo": 50, "hi": 100},
            "prompt": {"name": "fixed", "n": 20, "seed": 1, "lo": 512},
        })
        prompt_lengths, output_lengths = build_workload(cfg)
        assert prompt_lengths == [512] * 20
        assert len(output_lengths) == 20

    def test_build_fixed_training_cost(self):
        cfg = OmegaConf.create({"name": "fixed", "latency_s": 2.5, "peak_memory_bytes": 123})
        model = build_training_cost(cfg)
        assert isinstance(model.latency_model, FixedTrainingLatency)
        estimate = model.estimate(
            TrainingWorkload(sequence_lengths=(100,) * 8, micro_batch_size=1),
            TrainingParallelism(),
        )
        assert estimate.latency_s == 2.5
        assert estimate.peak_memory_bytes == 123

    def test_build_analytical_training_cost(self):
        cfg = OmegaConf.create({
            "name": "analytical",
            "model_params": 8_000_000_000,
            "peak_flops": 312e12,
            "memory_bandwidth": 2e12,
        })
        model = build_training_cost(cfg)
        assert isinstance(model.latency_model, AnalyticalTrainingLatency)
        estimate = model.estimate(
            TrainingWorkload(sequence_lengths=(1000,) * 8, micro_batch_size=1),
            TrainingParallelism(),
        )
        assert estimate.latency_s > 0
        assert estimate.peak_memory_bytes > 0

    def test_build_linear_training_cost(self):
        cfg = OmegaConf.create({
            "name": "linear",
            "time_per_token": 1e-4,
            "base_memory_bytes": 80_000_000_000,
        })
        model = build_training_cost(cfg)
        assert isinstance(model, LinearTrainingCost)
        assert isinstance(model.latency_model, LinearTrainingLatency)
        assert isinstance(model.memory_model, LinearTrainingMemory)
        estimate = model.estimate(
            TrainingWorkload(sequence_lengths=(1000,) * 8, micro_batch_size=1),
            TrainingParallelism(tp=2, pp=1, dp=2, cp=1),
        )
        assert estimate.latency_s == pytest.approx(0.8)
        assert estimate.peak_memory_bytes == 20_000_000_000

    def test_build_training_parallelism(self):
        cfg = OmegaConf.create({"parallelism": {"tp": 2, "pp": 2, "dp": 4, "cp": 1}})
        parallelism = build_training_parallelism(cfg)
        assert parallelism == TrainingParallelism(tp=2, pp=2, dp=4, cp=1)

    def test_build_sync_cost(self):
        cfg = OmegaConf.create({"name": "fixed", "sync_time": 0.5})
        model = build_sync_cost(cfg)
        assert isinstance(model, FixedSyncCost)

    def test_build_recompute_cost(self):
        cfg = OmegaConf.create({"name": "fixed", "time_per_token": 0.001})
        model = build_recompute_cost(cfg)
        assert isinstance(model, FixedRecomputeCost)


class TestOmegaConfIntegration:
    """Test that OmegaConf properly resolves structured configs."""

    def test_from_dataclass(self):
        cfg = OmegaConf.structured(SimulateConfig())
        assert cfg.job.batch_size == 8
        assert cfg.rollout_cost.name == "roofline"
        assert cfg.policy.name == "psrl"

    def test_override(self):
        cfg = OmegaConf.structured(SimulateConfig())
        cfg = OmegaConf.merge(cfg, {"job": {"batch_size": 16}, "policy": {"name": "areal"}})
        assert cfg.job.batch_size == 16
        assert cfg.policy.name == "areal"

    def test_yaml_roundtrip(self, tmp_path):
        cfg = OmegaConf.structured(SimulateConfig())
        yaml_path = tmp_path / "test.yaml"
        OmegaConf.save(cfg, yaml_path)
        loaded = OmegaConf.load(yaml_path)
        assert loaded.job.batch_size == cfg.job.batch_size
        assert loaded.rollout_cost.name == cfg.rollout_cost.name
