"""Tests for Hydra configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import dryrun.config
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
from dryrun.component.recompute.cost import FixedRecomputeCost, TrainingForwardRecomputeCost
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
    FixedTrainingCost,
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

    def test_dynamic_parallel_defaults_are_inert(self):
        cfg = SimulateConfig()
        assert cfg.dynamic_parallel.enabled is False
        assert cfg.dynamic_parallel.n_gpus == 4
        assert list(cfg.dynamic_parallel.tp_candidates) == [1, 2]
        assert cfg.dynamic_parallel.strategy == "search"
        assert list(cfg.dynamic_parallel.tp_per_instance) == []

    def test_dynamic_parallel_is_present_in_the_composed_config(self):
        """The CLI reads the COMPOSED config, not the dataclass.

        simulate.yaml enumerates its own top-level keys, so a dataclass field
        with no matching YAML block is absent from what Hydra composes and
        `cfg.dynamic_parallel.enabled` raises ConfigAttributeError at runtime while
        every dataclass-level test still passes. Measured: that was the state
        until the YAML block was added.
        """
        import os

        from hydra import compose, initialize_config_dir

        from dryrun.config import register_configs

        register_configs()
        config_dir = os.path.abspath("dryrun/config/defaults")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="simulate")
            assert cfg.dynamic_parallel.enabled is False
            assert cfg.dynamic_parallel.n_gpus == 4
            assert list(cfg.dynamic_parallel.tp_candidates) == [1, 2]
            assert cfg.dynamic_parallel.strategy == "search"
            assert list(cfg.dynamic_parallel.tp_per_instance) == []

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

    def test_build_unknown_distribution(self):
        cfg = OmegaConf.create({"name": "nonexistent"})
        with pytest.raises(ValueError, match="Unknown workload distribution"):
            build_distribution(cfg)

    def test_build_workload_prompt_and_output(self):
        cfg = OmegaConf.create({
            "output": {"name": "uniform", "n": 20, "seed": 1, "lo": 50, "hi": 100},
            "prompt": {"name": "fixed", "n": 20, "seed": 1, "lo": 512},
        })
        workload = build_workload(cfg)
        assert workload.n_requests == 20
        assert {workload.request(i)[1] for i in range(20)} == {512}

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
        model = build_recompute_cost(cfg, FixedTrainingCost(latency_s=1.0, peak_memory_bytes=0))
        assert isinstance(model, FixedRecomputeCost)

    def test_build_training_forward_recompute_cost(self):
        """The log-prob pass priced off the same training model as the optimizer step."""
        training = FixedTrainingCost(latency_s=100.0, peak_memory_bytes=0)
        cfg = OmegaConf.create(
            {
                "name": "training_forward",
                "hidden_size": 4096,
                "vocab_size": 151936,
                "parameter_count": 8_190_735_360,
            }
        )
        model = build_recompute_cost(cfg, training)
        assert isinstance(model, TrainingForwardRecomputeCost)
        # Full activation recompute makes a step 8 FLOPs per param per token and
        # the forward pass 2, so a quarter, plus the unembedding correction.
        assert model.forward_fraction("full") == pytest.approx(0.269, abs=0.002)
        # Without checkpointing a step is only 6, so the forward share is larger.
        assert model.forward_fraction("none") > model.forward_fraction("full")

    def test_training_forward_recompute_needs_the_geometry(self):
        cfg = OmegaConf.create({"name": "training_forward"})
        with pytest.raises(ValueError, match="hidden_size"):
            build_recompute_cost(cfg, FixedTrainingCost(latency_s=1.0, peak_memory_bytes=0))

    def test_unknown_recompute_model_raises(self):
        cfg = OmegaConf.create({"name": "nonesuch"})
        with pytest.raises(ValueError, match="Unknown recompute cost model"):
            build_recompute_cost(cfg, FixedTrainingCost(latency_s=1.0, peak_memory_bytes=0))


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


class TestWorkloadSource:
    def test_distributions_source_builds_single_response_groups(self):
        cfg = OmegaConf.create({
            "source": "distributions",
            "output": {"name": "uniform", "n": 20, "seed": 1, "lo": 50, "hi": 100},
            "prompt": {"name": "fixed", "n": 20, "seed": 1, "lo": 512},
        })
        workload = build_workload(cfg)
        assert workload.group_size == 1
        assert workload.n_requests == 20

    def test_source_defaults_to_distributions(self):
        cfg = OmegaConf.create({
            "output": {"name": "uniform", "n": 4, "seed": 1, "lo": 50, "hi": 100},
            "prompt": {"name": "fixed", "n": 4, "seed": 1, "lo": 512},
        })
        assert build_workload(cfg).group_size == 1

    def test_trace_source_reads_groups(self, tmp_path):
        from dryrun.workload.trace import RequestGroup, write_group_trace

        path = write_group_trace(
            [RequestGroup(0, 100, (7, 8)), RequestGroup(1, 200, (9, 10))],
            tmp_path / "joint.jsonl",
            metadata={"max_output_tokens": 10},
        )
        cfg = OmegaConf.create({"source": "trace", "trace": {"path": str(path), "n_groups": 0}})
        workload = build_workload(cfg)
        assert workload.group_size == 2
        assert workload.n_requests == 4

    def test_trace_source_requires_a_path(self):
        cfg = OmegaConf.create({"source": "trace", "trace": {"path": "", "n_groups": 0}})
        with pytest.raises(ValueError, match="workload.trace.path is required"):
            build_workload(cfg)

    def test_unknown_source_is_rejected(self):
        cfg = OmegaConf.create({"source": "nonsense"})
        with pytest.raises(ValueError, match="Unknown workload source"):
            build_workload(cfg)


class TestWorkloadConfigGroups:
    def test_dapo_trace_group_selects_the_trace_source(self):
        from hydra import compose, initialize_config_dir

        from dryrun.config import register_configs

        register_configs()
        config_dir = str(Path(dryrun.config.__file__).parent / "defaults")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="simulate", overrides=["workload=dapo_trace"])
        assert cfg.workload.source == "trace"


class TestDynamicParallelTableRoots:
    """Per-TP kernel table roots for the heterogeneous fleet.

    The composed cost model reads measured GEMM and attention tables that are sharded
    per TP degree, so one `rollout_cost.kernel_table_root` cannot serve every cohort.
    Pointing a tp=2 model at tp=1 tables raises KernelCoverageError rather than
    returning a wrong number, so the failure is loud -- but it must be configured.
    """

    @staticmethod
    def _hetero(**overrides):
        from omegaconf import OmegaConf

        base = {
            "enabled": True,
            "n_gpus": 4,
            "tp_candidates": [1, 2],
            "strategy": "search",
            "tp_per_instance": [],
            "kernel_table_roots": {},
        }
        base.update(overrides)
        return OmegaConf.create(base)

    def test_non_composed_models_need_no_tables(self):
        from dryrun.cli.main import _table_roots_per_tp

        assert _table_roots_per_tp(self._hetero(), (1, 2), "roofline") == {}

    def test_composed_requires_a_root_for_every_candidate(self):
        import pytest

        from dryrun.cli.main import _table_roots_per_tp

        partial = self._hetero(kernel_table_roots={1: "a/tp1"})
        with pytest.raises(ValueError, match="kernel_table_roots"):
            _table_roots_per_tp(partial, (1, 2), "composed")

    def test_composed_accepts_int_and_string_keys(self):
        from dryrun.cli.main import _table_roots_per_tp

        # A dataclass default yields int keys; a Hydra override
        # (`kernel_table_roots.2=...`) yields strings. Both must resolve.
        ints = self._hetero(kernel_table_roots={1: "a/tp1", 2: "a/tp2"})
        strs = self._hetero(kernel_table_roots={"1": "a/tp1", "2": "a/tp2"})
        expected = {1: "a/tp1", 2: "a/tp2"}
        assert _table_roots_per_tp(ints, (1, 2), "composed") == expected
        assert _table_roots_per_tp(strs, (1, 2), "composed") == expected

    def test_unprofiled_tp_degree_is_refused(self):
        """tp4/tp8 carry collective tables only, so they cannot price a step.

        Refusing at config time names the setting; letting it through surfaces a
        KernelCoverageError about a GEMM shape much further from the cause.
        """
        import pytest
        from omegaconf import OmegaConf

        from dryrun.cli.main import build_dynamic_parallel

        cfg = OmegaConf.create(
            {
                "dynamic_parallel": self._hetero(tp_candidates=[1, 4]),
                "rollout_cost": {"name": "roofline", "parallelism": {"tp": 1}},
                "rollout": {"router": {"strategy": "version_aware_tp"}},
            }
        )
        with pytest.raises(ValueError, match="unprofiled TP degrees"):
            build_dynamic_parallel(cfg)
