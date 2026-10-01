"""Hydra-based CLI entrypoint for DryRun-RL."""

from __future__ import annotations

import inspect
import os
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .._logging import dryrun_logger, setup_logging
from ..component.recompute.cost import (
    AnalyticalRecomputeCost,
    FixedRecomputeCost,
    RecomputeCostModel,
    TrainingForwardRecomputeCost,
)
from ..component.sync.cost import BandwidthSyncCost, FixedSyncCost, SyncCostModel
from ..config import register_configs
from ..cost.rollout import (
    ROLLOUT_MODEL_REGISTRY,
    DistServe,
    RolloutCostModel,
    UnifiedRoofline,
)
from ..cost.rollout.kv_cache import KvCacheGeometry, kv_blocks_from_memory
from ..cost.training import (
    TRAINING_MODEL_REGISTRY,
    AnalyticalTrainingCost,
    FixedTrainingCost,
    LinearTrainingCost,
    TrainingCostModel,
    TrainingParallelism,
)
from ..policy.areal import ArealPolicy
from ..policy.base import StalenessPolicy
from ..policy.psrl import PSRLPolicy
from ..policy.roll import RollPolicy
from ..policy.slime import SlimePolicy
from ..policy.verl import VerlPolicy
from ..profiler.schema import Parallelism, mib_to_bytes
from ..simulator import SimConfig, Simulator
from ..telemetry.store import JsonlStore
from ..visualizer.plots import SimPlotter
from ..workload.distributions import bimodal, fixed, lognormal, normal, powerlaw, uniform
from ..workload.grouped import GroupedWorkload

register_configs()


def _rollout_parallelism(cfg: DictConfig) -> Parallelism:
    value = cfg.get("parallelism", {})
    return Parallelism(
        tp=value.get("tp", 1),
        pp=value.get("pp", 1),
        dp=value.get("dp", 1),
        cp=value.get("cp", 1),
    )


def _build_composed_rollout(cfg: DictConfig, parallelism: Parallelism) -> RolloutCostModel:
    """Build a ComposedRollout from measured kernel tables."""
    from ..cost.kernel.database import KernelDatabase
    from ..cost.kernel.model_ops import ModelOps
    from ..cost.rollout.composed import ComposedRollout

    for field in ("kernel_table_root", "gpu", "model_path"):
        if not cfg.get(field, ""):
            raise ValueError(f"rollout_cost.{field} is required for the composed model.")
    database = KernelDatabase.from_root(
        cfg.kernel_table_root,
        gpu=cfg.gpu,
        utility_surface=cfg.get("utility_surface", "pchip"),
    )
    ops = ModelOps.from_checkpoint(cfg.model_path, tensor_parallel=parallelism.tp)
    return ComposedRollout(
        database,
        ops,
        tensor_parallel=parallelism.tp,
        stride=cfg.get("decode_integration_stride", 1024),
        attention_backend=cfg.get("attention_backend", "") or None,
        saturation_step_threshold=cfg.get("saturation_step_threshold", 1 << 62),
    )


def build_rollout_cost(cfg: DictConfig) -> RolloutCostModel:
    """Instantiate a rollout cost model from config."""
    name = cfg.get("name", "roofline")
    artifact_path = cfg.get("artifact_path", "")
    parallelism = _rollout_parallelism(cfg)
    model_type = ROLLOUT_MODEL_REGISTRY.get(name)
    if model_type is None:
        raise ValueError(f"Unknown rollout cost model: {name!r}.")
    if artifact_path:
        return model_type.from_artifact(artifact_path, parallelism)
    if name == "roofline":
        return UnifiedRoofline(
            F=cfg.get("F", 0.005),
            W=cfg.get("W", 0.001),
            G=cfg.get("G", 0.0001),
            A_p=cfg.get("A_p", 0.0),
            A_d=cfg.get("A_d", 1e-7),
            b=cfg.get("b", 16),
        )
    if name == "psrl":
        raise ValueError("rollout_cost.artifact_path is required for psrl.")
    if name == "distserve":
        return DistServe(
            h=cfg.get("h", 4096),
            m=cfg.get("m", 11008),
            C1=cfg.get("C1", 0.0),
            C2=cfg.get("C2", 0.0),
            C3=cfg.get("C3", 0.0),
            C4=cfg.get("C4", 0.0),
            C5=cfg.get("C5", 0.0),
            b=cfg.get("b", 16),
            F=cfg.get("F", 0.0),
        )
    if name == "composed":
        return _build_composed_rollout(cfg, parallelism)
    raise AssertionError(f"Unhandled rollout cost registry entry: {name!r}.")


def build_policy(cfg: DictConfig, max_staleness: int) -> StalenessPolicy:
    """Instantiate a staleness policy from config."""
    name = cfg.get("name", "psrl")
    if name == "psrl":
        return PSRLPolicy(
            proactive_filter=cfg.get("proactive_filter", False),
            proactive_threshold=cfg.get("proactive_threshold", 0),
        )
    if name == "areal":
        return ArealPolicy(max_inflight=cfg.get("max_inflight", None))
    if name == "roll":
        return RollPolicy(max_inflight=cfg.get("max_inflight", None))
    if name == "slime":
        return SlimePolicy(
            update_weights_interval=cfg.get("update_weights_interval", 1),
            over_sampling_ratio=cfg.get("over_sampling_ratio", 1.0),
            max_inflight=cfg.get("max_inflight", None),
        )
    if name == "verl":
        configured_threshold = cfg.get("staleness_threshold", None)
        return VerlPolicy(
            staleness_threshold=float(
                max_staleness if configured_threshold is None else configured_threshold
            ),
            require_batches=cfg.get("require_batches", 1),
            max_inflight=cfg.get("max_inflight", None),
        )
    raise ValueError(f"Unknown policy: {name!r}.")


def build_distribution(cfg: DictConfig) -> list[int]:
    """Generate a list of lengths from a single distribution config."""
    name = cfg.get("name", "uniform")
    n = cfg.get("n", 200)
    seed = cfg.get("seed", 42)
    if name == "uniform":
        return uniform(n, cfg.get("lo", 50), cfg.get("hi", 200), seed=seed)
    if name == "bimodal":
        return bimodal(
            n,
            cfg.get("short_len", 50),
            cfg.get("long_len", 500),
            cfg.get("long_frac", 0.25),
            seed=seed,
        )
    if name == "lognormal":
        return lognormal(
            n,
            cfg.get("mu", 4.0),
            cfg.get("sigma", 1.0),
            lo=cfg.get("lo", 1),
            hi=cfg.get("hi", None),
            seed=seed,
        )
    if name == "normal":
        return normal(
            n,
            cfg.get("mu", 512.0),
            cfg.get("sigma", 128.0),
            lo=cfg.get("lo", 1),
            hi=cfg.get("hi", None),
            seed=seed,
        )
    if name == "powerlaw":
        return powerlaw(
            n, cfg.get("alpha", 2.0), cfg.get("lo", 10), cfg.get("hi", 1000), seed=seed
        )
    if name == "fixed":
        return fixed(n, cfg.get("lo", 512), seed=seed)
    raise ValueError(f"Unknown workload distribution: {name!r}.")


def build_workload(cfg: DictConfig) -> GroupedWorkload:
    """Build the simulator's workload source from config."""
    source = cfg.get("source", "distributions")
    if source == "distributions":
        return GroupedWorkload.from_lengths(
            build_distribution(cfg.prompt),
            build_distribution(cfg.output),
        )
    if source == "trace":
        trace = cfg.trace
        path = trace.get("path", "")
        if not path:
            raise ValueError("workload.trace.path is required when workload.source is trace.")
        return GroupedWorkload.from_trace(path, n_groups=trace.get("n_groups", 0))
    raise ValueError(f"Unknown workload source: {source!r}.")


def build_training_cost(cfg: DictConfig) -> TrainingCostModel:
    """Instantiate composable training latency and memory models."""
    name = cfg.get("name", "fixed")
    if name == "fixed":
        return FixedTrainingCost(
            cfg.get("latency_s", 1.0),
            cfg.get("peak_memory_bytes", 0),
        )
    if name == "analytical":
        return AnalyticalTrainingCost(
            model_params=cfg.get("model_params", 0),
            peak_flops=cfg.get("peak_flops", 312e12),
            memory_bandwidth=cfg.get("memory_bandwidth", 2e12),
            overhead_factor=cfg.get("overhead_factor", 1.1),
            bytes_per_parameter=cfg.get("bytes_per_parameter", 16.0),
        )
    if name == "linear":
        return LinearTrainingCost(
            cfg.get("time_per_token", 1e-4),
            cfg.get("base_memory_bytes", 0),
        )
    model_type = TRAINING_MODEL_REGISTRY.get(name)
    if model_type is not None:
        artifact_path = cfg.get("artifact_path", "")
        if not artifact_path:
            raise ValueError(f"training_cost.artifact_path is required for {name}.")
        selectors = {
            "parameter_dtype": cfg.get("parameter_dtype", "bf16"),
            "gradient_dtype": cfg.get("gradient_dtype", "fp32"),
            "optimizer_state_dtype": cfg.get("optimizer_state_dtype", "fp32"),
            "extrapolate": list(cfg.get("extrapolate", [])) or None,
        }
        # Batching is an FSDP-only concept: profiling measures uniform shapes, and only
        # the FSDP model owns a planner that turns real trajectory lengths into
        # microbatches. Passing these to a model without a planner is a TypeError, so the
        # arguments are selected by what the model actually accepts rather than sent to
        # every registry entry.
        parameters = inspect.signature(model_type.from_artifact).parameters
        # FSDP2 shards optimizer state unconditionally, so it has no
        # `distributed_optimizer` selector at all and rejects one as unknown.
        if "distributed_optimizer" in parameters:
            selectors["distributed_optimizer"] = cfg.get("distributed_optimizer", True)
        if "batching_mode" in parameters:
            selectors["batching_mode"] = cfg.get("batching_mode", "pad_to_max")
        if "max_tokens_per_rank" in parameters:
            selectors["max_tokens_per_rank"] = cfg.get("max_tokens_per_rank", None)
        # An FSDP cell is a measurement of ONE mesh, so a multi-mesh artifact cannot infer
        # which one to answer for. These reach `from_artifact` through its **selectors, so
        # they are invisible to the signature check above and are passed by name.
        for field in ("shard_size", "replicate_size", "optimizer_offload", "reshard_after_forward"):
            value = cfg.get(field, None)
            if value is not None and "selectors" in parameters:
                selectors[field] = value
        return model_type.from_artifact(artifact_path, **selectors)
    raise ValueError(f"Unknown training cost model: {name!r}.")


def build_training_parallelism(cfg: DictConfig) -> TrainingParallelism:
    """Build the simulator's explicit training topology."""
    value = cfg.get("parallelism", {})
    return TrainingParallelism(
        tp=value.get("tp", 1),
        pp=value.get("pp", 1),
        dp=value.get("dp", 1),
        cp=value.get("cp", 1),
    )


def build_sync_cost(cfg: DictConfig) -> SyncCostModel:
    """Instantiate a sync cost model from config."""
    name = cfg.get("name", "fixed")
    if name == "fixed":
        return FixedSyncCost(sync_time=cfg.get("sync_time", 0.0))
    if name == "bandwidth":
        return BandwidthSyncCost()
    raise ValueError(f"Unknown sync cost model: {name!r}.")


def build_recompute_cost(
    cfg: DictConfig,
    training_cost: TrainingCostModel,
) -> RecomputeCostModel:
    """Instantiate a recompute cost model from config.

    Args:
        cfg: The `recompute_cost` config block.
        training_cost: The model pricing the optimizer step. The
            `training_forward` model prices the log-prob pass as a fraction of
            it, so the two cannot disagree about topology or sequence lengths.
    """
    name = cfg.get("name", "fixed")
    if name == "fixed":
        return FixedRecomputeCost(time_per_token=cfg.get("time_per_token", 0.0))
    if name == "analytical":
        return AnalyticalRecomputeCost(
            model_params=cfg.get("model_params", 0),
            peak_flops=cfg.get("peak_flops", 312e12),
            overhead_factor=cfg.get("overhead_factor", 1.1),
        )
    if name == "training_forward":
        for required in ("hidden_size", "vocab_size", "parameter_count"):
            if not cfg.get(required, 0):
                raise ValueError(
                    f"recompute_cost.{required} is required for the training_forward model, "
                    f"which corrects the forward share for the unembedding."
                )
        return TrainingForwardRecomputeCost(
            training_cost=training_cost,
            hidden_size=int(cfg.hidden_size),
            vocab_size=int(cfg.vocab_size),
            parameter_count=int(cfg.parameter_count),
        )
    raise ValueError(f"Unknown recompute cost model: {name!r}.")


def _length_cap(value) -> int | None:
    if value is None:
        return None
    return int(value)


def resolve_kv_blocks(rollout: DictConfig) -> int:
    """Return the KV capacity in blocks, computed from the model when configured.

    `rollout.kv_blocks` is a literal placeholder by default. Naming a
    `kv_cache_sizing.model_path` derives the capacity the way vLLM would, from
    the model's KV geometry and the memory budget, which is what lets a
    configuration be simulated before anyone has run it.
    """
    sizing = rollout.get("kv_cache_sizing", None)
    if sizing is None:
        return int(rollout.kv_blocks)

    model_path = sizing.get("model_path", None)
    if not model_path:
        return int(rollout.kv_blocks)

    geometry = KvCacheGeometry.from_hf_config(model_path)
    parameter_count = sizing.get("parameter_count", None)
    if not parameter_count:
        raise ValueError(
            "rollout.kv_cache_sizing.model_path needs parameter_count as well, since the weight "
            "footprint cannot be derived from the architecture fields alone."
        )
    return kv_blocks_from_memory(
        geometry,
        gpu_memory_mib=int(sizing.get("gpu_memory_mib", 97_871)),
        gpu_memory_utilization=float(sizing.get("gpu_memory_utilization", 0.9)),
        parameter_count=int(parameter_count),
        block_size=int(rollout.block_size),
        tp_size=int(sizing.get("tp_size", 1)),
        non_kv_overhead_bytes=int(sizing.get("non_kv_overhead_bytes", 0)),
    )


def run_simulate(cfg: DictConfig) -> None:
    """Run a simulation from a resolved Hydra config."""
    job = cfg.job
    rollout = cfg.rollout
    training = cfg.training_cost
    training_parallelism = build_training_parallelism(training)
    admit = rollout.admission_control
    log_telemetry = cfg.get("log_telemetry", True)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=cfg.get("log_level", "INFO"), log_file=output_dir / "run.log")

    memory_mib = training.get("gpu_memory_mib", None)
    workload = build_workload(cfg.workload)
    sim_cfg = SimConfig(
        batch_size=job.batch_size,
        max_staleness=job.max_staleness,
        n_versions=job.n_versions,
        sync_time=cfg.sync_cost.get("sync_time", 0.0),

        training_tp=training_parallelism.tp,
        training_pp=training_parallelism.pp,
        training_dp=training_parallelism.dp,
        training_cp=training_parallelism.cp,
        training_micro_batch_size=training.get("micro_batch_size", 1),
        # None keeps count-based grouping. Set it to the real trainer's
        # `ppo_max_token_len_per_gpu * sp_size` to reproduce dynamic batching.
        training_max_tokens_per_microbatch=training.get("max_tokens_per_microbatch"),
        training_schedule=training.get("schedule", "1f1b"),
        activation_recompute=training.get("activation_recompute", "none"),
        training_optimizer=training.get("optimizer", "adam"),
        training_gpu_memory_bytes=None if memory_mib is None else mib_to_bytes(int(memory_mib)),
        partial_rollout=rollout.partial_rollout,
        token_budget=rollout.token_budget,
        kv_blocks=resolve_kv_blocks(rollout),
        block_size=rollout.block_size,
        max_concurrency=rollout.get("max_concurrency", None),
        reject_if_kv_full=admit.get("reject_if_kv_full", True),
        reject_if_waiting=admit.get("reject_if_waiting", False),
        reject_if_running_full=admit.get("reject_if_running_full", False),
        n_instances=rollout.n_instances,
        workload=workload,
        max_prompt_length=_length_cap(job.get("max_prompt_length", None)),
        max_output_length=_length_cap(job.get("max_output_length", None)),
        livelock_rounds=rollout.get("livelock_rounds", 50),
        max_engine_iters=rollout.get("max_engine_iters", 10_000),
        log_telemetry=log_telemetry,
    )

    rollout_cost = build_rollout_cost(cfg.rollout_cost)
    policy = build_policy(cfg.policy, job.max_staleness)
    training_cost = build_training_cost(cfg.training_cost)
    sync_cost = build_sync_cost(cfg.sync_cost)
    recompute_cost = build_recompute_cost(cfg.recompute_cost, training_cost)

    dryrun_logger.info(
        f"Starting simulation: policy={policy.name}, n_versions={sim_cfg.n_versions}, "
        f"n_instances={sim_cfg.n_instances}, batch_size={sim_cfg.batch_size}, "
        f"max_prompt_length={sim_cfg.max_prompt_length!r}, "
        f"max_output_length={sim_cfg.max_output_length!r}, "
        f"log_telemetry={log_telemetry}."
    )

    # Only create the logs directory / JSONL files at all when telemetry
    # logging is switched on; `log_telemetry=False` must not touch disk.
    store = JsonlStore(output_dir / "logs", overwrite=True) if log_telemetry else None

    sim = Simulator(
        rollout_cost=rollout_cost,
        policy=policy,
        cfg=sim_cfg,
        training_cost=training_cost,
        sync_cost=sync_cost,
        recompute_cost=recompute_cost,
        store=store,
    )
    result = sim.run()

    print(f"Simulation complete: {result.versions_done} versions in {result.sim_time:.2f}s")
    print(f"  Consumed: {result.n_consumed} requests")
    print(f"  Dropped: {len(result.dropped)} requests")
    print(f"  Throughput: {result.throughput:.1f} tokens/s")
    if result.consumed:
        print(f"  Max staleness (true): {result.max_staleness_true}")
        staleness_counts = Counter(result.staleness_true_values)
        print("  Staleness distribution:")
        for staleness, count in sorted(staleness_counts.items()):
            percentage = count / result.n_consumed * 100
            print(f"    {staleness}: {count} requests ({percentage:.2f}%)")
    if result.livelocked:
        print("  WARNING: Simulation livelocked!")

    if store is not None:
        plotter = SimPlotter(store, output_dir / "plots")
        paths = plotter.plot_all()
        print(f"\n  Logs saved to: {output_dir / 'logs'}")
        print(f"  Plots saved to: {output_dir / 'plots'}")
        for p in paths:
            print(f"    {p.name}")
        store.close()
    else:
        dryrun_logger.info("log_telemetry=False: skipped writing logs and plots.")


_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "defaults")


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="simulate")
def main(cfg: DictConfig) -> None:
    """DryRun-RL simulation entrypoint."""
    run_simulate(cfg)


if __name__ == "__main__":
    main()
