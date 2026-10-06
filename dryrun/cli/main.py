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


# TP degrees with profiled GEMM and attention tables. kernel_tp4 and kernel_tp8 exist
# but carry collective tables only, so a composed model built at those degrees would
# query GEMM and attention shapes that were never measured.
_PROFILED_TP_DEGREES = frozenset({1, 2})


def _cohort_table_roots(
    heterogeneous: DictConfig,
    tp_candidates: tuple[int, ...],
    rollout_cost_name: str,
) -> dict[int, str]:
    """Return the per-TP kernel table root for each candidate TP degree.

    Only the composed cost model reads kernel tables; every other model is fitted and
    ignores them, so an empty mapping is valid there and this returns `{}`.

    Args:
        heterogeneous: The `heterogeneous` config subtree.
        tp_candidates: TP degrees the solver may allocate.
        rollout_cost_name: `rollout_cost.name`, which decides whether tables are needed.

    Returns:
        TP degree -> table root. Empty when the cost model does not read tables.

    Raises:
        ValueError: If the composed model is in use and any candidate TP has no entry.
            Falling back to the single `rollout_cost.kernel_table_root` is deliberately
            not done: that path is correct for at most one TP degree, and the others
            would fail later with a KernelCoverageError naming a shape rather than the
            missing configuration.
    """
    if rollout_cost_name != "composed":
        return {}
    # Keys arrive as ints from a dataclass default but as strings through a Hydra
    # override (`kernel_table_roots.2=...`), so normalise both.
    configured = {
        int(tp): str(root)
        for tp, root in dict(heterogeneous.get("kernel_table_roots", {}) or {}).items()
    }
    missing = [tp for tp in tp_candidates if tp not in configured]
    if missing:
        raise ValueError(
            f"heterogeneous.kernel_table_roots is missing an entry for TP "
            f"{sorted(missing)}. The composed cost model reads measured GEMM and "
            f"attention tables that are sharded per TP degree, so each candidate needs "
            f"its own root (e.g. 1: .../kernel_tp1/kernel, 2: .../kernel_tp2/kernel). "
            f"Configured: {sorted(configured)}."
        )
    return configured


def build_cohort_models(
    cfg: DictConfig,
) -> "tuple[dict[int, RolloutCostModel], dict[int, int], WaterFillingSolver]":
    """Build one cost model and KV capacity per TP candidate, plus the solver.

    Returns `(cost_models, kv_blocks_per_tp, solver)`. `Simulator` builds the fleet
    itself from these, because the router the fleet rebinds is the one `Simulator`
    constructs.

    Args:
        cfg: The full resolved Hydra config, containing `heterogeneous`, `rollout_cost`,
            and `rollout` sub-configs.

    Returns:
        A tuple of (cost_models, kv_blocks_per_tp, solver) ready to pass to `Simulator`.

    Raises:
        ValueError: If the router strategy is not `cohort_affinity`, since any other
            strategy ignores `cohort_id` and silently defeats the fleet split.
    """
    from omegaconf import OmegaConf

    from ..simulator.water_filling import WaterFillingSolver

    h = cfg.heterogeneous
    strategy_name = cfg.rollout.router.get("strategy", "request_num_balance")
    if strategy_name != "cohort_affinity":
        raise ValueError(
            f"heterogeneous.enabled=true requires rollout.router.strategy=cohort_affinity, "
            f"got {strategy_name!r}. Any other strategy ignores cohort_id, so old-version "
            f"requests would land on low-TP instances and the fleet split would measure "
            f"nothing."
        )

    tp_candidates = tuple(int(tp) for tp in h.tp_candidates)
    if not tp_candidates:
        raise ValueError("heterogeneous.tp_candidates must not be empty.")
    # Only TP 1 and 2 have profiled GEMM and attention tables. tp4/tp8 have collective
    # tables only, so a model built at those degrees would query shapes that were never
    # measured. Refusing here names the configuration; letting it through surfaces a
    # KernelCoverageError about a GEMM shape much further from the cause.
    if unsupported := sorted(set(tp_candidates) - _PROFILED_TP_DEGREES):
        raise ValueError(
            f"heterogeneous.tp_candidates contains unprofiled TP degrees "
            f"{unsupported}. Only {sorted(_PROFILED_TP_DEGREES)} have measured GEMM and "
            f"attention tables; tp4/tp8 carry collective tables only. Profile the "
            f"missing families to widen this set."
        )

    # One cost model per TP. `rollout_cost.parallelism.tp` is overridden per candidate:
    # ComposedRollout shards its ModelOps at construction, so a TP=2 cohort needs its
    # own model built at tp=2 rather than the tp=1 one with a scale factor.
    #
    # The composed model also needs a per-TP *table* root. Measured GEMM and attention
    # tables are sharded by TP degree, so one path cannot serve every cohort: a tp=2
    # model queries `(n, k)` slices and `num_kv_heads` values that only the tp=2 sweep
    # contains. Reading the wrong table raises KernelCoverageError rather than returning
    # a plausible wrong number, so the failure is loud -- but it has to be configured.
    cost_models: dict[int, RolloutCostModel] = {}
    base = OmegaConf.to_container(cfg.rollout_cost, resolve=True)
    table_roots = _cohort_table_roots(h, tp_candidates, str(base.get("name", "roofline")))
    for tp in tp_candidates:
        per_tp = OmegaConf.create(base)
        per_tp.parallelism.tp = tp
        if tp in table_roots:
            per_tp.kernel_table_root = table_roots[tp]
        cost_models[tp] = build_rollout_cost(per_tp)

    # KV capacity per TP. Sharding the weights across more ranks frees memory for
    # cache, so this genuinely differs per TP (31294 blocks at tp=1 against 69532 at
    # tp=2 for Qwen3-8B on an H20) and must not be collapsed to one number.
    kv_blocks_per_tp: dict[int, int] = {}
    for tp in tp_candidates:
        per_tp_rollout = OmegaConf.create(OmegaConf.to_container(cfg.rollout, resolve=True))
        if per_tp_rollout.get("kv_cache_sizing", None) is not None:
            per_tp_rollout.kv_cache_sizing.tp_size = tp
        kv_blocks_per_tp[tp] = resolve_kv_blocks(per_tp_rollout)

    solver = WaterFillingSolver(
        cost_models=cost_models,
        n_gpus=int(h.n_gpus),
        tp_candidates=tp_candidates,
        strategy=str(h.strategy),
        fixed_cohorts=[dict(c) for c in h.cohorts],
        max_concurrency=cfg.rollout.get("max_concurrency", None),
    )
    return cost_models, kv_blocks_per_tp, solver


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
    router = rollout.get("router", {})
    log_telemetry = cfg.get("log_telemetry", True)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=cfg.get("log_level", "INFO"), log_file=output_dir / "run.log")

    memory_mib = training.get("gpu_memory_mib", None)
    workload = build_workload(cfg.workload)
    # None adopts the workload's own arity; an explicit value is checked against
    # it by SimConfig, so a run against the wrong trace fails loudly.
    group_size = job.group_size if job.group_size is not None else workload.group_size
    sim_cfg = SimConfig(
        batch_size=job.batch_size,
        group_size=group_size,
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
        enable_prefix_caching=rollout.get("enable_prefix_caching", True),
        max_concurrency=rollout.get("max_concurrency", None),
        reject_if_kv_full=admit.get("reject_if_kv_full", True),
        reject_if_waiting=admit.get("reject_if_waiting", False),
        reject_if_running_full=admit.get("reject_if_running_full", False),
        routing_strategy=router.get("strategy", "request_num_balance"),
        routing_strategy_args=dict(router.get("strategy_args", {}) or {}),
        router_sort_key=router.get("sort_key", "short_length"),
        max_concurrent_seqs_per_instance=router.get("max_concurrent_seqs_per_instance", 0),
        router_reject_on_waiting=router.get("reject_on_waiting", False),
        enable_group_sticky=router.get("enable_group_sticky", False),
        max_waiting_after_preempt=router.get("max_waiting_after_preempt", None),
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

    cohort_cost_models = None
    kv_blocks_per_tp = None
    solver = None
    if cfg.get("heterogeneous", None) is not None and cfg.heterogeneous.enabled:
        cohort_cost_models, kv_blocks_per_tp, solver = build_cohort_models(cfg)

    dryrun_logger.info(
        f"Starting simulation: policy={policy.name}, n_versions={sim_cfg.n_versions}, "
        f"n_instances={sim_cfg.n_instances}, batch_size={sim_cfg.batch_size} groups "
        f"x group_size={sim_cfg.group_size} = {sim_cfg.responses_per_step} responses/step, "
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
        cohort_cost_models=cohort_cost_models,
        kv_blocks_per_tp=kv_blocks_per_tp,
        solver=solver,
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
