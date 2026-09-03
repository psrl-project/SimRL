"""Hydra structured configs for DryRun-RL."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParallelismConfig:
    """Explicit TP, PP, DP, and CP dimensions."""

    tp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1


@dataclass
class RolloutCostConfig:
    """Rollout inference cost model configuration."""

    name: str = "roofline"
    F: float = 0.005
    W: float = 0.001
    G: float = 0.0001
    A_p: float = 0.0
    A_d: float = 1e-7
    b: int = 16
    artifact_path: str = ""
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    h: int = 4096
    m: int = 11008
    C1: float = 0.0
    C2: float = 0.0
    C3: float = 0.0
    C4: float = 0.0
    C5: float = 0.0
    # ComposedRollout: predict from measured kernel tables instead of fitted coefficients.
    kernel_table_root: str = ""
    gpu: str = ""
    model_path: str = ""
    attention_backend: str = ""
    utility_surface: str = "pchip"
    decode_integration_stride: int = 1024
    # Per-request KV depth above which a decode segment uses the O(1) closed form instead of
    # the lattice integrator. Defaults to effectively disabled: measured end to end on the
    # DAPO trace the fast path gave 1.2x for 4.6% deviation, which is not a good trade.
    saturation_step_threshold: int = 1 << 62


@dataclass
class PolicyConfig:
    """Staleness policy configuration."""

    name: str = "psrl"
    proactive_filter: bool = False
    proactive_threshold: int = 0
    max_inflight: int | None = None
    update_weights_interval: int = 1
    over_sampling_ratio: float = 1.0
    staleness_threshold: float | None = None
    require_batches: int = 1


@dataclass
class DistributionConfig:
    """Length distribution configuration for one dimension (prompt or output).

    `lo` / `hi` describe the distribution's own support where that family needs
    them (for example uniform or powerlaw). Per-request caps belong on `JobConfig`.
    """

    name: str = "uniform"
    n: int = 200
    seed: int = 42
    # Kept because dryrun/config/defaults/workload/from_trace.yaml still sets them and
    # build_distribution still reads them. Retiring the from_trace workload is a separate
    # change that must drop the YAML group, the CLI branch, and these together.
    trace_path: str = ""
    trace_column: str = "length"
    lo: int = 50
    hi: int = 200
    short_len: int = 50
    long_len: int = 500
    long_frac: float = 0.25
    mu: float = 4.0
    sigma: float = 1.0
    alpha: float = 2.0


@dataclass
class JointTraceConfig:
    """Joint group trace source (see `dryrun-workload collect-groups`)."""

    path: str = ""
    n_groups: int = 0


@dataclass
class WorkloadConfig:
    """Workload source: synthetic marginal distributions or a joint group trace."""

    source: str = "distributions"
    output: DistributionConfig = field(default_factory=DistributionConfig)
    prompt: DistributionConfig = field(
        default_factory=lambda: DistributionConfig(name="fixed", lo=512, hi=512)
    )
    trace: JointTraceConfig = field(default_factory=JointTraceConfig)


@dataclass
class TrainingCostConfig:
    """Training latency, memory, and runtime parallelism configuration."""

    name: str = "fixed"
    artifact_path: str = ""
    latency_s: float = 1.0
    peak_memory_bytes: int = 0
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    micro_batch_size: int = 1
    schedule: str = "1f1b"
    activation_recompute: str = "none"
    optimizer: str = "adam"
    parameter_dtype: str = "bf16"
    gradient_dtype: str = "fp32"
    optimizer_state_dtype: str = "fp32"
    distributed_optimizer: bool = True
    # Feature names for which out-of-domain queries are allowed. Empty means strict.
    # Valid names: FSDP latency: num_microbatches, sequences_per_microbatch, tokens
    #              FSDP memory:  num_microbatches, activation_tokens
    #              Megatron latency: micro_batch_size, pipeline_slots, tokens
    #              Megatron memory:  activation_bytes, profiled_dp
    extrapolate: list = field(default_factory=list)
    gpu_memory_mib: int | None = None
    model_params: int = 0
    peak_flops: float = 312e12
    memory_bandwidth: float = 2e12
    overhead_factor: float = 1.1
    bytes_per_parameter: float = 16.0
    time_per_token: float = 1e-4
    base_memory_bytes: int = 0
    # How real trajectory lengths become microbatches. Profiling measures uniform
    # shapes, so this is a simulate-time choice, and the three modes span a measured
    # 1.17x to 4.34x latency range on DAPO-like bimodal lengths.
    batching_mode: str = "pad_to_max"
    # Token budget per rank for the token_budget mode. verl's default is 10240.
    max_tokens_per_rank: int | None = None
    # Which profiled FSDP mesh to answer for. A cell is a measurement of one mesh, so an
    # artifact spanning several shard degrees cannot infer the choice; null lets a
    # single-mesh artifact infer it. parallelism.dp must equal shard * replicate.
    shard_size: int | None = None
    replicate_size: int | None = None
    # Discrete FSDP cell selectors. Must match a profiled cell.
    optimizer_offload: bool = False
    reshard_after_forward: bool = True


@dataclass
class SyncCostConfig:
    """Weight sync cost configuration."""

    name: str = "fixed"
    sync_time: float = 0.0


@dataclass
class RecomputeCostConfig:
    """Recompute cost configuration."""

    name: str = "fixed"
    time_per_token: float = 0.0
    model_params: int = 0
    peak_flops: float = 312e12
    overhead_factor: float = 1.1


@dataclass
class JobConfig:
    """Basic job settings for one simulation run."""

    n_versions: int = 20
    batch_size: int = 8
    max_staleness: int = 2
    # Per-request caps applied after sampling. None means no truncation.
    max_prompt_length: int | None = None
    max_output_length: int | None = None


@dataclass
class AdmissionControlConfig:
    """Gates applied when a request is offered to an instance."""

    reject_if_kv_full: bool = True
    reject_if_waiting: bool = False
    reject_if_running_full: bool = False


@dataclass
class RolloutConfig:
    """Rollout engine knobs."""

    n_instances: int = 1
    partial_rollout: bool = True
    token_budget: int = 8192
    kv_blocks: int = 100_000
    block_size: int = 16
    max_concurrency: int | None = None
    admission_control: AdmissionControlConfig = field(default_factory=AdmissionControlConfig)
    livelock_rounds: int = 50
    max_engine_iters: int = 10_000


@dataclass
class SimulateConfig:
    """Top-level simulation configuration."""

    rollout_cost: RolloutCostConfig = field(default_factory=RolloutCostConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    training_cost: TrainingCostConfig = field(default_factory=TrainingCostConfig)
    sync_cost: SyncCostConfig = field(default_factory=SyncCostConfig)
    recompute_cost: RecomputeCostConfig = field(default_factory=RecomputeCostConfig)
    job: JobConfig = field(default_factory=JobConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    output_dir: str = "outputs"
    log_telemetry: bool = True
    log_level: str = "INFO"
