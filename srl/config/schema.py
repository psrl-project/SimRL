"""Hydra structured configs for SimRL."""

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
    # Kept because srl/config/defaults/workload/from_trace.yaml still sets them and
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
    """Joint group trace source (see `srl-workload collect-groups`)."""

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

    # fixed | analytical | training_forward. Prefer training_forward: it prices
    # the log-prob pass as a FLOP-derived fraction of the fitted training step,
    # so it inherits that model's topology and sequence-length behaviour rather
    # than needing an achieved-FLOPs number of its own.
    name: str = "fixed"
    time_per_token: float = 0.0
    model_params: int = 0
    peak_flops: float = 312e12
    overhead_factor: float = 1.1

    # training_forward's unembedding correction needs hidden_size, vocab_size and
    # parameter_count. They are NOT configured here: they describe the model whose
    # fitted training curve this scales, so they are read from the training
    # artifact (or the run's checkpoint when the training model has no fit behind
    # it). Configuring them separately let the correction describe one
    # architecture while the curve it multiplied described another.


@dataclass
class JobConfig:
    """Basic job settings for one simulation run."""

    n_versions: int = 20
    batch_size: int = 8
    """Prompt groups consumed per training step, not responses."""
    group_size: int | None = None
    """Responses per prompt. None reads it from the workload."""
    max_staleness: int = 2
    # Per-request caps applied after sampling. None means no truncation.
    max_prompt_length: int | None = None
    max_output_length: int | None = None


@dataclass
class AdmissionControlConfig:
    """Gates applied when a request is offered to an instance."""

    # Off because KV capacity is the scheduler's constraint, not an admission
    # one. `_engine_step` already enforces it the way vLLM does, by
    # leaving a request that does not fit on the instance's waiting queue.
    reject_if_kv_full: bool = False
    reject_if_waiting: bool = False
    reject_if_running_full: bool = False


@dataclass
class KvCacheSizingConfig:
    """How to obtain `rollout.kv_blocks` instead of taking the literal value.

    `kv_blocks` is the denominator of every reported `kv_cache_usage`, so a
    placeholder rescales the whole KV curve and makes it incomparable with a
    measured one. Naming a `model_path` replaces it with the capacity a real
    vLLM engine allocates for the same deployment.
    """

    # A HuggingFace checkpoint directory. Required for either path below.
    model_path: str | None = None

    # True starts a real vLLM engine on a cache miss and reuses the cached block
    # count thereafter. False uses arithmetic over config.json, which is only
    # honest for a dense model: vLLM measures its peak-activation and non-torch
    # memory at startup, and a hybrid model's mamba page alignment cannot be
    # reconstructed from a config file.
    probe: bool = True

    # Cache key component. Two GPUs with equal memory differ in context size and
    # fragmentation, so a capacity measured on one is invalid for the other.
    gpu_name: str = "unknown"

    gpu_memory_mib: int = 97_871
    gpu_memory_utilization: float = 0.9
    tp_size: int = 1

    # `probe: false` only. Weight size normally comes from the checkpoint's
    # safetensors index. This is the fallback for a single-shard checkpoint.
    parameter_count: int | None = None
    non_kv_overhead_bytes: int = 0

    # max_model_len, max_num_seqs, max_num_batched_tokens, dtype and block_size
    # are read from job.*, rollout.* and rollout_cost.* rather than duplicated
    # here, so they cannot drift from the run they describe.


@dataclass
class RouterConfig:
    """Cross-instance routing: where a request waits, and who takes it.

    Field names follow the PSRL run's own
    `psrl.rollout_coordination.routing_strategy.*` keys wherever one exists, so
    a setting can be transcribed rather than translated.
    """

    eviction: str = "vllm_lru"
    """vllm_lru | longest_ctx.

    Which resident the engine evicts when the KV pool cannot fund a decode
    step. Separate from `strategy` because the engine makes this call, not the
    router, and an experiment should be able to move one without the other.
    """
    strategy: str = "request_num_balance"
    """request_num_balance | round_robin | random | kv_cache_aware |
    prefix_aware | throughput_balance | length_aware.

    The default reproduces the former least-active selection exactly, so it is
    behaviour-preserving as well as what the matched PSRL run uses.
    """
    strategy_args: dict = field(default_factory=dict)
    """Per-strategy keyword arguments, e.g. cache_threshold for prefix_aware."""
    sort_key: str = "small_id"
    """Tie-break among same-version queued requests: small_id | short_length | long_length.

    `small_id` is arrival order, and it is the default because it is the only
    one of the three that does not make the result depend on how much of the
    dataset happens to be staged. A length key reorders the queue, so widening
    ingress changes which request is placed: measured over staged widths of
    3, 8 and 32 batches under `short_length`, the mean prompt length of the
    first 48 admissions moved 92.5 -> 86.7 -> 85.0. The length keys remain
    available for studying that effect deliberately.
    """
    queue_window: int = 32
    """Queued requests a ranking strategy chooses among per placement.

    Only a strategy implementing `rank_queued` reads it. The default keeps the
    cost of a placement bounded when a backlog reaches several hundred requests;
    a strategy ordering the whole backlog needs it at least that wide.
    """
    max_concurrent_seqs_per_instance: int = 0
    """Router-side cap on requests one instance may hold. 0 disables the gate.

    Distinct from `rollout.max_concurrency`, which is vLLM's own max_num_seqs
    inside the engine. This one decides eligibility before the request is
    offered at all.
    """
    reject_on_waiting: bool = False
    """Whether the router skips an instance that already has a queue of its own."""
    reject_preempted_on_waiting: bool = False
    """Whether only preemption loopbacks skip instances with a local queue."""
    enable_group_sticky: bool = False
    """Pin a prompt group's responses to the instance that accepted the first.

    False matches the matched run, which scatters all 160 groups across both
    instances. Turning it on concentrates a group's KV so its siblings share
    the prompt prefix.
    """
    max_waiting_after_preempt: int | None = None
    """Instance queue depth past which preempted requests return to the router.

    vLLM's `preemption_notification_threshold`: a preempted request is requeued
    locally first and only handed back once the local queue is congested,
    because requeueing into a congested instance makes it worse. None leaves
    every preemption local, which is the behaviour before this existed.
    """


@dataclass
class RolloutConfig:
    """Rollout engine knobs."""

    n_instances: int = 1
    partial_rollout: bool = True
    token_budget: int = 8192
    kv_blocks: int = 100_000
    block_size: int = 16
    enable_prefix_caching: bool = True
    """Whether the engine reuses KV blocks across requests sharing a prefix.

    Defaults to True because real vLLM does: a default that disagrees with the
    simulated system is a silent bias. Turning it off leaves the pool doing
    block accounting and simply records no hashes, so nothing can hit; it is a
    data state rather than a second code path.
    """
    max_concurrency: int | None = None
    admission_control: AdmissionControlConfig = field(default_factory=AdmissionControlConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    livelock_rounds: int = 50
    max_engine_iters: int = 10_000
    kv_cache_sizing: KvCacheSizingConfig = field(default_factory=KvCacheSizingConfig)


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
