"""Simulation configuration and result dataclasses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..component.rollout.engine import Request
from ..workload.grouped import GroupedWorkload


def clip_length(length: int, max_length: int | None) -> int:
    """Truncate `length` to `max_length` when a system cap is set."""
    if max_length is None:
        return length
    if max_length < 1:
        raise ValueError(f"Expected max_length >= 1, got {max_length!r}.")
    return min(length, max_length)


def clip_total_length(
    prompt_len: int,
    output_len: int,
    max_prompt_length: int | None,
    max_output_length: int | None,
) -> tuple[int, int]:
    """Cap a `(prompt, output)` pair the way the serving stack really caps it.

    The prompt is capped on its own, because truncation happens before the
    request is ever dispatched. The output is then capped on the **total**:

        output <= (max_prompt_length + max_output_length) - prompt_len

    which is what PSRL computes (`vllm_async_server.py`, the `max_tokens`
    fallback `min(response_length, response_length + prompt_length - len(prompt_ids))`).
    A short prompt therefore buys extra output budget, and the measured run
    shows it: 242 of 398 never-interrupted requests generated more than the
    10240 response cap, every one of them landing exactly on
    `11264 - prompt_len` (e.g. prompt 90 -> 11174 tokens).

    Capping the two dimensions independently understates a short-prompt
    request by up to `max_prompt_length` tokens of generation. On the DAPO
    trace, whose prompts are 90-200 tokens against a 1024 cap, that is ~900
    tokens per request, which made the simulated first batch finish at 342.6 s
    against a measured 386.6 s.

    Args:
        prompt_len: Sampled prompt length.
        output_len: Sampled output length.
        max_prompt_length: Prompt cap. None means uncapped.
        max_output_length: Output cap, and with the prompt cap it sets the
            total budget. None means uncapped.

    Returns:
        The capped `(prompt_len, output_len)` pair.

    TODO(lhy): Enrich what the trace records and make the sampling assumption
    selectable. A trace currently stores one `(prompt_len, output_lens)` tuple
    per group, which is the total-length distribution and nothing more. An
    agentic or interrupted trajectory is really a *sequence* of segments, so
    worth recording: per-segment lengths, the summed response length, and the
    turn count. Then expose switches at simulate time for which distribution to
    draw from: total length (today's behaviour), per-segment, or summed
    response, so the effect of each assumption can be measured instead of
    being baked in here.
    """
    prompt = clip_length(prompt_len, max_prompt_length)
    if max_output_length is None:
        return prompt, output_len
    if max_prompt_length is None:
        return prompt, clip_length(output_len, max_output_length)
    # The budget is the full context the stack is willing to hold, so a prompt
    # shorter than its cap leaves the remainder available for generation.
    budget = max_prompt_length + max_output_length - prompt
    # At least one token, so a prompt pinned at its cap still generates.
    return prompt, max(1, min(output_len, budget))


@dataclass
class SimConfig:
    """Configuration for a single simulation run."""

    batch_size: int = 8
    """Prompt groups consumed per training step.

    Responses per step is `batch_size * group_size`. GRPO scores a response against
    its own group, so a group is the indivisible unit of a training batch.
    """
    group_size: int = 1
    """Responses sampled per prompt, the recipe's `rollout_n`.

    Must equal `workload.group_size`. Only a trace carries real groups; the
    synthetic distributions produce one response per prompt.
    """
    max_staleness: int = 1
    n_versions: int = 20
    sync_time: float = 0.0
    training_tp: int = 1
    training_pp: int = 1
    training_dp: int = 1
    training_cp: int = 1
    training_micro_batch_size: int = 1
    training_max_tokens_per_microbatch: int | None = None
    """Token budget per training microbatch; None groups by `training_micro_batch_size`.

    Pass the already-multiplied budget: verl computes
    `max_token_len = ppo_max_token_len_per_gpu * sp_size`, and `sp_size` equals TP
    under Megatron sequence parallelism. See `TrainingWorkload`.
    """
    training_schedule: str = "1f1b"
    activation_recompute: str = "none"
    training_optimizer: str = "adam"
    training_gpu_memory_bytes: int | None = None
    partial_rollout: bool = True
    token_budget: int = 8192
    kv_blocks: int = 100_000
    block_size: int = 16
    enable_prefix_caching: bool = True
    max_concurrent: int | None = None
    max_concurrency: int | None = None
    reject_if_kv_full: bool = False
    reject_if_waiting: bool = False
    reject_if_running_full: bool = False
    routing_strategy: str = "request_num_balance"
    """Which instance a queued request is sent to.

    The default reproduces the former least-active selection exactly, so it is
    the behaviour-preserving choice as well as the one the matched PSRL run
    uses. See `component/rollout/router/strategies/`.
    """
    routing_strategy_args: dict = field(default_factory=dict)
    """Per-strategy keyword arguments, e.g. `cache_threshold` for prefix_aware."""
    router_sort_key: str = "small_id"
    """Tie-break among same-version requests: small_id, short_length, long_length.

    Arrival order by default: a length key makes placement depend on how much
    of the dataset is staged, which couples the result to an unrelated flow
    limit.
    """
    max_concurrent_seqs_per_instance: int = 0
    """Router-side cap on requests one instance may hold. 0 disables it."""
    router_reject_on_waiting: bool = False
    """Whether the router skips an instance that already has its own queue."""
    router_reject_preempted_on_waiting: bool = False
    """Whether preemption loopbacks skip instances with a local queue."""
    enable_group_sticky: bool = False
    """Pin a prompt group's responses to the instance that took the first one."""
    weight_load_time: float = 0.0
    """Rollout-side weight-load latency after a training step publishes.

    The real system pauses rollout while vLLM executes `pull_model` and
    `reset_prefix_cache`. Measured at 4-5 s on the matched PSRL run
    (the gap between train-end and the first nonzero `num_running_reqs`).
    This is separate from `sync_time`, which covers the NIXL transfer
    to the training nodes.
    """
    max_waiting_after_preempt: int | None = None
    """Instance queue depth past which preempted requests return to the router.

    vLLM's `preemption_notification_threshold`: a preempted request is requeued
    locally first, and only handed back once the local queue is congested,
    because requeueing into a congested instance makes it worse. None disables
    the path, leaving every preemption local.
    """
    workload: GroupedWorkload = field(
        default_factory=lambda: GroupedWorkload.from_lengths([512], [100])
    )
    max_prompt_length: int | None = None
    max_output_length: int | None = None
    n_instances: int = 1
    livelock_rounds: int = 50
    max_engine_iters: int = 10_000
    undoable_events: frozenset[str] = frozenset(
        {"request_complete", "preempt", "iteration_done"}
    )
    """Event types that get speculative undo in the engine loop.

    Default: all three → full speculation (current behavior, exact).
    Fast mode: ``{"request_complete"}`` → only completions are speculative;
    PREEMPT and ITERATION_DONE are committed without snapshots, eliminating
    the dominant undo bookkeeping cost. ``request_complete`` must always be
    present.
    """
    log_telemetry: bool = True
    """Whether to log rollout/train/request telemetry via `SimTelemetry`.

    When False, no JSONL files are written even if a `store` is passed to
    `Simulator`; this takes priority over `store`.
    """

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size counts prompt groups and must be >= 1, got {self.batch_size}.")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}.")
        # Agreement is required rather than derived, so a directly constructed
        # SimConfig cannot silently disagree with its own workload.
        if self.group_size != self.workload.group_size:
            raise ValueError(
                f"group_size {self.group_size} does not match the workload's "
                f"{self.workload.group_size}."
            )
        if "request_complete" not in self.undoable_events:
            raise ValueError(
                "undoable_events must contain 'request_complete': completions "
                "determine batch collection order and must be speculative."
            )

    @property
    def responses_per_step(self) -> int:
        """Responses in one training batch."""
        return self.batch_size * self.group_size


@dataclass
class SimResult:
    """Simulation output."""

    consumed: list[Request] = field(default_factory=list)
    dropped: list[Request] = field(default_factory=list)
    sim_time: float = 0.0
    versions_done: int = 0
    livelocked: bool = False
    train_timestamps: list[tuple[float, float]] = field(default_factory=list)
    train_peak_memory_bytes: list[int] = field(default_factory=list)

    @property
    def n_consumed(self) -> int:
        return len(self.consumed)

    @property
    def staleness_true_values(self) -> list[int]:
        return [r.staleness_true() for r in self.consumed]

    @property
    def staleness_reported_values(self) -> list[int]:
        return [r.staleness_reported() for r in self.consumed]

    @property
    def max_staleness_true(self) -> int:
        vals = self.staleness_true_values
        return max(vals) if vals else 0

    @property
    def max_staleness_reported(self) -> int:
        vals = self.staleness_reported_values
        return max(vals) if vals else 0

    @property
    def throughput(self) -> float:
        if self.sim_time <= 0:
            return 0.0
        total_tokens = sum(r.generated for r in self.consumed)
        return total_tokens / self.sim_time

    def scg(self, s: int) -> float:
        """Tokens from trajectories consumed with staleness_true <= s, per second.

        **Why tokens rather than a trajectory count.** SCG is goodput, and the
        useful work a training step receives is tokens, not trajectories: a
        dispersed workload's trajectories differ in length by more than an order
        of magnitude (measured p99/p50 = 4.6 on the long-context trace), so a
        count weights a 700-token response the same as a 30000-token one. Tokens
        also make this directly comparable with `throughput`, which is the same
        quantity without the staleness filter -- their ratio is the fraction of
        generated work that arrived fresh enough to use.

        Uses staleness_true (v_min of segments), not staleness_reported (v_max).
        The two differ on any trajectory interrupted by a weight update, and using
        v_max would understate staleness on exactly those partial-rollout cases.

        Args:
            s: Staleness bound. A trajectory counts when every token it carries
                was generated within `s` weight versions of its consumption.

        Returns:
            Tokens per simulated second. `scg(max_staleness_true)` equals
            `throughput` by construction, since no trajectory is then excluded.
        """
        if self.sim_time <= 0.0:
            return 0.0
        tokens = sum(r.generated for r in self.consumed if r.staleness_true() <= s)
        return tokens / self.sim_time

    def scg_count(self, s: int) -> float:
        """Trajectories consumed with staleness_true <= s, per simulated second.

        The count-based form `scg` used to return. Kept because the solver's
        value model reasons about completions rather than tokens, and because a
        trajectory count is the right unit when comparing fleets on a workload
        whose lengths are uniform.
        """
        if self.sim_time <= 0.0:
            return 0.0
        count = sum(1 for r in self.consumed if r.staleness_true() <= s)
        return count / self.sim_time

    def scg_curve(self, s_values: Sequence[int] | None = None) -> dict[int, float]:
        """SCG(S) in tokens/second for each S in s_values.

        Defaults to 0..max_staleness_true inclusive, which covers the full curve
        without requiring the caller to know the range in advance.
        """
        if s_values is None:
            max_s = self.max_staleness_true
            s_values = range(max_s + 1)
        return {s: self.scg(s) for s in s_values}

    def scg_w(self, w: Callable[[int], float]) -> float:
        """SCG_w = sum_k w(staleness_true(k)) / sim_time (§4.4 of the design spec).

        w(s) = 1[s <= S] recovers the hard-constraint SCG(S).
        """
        if self.sim_time <= 0.0:
            return 0.0
        return sum(w(r.staleness_true()) for r in self.consumed) / self.sim_time
