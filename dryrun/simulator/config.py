"""Simulation configuration and result dataclasses."""

from __future__ import annotations

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
    max_concurrent: int | None = None
    max_concurrency: int | None = None
    reject_if_kv_full: bool = True
    reject_if_waiting: bool = False
    reject_if_running_full: bool = False
    workload: GroupedWorkload = field(
        default_factory=lambda: GroupedWorkload.from_lengths([512], [100])
    )
    max_prompt_length: int | None = None
    max_output_length: int | None = None
    n_instances: int = 1
    livelock_rounds: int = 50
    max_engine_iters: int = 10_000
    log_telemetry: bool = True
    """Whether to log rollout/train/request telemetry via `SimTelemetry`.

    When False, no JSONL files are written even if a `store` is passed to
    `Simulator`; this takes priority over `store`.
    """


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
