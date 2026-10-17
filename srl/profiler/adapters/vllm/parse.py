"""Parsing of vLLM scheduler iteration logs into rollout payloads."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...schema import ProfileValidationError, RolloutPayload

_OUTPUT_FILE_PATTERN = re.compile(r"Output file:\s*(?P<path>.+?/run=\d+\.json)")
_RESULT_DIR_PATTERN = re.compile(r"result_dir=['\"](?P<path>[^'\"]+)['\"]")
_RESULT_FILENAME_PATTERN = re.compile(r"result_filename=['\"](?P<name>[^'\"]+)['\"]")
_ITERATION_PATTERN = re.compile(
    r"Iteration\((?P<index>\d+)\): "
    r"(?P<ctx_requests>\d+) context requests, "
    r"(?P<ctx_tokens>\d+) context tokens, "
    r"(?P<gen_requests>\d+) generation requests, "
    r"(?P<gen_tokens>\d+) generation tokens, "
    r"iteration elapsed time: (?P<elapsed_ms>[0-9.]+) ms"
)
_MEASUREMENT_START_MARKERS = (
    "Starting main benchmark run...",
    "Traffic request rate:",
    "Traffic ramp-up strategy:",
    "Starting profiler...",
    "Profiler started",
)
_MEASUREMENT_STOP_MARKERS = (
    "Stopping profiler",
    "Profiler stopped",
    "[END BENCHMARK]",
    "============ Serving Benchmark Result",
)
# No forward pass of a multi-billion-parameter model finishes this fast, so a
# shorter step is an async-scheduling overlap artifact, not a measurement.
_MIN_MEASURED_ELAPSED_MS = 1.0

# The same claim, made per token rather than per step. The absolute floor above is
# necessary but not sufficient: it only catches a step that is short in wall-clock terms,
# while an overlap artifact on a *large* prefill lands just above it and still reports an
# impossible rate. Measured on Qwen3.5-35B-A3B at tp=1, two steps survived collection at
# 16128 tokens in 1.080 ms (14.9M tok/s) and 11317 tokens in 1.030 ms (11.0M tok/s); the
# psrl fit then charged them a 424 ms prediction against a 1 ms "measurement", a 39178%
# residual that dragged tp=1 prefill MRE to 813%.
#
# 1M tok/s is far above anything an H20 reaches (a real 16k-token prefill measures ~400 ms,
# i.e. 40k tok/s) so this rejects artifacts without touching a single genuine step: the
# Qwen3-8B and DeepSeek-14B sweeps have zero steps above it. It is deliberately not tuned
# close to the real ceiling, because the quantity being rejected is off by two orders of
# magnitude and a loose bound cannot discard data that matters.
_MAX_MEASURED_TOKENS_PER_SECOND = 1_000_000.0


def _is_unmeasured(detail: _IterationDetail) -> bool:
    """
    Report whether a step's elapsed time is an overlap artifact rather than a measurement.

    Args:
        detail (_IterationDetail): One scheduler iteration.

    Returns:
        bool: True when the step is too short in absolute terms, or implies a
            physically impossible token rate.
    """
    if detail.elapsed_ms < _MIN_MEASURED_ELAPSED_MS:
        return True
    tokens = detail.ctx_tokens + detail.gen_tokens
    if tokens <= 0:
        return False
    return tokens / (detail.elapsed_ms / 1000.0) > _MAX_MEASURED_TOKENS_PER_SECOND


@dataclass(frozen=True)
class _IterationDetail:
    index: int
    ctx_requests: int
    ctx_tokens: int
    gen_requests: int
    gen_tokens: int
    elapsed_ms: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "index": self.index,
            "ctx_requests": self.ctx_requests,
            "ctx_tokens": self.ctx_tokens,
            "gen_requests": self.gen_requests,
            "gen_tokens": self.gen_tokens,
            "elapsed_ms": self.elapsed_ms,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> _IterationDetail:
        return cls(
            index=int(value["index"]),
            ctx_requests=int(value["ctx_requests"]),
            ctx_tokens=int(value["ctx_tokens"]),
            gen_requests=int(value["gen_requests"]),
            gen_tokens=int(value["gen_tokens"]),
            elapsed_ms=float(value["elapsed_ms"]),
        )


def _result_path_from_line(line: str) -> Path | None:
    output_match = _OUTPUT_FILE_PATTERN.search(line)
    if output_match is not None:
        return Path(output_match.group("path")).resolve()
    directory_match = _RESULT_DIR_PATTERN.search(line)
    filename_match = _RESULT_FILENAME_PATTERN.search(line)
    if directory_match is None or filename_match is None:
        return None
    return (Path(directory_match.group("path")) / filename_match.group("name")).resolve()


def parse_sweep_log(path: Path) -> dict[Path, list[_IterationDetail]]:
    """Parse a sweep log into per-result-file scheduler iteration timings."""
    if not path.exists():
        raise ProfileValidationError(
            f"Missing vLLM sweep log {path}; scheduler iteration timing is required."
        )
    observations: dict[Path, list[_IterationDetail]] = {}
    current_result: Path | None = None
    capturing = False
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            for line in raw_line.replace("\r", "\n").split("\n"):
                if not line:
                    continue
                result_path = _result_path_from_line(line)
                if result_path is not None:
                    current_result = result_path
                    observations.setdefault(current_result, [])
                    capturing = False
                    continue
                if any(marker in line for marker in _MEASUREMENT_START_MARKERS):
                    capturing = True
                if any(marker in line for marker in _MEASUREMENT_STOP_MARKERS):
                    capturing = False
                    continue
                iteration_match = _ITERATION_PATTERN.search(line)
                if (
                    capturing
                    and current_result is not None
                    and iteration_match is not None
                ):
                    values = iteration_match.groupdict()
                    observations[current_result].append(
                        _IterationDetail(
                            index=int(values["index"]),
                            ctx_requests=int(values["ctx_requests"]),
                            ctx_tokens=int(values["ctx_tokens"]),
                            gen_requests=int(values["gen_requests"]),
                            gen_tokens=int(values["gen_tokens"]),
                            elapsed_ms=float(values["elapsed_ms"]),
                        )
                    )
    return observations


@dataclass(frozen=True)
class TrustReport:
    """What `trusted_iterations` kept and why it stopped trusting a run."""

    kept: int
    dropped_unmeasured: int
    dropped_after_preemption: int
    truncation_iteration: int | None
    truncation_reason: str | None


def trusted_iterations(
    details: Sequence[_IterationDetail],
    *,
    input_len: int,
    prompt_token_budget: int,
) -> tuple[list[_IterationDetail], TrustReport]:
    """Keep only steps that equal-length synthetic formulas can describe.

    Two effects make a run untrustworthy:

    - **Preemption.** vLLM preempts by recompute once the KV cache fills. An
      evicted request re-enters the waiting queue and later re-prefills its
      prompt *plus* the tokens it already generated. After the first such
      resume the running set mixes request ages, so `ctxsum = R*(L+1+k)` no
      longer describes any later step and the rest of the run is dropped.
    - **Unmeasured steps.** Async scheduling overlaps a step with its
      neighbours, so some steps report a sub-millisecond elapsed time that is
      a pipelining artifact rather than a forward pass. Those steps are
      dropped individually; the surrounding steps remain valid.

    Args:
        details: Scheduler steps of one benchmark run, in iteration order.
        input_len: Prompt length shared by every request in the run.
        prompt_token_budget: Total prompt tokens the run may ever schedule,
            i.e. the sum of all request prompt lengths. Context beyond this is
            recompute of already-generated tokens.
    """
    if input_len <= 0:
        raise ProfileValidationError("input_len must be positive.")
    if prompt_token_budget <= 0:
        raise ProfileValidationError("prompt_token_budget must be positive.")
    kept: list[_IterationDetail] = []
    dropped_unmeasured = 0
    scheduled_prompt_tokens = 0
    for step_index, detail in enumerate(details):
        reason: str | None = None
        if detail.ctx_tokens > detail.ctx_requests * input_len:
            reason = "context-longer-than-prompt"
        elif scheduled_prompt_tokens + detail.ctx_tokens > prompt_token_budget:
            reason = "context-beyond-prompt-token-budget"
        elif detail.gen_tokens != detail.gen_requests:
            reason = "multiple-generation-tokens-per-request"
        if reason is not None:
            return kept, TrustReport(
                kept=len(kept),
                dropped_unmeasured=dropped_unmeasured,
                dropped_after_preemption=len(details) - step_index,
                truncation_iteration=detail.index,
                truncation_reason=reason,
            )
        scheduled_prompt_tokens += detail.ctx_tokens
        if _is_unmeasured(detail):
            dropped_unmeasured += 1
            continue
        kept.append(detail)
    return kept, TrustReport(
        kept=len(kept),
        dropped_unmeasured=dropped_unmeasured,
        dropped_after_preemption=0,
        truncation_iteration=None,
        truncation_reason=None,
    )


def _balanced_chunk_sum_l2(requests: int, tokens: int) -> float:
    """Sum of squared chunk lengths under as-equal-as-possible packing.

    When ``tokens`` divides ``requests``, this is ``R * (T/R)^2`` (same as
    ``T^2/R``). Otherwise the remainder is spread one token per request, which
    matches how a token budget is split across concurrent prefills.
    """
    if requests <= 0:
        return 0.0
    base, rem = divmod(tokens, requests)
    return float(rem * (base + 1) ** 2 + (requests - rem) * base * base)


def payload_from_iteration(
    detail: _IterationDetail,
    *,
    input_len: int,
    decode_ordinal: int,
    batch_step_index: int,
) -> RolloutPayload | None:
    """Fill attention features from an equal-length synthetic scheduler step.

    Callers must filter the run through `trusted_iterations` first; this
    function assumes every request shares `input_len` and `decode_ordinal`.
    """
    if input_len <= 0:
        raise ProfileValidationError("input_len must be positive.")
    if decode_ordinal < 0:
        raise ProfileValidationError("decode_ordinal must not be negative.")
    has_context = detail.ctx_requests > 0
    has_generation = detail.gen_requests > 0
    if not has_context and not has_generation:
        return None
    if has_context and has_generation:
        phase = "mixed"
    elif has_context:
        phase = "prefill"
    else:
        phase = "decode"
    sum_l2 = (
        _balanced_chunk_sum_l2(detail.ctx_requests, detail.ctx_tokens)
        if has_context
        else 0.0
    )
    ctxsum = (
        float(detail.gen_requests * (input_len + 1 + decode_ordinal))
        if has_generation
        else 0.0
    )
    n_tokens = detail.ctx_tokens + detail.gen_tokens
    request_count = detail.ctx_requests + detail.gen_requests
    if n_tokens <= 0 or request_count <= 0:
        return None
    return RolloutPayload(
        phase=phase,
        request_count=request_count,
        input_tokens=detail.ctx_tokens,
        output_tokens=detail.gen_tokens,
        n_tokens=n_tokens,
        sum_l=float(detail.ctx_tokens) + ctxsum,
        sum_l2=sum_l2,
        ctxsum=ctxsum,
        latency_s=detail.elapsed_ms / 1000.0,
        batch_step_index=batch_step_index,
    )
