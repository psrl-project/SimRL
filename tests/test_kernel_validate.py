"""Bucketing a real baseline and scoring a model against it."""

from __future__ import annotations

import argparse
import contextlib
import json

import pytest

from srl.cost.rollout import RolloutWorkload
from srl.cost.rollout.base import RolloutCostModel


class _Exact(RolloutCostModel):
    """A model that returns each bucket's own measured latency, so error must be zero."""

    def __init__(self, latencies: dict[tuple[int, int], float]) -> None:
        self._latencies = latencies

    def predict(self, workload: RolloutWorkload) -> float:
        return self._latencies[(workload.request_count, int(workload.n_tokens))]

    def decode_coeffs(self, n_d, ctxsum):
        return 0.0, 1.0e-3, 0.0


class _Biased(RolloutCostModel):
    """A model that is uniformly 10% slow, so MRE must be exactly 10."""

    def __init__(self, latencies) -> None:
        self._latencies = latencies

    def predict(self, workload: RolloutWorkload) -> float:
        return 1.1 * self._latencies[(workload.request_count, int(workload.n_tokens))]

    def decode_coeffs(self, n_d, ctxsum):
        return 0.0, 1.0e-3, 0.0


def _write_baseline(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({"payload": record}) + "\n")
    return path


def _decode(request_count, ctxsum, latency_s):
    return {
        "phase": "decode",
        "request_count": request_count,
        "n_tokens": request_count,
        "sum_l2": 0,
        "ctxsum": ctxsum,
        "latency_s": latency_s,
    }


def test_repeated_iterations_collapse_into_one_bucket(tmp_path):
    from srl.cost.kernel.validate import load_baseline

    path = _write_baseline(
        tmp_path / "vllm.jsonl",
        [_decode(16, 16 * 2048, 0.030 + 0.001 * i) for i in range(5)],
    )
    buckets = load_baseline(path)
    assert len(buckets) == 1
    assert buckets[0].sample_count == 5
    # The median, not the mean: a single slow outlier must not move the bucket.
    assert buckets[0].latency_s == pytest.approx(0.032)


def test_the_context_bucket_does_not_snap_to_a_power_of_two(tmp_path):
    """Snapping 7300 to 4096 once manufactured a 19.1% error that did not exist."""
    from srl.cost.kernel.validate import load_baseline

    path = _write_baseline(
        tmp_path / "vllm.jsonl", [_decode(16, 16 * 7300, 0.05) for _ in range(3)]
    )
    bucket = load_baseline(path)[0]
    assert bucket.ctxsum / bucket.request_count == pytest.approx(7296, abs=128)


def test_iterations_at_different_contexts_stay_in_different_buckets(tmp_path):
    from srl.cost.kernel.validate import load_baseline

    path = _write_baseline(
        tmp_path / "vllm.jsonl",
        [_decode(16, 16 * 1024, 0.02) for _ in range(3)]
        + [_decode(16, 16 * 8192, 0.06) for _ in range(3)],
    )
    assert len(load_baseline(path)) == 2


def test_thin_buckets_are_dropped(tmp_path):
    from srl.cost.kernel.validate import load_baseline

    path = _write_baseline(
        tmp_path / "vllm.jsonl",
        [_decode(16, 16 * 1024, 0.02) for _ in range(10)] + [_decode(1, 512, 0.01)],
    )
    buckets = load_baseline(path, min_samples=5)
    assert len(buckets) == 1
    assert buckets[0].request_count == 16


def test_prefill_and_decode_are_scored_separately(tmp_path):
    from srl.cost.kernel.validate import load_baseline, score_model

    records = [_decode(16, 16 * 2048, 0.030) for _ in range(3)]
    records += [
        {
            "phase": "prefill",
            "request_count": 4,
            "n_tokens": 4096,
            "sum_l2": 4 * 1024**2,
            "ctxsum": 0,
            "latency_s": 0.400,
        }
        for _ in range(3)
    ]
    buckets = load_baseline(_write_baseline(tmp_path / "vllm.jsonl", records))
    latencies = {(b.request_count, int(b.n_tokens)): b.latency_s for b in buckets}
    results = {result.phase: result for result in score_model(_Exact(latencies), buckets)}
    assert set(results) == {"decode", "prefill"}
    assert results["decode"].bucket_count == 1
    assert results["prefill"].bucket_count == 1


def test_an_exact_model_scores_zero_error(tmp_path):
    from srl.cost.kernel.validate import load_baseline, score_model

    path = _write_baseline(
        tmp_path / "vllm.jsonl",
        [_decode(count, count * 2048, 0.002 * count) for count in (8, 16, 32) for _ in range(3)],
    )
    buckets = load_baseline(path)
    latencies = {(b.request_count, int(b.n_tokens)): b.latency_s for b in buckets}
    (result,) = score_model(_Exact(latencies), buckets)
    assert result.mre_percent == pytest.approx(0.0, abs=1e-9)
    assert result.median_ratio == pytest.approx(1.0, rel=1e-9)
    assert result.passes(1.0)


def test_a_uniformly_biased_model_scores_its_bias(tmp_path):
    from srl.cost.kernel.validate import load_baseline, score_model

    path = _write_baseline(
        tmp_path / "vllm.jsonl",
        [_decode(count, count * 2048, 0.002 * count) for count in (8, 16, 32) for _ in range(3)],
    )
    buckets = load_baseline(path)
    latencies = {(b.request_count, int(b.n_tokens)): b.latency_s for b in buckets}
    (result,) = score_model(_Biased(latencies), buckets)
    assert result.mre_percent == pytest.approx(10.0, rel=1e-6)
    assert result.median_ratio == pytest.approx(1.1, rel=1e-9)
    assert not result.passes(5.0)


def test_a_mixed_iteration_is_labelled_mixed(tmp_path):
    """sum_l2 > 0 and ctxsum > 0 together is the case with no measured attention row."""
    from srl.cost.kernel.validate import load_baseline

    path = _write_baseline(
        tmp_path / "vllm.jsonl",
        [
            {
                "phase": "prefill",
                "request_count": 16,
                "n_tokens": 15361,
                "sum_l2": 16 * 960**2,
                "ctxsum": 16 * 512,
                "latency_s": 1.5745,
            }
            for _ in range(3)
        ],
    )
    assert load_baseline(path)[0].phase == "mixed"


def test_an_empty_baseline_yields_no_results(tmp_path):
    from srl.cost.kernel.validate import load_baseline, score_model

    path = _write_baseline(tmp_path / "vllm.jsonl", [])
    assert load_baseline(path) == ()
    assert score_model(_Exact({}), ()) == ()


def test_a_non_positive_latency_record_is_skipped(tmp_path):
    from srl.cost.kernel.validate import load_baseline

    path = _write_baseline(
        tmp_path / "vllm.jsonl",
        [_decode(16, 16 * 2048, 0.03) for _ in range(3)] + [_decode(16, 16 * 2048, 0.0)],
    )
    assert load_baseline(path)[0].sample_count == 3


def test_a_phase_dropped_by_min_samples_is_reported_not_silent(tmp_path):
    """A report naming only decode must not read as "prefill was never validated"."""
    import logging
    import os
    from pathlib import Path

    from srl._logging import srl_logger
    from srl.cli.profile import cmd_kernels_validate

    # `cmd_kernels_validate` resolves the config's checkpoint before it reaches the
    # coverage warning under test, and the example config reads that checkpoint from
    # `MODEL_PATH`. Without a local one the config load raises first and the assertions
    # below would never be exercised, so skip rather than pass vacuously.
    model_path = os.environ.get("MODEL_PATH", "")
    if not model_path or not (Path(model_path) / "config.json").is_file():
        pytest.skip("MODEL_PATH is not set to a local checkpoint")

    records = [_decode(16, 16 * 2048, 0.030) for _ in range(25)]
    records += [
        {
            "phase": "prefill",
            "request_count": 4,
            "n_tokens": 4096,
            "sum_l2": 4 * 1024**2,
            "ctxsum": 0,
            "latency_s": 0.400,
        }
        for _ in range(3)
    ]
    baseline = _write_baseline(tmp_path / "vllm.jsonl", records)

    # `srl_logger` sets propagate = False, so caplog cannot see it. Attach a handler.
    captured: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    srl_logger.addHandler(handler)
    # Only the coverage warning is under test. Let the later model construction fail.
    args = argparse.Namespace(
        # A real config: cmd_kernels_validate loads it before reaching the coverage warning,
        # so a bogus path would abort first and the test would pass vacuously.
        config="examples/profile/kernel/qwen3_8b_h20_tp1.yaml",
        baseline=str(baseline),
        model_path=None,
        surface="pchip",
        min_samples=20,
        gate=100.0,
        output=None,
    )
    try:
        with contextlib.suppress(Exception):
            cmd_kernels_validate(args)
    finally:
        srl_logger.removeHandler(handler)

    assert any("prefill" in message and "min-samples" in message for message in captured), captured
    assert not any("decode" in message for message in captured)
