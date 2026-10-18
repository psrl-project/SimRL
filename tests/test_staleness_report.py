"""Tests for the token-weighted staleness report."""

from __future__ import annotations

import pytest

from srl.component.rollout.engine import Request
from srl.simulator.report import summarize


def _req(n_tokens: int, v_gen: int, v_consume: int, rid: int) -> Request:
    r = Request(rid=rid, prompt_len=10, target_len=10 + n_tokens, v_traj=v_gen)
    r.add_tokens(v_gen, n_tokens)
    r.consume_version = v_consume
    return r


def test_report_is_token_weighted_and_reports_strictness() -> None:
    reqs = [_req(900, 0, 4, 0), _req(100, 3, 4, 1)]
    report = summarize(reqs, dropped=[], horizon=4)

    assert report.n_requests == 2
    assert report.n_tokens == 1000
    # 900 tokens at staleness 4, 100 tokens at staleness 1: (900*4 + 100*1) / 1000.
    assert report.mean == pytest.approx(3.7)
    assert report.max == 4
    assert report.strict is True

    report = summarize(reqs, dropped=[], horizon=3)
    assert report.strict is False


def test_percentiles_favor_the_heavier_token_mass() -> None:
    reqs = [_req(900, 0, 4, 0), _req(100, 3, 4, 1)]
    report = summarize(reqs, dropped=[], horizon=None)

    assert report.p50 == 4.0
    assert report.p95 == 4.0
    assert report.p99 == 4.0
    assert report.configured_horizon is None
    assert report.strict is None


def test_reported_matches_true_when_no_segment_was_interrupted() -> None:
    reqs = [_req(900, 0, 4, 0), _req(100, 3, 4, 1)]
    report = summarize(reqs, dropped=[], horizon=4)

    assert report.mean_reported == report.mean
    assert report.max_reported == report.max


def test_wasted_tokens_are_counted_by_reason() -> None:
    r = Request(rid=0, prompt_len=10, target_len=60, v_traj=0)
    r.add_tokens(0, 50)
    report = summarize([], dropped=[(r, "l3_stale")], horizon=2)
    assert report.wasted_tokens == {"l3_stale": 50}


def test_wasted_tokens_sum_across_multiple_drops_of_the_same_reason() -> None:
    a = Request(rid=0, prompt_len=10, target_len=60, v_traj=0)
    a.add_tokens(0, 50)
    b = Request(rid=1, prompt_len=10, target_len=40, v_traj=0)
    b.add_tokens(0, 20)
    report = summarize([], dropped=[(a, "l3_stale"), (b, "l3_stale")], horizon=2)
    assert report.wasted_tokens == {"l3_stale": 70}


def test_empty_report_has_no_strict_violation_and_zero_tokens() -> None:
    report = summarize([], dropped=[], horizon=4)
    assert report.n_requests == 0
    assert report.n_tokens == 0
    assert report.mean == 0.0
    assert report.strict is True
