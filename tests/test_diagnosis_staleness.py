"""The staleness reduction must account for every generated token exactly once."""

from __future__ import annotations

import pytest

from srl.diagnosis.staleness import (
    StalenessBreakdown,
    check_same_consumption,
    staleness_breakdown,
)
from srl.telemetry.store import JsonlStore

# Three consumed trajectories at two staleness levels, two discards with distinct
# reasons. Span is the largest `t`, which the last discard carries.
ROWS = [
    dict(event="consume", t=10.0, staleness_true=0, generated=100, reprefill_tokens=10),
    dict(event="consume", t=20.0, staleness_true=0, generated=200, reprefill_tokens=20),
    dict(event="consume", t=30.0, staleness_true=2, generated=300, reprefill_tokens=30),
    dict(event="drop", t=40.0, drop_reason="l3_stale", generated=50),
    dict(event="drop", t=50.0, drop_reason="l2_cancel", generated=25),
]


def _store(tmp_path, rows, name="logs"):
    """Write `rows` to a fresh `request` stream and hand back a reader."""
    root = tmp_path / name
    store = JsonlStore(root, overwrite=True)
    for row in rows:
        store.write("request", **row)
    store.close()
    return JsonlStore(root)


def test_consumed_tokens_bucket_by_true_staleness(tmp_path):
    bd = staleness_breakdown(_store(tmp_path, ROWS))
    assert bd.consumed_tokens == {0: 300, 2: 300}, f"Got {bd.consumed_tokens!r}."
    assert bd.n_consumed == 3, f"Got {bd.n_consumed!r}."
    assert bd.max_staleness == 2, f"Got {bd.max_staleness!r}."


def test_wasted_tokens_bucket_by_drop_reason(tmp_path):
    bd = staleness_breakdown(_store(tmp_path, ROWS))
    assert bd.wasted_tokens == {"l2_cancel": 25, "l3_stale": 50}, f"Got {bd.wasted_tokens!r}."
    assert bd.total_wasted_tokens == 75, f"Got {bd.total_wasted_tokens!r}."


def test_span_is_the_largest_timestamp(tmp_path):
    bd = staleness_breakdown(_store(tmp_path, ROWS))
    assert bd.span == 50.0, f"Got {bd.span!r}."


def test_throughput_counts_consumed_and_wasted(tmp_path):
    bd = staleness_breakdown(_store(tmp_path, ROWS))
    assert bd.total_consumed_tokens == 600, f"Got {bd.total_consumed_tokens!r}."
    assert bd.throughput == pytest.approx(675.0 / 50.0), f"Got {bd.throughput!r}."


def test_scg_is_cumulative_and_saturates(tmp_path):
    bd = staleness_breakdown(_store(tmp_path, ROWS))
    assert bd.scg(0) == pytest.approx(300.0 / 50.0), f"Got {bd.scg(0)!r}."
    assert bd.scg(1) == pytest.approx(300.0 / 50.0), f"Got {bd.scg(1)!r}."
    assert bd.scg(2) == pytest.approx(600.0 / 50.0), f"Got {bd.scg(2)!r}."
    assert bd.scg(9) == pytest.approx(bd.scg(2)), "SCG must not grow past the highest staleness."


def test_reprefill_tokens_are_reported_separately(tmp_path):
    bd = staleness_breakdown(_store(tmp_path, ROWS))
    assert bd.reprefill_tokens == 60, f"Got {bd.reprefill_tokens!r}."
    assert bd.total_consumed_tokens == 600, "Re-prefill must not be folded into consumed tokens."


def test_a_drop_without_a_reason_is_named_rather_than_dropped(tmp_path):
    rows = [ROWS[0], dict(event="drop", t=9.0, generated=7)]
    bd = staleness_breakdown(_store(tmp_path, rows))
    assert bd.wasted_tokens == {"unknown": 7}, f"Got {bd.wasted_tokens!r}."


def test_a_run_with_no_consumed_trajectories_is_refused(tmp_path):
    rows = [dict(event="drop", t=5.0, drop_reason="l3_stale", generated=10)]
    with pytest.raises(ValueError, match="no consumed trajectories"):
        staleness_breakdown(_store(tmp_path, rows))


def test_a_non_positive_span_is_refused(tmp_path):
    rows = [dict(event="consume", t=0.0, staleness_true=0, generated=100)]
    with pytest.raises(ValueError, match="non-positive span"):
        staleness_breakdown(_store(tmp_path, rows))


def test_a_training_tail_past_the_last_request_row_extends_the_span(tmp_path):
    """The clock can force-align to a training deadline with no request row at that instant."""
    root = tmp_path / "tail_logs"
    store = JsonlStore(root, overwrite=True)
    store.write("request", event="consume", t=10.0, staleness_true=0, generated=100)
    store.write("train", t=10.0, start=5.0, end=50.0)
    store.close()
    bd = staleness_breakdown(JsonlStore(root))
    assert bd.span == 50.0, f"Got {bd.span!r}."


def _bd(n_consumed):
    return StalenessBreakdown(
        span=10.0,
        n_consumed=n_consumed,
        consumed_tokens={0: 10},
        wasted_tokens={},
        reprefill_tokens=0,
    )


def test_the_gate_accepts_an_equal_consumption_set():
    check_same_consumption([("a", _bd(64)), ("b", _bd(64))])


def test_the_gate_refuses_an_unequal_consumption_set():
    with pytest.raises(ValueError, match="consumed different trajectory counts"):
        check_same_consumption([("a", _bd(64)), ("b", _bd(32))])
