"""Fleet state inside an idle window says whether the fleet was starved or merely waiting."""

from __future__ import annotations

from diagnosis_fixtures import run_mixed

from srl.diagnosis.critical_path import steps
from srl.diagnosis.fleet import idle_windows, instance_imbalance
from srl.diagnosis.timeline import request_phases


def test_busy_fraction_is_a_fraction_per_instance(tmp_path):
    store = run_mixed(tmp_path)
    timelines, _ = request_phases(store)
    found = steps(store, timelines)
    windows = idle_windows(store, found, timelines, staleness_horizon=1)
    assert windows, "The fixture produced no idle window."
    for window in windows:
        assert window.t1 >= window.t0, f"Window ends before it starts: {window!r}."
        for instance_id, fraction in window.busy_fraction.items():
            assert 0.0 <= fraction <= 1.0 + 1e-9, f"instance {instance_id} busy {fraction}."


def test_a_window_reports_whether_its_binding_group_is_reserved_at_the_limit(tmp_path):
    store = run_mixed(tmp_path)
    timelines, _ = request_phases(store)
    windows = idle_windows(store, steps(store, timelines), timelines, staleness_horizon=1)
    informative = [w for w in windows if w.binding_group_id is not None]
    assert informative, "No window named a binding group."
    at_limit_values = {w.binding_at_staleness_limit for w in informative}
    assert at_limit_values & {True, False}, (
        f"No window produced a real boolean, only saw {at_limit_values!r}."
    )
    for window in informative:
        assert window.binding_at_staleness_limit in (True, False, None), (
            f"Unexpected binding_at_staleness_limit: {window.binding_at_staleness_limit!r}."
        )
        if window.binding_reserved_in_buffer is not None:
            assert window.binding_reserved_in_buffer >= 0, (
                f"Negative buffer id: {window.binding_reserved_in_buffer!r}."
            )


def test_instance_imbalance_covers_every_instance_that_ran(tmp_path):
    store = run_mixed(tmp_path)
    rows = instance_imbalance(store)
    assert {row.instance_id for row in rows} == {0}, f"Unexpected instance ids: {rows!r}."
    for row in rows:
        assert row.busy_seconds > 0.0, f"instance {row.instance_id!r} reports zero busy seconds."
        assert row.tokens > 0, f"instance {row.instance_id!r} reports zero tokens."
        assert row.last_active >= 0.0, f"instance {row.instance_id!r} reports a negative last_active."


def test_busy_seconds_never_exceed_the_run_span(tmp_path):
    store = run_mixed(tmp_path)
    timelines, _ = request_phases(store)
    span = max(step.end for step in steps(store, timelines))
    for row in instance_imbalance(store):
        assert row.busy_seconds <= span + 1e-6, (
            f"instance {row.instance_id!r} busy_seconds {row.busy_seconds!r} exceeds span {span!r}."
        )
