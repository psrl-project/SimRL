"""Idle time must be attributed to a real request, or declared unexplained."""

from __future__ import annotations

import pytest
from diagnosis_fixtures import BLOCK_SIZE, _cost_model, run_mixed, run_single_request

from srl.diagnosis.critical_path import decompose, steps
from srl.diagnosis.pricer import SoloPricer
from srl.diagnosis.timeline import request_phases


def test_every_step_names_a_binding_request_from_its_own_batch(tmp_path):
    store = run_mixed(tmp_path)
    timelines, _ = request_phases(store)
    found = steps(store, timelines)
    assert found, "No training steps were recorded."
    for step in found:
        assert step.binding_rid is not None, f"version {step.version} has no binding request."
        assert timelines[step.binding_rid].group_id in step.group_ids
        assert step.binding_complete_time == pytest.approx(
            max(timelines[rid].complete_time for rid in timelines if timelines[rid].group_id in step.group_ids)
        )


def test_the_binding_request_is_the_last_of_the_batch_to_finish(tmp_path):
    store = run_mixed(tmp_path)
    timelines, _ = request_phases(store)
    for step in steps(store, timelines):
        members = [line for line in timelines.values() if line.group_id in step.group_ids]
        latest = max(members, key=lambda line: line.complete_time)
        assert step.binding_rid == latest.rid


def test_a_wait_not_ending_at_a_completion_is_declared_unexplained(tmp_path):
    store = run_mixed(tmp_path)
    timelines, _ = request_phases(store)
    for step in steps(store, timelines):
        assert step.wait_cause in ("completion", "no_wait", "unexplained")
        if step.wait_cause == "completion":
            assert step.binding_complete_time == pytest.approx(step.start, abs=1e-6)


def test_a_lone_request_has_no_interference(tmp_path):
    store = run_single_request(tmp_path)
    timelines, _ = request_phases(store)
    pricer = SoloPricer({1: _cost_model()}, BLOCK_SIZE)
    line = next(iter(timelines.values()))
    parts = decompose(line, pricer, {0: 1})
    assert parts.queue_seconds == pytest.approx(0.0, abs=1e-9)
    assert parts.reprefill_seconds == pytest.approx(0.0, abs=1e-9)
    assert parts.interference_seconds == pytest.approx(0.0, abs=0.05 * parts.observed_seconds)
    assert parts.solo_seconds > 0.0


def test_decomposition_components_sum_to_the_observed_lifetime(tmp_path):
    store = run_mixed(tmp_path)
    timelines, _ = request_phases(store)
    pricer = SoloPricer({1: _cost_model()}, BLOCK_SIZE)
    for line in timelines.values():
        parts = decompose(line, pricer, {0: 1, 1: 1})
        total = (
            parts.solo_seconds
            + parts.interference_seconds
            + parts.queue_seconds
            + parts.reprefill_seconds
            + parts.weight_load_seconds
        )
        assert total == pytest.approx(parts.observed_seconds, rel=1e-6, abs=1e-6)
