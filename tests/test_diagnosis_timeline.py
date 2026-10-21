"""Phase reconstruction must agree with the streams it did not read."""

from __future__ import annotations

import pytest
from diagnosis_fixtures import run_mixed, run_single_request

from srl.diagnosis.timeline import request_phases


def test_a_lone_request_has_no_queueing_and_no_interference(tmp_path):
    store = run_single_request(tmp_path)
    timelines, checks = request_phases(store)
    assert all(check.ok for check in checks), [c for c in checks if not c.ok]
    assert len(timelines) == 1
    line = next(iter(timelines.values()))
    assert line.seconds_in("router_queue") == pytest.approx(0.0, abs=1e-9)
    assert line.seconds_in("engine_wait") == pytest.approx(0.0, abs=1e-9)
    assert line.seconds_in("decode") > 0.0
    assert line.seconds_in("prefill") > 0.0
    assert line.seconds_in("reprefill") == pytest.approx(0.0, abs=1e-9)


def test_reconstructed_decode_tokens_match_the_request_stream(tmp_path):
    store = run_mixed(tmp_path)
    timelines, checks = request_phases(store)
    by_name = {check.name: check for check in checks}
    assert by_name["decode_tokens_match"].ok, by_name["decode_tokens_match"].detail
    assert by_name["completion_time_match"].ok, by_name["completion_time_match"].detail
    assert by_name["abort_count_match"].ok, by_name["abort_count_match"].detail


def test_phases_tile_each_request_lifetime_without_gaps_or_overlap(tmp_path):
    store = run_mixed(tmp_path)
    timelines, _ = request_phases(store)
    for line in timelines.values():
        covered = sum(phase.duration for phase in line.phases)
        lifetime = line.complete_time - line.admit_time
        assert covered == pytest.approx(lifetime, rel=1e-9, abs=1e-6), (
            f"rid {line.rid}: phases cover {covered} of {lifetime}."
        )
        ordered = sorted(line.phases, key=lambda p: p.t0)
        for earlier, later in zip(ordered, ordered[1:]):
            assert earlier.t1 == pytest.approx(later.t0, abs=1e-9)


def test_a_long_request_among_shorts_shows_reprefill_after_a_publish(tmp_path):
    store = run_mixed(tmp_path)
    timelines, _ = request_phases(store)
    aborted = [line for line in timelines.values() if line.num_aborts > 0]
    assert aborted, "The fixture must abort at least one request at a publish."
    for line in aborted:
        assert line.seconds_in("reprefill") > 0.0
        assert max(phase.attempt for phase in line.phases) >= 1


def test_a_preempted_request_reconstructs_without_losing_time(tmp_path):
    """A resident preempted mid-decode must still reconstruct a sane timeline.

    `_engine_phases` conservatively credits 0 decoded tokens when a resident
    vanishes from its instance's stream at a time other than its known
    `complete_time`, rather than guessing. This is the only fixture that
    preempts a request, so it is the only place that guard runs.
    """
    store = run_mixed(tmp_path)
    timelines, checks = request_phases(store)
    by_name = {check.name: check for check in checks}
    preempted = [line for line in timelines.values() if line.num_preemptions > 0]
    assert preempted, "The fixture must preempt at least one request."
    for line in preempted:
        covered = sum(phase.duration for phase in line.phases)
        lifetime = line.complete_time - line.admit_time
        assert covered == pytest.approx(lifetime, rel=1e-9, abs=1e-6), (
            f"rid {line.rid!r}: phases cover {covered!r} of {lifetime!r}."
        )
        ordered = sorted(line.phases, key=lambda p: p.t0)
        for earlier, later in zip(ordered, ordered[1:]):
            assert earlier.t1 == pytest.approx(later.t0, abs=1e-9), (
                f"rid {line.rid!r}: a gap or overlap separates two phases."
            )
        last_end = max(phase.t1 for phase in line.phases)
        assert last_end == pytest.approx(line.complete_time, abs=1e-6), (
            f"rid {line.rid!r}: reconstructed end {last_end!r} misses complete_time {line.complete_time!r}."
        )
        max_attempt = max(phase.attempt for phase in line.phases)
        assert max_attempt <= line.num_aborts + line.num_preemptions, (
            f"rid {line.rid!r}: {max_attempt!r} attempts exceeds {line.num_aborts!r} aborts "
            f"plus {line.num_preemptions!r} preemptions."
        )
        if line.reprefill_tokens > 0:
            assert line.seconds_in("reprefill") > 0.0, (
                f"rid {line.rid!r}: charged reprefill tokens but shows no reprefill phase."
            )
    for name in ("decode_tokens_match", "completion_time_match", "abort_count_match", "reprefill_present"):
        assert by_name[name].ok, by_name[name].detail
