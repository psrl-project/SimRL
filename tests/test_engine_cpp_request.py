"""Verify CppRequest matches Python Request semantics."""

import pytest

try:
    from dryrun._engine_cpp import CppRequest, ReqStatus, EventKind, CppEngineEvent
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ engine not compiled")


def test_request_fields_and_defaults():
    r = CppRequest(rid=1, prompt_len=100, target_len=200, v_traj=0)
    assert r.rid == 1
    assert r.prompt_len == 100
    assert r.target_len == 200
    assert r.generated == 0
    assert r.prefilled == 0
    assert r.ctx == 100
    assert r.remaining == 200
    assert r.status == ReqStatus.QUEUED
    assert r.group_id == 0
    assert r.instance_id is None


def test_add_tokens_creates_segments():
    r = CppRequest(rid=1, prompt_len=100, target_len=200, v_traj=0)
    r.add_tokens(0, 10)
    assert r.generated == 10
    assert r.ctx == 110
    assert r.remaining == 190
    assert len(r.segments) == 1
    assert r.segments[0].version == 0
    assert r.segments[0].tokens == 10


def test_add_tokens_extends_same_version():
    r = CppRequest(rid=1, prompt_len=100, target_len=200, v_traj=0)
    r.add_tokens(0, 10)
    r.add_tokens(0, 5)
    assert r.generated == 15
    assert len(r.segments) == 1
    assert r.segments[0].tokens == 15


def test_add_tokens_new_version_new_segment():
    r = CppRequest(rid=1, prompt_len=100, target_len=200, v_traj=0)
    r.add_tokens(0, 10)
    r.add_tokens(1, 5)
    assert len(r.segments) == 2
    assert r.v_min == 0
    assert r.v_max == 1


def test_add_tokens_zero_is_noop():
    r = CppRequest(rid=1, prompt_len=100, target_len=200, v_traj=0)
    r.add_tokens(0, 0)
    assert r.generated == 0
    assert len(r.segments) == 0


def test_v_min_v_max_empty_segments():
    r = CppRequest(rid=1, prompt_len=100, target_len=200, v_traj=3)
    assert r.v_min == 3
    assert r.v_max == 3


def test_event_kind_enum():
    assert EventKind.ITERATION_DONE != EventKind.REQUEST_COMPLETE
    assert EventKind.PREEMPT != EventKind.ITERATION_DONE


def test_engine_event_creation():
    e = CppEngineEvent()
    e.kind = EventKind.REQUEST_COMPLETE
    e.instance_id = 0
    e.t = 1.5
    assert e.kind == EventKind.REQUEST_COMPLETE
    assert e.t == 1.5
