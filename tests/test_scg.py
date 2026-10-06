from __future__ import annotations
from collections.abc import Callable
import pytest
from dryrun.simulator.config import SimResult
from dryrun.component.rollout.engine import Request


def _make_request(rid: int, v_min: int, v_max: int, consume_version: int) -> Request:
    """Build a minimal consumed Request with the given staleness parameters."""
    req = Request(rid=rid, prompt_len=10, target_len=20, v_traj=v_min)
    # staleness_true = consume_version - v_min; staleness_reported = consume_version - v_max
    # Simulate having generated tokens under v_min (and optionally v_max)
    req.add_tokens(v_min, 10)
    if v_max != v_min:
        req.add_tokens(v_max, 10)
    req.consume_version = consume_version
    return req


def _make_result(requests: list[Request], sim_time: float) -> SimResult:
    return SimResult(consumed=requests, sim_time=sim_time, versions_done=5)


def test_scg_all_within_budget():
    reqs = [_make_request(i, 0, 0, 1) for i in range(10)]  # staleness_true = 1
    result = _make_result(reqs, 10.0)
    assert result.scg(1) == pytest.approx(1.0)  # 10 consumed / 10 s


def test_scg_none_within_budget():
    reqs = [_make_request(i, 0, 0, 3) for i in range(10)]  # staleness_true = 3
    result = _make_result(reqs, 10.0)
    assert result.scg(2) == pytest.approx(0.0)


def test_scg_partial():
    reqs = (
        [_make_request(i, 0, 0, 1) for i in range(6)]    # staleness 1
        + [_make_request(i + 6, 0, 0, 3) for i in range(4)]  # staleness 3
    )
    result = _make_result(reqs, 10.0)
    assert result.scg(1) == pytest.approx(0.6)
    assert result.scg(3) == pytest.approx(1.0)


def test_scg_curve_default_range():
    reqs = (
        [_make_request(i, 0, 0, 1) for i in range(5)]    # staleness 1
        + [_make_request(i + 5, 0, 0, 2) for i in range(5)]  # staleness 2
    )
    result = _make_result(reqs, 10.0)
    curve = result.scg_curve()
    assert curve[0] == pytest.approx(0.0)
    assert curve[1] == pytest.approx(0.5)
    assert curve[2] == pytest.approx(1.0)


def test_scg_uses_staleness_true_not_reported():
    # v_traj=0, tokens under v=0 and v=2, consume_version=3
    # staleness_true  = 3 - 0 = 3  (uses v_min=0)
    # staleness_reported = 3 - 2 = 1  (uses v_max=2)
    req = _make_request(0, v_min=0, v_max=2, consume_version=3)
    result = _make_result([req], 1.0)
    assert result.scg(1) == pytest.approx(0.0)  # staleness_true=3 > 1
    assert result.scg(3) == pytest.approx(1.0)  # staleness_true=3 <= 3


def test_scg_w_hard_constraint_matches_scg():
    reqs = [_make_request(i, 0, 0, 1) for i in range(8)]
    result = _make_result(reqs, 8.0)
    w: Callable[[int], float] = lambda s: 1.0 if s <= 1 else 0.0
    assert result.scg_w(w) == pytest.approx(result.scg(1))


def test_scg_zero_sim_time():
    result = _make_result([], 0.0)
    assert result.scg(5) == pytest.approx(0.0)
