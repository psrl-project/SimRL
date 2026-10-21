"""A bound must be optimistic, labelled, and equal to the real span when nothing can improve."""

from __future__ import annotations

import pytest
from diagnosis_fixtures import BLOCK_SIZE, _cost_model, run_mixed, run_single_request

from srl.diagnosis.bounds import span_solo_all, span_tail_only
from srl.diagnosis.critical_path import steps
from srl.diagnosis.pricer import SoloPricer
from srl.diagnosis.timeline import request_phases


def _setup(store):
    timelines, _ = request_phases(store)
    return timelines, steps(store, timelines), SoloPricer({1: _cost_model()}, BLOCK_SIZE)


def test_a_lone_request_leaves_no_headroom(tmp_path):
    store = run_single_request(tmp_path)
    timelines, found = _setup(store)[:2]
    pricer = SoloPricer({1: _cost_model()}, BLOCK_SIZE)
    bound = span_solo_all(timelines, found, pricer, {0: 1}, charge_reprefill=True)
    observed = max(step.end for step in found)
    assert bound.span == pytest.approx(observed, rel=0.02), (
        f"Lone-request bound {bound.span!r} should match the observed span {observed!r}."
    )
    assert bound.headroom == pytest.approx(1.0, rel=0.02), (
        f"Lone-request headroom {bound.headroom!r} should be about 1.0."
    )


def test_every_bound_is_at_most_the_observed_span(tmp_path):
    store = run_mixed(tmp_path)
    timelines, found, pricer = _setup(store)
    observed = max(step.end for step in found)
    for bound in (
        span_solo_all(timelines, found, pricer, {0: 1, 1: 1}, charge_reprefill=True),
        span_solo_all(timelines, found, pricer, {0: 1, 1: 1}, charge_reprefill=False),
        span_tail_only(timelines, found, pricer, {0: 1, 1: 1}),
    ):
        assert bound.span <= observed + 1e-6, f"{bound.name} exceeds the observed span."
        assert bound.headroom >= 1.0 - 1e-9, f"{bound.name} reports headroom below 1.0: {bound.headroom!r}."
        assert bound.assumptions, f"{bound.name} carries no stated assumptions."


def test_zeroing_reprefill_is_never_the_weaker_bound(tmp_path):
    store = run_mixed(tmp_path)
    timelines, found, pricer = _setup(store)
    charged = span_solo_all(timelines, found, pricer, {0: 1, 1: 1}, charge_reprefill=True)
    free = span_solo_all(timelines, found, pricer, {0: 1, 1: 1}, charge_reprefill=False)
    assert free.span <= charged.span + 1e-9, (
        f"Zeroing reprefill gave a weaker bound: free={free.span!r} charged={charged.span!r}."
    )


def test_tail_only_is_weaker_than_solo_all(tmp_path):
    """Pricing only the binding chain solo cannot beat pricing everything solo."""
    store = run_mixed(tmp_path)
    timelines, found, pricer = _setup(store)
    tail = span_tail_only(timelines, found, pricer, {0: 1, 1: 1})
    every = span_solo_all(timelines, found, pricer, {0: 1, 1: 1}, charge_reprefill=True)
    assert every.span <= tail.span + 1e-6, (
        f"span_tail_only is tighter than span_solo_all: tail={tail.span!r} every={every.span!r}."
    )
