"""The default segment methods must reproduce the closed form exactly."""

from __future__ import annotations

import pytest

from dryrun.cost.rollout import PSRLFitted, UnifiedRoofline, advance

_BATCHES = (1, 4, 16, 48, 64)
_CONTEXTS = (1024.0, 8192.0, 65536.0, 262144.0)
_STEPS = (0, 1, 7, 256, 1024, 4096)


def _psrl() -> PSRLFitted:
    return PSRLFitted(
        prefill_attn_b=2.0e-4,
        prefill_attn_k=3.0e-9,
        prefill_other_floor=1.0e-3,
        prefill_other_b=5.0e-4,
        prefill_other_k=2.0e-7,
        decode_attn_b=1.0e-4,
        decode_attn_k=4.0e-9,
        decode_other_floor=2.0e-3,
        decode_other_b=6.0e-4,
        decode_other_k=1.0e-5,
    )


def _roofline() -> UnifiedRoofline:
    return UnifiedRoofline(F=0.002, W=0.0005, G=2.0e-7, A_p=3.0e-9, A_d=4.0e-9, b=16)


@pytest.mark.parametrize("model", [_psrl(), _roofline()], ids=["psrl", "roofline"])
def test_segment_time_equals_the_closed_form(model):
    for n_d in _BATCHES:
        for ctxsum in _CONTEXTS:
            floor, alpha, beta = model.decode_coeffs(n_d, ctxsum)
            for k in _STEPS:
                assert model.decode_segment_time(n_d, ctxsum, k) == advance.elapsed(
                    floor, alpha, beta, k
                )


@pytest.mark.parametrize("model", [_psrl(), _roofline()], ids=["psrl", "roofline"])
def test_steps_within_equals_the_closed_form(model):
    for n_d in _BATCHES:
        for ctxsum in _CONTEXTS:
            floor, alpha, beta = model.decode_coeffs(n_d, ctxsum)
            for budget in (0.0, 1.0e-3, 0.05, 1.0, 30.0):
                assert model.decode_steps_within(n_d, ctxsum, budget) == advance.steps_within(
                    floor, alpha, beta, budget
                )


@pytest.mark.parametrize("model", [_psrl(), _roofline()], ids=["psrl", "roofline"])
def test_saturated_steps_equals_the_closed_form(model):
    for n_d in _BATCHES:
        for ctxsum in _CONTEXTS:
            floor, alpha, beta = model.decode_coeffs(n_d, ctxsum)
            for k in _STEPS:
                expected = k - advance.crossing_step(floor, alpha, beta, k_cap=k)
                assert model.decode_saturated_steps(n_d, ctxsum, k) == expected


def test_a_zero_length_segment_costs_nothing():
    model = _psrl()
    assert model.decode_segment_time(16, 8192.0, 0) == 0
    assert model.decode_saturated_steps(16, 8192.0, 0) == 0
