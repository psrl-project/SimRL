"""The lattice integrator: accuracy, monotonicity, exact inversion, bounded lookups."""

from __future__ import annotations

import math

import pytest

from dryrun.cost.rollout.segment import LatticeIntegrator


def _linear(offset: int) -> float:
    """A cost that grows linearly, whose sum has a closed form."""
    return 1.0e-3 + 2.0e-6 * offset


def _spiky(offset: int) -> float:
    """A Gaussian spike, the adversarial shape that breaks a moving-sample trapezoid."""
    return 1.0e-3 + 5.0e-3 * math.exp(-(((offset - 2000) / 150.0) ** 2))


def _trough(offset: int) -> float:
    """A deep dip then a climb, the shape the real utility curve exhibits."""
    return 1.0e-3 * (1.0 + 3.0 / (1.0 + math.exp((offset - 1500) / 200.0)))


def test_a_zero_length_segment_costs_nothing():
    integrator = LatticeIntegrator(_linear, k_max=4096)
    assert integrator.total(0) == 0.0
    assert integrator.total(-5) == 0.0


def test_stride_one_reproduces_the_exact_sum():
    """At stride 1 the lattice nodes are the steps, so there is nothing left to approximate."""
    integrator = LatticeIntegrator(_linear, k_max=64, stride=1)
    for k in (1, 7, 32, 64):
        expected = sum(_linear(i) for i in range(k))
        assert integrator.total(k) == pytest.approx(expected, rel=1e-12)


def test_a_linear_cost_integrates_exactly_at_any_stride():
    """On an affine cost the lattice is exact to floating point, whatever the stride."""
    # The lattice trapezoid-integrates over [0, k] while a segment sums the steps
    # {0, ..., k-1}, and on an affine cost those differ by exactly (f(k) - f(0)) / 2: a
    # structural offset, not an approximation error. Folding it into the reference is what
    # makes "exact" testable at 1e-12. Comparing against the bare sum would only confirm
    # that 1.96e-4 offset, and a loose tolerance there would still pass a broken lattice.
    steps = 4096
    reference = sum(_linear(i) for i in range(steps)) + (_linear(steps) - _linear(0)) / 2.0
    for stride in (64, 256, 1024, 2048):
        integrator = LatticeIntegrator(_linear, k_max=steps, stride=stride)
        assert integrator.total(steps) == pytest.approx(reference, rel=1e-12)


def test_accuracy_improves_as_the_stride_shrinks_on_a_curved_cost():
    exact = sum(_trough(i) for i in range(4096))
    errors = [
        abs(LatticeIntegrator(_trough, k_max=4096, stride=stride).total(4096) - exact) / exact
        for stride in (2048, 512, 128)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.01


@pytest.mark.parametrize("cost", [_linear, _spiky, _trough], ids=["linear", "spiky", "trough"])
def test_the_total_is_strictly_increasing_in_k(cost):
    """Without this, steps_within has no unique inverse and instance.py's k == k_budget breaks."""
    integrator = LatticeIntegrator(cost, k_max=4096)
    previous = integrator.total(0)
    for k in range(1, 4097):
        current = integrator.total(k)
        assert current > previous, f"not increasing at k={k}: {previous} -> {current}"
        previous = current


@pytest.mark.parametrize("cost", [_linear, _spiky, _trough], ids=["linear", "spiky", "trough"])
def test_steps_within_exactly_inverts_total(cost):
    """Probes straddle lattice nodes at 1024, 2048, 3072 on purpose."""
    integrator = LatticeIntegrator(cost, k_max=4096)
    for k in (1, 7, 100, 1023, 1024, 1025, 2047, 2048, 2049, 3072, 4095, 4096):
        assert integrator.steps_within(integrator.total(k)) == k


def test_steps_within_rejects_a_budget_below_the_first_step():
    integrator = LatticeIntegrator(_linear, k_max=4096)
    assert integrator.steps_within(0.0) == 0
    assert integrator.steps_within(-1.0) == 0
    assert integrator.steps_within(0.5 * _linear(0)) == 0


def test_steps_within_saturates_at_k_max():
    integrator = LatticeIntegrator(_linear, k_max=100)
    assert integrator.steps_within(1.0e9) == 100


def test_total_clamps_at_k_max():
    """k_max is the hard ceiling on segment length; beyond it the total must not keep growing."""
    integrator = LatticeIntegrator(_linear, k_max=1024, stride=1024)
    assert integrator.total(3000) == pytest.approx(integrator.total(1024), rel=1e-12)


def test_construction_samples_nothing():
    """A composed step costs 83.6 us, so an eager lattice would cost 5.52 ms to build."""
    calls = []
    LatticeIntegrator(lambda offset: calls.append(offset) or _linear(offset), k_max=1 << 16)
    assert calls == []


def test_a_short_segment_only_samples_the_nodes_it_needs():
    """instance.py builds an integrator per decode event; most segments are short."""
    calls = []

    def counted(offset: int) -> float:
        calls.append(offset)
        return _linear(offset)

    integrator = LatticeIntegrator(counted, k_max=1 << 16, stride=1024)
    integrator.total(300)
    assert len(calls) == 2, f"a 300-step segment sampled {len(calls)} nodes"
    assert integrator.sample_count == 2


def test_the_lookup_count_is_bounded_by_the_stride_not_the_step_count():
    """The whole point: a 4096-step segment must not cost 4096 cost evaluations."""
    calls = []

    def counted(offset: int) -> float:
        calls.append(offset)
        return _linear(offset)

    integrator = LatticeIntegrator(counted, k_max=1 << 16, stride=1024)
    for k in range(0, 4097, 97):
        integrator.total(k)
    integrator.steps_within(1.0)
    assert len(calls) <= 8, f"expected a handful of lookups, made {len(calls)}"
    assert integrator.sample_count == len(calls)


def test_a_node_is_never_sampled_twice():
    """Repeated queries must reuse the cached lattice, not re-evaluate the cost."""
    calls = []

    def counted(offset: int) -> float:
        calls.append(offset)
        return _linear(offset)

    integrator = LatticeIntegrator(counted, k_max=1 << 16, stride=1024)
    for _ in range(50):
        integrator.total(2500)
    assert len(calls) == len(set(calls))


def test_a_non_positive_stride_is_rejected():
    with pytest.raises(ValueError, match="stride"):
        LatticeIntegrator(_linear, k_max=4096, stride=0)


def test_a_negative_k_max_is_rejected():
    with pytest.raises(ValueError, match="k_max"):
        LatticeIntegrator(_linear, k_max=-1)


def test_a_non_positive_step_cost_is_rejected():
    """A zero-cost step would make the total flat and destroy invertibility."""
    # Construction samples nothing, so the check can only fire when a node is first
    # evaluated. Asserting it at construction would contradict the lazy contract that
    # test_construction_samples_nothing pins down.
    integrator = LatticeIntegrator(lambda offset: 0.0, k_max=1024)
    with pytest.raises(ValueError, match="positive"):
        integrator.total(1)
