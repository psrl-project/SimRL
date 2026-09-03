"""Test closed-form advance: must match brute-force step summation."""

from fractions import Fraction

from dryrun.cost.rollout import advance


def test_elapsed_matches_brute_force():
    F, alpha, beta = Fraction(5), Fraction(2), Fraction(1)
    for k in range(20):
        brute = sum(max(F, alpha + beta * i) for i in range(k))
        assert advance.elapsed(F, alpha, beta, k) == brute, f"Mismatch at k={k}."


def test_elapsed_no_floor():
    F, alpha, beta = Fraction(0), Fraction(3), Fraction(2)
    for k in range(15):
        brute = sum(alpha + beta * i for i in range(k))
        assert advance.elapsed(F, alpha, beta, k) == brute


def test_elapsed_pure_floor():
    F, alpha, beta = Fraction(10), Fraction(1), Fraction(0)
    for k in range(10):
        assert advance.elapsed(F, alpha, beta, k) == F * k


def test_steps_within_inverse_of_elapsed():
    F, alpha, beta = Fraction(5), Fraction(2), Fraction(1)
    for k in range(1, 20):
        budget = advance.elapsed(F, alpha, beta, k)
        assert advance.steps_within(F, alpha, beta, budget) == k
        assert advance.steps_within(F, alpha, beta, budget - Fraction(1, 1000)) == k - 1


def test_steps_within_float():
    F, alpha, beta = 5.0, 2.0, 1.0
    for k in range(1, 15):
        budget = float(advance.elapsed(Fraction(5), Fraction(2), Fraction(1), k))
        result = advance.steps_within(F, alpha, beta, budget)
        assert result == k or result == k - 1


def test_next_completion():
    assert advance.next_completion([]) == (0, 0)
    assert advance.next_completion([5, 3, 3, 7]) == (3, 2)
    assert advance.next_completion([1]) == (1, 1)
    assert advance.next_completion([4, 4, 4]) == (4, 3)


def test_crossing_step_basics():
    assert advance.crossing_step(Fraction(10), Fraction(10), Fraction(1)) == 0
    assert advance.crossing_step(Fraction(10), Fraction(5), Fraction(1)) == 5
    assert advance.crossing_step(Fraction(10), Fraction(5), Fraction(2)) == 3
    assert advance.crossing_step(Fraction(10), Fraction(5), Fraction(0)) == 0


def test_steps_within_beta_zero_is_exact():
    """When beta == 0, steps_within(elapsed(k)) must equal k exactly for all k."""
    # Reproduces the reported off-by-one: F=0, alpha=0.007774277, beta=0.
    F, alpha, beta = 0.0, 0.007774277, 0.0
    for k in [7, 1023, 10000]:
        budget = advance.elapsed(F, alpha, beta, k)
        result = advance.steps_within(F, alpha, beta, budget)
        assert result == k, f"beta=0 inversion failed at k={k}: got {result}."


def test_steps_within_beta_zero_floor_dominates():
    """When beta == 0 and F > alpha, the floor governs."""
    F, alpha, beta = 0.002, 0.001, 0.0
    for k in [1, 5, 100, 1000]:
        budget = advance.elapsed(F, alpha, beta, k)
        result = advance.steps_within(F, alpha, beta, budget)
        assert result == k, f"beta=0 floor-governs inversion failed at k={k}: got {result}."
