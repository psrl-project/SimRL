"""Tests for length distribution generators."""

from __future__ import annotations

from dryrun.workload.distributions import fixed, normal


class TestNormal:
    def test_returns_n_samples(self):
        lengths = normal(20, mu=100.0, sigma=10.0, lo=1, hi=200, seed=0)
        assert len(lengths) == 20
        assert all(isinstance(length, int) for length in lengths)

    def test_clips_to_bounds(self):
        lengths = normal(50, mu=0.0, sigma=1.0, lo=8, hi=12, seed=1)
        assert all(8 <= length <= 12 for length in lengths)

    def test_seed_is_reproducible(self):
        assert normal(10, mu=64.0, sigma=8.0, hi=128, seed=7) == normal(
            10, mu=64.0, sigma=8.0, hi=128, seed=7
        )


class TestFixed:
    def test_returns_n_copies(self):
        assert fixed(5, 512) == [512] * 5

    def test_seed_is_accepted_but_ignored(self):
        assert fixed(3, 10, seed=99) == [10, 10, 10]
