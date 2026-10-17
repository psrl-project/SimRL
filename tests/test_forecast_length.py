"""Length predictors: the oracle, its deterministic noise, the floor, and the registry."""

from __future__ import annotations

import math
import statistics

import pytest

from srl.component.rollout.engine import Request, TrajView, traj_view
from srl.forecast import (
    LENGTH_PREDICTORS,
    NoisyOracleLength,
    OracleLength,
    build_length_predictor,
)


def traj(generated: int = 0, target_len: int = 8000, prompt_key: int = 3, prompt_len: int = 140) -> TrajView:
    return TrajView(
        generated=generated,
        prompt_len=prompt_len,
        target_len=target_len,
        prompt_key=prompt_key,
        v_traj=0,
        admit_time=0.0,
    )


class TestOracle:
    def test_it_predicts_the_trace_length(self):
        assert OracleLength().predict_total(traj(generated=500)) == 8000.0

    def test_remaining_is_total_minus_generated(self):
        assert OracleLength().remaining(traj(generated=500)) == 7500.0

    def test_remaining_is_floored_at_one(self):
        oracle = OracleLength()
        assert oracle.remaining(traj(generated=8000)) == 1.0
        assert oracle.remaining(traj(generated=9000)) == 1.0
        assert oracle.remaining(traj(generated=7999)) == 1.0

    def test_it_reads_a_projected_request(self):
        req = Request(rid=0, prompt_len=100, target_len=8000, v_traj=0, group_id=0, source_group_id=0)
        assert OracleLength().predict_total(traj_view(req)) == 8000.0


class TestNoisyOracle:
    def test_zero_noise_is_the_oracle(self):
        t = traj(generated=300)
        noisy = NoisyOracleLength(sigma=0.0)
        assert noisy.predict_total(t) == OracleLength().predict_total(t)
        assert noisy.remaining(t) == OracleLength().remaining(t)

    def test_the_same_request_is_mispredicted_the_same_way_across_instances(self):
        t = traj()
        a = NoisyOracleLength(sigma=0.5, seed=1)
        b = NoisyOracleLength(sigma=0.5, seed=1)
        assert a.predict_total(t) == a.predict_total(t) == b.predict_total(t)

    def test_noise_depends_on_the_hashed_identity(self):
        model = NoisyOracleLength(sigma=0.5, seed=1)
        reference = model.predict_total(traj())
        assert reference != NoisyOracleLength(sigma=0.5, seed=2).predict_total(traj())
        assert reference != model.predict_total(traj(prompt_key=4))
        assert reference != model.predict_total(traj(prompt_len=141))

    def test_noise_is_pinned_to_the_published_hash(self):
        """Sweeps recorded under a seed stay reproducible only while this formula is fixed."""
        for sigma, seed, expected in (
            (0.5, 1, 7017.369539721488),
            (0.5, 2, 10540.200916551836),
            (0.25, 0, 10103.150037237041),
        ):
            model = NoisyOracleLength(sigma=sigma, seed=seed)
            assert model.predict_total(traj()) == pytest.approx(expected, rel=1e-12)

    def test_the_error_has_the_requested_log_spread(self):
        model = NoisyOracleLength(sigma=0.5, seed=0)
        errors = [
            math.log(model.predict_total(traj(target_len=5000 + 37 * i, prompt_key=i)) / (5000 + 37 * i))
            for i in range(2000)
        ]
        assert statistics.pstdev(errors) == pytest.approx(0.5, abs=0.05)
        assert statistics.mean(errors) == pytest.approx(0.0, abs=0.05)

    def test_remaining_is_floored_at_one_when_the_prediction_undershoots(self):
        model = NoisyOracleLength(sigma=0.5, seed=1)
        t = traj(generated=7900)
        assert model.predict_total(t) < t.generated
        assert model.remaining(t) == 1.0

    def test_remaining_is_the_prediction_minus_generated_above_the_floor(self):
        model = NoisyOracleLength(sigma=0.5, seed=2)
        t = traj(generated=1000)
        assert model.remaining(t) == model.predict_total(t) - 1000

    def test_a_negative_sigma_is_refused(self):
        with pytest.raises(ValueError, match="sigma"):
            NoisyOracleLength(sigma=-0.1)


class TestRegistry:
    def test_it_names_exactly_the_oracle_and_the_noisy_oracle(self):
        assert set(LENGTH_PREDICTORS) == {"oracle", "noisy_oracle"}

    def test_it_builds_by_name(self):
        assert isinstance(build_length_predictor("oracle"), OracleLength)
        noisy = build_length_predictor("noisy_oracle", sigma=0.5, seed=7)
        assert isinstance(noisy, NoisyOracleLength)
        assert (noisy.sigma, noisy.seed) == (0.5, 7)

    def test_an_unknown_name_lists_the_available_ones(self):
        with pytest.raises(KeyError, match="empirical.*noisy_oracle.*oracle"):
            build_length_predictor("empirical")

    def test_bad_arguments_reach_the_constructor(self):
        with pytest.raises(TypeError):
            build_length_predictor("oracle", sigma=0.5)
