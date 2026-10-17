"""Length models: conditioning, monotonicity, and what the oracle may read."""

from __future__ import annotations

import pytest

from srl.value import EmpiricalLength, OracleLength, TrajView


def traj(generated: int = 0, target_len: int = 1000, prompt_key: int = 0) -> TrajView:
    return TrajView(
        generated=generated,
        prompt_len=512,
        target_len=target_len,
        prompt_key=prompt_key,
        v_traj=0,
        admit_time=0.0,
    )


def uniform_model(**kwargs) -> EmpiricalLength:
    """A model fed lengths spread evenly over (0, 10000]."""
    model = EmpiricalLength(max_length=16384, bin_width=256, min_samples=10, **kwargs)
    for length in range(100, 10001, 100):
        model.observe(traj(target_len=length), final_length=length)
    return model


class TestOracle:
    def test_a_bare_step_is_exact_at_the_true_remaining_length(self):
        model = OracleLength(band=0.0)
        t = traj(generated=400, target_len=1000)
        assert model.hit_prob(t, 599.0) == 0.0
        assert model.hit_prob(t, 601.0) == 1.0

    def test_the_band_keeps_v_centred_on_the_true_answer(self):
        """A bare step makes V correct and every derived ordering useless."""
        model = OracleLength(band=0.25)
        t = traj(generated=400, target_len=1000)
        assert model.hit_prob(t, 600.0) == pytest.approx(0.5)
        assert model.hit_prob(t, 300.0) == 0.0
        assert model.hit_prob(t, 900.0) == 1.0

    def test_the_oracle_reads_the_future_and_the_empirical_model_does_not(self):
        """The two differ on exactly one thing, which is the point of having both."""
        oracle = OracleLength()
        short = traj(generated=100, target_len=200)
        long = traj(generated=100, target_len=9000)
        assert oracle.hit_prob(short, 500.0) > oracle.hit_prob(long, 500.0)


class TestConditioning:
    def test_a_deeper_trajectory_scores_higher_on_the_same_budget(self):
        """The property the whole ordering rests on.

        Under a uniform length distribution a request 8000 tokens in is nearly
        certain to finish within 5000 more, while a fresh one is not. An
        unconditional CDF returns the same number for both, so this test fails
        loudly if the division by `1 - F(generated)` is ever dropped.
        """
        model = uniform_model(per_prompt=False)
        deep = model.hit_prob(traj(generated=8000), 5000.0)
        fresh = model.hit_prob(traj(generated=0), 5000.0)
        assert deep > fresh
        assert deep == pytest.approx(1.0, abs=0.02)
        assert fresh == pytest.approx(0.5, abs=0.05)

    def test_a_trajectory_past_every_observation_claims_nothing(self):
        model = uniform_model(per_prompt=False)
        assert model.hit_prob(traj(generated=20000), 5000.0) == 0.0


class TestPerPrompt:
    def test_a_prompts_own_evidence_moves_its_estimate(self):
        model = uniform_model(per_prompt=True, prior_strength=4.0)
        hard = traj(generated=0, prompt_key=99)
        before = model.hit_prob(hard, 3000.0)
        for _ in range(8):
            model.observe(hard, final_length=9500)
        after = model.hit_prob(hard, 3000.0)
        assert after < before

    def test_an_unseen_prompt_falls_back_to_global(self):
        model = uniform_model(per_prompt=True)
        globally = EmpiricalLength(max_length=16384, bin_width=256, min_samples=10, per_prompt=False)
        for length in range(100, 10001, 100):
            globally.observe(traj(target_len=length), final_length=length)
        t = traj(generated=1000, prompt_key=12345)
        assert model.hit_prob(t, 4000.0) == pytest.approx(globally.hit_prob(t, 4000.0))


class TestObservability:
    def test_the_empirical_model_never_reads_target_len(self):
        """An observable model must not see the future, even by accident."""
        model = uniform_model(per_prompt=False)
        honest = model.hit_prob(traj(generated=3000, target_len=3100), 2000.0)
        lying = model.hit_prob(traj(generated=3000, target_len=99999), 2000.0)
        assert honest == lying


class TestDecay:
    def test_decay_preserves_the_distribution_but_shrinks_the_evidence(self):
        model = uniform_model(per_prompt=False)
        t = traj(generated=2000)
        before = model.hit_prob(t, 3000.0)
        model.decay(0.5)
        assert model.hit_prob(t, 3000.0) == pytest.approx(before)
        assert model._global_n == pytest.approx(50.0)

    def test_decay_can_drop_a_model_below_ready(self):
        model = uniform_model(per_prompt=False)
        assert model.ready
        model.decay(0.01)
        assert not model.ready

    def test_an_invalid_factor_is_refused(self):
        with pytest.raises(ValueError, match="Decay factor"):
            uniform_model().decay(1.5)
