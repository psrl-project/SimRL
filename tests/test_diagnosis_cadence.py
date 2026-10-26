"""The cadence report's idle split and autocorrelation, and the contention-free bound."""

from __future__ import annotations

import json
import math

import pytest

from srl.diagnosis.cadence import cadence_report, span_contention_free


def write_logs(tmp_path, steps, rollout_rows):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    with (log_dir / "train.jsonl").open("w") as handle:
        for version, (start, end) in enumerate(steps, start=1):
            row = {"t": end, "start": start, "end": end, "dt": end - start, "version": version}
            handle.write(json.dumps(row) + "\n")
    with (log_dir / "rollout_step.jsonl").open("w") as handle:
        for instance_id, dt in rollout_rows:
            handle.write(json.dumps({"t": dt, "instance_id": instance_id, "dt": dt}) + "\n")
    return log_dir


class TestCadenceReport:
    def test_trainer_idle_is_the_gap_before_every_step(self, tmp_path):
        log_dir = write_logs(tmp_path, [(10.0, 20.0), (25.0, 35.0)], [(0, 30.0)])
        report = cadence_report(log_dir, n_instances=1)
        assert report.span == 35.0
        assert report.train_busy == 20.0
        assert report.trainer_idle == pytest.approx(15.0)

    def test_fleet_idle_is_one_minus_the_mean_busy_fraction(self, tmp_path):
        log_dir = write_logs(tmp_path, [(0.0, 100.0)], [(0, 80.0), (1, 60.0)])
        report = cadence_report(log_dir, n_instances=2)
        assert report.fleet_idle_frac == pytest.approx(0.3)

    def test_intervals_are_the_gaps_between_consecutive_step_ends(self, tmp_path):
        log_dir = write_logs(tmp_path, [(0.0, 10.0), (10.0, 25.0), (30.0, 45.0)], [(0, 45.0)])
        report = cadence_report(log_dir, n_instances=1)
        assert report.intervals == (15.0, 20.0)

    def test_a_period_two_series_has_a_negative_lag_one_autocorrelation(self, tmp_path):
        ends = [0.0]
        for index in range(12):
            ends.append(ends[-1] + (10.0 if index % 2 == 0 else 30.0))
        steps = [(ends[i], ends[i + 1]) for i in range(len(ends) - 1)]
        log_dir = write_logs(tmp_path, steps, [(0, ends[-1])])
        report = cadence_report(log_dir, n_instances=1)
        assert report.acf1 < -0.5
        assert report.acf2 > 0.5

    def test_a_constant_series_has_no_defined_autocorrelation(self, tmp_path):
        steps = [(10.0 * i, 10.0 * i + 5.0) for i in range(6)]
        log_dir = write_logs(tmp_path, steps, [(0, 60.0)])
        report = cadence_report(log_dir, n_instances=1)
        assert math.isnan(report.acf1)


class TestSpanContentionFree:
    def solo(self, prompt_len: int, output_len: int) -> float:
        return 0.001 * output_len

    def train(self, lengths) -> float:
        return 0.0001 * sum(lengths)

    def test_one_step_is_bounded_by_its_slowest_group(self, tmp_path):
        groups = [(100, [1_000, 2_000]), (100, [500, 400])]
        span = span_contention_free(
            groups,
            staleness=0,
            batch_size=2,
            n_versions=1,
            solo_latency=self.solo,
            train_latency=self.train,
        )
        # The slowest member needs 2.0 s, then the step trains on 3900 tokens plus 2 prompts.
        assert span == pytest.approx(2.0 + self.train([1_100, 2_100, 600, 500]))

    def test_staleness_lets_a_straggler_slip_to_a_later_buffer(self):
        groups = [(100, [50_000]), (100, [100]), (100, [100]), (100, [100])]
        tight = span_contention_free(
            groups, staleness=0, batch_size=2, n_versions=2,
            solo_latency=self.solo, train_latency=self.train,
        )
        loose = span_contention_free(
            groups, staleness=1, batch_size=2, n_versions=2,
            solo_latency=self.solo, train_latency=self.train,
        )
        assert loose < tight

    def test_buffer_membership_follows_occupy_not_a_global_eta_sort(self):
        groups = [
            (0, [100_000]), (0, [100_000]),
            (0, [1_000]), (0, [1_100]), (0, [1_200]), (0, [1_300]),
            (0, [1]), (0, [1]),
        ]

        def train(lengths) -> float:
            return 1e-9 * sum(lengths)

        span = span_contention_free(
            groups, staleness=2, batch_size=2, n_versions=2,
            solo_latency=self.solo, train_latency=train, weight_load=0.0,
        )
        # Buffer 1's true occupants stay groups 4, 5 (ready at 1.3s), not the
        # smaller-ETA groups 6, 7 that a global ETA sort would wrongly pick.
        train_end_0 = 1.1 + train([1_000, 1_100])
        expected = max(1.3, train_end_0) + train([1_200, 1_300])
        assert span == pytest.approx(expected, abs=1e-9)
        assert span != pytest.approx(1.3 + train([1, 1]), abs=1e-9)

    def test_too_few_groups_for_the_requested_steps_is_rejected(self):
        with pytest.raises(ValueError, match="needs at least"):
            span_contention_free(
                [(100, [100])], staleness=0, batch_size=2, n_versions=1,
                solo_latency=self.solo, train_latency=self.train,
            )
