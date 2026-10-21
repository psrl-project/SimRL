"""Figures must render from the diagnosis outputs without a display."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from diagnosis_fixtures import run_mixed

from srl.diagnosis.critical_path import steps
from srl.diagnosis.fleet import idle_windows, instance_imbalance
from srl.diagnosis.timeline import request_phases
from srl.visualizer.diagnosis import DiagnosisPlotter


def test_every_figure_is_written(tmp_path):
    store = run_mixed(tmp_path)
    timelines, _ = request_phases(store)
    found = steps(store, timelines)
    plotter = DiagnosisPlotter(
        store=store,
        timelines=timelines,
        steps_=found,
        windows=idle_windows(store, found, timelines, staleness_horizon=1),
        imbalance=instance_imbalance(store),
        out_dir=tmp_path / "figures",
    )
    paths = plotter.plot_all()
    assert len(paths) == 4, f"Expected 4 figures, got {len(paths)!r}: {paths!r}."
    for path in paths:
        assert path.exists(), f"{path} was not written."
        assert path.stat().st_size > 0, f"{path} is empty."
