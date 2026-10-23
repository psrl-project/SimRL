"""Segment heights must be exact, and the figure must render headless."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from srl.diagnosis.staleness import StalenessBreakdown
from srl.visualizer.staleness_sweep import (
    plot_staleness_throughput,
    stack_segments,
    staleness_levels,
    waste_reasons,
)

LOOSE = StalenessBreakdown(
    span=100.0,
    n_consumed=8,
    consumed_tokens={0: 1000, 1: 2000, 3: 3000},
    wasted_tokens={},
    reprefill_tokens=0,
)
TIGHT = StalenessBreakdown(
    span=200.0,
    n_consumed=8,
    consumed_tokens={0: 1000, 1: 1000},
    wasted_tokens={"l3_stale": 4000},
    reprefill_tokens=0,
)
PANELS = [("thr = 5", [("1", LOOSE), ("2", TIGHT)])]


def test_levels_span_every_panel_with_no_gaps():
    assert staleness_levels(PANELS) == [0, 1, 2, 3], (
        "Levels must be contiguous so a missing level reads as an absence, not a relabelled bar."
    )


def test_reasons_are_sorted_and_deduplicated():
    assert waste_reasons(PANELS) == ["l3_stale"]


def test_segment_heights_are_tokens_over_span():
    segments = stack_segments(PANELS[0][1], [0, 1, 2, 3], ["l3_stale"])
    assert segments["S0"] == pytest.approx([10.0, 5.0])
    assert segments["S1"] == pytest.approx([20.0, 5.0])
    assert segments["S2"] == pytest.approx([0.0, 0.0])
    assert segments["S3"] == pytest.approx([30.0, 0.0])
    assert segments["waste:l3_stale"] == pytest.approx([0.0, 20.0])


def test_the_stack_sums_to_full_generation_throughput():
    levels, reasons = [0, 1, 2, 3], ["l3_stale"]
    segments = stack_segments(PANELS[0][1], levels, reasons)
    totals = [sum(heights[i] for heights in segments.values()) for i in range(2)]
    assert totals[0] == pytest.approx(LOOSE.throughput)
    assert totals[1] == pytest.approx(TIGHT.throughput)


def test_the_figure_is_written(tmp_path):
    out = plot_staleness_throughput(
        PANELS,
        primary_label="num_warmup_batches (batches)",
        title="verl_v1",
        out_path=tmp_path / "fig.png",
        footnote="training cost extrapolated",
    )
    assert out.exists(), f"{out} was not written."
    assert out.stat().st_size > 0, f"{out} is empty."


def test_a_two_panel_figure_is_written(tmp_path):
    panels = [("thr = 1", [("1", TIGHT)]), ("thr = 5", [("1", LOOSE)])]
    out = plot_staleness_throughput(
        panels, primary_label="knob", title="two panels", out_path=tmp_path / "two.png"
    )
    assert out.exists(), f"{out} was not written."


def test_an_empty_panel_set_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no points"):
        plot_staleness_throughput([], primary_label="knob", title="t", out_path=tmp_path / "x.png")
