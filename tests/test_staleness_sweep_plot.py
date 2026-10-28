"""Segment heights must be exact, and the figure must render headless."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from srl.diagnosis.staleness import StalenessBreakdown
from srl.visualizer.staleness_sweep import (
    format_system_xlabel,
    orient_panels,
    plot_staleness_throughput,
    split_panel_name,
    stack_segments,
    staleness_cap,
    staleness_levels,
    staleness_series,
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


def test_split_panel_name_separates_the_shared_knob_from_the_value():
    assert split_panel_name("max_off_policy_threshold = 5") == ("max_off_policy_threshold", "5")


def test_system_xlabel_bolds_the_name_and_adds_a_colon():
    assert format_system_xlabel("areal max_head_offpolicyness") == r"$\mathbf{areal:}$ max_head_offpolicyness"
    assert format_system_xlabel("verl_v1 num_warmup_batches  (panels: max_off_policy_threshold)") == (
        r"$\mathbf{verl\_v1:}$ num_warmup_batches  (panels: max_off_policy_threshold)"
    )


def test_system_xlabel_leaves_a_single_token_unchanged():
    assert format_system_xlabel("knob") == "knob"


def test_split_panel_name_passes_through_a_name_with_no_separator():
    assert split_panel_name("") == ("", "")
    assert split_panel_name("no equals sign here") == ("", "no equals sign here")


def test_cap_is_the_knob_unless_the_comparison_is_off_by_one():
    assert staleness_cap("areal max_head_offpolicyness", "", "2") == 2
    assert staleness_cap("nemo_rl max_lookahead_versions", "", "3") == 2
    assert staleness_cap("verl_v1 num_warmup_batches", "num_warmup_batches=5", "2") == 1
    assert staleness_cap("verl_v1 num_warmup_batches (wait)", "num_warmup_batches=5", "1") == 0


def test_two_knob_panels_put_the_staleness_knob_on_x():
    panels = [
        ("max_off_policy_threshold = 1", [("1", TIGHT), ("2", LOOSE)]),
        ("max_off_policy_threshold = 2", [("1", LOOSE), ("2", TIGHT)]),
    ]
    oriented, label = orient_panels(panels, "verl_v1 num_warmup_batches")
    assert label == "verl_v1 max_off_policy_threshold"
    assert [name for name, _ in oriented] == ["num_warmup_batches=1", "num_warmup_batches=2"]
    assert [point for point, _ in oriented[0][1]] == ["1", "2"]
    assert oriented[0][1][1][1] is LOOSE


def test_a_one_knob_figure_is_not_reoriented():
    panels = [("", [("1", LOOSE)])]
    oriented, label = orient_panels(panels, "areal max_head_offpolicyness")
    assert label == "areal max_head_offpolicyness"
    assert oriented == panels


def test_a_ratio_axis_has_no_version_cap():
    assert staleness_cap("verl staleness_threshold", "", "2") is None


def test_staleness_series_is_each_point_s_own_achieved_staleness():
    assert staleness_series(PANELS[0][1]) == [3, 1]


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
        primary_label="num_warmup_batches",
        out_path=tmp_path / "fig.png",
    )
    assert out.exists(), f"{out} was not written."
    assert out.stat().st_size > 0, f"{out} is empty."
    svg = out.with_suffix(".svg")
    assert svg.exists() and svg.stat().st_size > 0, f"{svg} was not written."


def test_a_two_panel_figure_is_written(tmp_path):
    panels = [("thr = 1", [("1", TIGHT)]), ("thr = 5", [("1", LOOSE)])]
    out = plot_staleness_throughput(panels, primary_label="knob", out_path=tmp_path / "two.png")
    assert out.exists(), f"{out} was not written."


def test_an_empty_panel_set_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no points"):
        plot_staleness_throughput([], primary_label="knob", out_path=tmp_path / "x.png")


def test_a_mixed_empty_and_nonempty_panel_set_is_refused(tmp_path):
    panels = [("a", []), ("b", [("1", LOOSE)])]
    with pytest.raises(ValueError, match="panel 'a' has no points"):
        plot_staleness_throughput(panels, primary_label="k", out_path=tmp_path / "x.png")

