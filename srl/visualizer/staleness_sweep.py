"""Stacked bars: which staleness level produced each part of the throughput.

One bar per knob setting. The bar's height is the fleet's full generation
throughput, its lower segments are tokens that reached training keyed by the
staleness they carried, and its hatched upper segments are tokens that were
generated and discarded. The boundary between the two is the effective training
throughput, drawn as a line so a knob that converts useful work into waste reads
as the boundary falling while the bar keeps its height.

Importing `plots` applies the shared rcParams and supplies the figure DPI, so
these figures and the per-run ones agree without a second copy of the styling.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..diagnosis.staleness import StalenessBreakdown  # noqa: E402
from .plots import _BLUE_DARK, _FIG_DPI  # noqa: E402

Point = tuple[str, StalenessBreakdown]
Panel = tuple[str, Sequence[Point]]

# Discards are drawn outside the categorical palette: a neutral fill plus a
# hatch, so a waste segment can never be mistaken for a staleness level.
_WASTE_FILL = "#d9d7cf"
_WASTE_EDGE = "#6f6d66"
_WASTE_HATCHES = ("//", "xx", "..", "\\\\", "++")
_BAR_WIDTH = 0.68


def staleness_levels(panels: Sequence[Panel]) -> list[int]:
    """Contiguous staleness levels covering every point, lowest first.

    Contiguous rather than only the observed values: a level no point reached
    must occupy its own empty slot, or two bars would give the same colour to
    different staleness.
    """
    observed = {level for _, points in panels for _, bd in points for level in bd.consumed_tokens}
    if not observed:
        return []
    return list(range(min(observed), max(observed) + 1))


def waste_reasons(panels: Sequence[Panel]) -> list[str]:
    """Every discard reason any point recorded, sorted for a stable legend."""
    return sorted({reason for _, points in panels for _, bd in points for reason in bd.wasted_tokens})


def stack_segments(
    points: Sequence[Point],
    levels: Sequence[int],
    reasons: Sequence[str],
) -> dict[str, list[float]]:
    """Per-point segment heights in tokens per second, keyed by segment name.

    Consumed segments are keyed `S<level>` and discard segments `waste:<reason>`.
    Summing every value at one index gives that point's full generation
    throughput, which is the invariant the figure exists to show.

    Args:
        points (Sequence[Point]): Labelled breakdowns, in bar order.
        levels (Sequence[int]): Staleness levels to emit, from `staleness_levels`.
        reasons (Sequence[str]): Discard reasons to emit, from `waste_reasons`.

    Returns:
        dict[str, list[float]]: Segment name to one height per point.
    """
    segments: dict[str, list[float]] = {}
    for level in levels:
        segments[f"S{level}"] = [bd.consumed_tokens.get(level, 0) / bd.span for _, bd in points]
    for reason in reasons:
        segments[f"waste:{reason}"] = [bd.wasted_tokens.get(reason, 0) / bd.span for _, bd in points]
    return segments


def plot_staleness_throughput(
    panels: Sequence[Panel],
    primary_label: str,
    title: str,
    out_path: str | Path,
    footnote: str = "",
) -> Path:
    """Draw one stacked bar per point, one subplot per panel, shared y axis.

    Args:
        panels (Sequence[Panel]): `(panel_name, points)` pairs. A single panel
            may carry an empty name, which suppresses its subplot title.
        primary_label (str): X axis label, naming the knob and its unit.
        title (str): Figure title.
        out_path (str | Path): Destination PNG. Parent directories are created.
        footnote (str): Caveat text drawn bottom left, for example a note that
            the training cost is outside its fit domain.

    Returns:
        Path: The written PNG.

    Raises:
        ValueError: If no panel carries any point, or if one panel is empty
            while another is not.
    """
    out_path = Path(out_path)
    if not any(points for _, points in panels):
        raise ValueError("Cannot draw a staleness figure with no points.")
    empty_panels = [name for name, points in panels if not points]
    if empty_panels:
        raise ValueError(f"Cannot draw a staleness figure: panel {empty_panels[0]!r} has no points.")

    levels = staleness_levels(panels)
    reasons = waste_reasons(panels)
    colours = plt.cm.Blues(np.linspace(0.35, 0.95, max(len(levels), 1)))

    # +2.1 reserves room for the legend, which now sits outside the axes rather
    # than inside a `tight_layout` squeeze, so it never overlaps the rightmost bar.
    n_bars = sum(len(points) for _, points in panels)
    width = max(7.0, 1.1 * n_bars + 2.0 * len(panels)) + 2.1
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 5.6), sharey=True, squeeze=False)
    axes = axes[0]
    tallest_bar = 0.0

    for ax, (panel_name, points) in zip(axes, panels):
        segments = stack_segments(points, levels, reasons)
        xs = np.arange(len(points))
        bottom = np.zeros(len(points))

        for colour, level in zip(colours, levels):
            heights = np.array(segments[f"S{level}"])
            if not heights.any():
                continue
            ax.bar(
                xs,
                heights,
                _BAR_WIDTH,
                bottom=bottom,
                color=colour,
                edgecolor="white",
                linewidth=0.4,
                label=f"S = {level}",
            )
            bottom = bottom + heights

        consumed_top = bottom.copy()
        for index, reason in enumerate(reasons):
            heights = np.array(segments[f"waste:{reason}"])
            if not heights.any():
                continue
            ax.bar(
                xs,
                heights,
                _BAR_WIDTH,
                bottom=bottom,
                color=_WASTE_FILL,
                edgecolor=_WASTE_EDGE,
                linewidth=0.6,
                hatch=_WASTE_HATCHES[index % len(_WASTE_HATCHES)],
                label=f"discarded: {reason}",
            )
            bottom = bottom + heights
        tallest_bar = max(tallest_bar, float(bottom.max(initial=0.0)))

        # The consumed ceiling is the effective training throughput, which a bar
        # height alone hides once a discard segment sits on top of it.
        edges = np.append(xs - _BAR_WIDTH / 2, xs[-1] + _BAR_WIDTH / 2)
        ax.step(edges, np.append(consumed_top, consumed_top[-1]), where="post", color=_BLUE_DARK, linewidth=1.6)

        for index in range(len(points)):
            ax.annotate(
                f"{points[index][1].span:,.0f}s",
                (xs[index], bottom[index]),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=9,
                color="#555555",
            )

        ax.set_xticks(xs)
        ax.set_xticklabels([label for label, _ in points])
        ax.set_xlabel(primary_label)
        if panel_name:
            ax.set_title(panel_name)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Generation throughput [tokens / s]")
    # Headroom so a bar that reaches the data max still has room for its span
    # annotation below the panel title instead of crowding into it.
    axes[0].set_ylim(0, tallest_bar * 1.12 if tallest_bar > 0 else 1.0)

    # Gathered across panels, since a segment may be absent from the first one,
    # but ordered explicitly (S levels ascending, then discard reasons, then the
    # boundary line) rather than by whatever order a panel's own
    # `get_legend_handles_labels()` happens to return.
    handles: dict[str, object] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    ordered_labels = [f"S = {level}" for level in levels if f"S = {level}" in handles]
    ordered_labels += [f"discarded: {reason}" for reason in reasons if f"discarded: {reason}" in handles]
    ordered_labels.append("effective training throughput")
    handles["effective training throughput"] = plt.Line2D([], [], color=_BLUE_DARK, linewidth=1.6)
    fig.legend(
        [handles[label] for label in ordered_labels],
        ordered_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        title="bar segments",
    )

    fig.suptitle(title)
    if footnote:
        fig.text(0.01, 0.01, footnote, fontsize=9, color="#555555", va="bottom")
    fig.tight_layout(rect=(0.0, 0.04 if footnote else 0.0, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path
