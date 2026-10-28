"""Stacked bars: which staleness level produced each part of the throughput.

One bar per knob setting. The bar's height is end-to-end throughput: generated
tokens, consumed or discarded, divided by the wall-clock span through the last
training step. Lower segments are tokens that reached training, keyed by the
staleness they carried; a blue colorbar (0 to 6) is that key. Hatched upper
segments are tokens that were generated and discarded. Achieved max staleness is the line on the right axis. A point past the
cap that knob is designed to enforce is a red cross; a few knobs enforce
`value - 1` rather than the value itself.

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
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from ..diagnosis.staleness import StalenessBreakdown  # noqa: E402
from .plots import _FIG_DPI, _ORANGE  # noqa: E402

Point = tuple[str, StalenessBreakdown]
Panel = tuple[str, Sequence[Point]]

# Discards are drawn outside the categorical palette: a neutral fill plus a
# hatch, so a waste segment can never be mistaken for a staleness level.
_WASTE_FILL = "#d9d7cf"
_WASTE_EDGE = "#6f6d66"
_WASTE_HATCHES = ("//", "xx", "..", "\\\\", "++")
_BAR_WIDTH = 0.88
_EXCEED = "#d21f3c"

# Designed ceiling on consumed staleness, keyed by the sweep's primary label.
# `source` says whether the number lives on the bar (`x`) or the panel.
# Two-knob figures are reoriented so the staleness knob is the bar, and the
# cap is read from there. `offset` is -1 where the knob's own comparison
# allows one less than the number it is written as: verl's span is
# `version_diff + 1`, and NeMo's in-order gate blocks at `train + lookahead`,
# so the lead is `lookahead - 1`.
# A missing label, or an explicit None, means the axis is not a version cap
# (`verl`'s sample-overage ratio) and no cross is drawn.
_CapRule = tuple[str, int]
_CAP_RULES: dict[str, _CapRule | None] = {
    "psrl staleness": ("x", 0),
    "roll async_generation_ratio": ("x", 0),
    "trl max_staleness": ("x", 0),
    # slime's axes bound work and KV lifetime, not a trajectory's age.
    "slime group_prefetch_factor": None,
    "slime flush_cache_interval (0 = never abort)": None,
    "nemo_rl max_staleness_versions": ("x", 0),
    "nemo_rl max_lookahead_versions": ("x", -1),
    "skyrl max_staleness_steps": ("x", 0),
    "areal max_head_offpolicyness": ("x", 0),
    "openrlhf async_queue_size": ("x", 0),
    "dressage keep_versions": ("x", 0),
    "prime_rl max_off_policy_steps": ("x", 0),
    "prime_rl target_lag": ("x", 0),
    "verl_v1 num_warmup_batches": ("x", -1),
    "verl_v1 num_warmup_batches (wait)": ("x", -1),
    "verl staleness_threshold": None,
}


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


def split_panel_name(panel_name: str) -> tuple[str, str]:
    """Split a `"{knob} = {value}"` panel name into its two halves.

    Every multi-panel sweep names its panels this way (`sweep.sh`'s own
    `panel_name="${SECONDARY_KNOB##*.} = ${panel}"`), and the knob half is
    identical across every panel in a row -- repeating it as each panel's
    own title is what forces a panel wide enough to hold the whole string,
    working directly against a narrow, many-panels-per-row layout. Returns
    `("", panel_name)` unchanged when the `" = "` separator is absent (a
    single, untitled panel).
    """
    knob, sep, value = panel_name.partition(" = ")
    return (knob, value) if sep else ("", panel_name)


def orient_panels(panels: Sequence[Panel], primary_label: str) -> tuple[list[Panel], str]:
    """Put a two-knob sweep's staleness knob on each panel's x axis.

    Manifests store that knob as the panel (`"max_off_policy_threshold = 2"`)
    and the other knob as the bar. Each panel is then titled
    `num_warmup_batches=1`, and the shared x label names only the staleness
    knob. A one-knob figure is returned unchanged.
    """
    if not panels or any(not split_panel_name(name)[0] for name, _ in panels):
        return list(panels), primary_label

    secondary = split_panel_name(panels[0][0])[0]
    system, _, rest = primary_label.partition(" ")
    if not rest:
        primary_knob = primary_label
        suffix = ""
    else:
        primary_knob, sep, tail = rest.partition(" (")
        suffix = f" ({tail}" if sep else ""

    secondary_values = [split_panel_name(name)[1] for name, _ in panels]
    primary_values: list[str] = []
    lookup: dict[tuple[str, str], StalenessBreakdown] = {}
    for name, points in panels:
        sec = split_panel_name(name)[1]
        for label, bd in points:
            if label not in primary_values:
                primary_values.append(label)
            lookup[(label, sec)] = bd

    reoriented: list[Panel] = []
    for primary in primary_values:
        points = [(sec, lookup[(primary, sec)]) for sec in secondary_values if (primary, sec) in lookup]
        reoriented.append((f"{primary_knob}={primary}", points))
    return reoriented, f"{system} {secondary}{suffix}" if rest else f"{primary_label} {secondary}"


def staleness_cap(primary_label: str, panel_name: str, point_label: str) -> int | None:
    """Designed ceiling on consumed staleness for one bar, or None if the axis is not one.

    Compared with `>`: a bar whose achieved max equals the ceiling is inside
    the cap. The panel value is the secondary knob (`"threshold = 2"` → 2).
    """
    if primary_label not in _CAP_RULES:
        return None
    rule = _CAP_RULES[primary_label]
    if rule is None:
        return None
    source, offset = rule
    raw = point_label if source == "x" else split_panel_name(panel_name)[1]
    try:
        bound = int(raw)
    except ValueError:
        return None
    return bound + offset


def staleness_series(points: Sequence[Point]) -> list[int]:
    """Per-point achieved max staleness, in bar order: the right axis's line.

    This is the knob's own effect on the data, not the knob setting itself
    (already on the x axis): a knob that overshoots its own ceiling, or a
    discard-based bound that holds it exactly, both show up here.
    """
    return [bd.max_staleness for _, bd in points]


def stack_segments(
    points: Sequence[Point],
    levels: Sequence[int],
    reasons: Sequence[str],
) -> dict[str, list[float]]:
    """Per-point segment heights in tokens per second, keyed by segment name.

    Consumed segments are keyed `S<level>` and discard segments `waste:<reason>`.
    Summing every value at one index gives that point's end-to-end throughput
    (generated tokens over the span through the last training step).

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


# Scoped to this figure rather than the shared rcParams in `plots.py`: this
# family is read as a slide-sized chart (bars, line, colorbar) rather than a
# dense multi-panel diagnostic, so it earns its own, larger type.
_RC_OVERRIDES = {
    "font.size": 16,
    "axes.titlesize": 19,
    "axes.labelsize": 16,
    "figure.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    # `\mathbf` then matches the DejaVu Sans used for the rest of the xlabel.
    "mathtext.fontset": "dejavusans",
    # Outline the type in the SVG so the bold system name does not depend on
    # the viewer having DejaVu installed.
    "svg.fonttype": "path",
}
# Shared across every figure in the sweep. Nothing in this regime exceeds 6,
# and a fixed scale keeps one staleness the same blue in every preset.
_STALENESS_VMIN = 0
_STALENESS_VMAX = 6
# `Blues` starts at white, which disappears on the figure background.
_BLUES_FLOOR = 0.35
_BLUES_CEIL = 0.95


def format_system_xlabel(label: str) -> str:
    """Bold the leading system name and put a colon after it.

    ``"areal max_head_offpolicyness"`` becomes mathtext whose first word,
    plus the colon, is bold, and the knob name stays in the text font.
    A label with no space is returned unchanged.
    """
    system, sep, rest = label.partition(" ")
    if not sep:
        return label
    system_math = system.replace("_", r"\_")
    return rf"$\mathbf{{{system_math}:}}$ {rest}"


def _staleness_cmap() -> LinearSegmentedColormap:
    samples = plt.cm.Blues(np.linspace(_BLUES_FLOOR, _BLUES_CEIL, 256))
    return LinearSegmentedColormap.from_list("staleness_blues", samples)


def _colorbar_axes(fig: plt.Figure, twin: plt.Axes) -> plt.Axes:
    """A vertical strip just to the right of the max-staleness axis.

    Placed after layout, in figure coordinates, so it stays the rightmost
    element instead of fighting the twin spine for the same margin.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [twin.yaxis.label.get_window_extent(renderer)]
    tick_box = twin.yaxis.get_tightbbox(renderer)
    if tick_box is not None:
        boxes.append(tick_box)
    right = max(box.transformed(inv).x1 for box in boxes)
    host = twin.get_position()
    fig_width = fig.get_size_inches()[0]
    width = 0.18 / fig_width
    gap = 0.14 / fig_width
    return fig.add_axes([right + gap, host.y0, width, host.height])


def plot_staleness_throughput(
    panels: Sequence[Panel],
    primary_label: str,
    out_path: str | Path,
) -> Path:
    """Draw one stacked bar per point, one subplot per panel, shared y axis.

    Staleness is a blue colorbar on the far right, fixed from 0 to 6, rather
    than a categorical legend. A second axis overlays each point's achieved
    max staleness. Discard reasons, when any point has them, keep a legend
    so hatches stay distinguishable.

    Args:
        panels (Sequence[Panel]): `(panel_name, points)` pairs. A single panel
            may carry an empty name, which suppresses its subplot title.
        primary_label (str): X axis label, naming the knob and its unit. The
            first word is treated as the system name.
        out_path (str | Path): Destination PNG. An SVG with the same stem is
            written beside it. Parent directories are created.

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

    with plt.rc_context(_RC_OVERRIDES):
        return _draw(panels, primary_label, out_path)


def _draw(panels: Sequence[Panel], primary_label: str, out_path: Path) -> Path:
    cap_label = primary_label
    panels, axis_label = orient_panels(panels, primary_label)
    levels = staleness_levels(panels)
    reasons = waste_reasons(panels)
    cmap = _staleness_cmap()
    norm = Normalize(vmin=_STALENESS_VMIN, vmax=_STALENESS_VMAX)

    # Narrow per-bar and per-panel budgets, so a panel is only as wide as its
    # own bars need: many panels then fit in one row instead of one or two.
    # A long panel title (`num_warmup_batches=1`) gets extra width of its own,
    # or the titles run into each other. The last term is the staleness axis
    # plus the colorbar.
    n_bars = sum(len(points) for _, points in panels)
    title_chars = max((len(name) for name, _ in panels), default=0)
    bars_per_panel = max((len(points) for _, points in panels), default=1)
    title_pad = max(0.0, 0.105 * title_chars - 0.46 * bars_per_panel + 0.4)
    width = max(5.2, 0.46 * n_bars + (0.55 + title_pad) * len(panels) + 1.3)
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 4.8), sharey=True, squeeze=False)
    axes = axes[0]
    tallest_bar = 0.0
    tallest_staleness = 0
    twins = []
    waste_handles: dict[str, object] = {}

    for ax, (panel_name, points) in zip(axes, panels):
        segments = stack_segments(points, levels, reasons)
        xs = np.arange(len(points))
        bottom = np.zeros(len(points))
        # rcParams turn the grid on, and a twin axes would draw it again in
        # front of the bars. Horizontal rules across a stack read as white cuts.
        ax.grid(False)

        for level in levels:
            heights = np.array(segments[f"S{level}"])
            if not heights.any():
                continue
            ax.bar(
                xs,
                heights,
                _BAR_WIDTH,
                bottom=bottom,
                color=cmap(norm(level)),
                edgecolor="none",
                linewidth=0,
            )
            bottom = bottom + heights

        for index, reason in enumerate(reasons):
            heights = np.array(segments[f"waste:{reason}"])
            if not heights.any():
                continue
            label = f"discarded: {reason}"
            bars = ax.bar(
                xs,
                heights,
                _BAR_WIDTH,
                bottom=bottom,
                color=_WASTE_FILL,
                edgecolor=_WASTE_EDGE,
                linewidth=0.6,
                hatch=_WASTE_HATCHES[index % len(_WASTE_HATCHES)],
                label=label,
            )
            waste_handles.setdefault(label, bars)
            bottom = bottom + heights
        tallest_bar = max(tallest_bar, float(bottom.max(initial=0.0)))

        series = np.array(staleness_series(points), dtype=float)
        tallest_staleness = max(tallest_staleness, int(series.max(initial=0)))
        ceilings = [staleness_cap(cap_label, panel_name, label) for label, _ in points]
        exceeded = np.array(
            [cap is not None and int(value) > cap for value, cap in zip(series, ceilings)]
        )
        twin = ax.twinx()
        twin.grid(False)
        twin.plot(xs, series, color=_ORANGE, linewidth=1.8, zorder=3)
        if (~exceeded).any():
            twin.plot(
                xs[~exceeded],
                series[~exceeded],
                linestyle="None",
                marker="o",
                color=_ORANGE,
                markersize=5.5,
                zorder=4,
            )
        if exceeded.any():
            twin.plot(
                xs[exceeded],
                series[exceeded],
                linestyle="None",
                marker="x",
                color=_EXCEED,
                markersize=9,
                markeredgewidth=2.2,
                zorder=5,
            )
        twin.spines["right"].set_visible(True)
        # A twin axes' own background patch is opaque by default and sits
        # above the original axes, which would paint over the bars. Hiding it
        # leaves the line (drawn in the twin's own, later z-order) on top of
        # the bars instead of blotting them out.
        twin.patch.set_visible(False)
        twins.append(twin)

        ax.set_xticks(xs)
        ax.set_xticklabels([label for label, _ in points])
        # Bounds the bars exactly (half a slot of margin on each side)
        # instead of autoscale's default margin, which stacks extra blank
        # space onto every panel and widens the figure for nothing.
        ax.set_xlim(-0.5, len(points) - 0.5)
        _, panel_value = split_panel_name(panel_name)
        if panel_value:
            ax.set_title(panel_value, fontsize=plt.rcParams["xtick.labelsize"])

    axes[0].set_ylabel("Throughput [tokens / s]")
    axes[0].set_ylim(0, tallest_bar * 1.1 if tallest_bar > 0 else 1.0)

    fig.supxlabel(format_system_xlabel(axis_label))

    staleness_ylim = tallest_staleness * 1.15 if tallest_staleness > 0 else 1.0
    for twin in twins[:-1]:
        twin.set_ylim(0, staleness_ylim)
        twin.set_yticklabels([])
        twin.spines["right"].set_visible(False)
    twins[-1].set_ylim(0, staleness_ylim)
    twins[-1].set_ylabel("Max staleness", color=_ORANGE)
    twins[-1].spines["right"].set_color(_ORANGE)
    twins[-1].tick_params(axis="y", colors=_ORANGE)

    # Discard reasons only. A red cross marks a point past its cap, with no
    # legend entry. Reserve the top band before layout so the legend is not
    # covered by the panels.
    ordered_labels = [f"discarded: {reason}" for reason in reasons if f"discarded: {reason}" in waste_handles]
    top = 1.0
    if ordered_labels:
        fig.legend(
            [waste_handles[label] for label in ordered_labels],
            ordered_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(ordered_labels),
            frameon=False,
        )
        top = 0.88

    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(
        mappable,
        cax=_colorbar_axes(fig, twins[-1]),
        ticks=np.arange(_STALENESS_VMIN, _STALENESS_VMAX + 1),
    )
    colorbar.set_label("Staleness")
    colorbar.ax.grid(False)
    colorbar.outline.set_linewidth(0.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    svg_path = out_path.with_suffix(".svg")
    if svg_path != out_path:
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
