# Staleness-Throughput Sweep Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one stacked-bar figure per async-RL preset showing how that preset's own staleness knob redistributes the fleet's generation throughput across staleness levels and into discarded work.

**Architecture:** Three layers along the repo's existing seams. `srl/diagnosis/staleness.py` reduces a finished run's `request` JSONL stream into a `StalenessBreakdown` (pure, no matplotlib). `srl/visualizer/staleness_sweep.py` draws a set of those as stacked bars. `examples/simulate/staleness_throughput/` is a shared bash runner plus twelve ten-line per-preset grid declarations, driving the existing `examples/simulate/psrl_aligned.sh`.

**Tech Stack:** Python 3.12, matplotlib (Agg), numpy, pytest, Hydra overrides, bash. No new dependencies.

## Global Constraints

- Spec: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/docs/superpowers/specs/2026-10-23-staleness-throughput-sweep-design.md`
- Activate the environment before any run or test: `source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh`
- Coding style: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/.claude/coding-style.md`. Ruff `line-length = 119`, lint select `E,F,W,I`.
- Comments and docstrings must carry no measurements and no history (§5.0), and no `;` or dash characters in prose (§5.0.1). Numbers from the probe belong in commit messages, never in code comments.
- Identifier strings (dict keys, `==` comparands, Hydra override keys) are copied exactly: no capitalization change, no trailing period.
- No backward-compatibility shims. When `examples/analysis/plot_policy_cap_sweep.py` moves onto the library, its inline copy is deleted rather than kept as a fallback.
- `SIMRL_BACKEND=cpp` pinned for every simulation so all points come from one engine.
- Every simulation runs with `+log_exclude='instance,rollout_step'`. The shared filesystem quota is chronically full.
- Pinned regime for every point: trace `outputs/workload/qwen3_8b_dapo_38912_tp1/joint.jsonl`, `MAX_OUTPUT_LENGTH=38912`, `BATCH_SIZE=32`, `GROUP_SIZE=8`, `N_VERSIONS=12`, `N_INSTANCES=2`, `ROLLOUT_TP=1`, training TP2/PP3/DP1.

### Deviation from the spec, decided during planning

The spec names the reduction `srl/telemetry/reduce.py`. It goes in **`srl/diagnosis/staleness.py`** instead. `srl/diagnosis/` already exists for exactly this job ("Post-hoc diagnosis of a finished simulation. Reads only the JSONL telemetry streams a run already wrote"), and `srl/telemetry/` owns *writing* the schema. Putting a reader-reducer in the writer package would blur that boundary. Everything else in the spec stands.

---

## File Structure

| File | Responsibility |
|---|---|
| `srl/diagnosis/staleness.py` (create) | Reduce one run's `request` stream to `StalenessBreakdown`. Gate comparability. Pure. |
| `srl/diagnosis/__init__.py` (modify) | Re-export the two new public names. |
| `tests/test_diagnosis_staleness.py` (create) | Unit tests for the reduction and the gate, on handwritten rows. |
| `srl/visualizer/staleness_sweep.py` (create) | Stacked-bar figure plus the pure segment-height helpers it draws from. |
| `tests/test_staleness_sweep_plot.py` (create) | Numeric tests on the helpers, render test on the figure. |
| `srl/cli/staleness_plot.py` (create) | `srl-staleness-plot`: read a TSV manifest of points, reduce, gate, draw. |
| `pyproject.toml` (modify) | Register the console script. |
| `examples/analysis/plot_policy_cap_sweep.py` (modify) | Delete its inline reduction, import the library. |
| `examples/analysis/plot_verl_staleness_sweep.py` (modify) | Follow the renamed import surface. |
| `examples/simulate/staleness_throughput/common.sh` (create) | The pinned regime, one place. |
| `examples/simulate/staleness_throughput/sweep.sh` (create) | Shared runner: expand grid, run in parallel, gate, plot. |
| `examples/simulate/staleness_throughput/run_all.sh` (create) | All twelve presets. |
| `examples/simulate/staleness_throughput/README.md` (create) | Method, gates, the synchronous-degeneracy trap. |
| `examples/simulate/staleness_throughput/<preset>/grid.sh` (create x12) | Only that preset's knob, range, label, pins. |

---

## Task 1: The staleness reduction and its comparability gate

**Files:**
- Create: `srl/diagnosis/staleness.py`
- Modify: `srl/diagnosis/__init__.py`
- Test: `tests/test_diagnosis_staleness.py`

**Interfaces:**
- Consumes: `srl.telemetry.store.JsonlStore` (existing, `.iter(stream)` yields dicts).
- Produces: `StalenessBreakdown` frozen dataclass with fields `span: float`, `n_consumed: int`, `consumed_tokens: dict[int, int]`, `wasted_tokens: dict[str, int]`, `reprefill_tokens: int`; properties `max_staleness: int`, `total_consumed_tokens: int`, `total_wasted_tokens: int`, `throughput: float`; method `scg(s: int) -> float`. Module functions `staleness_breakdown(store: JsonlStore) -> StalenessBreakdown` and `check_same_consumption(points: Sequence[tuple[str, StalenessBreakdown]]) -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diagnosis_staleness.py`:

```python
"""The staleness reduction must account for every generated token exactly once."""

from __future__ import annotations

import pytest

from srl.diagnosis.staleness import (
    StalenessBreakdown,
    check_same_consumption,
    staleness_breakdown,
)
from srl.telemetry.store import JsonlStore

# Three consumed trajectories at two staleness levels, two discards with distinct
# reasons. Span is the largest `t`, which the last discard carries.
ROWS = [
    dict(event="consume", t=10.0, staleness_true=0, generated=100, reprefill_tokens=10),
    dict(event="consume", t=20.0, staleness_true=0, generated=200, reprefill_tokens=20),
    dict(event="consume", t=30.0, staleness_true=2, generated=300, reprefill_tokens=30),
    dict(event="drop", t=40.0, drop_reason="l3_stale", generated=50),
    dict(event="drop", t=50.0, drop_reason="l2_cancel", generated=25),
]


def _store(tmp_path, rows, name="logs"):
    """Write `rows` to a fresh `request` stream and hand back a reader."""
    root = tmp_path / name
    store = JsonlStore(root, overwrite=True)
    for row in rows:
        store.write("request", **row)
    store.close()
    return JsonlStore(root)


def test_consumed_tokens_bucket_by_true_staleness(tmp_path):
    bd = staleness_breakdown(_store(tmp_path, ROWS))
    assert bd.consumed_tokens == {0: 300, 2: 300}, f"Got {bd.consumed_tokens!r}."
    assert bd.n_consumed == 3, f"Got {bd.n_consumed!r}."
    assert bd.max_staleness == 2, f"Got {bd.max_staleness!r}."


def test_wasted_tokens_bucket_by_drop_reason(tmp_path):
    bd = staleness_breakdown(_store(tmp_path, ROWS))
    assert bd.wasted_tokens == {"l2_cancel": 25, "l3_stale": 50}, f"Got {bd.wasted_tokens!r}."
    assert bd.total_wasted_tokens == 75, f"Got {bd.total_wasted_tokens!r}."


def test_span_is_the_largest_timestamp(tmp_path):
    bd = staleness_breakdown(_store(tmp_path, ROWS))
    assert bd.span == 50.0, f"Got {bd.span!r}."


def test_throughput_counts_consumed_and_wasted(tmp_path):
    bd = staleness_breakdown(_store(tmp_path, ROWS))
    assert bd.total_consumed_tokens == 600, f"Got {bd.total_consumed_tokens!r}."
    assert bd.throughput == pytest.approx(675.0 / 50.0), f"Got {bd.throughput!r}."


def test_scg_is_cumulative_and_saturates(tmp_path):
    bd = staleness_breakdown(_store(tmp_path, ROWS))
    assert bd.scg(0) == pytest.approx(300.0 / 50.0), f"Got {bd.scg(0)!r}."
    assert bd.scg(1) == pytest.approx(300.0 / 50.0), f"Got {bd.scg(1)!r}."
    assert bd.scg(2) == pytest.approx(600.0 / 50.0), f"Got {bd.scg(2)!r}."
    assert bd.scg(9) == pytest.approx(bd.scg(2)), "SCG must not grow past the highest staleness."


def test_reprefill_tokens_are_reported_separately(tmp_path):
    bd = staleness_breakdown(_store(tmp_path, ROWS))
    assert bd.reprefill_tokens == 60, f"Got {bd.reprefill_tokens!r}."
    assert bd.total_consumed_tokens == 600, "Re-prefill must not be folded into consumed tokens."


def test_a_drop_without_a_reason_is_named_rather_than_dropped(tmp_path):
    rows = [ROWS[0], dict(event="drop", t=9.0, generated=7)]
    bd = staleness_breakdown(_store(tmp_path, rows))
    assert bd.wasted_tokens == {"unknown": 7}, f"Got {bd.wasted_tokens!r}."


def test_a_run_with_no_consumed_trajectories_is_refused(tmp_path):
    rows = [dict(event="drop", t=5.0, drop_reason="l3_stale", generated=10)]
    with pytest.raises(ValueError, match="no consumed trajectories"):
        staleness_breakdown(_store(tmp_path, rows))


def test_a_non_positive_span_is_refused(tmp_path):
    rows = [dict(event="consume", t=0.0, staleness_true=0, generated=100)]
    with pytest.raises(ValueError, match="non-positive span"):
        staleness_breakdown(_store(tmp_path, rows))


def _bd(n_consumed):
    return StalenessBreakdown(
        span=10.0,
        n_consumed=n_consumed,
        consumed_tokens={0: 10},
        wasted_tokens={},
        reprefill_tokens=0,
    )


def test_the_gate_accepts_an_equal_consumption_set():
    check_same_consumption([("a", _bd(64)), ("b", _bd(64))])


def test_the_gate_refuses_an_unequal_consumption_set():
    with pytest.raises(ValueError, match="consumed different trajectory counts"):
        check_same_consumption([("a", _bd(64)), ("b", _bd(32))])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
python -m pytest tests/test_diagnosis_staleness.py -q
```
Expected: collection error, `ModuleNotFoundError: No module named 'srl.diagnosis.staleness'`.

- [ ] **Step 3: Write the implementation**

Create `srl/diagnosis/staleness.py`:

```python
"""Which staleness level produced each part of a run's generation throughput.

Reduces the `request` stream into one stacked bar's worth of numbers: tokens
consumed at each observed `staleness_true`, tokens generated and then discarded
keyed by the reason, and the span they were produced in. Every segment over the
span sums to the fleet's full generation throughput, so tightening a staleness
bound shows up as tokens moving from a consumed level into a discard bucket
rather than disappearing from the accounting.

`span` is `max(t)` over all request rows, which is what `SimResult.sim_time` and
every existing sweep figure use, so a rate computed here is comparable with
those.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..telemetry.store import JsonlStore

# A discard row written before `drop_reason` existed, or by a path that supplies
# none, still carries tokens that were really spent.
_UNKNOWN_REASON = "unknown"


@dataclass(frozen=True)
class StalenessBreakdown:
    """One finished run, reduced to its staleness-by-throughput composition.

    Args:
        span (float): Largest timestamp in the request stream, in seconds. The
            denominator of every rate below.
        n_consumed (int): Trajectories that reached a training batch.
        consumed_tokens (dict[int, int]): `staleness_true` to generated tokens.
        wasted_tokens (dict[str, int]): `drop_reason` to generated tokens that
            never reached a training batch.
        reprefill_tokens (int): Tokens re-computed after a preemption or abort,
            over the consumed trajectories. Reported rather than stacked,
            because this is recomputation of tokens already counted once in
            `consumed_tokens` and adding it to the stack would double count.
    """

    span: float
    n_consumed: int
    consumed_tokens: dict[int, int]
    wasted_tokens: dict[str, int]
    reprefill_tokens: int

    @property
    def max_staleness(self) -> int:
        """Highest staleness any consumed trajectory reached."""
        return max(self.consumed_tokens) if self.consumed_tokens else 0

    @property
    def total_consumed_tokens(self) -> int:
        return sum(self.consumed_tokens.values())

    @property
    def total_wasted_tokens(self) -> int:
        return sum(self.wasted_tokens.values())

    @property
    def throughput(self) -> float:
        """Full generation throughput in tokens per second, trained on or not."""
        return (self.total_consumed_tokens + self.total_wasted_tokens) / self.span

    def scg(self, s: int) -> float:
        """Staleness-constrained goodput: consumed tokens at staleness <= `s`, per second.

        Defined for every `s`. Saturates at the consumed rate once nothing is
        excluded, so an `s` above this run's own maximum equals the value below it.
        """
        kept = sum(n for staleness, n in self.consumed_tokens.items() if staleness <= s)
        return kept / self.span


def staleness_breakdown(store: JsonlStore) -> StalenessBreakdown:
    """Reduce one run's `request` stream.

    Args:
        store (JsonlStore): A finished run's log store.

    Returns:
        StalenessBreakdown: The run's staleness composition.

    Raises:
        ValueError: If the stream holds no consumed trajectory with a recorded
            staleness, or if its span is not positive. Both make every rate
            meaningless, so they are refused rather than reported as zero.
    """
    span = 0.0
    consumed: dict[int, int] = {}
    wasted: dict[str, int] = {}
    n_consumed = 0
    reprefill = 0

    for row in store.iter("request"):
        span = max(span, float(row.get("t", 0.0) or 0.0))
        event = row.get("event")
        if event == "consume":
            staleness = row.get("staleness_true")
            if staleness is None:
                continue
            n_consumed += 1
            level = int(staleness)
            consumed[level] = consumed.get(level, 0) + int(row.get("generated", 0) or 0)
            reprefill += int(row.get("reprefill_tokens", 0) or 0)
        elif event == "drop":
            reason = row.get("drop_reason") or _UNKNOWN_REASON
            wasted[reason] = wasted.get(reason, 0) + int(row.get("generated", 0) or 0)

    if not n_consumed:
        raise ValueError(
            f"Run at {store.root!s} has no consumed trajectories with a recorded staleness."
        )
    if span <= 0.0:
        raise ValueError(f"Run at {store.root!s} has a non-positive span of {span!r}.")

    return StalenessBreakdown(
        span=span,
        n_consumed=n_consumed,
        consumed_tokens=consumed,
        wasted_tokens=wasted,
        reprefill_tokens=reprefill,
    )


def check_same_consumption(points: Sequence[tuple[str, StalenessBreakdown]]) -> None:
    """Refuse a comparison whose points did not finish the same training work.

    Throughput is a rate, so a point that ended early scores arbitrarily well.
    Consumed *tokens* may still differ between points, because a tighter bound
    resamples a different set of trajectories. That difference is real and is
    reported on the figure, so only the trajectory count is gated here.

    Args:
        points (Sequence[tuple[str, StalenessBreakdown]]): Labelled points.

    Raises:
        ValueError: If the consumed trajectory counts disagree.
    """
    counts = {bd.n_consumed for _, bd in points}
    if len(counts) > 1:
        detail = ", ".join(f"{label}: {bd.n_consumed}" for label, bd in points)
        raise ValueError(f"Points consumed different trajectory counts: {detail}.")
```

- [ ] **Step 4: Export the new names**

Modify `srl/diagnosis/__init__.py`. Add this import after the existing `from .pricer import SoloPricer` line, keeping the import block sorted by module name as ruff's isort rule requires:

```python
from .staleness import StalenessBreakdown, check_same_consumption, staleness_breakdown
```

The existing `__all__` is grouped by source module in import order, not alphabetised, so append a new group at the end of the list rather than interleaving:

```python
    "StalenessBreakdown",
    "check_same_consumption",
    "staleness_breakdown",
```

Also extend the module docstring's entry-point sentence, which currently names `request_phases`, `steps` and `bounds`, to mention `staleness_breakdown` for the staleness composition.

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_diagnosis_staleness.py -q
```
Expected: `12 passed`.

- [ ] **Step 6: Lint**

```bash
python -m ruff check srl/diagnosis/staleness.py srl/diagnosis/__init__.py tests/test_diagnosis_staleness.py
python -m ruff format --check srl/diagnosis/staleness.py || true
```
Expected: `All checks passed!` from `ruff check`.

- [ ] **Step 7: Commit**

```bash
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
git add srl/diagnosis/staleness.py srl/diagnosis/__init__.py tests/test_diagnosis_staleness.py
git commit -m "feat(diagnosis): reduce a run to its staleness-by-throughput composition

Consumed tokens keyed by staleness_true, discarded tokens keyed by
drop_reason, and the span, so every generated token is accounted for exactly
once and a tightened bound moves tokens between buckets instead of out of the
total. Re-prefill is reported beside the stack rather than in it, since those
tokens are recomputation of tokens already counted.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: The stacked-bar figure

**Files:**
- Create: `srl/visualizer/staleness_sweep.py`
- Test: `tests/test_staleness_sweep_plot.py`

**Interfaces:**
- Consumes: `StalenessBreakdown` from Task 1.
- Produces: type alias `Point = tuple[str, StalenessBreakdown]` and `Panel = tuple[str, Sequence[Point]]`. Functions `staleness_levels(panels) -> list[int]`, `waste_reasons(panels) -> list[str]`, `stack_segments(points, levels, reasons) -> dict[str, list[float]]`, and `plot_staleness_throughput(panels, primary_label, title, out_path, footnote="") -> Path`.

The numeric helpers are separate from the drawing so tests assert exact heights without parsing a PNG, matching how `SimPlotter._scg_curve_data` is already factored.

- [ ] **Step 1: Write the failing test**

Create `tests/test_staleness_sweep_plot.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_staleness_sweep_plot.py -q
```
Expected: collection error, `ModuleNotFoundError: No module named 'srl.visualizer.staleness_sweep'`.

- [ ] **Step 3: Write the implementation**

Create `srl/visualizer/staleness_sweep.py`:

```python
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
from .plots import _FIG_DPI  # noqa: E402

Point = tuple[str, StalenessBreakdown]
Panel = tuple[str, Sequence[Point]]

# Discards are drawn outside the categorical palette: a neutral fill plus a
# hatch, so a waste segment can never be mistaken for a staleness level.
_WASTE_FILL = "#d9d7cf"
_WASTE_EDGE = "#6f6d66"
_WASTE_HATCHES = ("//", "xx", "..", "\\\\", "++")
_BOUNDARY = "#104281"
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
        ValueError: If no panel carries any point.
    """
    out_path = Path(out_path)
    if not any(points for _, points in panels):
        raise ValueError("Cannot draw a staleness figure with no points.")

    levels = staleness_levels(panels)
    reasons = waste_reasons(panels)
    colours = plt.cm.Blues(np.linspace(0.35, 0.95, max(len(levels), 1)))

    n_bars = sum(len(points) for _, points in panels)
    width = max(7.0, 1.1 * n_bars + 2.0 * len(panels))
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 5.6), sharey=True, squeeze=False)
    axes = axes[0]

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

        # The consumed ceiling is the effective training throughput, which a bar
        # height alone hides once a discard segment sits on top of it.
        edges = np.append(xs - _BAR_WIDTH / 2, xs[-1] + _BAR_WIDTH / 2)
        ax.step(edges, np.append(consumed_top, consumed_top[-1]), where="post", color=_BOUNDARY, linewidth=1.6)

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

    axes[0].set_ylabel("Generation throughput  [tokens / s]")
    axes[0].set_ylim(bottom=0)

    # Gathered across panels, since a segment may be absent from the first one.
    handles: dict[str, object] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    handles["effective training throughput"] = plt.Line2D([], [], color=_BOUNDARY, linewidth=1.6)
    fig.legend(
        list(handles.values()),
        list(handles.keys()),
        loc="center right",
        frameon=False,
        title="bar segments",
    )

    fig.suptitle(title)
    if footnote:
        fig.text(0.01, 0.01, footnote, fontsize=9, color="#555555", va="bottom")
    fig.tight_layout(rect=(0.0, 0.04 if footnote else 0.0, 0.82, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_FIG_DPI)
    plt.close(fig)
    return out_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_staleness_sweep_plot.py -q
```
Expected: `7 passed`.

- [ ] **Step 5: Eyeball the figure once**

The tests prove the numbers and that a file appears. They cannot prove it is readable. Render the two-panel case and look at it:

```bash
python - <<'PY'
from pathlib import Path
from srl.diagnosis.staleness import StalenessBreakdown
from srl.visualizer.staleness_sweep import plot_staleness_throughput

loose = StalenessBreakdown(100.0, 8, {0: 1000, 1: 2000, 3: 3000}, {}, 0)
tight = StalenessBreakdown(200.0, 8, {0: 1000, 1: 1000}, {"l3_stale": 4000}, 0)
print(plot_staleness_throughput(
    [("thr = 1", [("1", tight), ("2", loose)]), ("thr = 5", [("1", loose), ("2", loose)])],
    primary_label="num_warmup_batches (batches)",
    title="smoke",
    out_path=Path("/tmp/staleness_smoke.png"),
    footnote="training cost outside its fit domain",
))
PY
```
Then Read `/tmp/staleness_smoke.png`. Confirm: legend is not clipped, span annotations do not collide, the boundary line sits at the top of the blue stack, the hatched segment is distinguishable from every blue. If any of these fail, adjust `width`, the `rect` of `tight_layout`, or `_BAR_WIDTH` and re-render.

- [ ] **Step 6: Lint and commit**

```bash
python -m ruff check srl/visualizer/staleness_sweep.py tests/test_staleness_sweep_plot.py
git add srl/visualizer/staleness_sweep.py tests/test_staleness_sweep_plot.py
git commit -m "feat(visualizer): stacked staleness-throughput bars

Bar height is full generation throughput, lower segments are tokens that
reached training keyed by staleness, hatched upper segments are discards keyed
by reason, and a line marks the boundary between them so converting useful
work into waste is visible at constant bar height.

Segment heights are computed by a pure helper so tests assert exact values
without parsing a PNG.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: The `srl-staleness-plot` CLI

**Files:**
- Create: `srl/cli/staleness_plot.py`
- Modify: `pyproject.toml`
- Test: `tests/test_staleness_plot_cli.py`

**Interfaces:**
- Consumes: `staleness_breakdown`, `check_same_consumption` (Task 1), `plot_staleness_throughput` (Task 2).
- Produces: `load_manifest(path: Path) -> list[Panel]` and `main() -> None`. Console script `srl-staleness-plot`.

The manifest is a three-column TSV, `panel<TAB>label<TAB>run_dir`, because `sweep.sh` writes it from bash and bash does not write JSON cleanly. An empty panel column means a single unnamed panel. Row order is bar order.

- [ ] **Step 1: Write the failing test**

Create `tests/test_staleness_plot_cli.py`:

```python
"""The manifest reader must preserve bar order and group panels."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from srl.cli.staleness_plot import load_manifest
from srl.telemetry.store import JsonlStore

ROWS = [
    dict(event="consume", t=10.0, staleness_true=0, generated=100),
    dict(event="consume", t=20.0, staleness_true=1, generated=100),
]


def _run(tmp_path, name):
    """Create a minimal finished run directory with a `logs/request.jsonl`."""
    store = JsonlStore(tmp_path / name / "logs", overwrite=True)
    for row in ROWS:
        store.write("request", **row)
    store.close()
    return tmp_path / name


def test_rows_keep_their_order_within_a_panel(tmp_path):
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b")
    manifest = tmp_path / "points.tsv"
    manifest.write_text(f"p1\t2\t{b}\np1\t1\t{a}\n", encoding="utf-8")
    panels = load_manifest(manifest)
    assert [name for name, _ in panels] == ["p1"]
    assert [label for label, _ in panels[0][1]] == ["2", "1"], "Bar order must follow row order."


def test_panels_are_grouped_in_first_appearance_order(tmp_path):
    a = _run(tmp_path, "a")
    manifest = tmp_path / "points.tsv"
    manifest.write_text(f"second\t1\t{a}\nfirst\t1\t{a}\nsecond\t2\t{a}\n", encoding="utf-8")
    panels = load_manifest(manifest)
    assert [name for name, _ in panels] == ["second", "first"]
    assert len(panels[0][1]) == 2
    assert len(panels[1][1]) == 1


def test_an_empty_panel_column_yields_one_unnamed_panel(tmp_path):
    a = _run(tmp_path, "a")
    manifest = tmp_path / "points.tsv"
    manifest.write_text(f"\t0\t{a}\n", encoding="utf-8")
    panels = load_manifest(manifest)
    assert [name for name, _ in panels] == [""]


def test_a_missing_run_directory_names_itself(tmp_path):
    manifest = tmp_path / "points.tsv"
    manifest.write_text(f"\t0\t{tmp_path / 'absent'}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="absent"):
        load_manifest(manifest)


def test_a_malformed_row_names_its_line_number(tmp_path):
    manifest = tmp_path / "points.tsv"
    manifest.write_text("only-one-column\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="line 1"):
        load_manifest(manifest)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_staleness_plot_cli.py -q
```
Expected: collection error, `ModuleNotFoundError: No module named 'srl.cli.staleness_plot'`.

- [ ] **Step 3: Write the implementation**

Create `srl/cli/staleness_plot.py`:

```python
"""Standalone CLI: draw a staleness-throughput sweep from finished runs.

Never re-simulates. Takes a TSV manifest of `panel<TAB>label<TAB>run_dir` rows,
reduces each run's `request` stream, enforces that every point finished the same
training work, and writes one figure. The manifest exists because the sweep
runner is bash, which writes columns far more cleanly than it writes JSON.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .._logging import setup_logging
from ..diagnosis.staleness import check_same_consumption, staleness_breakdown
from ..telemetry.store import JsonlStore
from ..visualizer.staleness_sweep import Panel, plot_staleness_throughput


def load_manifest(path: Path, logs_subdir: str = "logs") -> list[Panel]:
    """Read a point manifest into panels, preserving row order.

    Args:
        path (Path): TSV with one `panel`, `label`, `run_dir` row per bar. An
            empty panel column puts every row in one unnamed panel.
        logs_subdir (str): Name of the log directory under each `run_dir`.

    Returns:
        list[Panel]: Panels in first-appearance order, bars in row order.

    Raises:
        SystemExit: If a row is malformed or a run directory has no log stream.
            This is operator error in a generated file, so it reports the
            offending line rather than raising something a caller would catch.
    """
    grouped: dict[str, list] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise SystemExit(f"Malformed manifest {path!s} line {number}: expected 3 tab-separated fields.")
        panel, label, run_dir = (field.strip() for field in fields)
        logs = Path(run_dir) / logs_subdir
        if not (logs / "request.jsonl").is_file():
            raise SystemExit(f"No request stream under {run_dir!s} (manifest line {number}).")
        grouped.setdefault(panel, []).append((label, staleness_breakdown(JsonlStore(logs))))
    if not grouped:
        raise SystemExit(f"Manifest {path!s} holds no points.")
    return [(panel, points) for panel, points in grouped.items()]


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="srl-staleness-plot",
        description="Draw a staleness-throughput sweep figure from finished simulation runs.",
    )
    parser.add_argument("manifest", type=Path, help="TSV of panel/label/run_dir rows, one per bar.")
    parser.add_argument("--primary-label", required=True, help="X axis label, naming the knob and its unit.")
    parser.add_argument("--title", required=True, help="Figure title.")
    parser.add_argument("--out", type=Path, required=True, help="Destination PNG.")
    parser.add_argument("--footnote", default="", help="Caveat text drawn bottom left.")
    parser.add_argument(
        "--logs-subdir", default="logs", help="Name of the logs subdirectory under each run_dir."
    )
    parser.add_argument(
        "--allow-unequal-consumption",
        action="store_true",
        help="Draw even when points finished different amounts of training work. Off by default: "
        "throughput is a rate, so a point that ended early scores arbitrarily well.",
    )
    args = parser.parse_args()

    panels = load_manifest(args.manifest, logs_subdir=args.logs_subdir)
    every_point = [point for _, points in panels for point in points]
    if not args.allow_unequal_consumption:
        check_same_consumption(every_point)

    path = plot_staleness_throughput(
        panels,
        primary_label=args.primary_label,
        title=args.title,
        out_path=args.out,
        footnote=args.footnote,
    )

    header = f"  {'panel':<12}{'point':>8}{'span':>10}{'tok/s':>9}{'waste%':>8}  staleness"
    print(header)
    for panel, points in panels:
        for label, bd in points:
            share = 100.0 * bd.total_wasted_tokens / max(1, bd.total_consumed_tokens + bd.total_wasted_tokens)
            spread = " ".join(f"S{s}:{n / 1e6:.1f}M" for s, n in sorted(bd.consumed_tokens.items()))
            print(f"  {panel:<12}{label:>8}{bd.span:>10.0f}{bd.throughput:>9.0f}{share:>8.1f}  {spread}")
    print(f"\n  Every point consumed {every_point[0][1].n_consumed} trajectories.")
    print(f"  {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Register the console script**

Modify `pyproject.toml`. In `[project.scripts]`, after the `srl-psrl-plot` line, add:

```toml
srl-staleness-plot = "srl.cli.staleness_plot:main"
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_staleness_plot_cli.py -q
```
Expected: `5 passed`.

- [ ] **Step 6: Verify the entry point resolves end to end**

```bash
python -m pip install -e . --no-deps -q
srl-staleness-plot --help
```
Expected: the usage block, exit 0.

- [ ] **Step 7: Lint and commit**

```bash
python -m ruff check srl/cli/staleness_plot.py tests/test_staleness_plot_cli.py
git add srl/cli/staleness_plot.py tests/test_staleness_plot_cli.py pyproject.toml
git commit -m "feat(cli): add srl-staleness-plot

Reads a TSV manifest of panel/label/run_dir rows so the bash sweep runner can
generate it without writing JSON, reduces each run, gates on an equal
consumption set by default, and prints the table beside the figure.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Put the analysis scripts on the shared reduction

**Files:**
- Modify: `examples/analysis/plot_policy_cap_sweep.py`
- Modify: `examples/analysis/plot_verl_staleness_sweep.py`

`plot_policy_cap_sweep.py` carries its own `RunSummary` and `load_run`, which compute staleness-by-token exactly as Task 1 does. Two definitions of the study's central metric is how they drift. The inline copy is deleted, not kept as a fallback.

**Interfaces:**
- Consumes: `staleness_breakdown`, `check_same_consumption` from Task 1.
- Produces: `load_run(run_dir: Path, policy: str, cap: int) -> RunSummary` keeps its name and signature so `plot_verl_staleness_sweep.py` keeps working, but `RunSummary` becomes a thin wrapper holding `policy`, `cap`, and a `breakdown`, forwarding `span`, `n_consumed`, `scg`, `throughput`, and `max_staleness_seen`.

- [ ] **Step 1: Capture the current output as the reference**

Before changing anything, record what the existing script prints for a finished sweep, so the refactor can be shown to change nothing.

```bash
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
ls examples/analysis/verl_staleness_sweep/
```

If a finished sweep exists under `outputs/simulate/verl_staleness_sweep/`, run the existing plotter and save its stdout to `/tmp/cap_sweep_before.txt`. If none exists, synthesise two runs instead:

```bash
python - <<'PY'
from pathlib import Path
from srl.telemetry.store import JsonlStore

for name, span in (("thr1.0", 50.0), ("thr2.0", 40.0)):
    root = Path("/tmp/refactor_ref") / name / "logs"
    store = JsonlStore(root, overwrite=True)
    store.write("request", event="consume", t=10.0, staleness_true=0, generated=100)
    store.write("request", event="consume", t=span, staleness_true=1, generated=300)
    store.close()
print("written")
PY
cd examples/analysis && python - <<'PY' > /tmp/cap_sweep_before.txt
from pathlib import Path
from plot_policy_cap_sweep import load_run
for name in ("thr1.0", "thr2.0"):
    r = load_run(Path("/tmp/refactor_ref") / name, "verl", cap=0)
    print(name, r.span, r.n_consumed, r.scg(0), r.scg(1), r.throughput, r.max_staleness_seen)
PY
cat /tmp/cap_sweep_before.txt
```

- [ ] **Step 2: Replace the inline reduction**

In `examples/analysis/plot_policy_cap_sweep.py`, delete the `RunSummary` dataclass body, the whole `load_run` function, and the whole `check_comparable` function (lines 26 to 105 in the current file), and replace them with:

```python
from srl.diagnosis.staleness import StalenessBreakdown, check_same_consumption, staleness_breakdown
from srl.telemetry.store import JsonlStore


@dataclass(frozen=True)
class RunSummary:
    """One finished simulation, labelled by the sweep axis it came from.

    The staleness arithmetic lives in `srl.diagnosis.staleness` so this figure
    and the per-preset sweep figures cannot disagree about what SCG means.
    """

    policy: str
    cap: int
    breakdown: StalenessBreakdown

    @property
    def span(self) -> float:
        return self.breakdown.span

    @property
    def n_consumed(self) -> int:
        return self.breakdown.n_consumed

    @property
    def max_staleness_seen(self) -> int:
        return self.breakdown.max_staleness

    @property
    def throughput(self) -> float:
        return self.breakdown.throughput

    def scg(self, s: int) -> float:
        return self.breakdown.scg(s)


def load_run(run_dir: Path, policy: str, cap: int) -> RunSummary:
    """Reduce one run's request telemetry to a `RunSummary`.

    Raises:
        FileNotFoundError: If the run has no request stream.
        ValueError: If it has no consume rows or a non-positive span.
    """
    logs = run_dir / "logs"
    if not (logs / "request.jsonl").is_file():
        raise FileNotFoundError(f"No telemetry at {logs / 'request.jsonl'}.")
    return RunSummary(policy=policy, cap=cap, breakdown=staleness_breakdown(JsonlStore(logs)))


def check_comparable(runs: list[RunSummary]) -> None:
    """Refuse to plot runs that did not consume the same amount of work."""
    check_same_consumption([(f"{r.policy}/cap{r.cap}", r.breakdown) for r in runs])
```

Keep the existing `json` import only if something else in the file still uses it. Run `ruff check` in Step 4 to catch an import that is now unused.

- [ ] **Step 3: Note the one behaviour difference**

The old `RunSummary.throughput` divided only consumed tokens by span. The new one divides consumed *plus discarded*. That is the intended definition from the spec, and `throughput` is not used by either figure in this directory (both call `scg`), so no plot changes. Confirm that claim rather than assuming it:

```bash
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
grep -n "\.throughput" examples/analysis/*.py
```
Expected: no matches. If there are matches, the call sites must be changed to `scg(r.max_staleness_seen)` to keep their old meaning.

- [ ] **Step 4: Verify the refactor changed nothing**

```bash
cd examples/analysis && python - <<'PY' > /tmp/cap_sweep_after.txt
from pathlib import Path
from plot_policy_cap_sweep import load_run
for name in ("thr1.0", "thr2.0"):
    r = load_run(Path("/tmp/refactor_ref") / name, "verl", cap=0)
    print(name, r.span, r.n_consumed, r.scg(0), r.scg(1), r.throughput, r.max_staleness_seen)
PY
diff /tmp/cap_sweep_before.txt /tmp/cap_sweep_after.txt && echo "IDENTICAL"
```
Expected: `IDENTICAL`. The `throughput` column is the one allowed to differ, and in this fixture there are no discards so it must not. If `diff` reports only that column and the fixture has discards, that is the intended change.

- [ ] **Step 5: Lint and run the whole suite**

```bash
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
python -m ruff check examples/analysis/plot_policy_cap_sweep.py examples/analysis/plot_verl_staleness_sweep.py
python -m pytest tests/ -q -x
```
Expected: ruff passes, suite passes with no new failures.

- [ ] **Step 6: Commit**

```bash
git add examples/analysis/plot_policy_cap_sweep.py examples/analysis/plot_verl_staleness_sweep.py
git commit -m "refactor(analysis): one definition of staleness-by-token

plot_policy_cap_sweep carried its own copy of the reduction that
srl.diagnosis.staleness now owns. The copy is deleted rather than kept behind
a fallback, so the cap-sweep figures and the per-preset sweep figures cannot
drift apart on what SCG means.

RunSummary keeps its name and forwarding properties, so
plot_verl_staleness_sweep is unchanged. Its throughput property now counts
discarded tokens too, per the sweep spec; no figure in this directory reads it.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: The sweep harness, proven end to end on one preset

**Files:**
- Create: `examples/simulate/staleness_throughput/common.sh`
- Create: `examples/simulate/staleness_throughput/sweep.sh`
- Create: `examples/simulate/staleness_throughput/psrl/grid.sh`

**Interfaces:**
- Consumes: `examples/simulate/psrl_aligned.sh` (existing; appends to `EXTRA_OVERRIDES` rather than assigning it, so a caller's overrides survive). `srl-staleness-plot` from Task 3.
- Produces: a `grid.sh` contract that Task 6 fills in eleven more times. A grid declares exactly these shell variables: `PRESET` (string), `PRIMARY_KNOB` (Hydra key), `PRIMARY_VALUES` (array), `PRIMARY_LABEL` (string), `PINS` (array of `key=value`, may be empty), and optionally `SECONDARY_KNOB` (Hydra key) with `SECONDARY_VALUES` (array).

- [ ] **Step 1: Write the pinned regime**

Create `examples/simulate/staleness_throughput/common.sh`:

```bash
#!/usr/bin/env bash
# The one regime every preset's sweep is measured in.
#
# Sourced by sweep.sh before the preset's grid.sh. Every value is overridable
# from the environment, so a caller can move the whole study to another trace or
# another GPU split without editing twelve grids.
#
# The split and mesh are the ones examples/simulate/psrl_aligned.sh transcribes
# from the real PSRL run, so the cost models are asked for predictions in the
# configuration they were matched against.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ROOT_DIR

# Pinned rather than `auto`, so every point in a sweep comes from one engine.
export SIMRL_BACKEND="${SIMRL_BACKEND:-cpp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Near-untruncated DAPO lengths for Qwen3-8B. The dispersion is what generates
# staleness here, so a trace capped lower changes what is being measured.
export TRACE="${TRACE:-${ROOT_DIR}/outputs/workload/qwen3_8b_dapo_38912_tp1/joint.jsonl}"
export MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-38912}"

export BATCH_SIZE="${BATCH_SIZE:-32}"
export GROUP_SIZE="${GROUP_SIZE:-8}"
# Staleness cannot exceed the number of versions a run contains, so a short run
# reports its own length rather than the mechanism.
export N_VERSIONS="${N_VERSIONS:-12}"

export N_INSTANCES="${N_INSTANCES:-2}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export TRAINING_TP="${TRAINING_TP:-2}"
export TRAINING_PP="${TRAINING_PP:-3}"
export TRAINING_DP="${TRAINING_DP:-1}"

RUN_BASE="${RUN_BASE:-${ROOT_DIR}/outputs/simulate/staleness_throughput}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
REUSE="${REUSE:-true}"

# At cap 38912 an unsplittable sequence forms a microbatch wider than the
# training sweep covers, and the microbatch count exceeds its pipeline_slots
# range. Every figure carries this, so a reader never takes the training
# component for a measured number.
FOOTNOTE="${FOOTNOTE:-Training cost extrapolated beyond its fit domain (tokens, pipeline_slots). Rollout and staleness are in domain.}"
```

- [ ] **Step 2: Write the shared runner**

Create `examples/simulate/staleness_throughput/sweep.sh`:

```bash
#!/usr/bin/env bash
# Sweep one preset's staleness knob and draw its throughput composition.
#
# Usage:  bash sweep.sh <preset-dir>            e.g. bash sweep.sh psrl
#
# Reads the preset's grid.sh, runs every point in parallel against the regime in
# common.sh, refuses the comparison if the points did not finish the same
# training work, and writes one figure.
#
# ENV  RUN_BASE      where run directories go
#      MAX_PARALLEL  concurrent simulations
#      REUSE         true skips a point whose request stream already exists
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PRESET_DIR="${1:?Usage: sweep.sh <preset-dir>}"
GRID="${SCRIPT_DIR}/${PRESET_DIR%/}/grid.sh"
[[ -f "${GRID}" ]] || { echo "No grid at ${GRID}" >&2; exit 1; }

# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
SECONDARY_KNOB=""
SECONDARY_VALUES=()
PINS=()
# shellcheck source=/dev/null
source "${GRID}"

OUT_DIR="${SCRIPT_DIR}/${PRESET_DIR%/}/figures"
RUNS="${RUN_BASE}/${PRESET}"
mkdir -p "${OUT_DIR}" "${RUNS}"

# A rebuild of the engine mid-sweep kills running simulations with a native
# signal that reads like an engine bug, and silently mixes binaries when it does
# not, so the binary's identity is part of the result.
SO="$(echo "${ROOT_DIR}"/srl/_engine_cpp*.so)"
SO_MD5_BEFORE="$(md5sum "${SO}" | cut -d' ' -f1)"

# One panel per secondary value, or a single unnamed panel when there is none.
if ((${#SECONDARY_VALUES[@]} == 0)); then
    PANEL_KEYS=("")
else
    PANEL_KEYS=("${SECONDARY_VALUES[@]}")
fi

n_points=$(( ${#PRIMARY_VALUES[@]} * ${#PANEL_KEYS[@]} ))
echo "Sweeping ${PRESET}: ${PRIMARY_KNOB} = ${PRIMARY_VALUES[*]}"
if [[ -n "${SECONDARY_KNOB}" ]]; then
    echo "  panels: ${SECONDARY_KNOB} = ${SECONDARY_VALUES[*]}"
fi
((${#PINS[@]} == 0)) || echo "  pinned: ${PINS[*]}"
echo "  ${n_points} points, ${N_VERSIONS} versions, backend ${SIMRL_BACKEND}, up to ${MAX_PARALLEL} at once"

run_point () {  # $1=run_dir  $2=overrides
    local run_dir="$1" overrides="$2"
    mkdir -p "${run_dir}"
    (
        export OUTPUT_DIR="${run_dir}"
        export POLICY="${PRESET}"
        export EXTRA_OVERRIDES="${overrides} +log_exclude='instance,rollout_step'"
        bash "${ROOT_DIR}/examples/simulate/psrl_aligned.sh" > "${run_dir}/sweep.log" 2>&1
    )
    # Only request.jsonl is read. The rest is dead weight on a full quota.
    rm -rf "${run_dir}/logs/instance" "${run_dir}/logs/rollout_step.jsonl"
}

MANIFEST="${RUNS}/points.tsv"
: > "${MANIFEST}"
pids=()
labels=()

for panel in "${PANEL_KEYS[@]}"; do
    for value in "${PRIMARY_VALUES[@]}"; do
        if [[ -n "${panel}" ]]; then
            run_dir="${RUNS}/s${panel}_p${value}"
            overrides="${PRIMARY_KNOB}=${value} ${SECONDARY_KNOB}=${panel}"
            panel_name="${SECONDARY_KNOB##*.} = ${panel}"
        else
            run_dir="${RUNS}/p${value}"
            overrides="${PRIMARY_KNOB}=${value}"
            panel_name=""
        fi
        for pin in ${PINS[@]+"${PINS[@]}"}; do
            overrides="${overrides} ${pin}"
        done
        printf '%s\t%s\t%s\n' "${panel_name}" "${value}" "${run_dir}" >> "${MANIFEST}"

        if [[ "${REUSE}" == "true" && -f "${run_dir}/logs/request.jsonl" ]]; then
            echo "  [reuse] ${run_dir##*/}"
            continue
        fi
        while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n; done
        echo "  [run]   ${run_dir##*/}"
        run_point "${run_dir}" "${overrides}" &
        pids+=("$!")
        labels+=("${run_dir##*/}")
    done
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[index]}"; then
        echo "  FAILED ${labels[index]}, see ${RUNS}/${labels[index]}/sweep.log" >&2
        failed=1
    fi
done

if [[ "$(md5sum "${SO}" | cut -d' ' -f1)" != "${SO_MD5_BEFORE}" ]]; then
    echo "  ${SO} changed during the sweep. Points may come from different builds." >&2
    failed=1
fi
(( failed == 0 )) || exit 1

echo ""
srl-staleness-plot "${MANIFEST}" \
    --primary-label "${PRIMARY_LABEL}" \
    --title "${PRESET}: generation throughput by staleness of the data it produced" \
    --out "${OUT_DIR}/staleness_throughput.png" \
    --footnote "${FOOTNOTE}"
```

- [ ] **Step 3: Write the first grid**

Create `examples/simulate/staleness_throughput/psrl/grid.sh`:

```bash
# psrl: a Reserve/Occupy/Consume inventory whose window is `staleness`.
#
# The knob reaches the policy through SimConfig.staleness_horizon rather than a
# constructor argument, and the same value widens the ingress window, so setting
# the preset field is all that is needed.
PRESET=psrl
PRIMARY_KNOB=policy.staleness
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="psrl staleness (versions)"
PINS=()

# SECONDARY_KNOB=policy.proactive_filter
# SECONDARY_VALUES=(false true)
```

- [ ] **Step 4: Smoke the harness on two points**

Prove the plumbing before spending the full grid. Three versions and two values finish in about a minute.

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
N_VERSIONS=3 \
  RUN_BASE=/tmp/staleness_smoke_runs \
  bash -c 'source examples/simulate/staleness_throughput/common.sh; PRIMARY_VALUES=(1 2); bash examples/simulate/staleness_throughput/sweep.sh psrl'
```

That invocation cannot override the array, since `sweep.sh` sources the grid after `common.sh`. Use a temporary grid instead:

```bash
mkdir -p examples/simulate/staleness_throughput/_smoke
sed 's/^PRIMARY_VALUES=.*/PRIMARY_VALUES=(1 2)/' \
    examples/simulate/staleness_throughput/psrl/grid.sh \
    > examples/simulate/staleness_throughput/_smoke/grid.sh
N_VERSIONS=3 RUN_BASE=/tmp/staleness_smoke_runs \
    bash examples/simulate/staleness_throughput/sweep.sh _smoke
```

Expected: two `[run]` lines, the printed table with two rows showing equal trajectory counts, and a figure path. Then Read `examples/simulate/staleness_throughput/_smoke/figures/staleness_throughput.png` and confirm it is legible.

- [ ] **Step 5: Verify the gate actually fires**

A gate that never trips is not a gate. Force unequal consumption by giving one point fewer versions, and confirm the plot is refused:

```bash
rm -rf /tmp/gate_check && mkdir -p /tmp/gate_check
N_VERSIONS=3 RUN_BASE=/tmp/gate_check bash examples/simulate/staleness_throughput/sweep.sh _smoke > /dev/null 2>&1
N_VERSIONS=4 RUN_BASE=/tmp/gate_check REUSE=false bash -c '
  sed "s/^PRIMARY_VALUES=.*/PRIMARY_VALUES=(2)/" examples/simulate/staleness_throughput/psrl/grid.sh \
    > examples/simulate/staleness_throughput/_smoke/grid.sh
  bash examples/simulate/staleness_throughput/sweep.sh _smoke' > /dev/null 2>&1 || true
printf 'a\t1\t/tmp/gate_check/psrl/p1\na\t2\t/tmp/gate_check/psrl/p2\n' > /tmp/gate_check/mixed.tsv
srl-staleness-plot /tmp/gate_check/mixed.tsv --primary-label k --title t --out /tmp/gate_check/x.png
```
Expected: exits non-zero with `ValueError: Points consumed different trajectory counts:`.

- [ ] **Step 6: Clean up the smoke artifacts and commit**

```bash
rm -rf examples/simulate/staleness_throughput/_smoke /tmp/staleness_smoke_runs /tmp/gate_check
git add examples/simulate/staleness_throughput/common.sh \
        examples/simulate/staleness_throughput/sweep.sh \
        examples/simulate/staleness_throughput/psrl/grid.sh
git commit -m "feat(simulate): shared staleness-throughput sweep runner

common.sh holds the one regime, sweep.sh expands a preset's grid and runs it in
parallel against psrl_aligned.sh, and a grid.sh declares only that preset's
knob, range, label and mandatory pins. The engine binary's md5 is checked
across the sweep, because a rebuild mid-run mixes binaries between points.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: The remaining eleven grids, the all-preset driver, and the README

**Files:**
- Create: `examples/simulate/staleness_throughput/{verl_v1,verl,areal,roll,slime,skyrl,openrlhf,dressage,nemo_rl,trl,prime_rl}/grid.sh`
- Create: `examples/simulate/staleness_throughput/run_all.sh`
- Create: `examples/simulate/staleness_throughput/README.md`

**Interfaces:**
- Consumes: the `grid.sh` contract from Task 5.
- Produces: nothing other tasks read.

Every range below was validated by running its endpoints. Do not change a floor without re-running that endpoint.

- [ ] **Step 1: Write the eight six-point grids**

`verl/grid.sh`:
```bash
# verl legacy: `staleness_threshold` is a sample-overage RATIO, not a version
# count. It admits at most int(B * (1 + threshold)) groups between syncs, so the
# axis label must not claim versions even though it shares the 0..5 range.
PRESET=verl
PRIMARY_KNOB=policy.staleness_threshold
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="verl staleness_threshold (sample-overage ratio, not versions)"
PINS=()

# SECONDARY_KNOB=policy.require_batches
# SECONDARY_VALUES=(1 2)
```

`areal/grid.sh`:
```bash
# AReaL: a cumulative token bucket whose width is `max_head_offpolicyness`,
# reaching the policy through SimConfig.staleness_horizon.
PRESET=areal
PRIMARY_KNOB=policy.max_head_offpolicyness
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="areal max_head_offpolicyness (versions)"
PINS=()
```

`roll/grid.sh`:
```bash
# ROLL: `async_generation_ratio` is a ratio, but horizon takes its ceiling, so it
# acts as the version window for both the admission budget and the L2 GC.
PRESET=roll
PRIMARY_KNOB=policy.async_generation_ratio
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="roll async_generation_ratio (ceiling is the version window)"
PINS=()
```

`skyrl/grid.sh`:
```bash
# SkyRL: `max_staleness_steps` drives the admission quota and a violation
# counter and never discards, so this axis is expected to move span without
# bounding the composition. A flat figure here is the measurement.
PRESET=skyrl
PRIMARY_KNOB=policy.max_staleness_steps
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="skyrl max_staleness_steps (versions, warn-only)"
PINS=()
```

`trl/grid.sh`:
```bash
# TRL: the most direct bound in the study. `max_staleness` compares the
# generating version against the training version at dequeue.
PRESET=trl
PRIMARY_KNOB=policy.max_staleness
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="trl max_staleness (versions)"
PINS=()
```

`prime_rl/grid.sh`:
```bash
# prime-rl: `max_off_policy_steps` is one lower bound shared by the L2 cancel and
# the L3 sweep, so a change moves both at once.
PRESET=prime_rl
PRIMARY_KNOB=policy.max_off_policy_steps
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="prime_rl max_off_policy_steps (versions)"
PINS=()
```

`nemo_rl/grid.sh`:
```bash
# NeMo-RL: `sampler` must be pinned to windowed. Under the default in_order,
# horizon reads max_lookahead_versions instead and the L2 cancel never runs, so
# sweeping max_staleness_versions would move nothing.
PRESET=nemo_rl
PRIMARY_KNOB=policy.max_staleness_versions
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="nemo_rl max_staleness_versions (versions, sampler=windowed)"
PINS=(policy.sampler=windowed)

# SECONDARY_KNOB=policy.sample_freshest_first
# SECONDARY_VALUES=(false true)
```

- [ ] **Step 2: Write the three five-point grids**

Each floor is 1 because 0 is rejected or means the opposite of a tight bound.

`slime/grid.sh`:
```bash
# slime: no version comparison anywhere. The maximum staleness is identically
# `update_weights_interval`, which is a loop interval, so 0 is not a setting.
PRESET=slime
PRIMARY_KNOB=policy.update_weights_interval
PRIMARY_VALUES=(1 2 3 4 5)
PRIMARY_LABEL="slime update_weights_interval (training steps per sync)"
PINS=()

# SECONDARY_KNOB=policy.over_sampling_ratio
# SECONDARY_VALUES=(1.0 1.5)
```

`openrlhf/grid.sh`:
```bash
# OpenRLHF: a permit pool counted in whole batch lumps, not per-sample age.
# partial_rollout_enable must be on or the policy asserts for any queue size
# above 1. Turning it on also makes the pause mode in_place, which is the same
# switch in the real system rather than a separate choice.
PRESET=openrlhf
PRIMARY_KNOB=policy.async_queue_size
PRIMARY_VALUES=(1 2 3 4 5)
PRIMARY_LABEL="openrlhf async_queue_size (in-flight batch lumps)"
PINS=(policy.partial_rollout_enable=true)
```

`dressage/grid.sh`:
```bash
# Dressage: `keep_versions` counts distinct observed versions, keyed on each
# group's last trainable token. Zero means the filter is OFF, which is the loose
# end of the range rather than the tight one, so this axis starts at 1.
PRESET=dressage
PRIMARY_KNOB=policy.keep_versions
PRIMARY_VALUES=(1 2 3 4 5)
PRIMARY_LABEL="dressage keep_versions (distinct versions kept)"
PINS=()

# SECONDARY_KNOB=policy.max_partial_rollout_preempts
# SECONDARY_VALUES=(1 2 4)
```

- [ ] **Step 3: Write the one two-dimensional grid**

`verl_v1/grid.sh`:
```bash
# verl v1: the L1 feed and the L3 bound, swept together, because the bound can
# only bind once the feed lets rollout run ahead of training. Both assert
# strictly positive, so neither axis starts at 0.
PRESET=verl_v1
PRIMARY_KNOB=policy.num_warmup_batches
PRIMARY_VALUES=(1 2 3 4 5)
PRIMARY_LABEL="verl_v1 num_warmup_batches (batches prefetched)"
SECONDARY_KNOB=policy.max_off_policy_threshold
SECONDARY_VALUES=(1 2 3 4 5)
PINS=()

# SECONDARY_KNOB=policy.max_off_policy_strategy
# SECONDARY_VALUES=(drop wait)
```

- [ ] **Step 4: Write the driver**

Create `examples/simulate/staleness_throughput/run_all.sh`:

```bash
#!/usr/bin/env bash
# Sweep every preset. psrl and verl_v1 lead, because they are the contrast the
# study is built around: a bound that waits against a bound that discards.
#
# Presets run in sequence while their own points run in parallel, so one
# preset's failure does not hide eleven others. Each preset's exit status is
# reported at the end.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PRESETS=(
    psrl
    verl_v1
    verl
    areal
    roll
    slime
    skyrl
    openrlhf
    dressage
    nemo_rl
    trl
    prime_rl
)

declare -A status
for preset in "${PRESETS[@]}"; do
    echo ""
    echo "================ ${preset} ================"
    if bash "${SCRIPT_DIR}/sweep.sh" "${preset}"; then
        status["${preset}"]=ok
    else
        status["${preset}"]=FAILED
    fi
done

echo ""
echo "================ summary ================"
failed=0
for preset in "${PRESETS[@]}"; do
    printf '  %-10s %s\n' "${preset}" "${status[${preset}]}"
    [[ "${status[${preset}]}" == "ok" ]] || failed=1
done
exit "${failed}"
```

- [ ] **Step 5: Write the README**

Create `examples/simulate/staleness_throughput/README.md`:

```markdown
# Staleness-throughput sweeps

One figure per preset: how that system's own staleness knob redistributes the
fleet's generation throughput across staleness levels and into discarded work.

    bash sweep.sh psrl        # one preset
    bash run_all.sh           # all twelve

Design: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/docs/superpowers/specs/2026-10-23-staleness-throughput-sweep-design.md`

## Reading the figure

A bar's height is the fleet's full generation throughput, in tokens per second.
Its blue segments are tokens that reached a training batch, keyed by the
staleness they carried. Its hatched segments are tokens that were generated and
then thrown away, keyed by why. The line across the blue is the effective
training throughput.

A knob that tightens staleness does not shorten the bar. It moves tokens across
that line. That is the trade-off the figures exist to show.

The number above each bar is the simulated span. Every bar in a figure finished
the same number of training steps on the same number of trajectories, so span is
the honest cost axis and the gate below enforces the premise.

## The trap

With two rollout GPUs against six training GPUs on a long-tailed trace, a
preset whose admission is not opened runs **fully synchronously**: every token is
consumed at staleness 0, nothing is ever discarded, and no L3 threshold binds at
any value. A flat figure can mean the knob is not a staleness bound, or it can
mean admission never let rollout run ahead in the first place. Check the
staleness spread before concluding anything about a threshold.

Staleness also cannot exceed the number of versions in the run, so a short
`N_VERSIONS` reports the run length rather than the mechanism.

## Gates

`sweep.sh` refuses rather than warns.

- Every point must consume the same trajectory count. Throughput is a rate, so a
  point that ended early scores arbitrarily well. Consumed *tokens* may still
  differ, because a tighter bound resamples a different set of trajectories.
- The engine `.so` md5 must be unchanged across the sweep. Another session
  rebuilding it in place kills running simulations with a native signal that
  reads like an engine bug.
- `log_exclude=instance,rollout_step` on every run, and the instance stream
  deleted after. The shared filesystem quota is chronically full and a sweep that
  fills it produces truncated traces that look like failures.
- At cap 38912 the training cost is extrapolated on `tokens` and
  `pipeline_slots`. Every figure carries the footnote. Treat the training
  component as indicative; the rollout and staleness numbers are the claims.

## Two mandatory pins

Not optional panels. Without them the sweep is illegal or inert.

- `openrlhf` pins `partial_rollout_enable=true`. The policy asserts for any
  `async_queue_size` above 1 otherwise. This also makes the pause mode
  `in_place`, which is the same switch in the real system.
- `nemo_rl` pins `sampler=windowed`. Under the default `in_order`, `horizon`
  reads `max_lookahead_versions` and the L2 cancel never runs, so sweeping
  `max_staleness_versions` would move nothing.

## Axis floors

Five axes start at 1, each because 0 is rejected or inverted. Validated by
running the endpoints; do not change one without re-running it.

| preset | knob | why not 0 |
|---|---|---|
| `verl_v1` | `num_warmup_batches` | asserts `> 0`, or nothing is ever admitted |
| `verl_v1` | `max_off_policy_threshold` | asserts `> 0`, `None` to disable |
| `slime` | `update_weights_interval` | a loop interval, fails at 0 |
| `openrlhf` | `async_queue_size` | asserts `> 0` |
| `dressage` | `keep_versions` | 0 means the filter is **off**, the loose end |

`verl.staleness_threshold` shares the 0..5 range but is a sample-overage ratio,
not a version count. Its axis says so, and it is not commensurable with the
eleven version counts beside it.
```

- [ ] **Step 6: Verify every grid parses and declares the contract**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL/examples/simulate/staleness_throughput
for d in psrl verl_v1 verl areal roll slime skyrl openrlhf dressage nemo_rl trl prime_rl; do
  ( SECONDARY_KNOB=""; SECONDARY_VALUES=(); PINS=()
    source "$d/grid.sh"
    [[ -n "${PRESET}" && -n "${PRIMARY_KNOB}" && -n "${PRIMARY_LABEL}" && ${#PRIMARY_VALUES[@]} -gt 0 ]] \
      || { echo "INCOMPLETE $d"; exit 1; }
    [[ "${PRESET}" == "$d" ]] || { echo "MISMATCH $d declares ${PRESET}"; exit 1; }
    n=$(( ${#PRIMARY_VALUES[@]} * ( ${#SECONDARY_VALUES[@]} > 0 ? ${#SECONDARY_VALUES[@]} : 1 ) ))
    printf '  %-10s %-38s %d points\n' "$d" "${PRIMARY_KNOB}" "$n" ) || exit 1
done
```
Expected: twelve lines, no `INCOMPLETE` or `MISMATCH`.

- [ ] **Step 7: Confirm the point total matches the spec**

```bash
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL/examples/simulate/staleness_throughput
total=0
for d in psrl verl_v1 verl areal roll slime skyrl openrlhf dressage nemo_rl trl prime_rl; do
  ( SECONDARY_VALUES=(); source "$d/grid.sh"
    s=${#SECONDARY_VALUES[@]}; (( s == 0 )) && s=1
    echo $(( ${#PRIMARY_VALUES[@]} * s )) )
done | paste -sd+ | bc
```
Expected: `88`. If it differs, a grid's range drifted from the spec.

- [ ] **Step 8: Check every preset name is a real registry key**

```bash
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
python - <<'PY'
from pathlib import Path
from srl.policy.registry import POLICY_REGISTRY
root = Path("examples/simulate/staleness_throughput")
dirs = sorted(p.name for p in root.iterdir() if (p / "grid.sh").is_file())
missing = [d for d in dirs if d not in POLICY_REGISTRY]
unswept = [k for k in POLICY_REGISTRY if k not in dirs]
assert not missing, f"Not registry presets: {missing!r}."
assert not unswept, f"Registry presets with no grid: {unswept!r}."
print(f"All {len(dirs)} registry presets have a grid.")
PY
```
Expected: `All 12 registry presets have a grid.`

- [ ] **Step 9: Commit**

```bash
git add examples/simulate/staleness_throughput/
git commit -m "feat(simulate): the eleven remaining preset grids, driver and README

Every primary axis is 0..5, or 1..5 where 0 is rejected by an assert or means
the opposite of a tight bound. dressage keep_versions=0 turns its filter off,
so a 0-based axis would have put its loosest setting where every other preset
has its tightest. openrlhf pins partial_rollout_enable and nemo_rl pins
sampler=windowed, without which the sweep asserts or moves nothing.

88 points across twelve presets.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Run the sweep and record what it shows

**Files:**
- Create: `examples/simulate/staleness_throughput/RESULTS.md`
- Create: `examples/simulate/staleness_throughput/*/figures/staleness_throughput.png` (generated)

**Interfaces:**
- Consumes: everything above.
- Produces: the figures the blog post draws on, and a written record of which axes turned out to be informative.

- [ ] **Step 1: Confirm the inputs exist before spending the sweep**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
ls -l outputs/workload/qwen3_8b_dapo_38912_tp1/joint.jsonl
ls -l outputs/cost/qwen3_8b_tp2_pp3_thd/training_hydraulis.json
ls -d outputs/profile/qwen3_8b/kernel_tp1/kernel
df -h . | tail -1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1
```
Expected: all three paths present, free space available, GPU 0 idle. The KV capacity for this configuration is already cached, so no vLLM probe should start. If one does, let it finish once before launching the sweep, so twelve presets do not each start a server.

- [ ] **Step 2: Run the full sweep**

```bash
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
MAX_PARALLEL=16 bash examples/simulate/staleness_throughput/run_all.sh 2>&1 \
  | tee /tmp/staleness_sweep.log
```
Expected: twelve `ok` lines in the summary. Any `FAILED` preset is diagnosed from `outputs/simulate/staleness_throughput/<preset>/<point>/sweep.log` before continuing.

- [ ] **Step 3: Read every figure**

Read each of the twelve PNGs under `examples/simulate/staleness_throughput/*/figures/`. For each, record: does the staleness composition change across the axis, does a discard segment ever appear, and does span move. A preset where none of the three happens is either a knob that is not a staleness bound or a regime where admission never opened, and the two are distinguished by whether any point shows staleness above 0.

- [ ] **Step 4: Check the two predictions the spec made**

The spec predicted `skyrl` and `openrlhf` would be flat, and warned that a 0..5 ceiling might already be unbounded. Test both against the output:

```bash
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
python - <<'PY'
from pathlib import Path
from srl.diagnosis.staleness import staleness_breakdown
from srl.telemetry.store import JsonlStore

base = Path("outputs/simulate/staleness_throughput")
print(f"{'preset':<10}{'point':>7}{'span':>9}{'maxS':>6}{'waste%':>8}  spread")
for preset_dir in sorted(base.iterdir()):
    if not preset_dir.is_dir():
        continue
    for run in sorted(preset_dir.iterdir()):
        logs = run / "logs"
        if not (logs / "request.jsonl").is_file():
            continue
        bd = staleness_breakdown(JsonlStore(logs))
        total = bd.total_consumed_tokens + bd.total_wasted_tokens
        share = 100.0 * bd.total_wasted_tokens / max(1, total)
        spread = " ".join(f"S{s}:{n / 1e6:.1f}M" for s, n in sorted(bd.consumed_tokens.items()))
        print(f"{preset_dir.name:<10}{run.name:>7}{bd.span:>9.0f}{bd.max_staleness:>6}{share:>8.1f}  {spread}")
PY
```

Record which presets show a plateau across their whole range. Per the spec, the fix for a plateau is to re-centre that one preset's axis and say so on its figure, not to widen all twelve.

- [ ] **Step 5: Write the results record**

Create `examples/simulate/staleness_throughput/RESULTS.md` holding: the regime, the table from Step 4, and one line per preset stating whether its axis was informative and what it showed. State flat results as findings with their mechanism, not as omissions. Do not claim any preset is faster than another as a system: each figure compares one preset against itself.

- [ ] **Step 6: Run the full test suite once more**

```bash
python -m pytest tests/ -q
```
Expected: no failures.

- [ ] **Step 7: Commit the figures and the record**

```bash
git add examples/simulate/staleness_throughput/RESULTS.md \
        examples/simulate/staleness_throughput/*/figures/
git commit -m "feat(simulate): run the 88-point staleness-throughput sweep

Twelve figures plus the per-point table, on the pinned 2+6 GPU regime over the
near-untruncated DAPO trace. RESULTS.md records which axes moved the
composition, which only moved span, and which were flat with the mechanism
that explains each flat one.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** Figure definition, Task 2. Measured regime, Task 5 `common.sh`. Reduction, Task 1 (relocated to `srl/diagnosis/`, justified above). Figure module and CLI, Tasks 2 and 3. De-duplication of the inline reduction, Task 4. Harness layout with twelve `grid.sh` dirs, Tasks 5 and 6. Unified 0..5 and 1..5 axes with per-floor reasons, Task 6 Steps 1 to 3 and the README. The two mandatory pins, Task 6 Steps 1 and 2. All four gates, Task 5 `sweep.sh` plus Task 1's `check_same_consumption`, with the gate itself exercised in Task 5 Step 5. The 88-point total, asserted in Task 6 Step 7. The plateau caveat, Task 7 Step 4. Out-of-scope items are respected: no policy change, no refitting, no cross-preset speedup claim.

**Placeholder scan.** Every code step carries complete code. Every command carries its expected output. Task 7 Steps 3 and 5 are judgement steps rather than code, and they state exactly what to record and what not to claim.

**Type consistency.** `StalenessBreakdown` field names are identical in Tasks 1, 2, 3, 4, and 7. `staleness_breakdown` and `check_same_consumption` are spelled the same at all five call sites. `Point` and `Panel` are defined in Task 2 and imported by name in Task 3. `plot_staleness_throughput`'s keyword names match between Task 2's definition, Task 2 Step 5's smoke script, and Task 3's call. The `grid.sh` variable contract declared in Task 5 is the set Task 6 writes and Task 6 Step 6 validates.

**One fix applied during review.** Task 5 Step 4's first command cannot work, because `sweep.sh` sources the grid after `common.sh` and so overwrites an environment-supplied `PRIMARY_VALUES`. The step now shows the failing form, says why, and gives the temporary-grid form that does work.
