"""Row selection must be reproducible, bounded, and justified per row."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "workflow" / "analysis" / "scripts" / "select.py"
    spec = importlib.util.spec_from_file_location("select", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["select_rows"] = module
    spec.loader.exec_module(module)
    return module


def _summary():
    rows = []
    for point, speedups in (("pA", (1.00, 0.80)), ("pB", (1.00, 1.30)), ("pC", (1.00, 1.05))):
        for arm, speedup in zip(("control", "treatment"), speedups):
            rows.append(
                {
                    "point": point,
                    "arm": arm,
                    "row_id": f"{point}/{arm}",
                    "speedup": speedup,
                    "span": 100.0 / speedup,
                    "replay": {"script": "workflow/validate/experiments/x.py", "spec": {}},
                }
            )
    return {"rows": rows, "summary": {"failed": {}}}


def test_selection_is_bounded_and_every_row_carries_a_reason():
    picks = _load().pick(_summary(), control_arm="control", limit=6)
    assert 0 < len(picks) <= 6, f"Expected 1 to 6 picks, got {len(picks)!r}: {picks!r}."
    for pick in picks:
        assert pick["reason"], pick
    assert len({pick["row_id"] for pick in picks}) == len(picks), f"Duplicate row ids in picks: {picks!r}."


def test_the_worst_and_best_treatment_rows_are_both_selected():
    picks = _load().pick(_summary(), control_arm="control", limit=6)
    chosen = {pick["row_id"] for pick in picks}
    assert "pA/treatment" in chosen, "The lowest-speedup treatment row must be diagnosed."
    assert "pB/treatment" in chosen, "The highest-speedup treatment row must be diagnosed."


def test_a_row_that_failed_a_gate_is_never_selected():
    summary = _summary()
    summary["summary"]["failed"] = {"pA/treatment": ["tokens_per_group"]}
    picks = _load().pick(summary, control_arm="control", limit=6)
    assert "pA/treatment" not in {pick["row_id"] for pick in picks}, (
        f"A gate-failed row was selected anyway: {picks!r}."
    )


def test_a_row_without_a_replay_field_is_refused():
    summary = _summary()
    del summary["rows"][0]["replay"]
    with pytest.raises(ValueError, match="no replay field"):
        _load().pick(summary, control_arm="control", limit=6)
