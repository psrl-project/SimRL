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


def test_the_consumption_gate_exits_cleanly_rather_than_crashing(tmp_path, monkeypatch):
    """A violated consumption gate must surface as SystemExit, like the manifest's own errors."""
    a = _run(tmp_path, "a")
    store = JsonlStore(tmp_path / "b" / "logs", overwrite=True)
    store.write("request", event="consume", t=5.0, staleness_true=0, generated=50)
    store.close()
    manifest = tmp_path / "points.tsv"
    manifest.write_text(f"\t1\t{a}\n\t2\t{tmp_path / 'b'}\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "srl-staleness-plot",
            str(manifest),
            "--primary-label",
            "k",
            "--title",
            "t",
            "--out",
            str(tmp_path / "x.png"),
        ],
    )
    from srl.cli.staleness_plot import main

    with pytest.raises(SystemExit, match="consumed different trajectory counts"):
        main()


def test_a_run_with_no_consumed_trajectories_names_its_line(tmp_path):
    """A livelocked point must surface as a clean SystemExit, not an uncaught ValueError."""
    root = tmp_path / "livelocked" / "logs"
    store = JsonlStore(root, overwrite=True)
    store.write("request", event="enqueue", t=1.0)
    store.close()
    manifest = tmp_path / "points.tsv"
    manifest.write_text(f"\t0\t{tmp_path / 'livelocked'}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"no consumed trajectories.*manifest line 1"):
        load_manifest(manifest)


def test_allow_unequal_consumption_prints_the_differing_counts(tmp_path, monkeypatch, capsys):
    """Bypassing the gate must not claim a single count that is actually false."""
    a = _run(tmp_path, "a")
    store = JsonlStore(tmp_path / "b" / "logs", overwrite=True)
    store.write("request", event="consume", t=5.0, staleness_true=0, generated=50)
    store.close()
    manifest = tmp_path / "points.tsv"
    manifest.write_text(f"\t1\t{a}\n\t2\t{tmp_path / 'b'}\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "srl-staleness-plot",
            str(manifest),
            "--primary-label",
            "k",
            "--title",
            "t",
            "--out",
            str(tmp_path / "x.png"),
            "--allow-unequal-consumption",
        ],
    )
    from srl.cli.staleness_plot import main

    main()
    out = capsys.readouterr().out
    assert "Points consumed different trajectory counts: [1, 2]." in out, f"Got {out!r}."
    assert "Every point consumed" not in out, f"Got {out!r}."
