"""A summary row must carry enough to re-run exactly itself."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "workflow" / "validate" / "scripts" / "replayspec.py"
    spec = importlib.util.spec_from_file_location("replayspec", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["replayspec"] = module
    spec.loader.exec_module(module)
    return module


def test_row_id_joins_point_and_arm():
    assert _load().row_id("p256_n2", "control") == "p256_n2/control"


def test_replay_script_path_is_repo_relative():
    module = _load()
    field = module.replay_field(ROOT / "workflow" / "validate" / "experiments" / "smoke.py", {"cap": 0}, ROOT)
    assert field["script"] == "workflow/validate/experiments/smoke.py", f"Unexpected script path: {field!r}."
    assert field["spec"] == {"cap": 0}, f"Unexpected spec: {field!r}."


def test_an_absolute_path_outside_the_repo_is_refused():
    module = _load()
    with pytest.raises(ValueError, match="outside the repository"):
        module.replay_field(Path("/tmp/elsewhere.py"), {}, ROOT)


def test_attach_adds_row_id_and_replay_to_every_row():
    module = _load()
    rows = [{"point": "p1", "arm": "control"}, {"point": "p1", "arm": "treatment"}]
    out = module.attach(
        rows,
        ROOT / "workflow" / "validate" / "experiments" / "smoke.py",
        {"p1/control": {"cap": 0}, "p1/treatment": {"cap": 32}},
        ROOT,
    )
    assert [row["row_id"] for row in out] == ["p1/control", "p1/treatment"], f"Unexpected row ids: {out!r}."
    assert out[1]["replay"]["spec"] == {"cap": 32}, f"Unexpected replay spec: {out[1]!r}."
    assert all(row["replay"]["script"].startswith("workflow/validate/") for row in out), (
        f"A replay script path escaped workflow/validate/: {out!r}."
    )


def test_attach_refuses_a_row_with_no_spec():
    module = _load()
    with pytest.raises(KeyError, match="p1/treatment"):
        module.attach(
            [{"point": "p1", "arm": "treatment"}],
            ROOT / "workflow" / "validate" / "experiments" / "smoke.py",
            {},
            ROOT,
        )
