"""Every measured table must have a registered op, and the CLI must not hardcode a list."""

from __future__ import annotations

from pathlib import Path

import pytest

from srl.cost.kernel.ops.base import OP_REGISTRY, families

_REPO = Path(__file__).resolve().parents[1]
_TABLES = _REPO / "outputs" / "profile" / "qwen3_8b" / "kernel_tp1" / "kernel"


def test_the_registry_covers_every_op_module_in_a_fresh_interpreter():
    """Importing only `ops.base` must yield a registry holding every op module's ops.

    Two properties in one check, and neither survives in-process.

    Coverage: asserting a fixed three-name subset never notices a fourth or fifth op that
    stopped registering. Deriving the expected names from the modules under `ops/` does.

    Freshness: the registry is populated purely by the import side effects in
    `ops/__init__.py`, so an op whose import was dropped from that file still appears in
    `OP_REGISTRY` whenever any sibling test file imported the module itself. Under
    `pytest-randomly` that makes an in-process assertion pass or fail at random. This is
    exactly the production bug `PerfTableWriter.table_path("gemm")` hit: it raised
    `Registered ops: none` in a fresh interpreter while every unit test passed. The subprocess
    reproduces the fresh interpreter, following `tests/test_kernel_ops_base.py`'s torch probe.
    """
    import subprocess
    import sys

    probe = """
import pathlib

import srl.cost.kernel.ops as ops_pkg
import srl.cost.kernel.ops.base as base

modules = sorted(
    path.stem
    for path in pathlib.Path(ops_pkg.__file__).parent.glob("*.py")
    if path.stem not in ("__init__", "base")
)
assert modules, "found no op modules under ops/"
registered = sorted({op.__class__.__module__.rsplit(".", 1)[-1] for op in base.OP_REGISTRY.values()})
assert registered == modules, (
    f"ops/ holds {modules} but only {registered} registered; "
    f"an op module is missing from ops/__init__.py"
)
print("covered")
"""
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "covered" in result.stdout


def test_the_registry_covers_every_op_module_in_process():
    """The same coverage property, cheap and readable, for a fast local signal."""
    import pathlib

    import srl.cost.kernel.ops as ops_pkg

    modules = sorted(
        path.stem
        for path in pathlib.Path(ops_pkg.__file__).parent.glob("*.py")
        if path.stem not in ("__init__", "base")
    )
    registered = sorted(
        {op.__class__.__module__.rsplit(".", 1)[-1] for op in OP_REGISTRY.values()}
    )
    assert registered == modules, f"ops/ holds {modules} but only {registered} registered"


def test_the_cli_family_choices_come_from_the_registry():
    """A hardcoded choices list is the last place adding an op means editing the CLI."""
    from srl.cli.profile import build_parser

    parser = build_parser()
    args = parser.parse_args(["kernels", "run", "--config", "c.yaml"])
    assert args.family == "all"
    for family in families():
        parsed = parser.parse_args(["kernels", "run", "--config", "c.yaml", "--family", family])
        assert parsed.family == family


def test_an_unknown_family_is_rejected():
    from srl.cli.profile import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["kernels", "run", "--config", "c.yaml", "--family", "widgets"])


def test_the_family_choices_are_not_a_hardcoded_literal():
    """Derived choices are what make a new op need zero CLI edits.

    Asserted by registering a synthetic op and watching the parser's accepted choices grow,
    because greping for one retired spelling passes against any *other* hardcoded literal: a
    fully hardcoded five-family list satisfied the old assertion exactly. The derived list has
    to actually move for this to pass.
    """
    from srl.cli.profile import build_parser
    from srl.cost.kernel.ops.base import OP_REGISTRY

    before = _family_choices(build_parser())
    assert "widgets" not in before

    OP_REGISTRY["widget_probe"] = _WidgetProbeOp()
    try:
        after = _family_choices(build_parser())
    finally:
        OP_REGISTRY.pop("widget_probe", None)

    assert "widgets" in after, (
        "Registering an op in a new family did not change the CLI's --family choices, so the "
        "list is hardcoded rather than derived from the registry."
    )
    assert set(after) - set(before) == {"widgets"}
    # And the parser must accept the new family, not merely advertise it.
    OP_REGISTRY["widget_probe"] = _WidgetProbeOp()
    try:
        parsed = build_parser().parse_args(
            ["kernels", "run", "--config", "c.yaml", "--family", "widgets"]
        )
    finally:
        OP_REGISTRY.pop("widget_probe", None)
    assert parsed.family == "widgets"


def _family_choices(parser) -> tuple[str, ...]:
    """Return the `--family` choices `build_parser` produced, minus the `all` sentinel."""
    import argparse

    for action in parser._actions:
        if action.dest == "family":
            return tuple(name for name in action.choices if name != "all")
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                found = _family_choices_or_none(sub)
                if found is not None:
                    return found
    raise AssertionError("Could not locate the kernels run --family action on the parser.")


def _family_choices_or_none(parser) -> tuple[str, ...] | None:
    """Return `_family_choices` for `parser`, or None when this subtree has no `--family`."""
    try:
        return _family_choices(parser)
    except AssertionError:
        return None


class _WidgetProbeOp:
    """A synthetic op in a family no real op uses, for the derived-choices test."""

    name = "widget_probe"
    family = "widgets"
    slice_columns = ("dtype",)
    axis_columns = ("n",)
    fits_floor = False

    def sol(self, spec, *, slice_key, coords, floor_s=0.0):
        """Never called; the parser only reads `family`."""
        raise NotImplementedError("widget_probe is a parser fixture.")

    def parse_sweep(self, value):
        """Never called; the parser only reads `family`."""
        raise NotImplementedError("widget_probe is a parser fixture.")

    def table_columns(self):
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)

    def measure(self, model, config, *, writer_factory):
        """Never called; the parser only reads `family`."""
        raise NotImplementedError("widget_probe is a parser fixture.")


@pytest.mark.skipif(not _TABLES.is_dir(), reason="Measured perf tables are machine-local.")
def test_every_measured_parquet_has_a_registered_op():
    """An unregistered table is silently skipped by every report; that must be a failure."""
    stems = set()
    for parquet in (_TABLES / "h20").glob("*/*/*/*_perf.parquet"):
        stem = parquet.stem
        stems.add(stem[: stem.rfind("_perf")])
    assert stems, f"no perf tables found under {_TABLES / 'h20'}"
    unregistered = sorted(stems - set(OP_REGISTRY))
    assert not unregistered, f"perf tables with no registered op: {unregistered}"


@pytest.mark.skipif(not _TABLES.is_dir(), reason="Measured perf tables are machine-local.")
def test_every_measured_table_sits_in_its_ops_family_directory():
    """A table under the wrong family would be invisible to ops_for_family dispatch."""
    for parquet in (_TABLES / "h20").glob("*/*/*/*_perf.parquet"):
        stem = parquet.stem
        name = stem[: stem.rfind("_perf")]
        op = OP_REGISTRY.get(name)
        if op is None:
            continue
        family_dir = parquet.parents[2].name
        assert family_dir == op.family, f"{parquet} sits under {family_dir}, op declares {op.family}"
