"""Floors are keyed by op name, and an old phase-keyed file is rejected, not misread."""

from __future__ import annotations

import json

import pytest

from srl.cost.kernel.floor import load_floors, write_floors
from srl.cost.kernel.interp import KernelCoverageError


def test_floors_round_trip_under_op_name_keys(tmp_path):
    data = {
        "context_attention": 32e-6,
        "generation_attention": 22e-6,
        "fitted_mre_percent": {"context_attention": 5.7432, "generation_attention": 6.6293},
        "candidates_us": [0, 22, 32],
        "gpu": "h20",
        "backend": "vllm",
    }
    write_floors(tmp_path, data)
    loaded = load_floors(tmp_path, backend="vllm", warn_missing=False)
    assert loaded == {"context_attention": 32e-6, "generation_attention": 22e-6}


def test_an_old_phase_keyed_file_is_rejected_naming_the_fit_command(tmp_path):
    """Reading it leniently would apply a zero floor: 34.89% MRE against 6.63% fitted."""
    (tmp_path / "floors.json").write_text(
        json.dumps({"context": 32e-6, "generation": 22e-6, "gpu": "h20", "backend": "vllm"}),
        encoding="utf-8",
    )
    with pytest.raises(KernelCoverageError, match="fit-floor"):
        load_floors(tmp_path, backend="vllm", warn_missing=False)


def test_a_missing_file_falls_back_to_the_hardcoded_map_under_op_keys(tmp_path):
    from srl.cost.kernel.sol import FALLBACK_FLOORS_S

    loaded = load_floors(tmp_path, backend="vllm", warn_missing=True)
    assert loaded == FALLBACK_FLOORS_S["vllm"]
    # Derived from the registry rather than listed, so adding an op cannot leave this
    # assertion quietly describing a subset of what the fallback actually covers. What is
    # being asserted is that every key is an op name, which is the property that replaced
    # the old phase keying.
    from srl.cost.kernel.ops.base import OP_REGISTRY

    assert set(loaded) <= set(OP_REGISTRY)
    assert all(OP_REGISTRY[name].fits_floor for name in loaded)


def test_a_missing_file_returns_none_when_not_warning(tmp_path):
    assert load_floors(tmp_path, backend="vllm", warn_missing=False) is None


def test_the_fallback_map_is_keyed_by_op_name_not_phase():
    from srl.cost.kernel.ops.base import OP_REGISTRY, ops_for_family
    from srl.cost.kernel.sol import FALLBACK_FLOORS_S

    # The attention and elementwise values are the fitted ones and are pinned: they came from
    # a real sweep, and a silent change would move the gate.
    vllm = FALLBACK_FLOORS_S["vllm"]
    assert vllm["context_attention"] == 32e-6
    assert vllm["generation_attention"] == 22e-6
    assert vllm["elementwise"] == 2e-6
    assert FALLBACK_FLOORS_S["flashinfer"] == {
        "context_attention": 4e-6,
        "generation_attention": 6e-6,
    }
    # Every collective is measured through vLLM's dispatcher *and* through torch.distributed
    # as a control, so both subtrees need an entry for each: the backend selects which
    # subtree is read, and a gap degrades to a zero floor on one side only.
    collectives = {op.name for op in ops_for_family("collective")}
    assert collectives <= set(vllm)
    assert set(FALLBACK_FLOORS_S["nccl"]) == collectives
    # No phase keys survive anywhere: that spelling forced four separate translations.
    for backend, floors in FALLBACK_FLOORS_S.items():
        assert set(floors) <= set(OP_REGISTRY), backend
        assert "context" not in floors and "generation" not in floors, backend


def test_the_old_attention_floor_name_is_gone():
    """Leaving both names would let a caller read the phase-keyed map by accident."""
    import srl.cost.kernel.sol as sol_module

    assert not hasattr(sol_module, "ATTENTION_FLOOR_S")


def test_floor_fitting_skips_ops_that_declare_no_floor():
    """GEMM is excluded by fits_floor=False, not by absence from a hardcoded map."""
    import inspect

    from srl.cost.kernel import floor as floor_module

    source = inspect.getsource(floor_module)
    assert "_ATTENTION_OPS" not in source, "the phase map is still there"
    assert "fits_floor" in source, "fitting does not consult the op's floor policy"


def test_an_unregistered_op_key_in_floors_json_is_ignored(tmp_path):
    """A stale key from a removed op must not crash a load."""
    write_floors(
        tmp_path,
        {"context_attention": 32e-6, "retired_op": 9e-6, "gpu": "h20", "backend": "vllm"},
    )
    loaded = load_floors(tmp_path, backend="vllm", warn_missing=False)
    assert loaded == {"context_attention": 32e-6}


def test_the_database_reads_each_op_familys_own_floors_file(tmp_path):
    """floors.json lives inside the family's version directory, so the lookup must be scoped.

    Resolving it under a hardcoded attention path made every non-attention floor-taking op
    read the attention directory, silently applying a zero floor to an all_reduce whose fitted
    floor dominates its whole small-message regime.
    """
    from srl.cost.kernel.database import KernelDatabase
    from srl.cost.kernel.ops.base import op_for

    attention_dir = tmp_path / "h20" / "attention" / "vllm" / "0.1.0"
    attention_dir.mkdir(parents=True)
    write_floors(attention_dir, {"generation_attention": 22e-6, "gpu": "h20", "backend": "vllm"})
    collective_dir = tmp_path / "h20" / "collective" / "vllm" / "0.1.0"
    collective_dir.mkdir(parents=True)
    write_floors(collective_dir, {"all_reduce": 7e-6, "gpu": "h20", "backend": "vllm"})

    database = KernelDatabase.from_root(tmp_path, gpu="h20")
    assert database._floors_for_op(op_for("all_reduce"), "vllm") == {"all_reduce": 7e-6}
    assert database._floors_for_op(op_for("generation_attention"), "vllm") == {
        "generation_attention": 22e-6
    }


def _touch_perf_table(version_dir, op_name: str) -> None:
    """Create an empty parquet so `ops_in` reports this op as present in the directory."""
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / f"{op_name}_perf.parquet").write_bytes(b"")


def test_a_floor_taking_op_with_no_fallback_warns_by_name(tmp_path, caplog):
    """A fallback message is misleading when the fallback holds nothing for the op.

    The fallback map is keyed by backend as well as op, so a floor-taking op measured under a
    backend that has no entry for it still degrades to a zero floor even though a `floors.json`
    fallback message was logged. `elementwise` has a `vllm` entry but no `flashinfer` one, so a
    table filed there is the real instance of this. The warning must name the op rather than only
    the directory, so the loudness is proportional to what is actually missing.
    """
    import logging

    version_dir = tmp_path / "h20" / "elementwise" / "flashinfer" / "0.1.0"
    _touch_perf_table(version_dir, "elementwise")
    with caplog.at_level(logging.WARNING):
        load_floors(version_dir, backend="flashinfer", warn_missing=True)
    assert "elementwise" in caplog.text, f"Warning did not name the unfloored op: {caplog.text!r}"
    assert "zero floor" in caplog.text


def test_a_covered_floor_taking_op_gets_no_second_warning(tmp_path, caplog):
    """The per-op warning must not fire when the fallback does cover the op."""
    import logging

    version_dir = tmp_path / "h20" / "attention" / "vllm" / "0.1.0"
    _touch_perf_table(version_dir, "generation_attention")
    with caplog.at_level(logging.WARNING):
        load_floors(version_dir, backend="vllm", warn_missing=True)
    assert "zero floor" not in caplog.text, (
        f"generation_attention has a vllm fallback, so no per-op warning is due: {caplog.text!r}"
    )


def test_an_op_declaring_no_floor_never_triggers_the_per_op_warning(tmp_path, caplog):
    """GEMM has no fallback entry and needs none, so a gemm-only directory must stay quiet."""
    import logging

    version_dir = tmp_path / "h20" / "gemm" / "vllm" / "0.1.0"
    _touch_perf_table(version_dir, "gemm")
    with caplog.at_level(logging.WARNING):
        load_floors(version_dir, backend="vllm", warn_missing=True)
    assert "zero floor" not in caplog.text

