"""Tests for `srl.cost.kernel.floor`: fitting, persistence, and loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from srl.cost.kernel.floor import (
    CANDIDATE_FLOORS_US,
    fit_floors,
    fit_floors_report,
    load_floors,
    write_floors,
)
from srl.cost.kernel.hardware import load_gpu_spec

_VERSION = "0.26.1.dev0"
_BACKEND = "vllm"


# --- helpers ---


def _attention_version_dir(root: Path, backend: str = _BACKEND) -> Path:
    d = root / "h20" / "attention" / backend / _VERSION
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_generation_table(version_dir: Path, backend: str, points: list[tuple[int, int, float]]) -> None:
    """Write a minimal generation_attention parquet to `version_dir`."""
    count = len(points)
    path = version_dir / "generation_attention_perf.parquet"
    pq.write_table(
        pa.table(
            {
                "framework": [backend] * count,
                "version": [_VERSION] * count,
                "device": ["NVIDIA H20"] * count,
                "op_name": ["generation_attention"] * count,
                "kernel_source": ["flash_attn"] * count,
                "num_key_value_heads": [8] * count,
                "head_dim": [128] * count,
                "attn_dtype": ["bfloat16"] * count,
                "kv_cache_dtype": ["bfloat16"] * count,
                "block_size": [16] * count,
                "window_size": [0] * count,
                "num_heads": [32] * count,
                "batch_size": [p[0] for p in points],
                "step": [p[1] for p in points],
                "latency_s": [p[2] for p in points],
            }
        ),
        path,
    )


def _write_context_table(version_dir: Path, backend: str, points: list[tuple[int, int, float]]) -> None:
    """Write a minimal context_attention parquet to `version_dir`."""
    count = len(points)
    path = version_dir / "context_attention_perf.parquet"
    pq.write_table(
        pa.table(
            {
                "framework": [backend] * count,
                "version": [_VERSION] * count,
                "device": ["NVIDIA H20"] * count,
                "op_name": ["context_attention"] * count,
                "kernel_source": ["flash_attn"] * count,
                "num_key_value_heads": [8] * count,
                "head_dim": [128] * count,
                "attn_dtype": ["bfloat16"] * count,
                "kv_cache_dtype": ["bfloat16"] * count,
                "block_size": [16] * count,
                "window_size": [0] * count,
                "num_heads": [32] * count,
                "batch_size": [p[0] for p in points],
                "isl": [p[1] for p in points],
                "latency_s": [p[2] for p in points],
            }
        ),
        path,
    )


def _roofline_generation_points(floor_s: float) -> list[tuple[int, int, float]]:
    """Return generation points where floor_s is the binding constraint everywhere."""
    spec = load_gpu_spec("h20")
    from srl.cost.kernel.sol import generation_attention_sol

    points = []
    for batch_size in (1, 2, 4, 8, 16, 32):
        for step in (128, 256, 512, 1024, 2048, 4096):
            sol = generation_attention_sol(
                spec,
                batch_size=batch_size,
                step=step,
                num_heads=32,
                num_kv_heads=8,
                head_dim=128,
                attn_dtype="bfloat16",
                kv_cache_dtype="bfloat16",
                floor_s=floor_s,
            )
            points.append((batch_size, step, sol.latency_s))
    return points


# --- load_floors tests ---


def test_load_floors_returns_none_when_file_absent_and_warn_false(tmp_path):
    version_dir = _attention_version_dir(tmp_path)
    result = load_floors(version_dir, backend=_BACKEND, warn_missing=False)
    assert result is None


def test_load_floors_emits_warning_and_returns_fallback_when_absent(tmp_path, caplog):
    version_dir = _attention_version_dir(tmp_path)
    with caplog.at_level(logging.WARNING):
        result = load_floors(version_dir, backend=_BACKEND, warn_missing=True)
    assert result is not None
    # Warning names the missing directory.
    assert str(version_dir) in caplog.text
    # Fallback values come from FALLBACK_FLOORS_S for vllm.
    assert "context_attention" in result
    assert "generation_attention" in result
    assert result["context_attention"] == pytest.approx(32e-6)
    assert result["generation_attention"] == pytest.approx(22e-6)


def test_load_floors_returns_written_values(tmp_path):
    version_dir = _attention_version_dir(tmp_path)
    data = {
        "context_attention": 28e-6,
        "generation_attention": 18e-6,
        "fitted_mre_percent": {"context_attention": 5.0, "generation_attention": 7.0},
        "candidates_us": [0, 4, 8, 16, 22, 32],
        "gpu": "h20",
        "backend": _BACKEND,
    }
    write_floors(version_dir, data)
    result = load_floors(version_dir, backend=_BACKEND, warn_missing=False)
    assert result is not None
    assert result["context_attention"] == pytest.approx(28e-6)
    assert result["generation_attention"] == pytest.approx(18e-6)


def test_write_floors_produces_loadable_json(tmp_path):
    version_dir = _attention_version_dir(tmp_path)
    data = {
        "context_attention": 4e-6,
        "generation_attention": 6e-6,
        "fitted_mre_percent": {"context_attention": 9.87, "generation_attention": 6.01},
        "candidates_us": list(CANDIDATE_FLOORS_US),
        "gpu": "h20",
        "backend": "flashinfer",
    }
    path = write_floors(version_dir, data)
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["context_attention"] == pytest.approx(4e-6)
    assert loaded["backend"] == "flashinfer"


# --- floor_override in evaluate_holdout tests ---


def test_floor_override_is_used_instead_of_the_fallback_map(tmp_path):
    """Passing floor_override must change the MRE for attention ops."""
    from srl.cost.kernel.holdout import evaluate_holdout
    from srl.cost.kernel.ops.attention import GENERATION_ATTENTION_OP
    from srl.cost.kernel.table import KernelTable

    version_dir = _attention_version_dir(tmp_path)
    floor_s = 20e-6
    points = _roofline_generation_points(floor_s)
    _write_generation_table(version_dir, _BACKEND, points)
    path = version_dir / "generation_attention_perf.parquet"
    table = KernelTable.load(path, GENERATION_ATTENTION_OP)
    spec = load_gpu_spec("h20")

    # With the correct floor the MRE must be zero (uniform utility).
    result_correct = evaluate_holdout(
        table,
        op_name="generation_attention",
        backend=_BACKEND,
        gpu_spec=spec,
        fraction=1.0,
        floor_override=floor_s,
    )
    assert result_correct.point_count > 0
    assert result_correct.mre_percent == pytest.approx(0.0, abs=1e-6)

    # With a wrong floor the MRE must be non-zero.
    result_wrong = evaluate_holdout(
        table,
        op_name="generation_attention",
        backend=_BACKEND,
        gpu_spec=spec,
        fraction=1.0,
        floor_override=0.0,
    )
    assert result_wrong.mre_percent > 0.1


def test_floor_override_is_ignored_for_gemm(tmp_path):
    """floor_override must not affect GEMM: it has no dispatch floor."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from srl.cost.kernel.hardware import load_gpu_spec
    from srl.cost.kernel.holdout import evaluate_holdout
    from srl.cost.kernel.ops.gemm import GEMM_OP
    from srl.cost.kernel.sol import gemm_sol
    from srl.cost.kernel.table import KernelTable

    spec = load_gpu_spec("h20")
    ms = (512, 1024, 2048, 4096, 8192, 16384)
    shapes = [(m, 4096, 4096, gemm_sol(spec, m=m, n=4096, k=4096, dtype="bfloat16").latency_s / 0.5) for m in ms]
    path = tmp_path / "h20" / "gemm" / "vllm" / _VERSION / "gemm_perf.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(shapes)
    pq.write_table(
        pa.table(
            {
                "framework": ["vllm"] * count,
                "version": [_VERSION] * count,
                "device": ["NVIDIA H20"] * count,
                "op_name": ["gemm"] * count,
                "kernel_source": ["torch.nn.functional.linear"] * count,
                "gemm_dtype": ["bfloat16"] * count,
                "m": [s[0] for s in shapes],
                "n": [s[1] for s in shapes],
                "k": [s[2] for s in shapes],
                "latency_s": [s[3] for s in shapes],
            }
        ),
        path,
    )
    table = KernelTable.load(path, GEMM_OP)
    result_no_override = evaluate_holdout(
        table, op_name="gemm", backend="vllm", gpu_spec=spec, fraction=1.0
    )
    result_with_override = evaluate_holdout(
        table, op_name="gemm", backend="vllm", gpu_spec=spec, fraction=1.0, floor_override=999e-6
    )
    assert result_no_override.mre_percent == pytest.approx(result_with_override.mre_percent, abs=1e-9)


# --- fit_floors tests ---


def test_fit_floors_finds_zero_floor_when_data_has_no_floor(tmp_path):
    """When the measured latency equals pure roofline, floor=0 should win."""
    spec = load_gpu_spec("h20")
    from srl.cost.kernel.sol import generation_attention_sol

    version_dir = _attention_version_dir(tmp_path)
    points = []
    for batch_size in (1, 2, 4, 8, 16, 32):
        for step in (128, 256, 512, 1024, 2048, 4096):
            sol = generation_attention_sol(
                spec,
                batch_size=batch_size,
                step=step,
                num_heads=32,
                num_kv_heads=8,
                head_dim=128,
                attn_dtype="bfloat16",
                kv_cache_dtype="bfloat16",
                floor_s=0.0,
            )
            points.append((batch_size, step, sol.latency_s))
    _write_generation_table(version_dir, _BACKEND, points)
    result = fit_floors(version_dir, backend=_BACKEND, gpu="h20")
    # The minimiser must be the 0-floor candidate.
    assert result["generation_attention"] == pytest.approx(0.0, abs=1e-9)
    assert "fitted_mre_percent" in result
    assert result["fitted_mre_percent"]["generation_attention"] == pytest.approx(0.0, abs=1e-6)


def test_fit_floors_recovers_known_floor(tmp_path):
    """Fitting must recover the floor used to generate the data."""
    target_floor_us = 16.0
    version_dir = _attention_version_dir(tmp_path)
    points = _roofline_generation_points(target_floor_us * 1e-6)
    _write_generation_table(version_dir, _BACKEND, points)
    # Use fine candidates to bracket 16 us precisely.
    candidates = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0)
    result = fit_floors(version_dir, backend=_BACKEND, gpu="h20", candidates_us=candidates)
    assert result["generation_attention"] == pytest.approx(16e-6, abs=1e-9)
    assert result["fitted_mre_percent"]["generation_attention"] == pytest.approx(0.0, abs=1e-6)


def test_fit_floors_returns_metadata(tmp_path):
    version_dir = _attention_version_dir(tmp_path)
    points = _roofline_generation_points(10e-6)
    _write_generation_table(version_dir, _BACKEND, points)
    result = fit_floors(version_dir, backend=_BACKEND, gpu="h20", candidates_us=(0.0, 10.0, 20.0))
    assert result["gpu"] == "h20"
    assert result["backend"] == _BACKEND
    assert result["candidates_us"] == [0.0, 10.0, 20.0]


# --- holdout_report integration with floors.json ---


def test_holdout_report_uses_floors_json_when_present(tmp_path, caplog):
    """When floors.json is present the warning must NOT be emitted."""
    from srl.cost.kernel.holdout import holdout_report

    spec = load_gpu_spec("h20")
    from srl.cost.kernel.sol import generation_attention_sol

    version_dir = _attention_version_dir(tmp_path)
    floor_s = 22e-6
    points = []
    for batch_size in (1, 2, 4, 8, 16, 32):
        for step in (128, 256, 512, 1024, 2048, 4096):
            sol = generation_attention_sol(
                spec,
                batch_size=batch_size,
                step=step,
                num_heads=32,
                num_kv_heads=8,
                head_dim=128,
                attn_dtype="bfloat16",
                kv_cache_dtype="bfloat16",
                floor_s=floor_s,
            )
            points.append((batch_size, step, sol.latency_s))
    _write_generation_table(version_dir, _BACKEND, points)
    # Write the correct floors.json.
    write_floors(
        version_dir,
        {
            "context_attention": 32e-6,
            "generation_attention": floor_s,
            "fitted_mre_percent": {"context_attention": 5.0, "generation_attention": 0.0},
            "candidates_us": [0, 22],
            "gpu": "h20",
            "backend": _BACKEND,
        },
    )
    with caplog.at_level(logging.WARNING):
        results = holdout_report(tmp_path, gpu="h20", fraction=1.0)
    # No fallback warning should appear.
    assert "FALLBACK_FLOORS_S" not in caplog.text
    gen_results = [r for r in results if r.op_name == "generation_attention"]
    assert gen_results
    # With the correct floor loaded from floors.json the MRE must be near zero.
    assert gen_results[0].mre_percent == pytest.approx(0.0, abs=1e-6)


def test_holdout_report_warns_when_floors_json_absent(tmp_path, caplog):
    """When floors.json is absent holdout_report must emit a warning."""
    from srl.cost.kernel.holdout import holdout_report

    spec = load_gpu_spec("h20")
    from srl.cost.kernel.sol import generation_attention_sol

    version_dir = _attention_version_dir(tmp_path)
    floor_s = 22e-6
    points = []
    for batch_size in (1, 2, 4, 8, 16, 32):
        for step in (128, 256, 512, 1024, 2048, 4096):
            sol = generation_attention_sol(
                spec,
                batch_size=batch_size,
                step=step,
                num_heads=32,
                num_kv_heads=8,
                head_dim=128,
                attn_dtype="bfloat16",
                kv_cache_dtype="bfloat16",
                floor_s=floor_s,
            )
            points.append((batch_size, step, sol.latency_s))
    _write_generation_table(version_dir, _BACKEND, points)
    with caplog.at_level(logging.WARNING):
        holdout_report(tmp_path, gpu="h20", fraction=1.0)
    # Fallback warning must name the missing directory.
    assert "floors.json" in caplog.text
    assert str(version_dir) in caplog.text


# --- fit_floors_report tree-walk tests ---


def test_fit_floors_report_writes_json_per_version(tmp_path):
    version_dir = _attention_version_dir(tmp_path)
    points = _roofline_generation_points(10e-6)
    _write_generation_table(version_dir, _BACKEND, points)
    written = fit_floors_report(tmp_path, gpu="h20", candidates_us=(0.0, 10.0, 20.0))
    assert len(written) == 1
    assert written[0] == version_dir / "floors.json"
    assert written[0].exists()


def test_fit_floors_report_empty_when_no_attention(tmp_path):
    """When there are no attention parquets no floors.json files are written."""
    written = fit_floors_report(tmp_path, gpu="h20")
    assert written == []


def test_fit_floors_report_skips_gemm_family(tmp_path):
    """GEMM parquets do not get fitted; only attention dirs produce floors.json."""
    gemm_dir = tmp_path / "h20" / "gemm" / "vllm" / _VERSION
    gemm_dir.mkdir(parents=True, exist_ok=True)
    # Write a fake gemm parquet (not read during floor fitting).
    (gemm_dir / "gemm_perf.parquet").write_bytes(b"dummy")
    written = fit_floors_report(tmp_path, gpu="h20")
    assert written == []


def _kernel_tree(tmp_path: Path) -> Path:
    """Build a minimal table tree with floors.json, for the caching tests below."""
    root = tmp_path / "kernel"
    version_dir = _attention_version_dir(root)
    points = _roofline_generation_points(22e-6)
    _write_generation_table(version_dir, _BACKEND, points)
    _write_context_table(version_dir, _BACKEND, points)
    write_floors(
        version_dir,
        {"context_attention": 32e-6, "generation_attention": 22e-6, "gpu": "h20", "backend": _BACKEND},
    )
    return root


def test_floors_are_read_once_per_backend_not_once_per_query(tmp_path, monkeypatch):
    """A path-keyed cache still scanned the backend directory on every query."""
    import srl.cost.kernel.database as database_mod
    from srl.cost.kernel.database import AttentionQuery, KernelDatabase

    calls: list[str] = []
    real_load = database_mod._load_floors

    def counting(version_dir, *, backend, warn_missing=False):
        calls.append(backend)
        return real_load(version_dir, backend=backend, warn_missing=warn_missing)

    monkeypatch.setattr(database_mod, "_load_floors", counting)
    database = KernelDatabase.from_root(_kernel_tree(tmp_path), gpu="h20")
    query = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=8, head_dim=128, step=2048
    )
    for _ in range(25):
        database.query_attention(query)
    assert calls == [_BACKEND], calls


def test_a_query_does_not_scan_the_filesystem(tmp_path, monkeypatch):
    """listdir on every query made one attention query 2337 us; the interpolation is 17 us."""
    import os

    from srl.cost.kernel.database import AttentionQuery, KernelDatabase

    database = KernelDatabase.from_root(_kernel_tree(tmp_path), gpu="h20")
    query = AttentionQuery(
        phase="generation", batch_size=16, num_heads=32, num_kv_heads=8, head_dim=128, step=2048
    )
    database.query_attention(query)  # Warm the table, surface, and floor caches.

    listdirs: list[str] = []
    real_listdir = os.listdir
    monkeypatch.setattr(
        os, "listdir", lambda path=".": (listdirs.append(str(path)), real_listdir(path))[1]
    )
    for _ in range(10):
        database.query_attention(query)
    assert listdirs == [], listdirs
