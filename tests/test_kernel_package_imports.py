"""Package-level import isolation: GPU modules must not load eagerly."""

from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> str:
    """Run `code` in a clean subprocess and return combined stdout+stderr."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def test_import_config_leaves_torch_absent() -> None:
    """Importing `dryrun.profiler.kernels.config` must not load torch."""
    output = _run(
        "import dryrun.profiler.kernels.config, sys; "
        "assert 'torch' not in sys.modules, "
        "f'torch was loaded: {[k for k in sys.modules if k.startswith(\"torch\")]}'"
    )
    assert output == "", f"Unexpected output: {output}"


def test_import_attention_leaves_torch_absent() -> None:
    """Importing `dryrun.profiler.kernels.attention` must not load torch."""
    output = _run(
        "import dryrun.profiler.kernels.attention, sys; "
        "assert 'torch' not in sys.modules, "
        "f'torch was loaded: {[k for k in sys.modules if k.startswith(\"torch\")]}'"
    )
    assert output == "", f"Unexpected output: {output}"


def test_import_profile_gemm_lazy_resolves_correctly() -> None:
    """`from dryrun.profiler.kernels import profile_gemm` must resolve without error."""
    # We only check that the import completes and is callable, not that it runs on GPU.
    output = _run(
        "import sys; from dryrun.profiler.kernels import profile_gemm; "
        "assert callable(profile_gemm), 'profile_gemm is not callable'; "
        "assert 'torch' in sys.modules, 'torch was not loaded by profile_gemm import'"
    )
    assert output == "", f"Unexpected output: {output}"


def test_perf_table_writer_reachable_without_torch() -> None:
    """`dryrun.profiler.kernels.PerfTableWriter` must be accessible without torch."""
    output = _run(
        "import dryrun.profiler.kernels as k, sys; "
        "assert hasattr(k, 'PerfTableWriter'), 'PerfTableWriter missing'; "
        "assert 'torch' not in sys.modules, "
        "f'torch was loaded: {[m for m in sys.modules if m.startswith(\"torch\")]}'"
    )
    assert output == "", f"Unexpected output: {output}"
