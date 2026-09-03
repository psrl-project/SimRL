"""GEMM shape arithmetic must be importable without a GPU stack."""

from __future__ import annotations

import subprocess
import sys

_PROBE = """
import sys
import dryrun.profiler.kernels.shapes  # noqa: F401
assert "torch" not in sys.modules, "importing shapes pulled in torch"
assert "vllm" not in sys.modules, "importing shapes pulled in vllm"
print("clean")
"""


def test_importing_the_shape_module_does_not_import_torch():
    """A 4.2-second torch import has no place in a CPU-only cost-model path."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_the_shape_functions_are_reachable_from_both_modules():
    """gemm.py must keep re-exporting them so existing importers do not break."""
    from dryrun.profiler.kernels import gemm, shapes

    for name in (
        "GemmShape",
        "gemm_all_shapes",
        "gemm_sweep_shapes",
        "gemm_wgrad_shapes",
        "logits_gemm_shape",
        "logits_sweep_shapes",
        "transformer_gemm_shapes",
    ):
        assert getattr(gemm, name) is getattr(shapes, name)
