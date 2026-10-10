"""Verify UndoLog, KvBlock, and FreeBlockList via the pybind11 boundary.

Since UndoLog and FreeBlockList are internal (not exposed to Python), we test
them indirectly through CppInstance in Task 5. This file is a placeholder that
verifies compilation and will be expanded when CppInstance is available.

For now, just verify the module still imports after adding the undo sources.
"""
import pytest

try:
    from dryrun._engine_cpp import version
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ engine not compiled")


def test_module_still_loads_after_undo_sources():
    assert version() == "0.1.0"
