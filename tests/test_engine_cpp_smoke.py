# tests/test_engine_cpp_smoke.py
"""Verify the C++ engine module compiles, loads, and exposes its version."""

import pytest

try:
    from dryrun._engine_cpp import version
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ engine not compiled")


def test_version_returns_a_string():
    v = version()
    assert isinstance(v, str)
    assert len(v) > 0
