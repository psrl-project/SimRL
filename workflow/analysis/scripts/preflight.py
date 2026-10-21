"""Check that this session can produce a diagnosis, before any replay.

Every check is deterministic and read-only. Exit code is non-zero when a blocking check fails, so
an agent can gate on it.

Usage:
    python workflow/analysis/scripts/preflight.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Running this file directly puts its own directory at sys.path[0], which would shadow the
# standard library `select` module with the sibling script/select.py and break subprocess
# and matplotlib imports below.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [entry for entry in sys.path if entry not in ("", _SCRIPT_DIR)]

SCRATCH_ROOT = Path("/tmp/simrl_analysis")


def _check_workspace(blocking: list[str], report: dict) -> None:
    """Confirm SIMRL_WORKSPACE is set and holds env/simrl.sh."""
    value = os.environ.get("SIMRL_WORKSPACE")
    if not value:
        blocking.append("SIMRL_WORKSPACE is not set. Export it to the workspace directory.")
        return
    workspace = Path(value)
    report["workspace"] = str(workspace)
    env_script = workspace / "env" / "simrl.sh"
    report["env_script"] = str(env_script)
    if not env_script.is_file():
        blocking.append(f"Environment script not found: {env_script}")


def _check_diagnosis_import(blocking: list[str]) -> None:
    """Confirm the diagnosis library this unit calls actually imports."""
    try:
        import srl.diagnosis  # noqa: F401
    except ImportError as exc:
        blocking.append(f"srl.diagnosis does not import: {exc}")


def _check_matplotlib_agg(blocking: list[str], report: dict) -> None:
    """Confirm matplotlib imports and accepts the headless Agg backend."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        report["matplotlib_backend"] = matplotlib.get_backend()
    except ImportError as exc:
        blocking.append(f"matplotlib does not import: {exc}")


def _check_scratch_root(blocking: list[str], report: dict) -> None:
    """Confirm the local scratch root for replay traces is writable."""
    report["scratch_root"] = str(SCRATCH_ROOT)
    try:
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        probe = SCRATCH_ROOT / ".preflight_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        blocking.append(f"Scratch root {SCRATCH_ROOT} is not writable: {exc}")


def main() -> int:
    """Run every check, print a JSON report, and return the process exit code."""
    blocking: list[str] = []
    warnings: list[str] = []
    report: dict = {}

    _check_workspace(blocking, report)
    _check_diagnosis_import(blocking)
    _check_matplotlib_agg(blocking, report)
    _check_scratch_root(blocking, report)

    report["blocking"] = blocking
    report["warnings"] = warnings
    print(json.dumps(report, indent=2))
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
