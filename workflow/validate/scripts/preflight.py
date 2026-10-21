"""Check that this session can produce reportable simulator numbers, before any run.

Every check is deterministic and read-only. Exit code is non-zero when a blocking check fails, so
an agent can gate on it. Warnings do not block but must be quoted in the final report.

Usage:
    python workflow/validate/scripts/preflight.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sim  # noqa: E402


def _git(*args: str) -> str:
    """Return stdout of a git command run at the repository root."""
    return subprocess.run(
        ["git", *args], cwd=run_sim.ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def main() -> int:
    """Run every check, print a JSON report, and return the process exit code."""
    blocking: list[str] = []
    warnings: list[str] = []
    report: dict = {}

    try:
        report["workspace"] = str(run_sim.workspace())
        report["env_script"] = str(run_sim.activate_environment())
    except (RuntimeError, FileNotFoundError) as exc:
        blocking.append(str(exc))

    if not run_sim.SIMULATE.is_file():
        blocking.append(f"simulate.sh not found: {run_sim.SIMULATE}")

    md5 = run_sim.binary_md5()
    report["binary_md5"] = md5
    if not md5:
        blocking.append("No compiled simulator extension (*.so) found under srl/.")

    kv = os.environ.get("SIMRL_KV_CACHE")
    report["kv_cache"] = kv
    if not kv:
        warnings.append("SIMRL_KV_CACHE is unset. Every run re-probes KV capacity.")
    elif not Path(kv).is_file():
        warnings.append(f"SIMRL_KV_CACHE={kv} does not exist yet. The first run will probe and write it.")

    dirty = [line for line in _git("status", "--porcelain").splitlines() if "srl/" in line]
    report["git_head"] = _git("rev-parse", "--short", "HEAD")
    report["dirty_srl_files"] = dirty
    if dirty:
        warnings.append("Uncommitted changes under srl/. Record them, the baseline is this same tree.")

    report["blocking"] = blocking
    report["warnings"] = warnings
    print(json.dumps(report, indent=2))
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
