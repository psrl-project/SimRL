"""Run one declared experiment arm and record its output directory.

This module assembles a `simulate.sh` invocation and executes it. It performs no performance
arithmetic. Gates and summaries read the simulator output afterwards.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SIMULATE = ROOT / "examples" / "simulate" / "simulate.sh"


@dataclass(frozen=True)
class Arm:
    """One `simulate.sh` invocation.

    Attributes:
        name: Directory name for this arm under the experiment output root.
        env: `simulate.sh` environment variables, all strings.
        overrides: Extra Hydra overrides appended through `EXTRA_OVERRIDES`.
    """

    name: str
    env: dict[str, str] = field(default_factory=dict)
    overrides: tuple[str, ...] = ()

    def environ(self, output_dir: Path) -> dict[str, str]:
        """Return the full environment for this arm at `output_dir`."""
        out = dict(self.env)
        out["OUTPUT_DIR"] = str(output_dir)
        if self.overrides:
            out["EXTRA_OVERRIDES"] = " ".join(self.overrides)
        return out


def workspace() -> Path:
    """Return the workspace directory named by `SIMRL_WORKSPACE`.

    The workspace is the directory that holds `env/simrl.sh` next to the repository checkout.

    Raises:
        RuntimeError: When `SIMRL_WORKSPACE` is unset or is not a directory.
    """
    value = os.environ.get("SIMRL_WORKSPACE")
    if not value:
        raise RuntimeError("SIMRL_WORKSPACE is not set; export it to the workspace directory.")
    path = Path(value)
    if not path.is_dir():
        raise RuntimeError(f"SIMRL_WORKSPACE={value!r} is not a directory.")
    return path


def activate_environment() -> Path:
    """Return the environment script a subprocess should source.

    Raises:
        RuntimeError: When `SIMRL_WORKSPACE` is unset or invalid.
        FileNotFoundError: When `$SIMRL_WORKSPACE/env/simrl.sh` does not exist.
    """
    script = workspace() / "env" / "simrl.sh"
    if not script.is_file():
        raise FileNotFoundError(f"Environment script not found: {script}")
    return script


def binary_md5() -> dict[str, str]:
    """Hash every compiled simulator extension, to detect an in-place rebuild mid-sweep."""
    return {
        str(path.relative_to(ROOT)): hashlib.md5(path.read_bytes()).hexdigest()
        for path in sorted(ROOT.glob("srl/**/*.so"))
    }


def run_arm(arm: Arm, output_root: Path, *, timeout: float | None = None) -> dict:
    """
    Execute one arm and return a record of where it wrote its output.

    Args:
        arm (Arm): The invocation to run.
        output_root (Path): Experiment directory. The arm writes to `output_root / arm.name`.
        timeout (float | None): Seconds before the run is killed, None for no limit.

    Returns:
        dict: Arm name, output directory, exit code, and the binary hashes before the run.

    Raises:
        subprocess.CalledProcessError: When `simulate.sh` exits non-zero.
    """
    env_script = activate_environment()
    out_dir = output_root / arm.name
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **arm.environ(out_dir)}
    command = f"source {env_script} >/dev/null 2>&1; bash {SIMULATE}"
    record = {
        "arm": arm.name,
        "output_dir": str(out_dir),
        "kv_cache": env.get("SIMRL_KV_CACHE"),
        "md5_before": binary_md5(),
    }
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=ROOT,
        env=env,
        timeout=timeout,
        check=False,
        capture_output=True,
        text=True,
    )
    (out_dir / "simulate.stdout").write_text(completed.stdout)
    (out_dir / "simulate.stderr").write_text(completed.stderr)
    record["returncode"] = completed.returncode
    record["md5_after"] = binary_md5()
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)
    return record


@dataclass(frozen=True)
class Experiment:
    """A named set of arms that share a workload.

    Attributes:
        name: Output directory name under `outputs/simulate`.
        shared: Environment every arm starts from.
        arms: The arms to run.
    """

    name: str
    shared: dict[str, str]
    arms: tuple[Arm, ...]

    def output_root(self) -> Path:
        """Return this experiment's output root."""
        return ROOT / "outputs" / "simulate" / self.name

    def resolved_arms(self) -> tuple[Arm, ...]:
        """Return every arm with `shared` folded in, arm values winning."""
        return tuple(dataclasses.replace(arm, env={**self.shared, **arm.env}) for arm in self.arms)
