"""The torch-free boundary of the kernel profiler, asserted per module.

`sweep/`, `grid/`, `config.py` and two of the three `utils/` modules are reached from the cost
path, which prices a kernel from parquet. A simulator run that paid a 4.2-second torch import to
do that would still produce the right numbers, so nothing fails: the cost is invisible until
someone profiles the run. These tests are what make it visible.

**Asserted per module, by walking the directories.** The earlier version of this file named two
modules explicitly, which could not notice a third that started importing torch -- and the
layering introduced several. Discovering the module list from the filesystem means a new grid or
sweep is covered the moment it is added, and a `tests/` change is never needed to extend the
guarantee.

Each module is probed in its own subprocess. In-process, `sys.modules` already holds torch from
whichever sibling test ran first, so the assertion would pass unconditionally under
`pytest-randomly`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "dryrun" / "profiler" / "kernel"

# Modules that must stay free of the GPU stack. `utils/timing.py` is deliberately absent: it owns
# the CUDA event loop, which is why it is the one `utils` module `grid/` and `sweep/` may not
# import.
_TORCH_FREE_DIRECTORIES = ("sweep", "grid")
_TORCH_FREE_FILES = (
    "config.py",
    "utils/model_dims.py",
    "utils/validators.py",
)

# The measurement side. Its loops import torch inside `run()`, so the aggregate package is
# itself cheap to import; the probe below pins that rather than assuming it.
_GPU_DIRECTORIES = ("runner",)


def _module_name(path: Path) -> str:
    """Return the dotted import path of one module under the kernel package."""
    relative = path.relative_to(_PACKAGE_ROOT).with_suffix("")
    parts = [part for part in relative.parts if part != "__init__"]
    return ".".join(("dryrun", "profiler", "kernel", *parts))


def _torch_free_modules() -> list[str]:
    """Discover every module required to import without torch, packages included."""
    names = {_module_name(_PACKAGE_ROOT / name) for name in _TORCH_FREE_FILES}
    for directory in _TORCH_FREE_DIRECTORIES:
        for path in sorted((_PACKAGE_ROOT / directory).glob("*.py")):
            names.add(_module_name(path))
    # The package root itself: it re-exports `KernelProfileConfig`, so importing
    # `dryrun.profiler.kernel` must not drag the runners in either.
    names.add("dryrun.profiler.kernel")
    return sorted(names)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a clean subprocess and capture its output."""
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_the_discovery_found_the_modules_it_claims_to_guard():
    """A typo in a directory name would silently reduce this file to asserting nothing."""
    modules = _torch_free_modules()
    assert len(modules) >= 15, f"Discovered only {len(modules)} torch-free modules: {modules}"
    for expected in (
        "dryrun.profiler.kernel",
        "dryrun.profiler.kernel.config",
        "dryrun.profiler.kernel.sweep",
        "dryrun.profiler.kernel.sweep.gemm",
        "dryrun.profiler.kernel.grid.gemm",
        "dryrun.profiler.kernel.grid.elementwise",
        "dryrun.profiler.kernel.grid.optimizer",
        "dryrun.profiler.kernel.grid.attention_training",
        "dryrun.profiler.kernel.utils.model_dims",
    ):
        assert expected in modules, f"{expected} is not being guarded."


@pytest.mark.parametrize("module", _torch_free_modules())
def test_the_module_imports_without_the_gpu_stack(module: str):
    """Pricing a kernel from parquet must not pay a GPU-stack import."""
    result = _run(
        f"import {module}, sys\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'vllm', "
        "'transformer_engine', 'flashinfer'))\n"
        f"assert not loaded, '{module} pulled in ' + ', '.join(loaded)\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("directory", _GPU_DIRECTORIES)
def test_the_runner_package_defers_its_gpu_imports(directory: str):
    """Even `runner/` imports cleanly: every runner pulls torch inside `run()`.

    Not a requirement so much as a property worth pinning. `runner/__init__` aggregates all six
    runners, so a single module-scope `import torch` anywhere under it would make the aggregate
    expensive, and `cost/kernel/ops/<op>.py` imports a runner module by name inside `measure()`.
    The two backend modules `_vllm_attn` and `_flashinfer_attn` do import torch at module scope,
    which is why they are reached lazily from `runner/attention.py` rather than re-exported here.
    """
    result = _run(
        f"import dryrun.profiler.kernel.{directory} as pkg, sys\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'vllm', "
        "'transformer_engine', 'flashinfer'))\n"
        f"assert not loaded, 'importing {directory} pulled in ' + ', '.join(loaded)\n"
        "assert pkg.KernelRunner is not None\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_op_can_reach_its_runner():
    """Resolve the imports written inside each op's `measure()`, as written.

    This is the check a rename needs, and it has to read the *real* statements rather than a
    hand-kept list beside them: a list would resolve while the body it claims to mirror stayed
    stale. `parse_sweep` is exercised by every config test, but `measure` only runs during a real
    sweep, so a bad module path there surfaces on a GPU after the grid is built -- or on a
    cluster, after a cross-node rendezvous.

    Verified by mutation: rewriting one op's `measure` import back to the old `profiler.kernels`
    path makes this fail, where resolving the runners directly did not.

    Run in a subprocess because resolution loads torch, which the torch-free probes above must
    not find in `sys.modules`.
    """
    probe = """
import ast
import importlib
import inspect
import sys
import textwrap

import dryrun.cost.kernel.ops as _  # populates the registry
from dryrun.cost.kernel.ops.base import OP_REGISTRY


def lazy_imports(function):
    \"\"\"Return the ImportFrom nodes in a function, following a one-hop module-level delegate.

    The two attention ops share one `_measure_attention` helper, so their own `measure` bodies
    hold no import. Walking into a named module-level callee covers that without hardcoding it.
    \"\"\"
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    found = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    if found:
        return found, function
    module = sys.modules[function.__module__]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            delegate = getattr(module, node.func.id, None)
            if callable(delegate) and getattr(delegate, "__module__", None) == module.__name__:
                return lazy_imports(delegate)
    return [], function


checked = 0
for op in sorted(OP_REGISTRY.values(), key=lambda o: o.name):
    imports, source_fn = lazy_imports(type(op).measure)
    assert imports, f"{op.name}.measure has no import to resolve"
    owner = source_fn.__module__          # e.g. dryrun.cost.kernel.ops.optimizer
    for node in imports:
        # A `level` of N inside module `a.b.c.d` strips N trailing segments of that module's
        # dotted name: one dot means the containing package. Resolved by hand rather than by
        # passing the module name to import_module, whose `package` argument is treated as a
        # package and so lands one level too deep for a module.
        parts = owner.split(".")[: -node.level]
        target = ".".join(parts + ([node.module] if node.module else []))
        module = importlib.import_module(target)
        for alias in node.names:
            assert hasattr(module, alias.name), (
                f"{op.name}.measure imports {alias.name!r} from {target}, "
                f"which does not define it"
            )
            checked += 1

assert checked >= 12, f"resolved only {checked} measure-body imports across 12 ops"
print(f"resolved {checked}")
"""
    result = _run(probe)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "resolved" in result.stdout


def test_the_package_root_exports_only_the_config():
    """The 28-name lazy table is gone; callers name the module they want.

    A wide root is what the PEP 562 `__getattr__` existed to make safe, and keeping both the
    layering and the lazy table would leave two reachable paths to every symbol.
    """
    import dryrun.profiler.kernel as kernel

    assert kernel.__all__ == ["KernelProfileConfig"]
    assert not hasattr(kernel, "__getattr__"), (
        "The lazy-import table should have been removed with the flat layout."
    )
    for gone in ("profile_gemm", "GemmShape", "PerfTableWriter", "transformer_gemm_shapes"):
        assert not hasattr(kernel, gone), f"{gone} should no longer be a package-root export."
