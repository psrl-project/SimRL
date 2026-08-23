"""Emit Hydra workload overrides from a prompt/output distribution YAML file."""

from __future__ import annotations

import argparse
import sys

from omegaconf import DictConfig, OmegaConf


def emit_overrides(path: str) -> list[str]:
    """
    Flatten `prompt` and `output` mappings into Hydra dotted overrides.

    Args:
        path (str): YAML file with `prompt` and `output` distribution blocks.

    Returns:
        list[str]: Overrides such as `workload.prompt.hi=8192`.
    """
    cfg = OmegaConf.load(path)
    overrides: list[str] = []
    for dim in ("prompt", "output"):
        if dim not in cfg:
            raise SystemExit(f"Distribution config is missing {dim!r}.")
        node = cfg[dim]
        if not isinstance(node, (DictConfig, dict)):
            raise SystemExit(f"Distribution field {dim!r} must be a mapping.")
        for key, value in node.items():
            overrides.append(f"workload.{dim}.{key}={value}")
    return overrides


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print Hydra overrides for a workload YAML.")
    parser.add_argument("path", help="Path to distributions.yaml.")
    args = parser.parse_args(argv)
    sys.stdout.write("\n".join(emit_overrides(args.path)) + "\n")


if __name__ == "__main__":
    main()
