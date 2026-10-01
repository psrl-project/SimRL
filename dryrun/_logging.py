"""Central logging setup for DryRun-RL.

One place configures the ``"dryrun"`` logger. The formatter follows the
timestamp + source-location convention used by most mature logging setups
(psrl/verl included): one line per record with a millisecond timestamp,
level, logger name, PID, and ``file:line``, so any `dryrun_logger.info(...)`
call can be traced back to its call site without a debugger, e.g.

    2026-08-16 13:42:07.123 [INFO] dryrun (pid=12345) simulator.py:214 - ...

Every module in this package should log through the shared `dryrun_logger`
instance below (``from dryrun._logging import dryrun_logger``) rather than
calling `logging.getLogger("dryrun")` itself, so there is exactly one logger
to configure. This module never touches the root logger, so it plays nicely
whether or not the host process (e.g. Hydra) configures its own logging.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

dryrun_logger = logging.getLogger("dryrun")

_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s (pid=%(process)d) %(filename)s:%(lineno)d - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _coerce_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    try:
        return getattr(logging, level.upper())
    except AttributeError as exc:
        raise ValueError(f"Unknown log level: {level!r}.") from exc


def setup_logging(
    level: int | str | None = None,
    log_file: str | Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Attach a timestamped, source-located formatter to `dryrun_logger`.

    Idempotent by default: repeated calls (e.g. once per CLI entrypoint, or
    from several test modules importing this package) do not stack duplicate
    handlers unless `force=True`.

    Args:
        level: Log level name (e.g. `"DEBUG"`) or numeric level. Defaults to
            the `DRYRUN_LOG_LEVEL` environment variable, then `"INFO"`.
        log_file: If given, also write logs to this file (in addition to
            stderr): e.g. alongside a run's JSONL telemetry under
            `output_dir/run.log`. Parent directories are created as needed.
        force: Reconfigure even if `setup_logging` already ran once.

    Returns:
        The configured `dryrun_logger`.
    """
    global _configured
    if _configured and not force:
        return dryrun_logger

    resolved_level = _coerce_level(level if level is not None else os.environ.get("DRYRUN_LOG_LEVEL", "INFO"))
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))

    dryrun_logger.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        dryrun_logger.addHandler(handler)
    dryrun_logger.setLevel(resolved_level)
    # This logger owns its own output end to end; don't also emit through
    # the root logger's handlers (e.g. a host application's own config).
    dryrun_logger.propagate = False

    _configured = True
    return dryrun_logger
