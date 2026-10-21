# Lessons

Things a later run would otherwise rediscover: a check that fired and why, a row that is
unreplayable, a command that needed a workaround, a telemetry cost worth knowing before enabling a
stream. One bullet each, newest last, with the date. **Never record result numbers.** Remove a
bullet when it stops being true.

- 2026-10-22: Running `python workflow/analysis/scripts/preflight.py` directly puts
  `workflow/analysis/scripts/` at the front of `sys.path`, so the sibling `scripts/select.py`
  shadows the standard library `select` module and breaks any subsequent `import subprocess` or
  `import matplotlib`. `preflight.py` strips its own directory from `sys.path` before importing
  anything. Keep that guard if this file is edited.
