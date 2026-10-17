# Project Contract

## ALWAYS

- Apply first-principles thinking. Do not assume that I always have a clear understanding of what I want or how to achieve it. Stay cautious and start from the fundamental needs and problem. If the motivation or objective is unclear, pause and discuss it with me.
- When running scripts or inspecting the environment, please activate the conda environment by executing `source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh`. All dependencies and packages are installed within this environment.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.
- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.

## Coding Guidelines

Two reference files live under `.claude/`:

- **`.claude/coding-style.md`**: formatting rules, naming conventions, docstrings, logging, and annotation markers. Ordered by risk (silent-bug rules first).
- **`.claude/codebase-map.md`**: system architecture, directory tree, configuration hierarchy, quick-lookup indices, and import dependency graphs.

Claude must read and apply these guides when writing or modifying code.

## Compact Instructions

When compressing, preserve in priority order:

1. Architecture decisions (NEVER summarize)
2. Modified files and their key changes
3. Current verification status (pass/fail)
4. Open TODOs and rollback notes
5. Tool outputs (can delete, keep pass/fail only)
