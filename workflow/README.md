# Workflow

Each subfolder is one validation unit that can be handed to a subagent and run end to end.

| Folder | What it does |
|---|---|
| `validate/` | Decide whether a scheduling or system change improves end-to-end performance, using real simulator runs. |
| `analysis/` | Diagnose where span goes in one run. Replays a selected `validate` row with full telemetry and attributes its idle time. |

## Conventions

- A unit is self-contained: its `SKILL.md` is the entry point, its `scripts/` hold the code, its `reference/` holds the facts an agent needs.
- `.claude/skills/<name>/SKILL.md` and `.claude/agents/<name>.md` are thin pointers to the unit's `SKILL.md`, because only those paths are discovered automatically. Keep the content in `workflow/` and never duplicate it.
- Generated experiment scripts go in the unit's `experiments/`, which is gitignored.
- Results go under `outputs/`, never into `workflow/`.
- Paths come from the repository root (derived from the file location) or from `SIMRL_WORKSPACE`,
  the directory that holds `env/` next to the checkout. Never write an absolute path. A missing
  variable is an error, not a fallback.

## Adding a unit

1. Create `workflow/<unit>/SKILL.md` with the non-negotiable rules first, then the steps, then a
   reference table. Move facts into `reference/`, deterministic work into `scripts/`.
2. Add a `scripts/preflight.py` if the unit depends on the environment. Make it read-only and exit
   non-zero on a blocking failure.
3. Put the agent prompt in `workflow/<unit>/AGENT.md` (input contract, output contract, stop
   conditions) and cross-run lessons in `workflow/<unit>/memory/`.
4. Add the two pointers: `.claude/skills/<name>/SKILL.md` and `.claude/agents/<name>.md`. They hold
   only frontmatter (the agent's `skills:` preloads the pointer skill) and one line naming the
   file in `workflow/`. Do not use the `memory:` frontmatter field, it writes under `.claude/`.
