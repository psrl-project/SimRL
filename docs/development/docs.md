# Writing documentation

The site is Sphinx with MyST Markdown and the pydata theme. `docs/conf.py` holds the
configuration. This page covers building it, adding a page, and the two rules that turn a prose
mistake into a failed build.

---

## Build

```bash
pip install -r docs/requirements.txt
make -C docs html
make -C docs strict
make -C docs clean
make -C docs livehtml
```

`make -C docs html` builds the site into `docs/_build/html`. Open
`docs/_build/html/index.html` to read it.

`make -C docs strict` builds with `-W --keep-going`: warnings become errors and the build
continues past the first one so all problems are reported together. A page in no toctree, a broken
A cross-reference written with the doc role, an unknown lexer and a duplicated label all fail it.
Run this, not `html`, before merging a documentation change.

`make -C docs livehtml` runs `sphinx-autobuild` for a reloading local server. It needs
`sphinx-autobuild`, which is not in `docs/requirements.txt`.

Only `docs/` is built. `docs/index.md` and each section's `index.md` own a toctree.

---

## Site layout

```text
docs/
├── conf.py
├── Makefile, requirements.txt
├── index.md               Landing page and the top-level toctree
├── getting-started/       Installation, concepts, the staged walkthrough
├── workload/              Trace collection
├── profiling/             Profiling pipelines: vLLM, kernels, Megatron, FSDP, collectives
├── cost-models/           Rollout, training and kernel models
├── simulation/            Policies, routing, prefix cache, backends, outputs, domain gates
├── reference/             CLI, configuration, schemas, validation, limitations
├── development/           This section
├── design/                Design notes. Excluded from the site
└── superpowers/           Working notes. Excluded from the site
```

`exclude_patterns` in `docs/conf.py` drops `_build`, `superpowers` and `design`. The last two hold
working notes that are not part of the published site and may not exist in a clone. Never
Reference a page inside them with the doc role and never cite one as the source of a published
figure.

Adding a section means creating its `index.md` with orientation content and a toctree, adding the
section to the top-level toctree in `docs/index.md`, and adding the directory to `exclude_patterns`
only if it is not meant to be read.

---

## Adding or changing a page

1. Write the page under the section it belongs to, with a title as the first heading.
2. Add it to that section's `index.md` toctree. A page in no toctree is an orphan, and the strict
   build fails on it.
3. State every fact once, on the page that owns it. See below.
4. Run `make -C docs strict`.

A page that is superseded should be deleted, not excluded. An excluded file leaves two versions
on disk with a comment distinguishing them, and the comment is what goes stale next.

Changing a console script or a config key means changing the page that documents it in the same
change. {doc}`../reference/cli` and {doc}`../reference/configuration` are written from the code,
not from memory.

---

## Conventions that break the build

**A JSON fence with comments or elisions must be labelled `javascript`.** The JSON lexer rejects
`//` comments and `...`, and the strict build treats that as an error. The JavaScript lexer renders
JSON correctly and accepts both.

````
```javascript
{
  "key": "value",  // comment
  "other": "..."   // an elision
}
```
````

**Cross-references use the MyST doc role.** A raw Markdown link is not checked by Sphinx, so a
broken one ships silently. The doc role is resolved at build time and fails the strict build when
it points at no page.

```markdown
{doc}`kernel-ops`                       Same directory.
{doc}`../reference/validation`          From this directory to another section.
```

The path is relative to the containing file and omits the `.md` suffix.

**`literalinclude` paths are relative to the containing file.** Include the YAML defaults from
`docs/reference/configuration.md` as `../../srl/config/defaults/simulate.yaml`. The same path from
`docs/index.md` would not resolve.

**Every heading up to level 3 gets an anchor** (`myst_heading_anchors = 3`), so a heading can be
linked with `{ref}` or a fragment. Two identical headings on one page collide.

---

## Prose

`.claude/coding-style.md` governs prose here as it governs comments in code.

- **No semicolons.** Split the sentence or use a conjunction.
- **No em dashes or en dashes.** Use a period, a comma, a colon or parentheses. For a numeric range
  write `512 to 8192` or `512-8192`.
- **No absolute paths** in commands. Use `$MODEL_PATH` and paths relative to the repository. An
  absolute path works on one machine.
- **Sentence case in headings**, and one idea per paragraph.
- **Name the file or symbol** a claim rests on, so the next reader can check it.

These are checkable:

```bash
grep -rn ';' docs --include=*.md          # inspect each hit; code may contain semicolons
grep -rn '[—–]' docs --include=*.md
grep -rn '/apdcephfs\|/home/\|/path/to' docs --include=*.md
```

The second and third should return nothing outside a code block that quotes a real path on
purpose.

---

## One owner per fact

A measured number lives on exactly one page. Every other page names the result in a sentence and
links to the owner. When the number changes, one page changes and no other page can contradict it.

| Kind of fact | Owner |
|---|---|
| Console flags and environment variables | `reference/cli.md` |
| Config groups, keys and defaults | `reference/configuration.md` |
| `ProfileRecord`, `CostArtifact`, kernel tables, telemetry streams | `reference/schemas.md` |
| Accuracy figures, sample counts and gates | `reference/validation.md` |
| What the code does not do | `reference/limitations.md` |
| Package tree and dependency direction | `development/architecture.md` |

A number on any other page is a summary and must link to `reference/validation.md`.
