# Writing documentation

This page is the maintainer guide for the DryRun-RL Sphinx site. It is a page in the toctree and
is built as part of the site, because a maintainer guide that is never rendered goes stale at the
same rate as the site it is supposed to govern.

---

## Build commands

```bash
# Install documentation dependencies.
pip install -r docs/requirements.txt

# Build the HTML site.
make -C docs html

# Build with warnings as errors (what CI should run).
make -C docs strict

# Remove the build directory.
make -C docs clean

# Auto-reloading local server (requires sphinx-autobuild).
make -C docs livehtml
```

`make strict` passes `-W --keep-going` to `sphinx-build`. Warnings become errors, and the build
continues past the first error to collect all problems at once. An orphaned page (a page not
appearing in any toctree) and a broken cross-reference (`{doc}` pointing at a nonexistent file)
both fail this build. Run `make strict` rather than `make html` before merging documentation
changes.

`make livehtml` wraps `sphinx-autobuild` and starts a local server that reloads whenever a source
file changes. The `livehtml` target is defined in `docs/Makefile`.

---

## Site layout

The site source lives under `docs/`. One directory per section, each with an `index.md` that
carries real orientation content and a `toctree` directive listing the pages in that section.
Adding a page without placing it in a toctree causes `make strict` to fail with an orphan warning.

```
docs/
├── index.md                     <- landing page toctree
├── getting-started/
├── overview/
├── profiling/
├── cost-models/
├── workload/
├── simulation/
├── development/                 <- this section
└── reference/
```

`docs/superpowers/` and `docs/design/` hold AI scratch and design documents. Both directories are
in `exclude_patterns` in `docs/conf.py` and are not rendered. They are kept as source material,
and they are gitignored, so a fresh clone will not have them at all — never `{doc}`-reference
anything inside them, and do not cite a path under `.superpowers/` as evidence for a published
number.

When a page is superseded, delete it and its `exclude_patterns` entry in the same commit.
Excluding a stale file leaves two versions of the same content on disk with only a comment
distinguishing them, and the comment is what goes stale next. `docs/composed-rollout.md` and
`docs/kernel-ops.md` were handled this way once their content reached
`cost-models/composed-rollout.md`, `development/kernel-ops.md` and
`development/model-families.md`.

---

## The single-owner rule for facts

Every measured number lives on exactly one page. Other pages may state one sentence that names the
result and link to the owner. The motivation is concrete: before this rule was established, the
kernel holdout accuracy table existed in three files simultaneously and had three inconsistent
versions because only one of them was regenerated after the floor-fitting fix.

Current ownership map:

| Fact | Owner |
|---|---|
| Kernel holdout MRE, p90, and gate results | `profiling/kernels.md` |
| `ComposedRollout` end-to-end accuracy | `cost-models/composed-rollout.md` |
| The Hydraulis latency formula and coefficients | `cost-models/training-megatron.md` |
| Extrapolation semantics and domain-gate behaviour | `simulation/domain-gates.md` |
| Measurement sweep design and identifiability requirements | `profiling/methodology.md` |
| All accuracy statuses (pass/fail per model and GPU) | `reference/validation.md` |
| Known gaps, deferred work, and acceptance criteria | `reference/limitations.md` |

When you add a number to this site, decide which page owns it and link from all others. Do not
copy-paste the number into a second page "for convenience" — convenience turns into contradiction
on the first regeneration.

---

## Conventions that break the build

**JSON fences with comments or elisions must be labelled `javascript`.** The Sphinx JSON lexer
errors on `//` comments and `...` elisions. Since `make strict` treats warnings as errors, a JSON
fence containing either of those constructs fails the build. Use

````
```javascript
{
  "key": "value",  // comment explaining the field
  ...
}
```
````

instead of ` ```json `. This is not a workaround; the `javascript` lexer handles JSON correctly
and supports comments.

**Cross-references must use `` {doc}`path` ``.** Raw relative Markdown links (`[text](../foo.md)`)
are not validated by Sphinx. A broken raw link ships silently and creates a dead link in the
rendered site. A broken `{doc}` reference fails `make strict`. Use:

```
{doc}`../profiling/kernels`
{doc}`kernel-ops`
```

Paths are relative to the containing file. Within a directory, use the bare filename without
extension. Across directories, use the path from the project `docs/` root with a leading `../`.

**Every page must appear in a toctree.** An orphan page is a page that no toctree references. The
strict build flags orphans as warnings-turned-errors. If you add a page, add it to the nearest
`index.md` toctree in the same directory.

**`literalinclude` paths are relative to the containing file.** A `literalinclude` that references
`../../examples/profile/vllm/qwen3_8b_tp1.yaml` from `docs/profiling/vllm.md` works; the same
path from `docs/index.md` does not.

---

## No absolute paths

Commands in the documentation use `$MODEL_PATH` and repo-relative paths, never absolute paths like
`/path/to/checkpoint` or `/home/user/DryRun-RL`. An absolute path works on exactly one machine.
Export `MODEL_PATH` before running any profiling command:

```bash
export MODEL_PATH=/path/to/Qwen3-8B
```

All example configs and scripts read the checkpoint path from this variable.
