# Coding Style Guide

All new and modified code in SimRL must follow these conventions.
Sections are ordered by risk: rules that cause **silent runtime bugs** come first,
cosmetic formatting rules come last.

## Quick Decision Table

When writing a string, ask: **"Is this string a name the program looks up, or a message a human reads?"**

| Writing a… | Capitalize? | Period? | Backticks for identifiers? | `!r` for values? |
|------------|-------------|---------|---------------------------|-------------------|
| **Identifier string** (hasattr, getattr, ==, dict key, Ray method name, enum value) | No change | **NEVER** | N/A | N/A |
| **Assertion message** | Uppercase | Yes `.` | No (plain text) | Yes |
| **Log message** | Uppercase | Yes `.` / `...` / `!` | No (plain text) | Yes |
| **Comment: sentence** | Uppercase | Yes `.` | Yes `` `id` `` | N/A |
| **Comment: fragment** | lowercase OK | No | Yes `` `id` `` | N/A |
| **Annotation marker** | Uppercase | Yes `.` | Yes `` `id` `` | N/A |
| **Docstring prose** | Uppercase | Yes `.` | Yes `` `id` `` | N/A |
| **Docstring arg desc** | Uppercase | `.` optional | Yes for *other* identifiers | N/A |

For comments and docstrings, §5.0 also bounds *what* they may say (no history,
no measurements) and §5.0.1 bans `;` and dashes in all prose.

---

## §1. Identifier Strings vs. Human-Readable Messages ⚠️

> **This is the single most dangerous mistake in this codebase.
> Adding a period to an identifier string causes silent runtime failures : 
> no exception, no warning, just broken behavior.**

Strings fall into exactly two categories:

- **Identifier strings**: looked up or compared by the program:
  `hasattr(obj, "attr")`, `getattr`, `setattr`, `delattr`,
  `mode == "nixl_cpu"`, `unit in ("GB", "MB")`,
  Ray calls (`execute_all_async("method_name")`),
  dict keys, config keys, enum values.
  → **Never** add a period. **Never** change casing. Copy the name exactly.

- **Human-readable messages**: shown in logs, assertion failures, comments, docstrings.
  → Capitalize and punctuate per §2.

**Decision test**: if the string is passed to `==`, `in`, `hasattr`, `getattr`,
`execute_all_async`, `execute_rank_zero_async`, or used as a dict/config key,
it is an identifier. Everything else is a message.

```python
# GOOD: identifier strings are exact; message strings have period
assert hasattr(self, "device_mesh"), "device_mesh is not initialized."
assert self.srl_config.ps_mode in ("nixl_cpu", "nixl_gpu"), (
    f"Expected nixl_cpu or nixl_gpu, got {self.srl_config.ps_mode!r}."
)
self.ps_wg.execute_all_async("get_nixl_train_storage_client_name")
if config.routing_strategy.method == "throughput_balance":
    ...

# BAD: period leaked into identifier (silently broken!)
assert hasattr(self, "device_mesh."), "device_mesh is not initialized."
self.ps_wg.execute_all_async("get_nixl_train_storage_client_name.")
if config.routing_strategy.method == "throughput_balance.":
    ...
```

---

## §2. Capitalization, Punctuation & Identifier References

Three rules govern all human-readable text:

### 2.1 Sentence vs. Fragment

> **Sentence** (subject + verb) → **Uppercase** first letter + **period** `.`
> **Fragment** (no verb, short label) → **lowercase** OK + **no period**
> **When in doubt, treat it as a sentence.**

| Context | First letter | Ending | Example |
|---------|-------------|--------|---------|
| Comment: sentence | Uppercase | `.` | `# Set the rollout number based on batch size.` |
| Comment (fragment | lowercase OK |) | `# worker-side attributes` |
| Annotation marker | Uppercase | `.` | `# TODO(lhy): Add timeout handling.` |
| Docstring summary | Uppercase | `.` | `"""Wait for the NIXL push to complete."""` |
| Docstring arg desc | Uppercase | `.` optional | `timeout (float): Max wait time in seconds` |
| Assertion message | Uppercase | `.` | `"nixl_storage_client is not initialized."` |
| Log: normal | Uppercase | `.` | `"Initialized PS successfully."` |
| Log: ongoing | Uppercase | `...` | `"Getting the current PS model version..."` |
| Log: milestone | Uppercase | `!` | `"All configuration checks passed!"` |

### 2.2 After Colons

Text after `MARKER(author):` and `[prefix]:` always starts **uppercase**:

```python
# TODO(lhy): Add timeout handling for fault tolerance.     # GOOD
# TODO(lhy): add timeout handling for fault tolerance.     # BAD

srl_logger.warning(f"[{log_prefix}]: Failed to log {name}: {e}.")   # GOOD
srl_logger.warning(f"[{log_prefix}]: failed to log {name}: {e}.")   # BAD
```

### 2.3 Forbidden Endings

Never end a comment or message with `;` or `,`.

### 2.4 Backticks for Identifier References

The rule depends on whether the text is **read by a developer** (source code)
or **seen at runtime** (terminal output):

- **Comments & docstrings** (developer reads source) → wrap identifiers in backticks: `` `identifier` ``
- **Log messages & assertion messages** (shown at runtime) → plain text, no backticks

```python
# GOOD: backticks in comments
# The `rollout_coordinator` dispatches requests to `GenWorker` instances.

# BAD: identifier blends into prose
# The rollout_coordinator dispatches requests to GenWorker instances.

# GOOD: plain text in logs and assertions
srl_logger.debug("Calling pull_model_state_dict_nixl for model update.")
assert self.nixl_storage_client is not None, "nixl_storage_client is not initialized."
```

In docstrings, use backticks in prose paragraphs. In `Args` / `Returns` / `Raises`,
the parameter name itself needs no backticks, but references to *other* identifiers do:

```python
def push_model(self, version: int) -> None:
    """
    Push the model state dict to `PSStorageWorker` via NIXL.

    Args:
        version (int): The target model version to push.
            Passed to `PSManager.bump_version` after the push completes.
    """
```

### 2.5 `!r` for Ambiguous Dynamic Values

In log messages and assertion messages, use `!r` for any dynamic value whose type
might be ambiguous (strings, None, enums). This makes debugging output unambiguous:

```python
srl_logger.error(f"Unexpected status {status!r} for request {request_id}.")
assert self.srl_config.ps_mode in ("nixl_cpu", "nixl_gpu"), (
    f"Expected nixl_cpu or nixl_gpu, got {self.srl_config.ps_mode!r}."
)
```

---

## §3. Line Length & Line Breaking

**Hard limit: 119 characters** (enforced by Ruff).
**Never** use `\` for line continuation: always use parentheses.

### 3.1 Function Signatures

One parameter per line, 4-space indent, trailing comma, closing `)` on its own line.

```python
# GOOD
def push_model_state_dict_nixl(
    self,
    key: str,
    shards_to_transfer: list[int],
    target_client_name: str,
    next_ps_model_version: int,
) -> None:
    ...

# BAD
def push_model_state_dict_nixl(self, key: str, shards_to_transfer: list[int], target_client_name: str, next_ps_model_version: int) -> None:
```

### 3.2 Function Calls

Same pattern. Grouping on fewer lines is acceptable when only slightly over the limit.

```python
# GOOD: one per line
result = some_function(
    first_arg,
    second_arg,
    third_arg=value,
)

# GOOD: grouped, still under limit per line
result = some_function(
    first_arg, second_arg, third_arg=value,
)
```

### 3.3 Strings & f-strings

Use **implicit string concatenation** inside parentheses.

```python
# GOOD
srl_logger.info(
    f"Pushing key {key} shards {shards_to_transfer} to {target_client_name} "
    f"for version {next_ps_model_version} with {len(shards_to_transfer)} shards."
)

# BAD: backslash continuation
srl_logger.info(
    f"Pushing key {key} shards {shards_to_transfer} to {target_client_name} " \
    f"for version {next_ps_model_version} with {len(shards_to_transfer)} shards."
)
```

### 3.4 Conditionals

Break **before** each boolean operator.

```python
# GOOD
if (
    self.srl_config.ps_mode in ("nixl_cpu", "nixl_gpu")
    and self.nixl_storage_client is not None
    and version > self.current_version
):
    ...

# BAD: breaks after operator
if (self.srl_config.ps_mode in ("nixl_cpu", "nixl_gpu") and
    self.nixl_storage_client is not None and
    version > self.current_version):
    ...
```

### 3.5 Collections & Comprehensions

```python
# Collection: one element per line, trailing comma
supported_modes = [
    "nixl_cpu",
    "nixl_gpu",
    "cpu",
    "cpu_ref",
]

# Comprehension: one-liner if it fits
names = [w.name for w in workers if w.is_active]

# Comprehension: multi-line if long
names = [
    worker.name
    for worker in workers
    if worker.is_active and worker.role == Srl_Role.Actor
]
```

---

## §4. Docstrings

Use **Google-style**. Opening `"""` on its **own line**.

```python
def wait_for_nixl_push_completion(self, timeout: float | None = None) -> bool:
    """
    Wait for the NIXL push wait thread to complete.

    Args:
        timeout (float | None): Maximum wait time in seconds.
            If None, wait indefinitely.

    Returns:
        bool: True if completed successfully, False if timed out.

    Raises:
        RuntimeError: If the push thread encountered an unrecoverable error.
    """
```

- Summary: one sentence, uppercase, period.
- Sections: `Args:`, `Returns:`, `Raises:`, `Usage:`.
- Arg type in parentheses with `|` union: `timeout (float | None):`. Never write `optional`.
- Long arg descriptions: continuation at 8-space indent.
- Reference other identifiers with backticks (§2.4).
- Add `Usage:` with code block when call sequencing is non-obvious.

---

## §5. Comments & Annotation Markers

### 5.0 What a Comment Must Earn ⚠️

> **A comment earns its line by carrying information the code cannot. One line is
> the default, two the ceiling for ordinary code. Three only for a `NOTE` about a
> real design trap.**

Write the intuition: the constraint, the invariant, the unit, the failure mode
that made the code look the way it does. Then stop. If it needs more, it belongs
in the function docstring, or the code needs a better name.

**Never write these.**

| Anti-pattern | Why it is worthless |
|--------------|---------------------|
| **History** (`# Previously we called X`, `# An earlier version did Y`) | Nobody reads code to learn what it used to be. `git log` owns this. Delete the old approach rather than narrating it |
| **Measurements that justified a choice** (`# 65.53 GiB weight, verified to 0.999x`) | Dates instantly, nobody re-measures it. Belongs in the commit message |
| **Restating the code** (`# Increment the counter` above `counter += 1`) | Costs a line, adds nothing, rots on the next edit |
| **Comparison to alternatives** (`# We could have used a heap, but...`) | Design-space tourism. Keep only if a reader would actively try the alternative and break something |
| **Narrative markers** (`**Why this exists.**`, `**The bug this fixes.**`) | A comment is not an essay. State the fact |

```python
# BAD: history, measurements, and self-narration for a one-line fact
# Hybrid attention: only full-attention layers carry a KV cache. The linear
# layers use a separate Mamba state cache, so charging them would overstate
# capacity by the interval factor (4x here, 10 of 40 layers). An earlier
# version charged all 40 and was wrong by 9.4x against the real server.
num_layers = _full_attention_layer_count(config, num_layers)

# GOOD: one line, states what a reader cannot see
# Only full-attention layers carry a KV cache. The linear layers hold a
# per-sequence mamba state instead.
num_layers = _full_attention_layer_count(config, num_layers)
```

**Numbers.** Only when the number *is* the specification (`# vLLM caps this at
1024`). A number that merely motivated a choice goes in the commit message.

**Budget.** A module docstring is ~15 lines. Past that the content is reference
material: put it in `.claude/`, a README, or a memory note, and link to it.

**A missing comment is cheaper than a bloated one.** When nothing non-obvious is
happening, write nothing.

### 5.0.1 Banned characters in all prose ⚠️

In every comment, docstring, log message and assertion message, these are
forbidden anywhere in the line, not only at the end.

| Banned | Instead write |
|--------|---------------|
| `;` semicolon | `.` and a new sentence, or a conjunction |
| `—` em dash, `–` en dash, ` -- ` | `.` and a new sentence, or restructure |
| `. ` double space | one space |

Hyphenated compounds inside a word (`worker-side`, `non-KV`) are fine.

```python
# BAD: semicolon and dash splicing clauses
# Recovery succeeded; the slot is not wasted -- keep the data.

# GOOD: separate sentences
# Recovery succeeded, so the slot is not wasted. Keep the data.
```

Before finishing an edit, grep the diff: `git diff -U0 | grep -nE '^\+.*[;—–]'`
and confirm every hit is real code rather than prose.

### 5.1 Block Comments

Above the code, indented to match. Follow §2.1 (sentence vs. fragment).
Reference identifiers with backticks (§2.4).

```python
# Set the validation rollout number based on the global batch size.
config.actor_rollout_ref.rollout.val_rollout_n = ...

# worker-side attributes
self.worker_rank = worker_rank
```

- Blank line above when starting a new logical section.
- **No dashes inside comment text.** Use plain prose instead (e.g., `# worker-side` → `# worker side`).
- **No semicolons in comments or documentation.** Break into separate sentences instead.
- **No em dashes or en dashes in comments or documentation** (`: `, `–`). Replace with a period and a new sentence, or restructure the phrase.
- **One space after a period**, not two. Never write `. ` + extra space mid-sentence.

### 5.2 Inline Comments

Two spaces before `#`, one space after. Short fragment preferred.

```python
self.ray_worker_group_cls = ray_worker_group_cls  # used only on train side
x = compute_staleness()  # This resets every epoch.
```

- Fragment → no period. Sentence → period.
- Do **not** vertically align across lines.

### 5.3 Section Separators

Sparingly, only in files >150 lines:

```python
# --- Initialization ---

# --- Main Training Loop ---
```

### 5.4 Annotation Markers

Format: `# MARKER(author): Uppercase message ending with period.`

| Marker | Purpose |
|--------|---------|
| `NOTE(xx)` | Design decisions, non-obvious choices |
| `TODO(xx)` | Future improvements, missing features |
| `FIXME(xx)` | Known bugs that must be fixed |
| `HACK(xx)` | Temporary workarounds (must explain why) |

- Always include author initials: `NOTE(lhy)`, `TODO(claude)`.
- Colon immediately after `)`, one space, then the message.
- **Uppercase** first letter, ends with **period**.
- Multi-line: repeat `#` on each line, period on **last line only**.

```python
# GOOD
# NOTE(claude): We use a dict to store the PS handle and merge on the PS side.
# This is more efficient than calling `transfer_train_to_gen` for each key/shard,
# which would cause excessive remote calls and may crash the Ray actor.

# BAD: no period, lowercase start
# NOTE(claude): we use a dict to store the PS handle and merge on the PS side
```

---

## §6. Assertions

Every `assert` **must** include a message. Uppercase first letter, period at the end.
Plain text for identifiers (§2.4). Include actual value with `!r` (§2.5).

```python
# Single-line
assert self.nixl_storage_client is not None, "nixl_storage_client is not initialized."

# Multi-line: parenthesized, 4-space indent
assert self.srl_config.ps_mode in ("nixl_cpu", "nixl_gpu"), (
    "push_model_state_dict_nixl should only be used in nixl_cpu or nixl_gpu mode, "
    f"got: {self.srl_config.ps_mode!r}."
)
```

---

## §7. Logging

### 7.1 Setup

```python
srl_logger = logging.getLogger(__file__)
```

- Always `srl_logger`, never `print()`.
- Prefix with `[component_name]` at key entry points.

### 7.2 Message Format

- Follow §2 for capitalization and punctuation.
- **f-strings only**: never `%`-style or `.format()`.
- `!r` for ambiguous dynamic values (§2.5).
- Plain text for identifiers, no backticks (§2.4).

```python
srl_logger.debug("Getting the current PS model version...")
srl_logger.info("[validate_config] All configuration checks passed!")
srl_logger.info(f"PS mode set to {self.srl_config.ps_mode!r}.")
srl_logger.warning(f"[{log_prefix}]: Failed to log {name}: {e}.")
```

### 7.3 Multi-line

Implicit string concatenation, 4-space indent:

```python
srl_logger.info(
    f"Worker {self.worker_rank} completed NIXL push for version {version} "
    f"with {num_shards} shards to {target_name} in {elapsed:.2f}s."
)
```

### 7.4 Log Levels

| Level | Use for | Example |
|-------|---------|---------|
| `debug` | Internal state, fine-grained tracing | `"Getting the current PS model version..."` |
| `info` | Lifecycle events, milestones | `"Initialized parameter server successfully."` |
| `warning` | Recoverable issues, degraded state | `"Staleness buffer is full, dropping oldest entry."` |
| `error` | Failures that need attention | `"Failed to push model to PSStorageWorker: {e}."` |

---

## §8. Imports & Type Annotations

### 8.1 Imports

**isort** order (enforced by Ruff): stdlib → third-party → local. Blank line between groups.
Use explicit imports; avoid `from module import *`.

```python
import logging
import time

import ray
from omegaconf import DictConfig

from srl.utils.logger import (
    EventType,
    deprecated,
    get_ps_logger,
    get_worker_info,
    log_dual_events,
    log_single_event,
)
```

### 8.2 Type Annotations

- `X | None`, not `Optional[X]`.
- `list[int]`, `dict[str, Any]`, `tuple[int, ...]` (lowercase builtins).
- Complex types → `TypeAlias`:

```python
ShardMapping: TypeAlias = dict[str, list[int]]
```

---

## §9. Language

**English only.** No Chinese in comments, docstrings, or log messages.
