# Rollout Prefix Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the rollout engine a real KV block pool with prefix reuse, so a GRPO group pays for its shared prompt once instead of `group_size` times and a post-weight-sync re-prefill hits the blocks the engine kept.

**Architecture:** A new `engine/kv/` package holds a line-by-line port of the vLLM v1 block pool (blocks with `ref_cnt` plus a hash, a doubly linked LRU free list, a hash-to-block map, eviction on reuse). Block hashes are *derived* from workload provenance rather than read from a trace column, because we generate the workload and therefore know which tokens came from where. `NativeInstance` keeps its structure: occupancy queries route to the pool, and a cache hit is expressed by pre-setting `prefilled`, so the cost model is untouched.

Rollback gets a reusable mechanism rather than a one-off. `advance_one_event` is speculative, so every mutation must reverse. A small `undo` package provides `Undoable` (the restore protocol), `UndoLog` (an ordered collection of captured old values), and `NullRecorder` (the zero-cost no-op for committed paths). Pool state, request state, and instance state all register through the same interface, so a future subsystem needing rollback implements one protocol instead of extending a snapshot dataclass.

**Tech Stack:** Python 3.11+, a hand-written doubly linked free list (ported from vLLM; its `prev`/`next` pointers are what makes a mid-list removal reversible), pytest. No new third-party dependency. Validation uses the existing `dryrun/workload/inference.py` vLLM harness plus a `/metrics` reader.

## Global Constraints

- Read `.claude/coding-style.md` and `.claude/codebase-map.md` before writing code; apply both.
- Activate the environment first: `source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh`.
- No backward compatibility. Remove obsolete paths rather than adding fallbacks, shims, or migrations.
- `block_size` default is 16. Only **full** blocks receive a hash; a partially filled block is never cacheable.
- Prompt-interior blocks key on `source_group_id`, never `group_id`. `group_id` is monotonic and so changes on workload wraparound, which would miss reuse the real engine achieves.
- Fit no coefficients anywhere in this work. Every expected number is arithmetic over `prompt_len`, `block_size`, and the length distribution.
- `_kv_exhaustion_steps()` (`dryrun/component/rollout/engine/instance.py:234-278`) keeps its closed form. Do not replace it with a search.
- **Never snapshot the whole pool.** `copy.deepcopy` of a 31279-block pool measures 179 ms, which over 4949 steps across two instances is 1771 s against a 3349 s run. Capture cost must be proportional to the blocks an event touched (median 2), never to capacity. This is the rule `snapshot.py:78-83` already states.
- Run `pytest tests/ -q` after every task. A task that reddens an existing test is not done. Do not create git commits; the user commits separately.

---

## File Structure

| Path | Responsibility |
|---|---|
| `dryrun/component/rollout/engine/undo/__init__.py` | Package exports |
| `dryrun/component/rollout/engine/undo/protocol.py` | `Undoable` protocol, `NullRecorder` |
| `dryrun/component/rollout/engine/undo/log.py` | `UndoLog`: ordered old values, rollback in reverse |
| `dryrun/component/rollout/engine/undo/entries.py` | `AttrSnapshot`, `TruncateList`, `MappingKeysAdded` |
| `dryrun/component/rollout/engine/kv/__init__.py` | Package exports |
| `dryrun/component/rollout/engine/kv/block.py` | `KvBlock`: id, refcount, hash, free-list links |
| `dryrun/component/rollout/engine/kv/free_list.py` | `FreeBlockList`: doubly linked LRU queue |
| `dryrun/component/rollout/engine/kv/lineage.py` | Derive a request's block hashes from provenance |
| `dryrun/component/rollout/engine/kv/pool.py` | `BlockPool`: allocation, hash map, eviction |
| `dryrun/component/rollout/engine/kv/manager.py` | `KvCacheManager`: the methods `NativeInstance` calls |
| `dryrun/component/rollout/engine/instance.py` | Occupancy routes to the pool; hit pre-sets `prefilled` |
| `dryrun/component/rollout/engine/snapshot.py` | Rebuilt on `UndoLog` |
| `dryrun/component/rollout/engine/state.py` | `StepStat` gains hit/computed token counts |
| `dryrun/component/rollout/engine/request.py` | `Request` gains `prefix_hit_tokens` |
| `dryrun/config/schema.py` | `enable_prefix_caching` knob |
| `dryrun/cli/main.py` | Thread the knob into `SimConfig` |
| `dryrun/simulator/config.py` | `SimConfig.enable_prefix_caching` |
| `dryrun/telemetry/writer.py` | Emit the new `StepStat` fields |
| `dryrun/workload/prefix_probe.py` | Read vLLM's prefix-cache counters from `/metrics` |
| `tests/test_undo.py` | Task 1 |
| `tests/test_kv_pool.py` | Tasks 2-5 |
| `tests/test_kv_instance.py` | Tasks 6-8 |
| `tests/test_kv_config.py` | Task 9 |
| `tests/test_prefix_probe.py` | Task 10 |
| `examples/profile/vllm/prefix_cache_probe.yaml` | Task 11 fixture |
| `docs/simulation/prefix-cache.md` | Task 12 |

**Dependency order:** 1 → 2 → ... → 12, strictly linear. Task 11 needs a GPU; the rest do not.

---

### Task 1: The `undo` package

A reusable rollback mechanism, built before anything needs it, so the pool and the existing snapshot both register through one interface. Three pieces: a protocol saying what "restorable" means, a log that collects captured old values, and a null recorder for paths that commit and therefore never roll back.

The design rule comes from the existing snapshot's own comment (`snapshot.py:78-83`): capture cost is proportional to the live batch, never to the accumulated run. `UndoLog` enforces that structurally — you register the specific things you touched, so there is no API through which to capture a whole subsystem.

**Files:**
- Create: `dryrun/component/rollout/engine/undo/__init__.py`
- Create: `dryrun/component/rollout/engine/undo/protocol.py`
- Create: `dryrun/component/rollout/engine/undo/log.py`
- Create: `dryrun/component/rollout/engine/undo/entries.py`
- Test: `tests/test_undo.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Undoable` — runtime-checkable protocol with `restore() -> None`.
  - `Recorder` — protocol with `record(entry: Undoable) -> None` and `record_once(key: Hashable, capture: Callable[[], Undoable]) -> None`.
  - `NullRecorder()` — a `Recorder` that discards everything. Module-level singleton `NULL_RECORDER`.
  - `UndoLog()` — a `Recorder` with `rollback() -> None` (restores in reverse registration order), `clear() -> None`, `__len__`.
  - `AttrSnapshot(target: object, names: tuple[str, ...])` — classmethod `capture(target, *names) -> AttrSnapshot`, restores those attributes.
  - `TruncateList(target: list, length: int)` — classmethod `capture(target) -> TruncateList`, restores by truncating to the captured length.
  - `MappingKeysAdded(target: dict, keys: list)` — `note(key)` records a key added after capture; `restore` deletes exactly those.

- [ ] **Step 1: Write the failing test**

Create `tests/test_undo.py`:

```python
"""Tests for the reusable rollback mechanism.

`advance_one_event` is speculative: the driver advances every instance, commits
the earliest event, and rolls the rest back. So every mutation an event performs
must reverse, and this package is the one way to arrange that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from dryrun.component.rollout.engine.undo import (
    NULL_RECORDER,
    AttrSnapshot,
    MappingKeysAdded,
    TruncateList,
    Undoable,
    UndoLog,
)


@dataclass
class Thing:
    count: int = 0
    label: str = "a"
    items: list[int] = field(default_factory=list)


class TestAttrSnapshot:
    def test_restores_a_single_attribute(self):
        thing = Thing(count=5)
        entry = AttrSnapshot.capture(thing, "count")
        thing.count = 99
        entry.restore()
        assert thing.count == 5

    def test_restores_several_attributes_at_once(self):
        thing = Thing(count=5, label="a")
        entry = AttrSnapshot.capture(thing, "count", "label")
        thing.count, thing.label = 99, "z"
        entry.restore()
        assert (thing.count, thing.label) == (5, "a")

    def test_capturing_an_absent_attribute_raises_at_capture_time(self):
        """A typo must fail where it was written, not silently at rollback."""
        with pytest.raises(AttributeError, match="nope"):
            AttrSnapshot.capture(Thing(), "nope")

    def test_satisfies_the_undoable_protocol(self):
        assert isinstance(AttrSnapshot.capture(Thing(), "count"), Undoable)


class TestTruncateList:
    def test_restores_an_append_only_list_by_length(self):
        """The trick the existing snapshot already uses: an append-only list
        needs its length recorded, not its contents copied."""
        items = [1, 2, 3]
        entry = TruncateList.capture(items)
        items.extend([4, 5])
        entry.restore()
        assert items == [1, 2, 3]

    def test_an_untouched_list_is_unchanged(self):
        items = [1, 2]
        entry = TruncateList.capture(items)
        entry.restore()
        assert items == [1, 2]


class TestMappingKeysAdded:
    def test_removes_exactly_the_noted_keys(self):
        mapping = {"a": 1}
        entry = MappingKeysAdded.capture(mapping)
        mapping["b"] = 2
        entry.note("b")
        mapping["c"] = 3
        entry.note("c")
        entry.restore()
        assert mapping == {"a": 1}

    def test_leaves_keys_it_was_not_told_about(self):
        """It removes additions, it does not diff the mapping. A key added
        without a `note` is a caller bug, and silently removing it would hide
        that bug rather than surface it."""
        mapping = {"a": 1}
        entry = MappingKeysAdded.capture(mapping)
        mapping["b"] = 2
        entry.restore()
        assert mapping == {"a": 1, "b": 2}

    def test_a_noted_key_already_gone_is_not_an_error(self):
        """Eviction can remove a key the same event added."""
        mapping = {}
        entry = MappingKeysAdded.capture(mapping)
        entry.note("b")
        entry.restore()
        assert mapping == {}


class TestUndoLog:
    def test_rolls_back_in_reverse_registration_order(self):
        """Order matters: a later entry may depend on state an earlier one
        captured, so undo has to unwind, not replay."""
        order: list[str] = []

        @dataclass
        class Marker:
            name: str

            def restore(self) -> None:
                order.append(self.name)

        log = UndoLog()
        log.record(Marker("first"))
        log.record(Marker("second"))
        log.record(Marker("third"))
        log.rollback()
        assert order == ["third", "second", "first"]

    def test_record_once_captures_only_the_first_time_per_key(self):
        """A block touched twice in one event must restore to its value at the
        start of the event, not to the value after the first mutation."""
        thing = Thing(count=1)
        log = UndoLog()
        log.record_once("thing", lambda: AttrSnapshot.capture(thing, "count"))
        thing.count = 2
        log.record_once("thing", lambda: AttrSnapshot.capture(thing, "count"))
        thing.count = 3
        log.rollback()
        assert thing.count == 1

    def test_record_once_does_not_build_the_entry_when_already_seen(self):
        """The capture callable is the expensive part, so a repeat key must not
        invoke it at all."""
        calls = 0

        def capture():
            nonlocal calls
            calls += 1
            return AttrSnapshot.capture(Thing(), "count")

        log = UndoLog()
        log.record_once("k", capture)
        log.record_once("k", capture)
        assert calls == 1

    def test_distinct_keys_are_captured_separately(self):
        a, b = Thing(count=1), Thing(count=10)
        log = UndoLog()
        log.record_once("a", lambda: AttrSnapshot.capture(a, "count"))
        log.record_once("b", lambda: AttrSnapshot.capture(b, "count"))
        a.count, b.count = 7, 70
        log.rollback()
        assert (a.count, b.count) == (1, 10)

    def test_rollback_empties_the_log(self):
        thing = Thing(count=1)
        log = UndoLog()
        log.record(AttrSnapshot.capture(thing, "count"))
        log.rollback()
        assert len(log) == 0

    def test_rolling_back_twice_is_not_a_silent_no_op(self):
        """A double rollback means the caller lost track of event boundaries.
        Restoring stale values would corrupt state quietly."""
        log = UndoLog()
        log.record(AttrSnapshot.capture(Thing(), "count"))
        log.rollback()
        with pytest.raises(RuntimeError, match="empty"):
            log.rollback()

    def test_clear_discards_without_restoring(self):
        """Committing an event discards its log; the mutations stand."""
        thing = Thing(count=1)
        log = UndoLog()
        log.record(AttrSnapshot.capture(thing, "count"))
        thing.count = 2
        log.clear()
        assert thing.count == 2
        assert len(log) == 0

    def test_length_counts_registered_entries(self):
        log = UndoLog()
        assert len(log) == 0
        log.record(AttrSnapshot.capture(Thing(), "count"))
        assert len(log) == 1


class TestNullRecorder:
    def test_discards_entries(self):
        thing = Thing(count=1)
        NULL_RECORDER.record(AttrSnapshot.capture(thing, "count"))
        thing.count = 2
        assert thing.count == 2, "nothing was retained, so nothing can restore"

    def test_record_once_never_invokes_the_capture_callable(self):
        """Committed paths (`advance_to`) pay nothing. If the callable ran, the
        no-op would still cost an object allocation per touched block."""
        calls = 0

        def capture():
            nonlocal calls
            calls += 1
            return AttrSnapshot.capture(Thing(), "count")

        NULL_RECORDER.record_once("k", capture)
        assert calls == 0

    def test_is_interchangeable_with_a_real_log(self):
        """Call sites take a Recorder and never branch on which one they got."""
        for recorder in (UndoLog(), NULL_RECORDER):
            recorder.record_once("k", lambda: AttrSnapshot.capture(Thing(), "count"))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/DryRun-RL
pytest tests/test_undo.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'dryrun.component.rollout.engine.undo'`.

- [ ] **Step 3: Write minimal implementation**

Create `dryrun/component/rollout/engine/undo/protocol.py`:

```python
"""What it means for a mutation to be reversible.

`advance_one_event` is speculative: the driver advances every instance by one
event, commits the earliest, and discards the rest (`simulator.py:388-396` and
`:356-360`). That is how instances share one global clock, so reversibility is a
structural requirement of the engine loop rather than a safety net.
"""

from __future__ import annotations

from typing import Callable, Hashable, Protocol, runtime_checkable


@runtime_checkable
class Undoable(Protocol):
    """A captured old value that knows how to put itself back."""

    def restore(self) -> None:
        """Return the captured target to the state it held at capture time."""
        ...


class Recorder(Protocol):
    """Where a mutating method registers what it is about to overwrite.

    Methods take a `Recorder` rather than an optional log, so a committed path
    and a speculative one run the same code. `NULL_RECORDER` is the committed
    case.
    """

    def record(self, entry: Undoable) -> None:
        """Register an already-captured old value."""
        ...

    def record_once(self, key: Hashable, capture: Callable[[], Undoable]) -> None:
        """Register an old value the first time `key` is seen, and only then.

        `capture` is a callable rather than a value so that a repeat key costs
        nothing: a decode segment can touch the same block many times, and the
        first capture is the only correct one anyway.
        """
        ...


class NullRecorder:
    """A `Recorder` that keeps nothing, for paths that cannot be rolled back.

    `advance_to` commits as it goes, so recording there would allocate an entry
    per touched block and then drop it. `record_once` deliberately does not
    invoke its callable.
    """

    __slots__ = ()

    def record(self, entry: Undoable) -> None:
        pass

    def record_once(self, key: Hashable, capture: Callable[[], Undoable]) -> None:
        pass


NULL_RECORDER = NullRecorder()
"""Shared no-op recorder. Stateless, so one instance serves every caller."""
```

Create `dryrun/component/rollout/engine/undo/entries.py`:

```python
"""The three shapes of captured old value this engine actually needs.

Each mirrors a technique the original `InstanceSnapshot` arrived at by hand:
scalar fields stored and assigned back, append-only lists restored by length
rather than by copy, and additions to a mapping removed by key. Naming them
makes the techniques reusable instead of re-derived per subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, MutableMapping


@dataclass(slots=True)
class AttrSnapshot:
    """Old values of named attributes on one object.

    Args:
        target: The object whose attributes were captured.
        names: Attribute names, in capture order.
        values: The captured values, parallel to `names`.
    """

    target: Any
    names: tuple[str, ...]
    values: tuple[Any, ...]

    @classmethod
    def capture(cls, target: Any, *names: str) -> "AttrSnapshot":
        """Capture `names` from `target`.

        Raises:
            AttributeError: If any name is absent, so a typo fails where it was
                written rather than silently at rollback.
        """
        return cls(
            target=target,
            names=names,
            values=tuple(getattr(target, name) for name in names),
        )

    def restore(self) -> None:
        for name, value in zip(self.names, self.values):
            setattr(self.target, name, value)


@dataclass(slots=True)
class TruncateList:
    """An append-only list, restored by cutting it back to its old length.

    This is the technique `InstanceSnapshot` already applies to `step_stats` and
    `completed`: given the append-only invariant, a length is enough, so capture
    is O(1) rather than O(len).
    """

    target: list
    length: int

    @classmethod
    def capture(cls, target: list) -> "TruncateList":
        return cls(target=target, length=len(target))

    def restore(self) -> None:
        del self.target[self.length :]


@dataclass(slots=True)
class MappingKeysAdded:
    """Keys added to a mapping during one event, removed on rollback.

    Only noted keys are removed. This does not diff the mapping, because a key
    added without a `note` is a caller bug and quietly removing it would hide
    the bug instead of surfacing it.
    """

    target: MutableMapping
    keys: list[Hashable] = field(default_factory=list)

    @classmethod
    def capture(cls, target: MutableMapping) -> "MappingKeysAdded":
        return cls(target=target)

    def note(self, key: Hashable) -> None:
        """Record that `key` was just added."""
        self.keys.append(key)

    def restore(self) -> None:
        for key in self.keys:
            # Eviction can remove a key the same event added, so absence is
            # expected rather than exceptional.
            self.target.pop(key, None)
```

Create `dryrun/component/rollout/engine/undo/log.py`:

```python
"""An ordered collection of captured old values for one speculative event."""

from __future__ import annotations

from typing import Callable, Hashable

from .protocol import Undoable


class UndoLog:
    """Captured old values for one event, restorable in reverse order.

    Reverse order matters: a later entry may have been captured after an earlier
    one already changed something, so rollback has to unwind rather than replay.

    There is deliberately no "capture this whole object" entry point. Callers
    register the specific things they touch, which is what keeps cost
    proportional to the work an event did rather than to the size of the
    subsystem it did the work in. A whole-pool `copy.deepcopy` measured 179 ms
    per event, or 1771 s across a run whose real duration is 3349 s.
    """

    __slots__ = ("_entries", "_keys")

    def __init__(self) -> None:
        self._entries: list[Undoable] = []
        self._keys: set[Hashable] = set()

    def __len__(self) -> int:
        return len(self._entries)

    def record(self, entry: Undoable) -> None:
        """Register an already-captured old value."""
        self._entries.append(entry)

    def record_once(self, key: Hashable, capture: Callable[[], Undoable]) -> None:
        """Capture under `key` the first time it is seen, and only then.

        The first capture is the only correct one: an event that touches one
        block repeatedly must restore the value from before the event, not from
        between two of its own mutations. Passing a callable also means a repeat
        key costs nothing to build.
        """
        if key in self._keys:
            return
        self._keys.add(key)
        self._entries.append(capture())

    def rollback(self) -> None:
        """Restore every captured value, newest first, and empty the log.

        Raises:
            RuntimeError: If the log is empty. A second rollback means the
                caller lost track of event boundaries, and restoring stale
                values would corrupt state silently.
        """
        if not self._entries:
            raise RuntimeError(
                "Cannot roll back an empty undo log: either no event was "
                "recorded, or this event was already rolled back."
            )
        for entry in reversed(self._entries):
            entry.restore()
        self.clear()

    def clear(self) -> None:
        """Discard the log without restoring. This is what committing means."""
        self._entries.clear()
        self._keys.clear()
```

Create `dryrun/component/rollout/engine/undo/__init__.py`:

```python
"""Reusable rollback for speculative event advancement.

The engine advances each instance by one event, commits the earliest, and
discards the rest, so every mutation must reverse. This package is the one
mechanism for arranging that: a mutating method takes a `Recorder` and registers
what it is about to overwrite, and the caller decides whether that recorder is a
real `UndoLog` or `NULL_RECORDER`.

Adding rollback to a new subsystem therefore means implementing `Undoable` for
whatever it owns, not extending a snapshot dataclass that knows about every
subsystem at once.
"""

from .entries import AttrSnapshot, MappingKeysAdded, TruncateList
from .log import UndoLog
from .protocol import NULL_RECORDER, NullRecorder, Recorder, Undoable

__all__ = [
    "NULL_RECORDER",
    "AttrSnapshot",
    "MappingKeysAdded",
    "NullRecorder",
    "Recorder",
    "TruncateList",
    "UndoLog",
    "Undoable",
]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_undo.py -q && pytest tests/ -q
```

Expected: `tests/test_undo.py` 20 passed, whole suite green (nothing else imports this yet).

---

### Task 2: `KvBlock` and the undoable free list

A block is a slot in the pool, carrying a refcount, optionally a hash, and its position in the free list. The list and the block are co-designed: the `prev`/`next` pointers live on the block precisely so a mid-list removal can be reversed by re-linking, which is why they are one task.

**Files:**
- Create: `dryrun/component/rollout/engine/kv/__init__.py`
- Create: `dryrun/component/rollout/engine/kv/block.py`
- Create: `dryrun/component/rollout/engine/kv/free_list.py`
- Test: `tests/test_kv_pool.py`

**Interfaces:**
- Consumes: `Recorder`, `NULL_RECORDER`, `AttrSnapshot` from Task 1.
- Produces:
  - `KvBlock(block_id: int)` — attributes `block_id: int`, `ref_cnt: int` (starts 0), `block_hash: Hashable | None` (starts `None`), `prev_free: KvBlock | None`, `next_free: KvBlock | None`; methods `incr_ref()`, `decr_ref()`, `reset_hash()`. Assigning `block_hash` when already set raises `ValueError`.
  - `FreeBlockList(blocks: list[KvBlock])` — property `num_free: int`; methods `popleft(recorder) -> KvBlock`, `append(block, recorder) -> None`, `remove(block, recorder) -> None`, `ids() -> list[int]` (head-to-tail order, for tests and diagnostics).

- [ ] **Step 1: Write the failing test**

Create `tests/test_kv_pool.py`:

```python
"""Tests for the rollout KV block pool and provenance-derived block hashes."""

from __future__ import annotations

import pytest

from dryrun.component.rollout.engine.kv import FreeBlockList, KvBlock
from dryrun.component.rollout.engine.undo import NULL_RECORDER, UndoLog


class TestKvBlock:
    def test_starts_unheld_and_unhashed(self):
        block = KvBlock(block_id=7)
        assert block.block_id == 7
        assert block.ref_cnt == 0
        assert block.block_hash is None

    def test_refcount_tracks_holders(self):
        block = KvBlock(block_id=0)
        block.incr_ref()
        block.incr_ref()
        assert block.ref_cnt == 2
        block.decr_ref()
        assert block.ref_cnt == 1

    def test_reassigning_a_hash_is_a_bug_not_an_update(self):
        """One physical block holding two logical identities is the hardest bug
        class here, so assigning twice raises instead of silently overwriting."""
        block = KvBlock(block_id=0)
        block.block_hash = ("p", 42, 0)
        with pytest.raises(ValueError, match="already"):
            block.block_hash = ("p", 42, 1)

    def test_reset_hash_allows_reassignment(self):
        block = KvBlock(block_id=0)
        block.block_hash = ("p", 42, 0)
        block.reset_hash()
        assert block.block_hash is None
        block.block_hash = ("r", 9, 3)
        assert block.block_hash == ("r", 9, 3)


class TestFreeBlockList:
    def test_starts_with_every_block_in_order(self):
        blocks = [KvBlock(i) for i in range(4)]
        free = FreeBlockList(blocks)
        assert free.num_free == 4
        assert free.ids() == [0, 1, 2, 3]

    def test_popleft_takes_the_least_recently_freed(self):
        free = FreeBlockList([KvBlock(i) for i in range(3)])
        assert free.popleft(NULL_RECORDER).block_id == 0
        assert free.ids() == [1, 2]
        assert free.num_free == 2

    def test_append_puts_a_block_at_the_tail(self):
        blocks = [KvBlock(i) for i in range(3)]
        free = FreeBlockList(blocks)
        taken = free.popleft(NULL_RECORDER)
        free.append(taken, NULL_RECORDER)
        assert free.ids() == [1, 2, 0], "released blocks go to the back of the LRU queue"

    def test_remove_takes_a_block_out_of_the_middle(self):
        blocks = [KvBlock(i) for i in range(4)]
        free = FreeBlockList(blocks)
        free.remove(blocks[2], NULL_RECORDER)
        assert free.ids() == [0, 1, 3]
        assert free.num_free == 3

    def test_removing_a_block_that_is_not_in_the_list_raises(self):
        """A block outside the list has no neighbours to unlink, so proceeding
        would corrupt the list silently."""
        blocks = [KvBlock(i) for i in range(2)]
        free = FreeBlockList(blocks)
        free.remove(blocks[0], NULL_RECORDER)
        with pytest.raises(RuntimeError, match="not in the free list"):
            free.remove(blocks[0], NULL_RECORDER)

    def test_popleft_on_an_empty_list_raises(self):
        free = FreeBlockList([])
        with pytest.raises(RuntimeError, match="empty"):
            free.popleft(NULL_RECORDER)


class TestFreeListUndo:
    def test_undoing_popleft_restores_the_block_to_the_head(self):
        """Not the tail. Putting it back in the wrong place leaves `num_free`
        correct while silently rewriting LRU order, which `blocks_used()` cannot
        see and which decides every later eviction."""
        blocks = [KvBlock(i) for i in range(4)]
        free = FreeBlockList(blocks)
        log = UndoLog()
        free.popleft(log)
        log.rollback()
        assert free.ids() == [0, 1, 2, 3]
        assert free.num_free == 4

    def test_undoing_remove_restores_the_exact_position(self):
        blocks = [KvBlock(i) for i in range(5)]
        free = FreeBlockList(blocks)
        log = UndoLog()
        free.remove(blocks[2], log)
        log.rollback()
        assert free.ids() == [0, 1, 2, 3, 4]

    def test_undoing_append_removes_the_block_again(self):
        blocks = [KvBlock(i) for i in range(3)]
        free = FreeBlockList(blocks)
        taken = free.popleft(NULL_RECORDER)
        log = UndoLog()
        free.append(taken, log)
        log.rollback()
        assert free.ids() == [1, 2]
        assert free.num_free == 2

    def test_a_mixed_event_restores_the_whole_order(self):
        """One event can pop, remove and append. Rollback unwinds in reverse,
        so the list must come back byte for byte."""
        blocks = [KvBlock(i) for i in range(6)]
        free = FreeBlockList(blocks)
        before = free.ids()
        log = UndoLog()
        free.popleft(log)
        free.remove(blocks[3], log)
        free.append(blocks[0], log)
        assert free.ids() != before
        log.rollback()
        assert free.ids() == before

    def test_null_recorder_leaves_no_way_back(self):
        """Committed paths pass NULL_RECORDER, so the mutation stands."""
        blocks = [KvBlock(i) for i in range(3)]
        free = FreeBlockList(blocks)
        free.popleft(NULL_RECORDER)
        assert free.ids() == [1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/DryRun-RL
pytest tests/test_kv_pool.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'dryrun.component.rollout.engine.kv'`.

- [ ] **Step 3: Write minimal implementation**

Create `dryrun/component/rollout/engine/kv/block.py`:

```python
"""One slot in the KV block pool.

A block is identified by `block_id` and tracks three things: how many live
requests hold it (`ref_cnt`), which content it holds (`block_hash`), and where
it sits in the free list (`prev_free` / `next_free`).

The first two are deliberately independent. A block whose `ref_cnt` has fallen
to zero keeps its hash, so a later request with the same prefix can still find
and reuse it. That state -- free but still addressable -- is what makes reuse
across a weight update or an agentic round boundary possible at all.

The link fields live here rather than inside the list because they are what
makes a removal reversible: re-linking a block needs the neighbours it had, and
storing them on the node means the "old value" an undo wants is already
materialised. This is also why vLLM keeps them, and why swapping the list for an
`OrderedDict` would not work: `pop(key)` discards position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional


@dataclass
class KvBlock:
    """A pool slot, with a refcount, an optional content hash, and list links.

    Args:
        block_id: Index of this block in the pool. The sentinel head and tail of
            a `FreeBlockList` use -1, and are never handed out.
    """

    block_id: int
    ref_cnt: int = 0
    _block_hash: Optional[Hashable] = None
    prev_free: Optional["KvBlock"] = None
    next_free: Optional["KvBlock"] = None

    def incr_ref(self) -> None:
        self.ref_cnt += 1

    def decr_ref(self) -> None:
        self.ref_cnt -= 1

    @property
    def block_hash(self) -> Optional[Hashable]:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, value: Hashable) -> None:
        """Assign this block's content hash.

        Raises:
            ValueError: If a hash is already assigned. Overwriting would mean
                one physical block standing for two different prefixes, which
                corrupts every later hit silently rather than loudly.
        """
        if self._block_hash is not None:
            raise ValueError(
                f"Block {self.block_id} already holds hash {self._block_hash!r}; "
                f"call reset_hash() before assigning {value!r}."
            )
        self._block_hash = value

    def reset_hash(self) -> None:
        """Drop this block's content hash, making it assignable again."""
        self._block_hash = None
```

Create `dryrun/component/rollout/engine/kv/free_list.py`:

```python
"""The LRU queue of unheld blocks, as a reversible doubly linked list.

Ported from vLLM's `FreeKVCacheBlockQueue`. Head is least recently freed, so
head is the eviction victim; releases append at the tail.

Why a linked list rather than an `OrderedDict`: a prefix-cache hit on a block
that is currently free removes it from the *middle* of the queue, and undoing
that means putting it back exactly where it was. The node's own `prev_free` /
`next_free` are that position, so the information an undo needs already exists.
`OrderedDict.pop(key)` discards it, and re-inserting at the tail would leave
`num_free` correct while silently rewriting LRU order -- invisible to
`blocks_used()` and decisive for every later eviction.

Sentinel head and tail nodes mean a real block's links are never `None` while it
is in the list, which removes the branching vLLM's own comment calls out.
"""

from __future__ import annotations

from .block import KvBlock
from ..undo import AttrSnapshot, Recorder

_SENTINEL_ID = -1


class FreeBlockList:
    """Blocks with `ref_cnt == 0`, ordered least-recently-freed first.

    Args:
        blocks: Every block in the pool, in id order. All start free.
    """

    __slots__ = ("_head", "_tail", "_num_free")

    def __init__(self, blocks: list[KvBlock]) -> None:
        self._head = KvBlock(block_id=_SENTINEL_ID)
        self._tail = KvBlock(block_id=_SENTINEL_ID)
        self._num_free = len(blocks)

        previous = self._head
        for block in blocks:
            previous.next_free = block
            block.prev_free = previous
            previous = block
        previous.next_free = self._tail
        self._tail.prev_free = previous

    @property
    def num_free(self) -> int:
        return self._num_free

    def ids(self) -> list[int]:
        """Block ids from head to tail. For tests and diagnostics.

        Walks the list, so this is O(num_free) and must not be called from the
        engine's hot path. An undo test comparing LRU order is exactly the place
        it belongs: occupancy counts cannot detect a reordering.
        """
        out: list[int] = []
        node = self._head.next_free
        while node is not None and node is not self._tail:
            out.append(node.block_id)
            node = node.next_free
        return out

    def popleft(self, recorder: Recorder) -> KvBlock:
        """Take the least recently freed block.

        Args:
            recorder: Where to register the old links, so the block can be put
                back at the head rather than the tail.

        Raises:
            RuntimeError: If the list is empty. The caller is expected to have
                checked `num_free`; handing out a sentinel would corrupt the
                pool.
        """
        block = self._head.next_free
        if block is None or block is self._tail:
            raise RuntimeError("Cannot pop from an empty free list.")
        self._unlink(block, recorder)
        return block

    def append(self, block: KvBlock, recorder: Recorder) -> None:
        """Put a block at the tail, as the most recently freed."""
        last = self._tail.prev_free
        assert last is not None, "Sentinel tail always has a predecessor."
        self._record_links(block, recorder)
        self._record_links(last, recorder)
        self._record_links(self._tail, recorder)
        last.next_free = block
        block.prev_free = last
        block.next_free = self._tail
        self._tail.prev_free = block
        self._num_free += 1
        recorder.record(AttrSnapshot(self, ("_num_free",), (self._num_free - 1,)))

    def remove(self, block: KvBlock, recorder: Recorder) -> None:
        """Take a block out of the list, wherever it sits.

        Raises:
            RuntimeError: If the block is not currently in the list. It would
                have no neighbours to unlink, and proceeding would corrupt the
                list silently.
        """
        if block.prev_free is None or block.next_free is None:
            raise RuntimeError(
                f"Block {block.block_id} is not in the free list, so it cannot "
                f"be removed from it."
            )
        self._unlink(block, recorder)

    def _unlink(self, block: KvBlock, recorder: Recorder) -> None:
        """Detach `block`, recording enough to re-link it in place."""
        previous, following = block.prev_free, block.next_free
        assert previous is not None and following is not None
        self._record_links(block, recorder)
        self._record_links(previous, recorder)
        self._record_links(following, recorder)
        previous.next_free = following
        following.prev_free = previous
        block.prev_free = block.next_free = None
        self._num_free -= 1
        recorder.record(AttrSnapshot(self, ("_num_free",), (self._num_free + 1,)))

    @staticmethod
    def _record_links(block: KvBlock, recorder: Recorder) -> None:
        """Capture one node's links, once per event.

        Keyed on `id(block)` rather than `block_id` because the two sentinels
        share id -1, and they do get relinked.
        """
        recorder.record_once(
            ("links", id(block)),
            lambda: AttrSnapshot.capture(block, "prev_free", "next_free"),
        )
```

Create `dryrun/component/rollout/engine/kv/__init__.py`:

```python
"""KV block pool: occupancy, prefix reuse, and eviction.

This is a state machine, not a cost model. It fits no coefficient, reads no
artifact, and has no domain gate. `dryrun/cost/rollout/kv_cache.py` answers
*how many blocks exist*; this package answers *who holds them and what can
hit*. The only interface between the two is the `kv_blocks` integer.

Every mutating method takes a `Recorder` (see `..undo`), because
`advance_one_event` is speculative and the driver rolls back the events it does
not commit.
"""

from .block import KvBlock
from .free_list import FreeBlockList

__all__ = ["FreeBlockList", "KvBlock"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_kv_pool.py -q
```

Expected: 15 passed.

---

### Task 3: Provenance-derived block hashes

This is the design's pivot. vLLM hashes a block as `hash(parent_hash, block_tokens)`, so a block hash means *every token from position 0 to this block's end*. We hold no tokens, but we hold the provenance that produced them, which answers the same question: two contexts agree on a prefix exactly when they agree on its provenance.

**Files:**
- Create: `dryrun/component/rollout/engine/kv/lineage.py`
- Modify: `dryrun/component/rollout/engine/kv/__init__.py`
- Test: `tests/test_kv_pool.py`

**Interfaces:**
- Consumes: `Request` from `dryrun.component.rollout.engine.request`, using fields `prompt_len: int`, `generated: int`, `rid: int`, `source_group_id: int`.
- Produces: `block_hashes(req: Request, block_size: int) -> list[Hashable]`, one entry per full block of `req.prompt_len + req.generated`, in order. Prompt-interior entries are `("p", source_group_id, j)`; all others are `("r", rid, j)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kv_pool.py`:

```python
from dryrun.component.rollout.engine import Request
from dryrun.component.rollout.engine.kv import block_hashes


def _req(rid: int, prompt_len: int, source_group_id: int, generated: int = 0) -> Request:
    req = Request(
        rid=rid,
        prompt_len=prompt_len,
        target_len=4096,
        v_traj=0,
        source_group_id=source_group_id,
    )
    req.generated = generated
    return req


class TestBlockHashes:
    def test_only_full_blocks_get_a_hash(self):
        """vLLM's own rule: a partially filled block is not cacheable. At
        prompt_len=150 with block_size=16 that is 9 blocks over 144 tokens."""
        ids = block_hashes(_req(rid=1, prompt_len=150, source_group_id=42), block_size=16)
        assert len(ids) == 9

    def test_prompt_interior_blocks_key_on_the_shared_prompt(self):
        ids = block_hashes(_req(rid=1, prompt_len=150, source_group_id=42), block_size=16)
        assert ids[0] == ("p", 42, 0)
        assert ids[8] == ("p", 42, 8)

    def test_siblings_share_every_prompt_block(self):
        """Eight responses to one prompt are what GRPO samples, and the whole
        point of this module is that they pay for the prompt once."""
        a = block_hashes(_req(rid=1, prompt_len=150, source_group_id=42), block_size=16)
        b = block_hashes(_req(rid=2, prompt_len=150, source_group_id=42), block_size=16)
        assert a == b

    def test_different_prompts_share_nothing(self):
        a = block_hashes(_req(rid=1, prompt_len=150, source_group_id=42), block_size=16)
        b = block_hashes(_req(rid=2, prompt_len=150, source_group_id=43), block_size=16)
        assert set(a).isdisjoint(b)

    def test_the_block_straddling_the_prompt_end_is_private(self):
        """prompt_len=150 leaves block 9 covering tokens 144-159, of which only
        144-149 are prompt. Its content differs per response, so it cannot be
        shared even though most of its tokens are."""
        ids = block_hashes(
            _req(rid=7, prompt_len=150, source_group_id=42, generated=20), block_size=16
        )
        assert ids[8] == ("p", 42, 8)
        assert ids[9] == ("r", 7, 9)

    def test_generated_blocks_are_private_to_their_request(self):
        a = block_hashes(
            _req(rid=1, prompt_len=160, source_group_id=42, generated=64), block_size=16
        )
        b = block_hashes(
            _req(rid=2, prompt_len=160, source_group_id=42, generated=64), block_size=16
        )
        assert a[:10] == b[:10]
        assert set(a[10:]).isdisjoint(b[10:])

    def test_a_prompt_that_is_a_whole_number_of_blocks_shares_all_of_them(self):
        ids = block_hashes(_req(rid=1, prompt_len=160, source_group_id=42), block_size=16)
        assert len(ids) == 10
        assert all(h[0] == "p" for h in ids)

    def test_continued_context_reproduces_its_own_earlier_hashes(self):
        """A request resumed after a weight update re-prefills prompt+generated.
        Its generated blocks must hash exactly as they did when first filled,
        or resumption would never hit."""
        first = block_hashes(
            _req(rid=5, prompt_len=160, source_group_id=42, generated=64), block_size=16
        )
        later = block_hashes(
            _req(rid=5, prompt_len=160, source_group_id=42, generated=128), block_size=16
        )
        assert later[: len(first)] == first

    def test_a_context_shorter_than_one_block_has_no_hashes(self):
        assert block_hashes(_req(rid=1, prompt_len=10, source_group_id=1), block_size=16) == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_kv_pool.py -q
```

Expected: FAIL, `ImportError: cannot import name 'block_hashes'`.

- [ ] **Step 3: Write minimal implementation**

Create `dryrun/component/rollout/engine/kv/lineage.py`:

```python
"""Block hashes derived from workload provenance.

vLLM hashes a block as `hash(parent_block_hash, block_token_ids)`
(`third_party/vllm/vllm/v1/core/kv_cache_utils.py:617-623`). Chaining the
parent in is what makes a block hash mean *every token from position 0 to this
block's end*, so that

    A: [hello] [world]
    B: [bye]   [world]

do not collide on block 1: reuse is only valid when the whole prefix agrees.

We hold no token ids. We hold something that answers the same question: two
contexts agree on a prefix exactly when they agree on the provenance that
produced it. So the hash is written rather than computed, from two integers
already on `Request`.

This is where we diverge from Frontier, which must be handed a
`block_hash_ids` column on its trace because it replays real requests and
never sees their tokens. We generate the workload, so the provenance is known
and the trace format stays untouched.

**Known one-directional bias.** Two *different* prompts that happen to share
leading full blocks hit in the real engine and miss here. On RL workloads
distinct problems almost certainly differ at token 0, and every weight update
clears the hash map anyway, so this is second order -- but it only ever
under-reports hits, never over-reports. See `docs/reference/limitations.md`.
"""

from __future__ import annotations

from typing import Hashable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..request import Request


def block_hashes(req: "Request", block_size: int) -> list[Hashable]:
    """Hash ids for every full block of this request's current context.

    Args:
        req: The request. Its context is `prompt_len + generated`, which is the
            token count the engine knows about so far.
        block_size: Tokens per KV block.

    Returns:
        One id per full block, in block order. A block lying wholly inside the
        shared prompt keys on `source_group_id`, so siblings of one prompt
        agree on it. Every other block keys on `rid` and is private.

        `source_group_id` rather than `group_id` is deliberate: `group_id` is
        monotonic by design, so a repeating workload gives the same prompt a
        new id on wraparound and reuse the real engine achieves would be
        missed. `source_group_id` records which workload group the lengths came
        from, which is the prompt's actual identity.

    Raises:
        ValueError: If `block_size` is not positive.
    """
    if block_size <= 0:
        raise ValueError(f"Expected block_size > 0, got {block_size!r}.")

    context = req.prompt_len + req.generated
    # Only full blocks are cacheable, which is vLLM's own rule. The block
    # straddling the end of the prompt is therefore private even though most of
    # its tokens are shared.
    return [
        ("p", req.source_group_id, j)
        if (j + 1) * block_size <= req.prompt_len
        else ("r", req.rid, j)
        for j in range(context // block_size)
    ]
```

Replace `dryrun/component/rollout/engine/kv/__init__.py`:

```python
"""KV block pool: occupancy, prefix reuse, and eviction.

This is a state machine, not a cost model. It fits no coefficient, reads no
artifact, and has no domain gate. `dryrun/cost/rollout/kv_cache.py` answers
*how many blocks exist*; this package answers *who holds them and what can
hit*. The only interface between the two is the `kv_blocks` integer.
"""

from .block import KvBlock
from .lineage import block_hashes

__all__ = ["KvBlock", "block_hashes"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_kv_pool.py -q
```

Expected: 24 passed.

---

### Task 4: `BlockPool`

The pool owns allocation, the LRU free queue, the hash map, and eviction. Its central invariant: **a block may be in the free queue and in the hash map at the same time.** That is the "nobody holds it, but it is still addressable" state, and every cross-sync and cross-round hit depends on it.

**Files:**
- Create: `dryrun/component/rollout/engine/kv/pool.py`
- Modify: `dryrun/component/rollout/engine/kv/__init__.py`
- Test: `tests/test_kv_pool.py`

**Interfaces:**
- Consumes: `KvBlock` from Task 2, `Recorder` and the entry types from Task 1.
- Produces: `BlockPool(num_blocks: int, enable_caching: bool = True)` with properties `num_free_blocks: int`, `num_used_blocks: int`, `usage: float`, `eviction_count: int`; methods `cached_block(block_hash: Hashable) -> KvBlock | None`, `new_blocks(n: int) -> list[KvBlock]`, `touch(blocks: list[KvBlock]) -> None`, `free_blocks(blocks: Iterable[KvBlock]) -> None`, `cache_full_blocks(blocks: list[KvBlock], hashes: list[Hashable], n_cached: int, n_full: int) -> None`, `reset_hashes() -> None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kv_pool.py`:

```python
from dryrun.component.rollout.engine.kv import BlockPool


class TestBlockPoolAllocation:
    def test_starts_entirely_free(self):
        pool = BlockPool(num_blocks=10)
        assert pool.num_free_blocks == 10
        assert pool.num_used_blocks == 0
        assert pool.usage == 0.0

    def test_new_blocks_are_held_and_removed_from_free(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(3, NULL_RECORDER)
        assert len(blocks) == 3
        assert all(b.ref_cnt == 1 for b in blocks)
        assert pool.num_free_blocks == 7
        assert pool.usage == 0.3

    def test_overallocation_raises_rather_than_overcommitting(self):
        pool = BlockPool(num_blocks=2)
        with pytest.raises(ValueError, match="free"):
            pool.new_blocks(3, NULL_RECORDER)

    def test_freeing_returns_blocks_only_when_the_last_holder_leaves(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(2, NULL_RECORDER)
        pool.touch(blocks, NULL_RECORDER)  # a second holder
        pool.free_blocks(blocks, NULL_RECORDER)
        assert pool.num_free_blocks == 8, "still held by the second holder"
        pool.free_blocks(blocks, NULL_RECORDER)
        assert pool.num_free_blocks == 10


class TestBlockPoolHits:
    def test_a_cached_hash_is_findable(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(2, NULL_RECORDER)
        pool.cache_full_blocks(blocks, [("p", 1, 0), ("p", 1, 1)], n_cached=0, n_full=2, recorder=NULL_RECORDER)
        assert pool.cached_block(("p", 1, 0)) is blocks[0]
        assert pool.cached_block(("p", 1, 1)) is blocks[1]

    def test_an_unknown_hash_misses(self):
        assert BlockPool(num_blocks=10).cached_block(("p", 1, 0)) is None

    def test_a_freed_block_keeps_its_hash_and_still_hits(self):
        """The pool's central invariant. Reuse after a weight update and across
        an agentic round boundary both happen on blocks whose ref_cnt is zero
        but whose hash is still mapped."""
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(blocks, [("p", 1, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER)
        pool.free_blocks(blocks, NULL_RECORDER)
        assert pool.num_free_blocks == 10
        assert pool.cached_block(("p", 1, 0)) is blocks[0]

    def test_touch_revives_a_free_block_without_double_counting(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(blocks, [("p", 1, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER)
        pool.free_blocks(blocks, NULL_RECORDER)
        hit = pool.cached_block(("p", 1, 0))
        pool.touch([hit], NULL_RECORDER)
        assert hit.ref_cnt == 1
        assert pool.num_free_blocks == 9

    def test_cache_full_blocks_skips_already_cached_prefix(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(3, NULL_RECORDER)
        pool.cache_full_blocks(blocks, [("p", 1, 0), ("p", 1, 1), ("p", 1, 2)], n_cached=2, n_full=3, recorder=NULL_RECORDER)
        assert pool.cached_block(("p", 1, 0)) is None
        assert pool.cached_block(("p", 1, 2)) is blocks[2]

    def test_caching_is_a_no_op_when_disabled(self):
        pool = BlockPool(num_blocks=10, enable_caching=False)
        blocks = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(blocks, [("p", 1, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER)
        assert pool.cached_block(("p", 1, 0)) is None

    def test_reset_hashes_clears_the_map_but_not_occupancy(self):
        """A weight update calls reset_prefix_cache, which empties the block
        hash map. Blocks a live request still holds stay held."""
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(2, NULL_RECORDER)
        pool.cache_full_blocks(blocks, [("p", 1, 0), ("p", 1, 1)], n_cached=0, n_full=2, recorder=NULL_RECORDER)
        pool.reset_hashes(NULL_RECORDER)
        assert pool.cached_block(("p", 1, 0)) is None
        assert pool.num_free_blocks == 8
        assert blocks[0].block_hash is None


class TestBlockPoolEviction:
    def test_reuse_evicts_the_oldest_freed_block_first(self):
        pool = BlockPool(num_blocks=2)
        first = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(first, [("p", 1, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER)
        second = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(second, [("p", 2, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER)
        pool.free_blocks(first, NULL_RECORDER)
        pool.free_blocks(second, NULL_RECORDER)
        # Both are free and both still hit. Allocating one must take the
        # least recently freed, which is `first`.
        pool.new_blocks(1, NULL_RECORDER)
        assert pool.cached_block(("p", 1, 0)) is None, "oldest freed was evicted"
        assert pool.cached_block(("p", 2, 0)) is second[0], "newer one survived"
        assert pool.eviction_count == 1

    def test_eviction_frees_the_hash_for_reassignment(self):
        pool = BlockPool(num_blocks=1)
        blocks = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(blocks, [("p", 1, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER)
        pool.free_blocks(blocks, NULL_RECORDER)
        reused = pool.new_blocks(1, NULL_RECORDER)
        assert reused[0].block_hash is None
        pool.cache_full_blocks(reused, [("p", 9, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER)
        assert pool.cached_block(("p", 9, 0)) is reused[0]


def _pool_state(pool: BlockPool) -> tuple:
    """Everything an undo has to restore.

    Occupancy alone is not enough: a block put back in the wrong free-list slot
    leaves `num_used_blocks` right while rewriting LRU order, so the list order
    and every block's refcount and hash are all part of the state.
    """
    return (
        pool.num_used_blocks,
        pool.free_list_ids(),
        pool.eviction_count,
        dict(pool._hash_to_block),
        [(b.ref_cnt, b.block_hash) for b in pool.blocks],
    )


class TestBlockPoolUndo:
    def test_undoing_allocation_restores_occupancy_and_order(self):
        pool = BlockPool(num_blocks=10)
        before = _pool_state(pool)
        log = UndoLog()
        pool.new_blocks(3, log)
        log.rollback()
        assert _pool_state(pool) == before

    def test_undoing_a_hit_restores_the_free_list_position(self):
        """A hit removes a block from the middle of the LRU queue. Putting it
        back at the tail would keep occupancy correct and the eviction order
        wrong, which is why the list is linked rather than an OrderedDict."""
        pool = BlockPool(num_blocks=6)
        blocks = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(blocks, [("p", 1, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER)
        pool.free_blocks(blocks, NULL_RECORDER)
        before = _pool_state(pool)

        log = UndoLog()
        pool.touch([pool.cached_block(("p", 1, 0))], log)
        log.rollback()
        assert _pool_state(pool) == before

    def test_undoing_eviction_restores_the_hash_and_the_counter(self):
        """eviction_count feeds telemetry as eviction pressure, so an
        uncommitted event must not inflate it permanently."""
        pool = BlockPool(num_blocks=1)
        blocks = pool.new_blocks(1, NULL_RECORDER)
        pool.cache_full_blocks(blocks, [("p", 1, 0)], n_cached=0, n_full=1, recorder=NULL_RECORDER)
        pool.free_blocks(blocks, NULL_RECORDER)
        before = _pool_state(pool)

        log = UndoLog()
        pool.new_blocks(1, log)
        assert pool.cached_block(("p", 1, 0)) is None and pool.eviction_count == 1
        log.rollback()
        assert _pool_state(pool) == before

    def test_undoing_caching_removes_the_hashes_it_added(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(2, NULL_RECORDER)
        before = _pool_state(pool)
        log = UndoLog()
        pool.cache_full_blocks(
            blocks, [("p", 1, 0), ("p", 1, 1)], n_cached=0, n_full=2, recorder=log
        )
        log.rollback()
        assert _pool_state(pool) == before

    def test_undoing_reset_hashes_restores_the_whole_map(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(2, NULL_RECORDER)
        pool.cache_full_blocks(
            blocks, [("p", 1, 0), ("p", 1, 1)], n_cached=0, n_full=2, recorder=NULL_RECORDER
        )
        before = _pool_state(pool)

        log = UndoLog()
        pool.reset_hashes(log)
        assert pool.cached_block(("p", 1, 0)) is None
        log.rollback()
        assert _pool_state(pool) == before

    def test_undoing_a_mixed_event_restores_every_field(self):
        """One event can hit, allocate, cache and free. This is the test that
        would catch a mutation someone forgot to record."""
        pool = BlockPool(num_blocks=8)
        blocks = pool.new_blocks(2, NULL_RECORDER)
        pool.cache_full_blocks(
            blocks, [("p", 1, 0), ("p", 1, 1)], n_cached=0, n_full=2, recorder=NULL_RECORDER
        )
        pool.free_blocks(blocks, NULL_RECORDER)
        before = _pool_state(pool)

        log = UndoLog()
        pool.touch([pool.cached_block(("p", 1, 0))], log)
        fresh = pool.new_blocks(3, log)
        pool.cache_full_blocks(
            fresh, [("r", 9, 0), ("r", 9, 1), ("r", 9, 2)], n_cached=0, n_full=3, recorder=log
        )
        pool.free_blocks(fresh, log)
        assert _pool_state(pool) != before
        log.rollback()
        assert _pool_state(pool) == before

    def test_null_recorder_leaves_the_mutation_standing(self):
        pool = BlockPool(num_blocks=10)
        pool.new_blocks(3, NULL_RECORDER)
        assert pool.num_used_blocks == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_kv_pool.py -q
```

Expected: FAIL, `ImportError: cannot import name 'BlockPool'`.

- [ ] **Step 3: Write minimal implementation**

Create `dryrun/component/rollout/engine/kv/pool.py`:

```python
"""The KV block pool: allocation, LRU reuse, and eviction.

Ported from vLLM v1 (`vllm/v1/core/block_pool.py`) rather than approximated,
because what this has to reproduce is an engine's exact admission and reuse
behaviour under pressure, and that behaviour lives in the mechanism.

The central invariant: **a block may sit in the free queue and in the hash map
at the same time.** That is the "nobody holds it, but it is still addressable"
state. Reuse across a weight update and across an agentic round boundary both
land on exactly those blocks, so a pool that dropped hashes on release would
be wrong in the two cases that matter most here.

A hash is destroyed only when its block is handed to a new owner
(`new_blocks` reaching `_maybe_evict`), never on release.

Every mutating method takes a `Recorder`. Occupancy, hashes and list order are
all registered through it, so `undo_last_event` can unwind an uncommitted event
without the pool ever being copied. `NULL_RECORDER` is what committed paths pass.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Hashable, Optional

from ..undo import AttrSnapshot, MappingKeysAdded, Recorder
from .block import KvBlock
from .free_list import FreeBlockList


class BlockPool:
    """A fixed set of KV blocks, with prefix-cache bookkeeping.

    Args:
        num_blocks: Pool capacity, from `dryrun/cost/rollout/kv_cache.py`.
        enable_caching: When False, blocks are still allocated and freed but no
            hash is ever recorded, so nothing can hit. This is a data state,
            not a second code path.

    Raises:
        ValueError: If `num_blocks` is negative.
    """

    def __init__(self, num_blocks: int, enable_caching: bool = True) -> None:
        if num_blocks < 0:
            raise ValueError(f"Expected num_blocks >= 0, got {num_blocks!r}.")
        self.num_blocks = int(num_blocks)
        self.enable_caching = bool(enable_caching)
        self.blocks = [KvBlock(block_id=i) for i in range(self.num_blocks)]
        self._free = FreeBlockList(self.blocks)
        self._hash_to_block: dict[Hashable, KvBlock] = {}
        self.eviction_count = 0

    @property
    def num_free_blocks(self) -> int:
        return self._free.num_free

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self._free.num_free

    @property
    def usage(self) -> float:
        """Fraction of the pool currently held, 0.0 for an empty pool."""
        if self.num_blocks <= 0:
            return 0.0
        return self.num_used_blocks / self.num_blocks

    def free_list_ids(self) -> list[int]:
        """Free block ids, least recently freed first. Diagnostics and tests.

        O(num_free), so not for the hot path. Undo tests need it because
        occupancy counts cannot detect a reordering of the LRU queue.
        """
        return self._free.ids()

    def cached_block(self, block_hash: Hashable) -> Optional[KvBlock]:
        """The block holding `block_hash`, or None on a miss."""
        return self._hash_to_block.get(block_hash)

    def new_blocks(self, n: int, recorder: Recorder) -> list[KvBlock]:
        """Take `n` unheld blocks, evicting their hashes as they change hands.

        Args:
            n: How many blocks to allocate.
            recorder: Where to register old values for rollback.

        Returns:
            The allocated blocks, each with `ref_cnt == 1`.

        Raises:
            ValueError: If fewer than `n` blocks are free. The caller is
                expected to have checked capacity; overcommitting the pool
                would silently invent memory the engine does not have.
        """
        if n > self.num_free_blocks:
            raise ValueError(
                f"Cannot allocate {n} KV blocks: only {self.num_free_blocks} free."
            )
        taken: list[KvBlock] = []
        for _ in range(n):
            block = self._free.popleft(recorder)
            self._maybe_evict(block, recorder)
            self._record_block(block, recorder)
            block.incr_ref()
            taken.append(block)
        return taken

    def touch(self, blocks: list[KvBlock], recorder: Recorder) -> None:
        """Take a reference to blocks found by hash.

        A hit block may currently be free, in which case it has to leave the
        free list before anyone can grow into it. That removal is from the middle
        of the list, which is the case the linked list exists to make reversible.
        """
        for block in blocks:
            self._record_block(block, recorder)
            if block.ref_cnt == 0:
                self._free.remove(block, recorder)
            block.incr_ref()

    def free_blocks(self, blocks: Iterable[KvBlock], recorder: Recorder) -> None:
        """Release a reference, returning blocks whose last holder left.

        The hash is deliberately kept: a released block stays addressable so a
        later request with the same prefix can hit it.
        """
        for block in blocks:
            self._record_block(block, recorder)
            block.decr_ref()
            if block.ref_cnt == 0:
                self._free.append(block, recorder)

    def cache_full_blocks(
        self,
        blocks: list[KvBlock],
        hashes: list[Hashable],
        n_cached: int,
        n_full: int,
        recorder: Recorder,
    ) -> None:
        """Record hashes for blocks that have become full.

        Args:
            blocks: This request's blocks, in block order.
            hashes: This request's block hashes, in the same order.
            n_cached: How many leading blocks are already recorded.
            n_full: How many leading blocks are now full.
            recorder: Where to register old values for rollback.
        """
        if not self.enable_caching or n_cached >= n_full:
            return
        added = MappingKeysAdded.capture(self._hash_to_block)
        recorder.record(added)
        for index in range(n_cached, min(n_full, len(hashes), len(blocks))):
            block = blocks[index]
            if block.block_hash is None:
                self._record_block(block, recorder)
                block.block_hash = hashes[index]
                self._hash_to_block[block.block_hash] = block
                added.note(block.block_hash)

    def reset_hashes(self, recorder: Recorder) -> None:
        """Empty the hash map, leaving occupancy untouched.

        This is vLLM's `reset_prefix_cache`, which a weight update triggers via
        PSRL's `pull_model`. Blocks a live request still holds stay held; they
        simply stop being addressable by content.
        """
        if not self._hash_to_block:
            return
        recorder.record(_HashMapRestore(self, dict(self._hash_to_block)))
        for block in self._hash_to_block.values():
            self._record_block(block, recorder)
            block.reset_hash()
        self._hash_to_block.clear()

    def _maybe_evict(self, block: KvBlock, recorder: Recorder) -> bool:
        """Drop a block's hash as it is handed to a new owner."""
        if block.block_hash is None:
            return False
        evicted_hash = block.block_hash
        recorder.record(_HashMapEntry(self._hash_to_block, evicted_hash, block))
        recorder.record_once(
            ("evictions", id(self)),
            lambda: AttrSnapshot.capture(self, "eviction_count"),
        )
        self._hash_to_block.pop(evicted_hash, None)
        block.reset_hash()
        self.eviction_count += 1
        return True

    @staticmethod
    def _record_block(block: KvBlock, recorder: Recorder) -> None:
        """Capture a block's refcount and hash, once per event."""
        recorder.record_once(
            ("block", id(block)),
            lambda: AttrSnapshot.capture(block, "ref_cnt", "_block_hash"),
        )


class _HashMapEntry:
    """Restores one evicted `hash -> block` entry.

    Not an `AttrSnapshot`: the old value is a mapping entry rather than an
    attribute, and `MappingKeysAdded` is for additions rather than removals.
    """

    __slots__ = ("_mapping", "_key", "_block")

    def __init__(self, mapping: dict, key: Hashable, block: KvBlock) -> None:
        self._mapping, self._key, self._block = mapping, key, block

    def restore(self) -> None:
        self._mapping[self._key] = self._block


class _HashMapRestore:
    """Restores the whole hash map after a `reset_hashes`.

    This is the one entry whose size is proportional to the map rather than to
    the event, and it is correct for it to be: `reset_hashes` *is* an
    event-sized operation. It fires once per weight update, not per step.
    """

    __slots__ = ("_pool", "_entries")

    def __init__(self, pool: "BlockPool", entries: dict) -> None:
        self._pool, self._entries = pool, entries

    def restore(self) -> None:
        self._pool._hash_to_block.update(self._entries)
```

Replace `dryrun/component/rollout/engine/kv/__init__.py`:

```python
"""KV block pool: occupancy, prefix reuse, and eviction.

This is a state machine, not a cost model. It fits no coefficient, reads no
artifact, and has no domain gate. `dryrun/cost/rollout/kv_cache.py` answers
*how many blocks exist*; this package answers *who holds them and what can
hit*. The only interface between the two is the `kv_blocks` integer.

Every mutating method takes a `Recorder` (see `..undo`), because
`advance_one_event` is speculative and the driver rolls back the events it does
not commit.
"""

from .block import KvBlock
from .free_list import FreeBlockList
from .lineage import block_hashes
from .pool import BlockPool

__all__ = ["BlockPool", "FreeBlockList", "KvBlock", "block_hashes"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_kv_pool.py -q
```

Expected: 44 passed.

---

### Task 5: `KvCacheManager`

The four methods `NativeInstance` calls. This is where the longest-common-prefix hit lookup lives, including vLLM's full-hit edge case.

**Files:**
- Create: `dryrun/component/rollout/engine/kv/manager.py`
- Modify: `dryrun/component/rollout/engine/kv/__init__.py`
- Test: `tests/test_kv_pool.py`

**Interfaces:**
- Consumes: `BlockPool` (Task 4), `block_hashes` (Task 3).
- Produces: `KvCacheManager(num_blocks: int, block_size: int = 16, enable_caching: bool = True)` with attributes `pool: BlockPool`, `queries: int`, `hits: int`; properties `num_used_blocks: int`, `num_free_blocks: int`, `usage: float`; methods `computed_blocks(req) -> tuple[list[KvBlock], int]`, `can_allocate(req, n_tokens, hit_blocks) -> bool`, `allocate(req, n_tokens, hit_blocks) -> list[KvBlock] | None`, `free(req) -> None`, `reset_hashes() -> None`, `blocks_held(req) -> int`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kv_pool.py`:

```python
from dryrun.component.rollout.engine.kv import KvCacheManager


class TestComputedBlocks:
    def test_a_cold_manager_hits_nothing(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        blocks, tokens = mgr.computed_blocks(_req(rid=1, prompt_len=150, source_group_id=42))
        assert blocks == []
        assert tokens == 0

    def test_a_sibling_hits_every_shared_prompt_block(self):
        """prompt_len=150 shares 9 full blocks, so 144 of 150 prompt tokens are
        free for the second response onward. This is the limitation in
        docs/reference/limitations.md that reads 'Prompt prefill charged once
        per response'."""
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        first = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(first, n_tokens=150, hit_blocks=[])

        sibling = _req(rid=2, prompt_len=150, source_group_id=42)
        blocks, tokens = mgr.computed_blocks(sibling)
        assert len(blocks) == 9
        assert tokens == 144

    def test_hits_stop_at_the_first_miss(self):
        """Hashes carry chained semantics, so a miss at block j means content
        has diverged and later blocks must not be used even if they collide."""
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        req = _req(rid=1, prompt_len=160, source_group_id=42, generated=64)
        mgr.allocate(req, n_tokens=224, hit_blocks=[])
        mgr.free(req)
        mgr.pool.reset_hashes(NULL_RECORDER)

        # Re-cache only the first two blocks by replaying a short prefix.
        short = _req(rid=1, prompt_len=32, source_group_id=42)
        mgr.allocate(short, n_tokens=32, hit_blocks=[])

        blocks, tokens = mgr.computed_blocks(
            _req(rid=1, prompt_len=160, source_group_id=42, generated=64)
        )
        assert len(blocks) == 2
        assert tokens == 32

    def test_a_full_hit_hands_the_last_block_back(self):
        """vLLM's own edge case
        (vllm_v1_engine_replica_scheduler.py:1393-1399): a full hit leaves zero
        new tokens, but the model must forward at least one token to produce
        the next logits. Fires for real whenever prompt_len is a multiple of
        block_size."""
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        first = _req(rid=1, prompt_len=160, source_group_id=42)
        mgr.allocate(first, n_tokens=160, hit_blocks=[])

        sibling = _req(rid=2, prompt_len=160, source_group_id=42)
        blocks, tokens = mgr.computed_blocks(sibling)
        assert len(blocks) == 9, "ten blocks match, but the last is handed back"
        assert tokens == 144
        assert sibling.prompt_len - tokens == 16

    def test_counters_track_queries_and_hits_in_blocks(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        first = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(first, n_tokens=150, hit_blocks=[])
        mgr.computed_blocks(_req(rid=2, prompt_len=150, source_group_id=42))
        assert mgr.queries == 18, "9 blocks for the cold request, 9 for the sibling"
        assert mgr.hits == 9

    def test_disabled_caching_never_hits(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16, enable_caching=False)
        first = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(first, n_tokens=150, hit_blocks=[])
        blocks, tokens = mgr.computed_blocks(_req(rid=2, prompt_len=150, source_group_id=42))
        assert blocks == []
        assert tokens == 0


class TestAllocation:
    def test_allocation_accounts_occupancy(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        req = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(req, n_tokens=150, hit_blocks=[])
        assert mgr.num_used_blocks == 10, "ceil(150/16)"
        assert mgr.blocks_held(req) == 10

    def test_a_hit_sibling_only_allocates_its_own_tail(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        first = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(first, n_tokens=150, hit_blocks=[])
        used_after_first = mgr.num_used_blocks

        sibling = _req(rid=2, prompt_len=150, source_group_id=42)
        hit_blocks, hit_tokens = mgr.computed_blocks(sibling)
        sibling.prefilled = hit_tokens
        mgr.allocate(sibling, n_tokens=sibling.prompt_len - hit_tokens, hit_blocks=hit_blocks)
        assert mgr.num_used_blocks == used_after_first + 1, "only the private tail block"

    def test_can_allocate_refuses_what_does_not_fit(self):
        mgr = KvCacheManager(num_blocks=4, block_size=16)
        req = _req(rid=1, prompt_len=150, source_group_id=42)
        assert not mgr.can_allocate(req, n_tokens=150, hit_blocks=[])
        assert mgr.allocate(req, n_tokens=150, hit_blocks=[]) is None

    def test_free_releases_everything_the_request_held(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        req = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(req, n_tokens=150, hit_blocks=[])
        mgr.free(req)
        assert mgr.num_used_blocks == 0
        assert mgr.blocks_held(req) == 0

    def test_a_resumed_request_hits_its_own_generated_blocks(self):
        """Partial rollout: an abort keeps `generated` and zeroes `prefilled`,
        so the context must be re-prefilled. Its generated blocks hash exactly
        as they did when first filled, so they hit."""
        mgr = KvCacheManager(num_blocks=1000, block_size=16)
        req = _req(rid=5, prompt_len=160, source_group_id=42, generated=320)
        mgr.allocate(req, n_tokens=480, hit_blocks=[])
        mgr.free(req)

        req.prefilled = 0
        blocks, tokens = mgr.computed_blocks(req)
        assert tokens == 464, "30 of 30 full blocks hit, less the handed-back last one"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_kv_pool.py -q
```

Expected: FAIL, `ImportError: cannot import name 'KvCacheManager'`.

- [ ] **Step 3: Write minimal implementation**

Create `dryrun/component/rollout/engine/kv/manager.py`:

```python
"""Per-instance KV cache manager: the engine's view of the block pool.

Mirrors vLLM v1's `KVCacheManager`. `NativeInstance` talks to this and never to
`BlockPool` directly, so occupancy accounting and hit lookup stay in one place.
"""

from __future__ import annotations

from typing import Hashable, TYPE_CHECKING

from .block import KvBlock
from .lineage import block_hashes
from .pool import BlockPool

if TYPE_CHECKING:
    from ..request import Request


def _blocks_for(n_tokens: int, block_size: int) -> int:
    """Blocks needed to hold `n_tokens`, rounding up."""
    return (n_tokens + block_size - 1) // block_size


class KvCacheManager:
    """Allocation and prefix lookup for one engine instance.

    Args:
        num_blocks: Pool capacity.
        block_size: Tokens per block.
        enable_caching: When False no hash is recorded, so nothing hits.

    Attributes:
        queries: Blocks looked up since construction, counted per block to
            match vLLM's `vllm:prefix_cache_queries`.
        hits: Blocks found, matching `vllm:prefix_cache_hits`.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        enable_caching: bool = True,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"Expected block_size > 0, got {block_size!r}.")
        self.block_size = int(block_size)
        self.enable_caching = bool(enable_caching)
        self.pool = BlockPool(num_blocks, enable_caching=enable_caching)
        self._held: dict[int, list[KvBlock]] = {}
        self._n_cached: dict[int, int] = {}
        self.queries = 0
        self.hits = 0

    @property
    def num_used_blocks(self) -> int:
        return self.pool.num_used_blocks

    @property
    def num_free_blocks(self) -> int:
        return self.pool.num_free_blocks

    @property
    def usage(self) -> float:
        return self.pool.usage

    def blocks_held(self, req: "Request") -> int:
        """How many blocks this request currently holds."""
        return len(self._held.get(req.rid, ()))

    def computed_blocks(self, req: "Request") -> tuple[list[KvBlock], int]:
        """The longest cached prefix of this request's context.

        Walks the request's block hashes and stops at the first miss, because a
        hash carries chained semantics: a miss at block `j` means content has
        diverged there, so later blocks must not be reused even if their hashes
        happen to match.

        One edge case is ported verbatim from vLLM
        (`vllm_v1_engine_replica_scheduler.py:1393-1399`): on a *full* hit the
        last block is handed back. A full hit would leave zero new tokens, but
        the model must forward at least one token to produce the next logits.
        This fires for real whenever a group's `prompt_len` is a whole number
        of blocks.

        Args:
            req: The request being admitted or re-prefilled.

        Returns:
            The hit blocks and how many tokens they cover.
        """
        if not self.enable_caching:
            return [], 0

        hashes = block_hashes(req, self.block_size)
        hit: list[KvBlock] = []
        for block_hash in hashes:
            block = self.pool.cached_block(block_hash)
            if block is None:
                break
            hit.append(block)

        self.queries += len(hashes)
        self.hits += len(hit)

        tokens = len(hit) * self.block_size
        # A full hit must still leave one block of work to do.
        if hit and tokens >= req.prompt_len + req.generated:
            hit = hit[:-1]
            tokens -= self.block_size
            self.hits -= 1
        return hit, tokens

    def _new_blocks_needed(
        self,
        req: "Request",
        n_tokens: int,
        hit_blocks: list[KvBlock],
    ) -> int:
        """Blocks to allocate on top of what is held and what hit."""
        computed = req.prefilled + len(hit_blocks) * self.block_size
        required = _blocks_for(computed + n_tokens, self.block_size)
        return required - self.blocks_held(req) - len(hit_blocks)

    def can_allocate(
        self,
        req: "Request",
        n_tokens: int,
        hit_blocks: list[KvBlock],
    ) -> bool:
        """Whether `n_tokens` more of this request fit.

        Hit blocks that are currently free do not consume new capacity, so they
        are discounted from the free count rather than charged twice.
        """
        needed = self._new_blocks_needed(req, n_tokens, hit_blocks)
        revivable = sum(1 for b in hit_blocks if b.ref_cnt == 0)
        return needed <= self.pool.num_free_blocks - revivable

    def allocate(
        self,
        req: "Request",
        n_tokens: int,
        hit_blocks: list[KvBlock],
    ) -> list[KvBlock] | None:
        """Reserve room for `n_tokens` more of this request.

        Args:
            req: The request. `req.prefilled` must already reflect any hit.
            n_tokens: Tokens about to be computed.
            hit_blocks: Blocks from `computed_blocks`, adopted by this request.

        Returns:
            The newly allocated blocks, or None when capacity refuses.

        Raises:
            ValueError: If hit blocks are passed while caching is disabled,
                which would mean a caller invented a hit.
        """
        if hit_blocks and not self.enable_caching:
            raise ValueError("Hit blocks passed while prefix caching is disabled.")
        if not self.can_allocate(req, n_tokens, hit_blocks):
            return None

        self.pool.touch(hit_blocks)
        held = self._held.setdefault(req.rid, [])
        held.extend(hit_blocks)

        needed = self._new_blocks_needed(req, n_tokens, hit_blocks)
        fresh = self.pool.new_blocks(needed) if needed > 0 else []
        held.extend(fresh)

        if self.enable_caching:
            computed = req.prefilled + len(hit_blocks) * self.block_size
            n_full = (computed + n_tokens) // self.block_size
            self.pool.cache_full_blocks(
                held,
                block_hashes(req, self.block_size),
                n_cached=self._n_cached.get(req.rid, len(hit_blocks)),
                n_full=n_full,
            )
            self._n_cached[req.rid] = n_full
        return fresh

    def free(self, req: "Request") -> None:
        """Release every block this request holds, keeping their hashes.

        The hashes survive deliberately: a released block stays addressable, so
        a sibling or a later round can still hit it.
        """
        held = self._held.pop(req.rid, [])
        if held:
            self.pool.free_blocks(reversed(held))
        self._n_cached.pop(req.rid, None)

    def reset_hashes(self) -> None:
        """Empty the hash map. This is vLLM's `reset_prefix_cache`."""
        self.pool.reset_hashes(NULL_RECORDER)
        self._n_cached.clear()
```

Replace `dryrun/component/rollout/engine/kv/__init__.py`:

```python
"""KV block pool: occupancy, prefix reuse, and eviction.

This is a state machine, not a cost model. It fits no coefficient, reads no
artifact, and has no domain gate. `dryrun/cost/rollout/kv_cache.py` answers
*how many blocks exist*; this package answers *who holds them and what can
hit*. The only interface between the two is the `kv_blocks` integer.
"""

from .block import KvBlock
from .lineage import block_hashes
from .manager import KvCacheManager
from .pool import BlockPool

__all__ = ["BlockPool", "KvBlock", "KvCacheManager", "block_hashes"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_kv_pool.py -q
```

Expected: 55 passed.

---

### Task 6: Wire the pool into `NativeInstance` occupancy

Replace the derived occupancy sum with pool queries. No hit logic yet, so this must be behaviour-preserving: the existing suite is the test.

**Files:**
- Modify: `dryrun/component/rollout/engine/instance.py:20-74`
- Test: `tests/test_kv_instance.py`

**Interfaces:**
- Consumes: `KvCacheManager` (Task 5).
- Produces: `NativeInstance.__init__` gains keyword `enable_prefix_caching: bool = True`; attribute `self.kv: KvCacheManager`. `blocks_used()` and `free_blocks()` keep their signatures.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kv_instance.py`:

```python
"""Tests for prefix caching inside the rollout engine instance."""

from __future__ import annotations

import pytest

from dryrun.component.rollout.engine import NativeInstance, ReqStatus, Request
from dryrun.cost.rollout import PSRLFitted


def _cost() -> PSRLFitted:
    return PSRLFitted()


def _inst(**kwargs) -> NativeInstance:
    defaults = dict(
        cost=_cost(),
        token_budget=8192,
        kv_blocks=1000,
        block_size=16,
    )
    defaults.update(kwargs)
    return NativeInstance(**defaults)


def _req(rid: int, prompt_len: int, source_group_id: int, target_len: int = 64) -> Request:
    return Request(
        rid=rid,
        prompt_len=prompt_len,
        target_len=target_len,
        v_traj=0,
        group_id=source_group_id,
        source_group_id=source_group_id,
    )


class TestOccupancyComesFromThePool:
    def test_an_idle_instance_holds_nothing(self):
        inst = _inst()
        assert inst.blocks_used() == 0
        assert inst.free_blocks() == 1000

    def test_admitting_then_prefilling_charges_blocks(self):
        inst = _inst()
        assert inst.add_request(_req(rid=1, prompt_len=150, source_group_id=1), t=0.0)
        inst.advance_one_event(version=0)
        assert inst.blocks_used() > 0
        assert inst.blocks_used() + inst.free_blocks() == 1000

    def test_the_pool_and_the_instance_agree_after_a_full_run(self):
        inst = _inst()
        for rid in range(4):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=rid), t=0.0)
        for _ in range(200):
            if inst.advance_one_event(version=0) is None:
                break
        assert len(inst.completed) == 4
        assert inst.blocks_used() == 0, "every retired request released its blocks"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_kv_instance.py -q
```

Expected: FAIL. `NativeInstance.__init__` takes `cost` positionally but not as a keyword under that name, or the final assertion fails because `blocks_used()` still derives from `prefilled` and retired requests are not released.

- [ ] **Step 3: Write minimal implementation**

In `dryrun/component/rollout/engine/instance.py`, add the import at the top alongside the existing ones:

```python
from .kv import KvCacheManager
```

In `__init__`, add the parameter after `reject_if_running_full: bool = False`:

```python
        enable_prefix_caching: bool = True,
```

and replace the three KV lines in the body:

```python
        self.kv_blocks = kv_blocks
        self.block_size = block_size
```

with:

```python
        self.kv_blocks = kv_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        # Occupancy, hit lookup, and eviction all live in the pool. Disabling
        # prefix caching leaves the pool doing block accounting and simply
        # records no hashes, so there is no second code path to keep in step.
        self.kv = KvCacheManager(
            num_blocks=kv_blocks,
            block_size=block_size,
            enable_caching=enable_prefix_caching,
        )
```

Replace `blocks_used` and `free_blocks`:

```python
    def blocks_used(self) -> int:
        return self.kv.num_used_blocks

    def free_blocks(self) -> int:
        return self.kv.num_free_blocks
```

Every place that changes a request's `prefilled` must now tell the pool, and **`allocate` must be called before `prefilled` advances**, because it reads `req.prefilled` as the count already computed.

In `_prefill_budget_step`, replace the block

```python
            free -= blocks_for(req.prefilled + chunk, self.block_size) - have
            req.prefilled += chunk
```

with:

```python
            self.kv.allocate(req, n_tokens=chunk, hit_blocks=[])
            free -= blocks_for(req.prefilled + chunk, self.block_size) - have
            req.prefilled += chunk
```

In the decode loop of `_prefill_budget_step`, replace

```python
            r.add_tokens(version, 1)
            r.prefilled += 1
```

with:

```python
            self.kv.allocate(r, n_tokens=1, hit_blocks=[])
            r.add_tokens(version, 1)
            r.prefilled += 1
```

In `_decode_one_event` and in `advance_to`, replace both occurrences of

```python
                for r in self.running:
                    r.add_tokens(version, k)
                    r.prefilled += k
```

with:

```python
                for r in self.running:
                    self.kv.allocate(r, n_tokens=k, hit_blocks=[])
                    r.add_tokens(version, k)
                    r.prefilled += k
```

In `_retire`, release blocks as each request leaves:

```python
    def _retire(self, reqs: list[Request]) -> list[Request]:
        out = []
        for r in reqs:
            if r in self.running:
                self.running.remove(r)
            self.kv.free(r)
            r.status = ReqStatus.DONE
            r.complete_time = self.t_local
            self.completed.append(r)
            out.append(r)
        return out
```

In `_release`, the context is discarded and must be given back:

```python
    def _release(self, req: Request, keep_tokens: bool = True) -> None:
        req.num_aborts += 1
        if keep_tokens:
            req.reprefill_tokens += req.ctx
        self.kv.free(req)
        req.prefilled = 0
```

In `_preempt_one`, likewise:

```python
        self.running.remove(victim)
        victim.status = ReqStatus.PREEMPTED
        victim.num_preemptions += 1
        victim.reprefill_tokens += victim.ctx
        self.kv.free(victim)
        victim.prefilled = 0
        self.waiting.insert(0, victim)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_kv_instance.py -q && pytest tests/ -q
```

Expected: `tests/test_kv_instance.py` 3 passed, and the whole suite green. If `test_e2e.py` or `test_sim_telemetry.py` reddens, the pool and the requests disagree somewhere: find the `prefilled` mutation that skipped `self.kv`.

---

### Task 7: Rebuild `InstanceSnapshot` on `UndoLog`

**This is a correctness requirement the spec did not state.** `undo_last_event` (`instance.py:80`) is a live path: `simulator.py:358` calls it through `_undo_and_align` for every speculatively-advanced event that does not commit. `InstanceSnapshot` captures `prefilled` per request, which sufficed while occupancy was *derived* from `prefilled`. Now that the pool holds refcounts, a hash map and a free list, an uncommitted prefill step would roll back `prefilled` while leaving pool state mutated, and the two would silently disagree.

The pool already records its own old values (Task 4), so this task wires the recorder in and converts the existing per-request and per-instance captures onto the same mechanism. After it, `InstanceSnapshot` is a thin holder of one `UndoLog` rather than a dataclass that has to know every subsystem's fields.

**Files:**
- Modify: `dryrun/component/rollout/engine/snapshot.py`
- Modify: `dryrun/component/rollout/engine/instance.py`
- Modify: `dryrun/component/rollout/engine/kv/manager.py`
- Test: `tests/test_kv_instance.py`

**Interfaces:**
- Consumes: `UndoLog`, `AttrSnapshot`, `TruncateList`, `NULL_RECORDER` (Task 1); `BlockPool`'s recorder-taking methods (Task 4); `NativeInstance.kv` (Task 6).
- Produces: `InstanceSnapshot.capture(inst) -> InstanceSnapshot` unchanged in signature, now holding `log: UndoLog`; `restore(inst)` calls `log.rollback()`. `KvCacheManager` methods gain a trailing `recorder: Recorder = NULL_RECORDER` parameter. `NativeInstance._recorder` holds the active recorder, `NULL_RECORDER` outside a speculative event.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kv_instance.py`:

```python
def _instance_state(inst) -> tuple:
    """Everything an undo must restore, pool included.

    Occupancy alone is too weak: a block returned to the wrong free-list slot
    keeps `blocks_used()` correct while rewriting LRU order, which then decides
    every later eviction.
    """
    return (
        inst.t_local,
        [r.rid for r in inst.waiting],
        [r.rid for r in inst.running],
        len(inst.completed),
        len(inst.step_stats),
        inst.n_preemptions,
        inst.blocks_used(),
        inst.kv.pool.free_list_ids(),
        inst.kv.pool.eviction_count,
        inst.kv.queries,
        inst.kv.hits,
        [(r.rid, r.prefilled, r.generated, r.prefix_hit_tokens) for r in (*inst.waiting, *inst.running)],
    )


class TestUndoRestoresThePool:
    def test_undoing_an_event_restores_the_whole_instance(self):
        """simulator.py:358 undoes every speculatively advanced event that does
        not commit, so this is a live path rather than a safety net."""
        inst = _inst()
        assert inst.add_request(_req(rid=1, prompt_len=150, source_group_id=1), t=0.0)
        inst.advance_one_event(version=0)
        before = _instance_state(inst)

        inst.advance_one_event(version=0)
        inst.undo_last_event()
        assert _instance_state(inst) == before

    def test_undoing_restores_the_free_list_order_not_just_the_count(self):
        inst = _inst()
        for rid in range(4):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=rid), t=0.0)
        for _ in range(10):
            inst.advance_one_event(version=0)
        order_before = inst.kv.pool.free_list_ids()

        inst.advance_one_event(version=0)
        inst.undo_last_event()
        assert inst.kv.pool.free_list_ids() == order_before

    def test_undoing_restores_hit_bookkeeping(self):
        """queries and hits are compared against vllm:prefix_cache_* counters,
        so an uncommitted event must not inflate them."""
        inst = _inst()
        for rid in range(4):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=7), t=0.0)
        inst.advance_one_event(version=0)
        before = (inst.kv.queries, inst.kv.hits)

        inst.advance_one_event(version=0)
        inst.undo_last_event()
        assert (inst.kv.queries, inst.kv.hits) == before

    def test_repeated_advance_and_undo_does_not_drift(self):
        inst = _inst()
        for rid in range(4):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=rid), t=0.0)
        for _ in range(10):
            inst.advance_one_event(version=0)
        baseline = _instance_state(inst)
        for _ in range(20):
            inst.advance_one_event(version=0)
            inst.undo_last_event()
        assert _instance_state(inst) == baseline

    def test_a_decode_segment_is_undoable_too(self):
        """A decode segment advances every running request by k tokens at once,
        touching up to the whole pool. It must reverse like any other event."""
        inst = _inst()
        assert inst.add_request(_req(rid=1, prompt_len=150, source_group_id=1, target_len=512), t=0.0)
        for _ in range(20):
            inst.advance_one_event(version=0)
            if not inst.waiting:
                break
        before = _instance_state(inst)

        inst.advance_one_event(version=0)
        inst.undo_last_event()
        assert _instance_state(inst) == before

    def test_committed_advancement_is_not_undoable(self):
        """`advance_to` commits as it goes and passes NULL_RECORDER, so it pays
        nothing for recording and leaves no snapshot behind."""
        inst = _inst()
        assert inst.add_request(_req(rid=1, prompt_len=150, source_group_id=1), t=0.0)
        inst.advance_to(t_limit=1e9, version=0)
        assert inst._snapshot is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_kv_instance.py -q
```

Expected: FAIL on `test_undoing_an_event_restores_the_whole_instance`, because `blocks_used()` after the undo is higher than before: the uncommitted step's allocation survived.

- [ ] **Step 3: Write minimal implementation**

First give `KvCacheManager` a recorder parameter on every mutator. In `dryrun/component/rollout/engine/kv/manager.py`, add to the imports:

```python
from ..undo import NULL_RECORDER, AttrSnapshot, Recorder
```

Change the four mutating signatures and pass the recorder down. `computed_blocks` mutates the counters, so it records them too:

```python
    def computed_blocks(
        self, req: "Request", recorder: Recorder = NULL_RECORDER
    ) -> tuple[list[KvBlock], int]:
```

Immediately before `self.queries += len(hashes)`, add:

```python
        recorder.record_once(
            ("counters", id(self)),
            lambda: AttrSnapshot.capture(self, "queries", "hits"),
        )
```

Then thread the recorder through:

```python
    def allocate(
        self,
        req: "Request",
        n_tokens: int,
        hit_blocks: list[KvBlock],
        recorder: Recorder = NULL_RECORDER,
    ) -> list[KvBlock] | None:
```

with its three pool calls becoming `self.pool.touch(hit_blocks, recorder)`, `self.pool.new_blocks(needed, recorder)` and `self.pool.cache_full_blocks(..., recorder=recorder)`. The `_held` and `_n_cached` dictionaries are mutated too, so record them:

```python
        recorder.record_once(
            ("held", id(self), req.rid),
            lambda: _HeldBlocks.capture(self, req.rid),
        )
```

and add that entry type at the end of the module:

```python
class _HeldBlocks:
    """Restores one request's held-block list and cached-block count.

    Both are per-request dict entries rather than attributes, so neither
    `AttrSnapshot` nor `MappingKeysAdded` fits: the entry may be created,
    extended, or removed within one event.
    """

    __slots__ = ("_manager", "_rid", "_blocks", "_n_cached", "_existed")

    def __init__(self, manager: "KvCacheManager", rid: int) -> None:
        self._manager = manager
        self._rid = rid
        self._existed = rid in manager._held
        self._blocks = list(manager._held.get(rid, ()))
        self._n_cached = manager._n_cached.get(rid)

    @classmethod
    def capture(cls, manager: "KvCacheManager", rid: int) -> "_HeldBlocks":
        return cls(manager, rid)

    def restore(self) -> None:
        if self._existed:
            self._manager._held[self._rid] = self._blocks
        else:
            self._manager._held.pop(self._rid, None)
        if self._n_cached is None:
            self._manager._n_cached.pop(self._rid, None)
        else:
            self._manager._n_cached[self._rid] = self._n_cached
```

Apply the same treatment to `free` and `reset_hashes`:

```python
    def free(self, req: "Request", recorder: Recorder = NULL_RECORDER) -> None:
        recorder.record_once(
            ("held", id(self), req.rid),
            lambda: _HeldBlocks.capture(self, req.rid),
        )
        held = self._held.pop(req.rid, [])
        if held:
            self.pool.free_blocks(reversed(held), recorder)
        self._n_cached.pop(req.rid, None)

    def reset_hashes(self, recorder: Recorder = NULL_RECORDER) -> None:
        """Empty the hash map. This is vLLM's `reset_prefix_cache`."""
        recorder.record(_CachedCounts(self, dict(self._n_cached)))
        self.pool.reset_hashes(recorder)
        self._n_cached.clear()
```

with:

```python
class _CachedCounts:
    """Restores the per-request cached-block counts after a `reset_hashes`."""

    __slots__ = ("_manager", "_counts")

    def __init__(self, manager: "KvCacheManager", counts: dict[int, int]) -> None:
        self._manager, self._counts = manager, counts

    def restore(self) -> None:
        self._manager._n_cached.update(self._counts)
```

Now rewrite `dryrun/component/rollout/engine/snapshot.py` entirely:

```python
"""Rollback state for speculative event advancement.

The simulator advances each instance to its next event and discards the events
it does not commit, so every mutation an event performs must be reversible.

This is a thin adapter over `..undo`: capturing an instance means opening an
`UndoLog`, registering the old values of what the instance itself owns, and
handing that log to the engine so the pool can register its own. Restoring means
rolling the log back. There is no per-subsystem field list here, which is what
lets a new subsystem become undoable without editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .request import Request
from .undo import AttrSnapshot, TruncateList, UndoLog

if TYPE_CHECKING:
    from .instance import NativeInstance

_REQUEST_FIELDS = (
    "prefilled",
    "generated",
    "status",
    "complete_time",
    "num_preemptions",
    "num_aborts",
    "reprefill_tokens",
    "prefix_hit_tokens",
)


def record_request(req: Request, recorder) -> None:
    """Register one request's mutable state, once per event.

    `segments` is append-only except for its last entry, which is incremented in
    place, so the length alone is not enough -- hence the extra entry. This is
    the same correction the original hand-written snapshot made.
    """
    recorder.record_once(
        ("req", id(req)),
        lambda: AttrSnapshot.capture(req, *_REQUEST_FIELDS),
    )
    recorder.record_once(("segs", id(req)), lambda: TruncateList.capture(req.segments))
    if req.segments:
        recorder.record_once(
            ("seg_tail", id(req)),
            lambda: AttrSnapshot.capture(req.segments[-1], "tokens"),
        )


@dataclass
class InstanceSnapshot:
    """One event's worth of captured old values.

    Args:
        log: The undo log the event recorded into.
    """

    log: UndoLog

    @classmethod
    def capture(cls, inst: "NativeInstance") -> "InstanceSnapshot":
        """Open a log and register what the instance itself owns.

        `completed` is deliberately captured by length rather than per request.
        Events only mutate requests in `waiting` / `running`, and a request
        retired *by* this event was in `running` at capture time, so it is
        already covered. Including it would make capture cost grow with the
        whole run instead of with the live batch.
        """
        log = UndoLog()
        log.record(AttrSnapshot.capture(inst, "t_local", "n_preemptions"))
        log.record(AttrSnapshot.capture(inst, "waiting", "running"))
        log.record(TruncateList.capture(inst.completed))
        log.record(TruncateList.capture(inst.step_stats))
        for req in (*inst.waiting, *inst.running):
            record_request(req, log)
        return cls(log=log)

    def restore(self, inst: "NativeInstance") -> None:
        """Roll every captured value back, newest first."""
        del inst  # The log holds its own targets.
        self.log.rollback()
```

Note that `capture` records `waiting` and `running` as attributes, so `restore` reinstates the *list objects* that were current at capture time. `NativeInstance` must therefore replace those lists rather than mutate them in place when an event reorders the queues; it already does (`self.waiting = aborted[:]`, `self.running = []`), with two exceptions to fix below.

In `dryrun/component/rollout/engine/instance.py`, add the import:

```python
from .undo import NULL_RECORDER
```

In `__init__`, add after `self._snapshot = None`:

```python
        # The recorder every mutating pool call receives. Swapped for a real log
        # only while a speculative event is in flight.
        self._recorder = NULL_RECORDER
```

Replace `_save_snapshot` and `undo_last_event`:

```python
    def _save_snapshot(self) -> None:
        """Open an undo log for the event about to run."""
        self._snapshot = InstanceSnapshot.capture(self)
        self._recorder = self._snapshot.log

    def undo_last_event(self) -> None:
        """Reverse the last event, restoring request, instance and pool state."""
        assert self._snapshot is not None, "No snapshot to undo."
        self._snapshot.restore(self)
        self._snapshot = None
        self._recorder = NULL_RECORDER

    def _commit_event(self) -> None:
        """Keep the last event's mutations and close its log."""
        self._snapshot = None
        self._recorder = NULL_RECORDER
```

Pass `self._recorder` at every `self.kv` call site introduced in Task 6, for example:

```python
            self.kv.allocate(req, n_tokens=chunk, hit_blocks=[], recorder=self._recorder)
```

and in `_retire`, `_release` and `_preempt_one`:

```python
            self.kv.free(r, recorder=self._recorder)
```

The two places that mutate `waiting` in place must assign a fresh list instead, so the captured list object still describes the pre-event queue. In `_promote_ready`:

```python
    def _promote_ready(self) -> None:
        promoted = []
        for req in self.waiting:
            if not self._can_run_more() or req.prompt_len + req.generated - req.prefilled > 0:
                continue
            promoted.append(req)
        if not promoted:
            return
        self.waiting = [r for r in self.waiting if r not in promoted]
        self.running = [*self.running, *promoted]
        for req in promoted:
            req.status = ReqStatus.RUNNING
```

and in `_preempt_one`, replace `self.waiting.insert(0, victim)` with:

```python
        self.running = [r for r in self.running if r is not victim]
        self.waiting = [victim, *self.waiting]
```

removing the earlier `self.running.remove(victim)`. Do the same for the `self.waiting.remove(req)` / `self.running.append(req)` pairs in `_prefill_budget_step` and `_retire`: build a new list rather than mutating.

Finally, `advance_to` commits as it goes, so it must leave `_recorder` alone (it is already `NULL_RECORDER` there) and must not open a snapshot. It currently does not, so no change is needed; the test `test_committed_advancement_is_not_undoable` pins that.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_kv_instance.py -q && pytest tests/ -q
```

Expected: `tests/test_kv_instance.py` 6 passed, whole suite green. A failure in `test_sim_telemetry.py` or `test_e2e.py` here means a queue mutation still happens in place: find the `self.waiting.` or `self.running.` call that mutates rather than reassigns.

```

---

### Task 8: Apply hits, and split `abort_all` into free plus reset

The payload. A hit pre-sets `prefilled`, so the saved tokens leave `t2` and `n_tokens` by themselves and the cost model needs no change.

**Files:**
- Modify: `dryrun/component/rollout/engine/instance.py`
- Modify: `dryrun/component/rollout/engine/state.py`
- Modify: `dryrun/component/rollout/engine/request.py`
- Test: `tests/test_kv_instance.py`

**Interfaces:**
- Consumes: Tasks 5-7.
- Produces: `StepStat` gains `n_prefill_hit: int = 0` and `n_prefill_computed: int = 0`. `Request` gains `prefix_hit_tokens: int = 0`. `NativeInstance` gains `_apply_prefix_hit(req: Request) -> int` returning tokens saved.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kv_instance.py`:

```python
class TestWithinGroupPromptReuse:
    def test_siblings_pay_for_the_shared_prompt_once(self):
        """The limitation in docs/reference/limitations.md: 'When a prompt has n
        sampled responses, the simulator charges prefill n times.' At
        prompt_len=150 with block_size=16 the shared part is 9 full blocks over
        144 tokens, so seven of eight siblings skip 144 tokens each."""
        inst = _inst()
        for rid in range(8):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=99), t=0.0)
        for _ in range(500):
            if inst.advance_one_event(version=0) is None:
                break

        assert len(inst.completed) == 8
        saved = sum(r.prefix_hit_tokens for r in inst.completed)
        assert saved == 7 * 144

    def test_prompt_prefill_drops_to_the_predicted_token_count(self):
        inst = _inst()
        for rid in range(8):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=99), t=0.0)
        for _ in range(500):
            if inst.advance_one_event(version=0) is None:
                break

        prompt_work = sum(s.n_prefill_computed for s in inst.step_stats)
        assert prompt_work == 150 + 7 * 6, "150 cold, then six private tokens each"

    def test_different_prompts_share_nothing(self):
        inst = _inst()
        for rid in range(8):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=rid), t=0.0)
        for _ in range(500):
            if inst.advance_one_event(version=0) is None:
                break
        assert sum(r.prefix_hit_tokens for r in inst.completed) == 0

    def test_disabling_prefix_caching_restores_the_old_cost(self):
        inst = _inst(enable_prefix_caching=False)
        for rid in range(8):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=99), t=0.0)
        for _ in range(500):
            if inst.advance_one_event(version=0) is None:
                break
        assert sum(r.prefix_hit_tokens for r in inst.completed) == 0
        assert sum(s.n_prefill_computed for s in inst.step_stats) == 8 * 150


class TestWeightSyncClearsHashes:
    def test_abort_all_frees_blocks_and_clears_the_hash_map(self):
        """PSRL's pause_for_sync aborts every request, then pull_model calls
        reset_prefix_cache. Both steps, in that order."""
        inst = _inst()
        for rid in range(4):
            assert inst.add_request(_req(rid=rid, prompt_len=160, source_group_id=5), t=0.0)
        for _ in range(50):
            inst.advance_one_event(version=0)

        inst.abort_all()
        assert inst.blocks_used() == 0
        assert inst.kv.pool.cached_block(("p", 5, 0)) is None

    def test_a_resumed_group_still_hits_on_the_shared_prompt(self):
        """The post-sync hit rate is not zero: the first sibling repopulates
        the prompt hashes and the rest hit them. This is the mechanism behind
        the 9% measured in outputs/simulate/SMG_LOOPBACK_FIX.md."""
        inst = _inst(kv_blocks=4000)
        for rid in range(4):
            assert inst.add_request(
                _req(rid=rid, prompt_len=160, source_group_id=5, target_len=512), t=0.0
            )
        for _ in range(200):
            inst.advance_one_event(version=0)
        inst.abort_all()
        saved_before = sum(r.prefix_hit_tokens for r in inst.waiting)

        for _ in range(200):
            if inst.advance_one_event(version=1) is None:
                break
        saved_after = sum(
            r.prefix_hit_tokens for r in (*inst.waiting, *inst.running, *inst.completed)
        )
        assert saved_after > saved_before, "siblings hit the repopulated prompt blocks"

    def test_an_agentic_round_boundary_keeps_hashes(self):
        """`cancel(keep_tokens=True)` is the agentic round boundary: free the
        blocks, keep the hashes, so the next round hits. This is the interface
        the agentic spec builds on, and the reason free and reset are separate."""
        inst = _inst()
        req = _req(rid=1, prompt_len=160, source_group_id=5)
        assert inst.add_request(req, t=0.0)
        for _ in range(50):
            inst.advance_one_event(version=0)

        inst.cancel([req], keep_tokens=True)
        assert inst.blocks_used() == 0
        assert inst.kv.pool.cached_block(("p", 5, 0)) is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_kv_instance.py -q
```

Expected: FAIL, `AttributeError: 'Request' object has no attribute 'prefix_hit_tokens'`.

- [ ] **Step 3: Write minimal implementation**

In `dryrun/component/rollout/engine/request.py`, add after `reprefill_tokens: int = 0`:

```python
    prefix_hit_tokens: int = 0
    """Prompt tokens this request skipped because the KV cache already held them.

    Accumulates across resumptions, so a request interrupted by a weight update
    and re-prefilled records what the cache saved each time.
    """
```

In `dryrun/component/rollout/engine/state.py`, add to `StepStat` after `n_prefill: int`:

```python
    n_prefill_hit: int = 0
    n_prefill_computed: int = 0
```

and extend its docstring:

```python
    `n_prefill` is the work this step charged. `n_prefill_hit` is what the
    prefix cache absorbed and `n_prefill_computed` is what was really computed,
    named to match the real engine's own columns (`prefill_tokens` and
    `computed (cache miss)` in `outputs/simulate/SMG_LOOPBACK_FIX.md`) so
    `dryrun-compare` puts like against like.
```

In `dryrun/component/rollout/engine/snapshot.py`, add `prefix_hit_tokens` to `ReqSnapshot`: add the field

```python
    prefix_hit_tokens: int
```

to `capture`

```python
            prefix_hit_tokens=req.prefix_hit_tokens,
```

and to `restore`

```python
        req.prefix_hit_tokens = self.prefix_hit_tokens
```

In `dryrun/component/rollout/engine/instance.py`, add the hit application method after `free_blocks`:

```python
    def _apply_prefix_hit(self, req: Request) -> int:
        """Advance `req.prefilled` over whatever the cache already holds.

        A hit is expressed by moving `prefilled` rather than by a separate
        `num_computed_tokens` field, because both prefill paths already compute
        `prompt_len + generated - prefilled` as the work remaining. The saved
        tokens therefore drop out of `n_tokens` and `t2` on their own, and the
        cost model needs no change at all.

        Args:
            req: A request about to be prefilled, with `prefilled == 0`.

        Returns:
            Tokens the cache covered.
        """
        if not self.enable_prefix_caching or req.prefilled > 0:
            return 0
        hit_blocks, hit_tokens = self.kv.computed_blocks(req, recorder=self._recorder)
        if hit_tokens <= 0:
            return 0
        if not self.kv.can_allocate(req, n_tokens=1, hit_blocks=hit_blocks):
            return 0
        self.kv.allocate(req, n_tokens=0, hit_blocks=hit_blocks, recorder=self._recorder)
        req.prefilled = hit_tokens
        req.prefix_hit_tokens += hit_tokens
        return hit_tokens
```

In `_prefill_budget_step`, apply the hit before measuring the work. Replace

```python
        free = self.free_blocks()
        for req in sorted(self.waiting, key=lambda r: (r.dispatch_time, r.rid)):
            if budget <= 0:
                break
            if req.prefilled == 0:
                if self.max_concurrency is not None and n_in_batch >= self.max_concurrency:
                    break
                n_in_batch += 1
            need = req.prompt_len + req.generated - req.prefilled
```

with

```python
        free = self.free_blocks()
        prefill_hit = 0
        for req in sorted(self.waiting, key=lambda r: (r.dispatch_time, r.rid)):
            if budget <= 0:
                break
            if req.prefilled == 0:
                if self.max_concurrency is not None and n_in_batch >= self.max_concurrency:
                    break
                n_in_batch += 1
                prefill_hit += self._apply_prefix_hit(req)
                free = self.free_blocks()
            need = req.prompt_len + req.generated - req.prefilled
```

and record it on the `StepStat`. Replace

```python
            StepStat(
                n_tokens,
                n_decode,
                prefill_tokens,
                self.cost.saturated(n_tokens, t2, ctxsum, n_reqs),
```

with

```python
            StepStat(
                n_tokens,
                n_decode,
                prefill_tokens,
                n_prefill_hit=prefill_hit,
                n_prefill_computed=prefill_tokens,
                saturated=self.cost.saturated(n_tokens, t2, ctxsum, n_reqs),
```

Note that this turns the fourth positional argument into a keyword, so check the remaining arguments in that call still line up: `steps=1`, `dt=dt`, `phase="prefill"`, `t=self.t_local + dt` are already keywords.

Apply the same hit in `_step_time_estimate` so the time estimate matches what the step will do. This path predicts a step without performing it, so it must **not** move the counters that `vllm:prefix_cache_queries` is compared against. Add a non-counting lookup to `KvCacheManager` in `dryrun/component/rollout/engine/kv/manager.py`:

```python
    def peek_hit_tokens(self, req: "Request") -> int:
        """Tokens a hit would cover, without recording a query.

        `_step_time_estimate` predicts a step without performing it, so it must
        not move the counters that `vllm:prefix_cache_queries` is compared
        against.
        """
        if not self.enable_caching:
            return 0
        hashes = block_hashes(req, self.block_size)
        n_hit = 0
        for block_hash in hashes:
            if self.pool.cached_block(block_hash) is None:
                break
            n_hit += 1
        tokens = n_hit * self.block_size
        if n_hit and tokens >= req.prompt_len + req.generated:
            tokens -= self.block_size
        return tokens
```

Then in `_step_time_estimate`, replace

```python
            eff = effective.get(id(req), req.prefilled)
            need = req.prompt_len + req.generated - eff
```

with:

```python
            eff = effective.get(id(req), req.prefilled)
            if eff == 0:
                eff = self.kv.peek_hit_tokens(req)
                effective[id(req)] = eff
            need = req.prompt_len + req.generated - eff
```

Finally split `abort_all`. Replace its body (keeping the existing docstring and adding to it):

```python
        aborted = list(self.running) + list(self.waiting)
        for r in aborted:
            self._release(r, keep_tokens=True)
            r.status = ReqStatus.WAITING
        self.waiting = aborted[:]
        self.running = []
        # `_release` already freed each request's blocks. Clearing the hash map
        # is the second, separate step: it is what `pull_model` ->
        # `reset_prefix_cache` does, and it is deliberately *not* done at an
        # agentic round boundary, where `cancel(keep_tokens=True)` frees blocks
        # but leaves their hashes addressable so the next round hits.
        self.kv.reset_hashes(recorder=self._recorder)
        return aborted
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_kv_instance.py -q && pytest tests/ -q
```

Expected: `tests/test_kv_instance.py` 13 passed, whole suite green. If `test_e2e.py` shifts timings, that is the intended effect of prefix caching; confirm the direction is *less* prefill work, never more, and update the expected values in that test with a note naming this commit.

---

### Task 9: Config knob and telemetry

**Files:**
- Modify: `dryrun/config/schema.py:236-247`
- Modify: `dryrun/config/defaults/simulate.yaml`
- Modify: `dryrun/simulator/config.py`
- Modify: `dryrun/simulator/simulator.py:68-84`
- Modify: `dryrun/cli/main.py:388-422`
- Modify: `dryrun/telemetry/writer.py:105-120`
- Test: `tests/test_kv_config.py`

**Interfaces:**
- Consumes: Task 8.
- Produces: `RolloutConfig.enable_prefix_caching: bool = True`; `SimConfig.enable_prefix_caching: bool = True`; the `rollout_step` telemetry stream gains `n_prefill_hit` and `n_prefill_computed`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kv_config.py`:

```python
"""Tests that the prefix-caching knob reaches the engine and the telemetry."""

from __future__ import annotations

from dryrun.config.schema import RolloutConfig
from dryrun.simulator.config import SimConfig
from dryrun.workload.grouped import GroupedWorkload


def _workload() -> GroupedWorkload:
    return GroupedWorkload.from_lengths([150] * 8, [64] * 8)


class TestPrefixCachingKnob:
    def test_default_matches_the_simulated_engine_default(self):
        """Real vLLM ships with prefix caching on, so the default should be on:
        a default that does not match the simulated system is a silent bias."""
        assert RolloutConfig().enable_prefix_caching is True
        assert SimConfig(workload=_workload()).enable_prefix_caching is True

    def test_the_knob_reaches_every_instance(self):
        from dryrun.cost.rollout import PSRLFitted
        from dryrun.simulator.simulator import Simulator

        cfg = SimConfig(workload=_workload(), n_instances=2, enable_prefix_caching=False)
        sim = Simulator(cfg, rollout_cost=PSRLFitted())
        assert all(not inst.enable_prefix_caching for inst in sim.instances)
        assert all(not inst.kv.enable_caching for inst in sim.instances)


class TestTelemetryColumns:
    def test_step_stats_carry_hit_and_computed_tokens(self):
        from dryrun.component.rollout.engine import StepStat

        stat = StepStat(
            n_tokens=10,
            n_decode=4,
            n_prefill=6,
            saturated=False,
            n_prefill_hit=144,
            n_prefill_computed=6,
        )
        assert stat.n_prefill_hit == 144
        assert stat.n_prefill_computed == 6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_kv_config.py -q
```

Expected: FAIL, `AttributeError: 'RolloutConfig' object has no attribute 'enable_prefix_caching'`.

- [ ] **Step 3: Write minimal implementation**

In `dryrun/config/schema.py`, add to `RolloutConfig` after `block_size: int = 16`:

```python
    enable_prefix_caching: bool = True
    """Whether the engine reuses KV blocks across requests sharing a prefix.

    Defaults to True because real vLLM does: a default that disagrees with the
    simulated system is a silent bias. Turning it off leaves the pool doing
    block accounting and simply records no hashes, so nothing can hit; it is a
    data state rather than a second code path.
    """
```

In `dryrun/config/defaults/simulate.yaml`, under the `rollout:` block alongside `block_size`, add:

```yaml
  enable_prefix_caching: true
```

In `dryrun/simulator/config.py`, add to `SimConfig` after `block_size: int = 16`:

```python
    enable_prefix_caching: bool = True
```

In `dryrun/simulator/simulator.py`, add to the `NativeInstance(...)` construction after `reject_if_running_full=cfg.reject_if_running_full`:

```python
                    enable_prefix_caching=cfg.enable_prefix_caching,
```

In `dryrun/cli/main.py`, add to the `SimConfig(...)` call after `block_size=rollout.block_size`:

```python
        enable_prefix_caching=rollout.get("enable_prefix_caching", True),
```

In `dryrun/telemetry/writer.py`, add to the dict built around line 110, after `n_prefill=stat.n_prefill`:

```python
                n_prefill_hit=stat.n_prefill_hit,
                n_prefill_computed=stat.n_prefill_computed,
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_kv_config.py -q && pytest tests/ -q
```

Expected: `tests/test_kv_config.py` 3 passed, whole suite green. `tests/test_sim_telemetry.py` and `tests/test_telemetry.py` may assert on the `rollout_step` column set; add the two new columns there if so.

---

### Task 10: `/metrics` reader for vLLM's prefix-cache counters

**Files:**
- Create: `dryrun/workload/prefix_probe.py`
- Test: `tests/test_prefix_probe.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `parse_prefix_cache_counters(metrics_text: str) -> PrefixCacheCounters` where `PrefixCacheCounters` is a frozen dataclass with `queries: int`, `hits: int`, and a `hit_rate: float` property; and `read_prefix_cache_counters(base_url: str) -> PrefixCacheCounters`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prefix_probe.py`:

```python
"""Tests for reading vLLM's prefix-cache counters out of /metrics."""

from __future__ import annotations

import pytest

from dryrun.workload.prefix_probe import PrefixCacheCounters, parse_prefix_cache_counters

_SAMPLE = """\
# HELP vllm:prefix_cache_queries GPU prefix cache queries, in terms of number of queried blocks.
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries{engine="0",model_name="Qwen3-8B"} 1440.0
# HELP vllm:prefix_cache_hits GPU prefix cache hits, in terms of number of cached blocks.
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits{engine="0",model_name="Qwen3-8B"} 1008.0
"""


class TestParsing:
    def test_reads_both_counters(self):
        counters = parse_prefix_cache_counters(_SAMPLE)
        assert counters.queries == 1440
        assert counters.hits == 1008

    def test_hit_rate_is_blocks_over_blocks(self):
        """vLLM counts both in blocks, which is the pool's own unit, so no
        conversion is needed anywhere."""
        assert parse_prefix_cache_counters(_SAMPLE).hit_rate == pytest.approx(0.7)

    def test_sums_across_engines(self):
        text = (
            'vllm:prefix_cache_queries{engine="0"} 100.0\n'
            'vllm:prefix_cache_queries{engine="1"} 50.0\n'
            'vllm:prefix_cache_hits{engine="0"} 40.0\n'
            'vllm:prefix_cache_hits{engine="1"} 20.0\n'
        )
        counters = parse_prefix_cache_counters(text)
        assert counters.queries == 150
        assert counters.hits == 60

    def test_ignores_the_external_connector_counters(self):
        """`vllm:external_prefix_cache_*` are a different mechanism and must not
        be folded into the GPU-side numbers."""
        text = (
            'vllm:prefix_cache_queries{engine="0"} 100.0\n'
            'vllm:prefix_cache_hits{engine="0"} 40.0\n'
            'vllm:external_prefix_cache_queries{engine="0"} 999.0\n'
            'vllm:external_prefix_cache_hits{engine="0"} 999.0\n'
        )
        counters = parse_prefix_cache_counters(text)
        assert counters.queries == 100
        assert counters.hits == 40

    def test_absent_counters_raise_rather_than_reporting_zero(self):
        """A server without these counters means prefix caching was never on,
        and a silent 0.0 would read as 'measured no hits'."""
        with pytest.raises(ValueError, match="prefix_cache_queries"):
            parse_prefix_cache_counters("# nothing here\n")

    def test_zero_queries_has_no_hit_rate(self):
        counters = PrefixCacheCounters(queries=0, hits=0)
        assert counters.hit_rate == 0.0

    def test_more_hits_than_queries_is_rejected(self):
        with pytest.raises(ValueError, match="hits"):
            PrefixCacheCounters(queries=5, hits=6)

    def test_counters_subtract_for_a_windowed_reading(self):
        """The counters are cumulative, so a scenario measures a delta."""
        before = PrefixCacheCounters(queries=100, hits=40)
        after = PrefixCacheCounters(queries=250, hits=130)
        delta = after - before
        assert delta.queries == 150
        assert delta.hits == 90
        assert delta.hit_rate == pytest.approx(0.6)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_prefix_probe.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'dryrun.workload.prefix_probe'`.

- [ ] **Step 3: Write minimal implementation**

Create `dryrun/workload/prefix_probe.py`:

```python
"""Read vLLM's prefix-cache counters, for validating the simulated block pool.

`vllm:prefix_cache_queries` and `vllm:prefix_cache_hits`
(`third_party/vllm/vllm/v1/metrics/loggers.py:584-601`) are Prometheus counters
measured in **blocks**, which is the simulated pool's own unit. So a measured
hit rate and a simulated one compare directly, with no token conversion in
between and nothing to calibrate.

Both are cumulative since server start, so a scenario reads them before and
after and takes the difference.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass

_SAMPLE_LINE = re.compile(
    r"^vllm:(?P<name>prefix_cache_queries|prefix_cache_hits)\{[^}]*\}\s+(?P<value>[0-9.eE+-]+)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class PrefixCacheCounters:
    """Blocks queried and blocks found, summed across engines.

    Args:
        queries: Blocks looked up.
        hits: Blocks found.

    Raises:
        ValueError: If either is negative, or if hits exceed queries, which
            would mean the reading is not self-consistent.
    """

    queries: int
    hits: int

    def __post_init__(self) -> None:
        if self.queries < 0 or self.hits < 0:
            raise ValueError(f"Counters must be non-negative, got {self!r}.")
        if self.hits > self.queries:
            raise ValueError(
                f"More hits than queries ({self.hits} > {self.queries}), so this "
                f"reading is not self-consistent."
            )

    @property
    def hit_rate(self) -> float:
        """Fraction of queried blocks that were found, 0.0 when nothing was asked."""
        if self.queries == 0:
            return 0.0
        return self.hits / self.queries

    def __sub__(self, other: "PrefixCacheCounters") -> "PrefixCacheCounters":
        """The delta between two cumulative readings."""
        return PrefixCacheCounters(
            queries=self.queries - other.queries,
            hits=self.hits - other.hits,
        )


def parse_prefix_cache_counters(metrics_text: str) -> PrefixCacheCounters:
    """Extract the GPU prefix-cache counters from a Prometheus exposition.

    Args:
        metrics_text: The body of a `GET /metrics` response.

    Returns:
        The two counters, summed over every engine label.

    Raises:
        ValueError: If `vllm:prefix_cache_queries` is absent. That means the
            server never had prefix caching enabled, and reporting 0.0 would
            read as "measured no hits" rather than "measured nothing".
    """
    totals = {"prefix_cache_queries": 0, "prefix_cache_hits": 0}
    seen = set()
    for match in _SAMPLE_LINE.finditer(metrics_text):
        name = match.group("name")
        totals[name] += int(float(match.group("value")))
        seen.add(name)

    if "prefix_cache_queries" not in seen:
        raise ValueError(
            "No vllm:prefix_cache_queries sample in /metrics, so this server was "
            "not running with prefix caching enabled."
        )
    return PrefixCacheCounters(
        queries=totals["prefix_cache_queries"],
        hits=totals["prefix_cache_hits"],
    )


def read_prefix_cache_counters(base_url: str, timeout_s: float = 10.0) -> PrefixCacheCounters:
    """Fetch and parse `{base_url}/metrics`.

    Args:
        base_url: Server root, e.g. `http://127.0.0.1:8000`.
        timeout_s: Socket timeout.

    Returns:
        The counters at the moment of the call.
    """
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=timeout_s) as response:
        return parse_prefix_cache_counters(response.read().decode("utf-8"))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_prefix_probe.py -q && pytest tests/ -q
```

Expected: `tests/test_prefix_probe.py` 8 passed, whole suite green.

---

### Task 11: The four controlled vLLM scenarios

Requires a GPU. Every expected value is registered **before** measuring; nothing here is fitted. Scenario 4 runs first because LRU eviction is otherwise never executed: KV does not saturate under the current workload, so without it that path ships unexercised.

**Files:**
- Create: `examples/profile/vllm/prefix_cache_probe.yaml`
- Create: `examples/profile/vllm/prefix_cache_probe.sh`
- Test: manual, with results recorded in Task 12

**Interfaces:**
- Consumes: `read_prefix_cache_counters` (Task 10), `VllmServer` from `dryrun/workload/inference.py`.
- Produces: a measured-versus-simulated table consumed by Task 12.

- [ ] **Step 1: Write the fixture**

Create `examples/profile/vllm/prefix_cache_probe.yaml`:

```yaml
# Four controlled prefix-cache scenarios. Each row's expected hit count is
# arithmetic over prompt_len and block_size, registered before measuring.
#
# Scenario 4 is listed first deliberately: KV never saturates under the
# project's main workload, so LRU eviction is the one pool path that no other
# scenario reaches.
model: /path/to/Qwen3-8B
block_size: 16
# Never 1. A single-request point carries ~78 ms of server-side work and
# inflated prefill MRE to 46.98% once before; docs/reference/limitations.md
# records "never quote a figure from a num-prompts: 1 point" as a rule.
n: 8

scenarios:
  - name: lru_eviction
    description: Squeeze the pool until reuse must evict, the only path that reaches _maybe_evict.
    gpu_memory_utilization: 0.30
    prompt_len: 2048
    max_tokens: 256
    n_prompts: 64
    expect: hit_rate_falls_with_pressure

  - name: within_group_shared_prompt
    description: One prompt, eight samples. The shared prefix is paid for once.
    gpu_memory_utilization: 0.90
    prompt_len: 2048
    max_tokens: 64
    n_prompts: 1
    expect_shared_blocks: 128      # 2048 // 16
    expect_hit_rate: 0.875         # 7 of 8 requests hit every shared block

  - name: continued_prefix
    description: Round two's prompt is round one's prompt plus its output.
    gpu_memory_utilization: 0.90
    prompt_len: 2048
    max_tokens: 512
    n_prompts: 1
    rounds: 2
    expect_round2_hit_blocks: 160  # (2048 + 512) // 16

  - name: hit_rate_after_reset
    description: reset_prefix_cache, then resend. Hits fall to zero, then recover.
    gpu_memory_utilization: 0.90
    prompt_len: 2048
    max_tokens: 64
    n_prompts: 1
    reset_between: true
    expect_hit_rate_after_reset: 0.875
```

- [ ] **Step 2: Write the driver script**

Create `examples/profile/vllm/prefix_cache_probe.sh`:

```bash
#!/usr/bin/env bash
# Measure vLLM's prefix-cache hit rate in four controlled scenarios, to check
# the simulated block pool against the engine it models.
#
# Expected values live in prefix_cache_probe.yaml and were registered before
# measuring. Nothing here is fitted: a disagreement means the pool's semantics
# are wrong and the logic must change, not that a coefficient needs tuning.
set -euo pipefail

source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh
cd "$(git rev-parse --show-toplevel)"

MODEL="${MODEL:?set MODEL to a Qwen3-8B checkpoint directory}"
PORT="${PORT:-8000}"
OUT="${OUT:-outputs/profile/prefix_cache_probe}"
mkdir -p "$OUT"

python -m dryrun.workload.prefix_probe_run \
  --config examples/profile/vllm/prefix_cache_probe.yaml \
  --model "$MODEL" \
  --port "$PORT" \
  --output "$OUT/measured.json"

echo "Measured counters written to $OUT/measured.json"
echo "Compare against the simulator with:"
echo "  pytest tests/test_kv_instance.py -q   # same arithmetic, no GPU needed"
```

```bash
chmod +x examples/profile/vllm/prefix_cache_probe.sh
```

- [ ] **Step 3: Run scenario 4 first, then 1 through 3**

```bash
MODEL=/path/to/Qwen3-8B bash examples/profile/vllm/prefix_cache_probe.sh
```

Expected: `outputs/profile/prefix_cache_probe/measured.json` holding four entries, each with `queries`, `hits`, and `hit_rate`. Check each against the `expect_*` field in the yaml.

If a scenario disagrees with its registered expectation, **stop and diagnose the pool**. Do not adjust the expectation to match. Record the disagreement and its diagnosis in Task 12's document.

- [ ] **Step 4: Confirm the simulator reproduces the same arithmetic**

```bash
pytest tests/test_kv_instance.py tests/test_kv_pool.py -q
```

Expected: all pass. These encode the same expected hit counts without needing a GPU, so they are the regression guard once the GPU run has happened.

---

### Task 12: Acceptance gates and documentation

**Files:**
- Create: `docs/simulation/prefix-cache.md`
- Modify: `docs/reference/limitations.md`
- Modify: `docs/simulation/index.md`
- Modify: `outputs/simulate/ALIGNMENT_LOG.md`

**Interfaces:**
- Consumes: measured results from Task 11, the simulator from Tasks 1-9.
- Produces: documentation only.

- [ ] **Step 1: Re-run the aligned baseline and record the three gates**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/DryRun-RL
bash examples/simulate/psrl_baseline_aligned.sh
dryrun-compare outputs/simulate/psrl_baseline_aligned outputs/visualize/qwen3_8b_megatron_baseline
```

Record, from the comparison output:

| Gate | Before | After | Target |
|---|---|---|---|
| mean KV util | -24.1% | *measured* | toward 0 |
| re-prefill per interrupt | 8602 against 6003 (1.43x) | *measured* | toward 1.0 |
| peak KV util | +0.1% | *measured* | unchanged |
| mean running busy | -2.0% | *measured* | unchanged |
| span | +3.6% | *measured* | unchanged |

**If mean KV util has not converged toward 0, do not add a coefficient.** The spec registers this in advance: the correct response is to re-examine pool semantics or look for another mechanism. Write the finding down either way; a directional prediction that failed is a result, not an embarrassment.

- [ ] **Step 2: Write the documentation page**

Create `docs/simulation/prefix-cache.md`:

```markdown
# Prefix caching

The rollout engine reuses KV blocks between requests that share a prefix, which
is what the real vLLM does and what the simulator previously did not model at
all. Three measured consequences motivated building it; see
{doc}`../reference/limitations.md` for where each stood before.

## How a block is identified

vLLM hashes a block as `hash(parent_block_hash, block_token_ids)`, so a block
hash means *every token from position 0 to that block's end*. The chaining is
load-bearing: it is what stops `[hello][world]` and `[bye][world]` colliding on
their second block.

We hold no token ids. We hold the provenance that produced them, which answers
the same question, because two contexts agree on a prefix exactly when they
agree on the provenance of that prefix. So a block id is written rather than
computed:

| block `j` covering `[16j, 16j+16)` | id |
|---|---|
| whole block inside the shared prompt | `("p", source_group_id, j)` |
| otherwise | `("r", rid, j)` |

Only full blocks are cacheable, which is vLLM's own rule. At `prompt_len=150`
that makes blocks 0 through 8 shareable (144 tokens) and block 9 private,
because it straddles the end of the prompt.

This is where we diverge from Frontier, which must be handed a
`block_hash_ids` trace column because it replays real requests and never sees
their tokens. Generating our own workload means the trace format stays
untouched.

## What falls out

- **Within a group.** Eight responses to one prompt share `source_group_id`, so
  seven of them skip the shared prefix. At `prompt_len=150` a group drops from
  `8 x 150 = 1200` prefill tokens to `150 + 7 x 6 = 192`.
- **Partial rollout.** A resumed request's generated blocks hash exactly as they
  did when first filled, so re-prefill hits them.
- **Weight sync.** `abort_all` frees every block *and* clears the hash map, in
  that order, mirroring PSRL's `pause_for_sync` then `pull_model`. The post-sync
  hit rate is therefore not zero: the first sibling repopulates the prompt
  hashes and the rest hit them.

## Measured against the engine

Four controlled scenarios, run with `examples/profile/vllm/prefix_cache_probe.sh`.
Expected hit counts were registered in the fixture before measuring, and nothing
is fitted: `vllm:prefix_cache_queries` and `vllm:prefix_cache_hits` are counted
in blocks, which is the simulated pool's own unit, so the two compare directly.

| Scenario | Expected | Measured | Simulated |
|---|---|---|---|
| LRU eviction under pressure | hit rate falls | *fill from Task 11* | *fill* |
| Within-group shared prompt | 0.875 | *fill* | *fill* |
| Continued prefix | 160 blocks | *fill* | *fill* |
| Hit rate after reset | 0.875 | *fill* | *fill* |

## Turning it off

`rollout.enable_prefix_caching: false`. The pool still does block accounting and
simply records no hashes, so nothing can hit. It is a data state, not a second
code path, which is why the disabled case needs no separate validation.
```

Fill the table's `*fill*` cells from Task 11's `measured.json` and from
`pytest tests/test_kv_instance.py -q`.

- [ ] **Step 3: Update the limitations page**

In `docs/reference/limitations.md`, replace the **Prompt prefill charged once per response** entry under "Operational rough edges" with, under "Model approximations":

```markdown
**Prefix reuse is derived from provenance, so it under-reports hits in one
direction.** Block identity comes from `(source_group_id, block_index)` for
prompt-interior blocks and `(rid, block_index)` otherwise, rather than from a
hash of token ids the simulator never holds. Two *different* prompts that happen
to share leading full blocks hit in the real engine and miss here. On RL
workloads distinct problems almost certainly differ at token 0, and every weight
update clears the hash map anyway, so the effect is second order -- but it is
one-directional: this never over-reports a hit. A workload with a shared system
prompt or shared few-shot prefix would amplify it, and the fix at that point is
an optional explicit prefix-grouping key on the trace, not a change to the hash
rule. See {doc}`../simulation/prefix-cache`.

**Within-group reuse cannot be judged on the aligned baseline.**
`extract_workload_trace` chunks completed requests into groups of `group_size`
in request-id order, because the route trace does not record which prompt each
response belonged to. That reproduces the length distribution and group arity
exactly but not the grouping, so a hit rate measured there reflects fabricated
groups. This is why prefix caching is validated against a purpose-built vLLM
experiment instead; see {doc}`../simulation/prefix-cache`.

**No cross-instance reuse, and the router scatters groups.** The block pool is
per-instance, as in the real engine, while `_select_instance` picks the
least-active one, so a group's siblings land on different instances and
within-group reuse is diluted. Frontier makes this a hard gate: multi-replica
prefix caching requires a sticky scheduler or it raises. We do not port that
gate, because what PSRL actually does has not been verified and copying
Frontier's policy would weld in an unverified assumption. Routing is tracked
separately.
```

- [ ] **Step 4: Update the alignment log and the docs index**

In `outputs/simulate/ALIGNMENT_LOG.md`, under "Open gaps, largest first", replace item 1 with the measured outcome from Step 1, naming the commit. If mean KV util converged, say by how much; if it did not, say so and record the diagnosis.

In `docs/simulation/index.md`, add `prefix-cache` to the toctree.

- [ ] **Step 5: Verify docs build**

```bash
pytest tests/ -q
cd docs && make html 2>&1 | tail -20 && cd ..
```

Expected: suite green, docs build with no new warnings about the added page.

---

## Self-Review

**Spec coverage.** Walked each spec section against the plan.

| Spec item | Task |
|---|---|
| P1 pool under `engine/kv/` | 2, 4, 5 |
| P2 provenance-derived hashes | 3 |
| P3 hit pre-sets `prefilled` | 8 |
| P4 keep the hand-written doubly linked free list | 2 |
| P5 `abort_all` splits into free plus reset | 8 |
| P6 default on, disabling is a data state | 9 |
| P7 validate against a purpose-built experiment | 11 |
| P8 register expectations, fit nothing | 11 |
| P9 undo by touched-block old values, never a pool snapshot | 1, 2, 4, 7 |
| §3.2 pool invariant (free and hashed at once) | 4 |
| §3.3 full-hit edge case | 5 |
| §3.4 undo mechanism and the three extra state pieces | 1, 4, 7 |
| §3.5 seams, closed form untouched | 6 |
| §4 four scenarios | 11 |
| §4.1 telemetry fields | 8, 9 |
| §4.2 three gates plus no-regression set | 12 |
| §5 D1, D2, D3, D4 limitations | 12 |

**Two additions beyond the spec as first written.** Both are now in the spec
(commit `c1e557f`), recorded here because they changed the plan's shape.

Task 7 (undoable pool state) started as a plan-only addition: `undo_last_event`
is a live path via `simulator.py:358`, and pool state cannot be reconstructed
from `prefilled`, so without it an uncommitted prefill step leaves the pool and
the requests silently disagreeing. Task 1 then followed from measuring the
obvious implementation: `copy.deepcopy` of the pool costs 179 ms per event,
1771 s across a run whose real duration is 3349 s. That ruled out snapshotting
and motivated a reusable recorder rather than a bespoke fix.

**Verified before committing the plan.** The free-list and pool undo logic were
prototyped and exercised against the assertions written into Tasks 2 and 4 --
including a mixed event (hit, allocate, cache, free) restoring occupancy,
free-list order, eviction count, hash map and every block's refcount and hash.
All passed. The abstraction's own overhead measured 0.77 us for a median
two-block event, against 179 ms for the deepcopy it replaces. Both are
microbenchmarks, so Task 7's end-to-end run is what actually settles the cost.

**Placeholder scan.** The documentation table in Task 12 carries `*fill*` cells,
which are instructions to transcribe Task 11's measured output, not unspecified
work; Step 2 names the source for each. No other placeholders.

**Type consistency.** Checked the names used across tasks: `Undoable` /
`Recorder` / `UndoLog` / `NULL_RECORDER` / `AttrSnapshot` / `TruncateList` /
`MappingKeysAdded` (Task 1, consumed by 2, 4, 5, 7), `KvBlock.prev_free` /
`next_free` and `FreeBlockList.popleft` / `append` / `remove` / `ids` (Task 2,
consumed by 4), `block_hashes` (Tasks 3, 5), `BlockPool.cached_block` /
`new_blocks` / `touch` / `free_blocks` / `cache_full_blocks` / `reset_hashes` /
`free_list_ids` (Tasks 4, 5, 7), `KvCacheManager.computed_blocks` /
`can_allocate` / `allocate` / `free` / `reset_hashes` / `blocks_held` /
`peek_hit_tokens` (Tasks 5, 6, 8), `NativeInstance.kv` / `_recorder` /
`enable_prefix_caching` (Tasks 6, 7, 8, 9), `StepStat.n_prefill_hit` /
`n_prefill_computed` (Tasks 8, 9), `Request.prefix_hit_tokens` (Tasks 8, 12),
`PrefixCacheCounters.queries` / `hits` / `hit_rate` (Task 10).

Every pool and manager mutator takes a trailing `recorder`. `BlockPool`'s is
required and positional-or-keyword; `KvCacheManager`'s defaults to
`NULL_RECORDER` so tests and committed paths need not pass it. That asymmetry is
deliberate: the pool is internal and should never be called without a decision
about rollback, while the manager is the engine-facing surface.

`peek_hit_tokens` is introduced mid-Task-8 rather than in Task 5 where the rest
of the manager lives; that is deliberate, since its motivation (not perturbing
counters during a non-mutating estimate) only becomes visible once
`_step_time_estimate` needs it.
