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
