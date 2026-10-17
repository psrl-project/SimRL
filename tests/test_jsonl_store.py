"""Tests for the general-purpose JsonlStore telemetry sink."""

from __future__ import annotations

import json
from enum import Enum
from fractions import Fraction

from srl.telemetry.store import JsonlStore


class _Color(Enum):
    RED = "red"


class TestWriteIterRoundtrip:
    def test_single_stream(self, tmp_path):
        store = JsonlStore(tmp_path)
        store.write("train", t=1.0, version=1)
        store.write("train", t=2.0, version=2)
        rows = list(store.iter("train"))
        assert rows == [{"t": 1.0, "version": 1}, {"t": 2.0, "version": 2}]

    def test_preserves_append_order(self, tmp_path):
        store = JsonlStore(tmp_path)
        for i in range(50):
            store.write("s", i=i)
        rows = list(store.iter("s"))
        assert [r["i"] for r in rows] == list(range(50))

    def test_missing_stream_yields_nothing(self, tmp_path):
        store = JsonlStore(tmp_path)
        assert list(store.iter("does_not_exist")) == []

    def test_nested_stream_name_maps_to_subdirectory(self, tmp_path):
        store = JsonlStore(tmp_path)
        store.write("instance/0", t=0.0, n=1)
        store.write("instance/1", t=0.0, n=2)
        assert (tmp_path / "instance" / "0.jsonl").exists()
        assert (tmp_path / "instance" / "1.jsonl").exists()
        assert list(store.iter("instance/0")) == [{"t": 0.0, "n": 1}]
        assert list(store.iter("instance/1")) == [{"t": 0.0, "n": 2}]

    def test_each_line_is_valid_standalone_json(self, tmp_path):
        store = JsonlStore(tmp_path)
        store.write("s", a=1)
        store.write("s", a=2)
        path = tmp_path / "s.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # must not raise


class TestJsonSerialization:
    def test_enum_serialized_as_value(self, tmp_path):
        store = JsonlStore(tmp_path)
        store.write("s", color=_Color.RED)
        assert list(store.iter("s")) == [{"color": "red"}]

    def test_fraction_serialized_as_float(self, tmp_path):
        store = JsonlStore(tmp_path)
        store.write("s", dt=Fraction(1, 4))
        rows = list(store.iter("s"))
        assert rows[0]["dt"] == 0.25

    def test_none_serialized_as_null(self, tmp_path):
        store = JsonlStore(tmp_path)
        store.write("s", complete_time=None)
        assert list(store.iter("s")) == [{"complete_time": None}]


class TestListStreams:
    def test_lists_flat_and_nested_streams(self, tmp_path):
        store = JsonlStore(tmp_path)
        store.write("train", t=0.0)
        store.write("instance/0", t=0.0)
        store.write("instance/1", t=0.0)
        assert store.list_streams() == ["instance/0", "instance/1", "train"]

    def test_empty_store_has_no_streams(self, tmp_path):
        store = JsonlStore(tmp_path)
        assert store.list_streams() == []


class TestClose:
    def test_close_then_iter_still_works(self, tmp_path):
        store = JsonlStore(tmp_path)
        store.write("s", a=1)
        store.close()
        assert list(store.iter("s")) == [{"a": 1}]

    def test_close_is_idempotent(self, tmp_path):
        store = JsonlStore(tmp_path)
        store.write("s", a=1)
        store.close()
        store.close()  # must not raise


class TestOverwrite:
    def test_default_reopen_appends(self, tmp_path):
        first = JsonlStore(tmp_path)
        first.write("train", t=1.0)
        first.close()
        second = JsonlStore(tmp_path)
        second.write("train", t=2.0)
        second.close()
        assert list(second.iter("train")) == [{"t": 1.0}, {"t": 2.0}]

    def test_overwrite_replaces_existing_streams(self, tmp_path):
        first = JsonlStore(tmp_path)
        first.write("train", t=1.0)
        first.write("instance/0", t=1.0)
        first.write("instance/1", t=1.0)
        first.close()

        second = JsonlStore(tmp_path, overwrite=True)
        second.write("train", t=99.0)
        second.write("instance/0", t=99.0)
        second.close()

        assert list(second.iter("train")) == [{"t": 99.0}]
        assert list(second.iter("instance/0")) == [{"t": 99.0}]
        assert list(second.iter("instance/1")) == []
        assert second.list_streams() == ["instance/0", "train"]

    def test_overwrite_false_on_reopen_does_not_wipe(self, tmp_path):
        first = JsonlStore(tmp_path)
        first.write("train", t=1.0)
        first.close()
        reopened = JsonlStore(tmp_path)
        assert list(reopened.iter("train")) == [{"t": 1.0}]
