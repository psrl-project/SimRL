"""Tests for length distribution generators."""

from __future__ import annotations

from dryrun.workload.distributions import fixed, from_trace, normal


class TestNormal:
    def test_returns_n_samples(self):
        lengths = normal(20, mu=100.0, sigma=10.0, lo=1, hi=200, seed=0)
        assert len(lengths) == 20
        assert all(isinstance(length, int) for length in lengths)

    def test_clips_to_bounds(self):
        lengths = normal(50, mu=0.0, sigma=1.0, lo=8, hi=12, seed=1)
        assert all(8 <= length <= 12 for length in lengths)

    def test_seed_is_reproducible(self):
        assert normal(10, mu=64.0, sigma=8.0, hi=128, seed=7) == normal(
            10, mu=64.0, sigma=8.0, hi=128, seed=7
        )


class TestFixed:
    def test_returns_n_copies(self):
        assert fixed(5, 512) == [512] * 5

    def test_seed_is_accepted_but_ignored(self):
        assert fixed(3, 10, seed=99) == [10, 10, 10]


class TestFromTrace:
    def test_skips_metadata_lines(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        path.write_text('{"_meta": true, "stats": {}}\n{"length": 10}\n{"length": 20}\n')
        assert from_trace(str(path)) == [10, 20]

    def test_custom_column(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        path.write_text('{"response_length": 5}\n{"response_length": 15}\n')
        assert from_trace(str(path), column="response_length") == [5, 15]

    def test_limit(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        path.write_text('{"length": 1}\n{"length": 2}\n{"length": 3}\n')
        assert from_trace(str(path), limit=2) == [1, 2]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        path.write_text('{"length": 1}\n\n{"length": 2}\n')
        assert from_trace(str(path)) == [1, 2]

    def test_parquet(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = tmp_path / "trace.parquet"
        pq.write_table(pa.table({"length": [7, 8, 9]}), path)
        assert from_trace(str(path)) == [7, 8, 9]

    def test_parquet_respects_limit(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = tmp_path / "trace.parquet"
        pq.write_table(pa.table({"length": [1, 2, 3, 4]}), path)
        assert from_trace(str(path), limit=2) == [1, 2]
