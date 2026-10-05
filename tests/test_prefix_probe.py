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

    def test_reads_the_prometheus_total_suffix(self):
        """Prometheus exposes a counter registered as `x` under the sample name
        `x_total`, so the scraped name differs from the registered one. This is
        a real sample captured from a live vLLM 0.26 server; matching only the
        unsuffixed name made the reader report the counters as absent."""
        text = (
            'vllm:prefix_cache_queries_total{engine="0",model_name="Qwen3-8B"} 7445.0\n'
            'vllm:prefix_cache_queries_created{engine="0",model_name="Qwen3-8B"} 1.79e+09\n'
            'vllm:prefix_cache_hits_total{engine="0",model_name="Qwen3-8B"} 5952.0\n'
            'vllm:prefix_cache_hits_created{engine="0",model_name="Qwen3-8B"} 1.79e+09\n'
        )
        counters = parse_prefix_cache_counters(text)
        assert counters.queries == 7445
        assert counters.hits == 5952
        assert counters.hit_rate == pytest.approx(0.7995, abs=1e-4)

    def test_the_created_timestamp_sample_is_not_summed_in(self):
        """`_created` holds a unix timestamp. Folding it into the count would
        produce a nonsense total that still looks plausible."""
        text = (
            'vllm:prefix_cache_queries_total{engine="0"} 100.0\n'
            'vllm:prefix_cache_queries_created{engine="0"} 1791014117.2\n'
            'vllm:prefix_cache_hits_total{engine="0"} 40.0\n'
            'vllm:prefix_cache_hits_created{engine="0"} 1791014117.2\n'
        )
        counters = parse_prefix_cache_counters(text)
        assert counters.queries == 100
        assert counters.hits == 40
