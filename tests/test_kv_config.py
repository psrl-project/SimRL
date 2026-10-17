"""Tests that the prefix-caching knob reaches the engine and the telemetry."""

from __future__ import annotations

from srl.component.rollout.engine import StepStat
from srl.config.schema import RolloutConfig
from srl.cost.rollout import UnifiedRoofline
from srl.policy.areal import ArealPolicy
from srl.simulator.config import SimConfig
from srl.simulator.simulator import Simulator
from srl.workload.grouped import GroupedWorkload
from srl.workload.trace import RequestGroup


def _workload(group_size: int = 8) -> GroupedWorkload:
    """Four prompts, `group_size` responses each.

    Real groups rather than `from_lengths`, which produces one response per
    prompt and so cannot exercise within-group prefix sharing at all.
    """
    return GroupedWorkload.from_groups(
        [
            RequestGroup(group_id=gid, prompt_len=150, output_lens=tuple([64] * group_size))
            for gid in range(4)
        ]
    )


def _cost() -> UnifiedRoofline:
    return UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)


def _sim(store=None, **kwargs) -> Simulator:
    kwargs.setdefault("group_size", 8)
    cfg = SimConfig(workload=_workload(), **kwargs)
    return Simulator(
        rollout_cost=_cost(),
        policy=ArealPolicy(),
        cfg=cfg,
        store=store,
    )


class TestPrefixCachingKnob:
    def test_default_matches_the_simulated_engine_default(self):
        """Real vLLM ships with prefix caching on, so the default should be on:
        a default that does not match the simulated system is a silent bias."""
        assert RolloutConfig().enable_prefix_caching is True
        assert SimConfig(workload=_workload(), group_size=8).enable_prefix_caching is True

    def test_the_knob_reaches_every_instance(self):
        sim = _sim(n_instances=2, enable_prefix_caching=False)
        assert all(not inst.enable_prefix_caching for inst in sim.instances)
        assert all(not inst.kv.enable_caching for inst in sim.instances)

    def test_enabled_by_default_in_a_built_simulator(self):
        sim = _sim(n_instances=2)
        assert all(inst.enable_prefix_caching for inst in sim.instances)
        assert all(inst.kv.enable_caching for inst in sim.instances)


class TestTelemetryColumns:
    def test_step_stats_carry_hit_and_computed_tokens(self):
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

    def test_the_rollout_step_stream_reports_both_columns(self, tmp_path):
        """`srl-compare` lines these up against the real engine's own
        `prefill_tokens` / `computed (cache miss)` columns, so they have to
        reach the log rather than staying in memory."""
        import json

        from srl.telemetry.store import JsonlStore

        store = JsonlStore(tmp_path)
        sim = _sim(store=store, batch_size=2, n_versions=1, kv_blocks=2000)
        sim.run()
        store.close()

        rows = [
            json.loads(line)
            for line in (tmp_path / "rollout_step.jsonl").read_text().splitlines()
        ]
        assert rows, "the run produced no rollout steps"
        assert all("n_prefill_hit" in row for row in rows)
        assert all("n_prefill_computed" in row for row in rows)
        # Eight responses share one 150-token prompt, so the cache must absorb
        # the nine full prompt blocks for seven of them.
        assert sum(row["n_prefill_hit"] for row in rows) > 0
