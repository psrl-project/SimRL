#!/usr/bin/env python
"""G0 gate: how accurately does `forecast` predict the buffer READY and train-finish times?

Instruments a real, small `policy=psrl` simulator run. At every router dispatch, when the training
pipeline is about to be consulted, it calls `dryrun.forecast.fleet.forecast` on the live
`SchedulingRuntime` with the **oracle** remaining lengths. It then compares each prediction against
what actually happened:

    F_k  predicted -> the time psrl's buffer k first turned READY (observed on the policy)
    E_k  predicted -> `result.train_timestamps[k][1]`, the real publish after `weight_load_time`

The gate is the design doc's phase 3: median |relative error| <= 3%, p90 <= 8% on F_k, reported for
E_k separately since it compounds the training-cost model's own error. Oracle lengths remove
prediction error as a cause, so a miss is a forecaster bug, not a length-model one.

Usage:
    python examples/analysis/forecast_accuracy.py [--configs ...] [--out /tmp/forecast_accuracy.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from psrl.workers.ps.staleness_controller import BufferStatus

from dryrun.component.sync.cost import FixedSyncCost
from dryrun.cost.rollout import UnifiedRoofline
from dryrun.cost.training import LinearTrainingCost
from dryrun.forecast import OracleLength, forecast
from dryrun.policy.psrl import PSRLPolicy
from dryrun.simulator.config import SimConfig
from dryrun.simulator.runtime import SchedulingRuntime
from dryrun.simulator.simulator import Simulator
from dryrun.workload.grouped import GroupedWorkload
from dryrun.workload.trace import RequestGroup

GATE_F_MEDIAN = 0.03
GATE_F_P90 = 0.08


@dataclass(frozen=True)
class Regime:
    """A small, deterministic psrl run exercising one scheduling regime."""

    name: str
    staleness: int
    cap: int
    weight_load: float
    n_versions: int = 14
    batch: int = 8
    group_size: int = 4
    n_instances: int = 2
    time_per_token: float = 3.0e-4
    horizon: int = 2


@dataclass
class Sample:
    """One forecast call, and the buffer ids it pinned, waiting to be scored."""

    now: float
    b0: int
    ready: dict[int, float]
    train_end: dict[int, float]
    latency: float


@dataclass
class RunResult:
    regime: Regime
    true_ready: dict[int, float]
    true_end: dict[int, float]
    samples: list[Sample] = field(default_factory=list)
    n_dispatch: int = 0
    span: float = 0.0


class _ObservingPolicy(PSRLPolicy):
    """Record the first simulated time each buffer's status flips to READY."""

    def __init__(self) -> None:
        super().__init__()
        self.true_ready: dict[int, float] = {}

    def on_group_complete(self, group, st):
        action = super().on_group_complete(group, st)
        self._scan(st)
        return action

    def _scan(self, st) -> None:
        if self._inventory is None:
            return
        for buffer_id in self._inventory.buffers:
            if buffer_id in self.true_ready:
                continue
            status = self._inventory.get_buffer_status(buffer_id)
            if status in (BufferStatus.READY, BufferStatus.READY_WITH_CAPACITY):
                self.true_ready[buffer_id] = st.now


class _ForecastingSimulator(Simulator):
    """Run the sim and, at each dispatch, forecast the pipeline from the live state."""

    def __init__(self, *args, horizon: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._runtime = SchedulingRuntime(self)
        self.horizon = horizon
        self.samples: list[Sample] = []
        self.n_dispatch = 0

    def _dispatch_router(self) -> int:
        placed = super()._dispatch_router()
        self.n_dispatch += 1
        if self.train_version < self.cfg.n_versions:
            start = time.perf_counter()
            pipeline = forecast(self._runtime, OracleLength(), horizon_buffers=self.horizon)
            latency = time.perf_counter() - start
            if not pipeline.buffers:
                return placed
            self.samples.append(
                Sample(
                    now=self.t,
                    b0=pipeline.buffers[0],
                    ready=dict(zip(pipeline.buffers, pipeline.ready)),
                    train_end=dict(zip(pipeline.buffers, pipeline.train_end)),
                    latency=latency,
                )
            )
        return placed


def _workload(regime: Regime) -> GroupedWorkload:
    """A deterministic dispersed workload: four responses per group, lengths spread within a group."""
    lens = [(400, 900, 1800, 3200), (250, 600, 1200, 2600), (800, 1500, 2400, 4000), (200, 500, 1000, 2000)]
    groups = [
        RequestGroup(group_id=i, prompt_len=200, output_lens=lens[i % len(lens)])
        for i in range(regime.batch * (regime.n_versions + regime.staleness + 3))
    ]
    return GroupedWorkload.from_groups(groups)


def _build(regime: Regime) -> _ForecastingSimulator:
    cfg = SimConfig(
        batch_size=regime.batch,
        group_size=regime.group_size,
        max_staleness=regime.staleness,
        n_versions=regime.n_versions,
        n_instances=regime.n_instances,
        weight_load_time=regime.weight_load,
        sync_time=0.5,
        max_concurrent_seqs_per_instance=regime.cap,
        workload=_workload(regime),
        log_telemetry=False,
    )
    policy = _ObservingPolicy()
    sim = _ForecastingSimulator(
        rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
        training_cost=LinearTrainingCost(regime.time_per_token),
        sync_cost=FixedSyncCost(0.5),
        policy=policy,
        cfg=cfg,
        horizon=regime.horizon,
    )
    return sim


def run_regime(regime: Regime) -> RunResult:
    sim = _build(regime)
    result = sim.run()
    true_end = {index: end for index, (_, end) in enumerate(result.train_timestamps)}
    return RunResult(
        regime=regime,
        true_ready=sim.policy.true_ready,
        true_end=true_end,
        samples=sim.samples,
        n_dispatch=sim.n_dispatch,
        span=result.sim_time,
    )


def _relative(predicted: float, actual: float) -> float | None:
    """Predicted-vs-actual error, or None when the prediction is not a finite, scorable number."""
    if not math.isfinite(predicted) or actual <= 0:
        return None
    return abs(predicted - actual) / actual


def score(run: RunResult) -> dict:
    """Compare each sample's prediction against the buffer's actual F and E, keyed by buffer id."""
    f_errors: list[float] = []
    e_errors: list[float] = []
    skipped = 0
    for sample in run.samples:
        for buffer_id in (sample.b0, sample.b0 + 1):
            if buffer_id not in run.true_ready or buffer_id not in run.true_end:
                continue
            actual_f = run.true_ready[buffer_id]
            # Only a prediction made before the buffer turned READY is a prediction, not history.
            if sample.now >= actual_f:
                skipped += 1
                continue
            f_err = _relative(sample.ready.get(buffer_id, math.inf), actual_f)
            e_err = _relative(sample.train_end.get(buffer_id, math.inf), run.true_end[buffer_id])
            if f_err is not None:
                f_errors.append(f_err)
            if e_err is not None:
                e_errors.append(e_err)
    return {
        "regime": run.regime.name,
        "n_dispatch": run.n_dispatch,
        "n_samples": len(run.samples),
        "n_scored": len(f_errors),
        "n_skipped": skipped,
        "span": run.span,
        "latency_ms_mean": statistics.mean(s.latency for s in run.samples) * 1e3 if run.samples else 0.0,
        "latency_ms_p90": _pct([s.latency for s in run.samples], 0.9) * 1e3 if run.samples else 0.0,
        "F_median": statistics.median(f_errors) if f_errors else None,
        "F_p90": _pct(f_errors, 0.9),
        "F_max": max(f_errors) if f_errors else None,
        "E_median": statistics.median(e_errors) if e_errors else None,
        "E_p90": _pct(e_errors, 0.9),
        "E_max": max(e_errors) if e_errors else None,
    }


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[index]


def default_regimes() -> list[Regime]:
    """Regimes spanning the design doc's headroom: different S, a cap at the knee, and weight-load on."""
    return [
        Regime("s1_cap128_noLoad", staleness=1, cap=128, weight_load=0.0),
        Regime("s1_cap48_noLoad", staleness=1, cap=48, weight_load=0.0),
        Regime("s1_cap24_tight", staleness=1, cap=24, weight_load=0.0),
        Regime("s2_cap128_noLoad", staleness=2, cap=128, weight_load=0.0),
        Regime("s2_cap48_noLoad", staleness=2, cap=48, weight_load=0.0),
        Regime("s1_cap48_load4", staleness=1, cap=48, weight_load=4.0),
        Regime("s2_cap48_load4", staleness=2, cap=48, weight_load=4.0),
        Regime("s1_cap48_three_inst", staleness=1, cap=48, weight_load=0.0, n_instances=3),
        Regime("s1_cap48_one_inst", staleness=1, cap=48, weight_load=0.0, n_instances=1),
        Regime("s0_cap128_noLoad", staleness=0, cap=128, weight_load=0.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/forecast_accuracy.json"))
    args = parser.parse_args()

    rows = []
    print(f"{'regime':<22}{'disp':>6}{'scored':>7}{'F_med':>8}{'F_p90':>8}{'E_med':>8}{'E_p90':>8}{'lat_ms':>8}")
    for regime in default_regimes():
        run = run_regime(regime)
        row = score(run)
        rows.append(row)
        print(
            f"{row['regime']:<22}{row['n_dispatch']:>6}{row['n_scored']:>7}"
            f"{_fmt(row['F_median']):>8}{_fmt(row['F_p90']):>8}{_fmt(row['E_median']):>8}{_fmt(row['E_p90']):>8}"
            f"{row['latency_ms_mean']:>8.2f}"
        )

    all_f = [row["F_median"] for row in rows if row["F_median"] is not None]
    passed = all(
        row["F_median"] is not None and row["F_median"] <= GATE_F_MEDIAN and row["F_p90"] <= GATE_F_P90
        for row in rows
    )
    print(
        f"\nGate: F median <= {GATE_F_MEDIAN:.0%}, p90 <= {GATE_F_P90:.0%} per regime -> "
        f"{'PASS' if passed else 'FAIL'}"
    )
    print(f"Regimes with a scored F: {len(all_f)}/{len(rows)}, total scored {sum(r['n_scored'] for r in rows)}")
    args.out.write_text(json.dumps(rows, indent=1))
    print(f"Wrote {args.out}")


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.3%}"


if __name__ == "__main__":
    main()
