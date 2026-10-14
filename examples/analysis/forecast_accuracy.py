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

`--scale medium` reruns the same gate at the headroom-study scale (composed cost model, KV ~500k
tokens/instance, B=32 groups x 8) instead of the toy roofline regimes, since a scheduler built on
`forecast()` runs at that scale, not the toy one.

Usage:
    python examples/analysis/forecast_accuracy.py [--configs ...] [--out /tmp/forecast_accuracy.json]
    python examples/analysis/forecast_accuracy.py --scale medium [--out-medium ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from psrl.workers.ps.staleness_controller import BufferStatus

import dryrun.forecast.drain as drain_mod
from dryrun.component.sync.cost import FixedSyncCost
from dryrun.cost.rollout import UnifiedRoofline
from dryrun.cost.training import FixedTrainingCost, LinearTrainingCost
from dryrun.forecast import OracleLength, evaluate_placement, forecast
from dryrun.forecast.fleet import _queued_admitted, baseline_fleet, estimate_queued, members_of
from dryrun.policy.psrl import PSRLPolicy
from dryrun.simulator.config import SimConfig
from dryrun.simulator.runtime import SchedulingRuntime
from dryrun.simulator.simulator import Simulator
from dryrun.workload.grouped import GroupedWorkload
from dryrun.workload.trace import RequestGroup

ROOT = Path(__file__).resolve().parents[2]

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
    strategy: str = "request_num_balance"
    strategy_args: dict = field(default_factory=dict)


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
        routing_strategy=regime.strategy,
        routing_strategy_args=regime.strategy_args,
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


def _toy_regimes_for(strategy: str) -> list[Regime]:
    """`default_regimes()`, or the same axes re-run under `inventory_deadline`.

    A static `cap` is the knob `inventory_deadline` replaces, so forcing it to 0 (disabled) is what
    running that strategy "on its own terms" means: whatever it does is an effect of its own model,
    not of also being capped.
    """
    if strategy == "request_num_balance":
        return default_regimes()

    return [dataclasses.replace(r, strategy=strategy, cap=0) for r in default_regimes()]


def _run_toy(args: argparse.Namespace) -> None:
    rows = []
    print(f"{'regime':<22}{'disp':>6}{'scored':>7}{'F_med':>8}{'F_p90':>8}{'E_med':>8}{'E_p90':>8}{'lat_ms':>8}")
    for regime in _toy_regimes_for(args.strategy):
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


# --- medium scale: the headroom-study regime (composed model, KV ~500k tokens/instance) ---

MEDIUM_KERNEL_ROOT = ROOT / "outputs" / "profile" / "qwen3_8b" / "kernel_tp1" / "kernel"
MEDIUM_MODEL_PATH = "/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B"
MEDIUM_KV_TOKENS_PER_INSTANCE = 500_192
MEDIUM_BLOCK_SIZE = 16
MEDIUM_TOKEN_BUDGET = 12_800
MEDIUM_SAMPLE_STRIDE = 20
"""Every dispatch beyond this cadence is skipped, so a bounded-cap run with thousands of dispatches
still scores at most a couple hundred samples."""
MEDIUM_SWEEP_NS = (0, 10, 25, 50, 100, None)
"""Queue truncation levels for the latency-vs-queue-length table. `None` keeps the full queue."""

_replay_count = [0]
_original_replay = drain_mod._replay


def _counting_replay(*args, **kwargs):
    _replay_count[0] += 1
    return _original_replay(*args, **kwargs)


def _load_dispersed_trace_module():
    spec = importlib.util.spec_from_file_location(
        "make_dispersed_trace", ROOT / "examples" / "workload" / "make_dispersed_trace.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class MediumRegime:
    """One headroom-study-scale regime: composed model, 2 instances, B=32 groups x 8 responses."""

    name: str
    cv: float
    train_s: float
    staleness: int
    cap: int
    """Router's static per-instance concurrency cap. 0 disables it."""
    weight_load: float
    mean: int = 4000
    rho: float = 0.6
    length_cap: int = 16384
    n_versions: int = 8
    batch: int = 32
    group_size: int = 8
    n_instances: int = 2
    strategy: str = "request_num_balance"
    strategy_args: dict = field(default_factory=dict)


def medium_regimes() -> list[MediumRegime]:
    """Eight regimes spanning the design doc's axes without the full 32-way cross product."""
    return [
        MediumRegime("base", cv=0.5, train_s=150.0, staleness=1, cap=0, weight_load=0.0),
        MediumRegime("disperse", cv=1.0, train_s=150.0, staleness=1, cap=0, weight_load=0.0),
        MediumRegime("balanced_tau", cv=0.5, train_s=345.0, staleness=1, cap=0, weight_load=0.0),
        MediumRegime("staleness2", cv=0.5, train_s=150.0, staleness=2, cap=0, weight_load=0.0),
        MediumRegime("cap_knee", cv=0.5, train_s=150.0, staleness=1, cap=48, weight_load=0.0),
        MediumRegime("weight_load", cv=0.5, train_s=150.0, staleness=1, cap=0, weight_load=5.0),
        MediumRegime("combo_tight", cv=0.5, train_s=345.0, staleness=2, cap=48, weight_load=5.0),
        MediumRegime("combo_disperse", cv=1.0, train_s=345.0, staleness=2, cap=48, weight_load=0.0),
    ]


@dataclass
class MediumSample:
    """One sampled forecast call, plus everything needed to bucket its error by cause."""

    now: float
    version: int
    b0: int | None
    ready: dict
    train_end: dict
    latency: float
    qlen_total: int
    qlen_reserved: int
    overflow: bool
    cum_preemptions: int


@dataclass
class QueueSweepPoint:
    version: int
    qlen_full: int
    n: int | None
    latency: float
    replay_delta: int


@dataclass
class MediumRunResult:
    regime: MediumRegime
    true_ready: dict = field(default_factory=dict)
    true_end: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)
    sweep: list = field(default_factory=list)
    placement_eval_ms: float | None = None
    n_dispatch: int = 0
    span: float = 0.0
    total_preemptions: int = 0
    failed: str | None = None


def _total_preemptions(sim: Simulator) -> int:
    """Sum `num_preemptions` over every request the simulator currently knows about."""
    total = sum(r.num_preemptions for r in sim.consumed)
    total += sum(r.num_preemptions for r in sim.inflight)
    for group in sim.pending.values():
        total += sum(r.num_preemptions for r in group)
    for group in sim._staged_groups.values():
        total += sum(r.num_preemptions for r in group)
    return total


class _TruncatedQueueRuntime:
    """Wraps a `SchedulingRuntime`, keeping only the first `limit` queued members.

    `forecast()` only calls `queued()` for the router queue, so overriding it and delegating
    everything else reproduces the fleet forecast at a smaller queue without touching the engine.
    """

    def __init__(self, runtime: SchedulingRuntime, limit: int | None) -> None:
        self._runtime = runtime
        self._limit = limit

    def queued(self) -> dict:
        full = self._runtime.queued()
        if self._limit is None:
            return full
        out: dict = {}
        count = 0
        for group_id, reqs in full.items():
            if count >= self._limit:
                break
            out[group_id] = reqs
            count += len(reqs)
        return out

    def __getattr__(self, name):
        return getattr(self._runtime, name)


class _MediumObservingPolicy(PSRLPolicy):
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


class _MediumSimulator(Simulator):
    """Runs the sim and, at sampled dispatches, forecasts the pipeline and its KV overflow flag."""

    def __init__(self, *args, horizon: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._runtime = SchedulingRuntime(self)
        self.horizon = horizon
        self.samples: list[MediumSample] = []
        self.sweep: list[QueueSweepPoint] = []
        self.placement_eval_ms: float | None = None
        self.n_dispatch = 0
        self._last_version = -1

    def _dispatch_router(self) -> int:
        """Sample before dispatching on a publish, since `_publish_weights` enqueues the aborted
        set and the new batch, then dispatches once: by the time `super()` returns, that backlog
        is already placed as far as the cap allows, which is the state a scheduler would decide
        from."""
        after_publish = self.train_version != self._last_version
        in_horizon = self.train_version < self.cfg.n_versions
        if after_publish and in_horizon:
            self._sample(do_sweep=True)
        placed = super()._dispatch_router()
        self.n_dispatch += 1
        if not after_publish and in_horizon and self.n_dispatch % MEDIUM_SAMPLE_STRIDE == 0:
            self._sample(do_sweep=False)
        self._last_version = self.train_version
        return placed

    def _sample(self, *, do_sweep: bool) -> None:
        runtime = self._runtime
        queued = runtime.queued()
        qlen_total = sum(len(v) for v in queued.values())
        members = members_of(runtime)
        reserved_queued = _queued_admitted(runtime, members)
        qlen_reserved = sum(len(v) for v in reserved_queued.values())

        t0 = time.perf_counter()
        pipeline = forecast(runtime, OracleLength(), horizon_buffers=self.horizon)
        latency = time.perf_counter() - t0
        overflow = self._forecast_overflow(runtime, members)

        if pipeline.buffers:
            self.samples.append(
                MediumSample(
                    now=self.t,
                    version=self.train_version,
                    b0=pipeline.buffers[0],
                    ready=dict(zip(pipeline.buffers, pipeline.ready)),
                    train_end=dict(zip(pipeline.buffers, pipeline.train_end)),
                    latency=latency,
                    qlen_total=qlen_total,
                    qlen_reserved=qlen_reserved,
                    overflow=overflow,
                    cum_preemptions=_total_preemptions(self),
                )
            )

        if do_sweep:
            self._sweep_queue_length(runtime, qlen_reserved)
            if self.placement_eval_ms is None:
                self._time_evaluate_placement(runtime, members)

    def _forecast_overflow(self, runtime: SchedulingRuntime, members: dict) -> bool:
        """Re-derive the fleet `forecast()` builds internally, to read each instance's overflow flag."""
        remaining_of = OracleLength().remaining
        fleet = baseline_fleet(runtime, remaining_of)
        queued = _queued_admitted(runtime, members)
        snapshot = runtime.inventory()
        fleet, _ = estimate_queued(fleet, queued, snapshot.deadline, remaining_of, now=runtime.now)
        return any(fc.finish_times().kv_overflow for fc in fleet.instances.values())

    def _sweep_queue_length(self, runtime: SchedulingRuntime, qlen_full: int) -> None:
        for n in MEDIUM_SWEEP_NS:
            wrapped = _TruncatedQueueRuntime(runtime, n)
            before = _replay_count[0]
            t0 = time.perf_counter()
            forecast(wrapped, OracleLength(), horizon_buffers=self.horizon)
            latency = time.perf_counter() - t0
            self.sweep.append(
                QueueSweepPoint(
                    version=self.train_version,
                    qlen_full=qlen_full,
                    n=n,
                    latency=latency,
                    replay_delta=_replay_count[0] - before,
                )
            )

    def _time_evaluate_placement(self, runtime: SchedulingRuntime, members: dict) -> None:
        member = next(
            (req for reqs in runtime.queued().values() for req in reqs if req.group_id in members), None
        )
        if member is None:
            return
        instance_ids = [inst.instance_id for inst in runtime.instances]
        options = [*instance_ids[:2], None]
        t0 = time.perf_counter()
        evaluate_placement(runtime, OracleLength(), member, options, horizon_buffers=self.horizon)
        self.placement_eval_ms = (time.perf_counter() - t0) * 1e3


def _medium_workload(regime: MediumRegime, mdt_module) -> GroupedWorkload:
    n_groups = regime.batch * (regime.n_versions + regime.staleness + 3)
    rng = np.random.default_rng(0)
    lengths = mdt_module.draw(n_groups, regime.group_size, regime.mean, regime.cv, regime.rho, regime.length_cap, rng)
    prompts = np.clip(np.rint(rng.lognormal(np.log(140), 0.3, n_groups)), 16, 1024).astype(np.int64)
    groups = [
        RequestGroup(group_id=i, prompt_len=int(prompts[i]), output_lens=[int(x) for x in lengths[i]])
        for i in range(n_groups)
    ]
    return GroupedWorkload.from_groups(groups)


def _build_medium(regime: MediumRegime, cost_model, mdt_module) -> _MediumSimulator:
    kv_blocks = MEDIUM_KV_TOKENS_PER_INSTANCE // MEDIUM_BLOCK_SIZE
    cfg = SimConfig(
        batch_size=regime.batch,
        group_size=regime.group_size,
        max_staleness=regime.staleness,
        n_versions=regime.n_versions,
        n_instances=regime.n_instances,
        weight_load_time=regime.weight_load,
        sync_time=0.5,
        max_concurrent_seqs_per_instance=regime.cap,
        workload=_medium_workload(regime, mdt_module),
        log_telemetry=False,
        token_budget=MEDIUM_TOKEN_BUDGET,
        kv_blocks=kv_blocks,
        block_size=MEDIUM_BLOCK_SIZE,
        max_prompt_length=1024,
        max_output_length=regime.length_cap,
        routing_strategy=regime.strategy,
        routing_strategy_args=regime.strategy_args,
    )
    return _MediumSimulator(
        rollout_cost=cost_model,
        training_cost=FixedTrainingCost(regime.train_s),
        sync_cost=FixedSyncCost(0.5),
        policy=_MediumObservingPolicy(),
        cfg=cfg,
        backend="native",
    )


def _composed_cost_model():
    from dryrun.cost.kernel.database import KernelDatabase
    from dryrun.cost.kernel.model_ops import ModelOps
    from dryrun.cost.rollout.composed import ComposedRollout

    database = KernelDatabase.from_root(MEDIUM_KERNEL_ROOT, gpu="h20", utility_surface="pchip")
    ops = ModelOps.from_checkpoint(MEDIUM_MODEL_PATH, tensor_parallel=1)
    return ComposedRollout(database, ops, tensor_parallel=1)


def run_medium_regime(regime: MediumRegime, cost_model, mdt_module) -> MediumRunResult:
    sim = _build_medium(regime, cost_model, mdt_module)
    try:
        result = sim.run()
    except Exception as exc:  # noqa: BLE001
        return MediumRunResult(regime=regime, failed=f"{type(exc).__name__}: {exc}")
    true_end = {index: end for index, (_, end) in enumerate(result.train_timestamps)}
    return MediumRunResult(
        regime=regime,
        true_ready=sim.policy.true_ready,
        true_end=true_end,
        samples=sim.samples,
        sweep=sim.sweep,
        placement_eval_ms=sim.placement_eval_ms,
        n_dispatch=sim.n_dispatch,
        span=result.sim_time,
        total_preemptions=_total_preemptions(sim),
    )


def score_medium(run: MediumRunResult) -> dict:
    f_errors: list[float] = []
    e_errors: list[float] = []
    skipped = 0
    buckets: dict[str, list[float]] = {"overflow": [], "no_overflow": [], "preempt": [], "no_preempt": []}
    sorted_samples = sorted(run.samples, key=lambda s: s.now)
    for sample in run.samples:
        if sample.b0 is None:
            continue
        for offset, buffer_id in enumerate((sample.b0, sample.b0 + 1)):
            if buffer_id not in run.true_ready or buffer_id not in run.true_end:
                continue
            actual_f = run.true_ready[buffer_id]
            if sample.now >= actual_f:
                skipped += 1
                continue
            f_err = _relative(sample.ready.get(buffer_id, math.inf), actual_f)
            e_err = _relative(sample.train_end.get(buffer_id, math.inf), run.true_end[buffer_id])
            if f_err is not None:
                f_errors.append(f_err)
                if offset == 0:
                    buckets["overflow" if sample.overflow else "no_overflow"].append(f_err)
                    later = next((s for s in sorted_samples if s.now >= actual_f), None)
                    preempted = later is not None and later.cum_preemptions > sample.cum_preemptions
                    buckets["preempt" if preempted else "no_preempt"].append(f_err)
            if e_err is not None:
                e_errors.append(e_err)
    latencies = [s.latency for s in run.samples]
    qlens = [s.qlen_reserved for s in run.samples]
    return {
        "regime": run.regime.name,
        "failed": run.failed,
        "n_dispatch": run.n_dispatch,
        "n_samples": len(run.samples),
        "n_scored": len(f_errors),
        "n_skipped": skipped,
        "span": run.span,
        "total_preemptions": run.total_preemptions,
        "overflow_frac": statistics.mean(1.0 if s.overflow else 0.0 for s in run.samples) if run.samples else None,
        "placement_eval_ms": run.placement_eval_ms,
        "latency_ms_mean": statistics.mean(latencies) * 1e3 if latencies else 0.0,
        "latency_ms_p90": _pct(latencies, 0.9) * 1e3 if latencies else 0.0,
        "latency_ms_max": max(latencies) * 1e3 if latencies else 0.0,
        "qlen_mean": statistics.mean(qlens) if qlens else 0.0,
        "qlen_max": max(qlens) if qlens else 0,
        "F_median": statistics.median(f_errors) if f_errors else None,
        "F_p90": _pct(f_errors, 0.9),
        "F_max": max(f_errors) if f_errors else None,
        "E_median": statistics.median(e_errors) if e_errors else None,
        "E_p90": _pct(e_errors, 0.9),
        "bucket_medians": {k: (statistics.median(v) if v else None) for k, v in buckets.items()},
        "bucket_counts": {k: len(v) for k, v in buckets.items()},
    }


def _sweep_table(runs: list[MediumRunResult]) -> list[dict]:
    """Pool every regime's queue-truncation sweep into one latency/replay-count-vs-N table."""
    by_n: dict[int, list[QueueSweepPoint]] = {}
    for run in runs:
        for point in run.sweep:
            key = point.n if point.n is not None else point.qlen_full
            by_n.setdefault(key, []).append(point)
    rows = []
    for key in sorted(by_n):
        points = by_n[key]
        rows.append(
            {
                "n": key,
                "n_points": len(points),
                "mean_qlen_full": statistics.mean(p.qlen_full for p in points),
                "latency_ms_mean": statistics.mean(p.latency for p in points) * 1e3,
                "replay_mean": statistics.mean(p.replay_delta for p in points),
            }
        )
    return rows


def _medium_regimes_for(strategy: str) -> list[MediumRegime]:
    """`medium_regimes()`, or the same axes re-run under `inventory_deadline` with its cap forced
    off -- see `_toy_regimes_for`'s docstring for why that is what running it "on its own terms"
    means.
    """
    if strategy == "request_num_balance":
        return medium_regimes()

    return [dataclasses.replace(r, strategy=strategy, cap=0) for r in medium_regimes()]


def _run_medium(args: argparse.Namespace) -> None:
    mdt_module = _load_dispersed_trace_module()
    cost_model = _composed_cost_model()
    drain_mod._replay = _counting_replay

    rows = []
    runs = []
    print(
        f"{'regime':<16}{'disp':>6}{'samp':>6}{'scored':>7}{'F_med':>8}{'F_p90':>8}"
        f"{'E_med':>8}{'overfl%':>8}{'preempt':>8}{'lat_ms':>8}"
    )
    for regime in _medium_regimes_for(args.strategy):
        run = run_medium_regime(regime, cost_model, mdt_module)
        runs.append(run)
        if run.failed:
            print(f"{regime.name:<16} FAILED: {run.failed}")
            rows.append({"regime": regime.name, "failed": run.failed})
            continue
        row = score_medium(run)
        rows.append(row)
        print(
            f"{row['regime']:<16}{row['n_dispatch']:>6}{row['n_samples']:>6}{row['n_scored']:>7}"
            f"{_fmt(row['F_median']):>8}{_fmt(row['F_p90']):>8}{_fmt(row['E_median']):>8}"
            f"{100 * row['overflow_frac']:>7.1f}%{row['total_preemptions']:>8}"
            f"{row['latency_ms_mean']:>8.2f}"
        )

    scored = [row for row in rows if not row.get("failed") and row["F_median"] is not None]
    passed = len(scored) > 0 and all(
        row["F_median"] <= GATE_F_MEDIAN and row["F_p90"] <= GATE_F_P90 for row in scored
    )
    print(
        f"\nGate (medium scale): F median <= {GATE_F_MEDIAN:.0%}, p90 <= {GATE_F_P90:.0%} per regime -> "
        f"{'PASS' if passed else 'FAIL'}"
    )
    print(f"Regimes scored: {len(scored)}/{len(rows)}")

    sweep = _sweep_table(runs)
    print(f"\n{'N (or full qlen)':>16}{'n_points':>10}{'mean_qlen_full':>16}{'lat_ms':>10}{'replays':>10}")
    for point in sweep:
        print(
            f"{point['n']!s:>16}{point['n_points']:>10}{point['mean_qlen_full']:>16.1f}"
            f"{point['latency_ms_mean']:>10.2f}{point['replay_mean']:>10.1f}"
        )

    payload = {"regimes": rows, "sweep": sweep}
    args.out_medium.write_text(json.dumps(payload, indent=1))
    print(f"\nWrote {args.out_medium}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["toy", "medium"], default="toy")
    parser.add_argument(
        "--strategy",
        choices=["request_num_balance", "inventory_deadline"],
        default="request_num_balance",
        help="Which strategy drives placement, for the self-consistency check.",
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/forecast_accuracy.json"))
    parser.add_argument(
        "--out-medium", type=Path, default=ROOT / "outputs" / "analysis" / "forecast_accuracy_medium.json"
    )
    args = parser.parse_args()
    args.out_medium.parent.mkdir(parents=True, exist_ok=True)
    if args.scale == "toy":
        _run_toy(args)
    else:
        _run_medium(args)


if __name__ == "__main__":
    main()
