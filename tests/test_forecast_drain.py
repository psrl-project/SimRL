"""The drain forecast against the engine it mirrors, on both backends."""

from __future__ import annotations

import dataclasses
import math
import random
import statistics
from pathlib import Path

import pytest

from dryrun.component.rollout.engine import NativeInstance, Request, TrajView, traj_view
from dryrun.cost.rollout import UnifiedRoofline
from dryrun.forecast import (
    Arrival,
    InstanceForecast,
    InstanceState,
    Interrupt,
    OracleLength,
    Resident,
)

try:
    from dryrun._engine_cpp import CppInstance, CppRequest, PyWrappedCostModel

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = ROOT / "outputs/profile/qwen3_8b/kernel_tp1/kernel"
MODEL = Path("/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B")

BLOCK = 16
BUDGET = 8192
BIG_POOL = 1_000_000
EXACT = 1e-9
# Decode-only iterations behind a queue are priced from 16-step samples, which costs up to this much
# where a model leaves its floor inside one interval. It is rare, so only the worst case may reach it.
STRETCH = 2e-5
PRICING_PATH = 5e-4
LATTICE_ERROR = 2e-2

BACKENDS = ["native", pytest.param("cpp", marks=pytest.mark.skipif(not HAS_CPP, reason="C++ engine not compiled"))]


@pytest.fixture(scope="module", params=["composed", "roofline"])
def cost(request):
    if request.param == "roofline":
        return UnifiedRoofline(F=0.006, W=0.002, G=4e-6, A_p=3e-10, A_d=5e-8, b=16)
    if not (KERNEL_ROOT.is_dir() and MODEL.is_dir()):
        pytest.skip("Needs the profiled Qwen3-8B kernel tables and checkpoint.")
    from dryrun.cost.kernel.database import KernelDatabase
    from dryrun.cost.kernel.model_ops import ModelOps
    from dryrun.cost.rollout.composed import ComposedRollout

    database = KernelDatabase.from_root(str(KERNEL_ROOT), gpu="h20", utility_surface="pchip")
    return ComposedRollout(database, ModelOps.from_checkpoint(str(MODEL), tensor_parallel=1), tensor_parallel=1)


class TrackedInstance(NativeInstance):
    """A `NativeInstance` that records the pool's peak usage and whether it ever refused a decode block."""

    peak_blocks = 0
    refused = False

    def _retire(self, reqs):
        self.peak_blocks = max(self.peak_blocks, self.blocks_used())
        return super()._retire(reqs)

    def _decode_running_once(self, version):
        stepped, blocked = super()._decode_running_once(version)
        self.refused = self.refused or blocked
        return stepped, blocked


class Engine:
    """One engine instance of either backend with the driving helpers these tests share."""

    def __init__(
        self,
        backend: str,
        cost,
        *,
        kv_blocks: int = BIG_POOL,
        max_concurrency: int | None = None,
        chunk_cap: int | None = None,
        prefix_caching: bool = False,
    ) -> None:
        self.backend = backend
        self.cost = cost
        self.kv_blocks = kv_blocks
        self.max_concurrency = max_concurrency
        self.chunk_cap = chunk_cap
        self.requests: list = []
        if backend == "native":
            self.inst = TrackedInstance(
                cost,
                token_budget=BUDGET,
                kv_blocks=kv_blocks,
                block_size=BLOCK,
                max_concurrency=max_concurrency,
                chunk_cap=chunk_cap,
                enable_prefix_caching=prefix_caching,
            )
        else:
            self.inst = CppInstance(
                PyWrappedCostModel(cost),
                token_budget=BUDGET,
                kv_blocks=kv_blocks,
                block_size=BLOCK,
                max_concurrency=-1 if max_concurrency is None else max_concurrency,
                chunk_cap=-1 if chunk_cap is None else chunk_cap,
                enable_prefix_caching=prefix_caching,
            )

    def request(self, rid: int, prompt_len: int, target_len: int, generated: int = 0):
        if self.backend == "native":
            req = Request(
                rid=rid, prompt_len=prompt_len, target_len=target_len, v_traj=0, group_id=rid, source_group_id=rid
            )
        else:
            req = CppRequest(
                rid=rid, prompt_len=prompt_len, target_len=target_len, v_traj=0, group_id=rid, source_group_id=rid
            )
        req.generated = generated
        self.requests.append(req)
        return req

    def add(self, rid: int, prompt_len: int, target_len: int, t: float | None = None, generated: int = 0) -> None:
        self.inst.add_request(
            self.request(rid, prompt_len, target_len, generated), self.inst.t_local if t is None else t
        )

    @property
    def n_waiting(self) -> int:
        return len(self.inst.waiting) if self.backend == "native" else self.inst.n_waiting

    @property
    def n_stats(self) -> int:
        return len(self.inst.step_stats) if self.backend == "native" else self.inst.n_step_stats

    def last_phase(self) -> str:
        if self.backend == "native":
            return self.inst.step_stats[-1].phase
        return self.inst.step_stats_since(self.n_stats - 1)[-1].phase

    def state(self, remaining=None) -> InstanceState:
        return InstanceState.from_instance(
            self.inst,
            cost=self.cost,
            token_budget=BUDGET,
            max_concurrency=self.max_concurrency,
            remaining=remaining or OracleLength().remaining,
            chunk_cap=self.chunk_cap,
        )

    def sample(self) -> None:
        if self.backend == "native":
            self.inst.peak_blocks = max(self.inst.peak_blocks, self.inst.blocks_used())

    @property
    def observed_peak(self) -> int:
        return self.inst.peak_blocks

    @property
    def refused_a_block(self) -> bool | None:
        """Whether a decode step was refused a block, which the C++ instance does not report."""
        return self.inst.refused if self.backend == "native" else None

    def settle(self) -> None:
        while self.n_waiting:
            self.inst.advance_one_event_committed(0)
            self.sample()

    def drain(self, max_events: int = 300_000) -> str:
        """Run to completion. Returns `done`, or `preempted` or `stalled` when the pool cannot fund the batch."""
        before = self.inst.n_preemptions
        progress = self.progress()
        for _ in range(max_events):
            if not self.inst.n_active:
                return "done"
            event = self.inst.advance_one_event_committed(0)
            self.sample()
            if self.inst.n_preemptions > before:
                return "preempted"
            if event is not None and not event.completed:
                now = self.progress()
                if now == progress:
                    return "stalled"
                progress = now
        return "stalled"

    def progress(self) -> int:
        return sum(r.prefilled + r.generated for r in self.requests)

    def cut(self, t: float) -> None:
        """
        Bring the instance to `t` the way the simulator's engine loop does.

        Events ending by `t` are committed. The first one past it is undone and the clock is
        aligned through `advance_to`.
        """
        inst = self.inst
        while inst.t_local < t:
            event = inst.advance_one_event(0)
            if event is None:
                inst.t_local = t
                return
            if event.t > t:
                inst.undo_last_event()
                self.advance_to(t)
                return
            inst.commit_last_event()
            self.sample()

    def advance_to(self, t: float) -> None:
        """Advance to `t` through `advance_to` alone, whose iterations are priced by its own estimate."""
        inst = self.inst
        while inst.t_local < t:
            _, reached = inst.advance_to(t, 0)
            if reached:
                return
            if not inst.n_active:
                inst.t_local = t
                return


def assert_cases_close(case_worst: list[float]) -> None:
    """Every case within `STRETCH`, and the typical case exact, so that a bias cannot hide in the tolerance."""
    assert max(case_worst) <= STRETCH, f"max {max(case_worst)}."
    assert statistics.median(case_worst) <= EXACT, f"median {statistics.median(case_worst)}."


def assert_overflow_agrees(result, engine: Engine, outcome: str) -> None:
    """
    Check the flag against what the engine did.

    Preemption or a stall must be flagged. A decode step refused a block is the same fault even when
    a completion in that step frees room before anything is preempted, which only the native
    instance reports. So on native the flag is also required to be absent when the pool never
    refused anything.
    """
    if outcome != "done":
        assert result.kv_overflow, f"The engine {outcome} but the forecast raised no flag."
    elif engine.refused_a_block is not None:
        assert result.kv_overflow == engine.refused_a_block, (
            f"Flag {result.kv_overflow}, engine refused a block: {engine.refused_a_block}."
        )


def resident(key: int, ctx: int, prefilled: int, remaining: int, running: bool, prompt_len: int = 200) -> Resident:
    return Resident(
        key=key,
        group_id=key,
        prompt_key=key,
        prompt_len=min(prompt_len, ctx),
        ctx=ctx,
        prefilled=prefilled,
        remaining=remaining,
        running=running,
    )


def bare_state(cost, **overrides) -> InstanceState:
    fields = dict(
        cost=cost,
        block_size=BLOCK,
        kv_blocks=BIG_POOL,
        token_budget=BUDGET,
        max_concurrency=None,
        chunk_cap=None,
        tp=1,
        now=0.0,
        waiting=(),
        running=(),
    )
    return InstanceState(**{**fields, **overrides})


def relative_errors(forecast, engine: Engine, skip=()) -> list[float]:
    errors = []
    for req in engine.requests:
        if req.rid in forecast.finish_times and req.rid not in skip:
            truth = req.complete_time
            errors.append(abs(forecast.finish_times[req.rid] - truth) / truth)
    return errors


def random_specs(rng: random.Random, n: int, prompt=(100, 30000), target=(1, 5000)) -> list[tuple[int, int]]:
    return [(rng.randint(*prompt), rng.randint(*target)) for _ in range(n)]


def apply_interrupt(engine: Engine, ix: Interrupt) -> None:
    """Drive the real engine through one weight publish, mirroring `_Drain._apply_interrupt`.

    `abort_all` returns running then waiting, in the order `Router.enqueue` would have received them.
    `Router.dispatch` re-places a layer in `rid` order regardless of that push order
    (`SortKey.SMALL_ID`), so the re-add here uses `rid` order rather than the list `abort_all` handed back.
    """
    engine.cut(ix.time)
    aborted = engine.inst.abort_all(requeue=False)
    engine.inst.t_local = ix.time + ix.resume_delay
    for req in sorted(aborted, key=lambda r: r.rid):
        engine.inst.add_request(req, engine.inst.t_local)


class TestPureDecode:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_every_finish_time_matches_the_engine(self, backend, cost):
        errors = []
        for seed in range(12):
            rng = random.Random(seed)
            specs = random_specs(rng, rng.randint(1, 64))
            engine = Engine(backend, cost)
            for rid, (prompt, target) in enumerate(specs):
                engine.add(rid, prompt, target, 0.0)
            engine.settle()
            result = InstanceForecast(engine.state()).finish_times()
            engine.drain()
            errors += relative_errors(result, engine)
        assert max(errors) <= EXACT, f"max {max(errors)}, {len(errors)} requests."

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_requests_finishing_together_share_a_time(self, backend, cost):
        engine = Engine(backend, cost)
        for rid, (prompt, target) in enumerate([(300, 500), (900, 500), (50, 500), (4000, 120), (800, 120)]):
            engine.add(rid, prompt, target, 0.0)
        engine.settle()
        result = InstanceForecast(engine.state()).finish_times()
        engine.drain()
        assert result.finish_times[0] == result.finish_times[1] == result.finish_times[2]
        assert result.finish_times[3] == result.finish_times[4]
        assert max(relative_errors(result, engine)) <= EXACT


class TestArrivals:
    @staticmethod
    def scenario(backend, cost, seed, *, max_concurrency=None, n_arrivals=2):
        rng = random.Random(seed)
        n = rng.randint(8, 40)
        cap = n if max_concurrency == "full" else max_concurrency
        engine = Engine(backend, cost, max_concurrency=cap)
        for rid, (prompt, target) in enumerate(random_specs(rng, n, prompt=(100, 6000), target=(500, 4000))):
            engine.add(rid, prompt, target, 0.0)
        engine.settle()
        state = engine.state()
        arrivals = []
        for j in range(n_arrivals):
            arrivals.append(
                Arrival(
                    key=1000 + j,
                    arrive_time=state.now + rng.uniform(0.5, 30.0),
                    prompt_len=rng.randint(100, 4000),
                    generated=rng.choice([0, 0, rng.randint(1, 2000)]),
                    remaining=rng.randint(100, 3000),
                )
            )
        arrivals.sort(key=lambda a: a.arrive_time)
        return engine, state, tuple(arrivals)

    @staticmethod
    def inject(engine, arrivals, advance):
        for a in arrivals:
            advance(a.arrive_time)
            engine.add(a.key, a.prompt_len, a.generated + a.remaining, a.arrive_time, generated=a.generated)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_arrivals_into_a_running_batch(self, backend, cost):
        errors = []
        for seed in range(8):
            engine, state, arrivals = self.scenario(backend, cost, seed)
            result = InstanceForecast(state, arrivals).finish_times()
            self.inject(engine, arrivals, engine.cut)
            engine.drain()
            errors += relative_errors(result, engine)
        assert max(errors) <= EXACT, f"max {max(errors)}, {len(errors)} requests."

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_arrivals_when_the_batch_is_full(self, backend, cost):
        worst = []
        for seed in range(8):
            engine, state, arrivals = self.scenario(backend, cost, 100 + seed, max_concurrency="full")
            assert len(state.running) == state.max_concurrency
            result = InstanceForecast(state, arrivals).finish_times()
            self.inject(engine, arrivals, engine.cut)
            engine.drain()
            worst.append(max(relative_errors(result, engine)))
        assert_cases_close(worst)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_an_arrival_inside_a_prefill_iteration_restarts_it(self, backend, cost):
        """The engine drops the iteration in flight when the clock is aligned to an arrival, and runs it again."""
        rng = random.Random(30)
        engine = Engine(backend, cost)
        for rid, (prompt, target) in enumerate(random_specs(rng, 12, prompt=(100, 3000), target=(500, 3000))):
            engine.add(rid, prompt, target, 0.0)
        engine.settle()
        state = engine.state()
        first = Arrival(key=900, arrive_time=state.now + 1.0, prompt_len=9000, generated=0, remaining=300)
        second = Arrival(key=901, arrive_time=first.arrive_time + 0.004, prompt_len=1500, generated=0, remaining=200)
        result = InstanceForecast(state, (first, second)).finish_times()
        apart = InstanceForecast(state, (first, dataclasses.replace(second, arrive_time=first.arrive_time + 5.0)))
        assert result.finish_times[901] != apart.finish_times().finish_times[901]
        self.inject(engine, (first, second), engine.cut)
        engine.drain()
        assert max(relative_errors(result, engine)) <= STRETCH

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_the_only_deviation_from_advance_to_is_its_iteration_pricing(self, backend, cost):
        """`advance_to` prices a decode pass at the context before it, as `_step_time_estimate` does."""
        errors = []
        for seed in range(8):
            engine, state, arrivals = self.scenario(backend, cost, 200 + seed, max_concurrency="full")
            result = InstanceForecast(state, arrivals).finish_times()
            self.inject(engine, arrivals, engine.advance_to)
            engine.drain()
            errors += relative_errors(result, engine)
        assert max(errors) <= PRICING_PATH, f"max {max(errors)}, {len(errors)} requests."

    def test_an_arrival_into_an_empty_instance_waits_for_its_time(self, cost):
        state = bare_state(cost, now=3.0)
        arrival = Arrival(key=1, arrive_time=7.5, prompt_len=2000, generated=0, remaining=300)
        result = InstanceForecast(state, (arrival,)).finish_times()
        assert result.finish_times[1] > 7.5
        assert result.peak_blocks == math.ceil(2300 / BLOCK)
        assert not result.kv_overflow

    def test_an_arrival_dated_before_now_is_queued_at_now(self, cost):
        state = bare_state(cost, now=9.0)
        early = InstanceForecast(state, (Arrival(1, 2.0, 2000, 0, 300),)).finish_times()
        at_now = InstanceForecast(state, (Arrival(1, 9.0, 2000, 0, 300),)).finish_times()
        assert early.finish_times == at_now.finish_times


class TestPrefillBacklog:
    @staticmethod
    def case(backend, cost, rng):
        n = rng.randint(10, 60)
        cap = rng.choice([None, None, 16, 48])
        chunk = rng.choice([None, None, 2048])
        engine = Engine(backend, cost, max_concurrency=cap, chunk_cap=chunk)
        for rid, (prompt, target) in enumerate(random_specs(rng, n, prompt=(100, 12000), target=(200, 6000))):
            engine.add(rid, prompt, target, 0.0)
        engine.cut(rng.uniform(0.0, 30.0))
        if rng.random() < 0.7:
            engine.inst.abort_all(requeue=True)
        for j in range(rng.randint(0, 4)):
            engine.add(500 + j, rng.randint(100, 8000), rng.randint(100, 3000))
        return engine

    @pytest.mark.parametrize("backend, n_cases", [("native", 12), ("cpp", 100)])
    def test_the_forecast_after_an_abort_matches_the_engine(self, backend, n_cases, cost):
        if backend == "cpp" and not HAS_CPP:
            pytest.skip("C++ engine not compiled")
        rng = random.Random(7)
        worst = []
        for _ in range(n_cases):
            engine = self.case(backend, cost, rng)
            result = InstanceForecast(engine.state()).finish_times()
            assert not result.kv_overflow
            engine.drain()
            errors = relative_errors(result, engine)
            worst.append(max(errors))
        assert_cases_close(worst)


class TestPrefixSharing:
    """GRPO siblings share their prompt blocks, so a sibling's prefill skips what an earlier one computed."""

    @staticmethod
    def grouped_engine(backend, cost, rng, *, max_concurrency=None, first_prompt=None):
        engine = Engine(backend, cost, prefix_caching=True, max_concurrency=max_concurrency)
        prompts = {}
        rid = 0
        for group in range(rng.randint(2, 6)):
            prompts[group] = rng.choice([rng.randint(100, 3000), BLOCK * rng.randint(10, 190)])
            if group == 0 and first_prompt is not None:
                prompts[group] = first_prompt
            for _ in range(8):
                req = engine.request(rid, prompts[group], rng.randint(500, 10000))
                req.source_group_id = group
                req.group_id = group
                engine.inst.add_request(req, 0.0)
                rid += 1
        return engine, prompts

    @pytest.mark.parametrize("backend, n_cases", [("native", 6), ("cpp", 40)])
    @pytest.mark.parametrize("abort", [True, False])
    def test_sibling_backlogs_match_the_engine(self, backend, n_cases, abort, cost):
        if backend == "cpp" and not HAS_CPP:
            pytest.skip("C++ engine not compiled")
        rng = random.Random(21)
        worst = []
        for _ in range(n_cases):
            engine, _ = self.grouped_engine(backend, cost, rng, max_concurrency=rng.choice([None, 32]))
            engine.cut(rng.uniform(2.0, 30.0))
            if abort:
                engine.inst.abort_all(requeue=True)
            result = InstanceForecast(engine.state()).finish_times()
            assert not result.kv_overflow
            engine.drain()
            worst.append(max(relative_errors(result, engine)))
        assert_cases_close(worst)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_a_sibling_arrival_hits_the_cached_prompt(self, backend, cost):
        rng = random.Random(22)
        engine, prompts = self.grouped_engine(backend, cost, rng, first_prompt=2000)
        group = 0
        engine.settle()
        state = engine.state()
        arrival = Arrival(
            key=999,
            arrive_time=state.now + 3.0,
            prompt_len=prompts[group],
            generated=0,
            remaining=400,
            prompt_key=group,
        )
        stranger = dataclasses.replace(arrival, prompt_key=None)
        sibling = InstanceForecast(state, (arrival,)).finish_times()
        alone = InstanceForecast(state, (stranger,)).finish_times()
        engine.cut(arrival.arrive_time)
        req = engine.request(999, prompts[group], 400)
        req.source_group_id = group
        engine.inst.add_request(req, arrival.arrive_time)
        engine.drain()
        assert max(relative_errors(sibling, engine)) <= STRETCH
        assert alone.finish_times[999] > sibling.finish_times[999]

    def test_the_kv_peak_never_undercounts_a_shared_prompt(self, cost):
        rng = random.Random(23)
        for _ in range(4):
            engine, _ = self.grouped_engine("native", cost, rng)
            engine.cut(rng.uniform(2.0, 20.0))
            engine.inst.abort_all(requeue=True)
            result = InstanceForecast(engine.state()).finish_times()
            engine.drain()
            assert result.peak_blocks >= engine.observed_peak


class TestKvPool:
    @pytest.mark.parametrize("n_requests", [1, 7, 30])
    def test_the_predicted_peak_equals_the_pools_own(self, cost, n_requests):
        rng = random.Random(n_requests)
        engine = Engine("native", cost)
        for rid, (prompt, target) in enumerate(random_specs(rng, n_requests, prompt=(100, 12000), target=(1, 3000))):
            engine.add(rid, prompt, target, 0.0)
        result = InstanceForecast(engine.state()).finish_times()
        engine.drain()
        assert abs(result.peak_blocks - engine.observed_peak) <= n_requests
        assert result.peak_blocks == engine.observed_peak

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_a_pool_exactly_the_size_of_the_peak_does_not_overflow(self, backend, cost):
        specs = [(160, 200)] * 4
        peak = 4 * math.ceil(360 / BLOCK)
        engine = Engine(backend, cost, kv_blocks=peak)
        for rid, (prompt, target) in enumerate(specs):
            engine.add(rid, prompt, target, 0.0)
        result = InstanceForecast(engine.state()).finish_times()
        assert result.peak_blocks == peak
        assert not result.kv_overflow
        assert engine.drain() == "done"
        assert max(relative_errors(result, engine)) <= EXACT

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_a_pool_one_block_short_of_the_peak_overflows(self, backend, cost):
        peak = 4 * math.ceil(360 / BLOCK)
        engine = Engine(backend, cost, kv_blocks=peak - 1)
        for rid in range(4):
            engine.add(rid, 160, 200, 0.0)
        result = InstanceForecast(engine.state()).finish_times()
        assert result.peak_blocks == peak
        assert result.kv_overflow
        assert engine.drain() == "preempted"

    @pytest.mark.parametrize("backend, n_cases", [("native", 15), ("cpp", 80)])
    def test_overflow_is_flagged_exactly_when_the_engine_cannot_fund_the_batch(self, backend, n_cases, cost):
        if backend == "cpp" and not HAS_CPP:
            pytest.skip("C++ engine not compiled")
        rng = random.Random(11)
        outcomes = {"done": 0, "stuck": 0}
        worst = []
        for _ in range(n_cases):
            n = rng.randint(4, 40 if backend == "cpp" else 14)
            specs = random_specs(rng, n, prompt=(100, 12000), target=(200, 6000))
            floor = max(math.ceil((p + t) / BLOCK) for p, t in specs)
            cap = rng.choice([None, 8, 32])
            unbounded = Engine(backend, cost, max_concurrency=cap)
            for rid, (prompt, target) in enumerate(specs):
                unbounded.add(rid, prompt, target, 0.0)
            peak = InstanceForecast(unbounded.state()).finish_times().peak_blocks
            engine = Engine(
                backend, cost, kv_blocks=max(floor, int(peak * rng.uniform(0.9, 1.1))), max_concurrency=cap
            )
            for rid, (prompt, target) in enumerate(specs):
                engine.add(rid, prompt, target, 0.0)
            result = InstanceForecast(engine.state()).finish_times()
            outcome = engine.drain(max_events=100_000)
            assert_overflow_agrees(result, engine, outcome)
            outcomes["done" if outcome == "done" else "stuck"] += 1
            if not result.kv_overflow:
                worst.append(max(relative_errors(result, engine)))
        assert outcomes["done"] >= n_cases // 5 and outcomes["stuck"] >= n_cases // 5, f"Unbalanced cases {outcomes}."
        assert_cases_close(worst)

    @pytest.mark.parametrize("backend, n_cases", [("native", 10), ("cpp", 250)])
    def test_admission_into_a_pool_a_few_requests_wide_matches_the_engine(self, backend, n_cases, cost):
        if backend == "cpp" and not HAS_CPP:
            pytest.skip("C++ engine not compiled")
        rng = random.Random(12)
        outcomes = {"done": 0, "stuck": 0}
        worst = []
        for _ in range(n_cases):
            specs = random_specs(rng, rng.randint(6, 24), prompt=(500, 14000), target=(1, 40))
            floor = max(math.ceil((p + t) / BLOCK) for p, t in specs)
            engine = Engine(
                backend, cost, kv_blocks=rng.randint(floor, 4 * floor), max_concurrency=rng.choice([None, 4])
            )
            for rid, (prompt, target) in enumerate(specs):
                engine.add(rid, prompt, target, 0.0)
            result = InstanceForecast(engine.state()).finish_times()
            outcome = engine.drain(max_events=100_000)
            assert_overflow_agrees(result, engine, outcome)
            outcomes["done" if outcome == "done" else "stuck"] += 1
            if not result.kv_overflow:
                worst.append(max(relative_errors(result, engine)))
        assert outcomes["done"] >= n_cases // 5, f"Too few cases finish without preemption: {outcomes}."
        assert_cases_close(worst)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_a_decode_step_claims_its_growth_before_a_queued_request_is_admitted(self, backend, cost):
        """Three block-aligned residents need 3 of the 14 free blocks, so a 12 block request does not fit."""
        engine = Engine(backend, cost, kv_blocks=44)
        for rid, target in enumerate([5, 10, 20]):
            engine.add(rid, 160, target, 0.0)
        engine.settle()
        engine.add(3, 192, 30)
        state = engine.state()
        assert state.used_blocks == 30 and len(state.running) == 3
        result = InstanceForecast(state).finish_times()
        assert not result.kv_overflow
        assert engine.drain() == "done"
        assert max(relative_errors(result, engine)) <= EXACT

    def test_a_pool_nothing_can_leave_is_reported_as_overflow(self, cost):
        waiting = tuple(resident(i, 320, 160, 50, False) for i in range(2))
        state = bare_state(cost, kv_blocks=20, waiting=waiting)
        result = InstanceForecast(state).finish_times()
        assert result.kv_overflow
        assert all(math.isinf(t) for t in result.finish_times.values())


def synthetic_state(cost, rng: random.Random) -> InstanceState:
    cap = rng.choice([None, None, 16, 48])
    n_run = rng.randint(0, 40)
    if cap is not None:
        n_run = min(n_run, cap)
    running = []
    for i in range(n_run):
        ctx = rng.randint(100, 20000)
        running.append(resident(i, ctx, ctx, rng.randint(1, 3000), True))
    waiting = []
    for j in range(rng.randint(0, 20)):
        ctx = rng.randint(100, 12000)
        waiting.append(
            resident(1000 + j, ctx, rng.choice([0, 0, rng.randint(0, ctx - 1)]), rng.randint(1, 3000), False)
        )
    return bare_state(
        cost,
        max_concurrency=cap,
        chunk_cap=rng.choice([None, None, 2048]),
        now=5.0,
        waiting=tuple(waiting),
        running=tuple(running),
    )


class TestProperties:
    def test_an_arrival_never_makes_a_resident_finish_earlier(self, cost):
        """
        Strict for a closed-form model.

        The composed model integrates a segment on a lattice, so a queued arrival that makes the
        engine step one iteration at a time reprices the same decode by up to the lattice error.
        """
        slack = EXACT if isinstance(cost, UnifiedRoofline) else LATTICE_ERROR
        rng = random.Random(3)
        for _ in range(120):
            state = synthetic_state(cost, rng)
            if not state.running and not state.waiting:
                continue
            base = InstanceForecast(state)
            arrival = Arrival(
                key=99999,
                arrive_time=state.now + rng.choice([0.0, rng.uniform(0.0, 30.0)]),
                prompt_len=rng.randint(10, 8000),
                generated=rng.choice([0, rng.randint(0, 3000)]),
                remaining=rng.randint(1, 2000),
            )
            before = base.finish_times().finish_times
            after = base.with_arrival(arrival).finish_times().finish_times
            for key, t in before.items():
                assert after[key] >= t - slack * t, f"{key} finished earlier: {after[key]} < {t}."

    def test_the_order_of_the_running_batch_does_not_matter(self, cost):
        rng = random.Random(5)
        for _ in range(40):
            state = synthetic_state(cost, rng)
            shuffled = list(state.running)
            rng.shuffle(shuffled)
            other = dataclasses.replace(state, running=tuple(shuffled))
            arrival = Arrival(key=99999, arrive_time=state.now + 1.0, prompt_len=900, generated=0, remaining=400)
            assert (
                InstanceForecast(state, (arrival,)).finish_times()
                == InstanceForecast(other, (arrival,)).finish_times()
            )

    def test_the_order_arrivals_are_given_in_does_not_matter(self, cost):
        rng = random.Random(6)
        state = synthetic_state(cost, rng)
        arrivals = [
            Arrival(key=9000 + j, arrive_time=state.now + 2.0 * j, prompt_len=500 + j, generated=0, remaining=300)
            for j in range(4)
        ]
        forward = InstanceForecast(state, arrivals).finish_times()
        backward = InstanceForecast(state, arrivals[::-1]).finish_times()
        assert forward == backward

    def test_with_arrival_leaves_the_original_forecast_alone(self, cost):
        state = synthetic_state(cost, random.Random(8))
        base = InstanceForecast(state)
        first = base.finish_times()
        extended = base.with_arrival(
            Arrival(key=99999, arrive_time=state.now, prompt_len=400, generated=0, remaining=100)
        )
        assert extended is not base
        assert base.finish_times() is first
        assert 99999 in extended.finish_times().finish_times
        assert 99999 not in base.finish_times().finish_times

    def test_extra_arrivals_passed_to_finish_times_equal_with_arrival(self, cost):
        state = synthetic_state(cost, random.Random(9))
        arrival = Arrival(key=99999, arrive_time=state.now + 1.5, prompt_len=700, generated=20, remaining=120)
        assert (
            InstanceForecast(state).finish_times((arrival,))
            == InstanceForecast(state).with_arrival(arrival).finish_times()
        )

    def test_duplicate_keys_are_rejected(self, cost):
        state = synthetic_state(cost, random.Random(10))
        key = (state.running or state.waiting)[0].key if (state.running or state.waiting) else 1
        with pytest.raises(ValueError, match="unique"):
            InstanceForecast(
                state, (Arrival(key=key, arrive_time=0.0, prompt_len=10, generated=0, remaining=1),)
            ).finish_times()


class TestSnapshot:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_from_instance_reads_both_queues(self, backend, cost):
        engine = Engine(backend, cost, max_concurrency=4)
        for rid, (prompt, target) in enumerate([(500, 300)] * 6):
            engine.add(rid, prompt, target, 0.0)
        engine.inst.advance_one_event_committed(0)
        state = engine.state()
        assert len(state.running) + len(state.waiting) == 6
        assert all(r.running for r in state.running) and not any(r.running for r in state.waiting)
        assert state.used_blocks == engine.inst.blocks_used()
        assert state.kv_blocks == BIG_POOL and state.block_size == BLOCK and state.now == engine.inst.t_local
        for r in state.running:
            assert r.prefilled == r.ctx
        assert [r.key for r in state.waiting] == [r.rid for r in engine.inst.waiting]

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_an_aborted_request_keeps_its_tokens_and_loses_its_kv(self, backend, cost):
        engine = Engine(backend, cost)
        engine.add(0, 400, 900, 0.0)
        engine.cut(2.0)
        generated = engine.requests[0].generated
        assert generated > 0
        engine.inst.abort_all(requeue=True)
        (resident,) = engine.state().waiting
        assert resident.ctx == 400 + generated and resident.prefilled == 0
        assert resident.remaining == 900 - generated
        assert engine.state().used_blocks == 0

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_trajectory_views_carry_what_the_forecast_reads(self, backend, cost):
        engine = Engine(backend, cost)
        engine.add(7, 400, 900, 0.0)
        view = traj_view(engine.requests[0])
        assert isinstance(view, TrajView)
        assert (view.rid, view.group_id, view.prefilled, view.running) == (7, 7, 0, False)
        engine.settle()
        assert traj_view(engine.requests[0]).running

    def test_a_noisy_prediction_is_rounded_to_whole_tokens_and_floored_at_one(self):
        traj = TrajView(
            generated=10, prompt_len=50, target_len=100, prompt_key=0, v_traj=0, admit_time=0.0, rid=3, running=True
        )
        assert Resident.from_traj(traj, 12.6).remaining == 13
        assert Resident.from_traj(traj, 0.2).remaining == 1

    def test_invalid_states_are_rejected(self, cost):
        with pytest.raises(ValueError, match="positive"):
            bare_state(cost, kv_blocks=0)
        with pytest.raises(ValueError, match="matching status"):
            bare_state(cost, running=(resident(1, 10, 0, 5, False),))
        with pytest.raises(ValueError, match="at least one token"):
            bare_state(cost, running=(resident(1, 10, 10, 0, True),))
        with pytest.raises(ValueError, match="token budget"):
            bare_state(cost, token_budget=1, running=tuple(resident(i, 10, 10, 5, True) for i in range(2)))
        with pytest.raises(ValueError, match="at least one token"):
            Arrival(key=1, arrive_time=0.0, prompt_len=10, generated=0, remaining=0)


class TestInterrupts:
    """Scheduled weight publishes layered onto the replay, mirroring `Simulator._publish_weights`."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_single_interrupt_matches_the_engine(self, backend, cost):
        rng = random.Random(40)
        worst = []
        for _ in range(12):
            n = rng.randint(4, 40)
            engine = Engine(
                backend, cost, max_concurrency=rng.choice([None, None, 16]), chunk_cap=rng.choice([None, None, 2048])
            )
            for rid, (prompt, target) in enumerate(random_specs(rng, n, prompt=(100, 8000), target=(200, 4000))):
                engine.add(rid, prompt, target, 0.0)
            state = engine.state()
            ix = Interrupt(time=state.now + rng.uniform(1.0, 20.0), resume_delay=rng.choice([0.0, 0.5, 5.0, 20.0]))
            result = InstanceForecast(state, interrupts=(ix,)).finish_times()
            apply_interrupt(engine, ix)
            engine.drain()
            worst.append(max(relative_errors(result, engine)))
        assert_cases_close(worst)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_two_consecutive_interrupts(self, backend, cost):
        rng = random.Random(41)
        worst = []
        for _ in range(10):
            n = rng.randint(4, 30)
            engine = Engine(backend, cost, max_concurrency=rng.choice([None, 16]))
            for rid, (prompt, target) in enumerate(random_specs(rng, n, prompt=(100, 6000), target=(300, 3000))):
                engine.add(rid, prompt, target, 0.0)
            state = engine.state()
            t = state.now
            interrupts = []
            for _ in range(2):
                t += rng.uniform(1.0, 15.0)
                delay = rng.choice([0.0, 1.0, 8.0])
                interrupts.append(Interrupt(time=t, resume_delay=delay))
                t += delay
            result = InstanceForecast(state, interrupts=tuple(interrupts)).finish_times()
            for ix in interrupts:
                apply_interrupt(engine, ix)
            engine.drain()
            worst.append(max(relative_errors(result, engine)))
        assert_cases_close(worst)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_a_request_spans_two_publishes(self, backend, cost):
        engine = Engine(backend, cost)
        engine.add(0, 2000, 6000, 0.0)
        probe = Engine(backend, cost)
        probe.add(0, 2000, 6000, 0.0)
        probe.drain()
        total = probe.requests[0].complete_time
        state = engine.state()
        first = Interrupt(time=total * 0.3, resume_delay=total * 0.05)
        second = Interrupt(time=total * 0.65, resume_delay=total * 0.05)
        result = InstanceForecast(state, interrupts=(first, second)).finish_times()
        apply_interrupt(engine, first)
        apply_interrupt(engine, second)
        engine.drain()
        assert max(relative_errors(result, engine)) <= STRETCH

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_interrupt_during_a_prefill_backlog(self, backend, cost):
        rng = random.Random(42)
        worst = []
        for _ in range(8):
            n = rng.randint(10, 40)
            engine = Engine(
                backend, cost, max_concurrency=rng.choice([None, 16]), chunk_cap=rng.choice([None, 2048])
            )
            for rid, (prompt, target) in enumerate(random_specs(rng, n, prompt=(2000, 12000), target=(500, 4000))):
                engine.add(rid, prompt, target, 0.0)
            for _ in range(rng.randint(1, 3)):
                engine.inst.advance_one_event_committed(0)
            assert engine.n_waiting > 0, "Setup must still be mid-backlog."
            state = engine.state()
            ix = Interrupt(time=state.now, resume_delay=rng.choice([0.0, 2.0, 10.0]))
            result = InstanceForecast(state, interrupts=(ix,)).finish_times()
            apply_interrupt(engine, ix)
            engine.drain()
            worst.append(max(relative_errors(result, engine)))
        assert_cases_close(worst)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_interrupt_on_an_empty_instance_is_a_noop(self, backend, cost):
        rng = random.Random(43)
        engine = Engine(backend, cost)
        state = engine.state()
        ix = Interrupt(time=state.now + 5.0, resume_delay=3.0)
        specs = random_specs(rng, 6, prompt=(100, 4000), target=(100, 2000))
        arrivals = tuple(
            Arrival(key=rid, arrive_time=state.now + 10.0 + rid, prompt_len=prompt, generated=0, remaining=target)
            for rid, (prompt, target) in enumerate(specs)
        )
        result = InstanceForecast(state, arrivals, (ix,)).finish_times()
        assert result.reprefill_tokens == (0,)
        apply_interrupt(engine, ix)
        assert engine.n_waiting == 0
        for a in arrivals:
            engine.cut(a.arrive_time)
            engine.add(a.key, a.prompt_len, a.remaining, a.arrive_time)
        engine.drain()
        assert max(relative_errors(result, engine)) <= EXACT

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_an_arrival_inside_the_dead_window_waits_for_resume(self, backend, cost):
        rng = random.Random(44)
        engine = Engine(backend, cost)
        for rid, (prompt, target) in enumerate(random_specs(rng, 6, prompt=(200, 4000), target=(300, 2000))):
            engine.add(rid, prompt, target, 0.0)
        state = engine.state()
        ix = Interrupt(time=state.now + 1.0, resume_delay=5.0)
        arrival = Arrival(key=900, arrive_time=ix.time + 2.0, prompt_len=1500, generated=0, remaining=400)
        assert ix.time <= arrival.arrive_time < ix.time + ix.resume_delay
        result = InstanceForecast(state, (arrival,), (ix,)).finish_times()
        later = dataclasses.replace(arrival, arrive_time=ix.time + ix.resume_delay + 50.0)
        apart = InstanceForecast(state, (later,), (ix,)).finish_times()
        assert result.finish_times[900] != apart.finish_times[900]
        apply_interrupt(engine, ix)
        engine.add(arrival.key, arrival.prompt_len, arrival.remaining, engine.inst.t_local)
        engine.drain()
        assert max(relative_errors(result, engine)) <= STRETCH

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_resume_delay_zero_reprefills_without_an_idle_gap(self, backend, cost):
        rng = random.Random(45)
        worst = []
        for _ in range(8):
            n = rng.randint(4, 20)
            engine = Engine(backend, cost)
            for rid, (prompt, target) in enumerate(random_specs(rng, n, prompt=(200, 6000), target=(300, 3000))):
                engine.add(rid, prompt, target, 0.0)
            engine.settle()
            state = engine.state()
            ix = Interrupt(time=state.now + rng.uniform(0.5, 5.0), resume_delay=0.0)
            result = InstanceForecast(state, interrupts=(ix,)).finish_times()
            apply_interrupt(engine, ix)
            engine.drain()
            worst.append(max(relative_errors(result, engine)))
        assert_cases_close(worst)

    def test_reprefill_tokens_counts_every_aborted_context(self, cost):
        rng = random.Random(46)
        engine = Engine("native", cost)
        for rid, (prompt, target) in enumerate(random_specs(rng, 10, prompt=(100, 6000), target=(300, 3000))):
            engine.add(rid, prompt, target, 0.0)
        state = engine.state()
        ix = Interrupt(time=state.now + 2.0, resume_delay=3.0)
        result = InstanceForecast(state, interrupts=(ix,)).finish_times()
        engine.cut(ix.time)
        expected = sum(r.ctx for r in engine.requests)
        assert result.reprefill_tokens == (expected,)

    def test_reprefill_tokens_pads_zero_when_the_pool_overflows_first(self, cost):
        waiting = tuple(resident(i, 320, 160, 50, False) for i in range(2))
        state = bare_state(cost, kv_blocks=20, waiting=waiting)
        ix = Interrupt(time=1000.0, resume_delay=1.0)
        result = InstanceForecast(state, interrupts=(ix,)).finish_times()
        assert result.kv_overflow
        assert result.reprefill_tokens == (0,)

    def test_the_order_interrupts_are_given_in_does_not_matter(self, cost):
        rng = random.Random(47)
        state = synthetic_state(cost, rng)
        interrupts = [Interrupt(state.now + 5.0, 2.0), Interrupt(state.now + 15.0, 1.0)]
        forward = InstanceForecast(state, interrupts=interrupts).finish_times()
        backward = InstanceForecast(state, interrupts=interrupts[::-1]).finish_times()
        assert forward == backward

    def test_with_interrupt_leaves_the_original_forecast_alone(self, cost):
        state = synthetic_state(cost, random.Random(48))
        base = InstanceForecast(state)
        first = base.finish_times()
        extended = base.with_interrupt(Interrupt(state.now + 5.0, 1.0))
        assert extended is not base
        assert base.finish_times() is first
        assert base.interrupts == ()
        assert extended.interrupts == (Interrupt(state.now + 5.0, 1.0),)

    def test_extra_interrupts_passed_to_finish_times_equal_with_interrupt(self, cost):
        state = synthetic_state(cost, random.Random(49))
        ix = Interrupt(state.now + 3.0, 2.0)
        assert (
            InstanceForecast(state).finish_times(interrupts=(ix,))
            == InstanceForecast(state).with_interrupt(ix).finish_times()
        )

    def test_negative_resume_delay_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Interrupt(time=0.0, resume_delay=-1.0)
