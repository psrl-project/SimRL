# Cadence-Aware Lane Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the wall time of K async-PSRL training steps by giving long-tail requests low-concurrency "fast lane" instances, re-planned at every weight publish, and measure honestly whether it beats the strongest homogeneous baseline.

**Architecture:** A pure planner (`srl/forecast/lanes.py`) assigns unplaced requests to instances, prices the assignment through the existing `InstanceForecast` → `group_etas` → `PipelineForecast` chain, and greedily moves a late buffer's critical group onto fewer-occupant instances while the objective improves. Two thin router strategies consume it: `slack_lane` (the real algorithm) and `length_lane` (a static ablation). A diagnosis module decomposes idle time and computes the contention-free lower bound. The measurement phases are delegated to the `validate` and `analysis` subagents.

**Tech Stack:** Python 3.12, existing `srl` package, `psrl.workers.ps.staleness` inventory replay, pytest, Hydra/OmegaConf config, matplotlib for figures.

## Global Constraints

- Spec: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/docs/superpowers/specs/2026-10-26-cadence-lane-scheduling-design.md`. Read it before Task 1.
- Objective is **minimize the end time of the K-th training step**. Step-interval variance and oscillation period are diagnostics, never targets.
- Activate the environment before running anything: `source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh`.
- Repo root is `/apdcephfs_zwfy/share_303760348/lhy/SimRL`. All paths below are relative to it unless absolute.
- Every new mechanism ships behind a flag whose default reproduces the baseline exactly. A treatment arm byte-identical to its control is a wiring bug, not "no effect".
- No backward-compatibility layers, no fallbacks, no migrations (AGENTS.md). Remove obsolete paths instead.
- Comments carry no measurements and no history. Numbers go in reports and memory, not in code comments.
- Doc cross-references use absolute paths.
- `srl/` code follows `.claude/coding-style.md`: Google-style docstrings with typed `Args:`, no bare `except`, no silent fallbacks.
- Heterogeneous TP is out of scope. Every instance is TP1 in every arm.
- Lengths come from the oracle predictor (`length_model="oracle"`). Prediction error is out of scope.
- The lane cap is realized **by how many requests the plan assigns to an instance**, not by a new engine knob. The engine's `max_concurrency` stays uniform across instances in every arm.
- Workload: `outputs/workload/qwen3_8b_dapo_38912_tp1/joint.jsonl` (1024 groups x 8, cap 38912). Fleet: 16 x TP1 Qwen3-8B on H20, `kernel_tp1` tables, `token_budget=16384`, KV from probe (31155 blocks x 16 tokens), `weight_load_time=0`.
- Baseline reference run already on disk: `outputs/simulate/cadence_probe/baseline_s2_n16_tau190`. It is a hint, never a baseline: every comparison arm is re-run in the same session as its treatment.

---

## File Structure

| File | Responsibility |
|---|---|
| `srl/forecast/lanes.py` | **Create.** Pure lane planner: value types, LPT assignment, KV-safe concurrency, pricing, greedy upgrade search. No router, no engine, no simulator imports. |
| `srl/forecast/__init__.py` | **Modify.** Export the planner's public names. |
| `srl/component/rollout/router/strategies/slack_lane.py` | **Create.** Algorithm A adapter: builds a `LanePlan` at each dispatch burst and routes to it. |
| `srl/component/rollout/router/strategies/length_lane.py` | **Create.** Algorithm C: static fast instances behind a length threshold. |
| `srl/component/rollout/router/strategies/__init__.py` | **Modify.** Register both strategies. |
| `srl/config/schema.py` | **Modify.** Add both names to the `RouterConfig.strategy` docstring. |
| `srl/diagnosis/cadence.py` | **Create.** Idle decomposition, step-interval series with autocorrelation, and the contention-free `LB_phys` replay. |
| `tests/test_forecast_lanes.py` | **Create.** Planner unit tests. |
| `tests/test_router_lane_strategies.py` | **Create.** Both strategies' candidate rules. |
| `tests/test_diagnosis_cadence.py` | **Create.** Idle split, autocorrelation, and bound replay. |
| `workflow/validate/experiments/cadence_lanes.py` | **Create (gitignored).** Sweep driver: trace permutation, `T_sat` calibration, arms, gates. |

---

### Task 1: Lane planner value types and assignment

**Files:**
- Create: `srl/forecast/lanes.py`
- Test: `tests/test_forecast_lanes.py`

**Interfaces:**
- Consumes: `srl.component.rollout.engine.blocks_for(tokens: int, block_size: int) -> int`.
- Produces:
  - `LaneJob(rid: int, group_id: int, prompt_key: int, prompt_len: int, generated: int, remaining: int)` with `final_ctx: int` property.
  - `LaneSlot(instance_id: int, cost, block_size: int, kv_blocks: int, token_budget: int, max_concurrency: int | None, tp: int)`.
  - `lpt_assign(jobs: Sequence[LaneJob], slot_ids: Sequence[int], pinned: Mapping[int, int] = {}) -> dict[int, int]`
  - `kv_safe_concurrency(jobs: Sequence[LaneJob], slot: LaneSlot) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_forecast_lanes.py`:

```python
"""The lane planner: assignment, KV-safe concurrency, pricing, and the upgrade search."""

from __future__ import annotations

import pytest

from srl.forecast.lanes import LaneJob, LaneSlot, kv_safe_concurrency, lpt_assign


def job(rid: int, remaining: int, group_id: int = 0, prompt_len: int = 100) -> LaneJob:
    return LaneJob(
        rid=rid,
        group_id=group_id,
        prompt_key=group_id,
        prompt_len=prompt_len,
        generated=0,
        remaining=remaining,
    )


def slot(instance_id: int, kv_blocks: int = 1000) -> LaneSlot:
    return LaneSlot(
        instance_id=instance_id,
        cost=None,
        block_size=16,
        kv_blocks=kv_blocks,
        token_budget=8192,
        max_concurrency=None,
        tp=1,
    )


class TestLaneJob:
    def test_final_ctx_sums_prompt_generated_and_remaining(self):
        assert LaneJob(0, 0, 0, prompt_len=100, generated=50, remaining=200).final_ctx == 350

    def test_remaining_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="at least one token"):
            LaneJob(0, 0, 0, prompt_len=100, generated=0, remaining=0)


class TestLptAssign:
    def test_longest_job_goes_to_the_first_slot(self):
        got = lpt_assign([job(0, 100), job(1, 300), job(2, 200)], [7, 9])
        assert got[1] == 7
        assert got[2] == 9

    def test_balances_total_remaining_across_slots(self):
        jobs = [job(i, length) for i, length in enumerate([500, 400, 300, 200, 100])]
        got = lpt_assign(jobs, [0, 1])
        load = {0: 0, 1: 0}
        for j in jobs:
            load[got[j.rid]] += j.remaining
        assert abs(load[0] - load[1]) <= 100

    def test_pinned_jobs_keep_their_slot_and_count_as_load(self):
        jobs = [job(0, 1000), job(1, 10)]
        got = lpt_assign(jobs, [0, 1], pinned={0: 0})
        assert got[0] == 0
        assert got[1] == 1

    def test_ties_break_on_rid_so_the_result_is_deterministic(self):
        first = lpt_assign([job(0, 100), job(1, 100)], [0, 1])
        second = lpt_assign([job(1, 100), job(0, 100)], [0, 1])
        assert first == second

    def test_no_slots_is_rejected(self):
        with pytest.raises(ValueError, match="at least one slot"):
            lpt_assign([job(0, 100)], [])


class TestKvSafeConcurrency:
    def test_every_job_fits_when_the_pool_is_large(self):
        jobs = [job(i, 1000) for i in range(4)]
        assert kv_safe_concurrency(jobs, slot(0, kv_blocks=1000)) == 4

    def test_counts_only_the_jobs_whose_final_contexts_fit(self):
        # Each job's final ctx is 100 + 1500 = 1600 tokens = 100 blocks at block_size 16.
        jobs = [job(i, 1500) for i in range(10)]
        assert kv_safe_concurrency(jobs, slot(0, kv_blocks=250)) == 2

    def test_at_least_one_job_is_always_allowed(self):
        jobs = [job(0, 100_000)]
        assert kv_safe_concurrency(jobs, slot(0, kv_blocks=1)) == 1

    def test_no_jobs_means_no_concurrency(self):
        assert kv_safe_concurrency([], slot(0)) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
python -m pytest tests/test_forecast_lanes.py -v
```
Expected: collection error, `ModuleNotFoundError: No module named 'srl.forecast.lanes'`.

- [ ] **Step 3: Write the implementation**

Create `srl/forecast/lanes.py`:

```python
"""Plan which instance each unplaced request goes to, and how many share it.

A lane is not a configured pool. It is however many requests the plan puts on one instance:
few requests means low per-token latency, many means high aggregate throughput. The planner
therefore emits an assignment, and the instances that ended up holding few requests are the
fast lanes. Nothing here touches the engine, so the whole search is a pure function of the
jobs, the slots, and the inventory snapshot it is given.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..component.rollout.engine import blocks_for


@dataclass(frozen=True, slots=True)
class LaneJob:
    """One request the planner must place."""

    rid: int
    group_id: int
    prompt_key: int
    prompt_len: int
    generated: int
    remaining: int
    """Predicted decode tokens still to generate, at least 1."""

    def __post_init__(self) -> None:
        if self.remaining < 1:
            raise ValueError(f"A job must have at least one token left to generate, got {self.remaining!r}.")

    @property
    def final_ctx(self) -> int:
        """Context the job holds once it finishes, which bounds the blocks it will ever need."""
        return self.prompt_len + self.generated + self.remaining


@dataclass(frozen=True, slots=True)
class LaneSlot:
    """One instance the planner may assign jobs to, and the limits it prices against."""

    instance_id: int
    cost: object
    """The instance's rollout cost model, passed through to `InstanceState`."""
    block_size: int
    kv_blocks: int
    token_budget: int
    max_concurrency: int | None
    tp: int


def lpt_assign(
    jobs: Sequence[LaneJob],
    slot_ids: Sequence[int],
    pinned: Mapping[int, int] = field(default_factory=dict),
) -> dict[int, int]:
    """
    Assign every job to a slot, longest predicted remaining first, onto the least-loaded slot.

    Args:
        jobs (Sequence[LaneJob]): Jobs to place, including the pinned ones.
        slot_ids (Sequence[int]): Instance ids the assignment may use.
        pinned (Mapping[int, int]): Request id to the slot it is already on and must stay on.
            Its load counts against that slot, so the free jobs balance around it.

    Returns:
        dict[int, int]: Request id to slot id, covering every job.

    Raises:
        ValueError: If `slot_ids` is empty, or a pinned job names a slot outside `slot_ids`.
    """
    if not slot_ids:
        raise ValueError("lpt_assign needs at least one slot.")
    allowed = set(slot_ids)
    load = {slot_id: 0 for slot_id in slot_ids}
    out: dict[int, int] = {}
    for job in jobs:
        slot_id = pinned.get(job.rid)
        if slot_id is None:
            continue
        if slot_id not in allowed:
            raise ValueError(f"Pinned job {job.rid} names slot {slot_id!r}, which is not in slot_ids.")
        out[job.rid] = slot_id
        load[slot_id] += job.remaining
    free = sorted((j for j in jobs if j.rid not in out), key=lambda j: (-j.remaining, j.rid))
    for job in free:
        slot_id = min(slot_ids, key=lambda s: (load[s], s))
        out[job.rid] = slot_id
        load[slot_id] += job.remaining
    return out


def kv_safe_concurrency(jobs: Sequence[LaneJob], slot: LaneSlot) -> int:
    """
    Return how many of `jobs` may decode at once without the pool ever overflowing.

    A job's block usage grows monotonically to `blocks_for(final_ctx)`, so the sum of those
    bounds over a running set bounds that set's peak. Counting the longest jobs first makes
    the answer independent of the order `jobs` arrives in. The result is at least 1 whenever
    there is any job: an instance holding one request cannot do better by holding none, and
    the engine would preempt rather than stall.

    Args:
        jobs (Sequence[LaneJob]): The jobs assigned to this slot.
        slot (LaneSlot): The slot's block size and pool size.

    Returns:
        int: The largest safe running count, 0 when `jobs` is empty.
    """
    if not jobs:
        return 0
    used = 0
    count = 0
    for job in sorted(jobs, key=lambda j: (-j.final_ctx, j.rid)):
        used += blocks_for(job.final_ctx, slot.block_size)
        if used > slot.kv_blocks:
            break
        count += 1
    return max(1, count)
```

Note on `pinned`: a `field(default_factory=dict)` default in a plain function signature is wrong. Use `pinned: Mapping[int, int] | None = None` and normalise inside:

```python
def lpt_assign(
    jobs: Sequence[LaneJob],
    slot_ids: Sequence[int],
    pinned: Mapping[int, int] | None = None,
) -> dict[int, int]:
```
with `pinned = pinned or {}` as the first line after the empty-`slot_ids` check, and drop the unused `field` import.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_forecast_lanes.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add srl/forecast/lanes.py tests/test_forecast_lanes.py
git commit -m "feat(forecast): add lane planner value types and LPT assignment

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Price an assignment through the training pipeline

**Files:**
- Modify: `srl/forecast/lanes.py`
- Test: `tests/test_forecast_lanes.py`

**Interfaces:**
- Consumes: `LaneJob`, `LaneSlot`, `kv_safe_concurrency` (Task 1); `srl.forecast.InstanceState`, `Resident`, `InstanceForecast`, `group_etas`, `PipelineForecast`, `TrainState`; `psrl.workers.ps.staleness.InventorySnapshot`.
- Produces:
  - `Pricing(objective: float, pipeline: Pipeline, member_finish: dict[int, float], group_eta: dict[int, float])`
  - `state_for(slot: LaneSlot, jobs: Sequence[LaneJob], now: float) -> InstanceState`
  - `price(assignment: Mapping[int, int], jobs: Sequence[LaneJob], slots: Sequence[LaneSlot], *, snapshot: InventorySnapshot, train: TrainState, members: Mapping[int, Sequence[int]], finished: Mapping[int, float], now: float, horizon_buffers: int) -> Pricing`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_forecast_lanes.py`:

```python
import math

from psrl.workers.ps.staleness import BufferView, InventorySnapshot

from srl.cost.rollout import UnifiedRoofline
from srl.forecast import TrainState
from srl.forecast.lanes import price, state_for

ROOFLINE = UnifiedRoofline(F=0.006, W=0.002, G=4e-6, A_p=3e-10, A_d=5e-8, b=16)


def priced_slot(instance_id: int, kv_blocks: int = 100_000) -> LaneSlot:
    return LaneSlot(
        instance_id=instance_id,
        cost=ROOFLINE,
        block_size=16,
        kv_blocks=kv_blocks,
        token_budget=8192,
        max_concurrency=None,
        tp=1,
    )


class TestStateFor:
    def test_jobs_become_waiting_residents_with_no_kv_computed(self):
        state = state_for(priced_slot(0), [job(0, 100), job(1, 200)], now=5.0)
        assert state.now == 5.0
        assert state.running == ()
        assert {r.key for r in state.waiting} == {0, 1}
        assert all(r.prefilled == 0 for r in state.waiting)

    def test_concurrency_is_capped_by_the_pool(self):
        # Final ctx 100 + 1500 = 1600 tokens = 100 blocks each; a 250-block pool holds two.
        jobs = [job(i, 1500) for i in range(10)]
        assert state_for(priced_slot(0, kv_blocks=250), jobs, now=0.0).max_concurrency == 2

    def test_the_slot_cap_wins_when_it_is_tighter_than_the_pool(self):
        jobs = [job(i, 100) for i in range(10)]
        tight = dataclasses.replace(priced_slot(0), max_concurrency=3)
        assert state_for(tight, jobs, now=0.0).max_concurrency == 3


class TestPrice:
    def snapshot_one_buffer(self, group_ids):
        reserved = tuple((group_id, 0) for group_id in group_ids)
        return InventorySnapshot(
            staleness=0,
            num_entries=len(group_ids),
            next_buffer_id=1,
            buffers=(BufferView(buffer_id=0, ready_num_entries=len(group_ids), n_occupied=0, reserved=reserved),),
        )

    def test_the_objective_is_the_training_end_of_the_horizon(self):
        jobs = [job(0, 500, group_id=0), job(1, 500, group_id=1)]
        slots = [priced_slot(0), priced_slot(1)]
        pricing = price(
            {0: 0, 1: 1},
            jobs,
            slots,
            snapshot=self.snapshot_one_buffer([0, 1]),
            train=TrainState(training_end=None, now=0.0, cycle=10.0, weight_load=0.0),
            members={0: [0], 1: [1]},
            finished={},
            now=0.0,
            horizon_buffers=2,
        )
        assert pricing.pipeline.buffers == (0,)
        assert pricing.objective == pytest.approx(pricing.pipeline.train_end[0])
        assert pricing.objective > 10.0

    def test_splitting_two_long_jobs_beats_stacking_them(self):
        jobs = [job(0, 20_000, group_id=0), job(1, 20_000, group_id=1)]
        slots = [priced_slot(0), priced_slot(1)]
        common = dict(
            snapshot=self.snapshot_one_buffer([0, 1]),
            train=TrainState(training_end=None, now=0.0, cycle=10.0, weight_load=0.0),
            members={0: [0], 1: [1]},
            finished={},
            now=0.0,
            horizon_buffers=2,
        )
        split = price({0: 0, 1: 1}, jobs, slots, **common)
        stacked = price({0: 0, 1: 0}, jobs, slots, **common)
        assert split.objective < stacked.objective

    def test_a_finished_member_does_not_need_a_placement(self):
        jobs = [job(0, 500, group_id=0)]
        pricing = price(
            {0: 0},
            jobs,
            [priced_slot(0)],
            snapshot=self.snapshot_one_buffer([0]),
            train=TrainState(training_end=None, now=0.0, cycle=10.0, weight_load=0.0),
            members={0: [0, 99]},
            finished={99: 3.0},
            now=0.0,
            horizon_buffers=2,
        )
        assert math.isfinite(pricing.group_eta[0])

    def test_an_unplaced_member_makes_the_objective_infinite(self):
        jobs = [job(0, 500, group_id=0)]
        pricing = price(
            {0: 0},
            jobs,
            [priced_slot(0)],
            snapshot=self.snapshot_one_buffer([0]),
            train=TrainState(training_end=None, now=0.0, cycle=10.0, weight_load=0.0),
            members={0: [0, 99]},
            finished={},
            now=0.0,
            horizon_buffers=2,
        )
        assert pricing.objective == math.inf

    def test_an_assignment_naming_an_unknown_slot_is_rejected(self):
        with pytest.raises(ValueError, match="unknown slot"):
            price(
                {0: 5},
                [job(0, 500, group_id=0)],
                [priced_slot(0)],
                snapshot=self.snapshot_one_buffer([0]),
                train=TrainState(training_end=None, now=0.0, cycle=10.0, weight_load=0.0),
                members={0: [0]},
                finished={},
                now=0.0,
                horizon_buffers=2,
            )
```

Add `import dataclasses` to the test file's imports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_forecast_lanes.py -k "StateFor or Price" -v`
Expected: `ImportError: cannot import name 'price' from 'srl.forecast.lanes'`.

- [ ] **Step 3: Write the implementation**

Append to `srl/forecast/lanes.py` (and extend its imports):

```python
import math

from .drain import InstanceForecast
from .pipeline import Pipeline, PipelineForecast, TrainState, group_etas
from .snapshot import InstanceState, Resident
```

```python
@dataclass(frozen=True, slots=True)
class Pricing:
    """What one assignment costs the training pipeline."""

    objective: float
    """`J`, the sum of `E_k` over the priced horizon. `inf` when any buffer never becomes ready."""
    pipeline: Pipeline
    member_finish: dict[int, float]
    group_eta: dict[int, float]


def state_for(slot: LaneSlot, jobs: Sequence[LaneJob], now: float) -> InstanceState:
    """
    Project one slot and the jobs assigned to it into the drain forecast's input.

    Every job enters as a waiting resident with no KV computed, which is what a publish leaves
    behind: `abort_all` frees the pool and resets `prefilled`, keeping the generated tokens. The
    forecast's running batch is capped at `kv_safe_concurrency`, so the replay queues what the
    pool cannot hold instead of reporting an overflow that would void its finish times.

    Args:
        slot (LaneSlot): The instance's cost model and limits.
        jobs (Sequence[LaneJob]): Jobs assigned to this slot.
        now (float): Simulated time the forecast starts from.
    """
    safe = kv_safe_concurrency(jobs, slot)
    cap = safe if slot.max_concurrency is None else min(slot.max_concurrency, safe)
    waiting = tuple(
        Resident(
            key=job.rid,
            group_id=job.group_id,
            prompt_key=job.prompt_key,
            prompt_len=job.prompt_len,
            ctx=job.prompt_len + job.generated,
            prefilled=0,
            remaining=job.remaining,
            running=False,
        )
        for job in jobs
    )
    return InstanceState(
        cost=slot.cost,
        block_size=slot.block_size,
        kv_blocks=slot.kv_blocks,
        token_budget=slot.token_budget,
        max_concurrency=cap if jobs else None,
        chunk_cap=None,
        tp=slot.tp,
        now=now,
        waiting=waiting,
        running=(),
    )


def price(
    assignment: Mapping[int, int],
    jobs: Sequence[LaneJob],
    slots: Sequence[LaneSlot],
    *,
    snapshot,
    train: TrainState,
    members: Mapping[int, Sequence[int]],
    finished: Mapping[int, float],
    now: float,
    horizon_buffers: int,
) -> Pricing:
    """
    Price `assignment` as one training pipeline.

    Each slot's jobs are drained independently, since a closed assignment gives the instances
    nothing to share. The resulting member finish times become group ETAs, the inventory replay
    turns those into each buffer's READY time, and the training chain turns those into `E_k`.

    Args:
        assignment (Mapping[int, int]): Request id to slot id, covering every job.
        jobs (Sequence[LaneJob]): The jobs the assignment covers.
        slots (Sequence[LaneSlot]): Every slot the assignment may name.
        snapshot (InventorySnapshot): The staleness inventory to replay.
        train (TrainState): The training loop's state, priced for the next cycle.
        members (Mapping[int, Sequence[int]]): Group id to its member request ids.
        finished (Mapping[int, float]): Completion time of an already-finished member, by id.
        now (float): Simulated time the forecasts start from.
        horizon_buffers (int): Buffers the objective sums over, from the earliest pending one.

    Returns:
        Pricing: The objective and the pipeline it came from.

    Raises:
        ValueError: If `assignment` names a slot that is not in `slots`, or omits a job.
    """
    by_slot: dict[int, list[LaneJob]] = {slot.instance_id: [] for slot in slots}
    for job in jobs:
        slot_id = assignment.get(job.rid)
        if slot_id is None:
            raise ValueError(f"Assignment omits job {job.rid}.")
        if slot_id not in by_slot:
            raise ValueError(f"Assignment sends job {job.rid} to unknown slot {slot_id!r}.")
        by_slot[slot_id].append(job)

    member_finish: dict[int, float] = {}
    for slot in slots:
        placed = by_slot[slot.instance_id]
        if not placed:
            continue
        result = InstanceForecast(state_for(slot, placed, now)).finish_times()
        member_finish.update(result.finish_times)

    eta = group_etas(member_finish, members, finished)
    pipeline = PipelineForecast(snapshot, train).evaluate(eta)
    horizon = pipeline.train_end[:horizon_buffers]
    objective = math.inf if any(math.isinf(e) for e in horizon) else sum(horizon)
    return Pricing(objective=objective, pipeline=pipeline, member_finish=member_finish, group_eta=eta)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_forecast_lanes.py -v`
Expected: all passed (11 from Task 1 plus 8 here).

- [ ] **Step 5: Commit**

```bash
git add srl/forecast/lanes.py tests/test_forecast_lanes.py
git commit -m "feat(forecast): price a lane assignment through the training pipeline

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Greedy lane upgrade search

**Files:**
- Modify: `srl/forecast/lanes.py`
- Modify: `srl/forecast/__init__.py`
- Test: `tests/test_forecast_lanes.py`

**Interfaces:**
- Consumes: everything from Tasks 1-2.
- Produces:
  - `LanePlan(assignment: dict[int, int], fast_instances: frozenset[int], objective: float, moves: int)`
  - `plan_lanes(jobs, slots, *, snapshot, train, members, finished, now, pinned=None, horizon_buffers=2, c_max=4, eps_frac=0.01, max_moves=8) -> LanePlan`
- `srl/forecast/__init__.py` exports `LaneJob`, `LanePlan`, `LaneSlot`, `Pricing`, `kv_safe_concurrency`, `lpt_assign`, `plan_lanes`, `price`, `state_for`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_forecast_lanes.py`:

```python
from srl.forecast.lanes import LanePlan, plan_lanes


class TestPlanLanes:
    def fleet(self, n: int):
        return [priced_slot(i) for i in range(n)]

    def snapshot_two_buffers(self, early_groups, late_groups):
        """Buffer 0 holds `early_groups`, buffer 1 holds `late_groups`, staleness 1."""
        return InventorySnapshot(
            staleness=1,
            num_entries=max(len(early_groups), len(late_groups)),
            next_buffer_id=2,
            buffers=(
                BufferView(
                    buffer_id=0,
                    ready_num_entries=max(len(early_groups), len(late_groups)),
                    n_occupied=0,
                    reserved=tuple((g, 0) for g in early_groups),
                ),
                BufferView(
                    buffer_id=1,
                    ready_num_entries=max(len(early_groups), len(late_groups)),
                    n_occupied=0,
                    reserved=tuple((g, 1) for g in late_groups),
                ),
            ),
        )

    def test_a_balanced_fleet_needs_no_upgrade(self):
        jobs = [job(i, 2000, group_id=i) for i in range(4)]
        plan = plan_lanes(
            jobs,
            self.fleet(4),
            snapshot=self.snapshot_one_buffer_n(4),
            train=TrainState(training_end=None, now=0.0, cycle=1e6, weight_load=0.0),
            members={i: [i] for i in range(4)},
            finished={},
            now=0.0,
        )
        assert plan.moves == 0
        assert plan.fast_instances == frozenset()

    def test_a_straggler_is_moved_onto_a_slot_of_its_own(self):
        # Three short groups and one 30k straggler on two slots: LPT puts the straggler with a
        # short job, and the trainer waits on it. Making a slot fast must beat that.
        jobs = [job(0, 30_000, group_id=0)] + [job(i, 1_000, group_id=i) for i in (1, 2, 3)]
        plan = plan_lanes(
            jobs,
            self.fleet(2),
            snapshot=self.snapshot_one_buffer_n(4),
            train=TrainState(training_end=None, now=0.0, cycle=5.0, weight_load=0.0),
            members={i: [i] for i in range(4)},
            finished={},
            now=0.0,
            c_max=1,
        )
        straggler_slot = plan.assignment[0]
        assert [rid for rid, s in plan.assignment.items() if s == straggler_slot] == [0]
        assert straggler_slot in plan.fast_instances

    def test_the_plan_never_loses_to_the_bulk_only_assignment(self):
        jobs = [job(0, 30_000, group_id=0)] + [job(i, 2_000, group_id=i) for i in range(1, 8)]
        slots = self.fleet(4)
        common = dict(
            snapshot=self.snapshot_one_buffer_n(8),
            train=TrainState(training_end=None, now=0.0, cycle=5.0, weight_load=0.0),
            members={i: [i] for i in range(8)},
            finished={},
            now=0.0,
        )
        plan = plan_lanes(jobs, slots, **common)
        bulk = price(lpt_assign(jobs, [s.instance_id for s in slots]), jobs, slots, horizon_buffers=2, **common)
        assert plan.objective <= bulk.objective

    def test_max_moves_bounds_the_search(self):
        jobs = [job(i, 30_000, group_id=i) for i in range(6)]
        plan = plan_lanes(
            jobs,
            self.fleet(6),
            snapshot=self.snapshot_one_buffer_n(6),
            train=TrainState(training_end=None, now=0.0, cycle=5.0, weight_load=0.0),
            members={i: [i] for i in range(6)},
            finished={},
            now=0.0,
            max_moves=1,
        )
        assert plan.moves <= 1

    def test_pinned_jobs_are_never_moved(self):
        jobs = [job(0, 30_000, group_id=0)] + [job(i, 1_000, group_id=i) for i in (1, 2)]
        plan = plan_lanes(
            jobs,
            self.fleet(2),
            snapshot=self.snapshot_one_buffer_n(3),
            train=TrainState(training_end=None, now=0.0, cycle=5.0, weight_load=0.0),
            members={i: [i] for i in range(3)},
            finished={},
            now=0.0,
            pinned={0: 1},
        )
        assert plan.assignment[0] == 1

    def snapshot_one_buffer_n(self, n):
        return InventorySnapshot(
            staleness=0,
            num_entries=n,
            next_buffer_id=1,
            buffers=(BufferView(buffer_id=0, ready_num_entries=n, n_occupied=0, reserved=tuple((g, 0) for g in range(n))),),
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_forecast_lanes.py -k PlanLanes -v`
Expected: `ImportError: cannot import name 'plan_lanes'`.

- [ ] **Step 3: Write the implementation**

Append to `srl/forecast/lanes.py`:

```python
@dataclass(frozen=True, slots=True)
class LanePlan:
    """Where every job goes, and which instances ended up as fast lanes."""

    assignment: dict[int, int]
    fast_instances: frozenset[int]
    """Instances the search deliberately thinned out. Empty when the bulk plan was never beaten."""
    objective: float
    moves: int
    """Upgrades the search accepted, for telemetry and for bounding the search in tests."""


def plan_lanes(
    jobs: Sequence[LaneJob],
    slots: Sequence[LaneSlot],
    *,
    snapshot,
    train: TrainState,
    members: Mapping[int, Sequence[int]],
    finished: Mapping[int, float],
    now: float,
    pinned: Mapping[int, int] | None = None,
    horizon_buffers: int = 2,
    c_max: int = 4,
    eps_frac: float = 0.01,
    max_moves: int = 8,
) -> LanePlan:
    """
    Assign jobs to slots, then thin out slots until no late buffer can be improved.

    The search starts from the balanced assignment and repeatedly asks the same question: which
    pending buffer would make the trainer idle, and which group holds it up? It then tries to
    give that group's late members fewer neighbours, either on a slot already thinned out or by
    thinning a fresh one, and keeps the move that lowers the objective most. The number of fast
    lanes and their occupancy are outputs of this loop, never inputs.

    Args:
        jobs (Sequence[LaneJob]): Every job in flight or awaiting placement.
        slots (Sequence[LaneSlot]): The fleet.
        snapshot (InventorySnapshot): The staleness inventory to replay.
        train (TrainState): The training loop's state, held fixed across the search.
        members (Mapping[int, Sequence[int]]): Group id to its member request ids.
        finished (Mapping[int, float]): Completion time of an already-finished member, by id.
        now (float): Simulated time the forecasts start from.
        pinned (Mapping[int, int] | None): Jobs already placed, which keep their slot.
        horizon_buffers (int): Buffers the objective sums over.
        c_max (int): Jobs a fast lane may hold.
        eps_frac (float): Fraction of one training cycle a move must save to be accepted.
        max_moves (int): Upgrades the search may accept.

    Returns:
        LanePlan: The assignment, the fast lanes, the objective, and the moves accepted.

    Raises:
        ValueError: If `c_max` is below 1 or `slots` is empty.
    """
    if c_max < 1:
        raise ValueError(f"c_max must be at least 1, got {c_max!r}.")
    if not slots:
        raise ValueError("plan_lanes needs at least one slot.")
    pinned = pinned or {}
    slot_ids = [slot.instance_id for slot in slots]
    by_rid = {job.rid: job for job in jobs}
    pricing_args = dict(
        snapshot=snapshot,
        train=train,
        members=members,
        finished=finished,
        now=now,
        horizon_buffers=horizon_buffers,
    )

    assignment = lpt_assign(jobs, slot_ids, pinned=pinned)
    best = price(assignment, jobs, slots, **pricing_args)
    eps = eps_frac * train.cycle
    fast: set[int] = set()
    unfixable: set[int] = set()
    moves = 0

    while moves < max_moves:
        late = _first_late_buffer(best.pipeline, train, now, horizon_buffers, unfixable)
        if late is None:
            break
        buffer_id = best.pipeline.buffers[late]
        target = _target_time(best.pipeline, train, now, late)
        critical = _critical_group(snapshot, best.group_eta, members, buffer_id)
        if critical is None:
            unfixable.add(buffer_id)
            continue
        movers = [
            rid
            for rid in members[critical]
            if rid in by_rid and rid not in pinned and best.member_finish.get(rid, math.inf) > target
        ]
        candidates = _candidate_moves(movers, assignment, by_rid, slot_ids, fast, c_max, pinned)
        winner = None
        for cand_assignment, cand_fast in candidates:
            pricing = price(cand_assignment, jobs, slots, **pricing_args)
            if pricing.objective < best.objective - eps and (winner is None or pricing.objective < winner[0].objective):
                winner = (pricing, cand_assignment, cand_fast)
        if winner is None:
            unfixable.add(buffer_id)
            continue
        best, assignment, fast = winner[0], winner[1], set(winner[2])
        moves += 1

    return LanePlan(
        assignment=dict(assignment),
        fast_instances=frozenset(fast),
        objective=best.objective,
        moves=moves,
    )


def _target_time(pipeline: Pipeline, train: TrainState, now: float, index: int) -> float:
    """When buffer `index` must be ready for the trainer not to idle before it."""
    if index > 0:
        return pipeline.train_end[index - 1]
    return now if train.training_end is None else train.training_end + train.weight_load


def _first_late_buffer(
    pipeline: Pipeline,
    train: TrainState,
    now: float,
    horizon_buffers: int,
    skip: set[int],
) -> int | None:
    """Index of the earliest priced buffer the trainer would wait for, skipping `skip`."""
    for index in range(min(horizon_buffers, len(pipeline.buffers))):
        if pipeline.buffers[index] in skip:
            continue
        if pipeline.ready[index] > _target_time(pipeline, train, now, index):
            return index
    return None


def _critical_group(
    snapshot,
    group_eta: Mapping[int, float],
    members: Mapping[int, Sequence[int]],
    buffer_id: int,
) -> int | None:
    """
    The latest-finishing group that buffer `buffer_id` cannot be ready without.

    A group whose deadline is at or before `buffer_id` must be consumed by it at the latest, so
    the buffer cannot fill until every one of them has finished. The slowest of those is what
    sets the buffer's READY time, and the only group worth speeding up.
    """
    worst_eta = -math.inf
    worst: int | None = None
    for group_id in members:
        try:
            deadline = snapshot.deadline(group_id)
        except KeyError:
            continue
        if deadline > buffer_id:
            continue
        eta = group_eta.get(group_id, math.inf)
        if eta > worst_eta:
            worst_eta, worst = eta, group_id
    return worst


def _candidate_moves(
    movers: Sequence[int],
    assignment: Mapping[int, int],
    by_rid: Mapping[int, LaneJob],
    slot_ids: Sequence[int],
    fast: set[int],
    c_max: int,
    pinned: Mapping[int, int],
) -> list[tuple[dict[int, int], set[int]]]:
    """
    Enumerate the ways `movers` could get fewer neighbours.

    Two shapes only: join a fast lane that still has room, or thin a bulk lane out by
    rebalancing its current jobs over the remaining bulk lanes. Enumerating every bulk lane
    would price N nearly identical options, so only the emptiest one is offered.
    """
    if not movers:
        return []
    out: list[tuple[dict[int, int], set[int]]] = []
    load = {slot_id: 0 for slot_id in slot_ids}
    held = {slot_id: 0 for slot_id in slot_ids}
    for rid, slot_id in assignment.items():
        load[slot_id] += by_rid[rid].remaining
        held[slot_id] += 1

    for slot_id in sorted(fast):
        if held[slot_id] + len(movers) > c_max:
            continue
        candidate = dict(assignment)
        for rid in movers:
            candidate[rid] = slot_id
        out.append((candidate, set(fast)))

    bulk = [slot_id for slot_id in slot_ids if slot_id not in fast]
    if len(bulk) > 1 and len(movers) <= c_max:
        host = min(bulk, key=lambda s: (load[s], s))
        remaining_bulk = [slot_id for slot_id in bulk if slot_id != host]
        evicted = [by_rid[rid] for rid, slot_id in assignment.items() if slot_id == host and rid not in movers]
        candidate = {rid: slot_id for rid, slot_id in assignment.items() if slot_id != host}
        candidate.update(lpt_assign(evicted, remaining_bulk, pinned={r: s for r, s in pinned.items() if s != host}))
        for rid in movers:
            candidate[rid] = host
        out.append((candidate, {*fast, host}))
    return out
```

Then extend `srl/forecast/__init__.py`: add

```python
from .lanes import (
    LaneJob,
    LanePlan,
    LaneSlot,
    Pricing,
    kv_safe_concurrency,
    lpt_assign,
    plan_lanes,
    price,
    state_for,
)
```

and add `"LaneJob"`, `"LanePlan"`, `"LaneSlot"`, `"Pricing"`, `"kv_safe_concurrency"`, `"lpt_assign"`, `"plan_lanes"`, `"price"`, `"state_for"` to `__all__`, keeping it sorted.

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python -m pytest tests/test_forecast_lanes.py -v
python -m pytest tests/test_forecast_drain.py tests/test_forecast_pipeline.py tests/test_forecast_plan.py -q
```
Expected: the new file all passes; the three existing forecast suites still pass.

- [ ] **Step 5: Commit**

```bash
git add srl/forecast/lanes.py srl/forecast/__init__.py tests/test_forecast_lanes.py
git commit -m "feat(forecast): add greedy lane upgrade search

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The `slack_lane` routing strategy (Algorithm A)

**Files:**
- Create: `srl/component/rollout/router/strategies/slack_lane.py`
- Modify: `srl/component/rollout/router/strategies/__init__.py`
- Modify: `srl/config/schema.py:258-260` (the `RouterConfig.strategy` docstring)
- Test: `tests/test_router_lane_strategies.py`

**Interfaces:**
- Consumes: `plan_lanes`, `LaneJob`, `LaneSlot`, `LanePlan` (Task 3); `srl.forecast.fleet.members_of`, `sequence_hint`; `srl.forecast.build_length_predictor`; `srl.component.rollout.engine.{InstanceView, Request, traj_view}`; `register_strategy`; `SchedulingRuntime` (`bind_runtime`, `inflight`, `queued()`, `pending`, `instances`, `inventory()`, `train_state()`, `batch_size`, `group_size`, `cost_model()`).
- Produces: strategy name `"slack_lane"`, constructor
  `SlackLane(length_model="oracle", length_model_args=None, horizon_buffers=2, c_max=4, eps_frac=0.01, max_moves=8, enable_upgrade=True)`.
  `enable_upgrade=False` is the **A-nolane** arm: plan-based LPT with no fast lanes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_router_lane_strategies.py`:

```python
"""Candidate rules of the two lane strategies, without building a simulator."""

from __future__ import annotations

from srl.component.rollout.engine import InstanceView
from srl.component.rollout.router.strategy import RoutingContext, build_strategy, narrow_candidates


class FakeRequest:
    """The read surface `traj_view` needs from a request."""

    def __init__(self, rid: int, group_id: int, prompt_len: int, target_len: int, generated: int = 0):
        self.rid = rid
        self.group_id = group_id
        self.prompt_len = prompt_len
        self.target_len = target_len
        self.generated = generated
        self.prompt_key = group_id
        self.v_traj = 0
        self.admit_time = 0.0
        self.prefilled = 0
        self.running = False
        self.instance_id = None

    @property
    def ctx(self) -> int:
        return self.prompt_len + self.generated


def view(instance_id: int, residents=()) -> InstanceView:
    return InstanceView(
        instance_id=instance_id,
        n_waiting=0,
        n_running=len(residents),
        kv_utilization=0.0,
        total_ctx=0,
        tp=1,
        free_blocks=10_000,
        committed_blocks=0,
        residents=lambda: list(residents),
        kv_blocks=10_000,
        block_size=16,
    )


CONTEXT = RoutingContext(blocked_versions=(), fleet_min_tp=1, fleet_max_tp=1, now=0.0)


class TestSlackLaneWithoutAPlan:
    def test_it_offers_every_candidate_before_the_first_dispatch(self):
        strategy = build_strategy("slack_lane")
        req = FakeRequest(rid=0, group_id=0, prompt_len=100, target_len=1000)
        candidates = [view(0), view(1)]
        assert narrow_candidates(strategy, req, candidates, CONTEXT) == [0, 1]

    def test_it_falls_back_to_the_least_loaded_instance(self):
        strategy = build_strategy("slack_lane")
        req = FakeRequest(rid=0, group_id=0, prompt_len=100, target_len=1000)
        busy = view(0, residents=[FakeRequest(9, 9, 100, 50_000)])
        assert strategy.select_context(req, [busy, view(1)], CONTEXT) == 1


class TestLengthLane:
    def test_a_long_request_is_confined_to_the_fast_instances(self):
        strategy = build_strategy("length_lane", n_fast=1, threshold=20_000, c_fast=2)
        req = FakeRequest(rid=0, group_id=0, prompt_len=100, target_len=30_000)
        assert narrow_candidates(strategy, req, [view(0), view(1), view(2)], CONTEXT) == [0]

    def test_a_short_request_is_kept_out_of_the_fast_instances(self):
        strategy = build_strategy("length_lane", n_fast=1, threshold=20_000, c_fast=2)
        req = FakeRequest(rid=0, group_id=0, prompt_len=100, target_len=1_000)
        assert narrow_candidates(strategy, req, [view(0), view(1), view(2)], CONTEXT) == [1, 2]

    def test_a_long_request_spills_to_bulk_when_every_fast_lane_is_full(self):
        full = view(0, residents=[FakeRequest(8, 8, 100, 30_000), FakeRequest(9, 9, 100, 30_000)])
        strategy = build_strategy("length_lane", n_fast=1, threshold=20_000, c_fast=2)
        req = FakeRequest(rid=0, group_id=0, prompt_len=100, target_len=30_000)
        assert narrow_candidates(strategy, req, [full, view(1), view(2)], CONTEXT) == [1, 2]

    def test_n_fast_must_leave_a_bulk_instance(self):
        import pytest

        with pytest.raises(ValueError, match="at least one bulk"):
            build_strategy("length_lane", n_fast=0, threshold=20_000, c_fast=2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_router_lane_strategies.py -v`
Expected: `KeyError: "Unknown routing strategy 'slack_lane'"`.

- [ ] **Step 3: Write the implementation**

Create `srl/component/rollout/router/strategies/slack_lane.py`:

```python
"""Route by a lane plan rebuilt at every dispatch burst.

Where `length_aware` balances load and `inventory_deadline` prices one placement at a time,
this strategy plans the whole fleet at once: `plan_lanes` decides which instance every
unplaced request goes to, and how many requests share it. Planning the fleet as a unit is
what lets a straggler end up alone on an instance, which no per-placement rule can arrange,
and a weight publish is what makes it affordable: psrl aborts every in-flight request and
hands them all back at once, so the plan is rebuilt on an empty fleet at no extra cost.

Requests already decoding when a mid-burst dispatch happens are pinned to their instance:
moving one outside a publish would discard KV the plan has no way to pay back.
"""

from __future__ import annotations

from .....forecast import LaneJob, LaneSlot, build_length_predictor, plan_lanes
from .....forecast.fleet import members_of, sequence_hint
from ...engine import InstanceView, Request, traj_view
from ..strategy import register_strategy


@register_strategy("slack_lane")
class SlackLane:
    """Plan the whole fleet's placement at each burst, thinning instances for late stragglers."""

    name = "slack_lane"
    python_engine_only = True

    def __init__(
        self,
        length_model: str = "oracle",
        length_model_args: dict | None = None,
        horizon_buffers: int = 2,
        c_max: int = 4,
        eps_frac: float = 0.01,
        max_moves: int = 8,
        enable_upgrade: bool = True,
    ) -> None:
        """
        Args:
            length_model (str): Name in `LENGTH_PREDICTORS`.
            length_model_args (dict | None): Constructor keywords for that predictor.
            horizon_buffers (int): Pending buffers the objective sums over.
            c_max (int): Requests a fast lane may hold.
            eps_frac (float): Fraction of one training cycle an upgrade must save to be kept.
            max_moves (int): Upgrades one plan may accept.
            enable_upgrade (bool): Whether to search for fast lanes at all. False leaves the
                balanced assignment untouched, which isolates what the lanes themselves buy
                from what planning the fleet as a unit buys.
        """
        if horizon_buffers < 1:
            raise ValueError(f"horizon_buffers must be >= 1, got {horizon_buffers!r}.")
        self.predictor = build_length_predictor(length_model, **(length_model_args or {}))
        self.horizon_buffers = int(horizon_buffers)
        self.c_max = int(c_max)
        self.eps_frac = float(eps_frac)
        self.max_moves = int(max_moves)
        self.enable_upgrade = bool(enable_upgrade)
        self._runtime = None
        self._plan = None

    def bind_runtime(self, runtime) -> None:
        self._runtime = runtime

    def begin_dispatch(self, t: float) -> None:
        """Rebuild the plan for this burst, or drop it when the fleet has nothing to decide."""
        self._plan = self._build_plan(t)

    def narrow_candidates(self, req: Request, candidates: list[InstanceView], _context) -> list[int]:
        """Confine a planned request to its instance, and leave an unplanned one unconstrained."""
        plan = self._plan
        if plan is None:
            return list(range(len(candidates)))
        wanted = plan.assignment.get(req.rid)
        if wanted is None:
            # Not in the plan: a request enqueued after `begin_dispatch` priced the burst. It
            # may go anywhere the plan did not reserve for a fast lane.
            return [i for i, view in enumerate(candidates) if view.instance_id not in plan.fast_instances] or list(
                range(len(candidates))
            )
        matches = [i for i, view in enumerate(candidates) if view.instance_id == wanted]
        return matches or list(range(len(candidates)))

    def select_context(self, req: Request, candidates: list[InstanceView], _context) -> int | None:
        """Take the planned instance, else the one with the fewest predicted tokens left."""
        plan = self._plan
        if plan is not None:
            wanted = plan.assignment.get(req.rid)
            for index, view in enumerate(candidates):
                if view.instance_id == wanted:
                    return index
        return min(range(len(candidates)), key=lambda i: (self._remaining_load(candidates[i]), candidates[i].n_active))

    def _remaining_load(self, view: InstanceView) -> float:
        total = 0.0
        for traj in view.residents():
            total += max(0.0, self.predictor.predict_total(traj) - traj.generated)
        return total

    def _build_plan(self, t: float):
        runtime = self._runtime
        if runtime is None:
            return None
        instances = runtime.instances
        if len(instances) < 2:
            return None
        members = members_of(runtime)
        if not members:
            return None

        pinned: dict[int, int] = {}
        jobs: list[LaneJob] = []
        seen: set[int] = set()
        for inst in instances:
            for req in (*inst.waiting, *inst.running):
                jobs.append(self._job(req))
                pinned[req.rid] = inst.instance_id
                seen.add(req.rid)
        for group in runtime.queued().values():
            for req in group:
                if req.rid not in seen:
                    jobs.append(self._job(req))
                    seen.add(req.rid)
        if not jobs:
            return None

        finished = {rid: done for group in runtime.pending.values() for rid, done in group.items()}
        lengths = {job.rid: job.prompt_len + job.generated + job.remaining for job in jobs}
        hint = sequence_hint(
            {group_id: 0.0 for group_id in members},
            members,
            lengths,
            runtime.batch_size,
            runtime.group_size,
        )
        slots = [
            LaneSlot(
                instance_id=inst.instance_id,
                cost=runtime.cost_model(inst.tp),
                block_size=inst.block_size,
                kv_blocks=inst.kv_blocks,
                token_budget=runtime.token_budget,
                max_concurrency=None,
                tp=inst.tp,
            )
            for inst in instances
        ]
        return plan_lanes(
            jobs,
            slots,
            snapshot=runtime.inventory(),
            train=runtime.train_state(hint),
            members=members,
            finished=finished,
            now=t,
            pinned=pinned,
            horizon_buffers=self.horizon_buffers,
            c_max=self.c_max,
            eps_frac=self.eps_frac,
            max_moves=self.max_moves if self.enable_upgrade else 0,
        )

    def _job(self, req: Request) -> LaneJob:
        traj = traj_view(req)
        return LaneJob(
            rid=req.rid,
            group_id=req.group_id,
            prompt_key=traj.prompt_key,
            prompt_len=req.prompt_len,
            generated=req.generated,
            remaining=max(1, round(self.predictor.remaining(traj))),
        )
```

`runtime.token_budget` does not exist on `SchedulingRuntime`. Add it as a read-only property next to `weight_load` in `srl/simulator/runtime.py`:

```python
    @property
    def token_budget(self) -> int:
        return self._sim.cfg.token_budget
```

Create `srl/component/rollout/router/strategies/length_lane.py`:

```python
"""Confine long requests to a fixed set of instances behind a length threshold.

The static ablation for `slack_lane`. The fast set is the first `n_fast` instance ids, fixed
for the whole run, and membership is a request's own predicted total length against one
threshold. Nothing is replanned and nothing is priced, so a result matching `slack_lane`
means the planning and the slack signal bought nothing a fixed threshold does not.
"""

from __future__ import annotations

from .....forecast.length import build_length_predictor
from ...engine import InstanceView, Request, traj_view
from ..strategy import register_strategy


@register_strategy("length_lane")
class LengthLane:
    """Static fast instances for requests predicted to be long."""

    name = "length_lane"

    def __init__(
        self,
        n_fast: int,
        threshold: float,
        c_fast: int,
        length_model: str = "oracle",
        length_model_args: dict | None = None,
    ) -> None:
        """
        Args:
            n_fast (int): Instances reserved for long requests, taken as instance ids
                `0 .. n_fast - 1`.
            threshold (float): A request predicted to need at least this many total tokens
                prefers the fast instances.
            c_fast (int): Requests one fast instance may hold before it is treated as full.
            length_model (str): Name in `LENGTH_PREDICTORS`.
            length_model_args (dict | None): Constructor keywords for that predictor.

        Raises:
            ValueError: If `n_fast` is below 1, or `c_fast` is below 1.
        """
        if n_fast < 1:
            raise ValueError(f"n_fast must be at least 1, and must still leave at least one bulk instance, got {n_fast!r}.")
        if c_fast < 1:
            raise ValueError(f"c_fast must be at least 1, got {c_fast!r}.")
        self.n_fast = int(n_fast)
        self.threshold = float(threshold)
        self.c_fast = int(c_fast)
        self.predictor = build_length_predictor(length_model, **(length_model_args or {}))

    def narrow_candidates(self, req: Request, candidates: list[InstanceView], _context) -> list[int]:
        fast = [i for i, view in enumerate(candidates) if view.instance_id < self.n_fast]
        bulk = [i for i, view in enumerate(candidates) if view.instance_id >= self.n_fast]
        if not fast or not bulk:
            raise ValueError(
                f"length_lane needs at least one bulk instance beyond its {self.n_fast} fast ones, "
                f"and the fleet offered {len(candidates)} candidates."
            )
        if self.predictor.predict_total(traj_view(req)) < self.threshold:
            return bulk
        room = [i for i in fast if len(candidates[i].residents()) < self.c_fast]
        return room or bulk

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        return min(
            range(len(candidates)),
            key=lambda i: (self._remaining_load(candidates[i]), candidates[i].n_active),
        )

    def _remaining_load(self, view: InstanceView) -> float:
        total = 0.0
        for traj in view.residents():
            total += max(0.0, self.predictor.predict_total(traj) - traj.generated)
        return total
```

`test_n_fast_must_leave_a_bulk_instance` builds the strategy with `n_fast=0`, so the `n_fast < 1` guard must carry the words "at least one bulk" — it does.

Register both in `srl/component/rollout/router/strategies/__init__.py`, keeping the list alphabetical:

```python
from . import (  # noqa: F401
    inventory_deadline,
    kv_cache_aware,
    length_aware,
    length_lane,
    prefix_aware,
    random_choice,
    request_num_balance,
    round_robin,
    slack_lane,
    throughput_balance,
    tier_threshold,
)
```

In `srl/config/schema.py`, extend the `strategy` docstring's name list to read:

```
    """request_num_balance | round_robin | random | kv_cache_aware |
    prefix_aware | throughput_balance | length_aware | inventory_deadline |
    slack_lane | length_lane.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python -m pytest tests/test_router_lane_strategies.py -v
python -m pytest tests/test_config.py -q
```
Expected: the new file all passes; `test_config.py` still passes.

- [ ] **Step 5: Commit**

```bash
git add srl/component/rollout/router/strategies/slack_lane.py \
        srl/component/rollout/router/strategies/length_lane.py \
        srl/component/rollout/router/strategies/__init__.py \
        srl/config/schema.py srl/simulator/runtime.py \
        tests/test_router_lane_strategies.py
git commit -m "feat(router): add slack_lane and length_lane strategies

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Wire the strategies into a real run and confirm they take effect

**Files:**
- Test: `tests/test_router_lane_strategies.py` (append an end-to-end class)

**Interfaces:**
- Consumes: both strategies (Task 4); `srl.simulator.simulator.Simulator`, `srl.simulator.config.SimConfig`, `srl.policy.psrl.PSRLPolicy`, `srl.cost.rollout.UnifiedRoofline`, `srl.workload.grouped.GroupedWorkload`.
- Produces: nothing new. This task is the wiring gate: a treatment arm byte-identical to its control is a bug, and this test is what catches it before the sweep.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_router_lane_strategies.py`:

```python
class TestLaneStrategiesChangeARealRun:
    """Both strategies must move a real simulated run off the default router's result."""

    def build(self, strategy: str, strategy_args: dict):
        import random

        from srl.component.sync.cost import FixedSyncCost
        from srl.cost.rollout import UnifiedRoofline
        from srl.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel
        from srl.policy.psrl import PSRLPolicy
        from srl.simulator.config import SimConfig
        from srl.simulator.simulator import Simulator
        from srl.workload.grouped import GroupedWorkload

        rng = random.Random(7)
        groups = []
        for group_id in range(24):
            # One straggler per four groups, so a fast lane has something to isolate.
            base = 24_000 if group_id % 4 == 0 else rng.randint(800, 2_500)
            groups.append((120, [base if member == 0 else rng.randint(800, 2_500) for member in range(4)]))
        cfg = SimConfig(
            n_versions=4,
            batch_size=6,
            group_size=4,
            n_instances=4,
            kv_blocks=20_000,
            block_size=16,
            token_budget=8192,
            max_concurrency=64,
            weight_load_time=0.0,
            log_telemetry=False,
        )
        cfg.router.strategy = strategy
        cfg.router.strategy_args = dict(strategy_args)
        return Simulator(
            cfg=cfg,
            workload=GroupedWorkload(groups),
            policy=PSRLPolicy(),
            rollout_cost=UnifiedRoofline(F=0.006, W=0.002, G=4e-6, A_p=3e-10, A_d=5e-8, b=16),
            training_cost=TrainingCostModel(FixedTrainingLatency(40.0), FixedTrainingMemory(0)),
            sync_cost=FixedSyncCost(0.0),
        )

    def test_slack_lane_is_not_the_default_router(self):
        base = self.build("request_num_balance", {}).run()
        lane = self.build("slack_lane", {}).run()
        assert lane.sim_time != base.sim_time
        assert lane.versions_done == base.versions_done == 4

    def test_slack_lane_without_upgrades_differs_from_slack_lane_with_them(self):
        nolane = self.build("slack_lane", {"enable_upgrade": False}).run()
        full = self.build("slack_lane", {"enable_upgrade": True}).run()
        assert nolane.sim_time != full.sim_time

    def test_length_lane_is_not_the_default_router(self):
        base = self.build("request_num_balance", {}).run()
        lane = self.build("length_lane", {"n_fast": 1, "threshold": 20_000, "c_fast": 2}).run()
        assert lane.sim_time != base.sim_time
        assert lane.versions_done == base.versions_done == 4
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_router_lane_strategies.py -k ChangeARealRun -v`
Expected: fail. Either an exception from the strategy's first real `begin_dispatch` (the likely outcome, since Tasks 1-4 were only unit-tested), or an assertion that the spans are equal.

- [ ] **Step 3: Fix whatever the real run exposes**

Debug against the real simulator, not the unit fixtures. The likely failures, in the order they will appear:

1. `SimConfig` field names differ from the ones used above. Read `srl/simulator/config.py` and correct the constructor call; do not change `SimConfig`.
2. `runtime.inventory()` raises because the policy is not psrl. `slack_lane` is psrl-only by construction. Guard `_build_plan` so a non-psrl policy returns `None` rather than raising, and document that `slack_lane` degrades to least-loaded routing off psrl.
3. `sequence_hint` returns an empty tuple when no group has an ETA yet, and `runtime.train_state(())` then prices an empty step. Fall back to the mean predicted length over `jobs` when the hint is empty.
4. `plan_lanes` prices far too often and the test is slow. Confirm with `-durations`; if one `begin_dispatch` exceeds about 2 s, lower `max_moves` in the test rather than weakening the planner.

Fix the cause in `srl/` where the cause is in `srl/`. Keep the test's assertions as written.

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python -m pytest tests/test_router_lane_strategies.py -v --durations=5
python -m pytest tests/ -q -x
```
Expected: the new classes pass, and the full suite still passes. Record the full-suite pass count in the commit body.

- [ ] **Step 5: Commit**

```bash
git add srl/ tests/test_router_lane_strategies.py
git commit -m "test(router): gate both lane strategies against a real psrl run

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Cadence diagnosis and the contention-free bound

**Files:**
- Create: `srl/diagnosis/cadence.py`
- Test: `tests/test_diagnosis_cadence.py`

**Interfaces:**
- Consumes: `logs/train.jsonl` and `logs/rollout_step.jsonl` written by `srl.telemetry`; `psrl.workers.ps.staleness.{InventorySnapshot, BufferView, forecast_ready_times}`.
- Produces:
  - `CadenceReport(span: float, train_busy: float, trainer_idle: float, fleet_idle_frac: float, intervals: tuple[float, ...], acf1: float, acf2: float)`
  - `cadence_report(log_dir: Path, n_instances: int) -> CadenceReport`
  - `span_contention_free(groups: Sequence[tuple[int, Sequence[int]]], *, staleness: int, batch_size: int, n_versions: int, solo_latency: Callable[[int, int], float], train_latency: Callable[[Sequence[int]], float], weight_load: float = 0.0) -> float`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diagnosis_cadence.py`:

```python
"""The cadence report's idle split and autocorrelation, and the contention-free bound."""

from __future__ import annotations

import json
import math

import pytest

from srl.diagnosis.cadence import cadence_report, span_contention_free


def write_logs(tmp_path, steps, rollout_rows):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    with (log_dir / "train.jsonl").open("w") as handle:
        for version, (start, end) in enumerate(steps, start=1):
            handle.write(json.dumps({"t": end, "start": start, "end": end, "dt": end - start, "version": version}) + "\n")
    with (log_dir / "rollout_step.jsonl").open("w") as handle:
        for instance_id, dt in rollout_rows:
            handle.write(json.dumps({"t": dt, "instance_id": instance_id, "dt": dt}) + "\n")
    return log_dir


class TestCadenceReport:
    def test_trainer_idle_is_the_gap_before_every_step(self, tmp_path):
        log_dir = write_logs(tmp_path, [(10.0, 20.0), (25.0, 35.0)], [(0, 30.0)])
        report = cadence_report(log_dir, n_instances=1)
        assert report.span == 35.0
        assert report.train_busy == 20.0
        assert report.trainer_idle == pytest.approx(15.0)

    def test_fleet_idle_is_one_minus_the_mean_busy_fraction(self, tmp_path):
        log_dir = write_logs(tmp_path, [(0.0, 100.0)], [(0, 80.0), (1, 60.0)])
        report = cadence_report(log_dir, n_instances=2)
        assert report.fleet_idle_frac == pytest.approx(0.3)

    def test_intervals_are_the_gaps_between_consecutive_step_ends(self, tmp_path):
        log_dir = write_logs(tmp_path, [(0.0, 10.0), (10.0, 25.0), (30.0, 45.0)], [(0, 45.0)])
        report = cadence_report(log_dir, n_instances=1)
        assert report.intervals == (15.0, 20.0)

    def test_a_period_two_series_has_a_negative_lag_one_autocorrelation(self, tmp_path):
        ends = [0.0]
        for index in range(12):
            ends.append(ends[-1] + (10.0 if index % 2 == 0 else 30.0))
        steps = [(ends[i], ends[i + 1]) for i in range(len(ends) - 1)]
        log_dir = write_logs(tmp_path, steps, [(0, ends[-1])])
        report = cadence_report(log_dir, n_instances=1)
        assert report.acf1 < -0.5
        assert report.acf2 > 0.5

    def test_a_constant_series_has_no_defined_autocorrelation(self, tmp_path):
        steps = [(10.0 * i, 10.0 * i + 5.0) for i in range(6)]
        log_dir = write_logs(tmp_path, steps, [(0, 60.0)])
        report = cadence_report(log_dir, n_instances=1)
        assert math.isnan(report.acf1)


class TestSpanContentionFree:
    def solo(self, prompt_len: int, output_len: int) -> float:
        return 0.001 * output_len

    def train(self, lengths) -> float:
        return 0.0001 * sum(lengths)

    def test_one_step_is_bounded_by_its_slowest_group(self, tmp_path):
        groups = [(100, [1_000, 2_000]), (100, [500, 400])]
        span = span_contention_free(
            groups,
            staleness=0,
            batch_size=2,
            n_versions=1,
            solo_latency=self.solo,
            train_latency=self.train,
        )
        # The slowest member needs 2.0 s, then the step trains on 3900 tokens plus 2 prompts.
        assert span == pytest.approx(2.0 + self.train([1_100, 2_100, 600, 500]))

    def test_staleness_lets_a_straggler_slip_to_a_later_buffer(self):
        groups = [(100, [50_000]), (100, [100]), (100, [100]), (100, [100])]
        tight = span_contention_free(
            groups, staleness=0, batch_size=2, n_versions=2,
            solo_latency=self.solo, train_latency=self.train,
        )
        loose = span_contention_free(
            groups, staleness=1, batch_size=2, n_versions=2,
            solo_latency=self.solo, train_latency=self.train,
        )
        assert loose < tight

    def test_too_few_groups_for_the_requested_steps_is_rejected(self):
        with pytest.raises(ValueError, match="needs at least"):
            span_contention_free(
                [(100, [100])], staleness=0, batch_size=2, n_versions=1,
                solo_latency=self.solo, train_latency=self.train,
            )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diagnosis_cadence.py -v`
Expected: `ModuleNotFoundError: No module named 'srl.diagnosis.cadence'`.

- [ ] **Step 3: Write the implementation**

Create `srl/diagnosis/cadence.py`:

```python
"""Where an async run's wall time goes, and how little of it is unavoidable.

Two questions, both answered from a finished run rather than from a model. `cadence_report`
splits the span into training, trainer idle, and fleet idle, and reports the step-interval
series with its first two autocorrelations, which is what decides whether a run oscillates
with a period at all. `span_contention_free` removes every capacity limit and leaves only the
two constraints no scheduler can relax: a request cannot finish faster than it decodes alone,
and a training step cannot start before psrl's own reserve and occupy rules fill its buffer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from psrl.workers.ps.staleness import BufferView, InventorySnapshot, forecast_ready_times


@dataclass(frozen=True, slots=True)
class CadenceReport:
    """One run's wall-time split and its step cadence."""

    span: float
    train_busy: float
    trainer_idle: float
    """Span the trainer spent waiting for a buffer, including the wait before the first step."""
    fleet_idle_frac: float
    """One minus the mean per-instance busy fraction over the span."""
    intervals: tuple[float, ...]
    """Gaps between consecutive training-step end times."""
    acf1: float
    acf2: float
    """Lag-1 and lag-2 autocorrelation of `intervals`, `nan` when the series cannot support one."""


def cadence_report(log_dir: Path, n_instances: int) -> CadenceReport:
    """
    Build the cadence report from one run's telemetry.

    Args:
        log_dir (Path): The run's `logs/` directory, holding `train.jsonl` and
            `rollout_step.jsonl`.
        n_instances (int): Instances the run used, the denominator of the busy fraction.

    Raises:
        ValueError: If `train.jsonl` holds no step, or `n_instances` is below 1.
    """
    if n_instances < 1:
        raise ValueError(f"n_instances must be at least 1, got {n_instances!r}.")
    steps = [json.loads(line) for line in (log_dir / "train.jsonl").read_text().splitlines() if line]
    if not steps:
        raise ValueError(f"{log_dir / 'train.jsonl'} holds no training step.")
    steps.sort(key=lambda row: row["start"])

    span = steps[-1]["end"]
    train_busy = sum(row["end"] - row["start"] for row in steps)
    idle = 0.0
    previous = 0.0
    for row in steps:
        idle += max(0.0, row["start"] - previous)
        previous = row["end"]

    busy: dict[int, float] = {}
    rollout_path = log_dir / "rollout_step.jsonl"
    if rollout_path.exists():
        with rollout_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                busy[row["instance_id"]] = busy.get(row["instance_id"], 0.0) + row["dt"]
    mean_busy = sum(busy.values()) / n_instances / span if span > 0 else 0.0

    ends = [row["end"] for row in steps]
    intervals = tuple(ends[i + 1] - ends[i] for i in range(len(ends) - 1))
    return CadenceReport(
        span=span,
        train_busy=train_busy,
        trainer_idle=idle,
        fleet_idle_frac=1.0 - mean_busy,
        intervals=intervals,
        acf1=_autocorrelation(intervals, 1),
        acf2=_autocorrelation(intervals, 2),
    )


def _autocorrelation(series: Sequence[float], lag: int) -> float:
    """Sample autocorrelation at `lag`, `nan` when the series is too short or has no variance."""
    n = len(series)
    if lag < 1 or n <= lag + 1:
        return math.nan
    mean = sum(series) / n
    variance = sum((x - mean) ** 2 for x in series)
    if variance <= 0.0:
        return math.nan
    covariance = sum((series[i] - mean) * (series[i + lag] - mean) for i in range(n - lag))
    return covariance / variance


def span_contention_free(
    groups: Sequence[tuple[int, Sequence[int]]],
    *,
    staleness: int,
    batch_size: int,
    n_versions: int,
    solo_latency: Callable[[int, int], float],
    train_latency: Callable[[Sequence[int]], float],
    weight_load: float = 0.0,
) -> float:
    """
    Lower-bound the span of `n_versions` steps when no request ever shares an instance.

    Every group is admitted the moment psrl's window allows it and runs at its alone-on-one-GPU
    latency, so the only things left holding the run back are the longest member of each group
    and the training chain psrl's reserve and occupy rules impose on top. The bound ignores
    fleet capacity entirely: large headroom against it is necessary for a scheduler to win and
    never sufficient.

    Args:
        groups (Sequence[tuple[int, Sequence[int]]]): `(prompt_len, output_lens)` per group, in
            the order the workload stages them.
        staleness (int): psrl's staleness tolerance.
        batch_size (int): Groups consumed per training step.
        n_versions (int): Steps to bound.
        solo_latency (Callable[[int, int], float]): Seconds for one request of
            `(prompt_len, output_len)` alone on an instance.
        train_latency (Callable[[Sequence[int]], float]): Seconds to train on the given
            per-response sequence lengths.
        weight_load (float): Delay between a publish and the fleet resuming.

    Returns:
        float: The earliest the `n_versions`-th step could end.

    Raises:
        ValueError: If there are fewer groups than `(n_versions + staleness) * batch_size`.
    """
    needed = (n_versions + staleness) * batch_size
    if len(groups) < needed:
        raise ValueError(f"The bound needs at least {needed} groups for {n_versions} steps, got {len(groups)}.")

    next_group = 0
    train_end: float | None = None
    now = 0.0
    eta: dict[int, float] = {}
    admitted: dict[int, tuple[int, Sequence[int]]] = {}
    occupied: dict[int, int] = {}
    reserved: dict[int, list[tuple[int, int]]] = {}
    consumed: set[int] = set()

    for version in range(n_versions):
        resume = now if train_end is None else train_end + weight_load
        while next_group < len(groups) and sum(
            len(members) for members in reserved.values()
        ) + sum(occupied.values()) < (staleness + 1) * batch_size:
            prompt_len, output_lens = groups[next_group]
            admitted[next_group] = (prompt_len, output_lens)
            eta[next_group] = resume + max(solo_latency(prompt_len, length) for length in output_lens)
            reserved.setdefault(min(version + staleness, version + staleness), []).append((next_group, version))
            next_group += 1

        snapshot = InventorySnapshot(
            staleness=staleness,
            num_entries=batch_size,
            next_buffer_id=version + staleness + 1,
            buffers=tuple(
                BufferView(
                    buffer_id=buffer_id,
                    ready_num_entries=batch_size,
                    n_occupied=occupied.get(buffer_id, 0),
                    reserved=tuple(reserved.get(buffer_id, ())),
                )
                for buffer_id in range(version, version + staleness + 1)
            ),
        )
        ready = forecast_ready_times(snapshot, eta)
        f_k = ready.get(version, math.inf)
        if not math.isfinite(f_k):
            raise ValueError(f"Buffer {version} can never fill from the groups supplied.")

        lengths = []
        for group_id in sorted(eta, key=lambda g: eta[g]):
            if group_id in consumed:
                continue
            prompt_len, output_lens = admitted[group_id]
            lengths.extend(prompt_len + length for length in output_lens)
            consumed.add(group_id)
            if len(consumed) % batch_size == 0:
                break
        start = max(f_k, now if train_end is None else train_end + weight_load)
        train_end = start + train_latency(lengths)
        now = train_end
        occupied = {}
        reserved = {buffer_id: members for buffer_id, members in reserved.items() if buffer_id > version}

    return float(train_end)
```

The `span_contention_free` body above is the shape to aim for, and its inventory bookkeeping is the part most likely to be wrong on the first attempt. Treat the three tests as the specification: `test_one_step_is_bounded_by_its_slowest_group` pins the single-step arithmetic exactly, `test_staleness_lets_a_straggler_slip_to_a_later_buffer` pins the direction staleness must move the bound, and `test_too_few_groups_for_the_requested_steps_is_rejected` pins the guard. Iterate on the implementation until all three pass without changing their assertions. If replaying psrl's reserve rule per step proves unwieldy, replace the per-step snapshot construction with one call to `forecast_ready_times` over a snapshot holding every admitted group, and derive each step's start from the returned READY times — the tests do not constrain which of the two shapes is used.

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python -m pytest tests/test_diagnosis_cadence.py -v
python -m pytest tests/test_diagnosis_bounds.py tests/test_diagnosis_staleness.py -q
```
Expected: the new file all passes; the existing diagnosis suites still pass.

- [ ] **Step 5: Commit**

```bash
git add srl/diagnosis/cadence.py tests/test_diagnosis_cadence.py
git commit -m "feat(diagnosis): add cadence report and contention-free span bound

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Sweep driver and one smoked point

**Files:**
- Create: `workflow/validate/experiments/cadence_lanes.py` (gitignored by `.gitignore:279`)
- Read first: `workflow/validate/experiments/smoke.py`, `workflow/validate/reference/simulate.md`, `workflow/validate/reference/gates.md`, `workflow/validate/reference/reading-results.md`, `workflow/validate/memory/lessons.md`

**Interfaces:**
- Consumes: `workflow/validate/scripts/{collect,gates,replayspec}.py`; both strategies (Tasks 4-5); `cadence_report` and `span_contention_free` (Task 6).
- Produces: a module exposing `POINTS`, `ARMS`, `build(spec, output_dir=None) -> (Simulator, int)`, `calibrate_t_sat(seed) -> float`, and `main() -> int`, writing `outputs/simulate/cadence_lanes/summary.json`.

- [ ] **Step 1: Set up the workspace and read the contracts**

Run:
```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
export SIMRL_WORKSPACE=/apdcephfs_zwfy/share_303760348/lhy
python workflow/validate/scripts/preflight.py
```
Expected: exit 0. `SIMRL_WORKSPACE` points at the directory **above** the repo, the one holding `env/simrl.sh`; pointing it at the repo is the documented way to make preflight fail. Quote any warning it prints in the Task 8 report.

Then read the four reference files and `memory/lessons.md` listed above. They define the gate thresholds, the `simulate.sh` env contract, and the result fields the driver must emit.

- [ ] **Step 2: Write the driver**

Create `workflow/validate/experiments/cadence_lanes.py`. It must do exactly this, and the shape to follow for the harness plumbing (`collect.measure`, `collect.compare_rows`, `replayspec.attach`, `gates.evaluate_point`, `gates.summarize`) is `workflow/validate/experiments/smoke.py`'s `main`:

1. **Permuted traces.** For `seed in (0, 1, 2)`, read `outputs/workload/qwen3_8b_dapo_38912_tp1/joint.jsonl`, keep its first line (the `_meta` header) as is, shuffle the group lines with `random.Random(seed)`, and write `outputs/workload/cadence_lanes_seed{seed}/joint.jsonl`. Seed 0 must be file order, unshuffled. Skip writing when the file already exists.
2. **`calibrate_t_sat(seed)`.** Run one rollout-bound arm per seed: the same fleet, `training_cost=linear` with `time_per_token=1e-9`, `policy.staleness=2`, `strategy="request_num_balance"`, `n_versions=8`. Return `consumed_tokens / span`. Cache the value in `outputs/simulate/cadence_lanes/t_sat.json` keyed by seed, so a resumed sweep does not recompute it.
3. **`POINTS`.** The 27 points of `staleness in (1, 2, 3)` x `rho in (0.9, 1.0, 1.1)` x `seed in (0, 1, 2)`, each carrying `n_versions=24`, `batch=32`, `group_size=8`, `n_instances=16`, `max_output_length=38912`, `token_budget=16384`, and `time_per_token = rho / t_sat(seed)`.
4. **`ARMS`.** Exactly these keys, so the report's arm names are stable:
   - `b0_default`: `strategy="request_num_balance"`, `cap=0`.
   - `b1_cap16`, `b1_cap24`, `b1_cap32`, `b1_cap48`: `strategy="request_num_balance"` with `rollout.router.max_concurrent_seqs_per_instance` at 16/24/32/48.
   - `a_nolane`: `strategy="slack_lane"`, `strategy_args={"enable_upgrade": False}`.
   - `a_slack`: `strategy="slack_lane"`, `strategy_args={}`.
   - `a_slack_cap`: `strategy="slack_lane"`, `strategy_args={}`, `cap` = the per-point winning `b1_*` cap.
   - `c_k{k}_L{L}_c{c}` for `k in (1, 2, 4)`, `L in (20000, 30000)`, `c in (2, 4)`: `strategy="length_lane"`, `strategy_args={"n_fast": k, "threshold": L, "c_fast": c}`.
5. **`build(spec, output_dir=None)`.** Construct the `Simulator` in process. Drive the same `Simulator` `simulate.sh` would, with the composed cost model off `outputs/profile/qwen3_8b/kernel_tp1/kernel`, `MODEL_PATH=/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B`, KV from the probe, and `weight_load_time=0.0`. When `output_dir` is set, attach `SimTelemetry(JsonlStore(output_dir, overwrite=True))` so Task 9 has streams to replay.
6. **Baseline selection.** `b1_best` per point is the `b1_cap*` arm with the smallest span **among those that passed the gates**. Speedup is reported against `b1_best` as primary and `b0_default` as secondary. For the `c_*` family report both the per-point best arm (labelled in-sample) and the single arm with the best geometric mean across all points (labelled the honest number).
7. **Gates.** `gates.evaluate_point(row, control, n_versions=24, batch_size=32)` for every arm against `b0_default`, plus these, enforced in the driver:
   - `versions_done == 24`;
   - `livelocked is False`;
   - `consumed_groups == 24 * 32`;
   - `tokens_per_group` within 1% of `b0_default` at the same point;
   - `max_staleness_true <= point["staleness"]`.

   Record `preemptions` and total `reprefill_tokens` on every row without gating on them.
8. **Bound.** Per point, compute `LB_roll = consumed_tokens(b0) / t_sat(seed)` and `LB_phys = span_contention_free(...)` with `solo_latency` from the composed TP1 cost model and `train_latency` from the point's own `time_per_token`. Store `LB = max(LB_roll, LB_phys)` and, per arm, `headroom_capture = (span_b1_best - span_arm) / (span_b1_best - LB)`.
9. **Output.** Write `outputs/simulate/cadence_lanes/summary.json` with `{"rows": [...], "points": {...}, "t_sat": {...}, "summary": gates.summarize(verdicts)}`. Every row carries `row_id` from `replayspec.attach` so Task 9 can replay it.

- [ ] **Step 3: Smoke exactly one point**

Run one point, short, both arms:
```bash
python workflow/validate/experiments/cadence_lanes.py --smoke --staleness 2 --rho 1.0 --seed 0 --n-versions 6
```
Confirm all four:
1. `b0_default` reproduces the default router (span within 0.5% of a `request_num_balance` run at the same settings — run it twice to confirm determinism first);
2. `a_slack` differs from `b0_default`;
3. `a_slack` differs from `a_nolane`;
4. every gate verdict prints, and `consumed_groups == 6 * 32` on both arms.

If 2 or 3 fails, the strategy is not wired and Task 5's gate missed it. Fix that before sweeping. If `t_sat` calibration takes more than about 10 minutes per seed, lower its `n_versions` to 6 and record the change.

- [ ] **Step 4: Verify the smoke output**

Run: `python -c "import json; d=json.load(open('outputs/simulate/cadence_lanes/summary.json')); print(json.dumps(d['summary'], indent=1)); [print(r['arm'], round(r['span'],1), r['consumed_groups'], r['preemptions']) for r in d['rows']]"`
Expected: one row per smoked arm, equal `consumed_groups`, and distinct spans for `b0_default` / `a_nolane` / `a_slack`.

- [ ] **Step 5: Commit**

The driver is gitignored, so commit only what moved in `srl/`:
```bash
git status --short srl/ tests/
git add -u srl/ tests/
git commit -m "fix: whatever the smoke run exposed in srl/

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```
If `git status` shows nothing, skip the commit and record that the smoke run needed no `srl/` change.

---

### Task 8: Run the sweep and gate it

**Files:**
- Write: `outputs/simulate/cadence_lanes/summary.json`, plots under `outputs/simulate/cadence_lanes/figures/`
- Write: `workflow/validate/reports/2026-10-26-cadence-lanes.md` (from `workflow/validate/reference/report.md`)
- Append: `workflow/validate/memory/lessons.md` (process lessons only, no result numbers)

**Interfaces:**
- Consumes: the driver from Task 7.
- Produces: the gated result table, the geometric means, and the per-point `headroom_capture` that Task 10's decision rule reads.

- [ ] **Step 1: Delegate the sweep to the validate subagent**

Dispatch the `validate` subagent with `model: "haiku"` for the mechanical middle only — smoke, sweep, gates, `results.json`, plots. Do not delegate script-authoring (done in Task 7) or report-writing (Step 3 below) to a Haiku-backed call; that is where a wrong workload shape or a silently dropped failing point slips through.

The prompt must carry, verbatim:
- `SIMRL_WORKSPACE=/apdcephfs_zwfy/share_303760348/lhy` and the env activation line;
- the driver path `workflow/validate/experiments/cadence_lanes.py` and that it is already written and smoked;
- the 27 points, the arm list, and the five extra gates from Task 7 step 7;
- "Report `b1_best` as the primary baseline and `b0_default` as secondary";
- "Drop every point that fails a gate and list it with its value. Do not explain a failing gate away.";
- "Do not tune any knob on the points you report over.";
- the output paths above.

- [ ] **Step 2: Verify the subagent's result set yourself**

Do not take the subagent's summary on trust. Run:
```bash
python workflow/validate/scripts/gates.py outputs/simulate/cadence_lanes/summary.json
python -c "
import json, statistics as st
d=json.load(open('outputs/simulate/cadence_lanes/summary.json'))
rows=[r for r in d['rows'] if r.get('gated')]
print('gated rows', len(rows), 'of', len(d['rows']))
for arm in ('a_nolane','a_slack','a_slack_cap'):
    s=[r['speedup_vs_b1'] for r in rows if r['arm']==arm]
    if s: print(arm, 'n', len(s), 'geomean', round(st.geometric_mean(s),4), 'min', round(min(s),4), 'max', round(max(s),4))
"
```
Confirm by hand: the gated row count, that no arm's row set is larger than 27, and that every dropped point appears in the report's failure list with its value.

- [ ] **Step 3: Write the report yourself**

Fill `workflow/validate/reports/2026-10-26-cadence-lanes.md` from `workflow/validate/reference/report.md`. It must state:
- the gated table, led by `a_slack` vs `b1_best`;
- the geometric mean per arm, with the point count each is over;
- the concurrency and fast-lane count `a_slack` chose for itself, over time;
- `headroom_capture` per arm: median, min, max;
- the step-interval autocorrelations, and whether a period-2 oscillation is present — this is the explicit test of the user's prior, and a negative answer is a result;
- every gate that failed, with the point and the value;
- a "Not established" section naming at least: whether any result holds off the oracle predictor, whether it holds under KV pressure tight enough to preempt, and whether it holds on a real agentic trace rather than this single math-task trace.

- [ ] **Step 4: Append process lessons**

Append to `workflow/validate/memory/lessons.md` under its own convention: process lessons only, no result numbers.

- [ ] **Step 5: Commit**

```bash
git add workflow/validate/reports/2026-10-26-cadence-lanes.md workflow/validate/memory/lessons.md
git commit -m "docs(validate): report the cadence-lane sweep

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```
`outputs/` is gitignored, so the result set and figures stay on disk only.

---

### Task 9: Explain the win or the loss

**Files:**
- Write: `outputs/analysis/cadence_lanes/<row>/` per selected row
- Write: `workflow/analysis/reports/2026-10-26-cadence-lanes.md`

**Interfaces:**
- Consumes: `outputs/simulate/cadence_lanes/summary.json` (Task 8); `cadence_report` (Task 6); `workflow/analysis/scripts/{preflight,select,replay,diagnose}.py`.
- Produces: the idle-time diff between control and treatment that explains the Task 8 verdict.

- [ ] **Step 1: Delegate to the analysis subagent**

Dispatch the `analysis` subagent. Its contract forbids it from claiming a speedup; it explains one. The prompt must carry:
- `SIMRL_WORKSPACE=/apdcephfs_zwfy/share_303760348/lhy` and the env activation line;
- `outputs/simulate/cadence_lanes/summary.json` as the result set;
- the rows to diagnose: for the three points with the **largest** `a_slack` speedup and the three with the **smallest**, both the `b1_best` row and the `a_slack` row, so every point has a control and a treatment to diff (6 points x 2 rows = 12 rows; if that exceeds the unit's 6-row budget, take 2 largest and 1 smallest);
- "For each point, diff the control's and the treatment's idle decomposition: which component of idle time moved, and by how much.";
- "Also run `srl.diagnosis.cadence.cadence_report` on both rows' log directories and report the trainer-idle and fleet-idle split side by side.";
- "Rank hypotheses by share of total idle time. Name the `validate` experiment that would falsify each.";
- the fidelity gate is non-negotiable: any False means stop and report the mismatch.

- [ ] **Step 2: Read the diagnosis and reconcile it with Task 8**

Check that the diagnosis and the sweep agree about direction. Specifically: if `a_slack` won, the diff must show trainer idle falling; if it lost, it must show which idle component rose instead. A diagnosis that cannot account for the sign of the Task 8 result means one of the two is wrong — say so rather than picking the convenient one.

- [ ] **Step 3: Record the mechanism**

Write the reconciliation into `workflow/analysis/reports/2026-10-26-cadence-lanes.md` (from `workflow/analysis/reference/report.md`), and append mechanism lessons to `workflow/analysis/memory/lessons.md` under its own rules.

- [ ] **Step 4: Commit**

```bash
git add workflow/analysis/reports/2026-10-26-cadence-lanes.md workflow/analysis/memory/lessons.md
git commit -m "docs(analysis): explain the cadence-lane sweep's idle-time split

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Decide on Algorithm B, and build it only if the rule fires

**Files:**
- Read: `outputs/simulate/cadence_lanes/summary.json`
- Conditionally modify: `srl/forecast/lanes.py`, `srl/component/rollout/router/strategies/slack_lane.py`, `tests/test_forecast_lanes.py`

**Interfaces:**
- Consumes: `headroom_capture` per point (Task 8).
- Produces: either a recorded decision not to build B, or `plan_cadence(...)` plus a `cadence_target` flag on `SlackLane`.

- [ ] **Step 1: Evaluate the decision rule**

Run:
```bash
python -c "
import json, statistics as st
d=json.load(open('outputs/simulate/cadence_lanes/summary.json'))
rows=[r for r in d['rows'] if r.get('gated') and r['arm']=='a_slack']
live=[r for r in rows if (r['span_b1_best']-r['lb'])>=0.05*r['span_b1_best']]
h=[r['headroom_capture'] for r in live]
print('points with >=5% headroom:', len(live), 'of', len(rows))
print('median headroom capture:', round(st.median(h),3) if h else None)
print('BUILD B' if h and st.median(h)<0.5 else 'DO NOT BUILD B')
"
```

The rule, from the spec: build B if the median `headroom_capture` for `a_slack` is **below 0.5** over the points where `span_b1_best - LB >= 0.05 * span_b1_best`.

- [ ] **Step 2: If the rule says do not build, record that and stop**

Append the decision, with the two numbers it rests on, to `workflow/validate/reports/2026-10-26-cadence-lanes.md` under a "Algorithm B decision" heading. Then write one memory file per the memory instructions — `type: project`, naming the mechanism, the objective, the measured direction, and the regime it was measured in — and add its one-line pointer to `MEMORY.md`. Commit the report change. **Task 10 ends here.**

- [ ] **Step 3: If the rule says build, add `plan_cadence`**

Only reached when Step 1 printed `BUILD B`. Add to `srl/forecast/lanes.py`:

```python
def plan_cadence(
    jobs: Sequence[LaneJob],
    slots: Sequence[LaneSlot],
    *,
    snapshot,
    train: TrainState,
    members: Mapping[int, Sequence[int]],
    finished: Mapping[int, float],
    now: float,
    pinned: Mapping[int, int] | None = None,
    horizon_buffers: int = 2,
    c_max: int = 4,
    eps_frac: float = 0.01,
    max_moves: int = 8,
    bisect_steps: int = 8,
) -> LanePlan:
    """
    Plan for the smallest uniform step period the fleet can actually hold.

    `plan_lanes` improves the nearest late buffer first, so it will spend its move budget on
    buffer k and leave k+1 late. This asks the opposite question: what is the smallest period
    `P` such that spacing every deadline `P` apart leaves a plan that meets all of them? The
    answer is found by bisection, and the plan that achieved it is returned.

    Args:
        bisect_steps (int): Bisection iterations between `train.cycle` and the bulk plan's own
            mean period. Eight brings the bracket inside 0.4% of its width.

    Returns:
        LanePlan: The plan for the smallest feasible period, or the `plan_lanes` result when no
            period in the bracket is feasible.
    """
```

Implement it as: bisect `P` in `[train.cycle, bulk_mean_period]`; for each `P`, build deadlines `D_k = target_0 + (k + 1) * P`, run the same move set as `plan_lanes` but accepting a move when it brings every buffer in the horizon under its `D_k` rather than when it lowers `J`; keep the smallest feasible `P`. Add a `cadence_target: bool = False` flag to `SlackLane` that routes `_build_plan` to `plan_cadence`.

Write its tests first, in `tests/test_forecast_lanes.py`, mirroring `TestPlanLanes`: a balanced fleet needs no upgrade; a feasible tighter period is found when one exists; `plan_cadence` never returns a worse objective than `plan_lanes` on the same input; `bisect_steps=0` falls back to `plan_lanes` exactly.

- [ ] **Step 4: Measure B as one more arm**

Re-run Task 8's sweep with `b_cadence` added to `ARMS` (`strategy="slack_lane"`, `strategy_args={"cadence_target": True}`), through the `validate` subagent, on the same 27 points. Report `b_cadence` against `b1_best` and against `a_slack`.

- [ ] **Step 5: Commit**

```bash
git add srl/forecast/lanes.py srl/component/rollout/router/strategies/slack_lane.py \
        tests/test_forecast_lanes.py workflow/validate/reports/2026-10-26-cadence-lanes.md
git commit -m "feat(forecast): add cadence-target lane planning

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** Every section of the spec maps to a task:

| Spec section | Task |
|---|---|
| Algorithm A step 1 (bulk-only plan) | 1 (`lpt_assign`), 3 (`plan_lanes` entry) |
| Algorithm A step 2 (price the plan, `J = sum E_k`) | 2 (`price`) |
| Algorithm A step 3 (greedy upgrade, `c_max`, `eps_frac`, `max_moves`) | 3 |
| Algorithm A step 4 (`narrow_candidates`, `select_context`, fast lane reverts) | 4 |
| A-nolane | 4 (`enable_upgrade=False`), 7 (`a_nolane` arm) |
| A+cap | 7 (`a_slack_cap` arm) |
| Algorithm C, 12 arms, best-per-point and best-overall | 4 (`length_lane`), 7 (`c_*` arms, step 6) |
| Algorithm B, conditional | 10 |
| `LB_roll`, `LB_phys`, headroom capture | 6 (`span_contention_free`), 7 (step 8) |
| Decision rule for B | 10 (step 1) |
| Experiment fleet, workload, 3 permutations, K=24, linear training, `T_sat` | 7 (steps 2.1-2.3) |
| 9 regimes x 3 seeds = 27 points | 7 (step 2.3) |
| Baselines B0, B1 | 7 (`ARMS`) |
| All five gates | 7 (step 7), 8 (step 2) |
| Metrics incl. step-interval autocorrelation, fast-lane count | 6, 8 (step 3) |
| Code layout (`lanes.py`, both strategies, `cadence.py`, registry, schema, tests, driver) | 1-7 |
| Out of scope (hetero TP, early admission, KV migration, prediction error, warmup) | Global Constraints |

**Two deliberate deviations from the spec, both narrowing:**
1. The spec's step 1 says requests that do not fit "stay queued, in priority order". The plan instead assigns every job and caps each instance's **forecast** concurrency at `kv_safe_concurrency`, because the engine's own per-instance waiting queue already implements holding back, and the `A+cap` arm supplies the explicit hold-back lever through the existing uniform cap. No `held` list is built. This is recorded here so the Task 8 report can state it.
2. The spec's `pick_queued` ordering by group slack is dropped. `narrow_candidates` plus `select_context` already force the plan's assignment, and `inventory_deadline`'s own `pick_queued` is the complexity this plan is trying not to repeat. If the Task 9 diagnosis shows queue order is what binds, that is the next experiment, not a silent addition here.

**Placeholder scan.** No "TBD", no "add error handling", no "similar to Task N". Task 5 step 3 and Task 6 step 3 deliberately describe *debugging* against named tests rather than showing final code, because both are places where the first implementation will be wrong against the real engine; in both cases the tests are given in full and are the specification.

**Type consistency.** Checked across tasks: `LaneJob`/`LaneSlot`/`LanePlan`/`Pricing` field names match between Tasks 1, 2, 3, and 4. `plan_lanes`'s keyword is `members` (not `members_of`, which is the imported *function* from `srl.forecast.fleet`) in Tasks 2, 3, 4, and 10. `price(...)` takes `horizon_buffers` as a keyword in Tasks 2, 3, and 10. `kv_safe_concurrency(jobs, slot)` is positional in Tasks 1 and 2. `cadence_report(log_dir, n_instances)` is positional in Tasks 6, 8, and 9. `span_contention_free` is keyword-only after `groups` in Tasks 6 and 7. `SlackLane`'s flags are `enable_upgrade` (Task 4) and `cadence_target` (Task 10); `LengthLane`'s are `n_fast`, `threshold`, `c_fast` (Tasks 4 and 7, matching the `c_k{k}_L{L}_c{c}` arm names).

---

## Execution Handoff

Plan complete and saved to `/apdcephfs_zwfy/share_303760348/lhy/SimRL/docs/superpowers/plans/2026-10-26-cadence-lane-scheduling.md`.
