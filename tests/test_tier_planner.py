"""Length-aware tier planner: the drain model, the three solvers, the estimator, the router.

Each test names what it would catch if the code were wrong. The cost model is an
analytic stand-in, so none of this needs kernel tables or a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from dryrun.component.rollout.engine import InstanceView, Request
from dryrun.component.rollout.router import build_strategy
from dryrun.component.rollout.router.estimator import LengthEstimator
from dryrun.simulator.dynamic_parallel import PlannerSolver, RichRequestSnapshot, TierPlanner
from dryrun.simulator.dynamic_parallel.planner import StepCurve

KV = {1: 500_000, 2: 1_100_000, 4: 2_300_000}


class _Cost:
    """Decode step time with a TP-shrinking weight read and a fixed floor.

    Per step: a floor that does not shrink with TP, a weight read split across the
    ranks, and a KV read split across them too. The same three terms the real
    model's TP trade-off comes from.
    """

    def __init__(self, tp: int) -> None:
        self.tp = tp

    def step_time(self, n_tokens, t2, ctxsum, n_reqs=0, prefill_ctx=0.0) -> float:
        floor = 0.0012 + 0.0007 * self.tp
        weights = 0.0042 / self.tp
        kv = 3.0e-9 * float(ctxsum) / self.tp
        gemm = 1.5e-5 * float(n_tokens) / self.tp
        return floor + weights + kv + gemm


def _planner(algorithm: str = "pp", n_gpus: int = 8, tps=(1, 2, 4), **kw) -> TierPlanner:
    return TierPlanner(
        {tp: _Cost(tp) for tp in tps},
        {tp: KV[tp] for tp in tps},
        n_gpus,
        tps,
        algorithm=algorithm,
        **kw,
    )


def _two_class(n_short: int = 508, n_long: int = 4, short: int = 4096, long: int = 32768):
    rem = np.array([float(long)] * n_long + [float(short)] * n_short)
    return np.full(len(rem), 64.0), rem


# --- the drain model ---------------------------------------------------------


def test_drain_of_one_request_is_its_length_times_the_step_time():
    """Catches a drain model that charges the wrong number of steps or batch size."""
    curve = StepCurve(_Cost(1), 1, KV[1])
    seconds = curve.drain(np.array([64.0]), np.array([1000.0]))
    step = float(curve.tau(1, 64.0 + 500.0))
    assert seconds == pytest.approx(1000 * step, rel=0.02)


def test_adding_a_resident_never_makes_the_instance_finish_sooner():
    """Catches a drain model where more batch-mates look free or helpful."""
    curve = StepCurve(_Cost(1), 1, KV[1])
    ctx = np.full(40, 64.0)
    base = curve.drain(ctx[:20], np.full(20, 2000.0))
    more = curve.drain(ctx, np.full(40, 2000.0))
    assert more >= base


def test_a_cap_below_the_offered_load_lifts_the_drain_time():
    """Catches a cap that is ignored: 200 requests on a cap of 10 cannot finish at batch 200."""
    curve = StepCurve(_Cost(1), 1, KV[1])
    ctx, rem = np.full(200, 64.0), np.full(200, 1000.0)
    assert curve.drain(ctx, rem, cap=10) > 3.0 * curve.drain(ctx, rem, cap=None)


def test_the_event_simulation_agrees_with_the_closed_form_when_nothing_binds():
    """Catches an event loop that mis-advances ctx or mis-times a completion."""
    curve = StepCurve(_Cost(2), 2, KV[2])
    rng = np.random.default_rng(1)
    ctx, rem = np.full(24, 64.0), np.sort(rng.integers(500, 4000, 24).astype(float))[::-1]
    assert curve.simulate(ctx, rem, cap=1024) == pytest.approx(curve.drain(ctx, rem), rel=0.08)


def test_slack_relaxes_a_request_that_may_take_extra_cycles():
    """Catches deadline weighting that is applied backwards."""
    curve = StepCurve(_Cost(1), 1, KV[1])
    ctx, rem = np.full(8, 64.0), np.linspace(1000, 8000, 8)
    assert curve.drain(ctx, rem, slack=np.full(8, 1.0)) < curve.drain(ctx, rem)


# --- the three solvers ---------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["pp", "twopool", "dp"])
def test_every_solver_spends_the_gpu_budget_exactly(algorithm):
    """Catches a plan that overspends or leaves GPUs idle, which would change the fleet size."""
    ctx, rem = _two_class()
    plan = _planner(algorithm).plan(ctx, rem)
    assert sum(plan.tp_per_instance) == 8
    assert sum(len(t.instances) * t.tp for t in plan.tiers) <= 8


@pytest.mark.parametrize("algorithm", ["pp", "twopool"])
def test_the_longest_requests_go_to_the_highest_tp(algorithm):
    """Catches tiers listed in the wrong order, which would put stragglers on the slow shape.

    Holds by construction for the two solvers that enumerate TP groups in descending
    order. The DP is free to choose any TP for any range, so it is checked below.
    """
    ctx, rem = _two_class()
    plan = _planner(algorithm).plan(ctx, rem)
    assert plan.is_heterogeneous
    assert plan.tiers[0].tp == max(t.tp for t in plan.tiers)
    assert plan.tiers[0].lo > plan.tiers[-1].lo


def test_the_dp_gives_stragglers_a_tier_of_their_own():
    """Catches a class partition that averages a straggler into the bulk.

    A class is priced at its mean length. Before classes were cut at sharp length
    drops, the four 32768-token requests shared a class with 4096-token ones and the
    DP planned for a workload with no tail at all.
    """
    ctx, rem = _two_class()
    plan = _planner("dp").plan(ctx, rem)
    assert plan.tiers[0].n_items == 4
    assert plan.tiers[0].lo > plan.tiers[-1].lo


@pytest.mark.parametrize("algorithm", ["pp", "twopool", "dp"])
def test_every_request_is_assigned_to_exactly_one_tier(algorithm):
    """Catches a partition that drops or duplicates a request."""
    ctx, rem = _two_class()
    plan = _planner(algorithm).plan(ctx, rem, ids=np.arange(len(rem)))
    assert sorted(plan.assignment) == list(range(len(rem)))
    assert sum(t.n_items for t in plan.tiers) == len(rem)


def test_a_two_class_workload_isolates_the_stragglers_on_the_high_tp_tier():
    """Catches a planner that fails to find the tail isolation the whole design rests on."""
    ctx, rem = _two_class()
    plan = _planner("twopool").plan(ctx, rem, ids=np.arange(len(rem)))
    long_ids = {i for i in range(4)}
    tail = {rid for rid, tier in plan.assignment.items() if tier == 0}
    assert long_ids <= tail
    assert len(tail) < 40


def test_a_uniform_workload_gets_no_heterogeneous_fleet_under_the_margin():
    """Catches a planner that chases a gain smaller than the model's own error."""
    rem = np.full(512, 4096.0)
    plan = _planner("pp", switch_margin=0.05).plan(np.full(512, 64.0), rem)
    assert not plan.is_heterogeneous


def test_the_switch_margin_only_ever_falls_back_to_a_uniform_fleet():
    """Catches a margin that picks a third, worse plan instead of the uniform one."""
    ctx, rem = _two_class(n_long=1)
    free = _planner("pp", switch_margin=0.0).plan(ctx, rem)
    guarded = _planner("pp", switch_margin=0.99).plan(ctx, rem)
    assert not guarded.is_heterogeneous
    assert guarded.predicted_makespan_s >= free.predicted_makespan_s * 0.999


def test_the_parametric_partition_is_no_worse_than_an_even_split():
    """Catches the contiguous partition returning something an even split would beat."""
    ctx, rem = _two_class()
    planner = _planner("pp")
    best = planner.plan(ctx, rem).predicted_makespan_s
    even = max(
        planner._tier_sim(1, 4, ctx[:256], rem[:256], 64, None),
        planner._tier_sim(4, 1, ctx[256:], rem[256:], 64, None),
    )
    assert best <= even


def test_an_empty_snapshot_yields_a_valid_plan():
    """Catches a crash on the first plan of a run, before any request exists."""
    plan = _planner().plan(np.zeros(0), np.zeros(0))
    assert sum(plan.tp_per_instance) == 8


def test_an_unknown_algorithm_is_refused():
    with pytest.raises(ValueError, match="Unknown planner algorithm"):
        _planner("bogus")


def test_a_missing_cost_model_is_refused():
    with pytest.raises(ValueError, match="No cost model"):
        TierPlanner({1: _Cost(1)}, {1: KV[1]}, 4, (1, 2))


# --- the solver adapter ----------------------------------------------------------


def test_the_adapter_turns_staleness_into_slack_and_keeps_ids():
    """Catches an adapter that drops the request id, which breaks the planner's own assignment."""
    solver = PlannerSolver(_planner("twopool"), max_staleness=1)
    snap = [RichRequestSnapshot(v_traj=1, ctx=64, remaining=int(r), rid=i) for i, r in enumerate(_two_class()[1])]
    plan = solver.solve(snap, rollout_version=1)
    assert solver.wants_pending is True
    assert set(plan.assignment) == set(range(len(snap)))


def test_an_adapter_snapshot_without_ids_still_plans():
    """Catches a crash when the estimator hands the planner anonymous requests."""
    solver = PlannerSolver(_planner("pp"), max_staleness=0)
    snap = [RichRequestSnapshot(v_traj=0, ctx=64, remaining=int(r)) for r in _two_class()[1]]
    assert solver.solve(snap, 0).assignment == {}


# --- the length estimator ---------------------------------------------------------


def _req(rid: int, target: int, generated: int = 0, prompt: int = 0) -> Request:
    r = Request(rid=rid, prompt_len=64, target_len=target, v_traj=0, source_group_id=prompt)
    r.generated = generated
    return r


def test_the_oracle_reads_the_true_remaining_length():
    assert LengthEstimator("oracle").remaining(_req(0, 5000, 1000)) == 4000.0


def test_survival_raises_the_estimate_of_a_request_that_outlived_the_bulk():
    """Catches a survival estimate that ignores how far a request has already run."""
    est = LengthEstimator("online", use_history=False)
    for i in range(200):
        est.observe(_req(i, 2000, 2000))
    for i in range(10):
        est.observe(_req(1000 + i, 20000, 20000))
    young, old = _req(5000, 0, 100), _req(5001, 0, 5000)
    assert est.remaining(old) > est.remaining(young) / 2
    assert est.remaining(old) > 5000


def test_prompt_history_predicts_the_next_round_of_the_same_prompt():
    """Catches history that is stored under the wrong key, which would make it useless."""
    est = LengthEstimator("online")
    est.observe(_req(0, 9000, 9000, prompt=7))
    assert est.remaining(_req(1, 0, 0, prompt=7)) == pytest.approx(9000.0)
    assert est.history_total(_req(2, 0, 0, prompt=8)) is None


def test_unknown_requests_are_filled_with_population_quantiles_not_a_constant():
    """Catches a planner that sees every unlabeled request as the same length."""
    est = LengthEstimator("online")
    for i in range(400):
        est.observe(_req(i, 1000 + 20 * i, 1000 + 20 * i, prompt=10_000 + i))
    fresh = [_req(9000 + i, 0, 0, prompt=i) for i in range(50)]
    filled = est.plan_remaining(fresh)
    assert filled.max() > 2 * filled.min()


def test_the_noisy_predictor_is_deterministic_per_request():
    """Catches noise that is redrawn at every query, which would make a request flip tiers."""
    est = LengthEstimator("noisy", noise_sigma=0.5, seed=3)
    r = _req(11, 6000)
    assert est.remaining(r) == est.remaining(r)


def test_a_missed_request_looks_average():
    est = LengthEstimator("noisy", noise_sigma=0.0, miss_rate=1.0, prior_total=4096.0)
    assert est.remaining(_req(5, 30000)) == pytest.approx(4096.0)


def test_an_unknown_estimator_mode_is_refused():
    with pytest.raises(ValueError, match="Unknown estimator mode"):
        LengthEstimator("psychic")


# --- the router strategy ----------------------------------------------------------


def _view(instance_id: int, tp: int, n_active: int = 0) -> InstanceView:
    return InstanceView(
        instance_id=instance_id, n_waiting=0, n_running=n_active, kv_utilization=0.0, total_ctx=0, tp=tp
    )


def _adopt(strategy, plan) -> None:
    strategy.update_plan(plan)


def test_a_request_goes_only_to_its_planned_tier_and_waits_when_it_is_full():
    """Catches spill across tiers, which would put a straggler into a throughput batch."""
    ctx, rem = _two_class()
    plan = _planner("twopool").plan(ctx, rem, ids=np.arange(len(rem)))
    strategy = build_strategy("planned_tier_tp")
    _adopt(strategy, plan)
    reqs = [_req(i, int(r)) for i, r in enumerate(rem)]
    high = [i for i, tp in enumerate(plan.tp_per_instance) if tp == max(plan.tp_per_instance)]
    views = [_view(i, tp) for i, tp in enumerate(plan.tp_per_instance)]
    assert set(strategy._allowed(reqs[0], views)) <= set(high)
    full = [_view(i, tp, n_active=10_000) for i, tp in enumerate(plan.tp_per_instance)]
    assert strategy._allowed(reqs[0], full) == []


def test_queued_requests_are_offered_longest_first_and_a_full_tier_never_blocks_the_rest():
    """Catches head-of-line blocking, the failure the whole queue scan exists to avoid."""
    ctx, rem = _two_class()
    plan = _planner("twopool").plan(ctx, rem, ids=np.arange(len(rem)))
    assert plan.is_heterogeneous
    strategy = build_strategy("planned_tier_tp")
    _adopt(strategy, plan)
    reqs = [_req(i, int(r)) for i, r in enumerate(rem)]
    window = [reqs[0], reqs[10], reqs[11]]
    free = [_view(i, tp) for i, tp in enumerate(plan.tp_per_instance)]
    assert strategy.pick_queued(window, free) is reqs[0]
    top = max(plan.tp_per_instance)
    tail_full = [_view(i, tp, n_active=10_000 if tp == top else 0) for i, tp in enumerate(plan.tp_per_instance)]
    chosen = strategy.pick_queued(window, tail_full)
    assert chosen is not None and chosen.target_len == 4096


def test_rescue_moves_a_request_that_has_outgrown_its_tier():
    """Catches a rescue that never fires, or one that moves a request with no progress."""
    ctx, rem = _two_class()
    plan = _planner("twopool").plan(ctx, rem, ids=np.arange(len(rem)))
    strategy = build_strategy("planned_tier_tp", length_source="noisy", rescue=True, rescue_min_generated=256)
    _adopt(strategy, plan)
    low = [i for i, tp in enumerate(plan.tp_per_instance) if tp == min(plan.tp_per_instance)][0]
    straggler = _req(900, 30000, generated=9000)
    fresh = _req(901, 30000, generated=0)
    view = InstanceView(
        instance_id=low,
        n_waiting=0,
        n_running=2,
        kv_utilization=0.0,
        total_ctx=0,
        tp=plan.tp_per_instance[low],
        decoding=lambda: [(None, straggler), (None, fresh)],
    )
    strategy.observe_fleet([_view(i, tp) for i, tp in enumerate(plan.tp_per_instance)])
    assert strategy.choose_pause(view, 0.0) is straggler


def test_without_rescue_the_strategy_never_pauses_anything():
    strategy = build_strategy("planned_tier_tp")
    assert strategy.choose_pause(_view(0, 1), 0.0) is None


def test_an_unknown_order_is_refused():
    with pytest.raises(ValueError, match="Unknown order"):
        build_strategy("planned_tier_tp", order="sideways")
