# tests/test_water_filling.py
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from dryrun.simulator.water_filling import (
    CohortPlan,
    FleetPlan,
    RequestSnapshot,
    WaterFillingSolver,
)


def _mock_cost_models(tp_set=(1, 2)):
    """Return mock ComposedRollout objects that report a step_time inversely proportional to TP."""
    models = {}
    for tp in tp_set:
        m = MagicMock()
        # step_time decreases with TP: TP=2 is ~1.5x faster per token at moderate batch
        m.step_time.side_effect = lambda n, t2, ctx, reqs, _tp=tp: 0.010 / _tp
        models[tp] = m
    return models


def _snap(v_traj: int, ctx: int = 5000) -> RequestSnapshot:
    return RequestSnapshot(v_traj=v_traj, ctx=ctx, instance_id=0)


def test_fleet_plan_budget_invariant():
    plan = FleetPlan(cohorts=(CohortPlan(tp=2, n_instances=1, max_version=0),
                               CohortPlan(tp=1, n_instances=2, max_version=None)),
                     n_gpus=4)
    assert sum(c.tp * c.n_instances for c in plan.cohorts) == 4


def test_fleet_plan_budget_violated():
    # tp=2,n=2 + tp=1,n=2 => 6 GPU-ranks, but n_gpus=4 => budget error.
    # Distinct max_version values ensure the budget check fires, not the catch-all check.
    with pytest.raises(ValueError, match="GPU budget"):
        FleetPlan(cohorts=(CohortPlan(tp=2, n_instances=2, max_version=0),
                            CohortPlan(tp=1, n_instances=2, max_version=None)),
                  n_gpus=4)


def test_fleet_plan_catchall_independent_of_budget():
    # Budget is correct (tp=2,n=1 + tp=1,n=2 = 4), but two catch-alls => routing error.
    # Message must NOT mention "GPU budget".
    with pytest.raises(ValueError) as exc_info:
        FleetPlan(cohorts=(CohortPlan(tp=2, n_instances=1, max_version=None),
                            CohortPlan(tp=1, n_instances=2, max_version=None)),
                  n_gpus=4)
    assert "GPU budget" not in str(exc_info.value)
    assert "catch-all" in str(exc_info.value)


def test_solver_fixed_strategy():
    models = _mock_cost_models()
    solver = WaterFillingSolver(
        cost_models=models,
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="fixed",
        fixed_cohorts=[{"tp": 1, "n_instances": 4}],
        max_concurrency=None,
    )
    snapshot = [_snap(v_traj=0) for _ in range(20)]
    plan = solver.solve(snapshot, rollout_version=1)
    assert len(plan.cohorts) == 1
    assert plan.cohorts[0].tp == 1
    assert plan.cohorts[0].n_instances == 4


def test_solver_fixed_two_cohorts():
    models = _mock_cost_models()
    solver = WaterFillingSolver(
        cost_models=models,
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="fixed",
        fixed_cohorts=[{"tp": 2, "n_instances": 2}],
        max_concurrency=None,
    )
    plan = solver.solve([], rollout_version=0)
    assert plan.cohorts[0].tp == 2
    assert plan.cohorts[0].n_instances == 2


def test_solver_search_degenerate_no_old():
    """When all requests are new-version, solver returns a single homogeneous cohort.

    NOTE: this test passes even without the r_old==0 guard because the mock's
    step_time (0.010/tp) is batch-insensitive: concentration buys nothing, so
    the degenerate plan collapses to the right answer by accident.  The test
    that actually guards the fix is
    ``test_solver_no_stranded_old_cohort_batch_sensitive`` below, which uses a
    batch-superlinear model where concentration genuinely outscores spreading.
    """
    models = _mock_cost_models()
    solver = WaterFillingSolver(
        cost_models=models,
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="search",
        fixed_cohorts=[],
        max_concurrency=None,
    )
    # All requests have v_traj == rollout_version, so R_old == 0
    snapshot = [_snap(v_traj=1) for _ in range(100)]
    plan = solver.solve(snapshot, rollout_version=1)
    assert len(plan.cohorts) == 1
    assert sum(c.tp * c.n_instances for c in plan.cohorts) == 4


def test_solver_search_produces_valid_budget():
    models = _mock_cost_models()
    solver = WaterFillingSolver(
        cost_models=models,
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="search",
        fixed_cohorts=[],
        max_concurrency=None,
    )
    snapshot = (
        [_snap(v_traj=0) for _ in range(8)]    # old-version (small cohort)
        + [_snap(v_traj=1) for _ in range(200)]  # new-version (large cohort)
    )
    plan = solver.solve(snapshot, rollout_version=1)
    assert sum(c.tp * c.n_instances for c in plan.cohorts) == 4


def test_solver_target_len_not_accessible():
    """RequestSnapshot must not have target_len; the firewall is structural."""
    snap = RequestSnapshot(v_traj=0, ctx=1000, instance_id=None)
    assert not hasattr(snap, "target_len")
    assert not hasattr(snap, "remaining")


def test_solver_search_ties_prefer_fewer_cohorts():
    """When two configurations score identically, prefer fewer cohorts (homogeneous)."""
    # Use a mock that scores all fleets equally — tie-break must favour 1 cohort
    models: dict = {}
    for tp in (1, 2):
        m = MagicMock()
        m.step_time.return_value = 0.010  # all fleets score the same
        models[tp] = m
    solver = WaterFillingSolver(
        cost_models=models,
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="search",
        fixed_cohorts=[],
        max_concurrency=None,
    )
    snapshot = [_snap(v_traj=0, ctx=100) for _ in range(50)]
    plan = solver.solve(snapshot, rollout_version=1)
    # Tie → prefer 1 cohort
    assert len(plan.cohorts) == 1


# ---------------------------------------------------------------------------
# Regression: n_old==0 must not undercount new-cohort load (bug fix)
# ---------------------------------------------------------------------------

def test_solver_search_n_old_zero_load_absorbed():
    """
    When n_old==0, the catch-all routes OLD requests into the new cohort.
    The scorer must include those requests in the new-cohort load; otherwise
    single-cohort plans are scored too cheaply and the solver incorrectly
    splits into two cohorts.

    Reproduction:
      r_old=10 (v_traj=0), r_new=90 (v_traj=1), rollout_version=1,
      n_gpus=4, tp_candidates=(1,2), step_time=0.010/tp.

    Without the fix the solver returns a 2-cohort split
    [(tp=2,n=1,max_version=0), (tp=2,n=1,max_version=None)] scoring 20000
    while single-cohort (tp=2,n=2) is mis-scored at 18000 instead of 20000.
    With the fix, the scores tie at 20000 and the tie-break picks 1 cohort.
    """
    models = _mock_cost_models(tp_set=(1, 2))
    solver = WaterFillingSolver(
        cost_models=models,
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="search",
        fixed_cohorts=[],
        max_concurrency=None,
    )
    snapshot = (
        [_snap(v_traj=0, ctx=5000) for _ in range(10)]   # old requests
        + [_snap(v_traj=1, ctx=5000) for _ in range(90)]  # new requests
    )
    plan = solver.solve(snapshot, rollout_version=1)
    # Must be a single cohort (catch-all absorbs old requests into new cohort)
    assert len(plan.cohorts) == 1, (
        f"Expected 1 cohort but got {len(plan.cohorts)}: {plan.cohorts}"
    )
    assert plan.cohorts[0].tp == 2
    assert plan.cohorts[0].n_instances == 2


# ---------------------------------------------------------------------------
# Scorer liveness: a dead _score must be caught
# ---------------------------------------------------------------------------

def test_solver_score_liveness_large_new_cohort():
    """
    With a cost model where step_time grows linearly with batch size
    (step_time = 0.002 * reqs / tp), spreading 400 requests across more
    replicas wins.  With n_gpus=4 and tp_candidates=(1,2):

      - 4×TP=1: each instance gets 100 reqs, step_time=0.2s, rate=100/0.2=500 tok/s
        per instance → 4×500 = 2000 total
      - 2×TP=2: each instance gets 200 reqs, step_time=0.2s, rate=200/0.2=1000 tok/s
        per instance → 2×1000 = 2000 total — tie at this batch scale

    Use step_time=0.002*reqs/tp so that 4×TP=1 gives per-instance rate
    100/(0.002*100/1)=500 and 2×TP=2 gives 200/(0.002*200/2)=1000.
    4×TP=1 total = 2000, 2×TP=2 total = 2000 — but the solver prefers
    1 cohort on a tie.  However with 0.001*reqs/tp the TP=1 rate is
    500/instance × 4 = 2000, TP=2 rate is 1000/instance × 2 = 2000.
    Let us pick a model that cleanly favours TP=1 by using a quadratic
    penalty on batch: step_time = 0.001 * reqs**2 / tp.
      - 4×TP=1: batch=100, step=0.001*10000/1=10s, rate=100/10=10 → total 40
      - 2×TP=2: batch=200, step=0.001*40000/2=20s, rate=200/20=10 → total 20

    So 4×TP=1 scores 40 > 2×TP=2 scores 20.  With _score returning 0.0
    (dead scorer) all plans tie and the tie-break picks 1 cohort with TP=2
    (enumerated first due to descending sort) — NOT the expected 4×TP=1.
    """
    models: dict = {}
    for tp in (1, 2):
        m = MagicMock()
        # quadratic batch penalty: spreading to more instances wins
        m.step_time.side_effect = lambda n, t2, ctx, reqs, _tp=tp: 0.001 * reqs**2 / _tp
        models[tp] = m
    solver = WaterFillingSolver(
        cost_models=models,
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="search",
        fixed_cohorts=[],
        max_concurrency=None,
    )
    snapshot = [_snap(v_traj=1, ctx=500) for _ in range(400)]
    plan = solver.solve(snapshot, rollout_version=1)
    # quadratic penalty on batch makes 4×TP=1 score higher than 2×TP=2
    assert len(plan.cohorts) == 1
    assert plan.cohorts[0].tp == 1
    assert plan.cohorts[0].n_instances == 4


# ---------------------------------------------------------------------------
# Minor: r < n_instances must not crash (clamp behaviour documented)
# ---------------------------------------------------------------------------

def test_solver_score_r_less_than_n_instances():
    """
    When there are fewer requests than instances (r < n_instances), _score
    clamps batch to max(1, r // n_instances) = 1.  This overestimates work
    for tiny cohorts (each idle instance is priced as if running 1 request),
    which is a known approximation.  The solver must still return a valid plan.
    """
    models = _mock_cost_models(tp_set=(1, 2))
    solver = WaterFillingSolver(
        cost_models=models,
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="search",
        fixed_cohorts=[],
        max_concurrency=None,
    )
    # Only 2 requests for a fleet that could hold 4 TP=1 instances
    snapshot = [_snap(v_traj=1, ctx=200) for _ in range(2)]
    plan = solver.solve(snapshot, rollout_version=1)
    # Must satisfy GPU budget (and not crash)
    assert sum(c.tp * c.n_instances for c in plan.cohorts) == 4


def test_absorbed_cohort_depth_is_request_weighted():
    """
    When the catch-all absorbs old requests, the new cohort's DEPTH must become
    the request-weighted mean of the two partitions, not the new partition's own.

    `test_solver_search_n_old_zero_load_absorbed` cannot catch this: it gives
    every request ctx=5000, so the weighted mean equals `d_new` and a trivial
    `d_new_eff = d_new` passes it. Attention cost is quadratic in sequence
    length, so pricing 10 trajectories at 50k context as if they were 100
    tokens understates the cohort by orders of magnitude -- exactly the class
    of silent mis-scoring the absorption fix exists to prevent.

    Asserts on the depth actually handed to the cost model rather than on the
    chosen plan, because the plan can coincide for unrelated reasons.
    """
    queried: list[tuple[int, float]] = []

    def _recording_models() -> dict[int, MagicMock]:
        models: dict[int, MagicMock] = {}
        for tp in (1, 2):
            model = MagicMock()

            def step_time(n_tokens, t2, ctxsum, n_reqs, _tp=tp):
                queried.append((n_reqs, ctxsum))
                return 0.010 / _tp

            model.step_time.side_effect = step_time
            models[tp] = model
        return models

    solver = WaterFillingSolver(
        cost_models=_recording_models(),
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="search",
        fixed_cohorts=[],
        max_concurrency=None,
    )
    # 10 deep + 90 shallow. Weighted mean = (50000*10 + 100*90) / 100 = 5090.
    snapshot = (
        [_snap(v_traj=0, ctx=50_000) for _ in range(10)]
        + [_snap(v_traj=1, ctx=100) for _ in range(90)]
    )
    solver.solve(snapshot, rollout_version=1)

    per_request_depths = {
        round(ctxsum / n_reqs) for n_reqs, ctxsum in queried if n_reqs > 0
    }
    assert 5090 in per_request_depths, (
        "The absorbed cohort was never priced at the request-weighted mean depth "
        f"5090; depths queried were {sorted(per_request_depths)}. A trivial "
        "d_new_eff = d_new would query 100 instead."
    )


# ---------------------------------------------------------------------------
# Regression: r_old==0 must not produce an old cohort (stranded GPUs)
# ---------------------------------------------------------------------------

def test_solver_no_stranded_old_cohort_batch_sensitive():
    """
    Regression for the stranded-GPU defect: when r_old==0, no cohort with a
    finite max_version may be emitted, regardless of how the cost model scores
    different batch sizes.

    Root cause: with a batch-superlinear cost model (step_time inversely
    proportional to reqs*tp), concentrating all requests on one instance gives
    a higher token rate than splitting them across two.  Before the fix, the
    solver would emit:

        [(tp=2, n=1, max_version=0), (tp=2, n=1, max_version=None)]

    The first cohort (max_version=0) can never receive a request because all
    requests have v_traj=1 == rollout_version; its 2 GPUs are stranded.  The
    solver chose it anyway because (score=72) > (score=36) for the two-instance
    alternative, paying for concentration with idle GPUs.

    The fix adds ``if n_old > 0 and r_old == 0: continue`` in _search_plan.

    Why NOT add the symmetric guard (n_new>0 and r_new==0):
        The new cohort is the catch-all (max_version=None).  When n_old==0 it
        also absorbs old requests; more importantly, new requests will arrive
        during the interval as the simulator admits work.  The catch-all is
        never stranded in the same structural sense as an old cohort whose
        version predicate can never be satisfied.

    This test intentionally uses a batch-sensitive (superlinear) mock.
    ``test_solver_search_degenerate_no_old`` uses a batch-insensitive mock and
    passes for the wrong reason; this test is the effective guard.
    """
    models: dict = {}
    for tp in (1, 2):
        m = MagicMock()
        # step_time = 1 / (reqs * tp): throughput-per-instance scales as reqs^2*tp,
        # so a larger batch on fewer instances always outscores smaller batches on
        # more instances.  This reproduces the real-model behaviour at small batch
        # sizes where GPU utilisation is memory-bandwidth-bound and larger batches
        # amortize per-step fixed costs.
        m.step_time.side_effect = lambda n, t2, ctx, reqs, _tp=tp: 1.0 / (reqs * _tp)
        models[tp] = m
    solver = WaterFillingSolver(
        cost_models=models,
        n_gpus=4,
        tp_candidates=(1, 2),
        strategy="search",
        fixed_cohorts=[],
        max_concurrency=None,
    )
    # r_old==0: every request is at the current rollout version.
    snapshot = [RequestSnapshot(v_traj=1, ctx=2000, instance_id=0) for _ in range(6)]
    plan = solver.solve(snapshot, rollout_version=1)

    # With r_old==0, every emitted cohort must be able to serve at least one
    # request.  The only cohort that can do so is the catch-all (max_version=None);
    # any finite max_version < rollout_version would match no request.
    for cohort in plan.cohorts:
        assert cohort.max_version is None, (
            f"Stranded cohort emitted: max_version={cohort.max_version!r} but "
            f"r_old==0 so no request satisfies v_traj<={cohort.max_version}.  "
            f"Full plan: {plan.cohorts}"
        )
    assert sum(c.tp * c.n_instances for c in plan.cohorts) == 4
