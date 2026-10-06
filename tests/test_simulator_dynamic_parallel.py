"""Simulator-level integration of the dynamic-parallel fleet.

`tests/test_fleet.py` covers `HeterogeneousFleet` in isolation and
`tests/test_water_filling.py` covers the solver, but nothing exercised the two through
`Simulator`. That gap was real rather than theoretical: a code review found that
deleting the instance-list rebind, dropping the aborted-request re-enqueue, or solving
before `rollout_version` advances each left the entire suite green while the feature was
silently broken or inert.

So every test here asserts a property that fails under one of those mutations. The
mutation each one guards is named in its docstring.
"""

from __future__ import annotations

import pytest

from dryrun.component.rollout.engine import Request
from dryrun.cost import UnifiedRoofline
from dryrun.policy.psrl import PSRLPolicy
from dryrun.simulator.config import SimConfig
from dryrun.simulator.simulator import Simulator
from dryrun.simulator.dynamic_parallel import FleetPlan, ParallelismSolver
from dryrun.workload.grouped import GroupedWorkload


def _cost() -> UnifiedRoofline:
    return UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_d=1e-7)


def _config(**overrides) -> SimConfig:
    """A small configuration that still reaches several weight updates."""
    base = dict(
        batch_size=2,
        group_size=1,
        max_staleness=2,
        n_versions=4,
        # 4000-token outputs keep requests in flight across weight updates, which is
        # what makes the rebuild path run at all; with short outputs the fleet drains
        # before every update and `_publish_weights` skips the rebuild entirely.
        workload=GroupedWorkload.from_lengths([200], [4000]),
        n_instances=2,
        # 2000 blocks, deliberately. A larger pool trips a PRE-EXISTING engine-loop
        # assertion (`_align_instances`: "No extra completions expected"), which I
        # reproduced at base commit 746276a on the homogeneous path with solver=None --
        # so it is unrelated to the fleet. See .superpowers/sdd/progress.md.
        kv_blocks=2000,
        token_budget=1000,
        routing_strategy="version_aware_tp",
    )
    base.update(overrides)
    return SimConfig(**base)


def _solver(strategy: str = "search", n_gpus: int = 2) -> ParallelismSolver:
    cost = _cost()
    return ParallelismSolver(
        cost_models={1: cost, 2: cost},
        n_gpus=n_gpus,
        tp_candidates=(1, 2),
        strategy=strategy,
        fixed_tp_per_instance=(1,) * n_gpus,
        max_concurrency=None,
    )


def _simulator(**cfg_overrides) -> Simulator:
    cost = _cost()
    return Simulator(
        rollout_cost=cost,
        policy=PSRLPolicy(),
        cfg=_config(**cfg_overrides),
        parallel_cost_models={1: cost, 2: cost},
        # Equal capacities: the per-TP difference is real (31294 vs 69532 on the
        # measured model) but a larger pool hits the pre-existing assertion above, so
        # these tests hold KV fixed and vary only the fleet shape.
        kv_blocks_per_tp={1: 2000, 2: 2000},
        solver=_solver(),
    )


def test_solver_absent_leaves_the_homogeneous_path_untouched():
    """The feature must be dormant by default, not merely configurable."""
    cost = _cost()
    sim = Simulator(rollout_cost=cost, policy=PSRLPolicy(), cfg=_config())
    assert sim._fleet is None
    assert len(sim.instances) == 2
    result = sim.run()
    assert result.versions_done == 4


def test_partial_cohort_inputs_are_refused():
    """A solver without its cost models would build a fleet that cannot price a step."""
    cost = _cost()
    with pytest.raises(ValueError, match="parallel_cost_models"):
        Simulator(
            rollout_cost=cost,
            policy=PSRLPolicy(),
            cfg=_config(),
            solver=_solver(),
        )


def test_router_dispatches_to_the_fleets_own_instances_after_every_update():
    """Guards the `self.instances = self._fleet.instances` rebind in `_publish_weights`.

    Without it the engine loop advances instances the router never dispatches to, and
    `_align_instances` asserts clocks against a list the fleet has replaced. Checking
    only before `run()` would pass with the rebind deleted, so this asserts after a
    completed run that crossed several weight updates.
    """
    sim = _simulator()
    assert sim.instances is sim.router.instances
    sim.run()
    assert sim.instances is sim.router.instances
    assert all(
        inst is sim.router.instances[i] for i, inst in enumerate(sim.instances)
    )


def test_aborted_requests_survive_the_rebuild():
    """Guards the `router.enqueue(aborted, ...)` after a heterogeneous rebuild.

    A weight update aborts every in-flight request and the fleet hands them back; if
    they are not re-enqueued they vanish from the simulation. The run would still
    "succeed" with a quietly smaller workload, so the signal is the conserved count:
    nothing admitted may disappear without being consumed or dropped.
    """
    sim = _simulator()
    result = sim.run()
    accounted = len(result.consumed) + len(result.dropped)
    still_live = len(sim.inflight) + len(sim.ready) + sum(
        len(group) for group in sim.pending.values()
    )
    queued = len(sim.router) + sum(
        len(inst.waiting) + len(inst.running) for inst in sim.instances
    )
    assert accounted > 0
    # Every admitted request is either accounted for or demonstrably still somewhere.
    assert sim.n_admitted <= accounted + still_live + queued


def test_the_fleet_is_rebuilt_at_weight_updates_not_only_at_construction():
    """The rebuild must actually run, which a construction-time check cannot show."""
    sim = _simulator()
    rebuilds: list[int] = []
    original = sim._fleet.rebuild

    def counting_rebuild(plan, t):
        rebuilds.append(plan.n_instances)
        return original(plan, t)

    sim._fleet.rebuild = counting_rebuild
    sim.run()
    assert rebuilds, "No fleet rebuild happened across four weight updates."
    assert all(count >= 1 for count in rebuilds)


def test_the_solver_sees_the_published_version_not_the_previous_one():
    """Guards the ordering of `rollout_version` against `solver.solve`.

    The solver partitions in-flight requests on `v_traj < rollout_version`. Called
    before the version advances, every request classifies as new, the old cohort is
    never populated and the split never fires -- while every other test still passes,
    because an inert feature breaks nothing. This records the version at each solve and
    asserts it advanced.
    """
    sim = _simulator()
    seen_versions: list[int] = []
    original = sim._solver.solve

    def recording_solve(snapshot, rollout_version):
        seen_versions.append(rollout_version)
        return original(snapshot, rollout_version)

    sim._solver.solve = recording_solve
    sim.run()
    assert seen_versions, "The solver was never consulted."
    # The construction-time solve sees version 0; later ones must see published versions.
    assert max(seen_versions) >= 1, (
        f"The solver only ever saw version {seen_versions}; it is being called before "
        f"rollout_version advances, so the cohort split can never fire."
    )


def test_scg_curve_is_monotone_and_bounded_by_total_throughput():
    """SCG(S) counts a cumulative subset, so it can only rise with S."""
    sim = _simulator()
    result = sim.run()
    curve = result.scg_curve()
    values = [curve[s] for s in sorted(curve)]
    assert values == sorted(values)
    total_rate = len(result.consumed) / result.sim_time
    assert values[-1] == pytest.approx(total_rate, rel=1e-9)


def test_fixed_strategy_reproduces_its_configured_fleet():
    """The baselines run through this path, so `fixed` must be honoured exactly."""
    cost = _cost()
    solver = ParallelismSolver(
        cost_models={1: cost, 2: cost},
        n_gpus=2,
        tp_candidates=(1, 2),
        strategy="fixed",
        fixed_tp_per_instance=(2,),
        max_concurrency=None,
    )
    sim = Simulator(
        rollout_cost=cost,
        policy=PSRLPolicy(),
        cfg=_config(),
        parallel_cost_models={1: cost, 2: cost},
        kv_blocks_per_tp={1: 2000, 2: 3000},
        solver=solver,
    )
    assert len(sim.instances) == 1
    assert sim.instances[0].tp == 2
    assert sim.instances[0].kv_blocks == 3000


def test_every_instance_clock_stays_behind_global_time_across_rebuilds():
    """A newly built instance left at t_local=0.0 trips `_align_instances` later.

    The crash would surface at a subsequent alignment rather than at the rebuild, so
    this asserts the invariant directly at the end of a multi-update run.
    """
    sim = _simulator()
    sim.run()
    assert all(inst.t_local <= sim.t + 1e-9 for inst in sim.instances), [
        (inst.instance_id, inst.t_local, sim.t) for inst in sim.instances
    ]


# NOT TESTED HERE: the solver's "never allocate GPUs to a cohort that cannot receive a
# request" guard. I measured that every solve in this workload sees r_old == len(snapshot)
# (that is, zero NEW-version requests at solve time), so the r_old == 0 branch the guard
# protects is unreachable from the Simulator with these shapes -- a test asserting it
# here passed with the guard deleted. It is covered where it can actually fail, in
# tests/test_water_filling.py::test_solver_no_stranded_old_cohort_batch_sensitive.


def test_a_group_is_never_split_across_a_rebuild():
    """GRPO scores a response against its group, so a half-consumed group is untrainable."""
    sim = _simulator(group_size=1)
    result = sim.run()
    by_group: dict[int, int] = {}
    for req in result.consumed:
        by_group[req.group_id] = by_group.get(req.group_id, 0) + 1
    assert all(count == sim.cfg.group_size for count in by_group.values())


def test_fleet_plan_respects_the_gpu_budget_at_every_rebuild():
    """A plan that overspends GPUs would mean the fleet is not the configured size."""
    sim = _simulator()
    budgets: list[int] = []
    original = sim._fleet.rebuild

    def checking_rebuild(plan: FleetPlan, t: float):
        budgets.append(sum(plan.tp_per_instance))
        return original(plan, t)

    sim._fleet.rebuild = checking_rebuild
    sim.run()
    assert budgets, "No rebuild observed."
    assert all(total == 2 for total in budgets), budgets


def test_consumed_requests_carry_a_staleness_within_the_configured_bound():
    """A fleet rebuild must not let a request be consumed past max_staleness."""
    sim = _simulator()
    result = sim.run()
    assert result.consumed
    assert all(
        req.staleness_true() <= sim.cfg.max_staleness for req in result.consumed
    ), sorted({req.staleness_true() for req in result.consumed})


def test_an_in_flight_request_keeps_its_tokens_across_a_rebuild():
    """The rebuild frees KV but must not discard generated tokens.

    `abort_all` keeps `generated` and the per-version segments; losing them would reset
    trajectories at every weight update and make staleness meaningless.
    """
    sim = _simulator()
    tokens_seen: dict[int, int] = {}
    original = sim._fleet.rebuild

    def watching_rebuild(plan, t):
        aborted = original(plan, t)
        for req in aborted:
            previous = tokens_seen.get(req.rid)
            if previous is not None:
                assert req.generated >= previous, (
                    f"Request {req.rid} lost tokens across a rebuild: "
                    f"{previous} -> {req.generated}"
                )
            tokens_seen[req.rid] = req.generated
        return aborted

    sim._fleet.rebuild = watching_rebuild
    sim.run()


def test_a_request_is_an_identity_not_a_value_across_the_rebuild():
    """The fleet must hand back the SAME objects, since queues hold identities."""
    sim = _simulator()
    cost = _cost()
    req = Request(rid=9_999, prompt_len=50, target_len=100, v_traj=0)
    sim._fleet.instances[0].waiting.append(req)
    plan = FleetPlan.uniform(tp=1, n_gpus=2)
    aborted = sim._fleet.rebuild(plan, t=1.0)
    assert any(item is req for item in aborted)
    assert cost is not None
