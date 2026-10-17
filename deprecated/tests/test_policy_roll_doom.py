"""Tests for `RollDoomPolicy`: abandoning trajectories that cannot meet the deadline.

The policy is the one intervention measured to raise SCG (1.11x at S=1, 1.47x at
S=2 on the heavy-tail trace), so the properties that make it sound are worth
pinning: it must not abandon work it cannot judge, it must leave `RollPolicy`'s
accounting intact, and `abandon=False` must be indistinguishable from `RollPolicy`.
"""

from __future__ import annotations

import pytest

from srl.component.rollout.engine import Request
from srl.policy.base import SimState
from srl.policy.roll import RollPolicy
from srl.policy.roll_doom import RollDoomPolicy


def make_request(
    rid: int, group_id: int, generated: int, target_len: int, admit_time: float = 0.0
) -> Request:
    """Build a request that has already generated `generated` tokens."""
    req = Request(
        rid=rid,
        group_id=group_id,
        group_index=rid % 2,
        prompt_len=100,
        target_len=target_len,
        v_traj=0,
        admit_time=admit_time,
    )
    req.generated = generated
    return req


def teach_cycle(policy: RollDoomPolicy, seconds: float) -> None:
    """Make `policy` observe one training cycle of `seconds`.

    The deadline is denominated in cycles, so a policy that has seen none cannot
    judge feasibility at all; every test that expects abandonment must do this
    first.
    """
    policy.on_train_version_advance(0, SimState(now=0.0, train_version=0, max_staleness=1))
    policy.on_train_version_advance(
        1, SimState(now=seconds, train_version=1, max_staleness=1)
    )


def admit(policy: RollDoomPolicy, st: SimState, *groups: list[Request]) -> None:
    """Run the admission path for each group."""
    policy.admit_quota(st)
    for group in groups:
        policy.on_admit(group, st)


def test_roll_frees_quota_for_each_permanently_dropped_group():
    """Expiring a cohort must return its groups' capacity to the quota.

    This is how ROLL replaces work it garbage-collected. It used to be a
    dedicated `replacement_groups_on_drop` hook feeding a simulator-side credit
    counter; the counter turned out to gate admission for every policy, so it
    and the hook are gone. The replacement itself is unchanged, and now rides on
    the quota: a dropped group releases what it held, so the next staging pass
    grants a fresh one.
    """
    policy = RollPolicy()
    st = SimState(batch_size=2, group_size=2, max_staleness=0, train_version=0)
    groups = [
        [make_request(0, 0, 0, 100), make_request(1, 0, 0, 100)],
        [make_request(2, 1, 0, 100), make_request(3, 1, 0, 100)],
    ]

    policy.admit_quota(st)
    for group in groups:
        policy.on_admit(group, st)
    after_admit = policy.admit_quota(st)

    # Advance past the staleness floor so both cohorts are collectable, then
    # expire them: `expire` is what decrements ROLL's own occupancy counters.
    policy.on_train_version_advance(1, SimState(batch_size=2, group_size=2, train_version=1))
    expired = policy.expire(
        SimState(
            batch_size=2,
            group_size=2,
            max_staleness=0,
            train_version=2,
            inflight=[req for group in groups for req in group],
        ),
        expected_train_version=3,
    )

    assert len(expired) == 4, "Both groups should have been garbage-collected."
    assert policy.admit_quota(
        SimState(batch_size=2, group_size=2, max_staleness=0, train_version=2)
    ) > after_admit, "Expiry did not return the dropped groups' capacity to the quota."


def test_abandons_only_the_infeasible_group():
    """A group that cannot finish in its remaining cycles is abandoned; a fast one is not."""
    policy = RollDoomPolicy(
        abandon=True, safety_margin=1.0, min_generated=100, require_observed_waste=False
    )
    # 100 tok/s for both: 1000 remaining needs 10 s, 50000 remaining needs 500 s.
    fast = [make_request(0, 0, 5000, 6000), make_request(1, 0, 5000, 6000)]
    doomed = [make_request(2, 1, 5000, 55000), make_request(3, 1, 5000, 55000)]
    st = SimState(
        now=50.0,
        train_version=0,
        batch_size=2,
        group_size=2,
        max_staleness=1,
        inflight=[*fast, *doomed],
    )
    admit(policy, st, fast, doomed)
    teach_cycle(policy, 100.0)

    victims = policy.expire(st, expected_train_version=1)

    assert sorted(r.rid for r in victims) == [2, 3]
    assert policy.abandoned == 1
    # The tokens the abandoned group would still have had to produce.
    assert policy.abandoned_tokens_saved == 2 * (55000 - 5000)


def test_abandonment_releases_the_admission_budget():
    """`running_prompts` must drop, or admission is throttled for the rest of the run."""
    policy = RollDoomPolicy(
        abandon=True, safety_margin=1.0, min_generated=100, require_observed_waste=False
    )
    doomed = [make_request(0, 0, 5000, 55000), make_request(1, 0, 5000, 55000)]
    st = SimState(
        now=50.0,
        train_version=0,
        batch_size=2,
        group_size=2,
        max_staleness=1,
        inflight=list(doomed),
    )
    admit(policy, st, doomed)
    teach_cycle(policy, 100.0)
    assert policy.running_prompts == 1

    policy.expire(st, expected_train_version=1)

    assert policy.running_prompts == 0
    # The cohort must no longer believe it owns the group, or GC would double-count.
    assert 0 not in policy.cohorts[0].running


def test_never_abandons_below_min_generated():
    """A request with too little history has no trustworthy rate to project from."""
    policy = RollDoomPolicy(abandon=True, safety_margin=1.0, min_generated=1000)
    # Only 10 tokens generated: hopeless by projection, but unjudgeable.
    young = [make_request(0, 0, 10, 50000), make_request(1, 0, 10, 50000)]
    st = SimState(
        now=50.0,
        train_version=0,
        batch_size=2,
        group_size=2,
        max_staleness=1,
        inflight=list(young),
    )
    admit(policy, st, young)
    teach_cycle(policy, 100.0)

    assert policy.expire(st, expected_train_version=1) == []
    assert policy.abandoned == 0


def test_no_abandonment_before_a_cycle_is_observed():
    """Without a measured cycle duration there is no deadline, so nothing is judged."""
    policy = RollDoomPolicy(abandon=True, safety_margin=1.0, min_generated=100)
    doomed = [make_request(0, 0, 5000, 55000), make_request(1, 0, 5000, 55000)]
    st = SimState(
        now=50.0,
        train_version=0,
        batch_size=2,
        group_size=2,
        max_staleness=1,
        inflight=list(doomed),
    )
    admit(policy, st, doomed)
    # Deliberately no teach_cycle call.

    assert policy.expire(st, expected_train_version=1) == []
    assert policy.abandoned == 0


def test_safety_margin_controls_aggressiveness():
    """A lenient margin spares a group that an aggressive one abandons.

    `consumed_groups`/`gc_dropped` are seeded past `waste_rate_trusted` so the
    configured margin is used verbatim. Without that the adaptive floor
    (`aggressive_margin`) applies and `safety_margin` is deliberately ignored --
    which is the subject of its own test below.
    """
    # 100 tok/s, 12000 remaining -> 120 s needed against a 100 s cycle budget.
    def run(margin: float) -> int:
        policy = RollDoomPolicy(
            abandon=True,
            safety_margin=margin,
            min_generated=100,
            require_observed_waste=False,
        )
        # Plenty of observed waste, so the margin is trusted as configured.
        policy.gc_dropped = 30
        policy.consumed_groups = 70
        group = [make_request(0, 0, 5000, 17000), make_request(1, 0, 5000, 17000)]
        st = SimState(
            now=50.0,
            train_version=0,
            batch_size=2,
            group_size=2,
            max_staleness=1,
            inflight=list(group),
        )
        admit(policy, st, group)
        teach_cycle(policy, 100.0)
        policy.expire(st, expected_train_version=1)
        return policy.abandoned

    assert run(1.0) == 1, "120 s of work against a 100 s budget should be abandoned."
    assert run(2.0) == 0, "A 2x margin buys 200 s, which is enough."


def test_abandon_false_matches_roll_exactly():
    """With the feature off, expiry must be `RollPolicy`'s and nothing else."""
    doomed = [make_request(0, 0, 5000, 55000), make_request(1, 0, 5000, 55000)]
    st = SimState(
        now=50.0,
        train_version=0,
        batch_size=2,
        group_size=2,
        max_staleness=1,
        inflight=list(doomed),
    )

    off = RollDoomPolicy(abandon=False, min_generated=100)
    admit(off, st, doomed)
    teach_cycle(off, 100.0)

    plain = RollPolicy()
    plain.admit_quota(st)
    plain.on_admit(doomed, st)
    plain.on_train_version_advance(0, SimState(now=0.0, train_version=0, max_staleness=1))
    plain.on_train_version_advance(1, SimState(now=100.0, train_version=1, max_staleness=1))

    assert off.expire(st, 1) == plain.expire(st, 1) == []
    assert off.running_prompts == plain.running_prompts


def test_gc_victims_are_not_double_counted():
    """A group already collected by GC must not also be charged as abandoned."""
    policy = RollDoomPolicy(abandon=True, safety_margin=1.0, min_generated=100)
    doomed = [make_request(0, 0, 5000, 55000), make_request(1, 0, 5000, 55000)]
    st = SimState(
        now=50.0,
        train_version=0,
        batch_size=2,
        group_size=2,
        max_staleness=1,
        inflight=list(doomed),
    )
    admit(policy, st, doomed)
    teach_cycle(policy, 100.0)

    # expected_train_version=2 puts cohort 0 past the GC floor (max_gc_step = 0).
    victims = policy.expire(st, expected_train_version=2)

    assert sorted(r.rid for r in victims) == [0, 1]
    # GC owns these, so the abandonment counter must stay clean.
    assert policy.abandoned == 0
    assert policy.gc_dropped == 1


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_rejects_non_positive_safety_margin(bad: float):
    with pytest.raises(ValueError, match="safety_margin"):
        RollDoomPolicy(safety_margin=bad)


def test_rejects_non_positive_min_generated():
    with pytest.raises(ValueError, match="min_generated"):
        RollDoomPolicy(min_generated=0)


def test_require_observed_waste_keeps_the_policy_inert_without_gc_drops():
    """With no GC drop yet, nothing is abandoned: a workload with no hopeless work
    must not be harmed.

    Measured motivation: the aligned DAPO trace GC-drops zero groups and
    abandonment there cost 0.861x, while the heavy-tail trace drops 528-752 and
    abandonment won 1.11-1.47x. The guard is what makes the policy safe to leave on.
    """
    doomed = [make_request(0, 0, 5000, 55000), make_request(1, 0, 5000, 55000)]
    st = SimState(
        now=50.0,
        train_version=0,
        batch_size=2,
        group_size=2,
        max_staleness=1,
        inflight=list(doomed),
    )

    guarded = RollDoomPolicy(abandon=True, min_generated=100, require_observed_waste=True)
    admit(guarded, st, doomed)
    teach_cycle(guarded, 100.0)
    assert guarded.gc_dropped == 0
    assert guarded.expire(st, expected_train_version=1) == []
    assert guarded.abandoned == 0

    # Same state, guard off: the hopeless group is abandoned.
    ungated = RollDoomPolicy(abandon=True, min_generated=100, require_observed_waste=False)
    admit(ungated, st, doomed)
    teach_cycle(ungated, 100.0)
    assert len(ungated.expire(st, expected_train_version=1)) == 2
    assert ungated.abandoned == 1


def test_margin_adapts_down_when_waste_is_scarce():
    """With little evidence of waste, the margin is pulled to `aggressive_margin`.

    A binary gate was too coarse: at DAPO S=2 the baseline discards 32 of 1568
    groups, so a presence-only gate opens, and there only an aggressive margin paid
    (1.058x at 0.7 against losses at 1.0 and 1.5). Scaling the margin with the
    observed discard rate is what makes one setting correct at both ends.
    """
    policy = RollDoomPolicy(
        abandon=True, safety_margin=1.0, aggressive_margin=0.7, waste_rate_trusted=0.15
    )

    # No evidence at all: clamp to the aggressive floor.
    assert policy._waste_rate() == 0.0
    assert policy._effective_margin() == pytest.approx(0.7)

    # 2% discarded, as measured on DAPO at S=2: still close to the floor.
    policy.gc_dropped, policy.consumed_groups = 32, 1536
    assert policy._effective_margin() == pytest.approx(0.741, abs=5e-3)

    # 26.9% discarded, as measured on the heavy-tail trace: margin fully trusted.
    policy.gc_dropped, policy.consumed_groups = 752, 2048
    assert policy._effective_margin() == pytest.approx(1.0)


def test_consumed_groups_tracks_the_waste_denominator():
    """`on_group_complete` must count, or the waste rate has no denominator."""
    policy = RollDoomPolicy()
    group = [make_request(0, 0, 100, 200), make_request(1, 0, 100, 200)]
    st = SimState(
        now=0.0,
        train_version=0,
        batch_size=2,
        group_size=2,
        max_staleness=1,
        inflight=list(group),
    )
    admit(policy, st, group)
    assert policy.consumed_groups == 0

    policy.on_group_complete(group, st)

    assert policy.consumed_groups == 1
