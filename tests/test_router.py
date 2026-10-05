"""The router's queue, dispatch, group pinning, and preemption loopback."""

from __future__ import annotations

import pytest

from dryrun.component.rollout.engine import NativeInstance, Request
from dryrun.component.rollout.router import (
    AdmissionGate,
    Router,
    RouterQueue,
    SortKey,
    build_strategy,
)
from dryrun.cost.rollout import LinearLPS


def _req(rid: int, prompt_len: int = 100, group_id: int = 0, v_traj: int = 0) -> Request:
    return Request(
        rid=rid,
        prompt_len=prompt_len,
        target_len=50,
        v_traj=v_traj,
        group_id=group_id,
        group_index=rid,
    )


def _instances(n: int, kv_blocks: int = 1000, **kwargs) -> list[NativeInstance]:
    return [
        NativeInstance(
            LinearLPS(),
            token_budget=8192,
            kv_blocks=kv_blocks,
            instance_id=i,
            **kwargs,
        )
        for i in range(n)
    ]


def _router(instances, strategy: str = "request_num_balance", **kwargs) -> Router:
    return Router(build_strategy(strategy), instances, **kwargs)


class TestQueueOrdering:
    def test_shortest_prompt_first(self):
        queue = RouterQueue(SortKey.SHORT_LENGTH)
        for rid, length in enumerate([300, 100, 200]):
            queue.push(_req(rid, prompt_len=length), t=0.0)
        assert [r.prompt_len for r in queue.drain()] == [100, 200, 300]

    def test_longest_prompt_first(self):
        queue = RouterQueue(SortKey.LONG_LENGTH)
        for rid, length in enumerate([300, 100, 200]):
            queue.push(_req(rid, prompt_len=length), t=0.0)
        assert [r.prompt_len for r in queue.drain()] == [300, 200, 100]

    def test_small_id_first(self):
        queue = RouterQueue(SortKey.SMALL_ID)
        for rid in (5, 1, 3):
            queue.push(_req(rid), t=0.0)
        assert [r.rid for r in queue.drain()] == [1, 3, 5]

    def test_lower_version_outranks_a_shorter_prompt(self):
        """An older trajectory is closer to expiring, so version dominates."""
        queue = RouterQueue(SortKey.SHORT_LENGTH)
        queue.push(_req(0, prompt_len=10, v_traj=5), t=0.0)
        queue.push(_req(1, prompt_len=999, v_traj=1), t=0.0)
        assert [r.rid for r in queue.drain()] == [1, 0]

    def test_equal_keys_keep_arrival_order(self):
        queue = RouterQueue(SortKey.SHORT_LENGTH)
        for rid in range(5):
            queue.push(_req(rid, prompt_len=100), t=0.0)
        assert [r.rid for r in queue.drain()] == [0, 1, 2, 3, 4]

    def test_a_looped_back_request_keeps_its_original_priority(self):
        """Keying on `ctx` would push a resumed request further back each time
        it returns, starving exactly the requests that have already paid most."""
        queue = RouterQueue(SortKey.SHORT_LENGTH)
        resumed = _req(0, prompt_len=100)
        resumed.add_tokens(0, 5000)  # a long partial rollout
        queue.push(resumed, t=0.0)
        queue.push(_req(1, prompt_len=200), t=0.0)
        assert [r.rid for r in queue.drain()] == [0, 1]

    def test_remove_reports_only_what_it_held(self):
        queue = RouterQueue()
        held, absent = _req(0), _req(1)
        queue.push(held, t=0.0)
        assert queue.remove([held, absent]) == [held]
        assert len(queue) == 0

    def test_remove_preserves_order_of_the_rest(self):
        queue = RouterQueue(SortKey.SHORT_LENGTH)
        reqs = [_req(i, prompt_len=100 * (i + 1)) for i in range(4)]
        for req in reqs:
            queue.push(req, t=0.0)
        queue.remove([reqs[1]])
        assert [r.rid for r in queue.drain()] == [0, 2, 3]

    def test_enqueue_records_the_wait_start(self):
        queue = RouterQueue()
        req = _req(0)
        queue.push(req, t=12.5)
        assert req.router_enqueue_time == 12.5


class TestDispatch:
    def test_it_places_what_it_can(self):
        instances = _instances(2)
        router = _router(instances)
        router.enqueue([_req(i) for i in range(4)], t=0.0)
        placed = router.dispatch(t=0.0)
        assert len(placed) == 4
        assert len(router) == 0
        assert sum(inst.n_active for inst in instances) == 4

    def test_balance_spreads_across_instances(self):
        instances = _instances(2)
        router = _router(instances)
        router.enqueue([_req(i) for i in range(4)], t=0.0)
        router.dispatch(t=0.0)
        assert [inst.n_active for inst in instances] == [2, 2]

    def test_a_refused_request_dwells_instead_of_being_lost(self):
        """The whole point of the queue: no rollback, no drop."""
        instances = _instances(1)
        router = _router(instances, gate=AdmissionGate(max_concurrent_seqs_per_instance=2))
        router.enqueue([_req(i) for i in range(5)], t=0.0)
        placed = router.dispatch(t=0.0)
        assert len(placed) == 2
        assert len(router) == 3

    def test_capacity_released_later_drains_the_queue(self):
        instances = _instances(1)
        router = _router(instances, gate=AdmissionGate(max_concurrent_seqs_per_instance=2))
        router.enqueue([_req(i) for i in range(4)], t=0.0)
        router.dispatch(t=0.0)
        instances[0].cancel(instances[0].waiting[:1], keep_tokens=False)
        assert len(router.dispatch(t=1.0)) == 1

    def test_dispatch_stops_at_the_first_request_nothing_can_take(self):
        """Head-of-line, as SMG's peek-then-pop is. Skipping the head to place
        a later request would starve whatever needs the most room."""
        instances = _instances(1, kv_blocks=64, block_size=16)
        # `long_length` puts the big request at the head, so it is genuinely the
        # blocker rather than being sorted out of the way.
        router = _router(instances, sort_key=SortKey.LONG_LENGTH)
        # Fill most of the pool: admitting the big request would reserve
        # ceil((800 + 400) / 16) = 75 blocks against the 64 available, so it is
        # refused, while the small one would need only 51 and still fits. The
        # small one must not be allowed to jump the queue.
        instances[0].add_request(_req(9, prompt_len=800), 0.0)
        router.enqueue([_req(0, prompt_len=400), _req(1, prompt_len=16)], t=0.0)
        placed = router.dispatch(t=0.0)
        assert placed == []
        assert len(router) == 2

    def test_has_dispatchable_tracks_the_queue(self):
        router = _router(_instances(1))
        assert not router.has_dispatchable()
        router.enqueue([_req(0)], t=0.0)
        assert router.has_dispatchable()

    def test_dispatch_inside_a_speculative_event_is_refused(self):
        """The invariant the no-undo design rests on, asserted rather than assumed."""
        instances = _instances(1)
        router = _router(instances)
        router.enqueue([_req(0)], t=0.0)
        instances[0].add_request(_req(99), 0.0)
        instances[0].advance_one_event(version=0)  # leaves a snapshot open
        with pytest.raises(AssertionError, match="speculative"):
            router.dispatch(t=0.0)

    def test_committing_an_event_releases_the_guard(self):
        instances = _instances(1)
        router = _router(instances)
        instances[0].add_request(_req(99), 0.0)
        instances[0].advance_one_event(version=0)
        instances[0].commit_last_event()
        router.enqueue([_req(0)], t=0.0)
        assert len(router.dispatch(t=0.0)) == 1


class TestRerouteAccounting:
    def test_a_first_placement_is_not_a_reroute(self):
        router = _router(_instances(2))
        router.enqueue([_req(0)], t=0.0)
        router.dispatch(t=0.0)
        assert router.n_routed == 1
        assert router.n_reroutes == 0

    def test_a_second_placement_counts_as_a_reroute(self):
        instances = _instances(2)
        router = _router(instances)
        req = _req(0)
        router.enqueue([req], t=0.0)
        router.dispatch(t=0.0)
        first = req.instance_id
        instances[first].cancel([req], keep_tokens=True, requeue=False)
        router.enqueue([req], t=1.0)
        router.dispatch(t=1.0)
        assert router.n_reroutes == 1
        assert req.num_reroutes == 1

    def test_a_cross_instance_reroute_is_counted_separately(self):
        """The quantity the real run puts at 58.5%, so it must be measurable."""
        instances = _instances(2)
        router = _router(instances)
        req = _req(0)
        router.enqueue([req], t=0.0)
        router.dispatch(t=0.0)
        origin = req.instance_id
        instances[origin].cancel([req], keep_tokens=True, requeue=False)
        # Load the origin so balance sends the request to the other instance.
        for filler in range(5):
            instances[origin].add_request(_req(100 + filler), 0.0)
        router.enqueue([req], t=1.0)
        router.dispatch(t=1.0)
        assert req.instance_id != origin
        assert router.n_cross_instance_reroutes == 1


class TestGroupSticky:
    def test_siblings_follow_the_first_placement(self):
        instances = _instances(2)
        router = _router(instances, enable_group_sticky=True)
        router.enqueue([_req(i, group_id=7) for i in range(4)], t=0.0)
        router.dispatch(t=0.0)
        assert len({inst.n_active for inst in instances}) == 2  # all on one
        assert max(inst.n_active for inst in instances) == 4

    def test_disabled_by_default_so_groups_scatter(self):
        """The matched run sets enable_group_sticky=False and scatters all 160
        groups 4/4, so scattering is the faithful default."""
        instances = _instances(2)
        router = _router(instances)
        router.enqueue([_req(i, group_id=7) for i in range(4)], t=0.0)
        router.dispatch(t=0.0)
        assert [inst.n_active for inst in instances] == [2, 2]

    def test_the_pin_is_soft_when_its_instance_cannot_help(self):
        """A hard pin would dwell a request behind one busy instance while
        others idle. SMG falls through instead."""
        instances = _instances(2)
        router = _router(
            instances,
            gate=AdmissionGate(max_concurrent_seqs_per_instance=2),
            enable_group_sticky=True,
        )
        router.enqueue([_req(i, group_id=7) for i in range(4)], t=0.0)
        router.dispatch(t=0.0)
        assert [inst.n_active for inst in instances] == [2, 2]

    def test_releasing_a_pin_frees_the_group(self):
        """A held pin would let a repeating workload be routed by a group that
        no longer exists."""
        instances = _instances(2)
        router = _router(instances, enable_group_sticky=True)
        router.enqueue([_req(0, group_id=7)], t=0.0)
        router.dispatch(t=0.0)
        assert 7 in router._group_pin
        router.release_group_pins({7})
        assert 7 not in router._group_pin

    def test_a_held_pin_keeps_sending_siblings_to_one_instance(self):
        """The contrast case: without the release the group stays welded on."""
        instances = _instances(2)
        router = _router(instances, enable_group_sticky=True)
        for rid in range(3):
            router.enqueue([_req(rid, group_id=7)], t=float(rid))
            router.dispatch(t=float(rid))
        assert sorted(inst.n_active for inst in instances) == [0, 3]


class TestPreemptLoopback:
    def test_it_is_off_by_default(self):
        """Without a threshold every preemption stays local, as vLLM's does."""
        instances = _instances(1)
        router = _router(instances)
        for i in range(10):
            instances[0].add_request(_req(i), 0.0)
        assert router.reclaim_preempted(t=0.0) == []

    def test_below_the_threshold_nothing_is_reclaimed(self):
        instances = _instances(1)
        router = _router(instances, max_waiting_after_preempt=100)
        for i in range(10):
            instances[0].add_request(_req(i), 0.0)
        assert router.reclaim_preempted(t=0.0) == []

    def test_overflow_returns_to_the_router(self):
        instances = _instances(1)
        router = _router(instances, max_waiting_after_preempt=2)
        for i in range(5):
            req = _req(i)
            instances[0].add_request(req, 0.0)
            req.num_preemptions = 1  # these were preempted, not fresh
        reclaimed = router.reclaim_preempted(t=0.0)
        assert len(reclaimed) == 3
        assert len(instances[0].waiting) == 2
        assert len(router) == 3

    def test_only_preempted_requests_are_taken(self):
        """A fresh admission has not been preempted, so it is not overflow."""
        instances = _instances(1)
        router = _router(instances, max_waiting_after_preempt=1)
        for i in range(4):
            instances[0].add_request(_req(i), 0.0)
        assert router.reclaim_preempted(t=0.0) == []

    def test_a_mid_prefill_request_is_never_taken(self):
        """It holds blocks without decoding, so moving it throws away prefill
        the instance has already paid for."""
        instances = _instances(1)
        router = _router(instances, max_waiting_after_preempt=1)
        for i in range(4):
            req = _req(i)
            instances[0].add_request(req, 0.0)
            req.num_preemptions = 1
            req.prefilled = 50  # mid-prefill
        assert router.reclaim_preempted(t=0.0) == []

    def test_a_reclaimed_request_keeps_its_generated_tokens(self):
        instances = _instances(1)
        router = _router(instances, max_waiting_after_preempt=0)
        req = _req(0)
        instances[0].add_request(req, 0.0)
        req.num_preemptions = 1
        req.add_tokens(0, 300)
        router.reclaim_preempted(t=0.0)
        assert req.generated == 300
        assert req.instance_id == 0  # the hint survives, for reroute accounting
