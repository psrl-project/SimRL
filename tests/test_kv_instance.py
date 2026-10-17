"""Tests for prefix caching inside the rollout engine instance."""

from __future__ import annotations

import random

import pytest

from srl.component.rollout.engine import NativeInstance, Request
from srl.cost.rollout import UnifiedRoofline


def _inst(**kwargs) -> NativeInstance:
    defaults = dict(
        cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
        token_budget=8192,
        kv_blocks=1000,
        block_size=16,
    )
    defaults.update(kwargs)
    return NativeInstance(**defaults)


def _req(rid: int, prompt_len: int, source_group_id: int, target_len: int = 64) -> Request:
    return Request(
        rid=rid,
        prompt_len=prompt_len,
        target_len=target_len,
        v_traj=0,
        group_id=source_group_id,
        source_group_id=source_group_id,
    )


class TestOccupancyComesFromThePool:
    def test_an_idle_instance_holds_nothing(self):
        inst = _inst()
        assert inst.blocks_used() == 0
        assert inst.free_blocks() == 1000

    def test_admitting_then_prefilling_charges_blocks(self):
        inst = _inst()
        assert inst.add_request(_req(rid=1, prompt_len=150, source_group_id=1), t=0.0)
        inst.advance_one_event(version=0)
        assert inst.blocks_used() > 0
        assert inst.blocks_used() + inst.free_blocks() == 1000

    def test_the_pool_and_the_instance_agree_after_a_full_run(self):
        inst = _inst()
        for rid in range(4):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=rid), t=0.0)
        for _ in range(200):
            if inst.advance_one_event(version=0) is None:
                break
        assert len(inst.completed) == 4
        assert inst.blocks_used() == 0, "every retired request released its blocks"

    def test_the_pool_is_the_source_of_truth_for_occupancy(self):
        """Not a sum derived from `prefilled`. The distinction matters once
        blocks are shared: two requests holding one block must be counted once,
        which a per-request sum cannot do."""
        inst = _inst()
        assert inst.add_request(_req(rid=1, prompt_len=150, source_group_id=1), t=0.0)
        inst.advance_one_event(version=0)
        assert inst.blocks_used() == inst.kv.num_used_blocks
        assert inst.free_blocks() == inst.kv.num_free_blocks


def _instance_state(inst) -> tuple:
    """Everything an undo must restore, pool included.

    Occupancy alone is too weak: a block returned to the wrong free-list slot
    keeps `blocks_used()` correct while rewriting LRU order, which then decides
    every later eviction.
    """
    return (
        inst.t_local,
        [r.rid for r in inst.waiting],
        [r.rid for r in inst.running],
        len(inst.completed),
        len(inst.step_stats),
        inst.n_preemptions,
        inst.blocks_used(),
        inst.kv.pool.free_list_ids(),
        inst.kv.pool.eviction_count,
        inst.kv.queries,
        inst.kv.hits,
        [
            (r.rid, r.prefilled, r.generated, r.prefix_hit_tokens)
            for r in (*inst.waiting, *inst.running)
        ],
    )


class TestUndoRestoresThePool:
    def test_undoing_an_event_restores_the_whole_instance(self):
        """simulator.py:358 undoes every speculatively advanced event that does
        not commit, so this is a live path rather than a safety net."""
        inst = _inst()
        assert inst.add_request(_req(rid=1, prompt_len=150, source_group_id=1), t=0.0)
        inst.advance_one_event(version=0)
        before = _instance_state(inst)

        inst.advance_one_event(version=0)
        inst.undo_last_event()
        assert _instance_state(inst) == before

    def test_undoing_restores_the_free_list_order_not_just_the_count(self):
        inst = _inst()
        # Staggered target_len: each closed-form decode segment retires one
        # request, so work stays in flight across many events.
        for rid in range(8):
            assert inst.add_request(
                _req(rid=rid, prompt_len=150, source_group_id=rid, target_len=64 * (rid + 1)),
                t=0.0,
            )
        for _ in range(3):
            inst.advance_one_event(version=0)
        order_before = inst.kv.pool.free_list_ids()

        assert inst.advance_one_event(version=0) is not None, "needs a live event"
        inst.undo_last_event()
        assert inst.kv.pool.free_list_ids() == order_before

    def test_undoing_restores_hit_bookkeeping(self):
        """queries and hits are compared against vllm:prefix_cache_* counters,
        so an uncommitted event must not inflate them."""
        inst = _inst()
        for rid in range(4):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=7), t=0.0)
        inst.advance_one_event(version=0)
        before = (inst.kv.queries, inst.kv.hits)

        inst.advance_one_event(version=0)
        inst.undo_last_event()
        assert (inst.kv.queries, inst.kv.hits) == before

    def test_repeated_advance_and_undo_does_not_drift(self):
        """Twenty speculative advances, each rolled back. The driver does this
        every iteration: it advances every instance and keeps only one."""
        inst = _inst()
        for rid in range(8):
            assert inst.add_request(
                _req(rid=rid, prompt_len=150, source_group_id=rid, target_len=64 * (rid + 1)),
                t=0.0,
            )
        for _ in range(3):
            inst.advance_one_event(version=0)
        baseline = _instance_state(inst)
        for _ in range(20):
            assert inst.advance_one_event(version=0) is not None, "needs a live event"
            inst.undo_last_event()
        assert _instance_state(inst) == baseline

    def test_a_decode_segment_is_undoable_too(self):
        """A decode segment advances every running request by k tokens at once,
        touching up to the whole pool. It must reverse like any other event."""
        inst = _inst()
        assert inst.add_request(
            _req(rid=1, prompt_len=150, source_group_id=1, target_len=512), t=0.0
        )
        for _ in range(20):
            inst.advance_one_event(version=0)
            if not inst.waiting:
                break
        before = _instance_state(inst)

        inst.advance_one_event(version=0)
        inst.undo_last_event()
        assert _instance_state(inst) == before

    def test_an_idle_instance_drops_its_stale_snapshot(self):
        """A snapshot must exist exactly while an uncommitted event is in
        flight. Otherwise an undo after an idle call rewinds to whichever event
        came before, which is a silent corruption rather than a no-op."""
        inst = _inst()
        assert inst.add_request(_req(rid=1, prompt_len=150, source_group_id=1), t=0.0)
        while inst.advance_one_event(version=0) is not None:
            pass
        assert inst._snapshot is None

    def test_committed_advancement_leaves_no_snapshot(self):
        """`advance_to` commits as it goes and passes NULL_RECORDER, so it pays
        nothing for recording and leaves nothing behind to undo."""
        inst = _inst()
        assert inst.add_request(_req(rid=1, prompt_len=150, source_group_id=1), t=0.0)
        inst.advance_to(t_limit=1e9, version=0)
        assert inst._snapshot is None


class TestWithinGroupPromptReuse:
    def test_siblings_pay_for_the_shared_prompt_once(self):
        """The limitation in docs/reference/limitations.md: 'When a prompt has n
        sampled responses, the simulator charges prefill n times.' At
        prompt_len=150 with block_size=16 the shared part is 9 full blocks over
        144 tokens, so seven of eight siblings skip 144 tokens each."""
        inst = _inst()
        for rid in range(8):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=99), t=0.0)
        for _ in range(500):
            if inst.advance_one_event(version=0) is None:
                break

        assert len(inst.completed) == 8
        saved = sum(r.prefix_hit_tokens for r in inst.completed)
        assert saved == 7 * 144

    def test_prompt_prefill_drops_to_the_predicted_token_count(self):
        inst = _inst()
        for rid in range(8):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=99), t=0.0)
        for _ in range(500):
            if inst.advance_one_event(version=0) is None:
                break

        prompt_work = sum(s.n_prefill_computed for s in inst.step_stats)
        assert prompt_work == 150 + 7 * 6, "150 cold, then six private tokens each"

    def test_different_prompts_share_nothing(self):
        inst = _inst()
        for rid in range(8):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=rid), t=0.0)
        for _ in range(500):
            if inst.advance_one_event(version=0) is None:
                break
        assert sum(r.prefix_hit_tokens for r in inst.completed) == 0

    def test_disabling_prefix_caching_restores_the_old_cost(self):
        inst = _inst(enable_prefix_caching=False)
        for rid in range(8):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=99), t=0.0)
        for _ in range(500):
            if inst.advance_one_event(version=0) is None:
                break
        assert sum(r.prefix_hit_tokens for r in inst.completed) == 0
        assert sum(s.n_prefill_computed for s in inst.step_stats) == 8 * 150

    def test_a_shared_block_is_counted_once_in_occupancy(self):
        """The reason occupancy had to move into the pool: a per-request sum
        would charge each sibling for the blocks they share."""
        inst = _inst()
        for rid in range(8):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=99), t=0.0)
        inst.advance_one_event(version=0)
        # 9 shared prompt blocks plus one private tail block per sibling.
        assert inst.blocks_used() == 9 + 8


class TestWeightSyncClearsHashes:
    def test_abort_all_frees_blocks_and_clears_the_hash_map(self):
        """PSRL's pause_for_sync aborts every request, then pull_model calls
        reset_prefix_cache. Both steps, in that order."""
        inst = _inst()
        for rid in range(4):
            assert inst.add_request(_req(rid=rid, prompt_len=160, source_group_id=5), t=0.0)
        for _ in range(50):
            inst.advance_one_event(version=0)

        inst.abort_all()
        assert inst.blocks_used() == 0
        assert inst.kv.pool.cached_block(("p", 5, 0)) is None

    def test_a_resumed_group_still_hits_on_the_shared_prompt(self):
        """The post-sync hit rate is not zero: the first sibling repopulates
        the prompt hashes and the rest hit them. This is the mechanism behind
        the 9% measured in outputs/simulate/SMG_LOOPBACK_FIX.md."""
        inst = _inst(kv_blocks=4000)
        for rid in range(4):
            assert inst.add_request(
                _req(rid=rid, prompt_len=160, source_group_id=5, target_len=512), t=0.0
            )
        for _ in range(3):
            inst.advance_one_event(version=0)
        inst.abort_all()
        saved_before = sum(r.prefix_hit_tokens for r in inst.waiting)

        for _ in range(200):
            if inst.advance_one_event(version=1) is None:
                break
        saved_after = sum(
            r.prefix_hit_tokens for r in (*inst.waiting, *inst.running, *inst.completed)
        )
        assert saved_after > saved_before, "siblings hit the repopulated prompt blocks"

    def test_an_agentic_round_boundary_keeps_hashes(self):
        """`cancel(keep_tokens=True)` is the agentic round boundary: free the
        blocks, keep the hashes, so the next round hits. This is the interface
        the agentic spec builds on, and the reason free and reset are separate."""
        inst = _inst()
        req = _req(rid=1, prompt_len=160, source_group_id=5)
        assert inst.add_request(req, t=0.0)
        inst.advance_one_event(version=0)

        inst.cancel([req], keep_tokens=True)
        assert inst.blocks_used() == 0
        assert inst.kv.pool.cached_block(("p", 5, 0)) is not None


class TestThePoolNeverOvercommits:
    """The pool is authoritative, so `prefilled` must never run past it.

    The old accounting derived occupancy from `prefilled`, so a step could
    advance into blocks it had not reserved and detect the overflow afterwards.
    With a real pool that silently desynchronises the two: a request reports a
    2000-token context while holding blocks for 600, and every occupancy number
    downstream is wrong. These tests pin the invariant at the three places a
    step can advance: a prefill chunk, a per-token decode inside a prefill step,
    and a closed-form decode segment.
    """

    @staticmethod
    def _derived_blocks(inst) -> int:
        """What occupancy would be if each request held its context privately."""
        from srl.component.rollout.engine import blocks_for

        return sum(
            blocks_for(r.prefilled, inst.block_size)
            for r in (*inst.waiting, *inst.running)
        )

    def test_held_blocks_cover_every_context_under_moderate_pressure(self):
        """The shape that actually catches an unreserved advance.

        With caching off nothing is shared, so the pool's occupancy must equal
        the per-request sum at every step. Reverting either the prefill or the
        decode guard makes this drift by 9 blocks; the 8-request shapes happen to
        be immune, which is why the count and capacity matter here.
        """
        inst = _inst(kv_blocks=500, max_concurrency=1024, enable_prefix_caching=False)
        for rid in range(32):
            assert inst.add_request(
                _req(rid=rid, prompt_len=200, source_group_id=rid, target_len=500), t=0.0
            )
        for _ in range(1500):
            if inst.advance_one_event(version=0) is None:
                break
            assert inst.blocks_used() == self._derived_blocks(inst)
            assert inst.blocks_used() <= inst.kv_blocks

    def test_held_blocks_cover_every_context_at_the_baseline_shape(self):
        """64 requests of the aligned baseline's own length, squeezed. Drifts by
        21 blocks without the guards."""
        inst = _inst(kv_blocks=1000, max_concurrency=1024, enable_prefix_caching=False)
        for rid in range(64):
            assert inst.add_request(
                _req(rid=rid, prompt_len=137, source_group_id=rid, target_len=10240),
                t=0.0,
            )
        for _ in range(1500):
            if inst.advance_one_event(version=0) is None:
                break
            assert inst.blocks_used() == self._derived_blocks(inst)
            assert inst.blocks_used() <= inst.kv_blocks

    def test_occupancy_never_exceeds_capacity_when_far_oversubscribed(self):
        """8 requests of 137+2000 tokens need 1072 blocks against a 200 cap, so
        the pool must refuse and preempt rather than overcommit."""
        inst = _inst(kv_blocks=200, max_concurrency=1024)
        for rid in range(8):
            assert inst.add_request(
                _req(rid=rid, prompt_len=137, source_group_id=rid, target_len=2000), t=0.0
            )
        for _ in range(2000):
            if inst.advance_one_event(version=0) is None:
                break
            assert inst.blocks_used() <= inst.kv_blocks

    def test_a_feasible_workload_still_completes(self):
        """The refusal path must not turn into a livelock when the work fits.
        Staggered lengths so decode segments are interrupted and blocks grow."""
        inst = _inst(kv_blocks=4000, max_concurrency=1024)
        for rid in range(8):
            assert inst.add_request(
                _req(
                    rid=rid,
                    prompt_len=137,
                    source_group_id=rid,
                    target_len=200 * (rid + 1),
                ),
                t=0.0,
            )
        for _ in range(5000):
            if inst.advance_one_event(version=0) is None:
                break
        assert len(inst.completed) == 8
        assert inst.n_preemptions == 0, "ample capacity should need no preemption"
        assert inst.blocks_used() == 0

    def test_advance_to_also_respects_capacity(self):
        """`advance_to` is the committed path and shares the decode logic, so it
        needs the same guarantee."""
        inst = _inst(kv_blocks=200, max_concurrency=1024, enable_prefix_caching=False)
        for rid in range(8):
            assert inst.add_request(
                _req(rid=rid, prompt_len=137, source_group_id=rid, target_len=2000), t=0.0
            )
        for _ in range(50):
            inst.advance_to(t_limit=inst.t_local + 0.5, version=0)
            assert inst.blocks_used() <= inst.kv_blocks
            assert inst.blocks_used() == self._derived_blocks(inst)


class TestExhaustionAccountsForSharing:
    """`_kv_exhaustion_steps` must measure growth, not an absolute sum.

    Its increments are residue-bucketed differences of `blocks_for`, which stay
    correct under sharing because decode only ever grows *private* blocks. Its
    intercept has to come from the pool. Summing `blocks_for(prefilled)` instead
    over-counts a shared prompt by the group size -- 896 against a real 134 for
    eight siblings -- which reports exhaustion immediately, preempts, lets the
    request back in, and livelocks with the pool nearly empty.
    """

    @staticmethod
    def _exact(inst) -> int | None:
        """Brute force: first k whose pool occupancy would exceed capacity.

        The pool after k steps is `pool(0)` plus each running request's private
        growth, since sharing is fixed at the prompt.
        """
        from srl.component.rollout.engine import blocks_for

        cap = max((r.remaining for r in inst.running), default=0)
        if cap == 0:
            return None
        base = inst.blocks_used()
        for k in range(0, cap + 1):
            grown = base + sum(
                blocks_for(r.prefilled + k, inst.block_size)
                - blocks_for(r.prefilled, inst.block_size)
                for r in inst.running
            )
            if grown > inst.kv_blocks:
                return k
        return None

    def test_matches_brute_force_when_siblings_share_a_prompt(self):
        inst = _inst(kv_blocks=1000)
        for rid in range(8):
            inst.add_request(
                _req(rid=rid, prompt_len=2048, source_group_id=99, target_len=400), t=0.0
            )
        assert len(inst.waiting) == 8, "admission queues every request"
        for _ in range(6):
            if inst.advance_one_event(version=0) is None:
                break
            assert inst._kv_exhaustion_steps() == self._exact(inst)

    def test_matches_brute_force_without_sharing(self):
        inst = _inst(kv_blocks=1000)
        for rid in range(8):
            inst.add_request(
                _req(rid=rid, prompt_len=300, source_group_id=rid, target_len=400), t=0.0
            )
        assert len(inst.waiting) == 8, "short prompts all fit"
        for _ in range(6):
            if inst.advance_one_event(version=0) is None:
                break
            assert inst._kv_exhaustion_steps() == self._exact(inst)

    def test_a_group_with_room_to_spare_never_preempts(self):
        """The livelock guard. Eight siblings of a 2048-token prompt fit in 1000
        blocks with hundreds to spare, so preemption here is spurious."""
        inst = _inst(kv_blocks=1000)
        for rid in range(8):
            inst.add_request(
                _req(rid=rid, prompt_len=2048, source_group_id=99, target_len=400), t=0.0
            )
        for _ in range(2000):
            if inst.advance_one_event(version=0) is None:
                break
        assert inst.n_preemptions == 0, "the pool was never short of blocks"
        assert len(inst.completed) == 8, "every sibling runs to completion"

    def test_sharing_never_makes_the_estimate_optimistic(self):
        """The intercept must never be below the pool's true occupancy, or the
        estimate would authorise a step the pool cannot fund."""
        from srl.component.rollout.engine import blocks_for

        inst = _inst(kv_blocks=4000)
        for rid in range(8):
            assert inst.add_request(
                _req(rid=rid, prompt_len=2048, source_group_id=99, target_len=400), t=0.0
            )
        for _ in range(6):
            if inst.advance_one_event(version=0) is None:
                break
        derived = sum(
            blocks_for(r.prefilled, inst.block_size)
            for r in (*inst.waiting, *inst.running)
        )
        assert inst.blocks_used() <= derived, "sharing only ever lowers occupancy"


class TestAFullPoolStillMakesProgress:
    """A workload larger than the pool must run in waves, not stall.

    Three mechanisms have to agree for this, and each one alone looks fine:

    1. The step serves `running` before `waiting`, as vLLM does. Prefilling
       first takes the blocks decode then needs, so the step overflows and
       preempts the request it just admitted.
    2. A step that could not fund every decode token admits no new work, which
       is vLLM's `if not preempted_reqs:` guard.
    3. A fresh admission must leave the resident batch a watermark. Without it
       every request gets a toe-hold and none can finish: 128 requests needing
       78,335 blocks against a 31,294-block pool end at `running=1,
       waiting=127` with the pool full and nothing able to advance.

    Capacity allows roughly 61 residents here, and the real engine's measured
    median running on this workload is 59, so the wave size is right too.
    """

    @staticmethod
    def _oversubscribed(inst) -> int:
        """Admit 128 requests of ~11k tokens each. Returns how many landed."""
        admitted = 0
        for rid in range(128):
            # Lengths near the real trace's median: prompt ~140, output ~11k.
            if inst.add_request(
                _req(rid=rid, prompt_len=140, source_group_id=rid // 8, target_len=11000),
                t=0.0,
            ):
                admitted += 1
        return admitted

    def test_an_oversubscribed_batch_drains_completely(self):
        inst = _inst(kv_blocks=31294, token_budget=12800, max_concurrency=1024)
        admitted = self._oversubscribed(inst)
        for _ in range(200_000):
            if inst.advance_one_event(version=0) is None:
                break
        assert len(inst.completed) == admitted, (
            f"stalled with {len(inst.running)} running and {len(inst.waiting)} waiting; "
            f"a pool-bound workload must run in waves rather than deadlock"
        )

    def test_the_batch_never_collapses_to_a_single_request(self):
        """The failure signature: preemption drains the batch to one request
        that holds the pool while everyone else waits forever."""
        inst = _inst(kv_blocks=31294, token_budget=12800, max_concurrency=1024)
        self._oversubscribed(inst)
        for _ in range(4000):
            if inst.advance_one_event(version=0) is None:
                break
            # Once the pool is full there should still be a real batch resident.
            if inst.free_blocks() == 0 and inst.waiting:
                assert len(inst.running) > 1, (
                    "pool full with a single resident request: the rest can never "
                    "be re-admitted, so the instance is deadlocked"
                )

    def test_an_eviction_leaves_the_batch_able_to_decode(self):
        """Preemption must clear the stall it was called for.

        Not a bound on how often preemption happens: an oversubscribed pool
        preempts often and that is correct. What must hold is that the pool can
        fund one decode token per resident afterwards, so the engine advances
        instead of re-entering the same stall with nothing changed.
        """
        inst = _inst(kv_blocks=31_294, token_budget=12_800, max_concurrency=1024)
        for rid in range(188):
            # Staggered lengths: equal-length requests retire together and
            # release in bulk, which never exercises the stall.
            req = _req(rid=rid, prompt_len=140, source_group_id=rid, target_len=11_300)
            req.generated = 9_000 + 37 * rid if rid < 60 else 100 + rid % 80
            inst.add_request(req, t=0.0)
        for _ in range(400):
            inst.advance_to(t_limit=inst.t_local + 50.0, version=0)
            if not inst.running and not inst.waiting:
                break
            if inst.free_blocks() == 0 and inst.running:
                boundary = sum(
                    1 for r in inst.running if (r.prefilled % inst.block_size) == 0
                )
                assert boundary == 0, (
                    f"pool empty with {boundary} residents on a block boundary: "
                    f"the next decode step cannot advance them"
                )
        assert len(inst.completed) == 188, (
            f"only {len(inst.completed)} of 188 finished; a pool-bound batch must "
            f"drain in waves rather than stall"
        )

    def test_a_full_pool_alone_is_not_a_stall(self):
        """Preemption needs a refused decode token, not merely a full pool.

        Asking whether any resident *could* need a block reads a full pool as
        stalled whenever one sits on a block boundary, which at 175 running has
        probability 1.0. On the matched run that fired on 96% of a plateau's
        prefill steps, every one of which decoded 93-182 tokens.
        """
        inst = _inst(kv_blocks=200, token_budget=12_800, max_concurrency=1024)
        for rid in range(8):
            req = _req(rid=rid, prompt_len=320, source_group_id=rid, target_len=2_000)
            inst.add_request(req, t=float(rid))
        while inst.free_blocks() > 0:
            if inst.advance_one_event(version=0) is None:
                break
            inst.commit_last_event()
        assert inst.free_blocks() == 0, "the fixture must drive the pool full"
        assert len(inst.running) > 1, "the fixture must leave a resident batch"
        # A request sitting exactly on a block boundary is what the old proxy
        # keyed on, so put one there rather than waiting for the batch to drift
        # into that state.
        inst.running[0].prefilled -= inst.running[0].prefilled % inst.block_size
        boundary = sum(1 for r in inst.running if (r.prefilled % inst.block_size) == 0)
        assert boundary > 0
        # The pool granted every decode token it was asked for on the last step,
        # so this is not a stall however many residents sit on a boundary.
        inst._last_decode_blocked = False
        assert not inst._deadlocked(), (
            f"full pool with {boundary} of {len(inst.running)} residents on a block "
            f"boundary read as a stall, though the decode pass was never refused"
        )
        # And the converse: a genuine refusal must still be reported.
        inst._last_decode_blocked = True
        assert inst._deadlocked(), "a refused decode token must still count as a stall"

    def test_a_genuine_stall_is_still_broken_by_preemption(self):
        """The other side of the same rule: the escape must survive.

        With the deadlock path removed this workload ends at running=49,
        waiting=139 and completes nothing, so the predicate cannot simply be
        dropped once its false positives are gone.
        """
        inst = _inst(kv_blocks=31_294, token_budget=12_800, max_concurrency=1024)
        for rid in range(188):
            req = _req(rid=rid, prompt_len=140, source_group_id=rid, target_len=11_300)
            req.generated = 9_000 + 37 * rid if rid < 60 else 100 + rid % 80
            inst.add_request(req, t=float(rid))
        # Draining this batch takes ~6100 simulated seconds.
        for _ in range(2000):
            inst.advance_to(t_limit=inst.t_local + 5.0, version=0)
            if not inst.running and not inst.waiting:
                break
        assert inst.n_preemptions > 0, "a pool-bound batch has to preempt to advance"
        assert len(inst.completed) == 188, (
            f"only {len(inst.completed)} of 188 finished; the stall was never broken"
        )

    def test_a_pool_with_room_admits_everything_at_once(self):
        """The control: the watermark must not throttle a batch that fits."""
        inst = _inst(kv_blocks=200_000, token_budget=12800, max_concurrency=1024)
        for rid in range(64):
            assert inst.add_request(
                _req(rid=rid, prompt_len=140, source_group_id=rid // 8, target_len=2000),
                t=0.0,
            )
        inst.advance_one_event(version=0)
        assert len(inst.running) == 64, "ample capacity should admit the whole batch"
        assert inst.n_preemptions == 0


class TestTheEstimateMatchesTheStep:
    """`_step_time_estimate` must price what `_engine_step` does.

    `advance_to` has no snapshot, so it asks the twin how long the next
    iteration will take before performing it. Any disagreement advances the
    clock by work that does not happen: the two differed by 369.6% on one
    workload's `t_local` before the twin was given the step's own gates.
    """

    @staticmethod
    def _priced(inst) -> int:
        """Tokens each path hands the cost model for the current state."""
        captured: list[int] = []
        inner = inst.cost

        class Spy:
            def __getattr__(self, name):
                return getattr(inner, name)

            def step_time(self, n_tokens, t2, ctxsum, n_reqs=0, prefill_ctx=0.0):
                captured.append(n_tokens)
                return inner.step_time(n_tokens, t2, ctxsum, n_reqs, prefill_ctx)

        inst.cost = Spy()
        try:
            inst._step_time_estimate()
            inst._save_snapshot()
            inst._engine_step(0)
            inst.undo_last_event()
        finally:
            inst.cost = inner
        return captured[0], captured[1]

    @pytest.mark.parametrize("kv_blocks", [500, 2000, 31_294])
    def test_the_twin_prices_the_same_tokens(self, kv_blocks):
        """Swept rather than spot-checked: the gates drifted apart silently.

        Prefix caching is off because `peek_hit_tokens` deliberately does not
        reserve the blocks a hit claims, so the twin cannot see what the real
        step's `_apply_prefix_hit` consumes. That is a known approximation,
        measured at 6 divergent states in 540 trials, against 0 in 19,408 with
        caching off.
        """
        rng = random.Random(kv_blocks)
        inst = _inst(kv_blocks=kv_blocks, token_budget=12_800,
                     max_concurrency=1024, enable_prefix_caching=False)
        # `add_request` refuses a request that can never fit even in an empty
        # pool, so prompt and target together stay under half of it. The batch
        # as a whole still oversubscribes, which is the regime being probed.
        budget = kv_blocks * 16 // 2
        for rid in range(40):
            prompt = min(rng.choice([140, 1024, 4096]), budget // 2)
            target = budget - prompt
            req = _req(rid=rid, prompt_len=prompt,
                       source_group_id=rid // 8, target_len=target)
            req.generated = rng.randint(0, max(1, target - 1))
            inst.add_request(req, t=float(rid))
        probed = 0
        for _ in range(150):
            if inst.waiting:
                twin, real = self._priced(inst)
                assert twin == real, (
                    f"twin priced {twin} tokens, the step did {real}, with "
                    f"running={len(inst.running)} waiting={len(inst.waiting)} "
                    f"free={inst.free_blocks()}"
                )
                probed += 1
            if inst.advance_one_event(version=0) is None:
                break
            inst.commit_last_event()
            if not inst.running and not inst.waiting:
                break
        assert probed > 20, f"only {probed} states exercised the prefill path"


class TestUndoJournalComposesOverManyMutations:
    """The free list rolls back by replaying inverse operations rather than by
    snapshotting node links, so an event that mutates the same node repeatedly
    must unwind through every step. These drive a pool under real pressure,
    where one event preempts several requests and each eviction both unlinks and
    appends -- the case a single keyed capture per node could get wrong.
    """

    def test_a_preempting_event_restores_the_pool_exactly(self):
        # A pool too small for the batch, so events preempt and each preemption
        # frees a long context: many list mutations inside one event.
        inst = _inst(kv_blocks=200, token_budget=2048)
        for rid in range(12):
            inst.add_request(
                _req(rid=rid, prompt_len=400, source_group_id=rid, target_len=600),
                t=float(rid),
            )
        undone = 0
        for _ in range(200):
            if inst.advance_one_event(version=0) is None:
                break
            inst.commit_last_event()
            before = _instance_state(inst)
            if inst.advance_one_event(version=0) is None:
                break
            inst.undo_last_event()
            assert _instance_state(inst) == before
            undone += 1
            inst.advance_one_event(version=0)
            inst.commit_last_event()
        assert undone > 20, f"only {undone} events were undone"

    def test_undo_after_an_abort_restores_free_list_order(self):
        """`abort_all` frees every resident request and resets the hash map, so
        it is the largest single-event mutation the list sees."""
        inst = _inst(kv_blocks=400)
        for rid in range(10):
            inst.add_request(
                _req(rid=rid, prompt_len=300, source_group_id=rid, target_len=400),
                t=float(rid),
            )
        for _ in range(5):
            if inst.advance_one_event(version=0) is None:
                break
            inst.commit_last_event()
        before = _instance_state(inst)

        inst._save_snapshot()
        inst.abort_all(requeue=True)
        inst.undo_last_event()
        assert _instance_state(inst) == before

    def test_randomised_events_round_trip(self):
        rng = random.Random(1234)
        inst = _inst(kv_blocks=300, token_budget=4096)
        for rid in range(16):
            req = _req(
                rid=rid,
                prompt_len=rng.choice([160, 320, 800]),
                source_group_id=rid // 4,
                target_len=rng.choice([200, 500, 900]),
            )
            inst.add_request(req, t=float(rid))
        checked = 0
        for _ in range(300):
            before = _instance_state(inst)
            event = inst.advance_one_event(version=0)
            if event is None:
                break
            if rng.random() < 0.5:
                inst.undo_last_event()
                assert _instance_state(inst) == before, "undo did not round-trip"
                checked += 1
                event = inst.advance_one_event(version=0)
                if event is None:
                    break
            inst.commit_last_event()
        assert checked > 40, f"only {checked} undos were exercised"
