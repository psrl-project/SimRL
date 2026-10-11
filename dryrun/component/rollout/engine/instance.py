"""Single rollout instance: event-driven with closed-form segment advancement.

Models one inference engine instance with continuous batching, KV cache
accounting, and preemption.
"""

from __future__ import annotations

from ....cost.rollout import RolloutCostModel, advance
from .event import EngineEvent, EventKind
from .kv import KvCacheManager
from .request import ReqStatus, Request
from .snapshot import InstanceSnapshot
from .state import InstanceLoad, StepStat
from .undo import NULL_RECORDER


def blocks_for(n_tokens: int, block_size: int) -> int:
    return (n_tokens + block_size - 1) // block_size


class NativeInstance:
    """A single simulated engine instance."""

    def __init__(
        self,
        cost: RolloutCostModel,
        token_budget: int,
        kv_blocks: int,
        block_size: int = 16,
        max_running: int | None = None,
        max_concurrency: int | None = None,
        chunk_cap: int | None = None,
        instance_id: int = 0,
        tp: int = 1,
        reject_if_kv_full: bool = False,
        reject_if_waiting: bool = False,
        reject_if_running_full: bool = False,
        enable_prefix_caching: bool = True,
    ):
        self.cost = cost
        self.M = token_budget
        self.kv_blocks = kv_blocks
        self.block_size = block_size
        self.max_running = max_running
        self.max_concurrency = max_concurrency
        self.chunk_cap = chunk_cap
        self.instance_id = instance_id
        self.reject_if_kv_full = reject_if_kv_full
        self.reject_if_waiting = reject_if_waiting
        self.reject_if_running_full = reject_if_running_full
        self.tp = tp
        """Tensor-parallel degree this instance runs at.

        Recorded here rather than derived from the cost model because only
        ComposedRollout carries a TP degree; the fitted models do not. A TP-aware
        router reads it, and `DynamicParallelFleet` keys instance reuse on it.
        """
        self.enable_prefix_caching = enable_prefix_caching

        # Occupancy, hit lookup, and eviction all live in the pool. Disabling
        # prefix caching leaves the pool doing block accounting and simply
        # records no hashes, so there is no second code path to keep in step.
        self.kv = KvCacheManager(
            num_blocks=kv_blocks,
            block_size=block_size,
            enable_caching=enable_prefix_caching,
        )

        self.t_local: float = 0.0
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.step_stats: list[StepStat] = []
        self.n_preemptions: int = 0
        self.completed: list[Request] = []
        self._last_decode_blocked: bool = False
        """Whether the last prefill step's decode pass was refused a block.

        The authority on whether the batch is stuck. A full pool is not enough:
        a request mid-block decodes without asking for anything.
        """
        self._snapshot: InstanceSnapshot | None = None
        # The recorder every mutating pool call receives. Swapped for a real
        # log only while a speculative event is in flight.
        self._recorder = NULL_RECORDER

    @property
    def n_active(self) -> int:
        return len(self.waiting) + len(self.running)

    def get_load(self) -> InstanceLoad:
        return InstanceLoad(
            instance_id=self.instance_id,
            n_waiting=len(self.waiting),
            n_running=len(self.running),
            kv_utilization=self.blocks_used() / max(1, self.kv_blocks),
            total_ctx=sum(r.ctx for r in self.running),
            tp=self.tp,
        )

    def blocks_used(self) -> int:
        return self.kv.num_used_blocks

    def free_blocks(self) -> int:
        return self.kv.num_free_blocks

    def committed_blocks(self) -> int:
        """Blocks the resident requests need for the context they already hold.

        Counts queued requests too, which is the point: one that has not been
        prefilled holds no blocks yet but is going to, and a router comparing
        raw `free_blocks` across instances would see no difference between an
        empty instance and one with a hundred requests about to start. This is
        the same sum `add_request` reserves on.
        """
        committed = sum(r.ctx for r in self.waiting + self.running)
        return blocks_for(committed, self.block_size)

    @property
    def has_pending_event(self) -> bool:
        """Whether a speculative event is open."""
        return self._snapshot is not None

    def peek_hit_tokens(self, req: Request) -> int:
        """Tokens a prefix cache hit would cover, without recording a query."""
        return self.kv.peek_hit_tokens(req)

    def _apply_prefix_hit(self, req: Request) -> int:
        """Advance `req.prefilled` over whatever the cache already holds.

        A hit is expressed by moving `prefilled` rather than by a separate
        `num_computed_tokens` field, because both prefill paths already compute
        `prompt_len + generated - prefilled` as the work remaining. The saved
        tokens therefore drop out of `n_tokens` and `t2` on their own, and the
        cost model needs no change at all.

        Args:
            req: A request about to be prefilled, with `prefilled == 0`.

        Returns:
            Tokens the cache covered, 0 when nothing hit.
        """
        if not self.enable_prefix_caching or req.prefilled > 0:
            return 0
        hit_blocks, hit_tokens = self.kv.computed_blocks(req, recorder=self._recorder)
        if hit_tokens <= 0:
            return 0
        if not self.kv.can_allocate(req, n_tokens=1, hit_blocks=hit_blocks):
            return 0
        self.kv.allocate(req, n_tokens=0, hit_blocks=hit_blocks, recorder=self._recorder)
        self._recorder.record_attrs_once(
            ("hit", id(req)), req, "prefilled", "prefix_hit_tokens"
        )
        req.prefilled = hit_tokens
        req.prefix_hit_tokens += hit_tokens
        return hit_tokens

    def _save_snapshot(self) -> None:
        """Open an undo log for the event about to run."""
        self._snapshot = InstanceSnapshot.capture(self)
        self._recorder = self._snapshot.log

    def commit_last_event(self) -> None:
        """Accept the last event, discarding the undo log it recorded into.

        The counterpart to `undo_last_event`. Without it a snapshot outlives
        the event that opened it, with two consequences: the log keeps growing
        as later committed work records into it, and an `undo_last_event`
        reaching that state would roll back work that was already committed.
        Both were latent while the snapshot was only ever read right after an
        event, but a router that asks whether an instance is mid-event needs
        the answer to be true only when one really is.

        Idempotent, so a caller need not track whether an event is open.
        """
        self._snapshot = None
        self._recorder = NULL_RECORDER

    def undo_last_event(self) -> None:
        """Reverse the last event, restoring request, instance and pool state."""
        assert self._snapshot is not None, "No snapshot to undo."
        self._snapshot.restore(self)
        self._snapshot = None
        self._recorder = NULL_RECORDER

    def add_request(self, req: Request, t: float) -> bool:
        """
        Try to admit `req` onto this instance.

        Returns False when the instance is temporarily unable to take more work
        (concurrency / waiting / reserved-KV gates). Raises if the request can
        never fit in this instance's KV cache, even when empty.
        """
        need = blocks_for(req.prompt_len + req.target_len, self.block_size)
        if need > self.kv_blocks:
            raise ValueError(
                f"Request {req.rid} needs {need} KV blocks "
                f"(prompt_len={req.prompt_len}, target_len={req.target_len}, "
                f"block_size={self.block_size}) but instance {self.instance_id} "
                f"only has {self.kv_blocks} blocks. So it can never be admitted."
            )
        if self.max_running is not None and self.n_active >= self.max_running:
            return False
        if self.reject_if_waiting and self.waiting:
            return False
        if self.reject_if_running_full and not self._can_schedule_new():
            return False
        if self.reject_if_kv_full:
            # Off by default. vLLM admits into the waiting queue and lets
            # `_engine_step` decide what fits, so this gate duplicates
            # that limit earlier and counts queued context as already committed.
            committed = sum(r.ctx for r in self.waiting + self.running)
            if blocks_for(committed + req.prompt_len, self.block_size) > self.kv_blocks:
                return False
        req.status = ReqStatus.WAITING
        req.dispatch_time = t
        req.instance_id = self.instance_id
        self.waiting.append(req)
        return True

    def _n_in_batch(self) -> int:
        """vLLM running-queue size: decoding plus already-started prefills."""
        return len(self.running) + sum(1 for r in self.waiting if r.prefilled > 0)

    def _can_run_more(self) -> bool:
        return self.max_concurrency is None or len(self.running) < self.max_concurrency

    def _can_schedule_new(self) -> bool:
        """Whether a not-yet-started waiting request can join the batch.

        Mirrors vLLM `max_num_seqs`: waiting is only pulled into the running
        queue while `len(running) < max_num_seqs`. Mid-prefill requests already
        occupy a slot (`prefilled > 0`).
        """
        return self.max_concurrency is None or self._n_in_batch() < self.max_concurrency

    def _promote_ready(self) -> None:
        for req in list(self.waiting):
            if not self._can_run_more():
                break
            if req.prompt_len + req.generated - req.prefilled <= 0:
                self.waiting.remove(req)
                req.status = ReqStatus.RUNNING
                self.running.append(req)

    def _decode_running_once(self, version: int) -> tuple[list[Request], bool]:
        """Advance every running request by one decode token.

        Served before the waiting queue, as vLLM does, so that a full pool
        starves new admissions rather than the batch already in flight.

        Args:
            version: Weight version producing these tokens.

        Returns:
            The requests that advanced, and whether the pool refused any of
            them. A refusal means the step is KV-bound: the caller admits no new
            prefill work, matching vLLM's `if not preempted_reqs:` guard.
        """
        n_decode = min(len(self.running), self.M)
        decoding = self.running[:n_decode]
        stepped: list[Request] = []
        blocked = False
        # `blocks_for(p + 1) - blocks_for(p) == 1` exactly when
        # `p % block_size == 0`, which is the `grows` flag below. Tracking free
        # blocks incrementally avoids recomputing the occupancy sum per request.
        oldest = (
            min(self.running, key=lambda r: (r.dispatch_time, r.rid))
            if self.running
            else None
        )
        free = self.free_blocks()
        for r in decoding:
            grows = (r.prefilled % self.block_size) == 0
            if grows and free <= 0 and r is not oldest:
                blocked = True
                continue
            if self.kv.allocate(r, n_tokens=1, hit_blocks=[], recorder=self._recorder) is None:
                # The `grows`/`free` test above approximates pool capacity; the
                # pool itself is the authority, so skip rather than advance past
                # what it granted.
                blocked = True
                continue
            r.add_tokens(version, 1)
            r.prefilled += 1
            if grows:
                free -= 1
            stepped.append(r)
        return stepped, blocked

    def _prefill_chunk(self, req: Request, prefilled: int, budget: int, free: int) -> int:
        """Tokens of `req` the next prefill would compute, or 0 for none.

        The one place chunk sizing lives, so the step and its non-mutating twin
        cannot answer differently. Reads the pool but never writes, which is
        what lets `_step_time_estimate` call it.

        Args:
            req: The request about to be prefilled.
            prefilled: Tokens already computed. The twin tracks this locally
                rather than reading `req.prefilled`, which it must not advance.
            budget: Tokens the step has left.
            free: Blocks the caller believes are free, as a running tally.

        Returns:
            The chunk size, clamped by the budget, the chunk cap, the blocks
            the tally allows, and finally by what the pool will actually grant.
        """
        chunk = min(req.prompt_len + req.generated - prefilled, budget)
        if self.chunk_cap:
            chunk = min(chunk, self.chunk_cap)
        have = blocks_for(prefilled, self.block_size)
        chunk = min(chunk, (have + max(0, free)) * self.block_size - prefilled)
        # The pool is the authority on capacity, not the running `free` tally:
        # a hit may have revived blocks, and the tally cannot see sharing.
        # Shrinking here rather than advancing past what the pool grants is
        # what stops the two from disagreeing and overcommitting the cache.
        while chunk > 0 and not self.kv.can_allocate(req, chunk, []):
            chunk -= self.block_size
        return max(0, chunk)

    def _engine_step(self, version: int):
        """Run one engine iteration: decode the running batch, then prefill.

        One iteration, in vLLM's sense of one forward pass. It decodes every
        running request by a token and spends whatever token budget is left
        admitting and prefilling from the queue, so it is not a "prefill step"
        even though only this path prefills. On an oversubscribed pool the
        prefill half usually admits nothing.

        The order is load-bearing. Prefilling first takes the blocks decode
        then needs, so the step overflows, preempts the request it just
        admitted, and repeats at one block of progress. Serving decode first
        means a full pool simply stops admitting.

        Returns:
            `(dt, newly_done)`. `dt` is the iteration's latency. `newly_done`
            are requests the decode pass finished, left for the caller to
            retire so it can shape the event.
        """
        n_d = len(self.running)
        n_decode = min(n_d, self.M)
        budget = self.M - n_decode
        t2 = 0

        prefill_ctx = 0
        prefill_tokens = 0
        n_prefill_reqs = 0
        # Blocks and slots promised to admissions made in this step. They hold
        # nothing yet, so `free_blocks()` cannot see them.
        admitted_blocks = 0
        admitted_count = 0
        finished_prefill: list[Request] = []
        n_in_batch = self._n_in_batch()

        # vLLM's order: "First, schedule the RUNNING requests" then "Next,
        # schedule the WAITING requests" (`vllm/v1/core/sched/scheduler.py`).
        stepped, decode_blocked = self._decode_running_once(version)
        newly_done = [r for r in stepped if r.remaining <= 0]

        free = self.free_blocks()
        prefill_hit = 0
        # vLLM also refuses to admit anything in a step that had to preempt
        # (`if not preempted_reqs:` guards its waiting loop). A pool too full to
        # fund one decode token per running request is in no position to take on
        # new work either.
        # List order is the queue policy, not an accident: `_preempt_one`
        # prepends its victim so the next step readmits it first, as vLLM's
        # `waiting.prepend_request` does. Re-sorting by dispatch time discarded
        # that, so an evicted request went to the back and the step evicted
        # another instead, walking the batch from newest to oldest.
        for req in list(self.waiting):
            if budget <= 0:
                break
            if req.prefilled == 0:
                # A KV-bound step admits no *new* work, as vLLM's waiting loop
                # is guarded by `if not preempted_reqs:`. Requests already
                # mid-prefill are exempt and must stay so: they are holding
                # blocks without decoding, so starving them leaves the pool
                # pinned by requests that can never finish.
                if decode_blocked:
                    break
                if self.max_concurrency is not None and n_in_batch >= self.max_concurrency:
                    break
                # Admission is all-or-nothing: a request enters the batch only
                # if the pool can hold the context it needs, plus one block of
                # headroom per resident request so the batch already running can
                # still take its next decode token.
                #
                # vLLM gets this from the shape of its waiting loop rather than
                # from a reserve. It *peeks* at the head of the queue
                # (`scheduler.py:679`), tries to allocate, and on failure does
                # `if new_blocks is None: break` -- leaving the request queued
                # and popping it only after a successful allocation
                # (`:978`). So a request that does not fit never gets a partial
                # foothold.
                #
                # Without this the engine hands every request a sliver and none
                # can finish: 128 requests needing 78,335 blocks against a
                # 31,294-block pool pin the instance at `running=1,
                # waiting=127`. Capacity allows about 61 residents, and the real
                # engine's measured median running on this workload is 59.
                #
                # Note this is *not* vLLM's `--watermark`, which defaults to 0.0
                # and was not set in the run being matched. That knob reserves a
                # fraction of the whole pool to make preemption rarer; this is
                # the structural admission test that holds even with it off.
                if self.running or admitted_blocks:
                    want = req.prompt_len + req.generated
                    # Earlier admissions in THIS step hold no blocks yet: nothing
                    # is prefilled until below, so `free_blocks()` still reports
                    # the value the first of them saw. Charging them here is what
                    # vLLM gets from `allocate_slots` reserving per request, and
                    # without it a whole backlog passes one gate reading: 129
                    # requests admitted against 1420 free blocks, then evicted one
                    # per step for the rest of the segment.
                    residents = len(self.running) + admitted_count
                    headroom = residents * self.block_size
                    if blocks_for(want, self.block_size) + blocks_for(
                        headroom, self.block_size
                    ) > self.free_blocks() - admitted_blocks:
                        break
                    admitted_blocks += blocks_for(want, self.block_size)
                    admitted_count += 1
                n_in_batch += 1
                prefill_hit += self._apply_prefix_hit(req)
                free = self.free_blocks()
            need = req.prompt_len + req.generated - req.prefilled
            if need <= 0:
                finished_prefill.append(req)
                continue
            have = blocks_for(req.prefilled, self.block_size)
            chunk = self._prefill_chunk(req, req.prefilled, budget, free)
            if chunk <= 0:
                break
            granted = self.kv.allocate(
                req, n_tokens=chunk, hit_blocks=[], recorder=self._recorder
            )
            assert granted is not None, (
                f"KV pool refused {chunk} prefill tokens for request {req.rid} "
                f"after can_allocate approved them."
            )
            free -= blocks_for(req.prefilled + chunk, self.block_size) - have
            # A chunk attends over the keys already written for this request as
            # well as its own, so the extent is `prefilled + chunk`. Charging
            # `chunk` alone prices a resume as if its context were absent, which
            # understates a 12800-token step by 18% at an 11k context.
            extent = req.prefilled + chunk
            prefill_ctx += extent
            req.prefilled += chunk
            prefill_tokens += chunk
            n_prefill_reqs += 1
            t2 += extent * extent
            budget -= chunk
            if req.prefilled >= req.prompt_len + req.generated:
                finished_prefill.append(req)

        ctxsum = sum(r.ctx for r in self.running)
        n_tokens = n_decode + prefill_tokens
        n_reqs = n_d + n_prefill_reqs
        dt = self.cost.step_time(n_tokens, t2, ctxsum, n_reqs, prefill_ctx)
        assert dt > 0, f"Cost model returned non-positive step time: {dt!r}."
        self.step_stats.append(
            StepStat(
                n_tokens,
                n_decode,
                prefill_tokens,
                saturated=self.cost.saturated(n_tokens, t2, ctxsum, n_reqs),
                n_prefill_hit=prefill_hit,
                n_prefill_computed=prefill_tokens,
                steps=1,
                dt=dt,
                phase="mixed",
                # self.t_local is not yet incremented by the caller at this point.
                t=self.t_local + dt,
            )
        )

        for req in finished_prefill:
            if not self._can_run_more():
                break
            self.waiting.remove(req)
            req.status = ReqStatus.RUNNING
            self.running.append(req)
        self._last_decode_blocked = decode_blocked
        return dt, newly_done

    def _decode_batch(self):
        """Return the running batch size and total context, the segment methods' arguments."""
        return len(self.running), sum(r.ctx for r in self.running)

    def step_time_delta(self, req: Request) -> float:
        """How much one more decode step would cost if `req` joined this batch.

        What a throughput-balancing router asks before placing a request. The
        comparison is against the current batch's own step, so the answer is
        the marginal cost rather than the absolute one, and an idle instance
        correctly reports the cost of starting a batch of one.

        Nothing is mutated: the request is priced as if it were resident, but
        it is not added to any queue.

        Args:
            req: The request being considered for placement.

        Returns:
            Predicted increase in one decode step's latency, in seconds, never
            negative.
        """
        n_d, ctxsum = self._decode_batch()
        before = self.cost.step_time(n_d, 0, ctxsum, n_d) if n_d else 0.0
        after = self.cost.step_time(n_d + 1, 0, ctxsum + req.ctx, n_d + 1)
        return max(0.0, after - before)

    def _kv_exhaustion_steps(self) -> int | None:
        """
        Steps of decode until this instance would exceed its KV budget.

        Returns the smallest `k >= 0` such that every running request gaining `k`
        tokens overflows `kv_blocks`, or None when even the longest remaining
        request finishes before that happens.
        """
        B = self.block_size
        # `blocks_for(p) == p // B + blocks_for(p % B)`, so the running batch's
        # growth over k steps is `sum(count * blocks_for(residue + k)) - base`,
        # bucketed by residue modulo B, independent of how many requests there
        # are. Waiting requests contribute nothing: they hold blocks but do not
        # grow during a decode segment.
        #
        # Only the *increments* may be computed from `prefilled`. The intercept
        # must come from the pool, because prefix sharing makes a sum over
        # `blocks_for(prefilled)` an over-estimate -- 896 against a real 134 in
        # the GRPO shape. Using it drives `k` to 0, preempts, lets the request
        # back in, and livelocks with the pool three-quarters empty.
        #
        # The increments stay valid because sharing is fixed at the prompt:
        # siblings diverge from their first generated token, so every block a
        # request grows during decode is private, and the difference of the two
        # `blocks_for` calls is exactly that request's growth.
        residues = [0] * B
        for r in self.running:
            residues[r.prefilled % B] += 1
        base = self.blocks_used()
        if base > self.kv_blocks:
            return 0

        cap = max((r.remaining for r in self.running), default=0)
        if cap == 0:
            return None

        # blocks(k) is periodic-linear rather than merely monotone:
        #     blocks(k + B) == blocks(k) + len(running)   (each request rolls one block)
        #     blocks(s + 1) - blocks(s) == residues[(B - s) % B]
        # So for each phase s in [0, B) the overflow point is the first term of an
        # arithmetic progression above the budget, which is one division. Taking the
        # min over the B phases is the exact answer: no search, and cost depends on
        # block_size alone, not on cap or the batch size.
        n_run = len(self.running)
        target = self.kv_blocks + 1
        best = cap + 1
        blocks_s = base
        for s in range(B):
            need = target - blocks_s
            steps = s if need <= 0 else -(-need // n_run) * B + s
            if steps < best:
                best = steps
            blocks_s += residues[(B - s) % B]
        return best if best <= cap else None

    def _step_time_estimate(self) -> float:
        """
        Compute the step time for the next prefill iteration without mutating state.

        Uses the same allocation logic as `_engine_step` but tracks
        `prefilled` deltas locally so no request field is modified.
        """
        n_d = len(self.running)
        n_decode = min(n_d, self.M)
        budget = self.M - n_decode
        t2 = 0
        prefill_ctx = 0
        prefill_tokens = 0
        free = self.free_blocks()
        effective: dict[int, int] = {}  # id(req) -> effective prefilled
        n_prefill_reqs = 0
        admitted_blocks = 0
        admitted_count = 0
        n_in_batch = self._n_in_batch()

        # List order is the queue policy, not an accident: `_preempt_one`
        # prepends its victim so the next step readmits it first, as vLLM's
        # `waiting.prepend_request` does. Re-sorting by dispatch time discarded
        # that, so an evicted request went to the back and the step evicted
        # another instead, walking the batch from newest to oldest.
        for req in list(self.waiting):
            if budget <= 0:
                break
            if req.prefilled == 0:
                # The same two gates `_engine_step` applies, in the same order.
                # Without them this priced a full 12800-token chunk the step
                # then refuses, reading 0.857 s against a real 0.055 s on a
                # blocked queue, and `advance_to` advanced its clock by that.
                if self._last_decode_blocked:
                    break
                if self.max_concurrency is not None and n_in_batch >= self.max_concurrency:
                    break
                if self.running or admitted_blocks:
                    want = req.prompt_len + req.generated
                    residents = len(self.running) + admitted_count
                    headroom = residents * self.block_size
                    # `free` rather than `self.free_blocks()`: the real step
                    # re-reads a pool its own allocations have drained, while
                    # nothing here writes, so the live value would still report
                    # the blocks earlier requests in this same loop consumed.
                    # The twin then admitted one request more than the step
                    # does, pricing 12800 tokens against a real 5452.
                    if blocks_for(want, self.block_size) + blocks_for(
                        headroom, self.block_size
                    ) > free - admitted_blocks:
                        break
                    admitted_blocks += blocks_for(want, self.block_size)
                    admitted_count += 1
                n_in_batch += 1
            eff = effective.get(id(req), req.prefilled)
            if eff == 0:
                eff = self.kv.peek_hit_tokens(req)
                effective[id(req)] = eff
                # A hit costs blocks in the real step, where `_apply_prefix_hit`
                # allocates them. Peeking alone leaves the tally believing that
                # room is still free, and the twin then prices a chunk the step
                # cannot take: 12800 tokens against a real 1024 on a 5296-token
                # hit, which is 362% of the step.
                free -= blocks_for(eff, self.block_size)
            need = req.prompt_len + req.generated - eff
            if need <= 0:
                continue
            have = blocks_for(eff, self.block_size)
            chunk = self._prefill_chunk(req, eff, budget, free)
            if chunk <= 0:
                break
            free -= blocks_for(eff + chunk, self.block_size) - have
            effective[id(req)] = eff + chunk
            # Mirrors `_engine_step`: the chunk attends over the keys
            # already written for this request as well as its own.
            extent = eff + chunk
            prefill_ctx += extent
            prefill_tokens += chunk
            n_prefill_reqs += 1
            t2 += extent * extent
            budget -= chunk

        ctxsum = sum(r.ctx for r in self.running)
        n_tokens = n_decode + prefill_tokens
        return self.cost.step_time(
            n_tokens, t2, ctxsum, n_d + n_prefill_reqs, prefill_ctx
        )

    def _iteration_event(self, version: int) -> EngineEvent:
        """Run one engine iteration and report what it did."""
        dt, newly = self._engine_step(version)
        self.t_local += dt
        if newly:
            done = self._retire(newly)
            # Retiring frees blocks, so a step that completes may no longer be
            # stuck. Ask again rather than assuming either way, and report both
            # outcomes: the kind the engine loop dispatches on is the
            # completion, but the eviction still happened.
            evicted = self._preempt_for_progress() if self._deadlocked() else []
            return EngineEvent(
                kind=EventKind.REQUEST_COMPLETE,
                instance_id=self.instance_id,
                t=self.t_local,
                completed=done,
                preempted=evicted,
            )
        if self._deadlocked():
            return EngineEvent(
                kind=EventKind.PREEMPT,
                instance_id=self.instance_id,
                t=self.t_local,
                preempted=self._preempt_for_progress(),
            )
        return EngineEvent(
            kind=EventKind.ITERATION_DONE,
            instance_id=self.instance_id,
            t=self.t_local,
        )

    def _decode_segment(self, version: int, k: int) -> None:
        """Advance every running request by `k` decode tokens.

        Args:
            version: Weight version producing these tokens.
            k: Steps to advance. Must be a count the pool can satisfy for every
                running request: `_kv_exhaustion_steps` returns the first
                *overflowing* step, so callers pass `k - 1` when it binds.

        Raises:
            AssertionError: If the pool refuses. Advancing `prefilled` past what
                the pool granted would leave the two disagreeing, which is the
                bug this assert exists to catch rather than to tolerate.
        """
        for r in self.running:
            granted = self.kv.allocate(
                r, n_tokens=k, hit_blocks=[], recorder=self._recorder
            )
            assert granted is not None, (
                f"KV pool refused {k} decode tokens for request {r.rid} "
                f"(prefilled={r.prefilled}, free={self.free_blocks()}). The step "
                f"count must be one the pool can satisfy for every request."
            )
            r.add_tokens(version, k)
            r.prefilled += k

    def _decode_segment_event(self, version: int) -> EngineEvent:
        """Advance one full decode segment (until completion or KV exhaustion) and return the event."""
        n_d, ctxsum = self._decode_batch()
        k_done, _ = advance.next_completion([r.remaining for r in self.running])
        k_kv = self._kv_exhaustion_steps()
        # `_kv_exhaustion_steps` reports the first step that *overflows*, so the
        # last step that fits is one fewer. The old accounting could advance into
        # the overflow and detect it afterwards, because occupancy was recomputed
        # from `prefilled`; the pool is authoritative now, so the segment has to
        # stop short and preempt instead.
        kv_bound = k_kv is not None and k_kv <= k_done
        k = max(0, k_kv - 1) if kv_bound else k_done

        if k > 0:
            self._decode_segment(version, k)
            dt = self.cost.decode_segment_time(n_d, ctxsum, k)
            self.t_local += dt
            saturated_steps = self.cost.decode_saturated_steps(n_d, ctxsum, k)
            self.step_stats.append(
                StepStat(
                    n_tokens=n_d * k,
                    n_decode=n_d * k,
                    n_prefill=0,
                    saturated=saturated_steps > 0,
                    steps=k,
                    saturated_steps=saturated_steps,
                    dt=dt,
                    phase="decode",
                    t=self.t_local,
                )
            )

        if kv_bound:
            return EngineEvent(
                kind=EventKind.PREEMPT,
                instance_id=self.instance_id,
                t=self.t_local,
                preempted=self._preempt_until_decodable(),
            )

        newly = [r for r in self.running if r.remaining <= 0]
        done = self._retire(newly)
        return EngineEvent(
            kind=EventKind.REQUEST_COMPLETE,
            instance_id=self.instance_id,
            t=self.t_local,
            completed=done,
        )

    def advance_one_event(self, version: int) -> EngineEvent | None:
        """
        Advance this instance to its next discrete event.

        Saves a snapshot for undo before mutating state. Returns the event, or
        None if the instance is idle (no waiting or running requests).

        An idle instance drops any snapshot it was holding, so a snapshot exists
        exactly while an uncommitted event is in flight. Without that, an undo
        following an idle call would rewind to whichever event came before,
        rather than being a no-op.

        A non-empty queue takes the iteration path even when nothing in it can
        be admitted, which on an oversubscribed plateau is 99% of iterations.
        Collapsing those into a decode segment is sound for one instance and
        was measured so: identical completions and preemptions, `t_local`
        within 0.016%, 28x fewer events, and the whole aligned run 10x faster.

        It still moves the fleet. `REQUEST_COMPLETE` drives `_undo_and_align`,
        which advances every *other* instance to that moment, so a coarser
        event grid relocates those points and the instances interleave
        differently. Span went +2.3% to +5.2% against the matched run, both
        before and after `_step_time_estimate` was corrected, so the estimate
        was not the cause. Event granularity is part of the multi-instance
        timing model here, not only a cost.
        """
        self._promote_ready()
        if not self.running and not self.waiting:
            self._snapshot = None
            self._recorder = NULL_RECORDER
            return None
        self._save_snapshot()
        if self.waiting:
            return self._iteration_event(version)
        return self._decode_segment_event(version)

    def advance_one_event_committed(self, version: int) -> EngineEvent | None:
        """Advance without recording undo state. The event cannot be rolled back.

        Identical to `advance_one_event` in what it computes, but no snapshot is
        taken and the recorder stays `NULL_RECORDER` throughout, so KV pool and
        free-list mutations carry zero undo bookkeeping cost.

        Use this when the caller knows it will commit this event unconditionally
        — i.e., the event type is not in the undoable set. A later
        `_undo_and_align` that encounters an instance in this state will skip it
        (nothing to undo) and tolerate its clock being ahead of the alignment
        target.
        """
        self._snapshot = None
        self._recorder = NULL_RECORDER
        self._promote_ready()
        if not self.running and not self.waiting:
            return None
        if self.waiting:
            return self._iteration_event(version)
        return self._decode_segment_event(version)

    def run_until(self, t_limit: float, version: int) -> list[Request]:
        done = []
        while self.t_local < t_limit:
            got, reached = self.advance_to(t_limit, version)
            done.extend(got)
            if reached:
                break
            if not self.running and not self.waiting:
                break
        return done

    def advance_to(
        self,
        t_limit: float,
        version: int,
        max_steps: int = 100_000,
    ) -> tuple[list[Request], bool]:
        """
        Advance this instance up to t_limit. Returns (completed, reached_limit).
        """
        done: list[Request] = []
        steps = 0
        while self.t_local < t_limit:
            self._promote_ready()
            if self.waiting:
                if steps >= max_steps:
                    return done, False
                steps += 1
                dt = self._step_time_estimate()
                if self.t_local + dt > t_limit:
                    self.t_local = t_limit
                    return done, True
                _, newly = self._engine_step(version)
                self.t_local += dt
                done.extend(self._retire(newly))
                if newly:
                    return done, False
                if self._deadlocked():
                    if not self._preempt_for_progress():
                        self.t_local = t_limit
                        return done, True
                continue

            if not self.running:
                self.t_local = t_limit
                return done, True

            n_d, ctxsum = self._decode_batch()
            k_done, _ = advance.next_completion([r.remaining for r in self.running])
            k_kv = self._kv_exhaustion_steps()
            # One short of the first overflowing step, as in `_decode_segment_event`:
            # the pool is authoritative, so the segment must not advance into a
            # block it was never granted.
            kv_bound = k_kv is not None and k_kv <= k_done
            k_int = max(0, k_kv - 1) if kv_bound else k_done
            budget = t_limit - self.t_local
            k_budget = self.cost.decode_steps_within(n_d, ctxsum, budget)

            k = min(k_int, k_budget)
            if k > 0:
                self._decode_segment(version, k)
                dt = self.cost.decode_segment_time(n_d, ctxsum, k)
                self.t_local += dt
                saturated_steps = self.cost.decode_saturated_steps(n_d, ctxsum, k)
                self.step_stats.append(
                    StepStat(
                        n_tokens=n_d * k,
                        n_decode=n_d * k,
                        n_prefill=0,
                        saturated=saturated_steps > 0,
                        steps=k,
                        saturated_steps=saturated_steps,
                        dt=dt,
                        phase="decode",
                        t=self.t_local,
                    )
                )

            if self.blocks_used() > self.kv_blocks:
                self._preempt_until_decodable()
                return done, False

            if k == k_budget and k < k_int:
                self.t_local = t_limit
                return done, True

            newly = [r for r in self.running if r.remaining <= 0]
            if newly:
                done.extend(self._retire(newly))
                return done, False
            if kv_bound and k >= k_int:
                self._preempt_until_decodable()
                return done, False
            if k == 0:
                self.t_local = t_limit
                return done, True
        return done, True

    def _deadlocked(self) -> bool:
        """Whether the pool is too full for the batch to make progress.

        A request sitting exactly on a block boundary needs a new block for its
        next token; one mid-block does not. So with no free blocks the step's
        decode throughput is only the mid-block requests, and the boundary ones
        are stalled.

        The test is whether the pool actually refused a decode token, not
        whether some request *could* need one. Asking the latter makes a full
        pool look deadlocked whenever any resident sits on a block boundary,
        which at 175 running has probability 1.0: it fired on 96% of the
        plateau's prefill steps, every one of which decoded 93-182 tokens.
        Each false positive preempted a request, which is the oscillation.

        `_decode_running_once` already reports the real answer, since it is the
        code that asks the pool for those blocks.
        """
        if not self.running or self.free_blocks() > 0:
            return False
        prefill_blocked = any(
            r.prompt_len + r.generated - r.prefilled > 0 for r in self.waiting
        )
        # A full pool with nothing waiting is still stalled if every running
        # request needs a block it cannot get; preemption is the only way out.
        return self._last_decode_blocked and (prefill_blocked or len(self.running) > 1)

    def _preempt_for_progress(self) -> list[Request]:
        return self._preempt_until_decodable()

    def _retire(self, reqs: list[Request]) -> list[Request]:
        out = []
        for r in reqs:
            if r in self.running:
                self.running.remove(r)
            self.kv.free(r, recorder=self._recorder)
            r.status = ReqStatus.DONE
            r.complete_time = self.t_local
            self.completed.append(r)
            out.append(r)
        return out

    def _preempt_one(self, victim: Request | None = None) -> Request | None:
        if not self.running:
            return None
        if victim is None:
            victim = self.running[-1]
        self.running.remove(victim)
        victim.status = ReqStatus.PREEMPTED
        victim.num_preemptions += 1
        victim.reprefill_tokens += victim.ctx
        self.kv.free(victim, recorder=self._recorder)
        victim.prefilled = 0
        self.waiting.insert(0, victim)
        self.n_preemptions += 1
        return victim

    def _preempt_until_decodable(self) -> list[Request]:
        """Evict until the resident batch can fund its next decode token.

        vLLM preempts inside `allocate_slots`'s retry loop, so one scheduler
        step evicts as many requests as the allocation needs rather than one
        (`v1/core/sched/scheduler.py`, the `while True` around `allocate_slots`).
        The headroom asked for is one block per resident sitting on a block
        boundary, which is what the next decode step consumes.

        Returns:
            The requests evicted, in eviction order.
        """
        evicted: list[Request] = []
        while self.running:
            needed = sum(
                1
                for r in self.running
                if (r.prefilled % self.block_size) == 0
            )
            if self.free_blocks() >= needed:
                break
            # The oldest request is never a victim: it is the one guaranteed to
            # finish and release, so evicting it is how a pool-bound instance
            # livelocks.
            oldest = min(
                self.running + self.waiting, key=lambda r: (r.dispatch_time, r.rid)
            )
            victims = [r for r in self.running if r is not oldest]
            if not victims:
                break
            victim = self._preempt_one(max(victims, key=lambda r: (r.dispatch_time, r.rid)))
            if victim is None:
                break
            evicted.append(victim)
        return evicted

    def _release(self, req: Request, keep_tokens: bool = True) -> None:
        req.num_aborts += 1
        if keep_tokens:
            req.reprefill_tokens += req.ctx
        self.kv.free(req, recorder=self._recorder)
        req.prefilled = 0

    def abort_all(self, requeue: bool = True) -> list[Request]:
        """Interrupt every request for a weight update, keeping its tokens but not its KV.

        Args:
            requeue: Whether the aborted requests stay on this instance's
                waiting list. False empties the instance and hands them back,
                which is what a router does so they can be placed again. The
                real gateway re-routes 58.5% of them to a *different*
                instance, so keeping them pinned here is not neutral.

        The request resumes rather than restarting: it keeps `generated` and the
        segments recording which weight version produced what. What it loses is
        the KV cache, and it has to: those tensors were produced by the old
        weights, so reusing them would mix two policies inside one sequence.
        PSRL makes that explicit: `pause_for_sync` calls
        `pause_generation(mode="abort")`, whose default mode reaches
        `scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)` and
        frees every block, and `pull_model` then calls `reset_prefix_cache`,
        which empties the block hash map so nothing can hit on resume.

        Setting `prefilled = 0` is therefore the whole model of a weight update:
        the accumulated context must be re-prefilled before generation can
        continue, and `reprefill_tokens` accumulates how much that cost.

        Verified against a real run of the fixed gateway. Before the smg fix the
        loopback re-dispatched with the *original* prompt and regenerated from
        scratch, which made `prefill_tokens` track the prompt sum (58,280 tokens,
        1.71 s over 10 steps) and wasted 1.7M decode tokens. With the prompt now
        carrying the accumulated output, prefill jumps to 197,097 tokens / 24.4 s
        in two steps and the prefix-cache hit rate collapses from 75% to 9%,
        because `reset_prefix_cache` leaves a long continued prompt with nothing
        to hit against. See `outputs/simulate/SMG_LOOPBACK_FIX.md`.
        """
        aborted = list(self.running) + list(self.waiting)
        for r in aborted:
            self._release(r, keep_tokens=True)
            r.status = ReqStatus.WAITING
        self.waiting = aborted[:] if requeue else []
        self.running = []
        # `_release` already freed each request's blocks. Clearing the hash map
        # is the second, separate step: it is what `pull_model` ->
        # `reset_prefix_cache` does, and it is deliberately *not* done at an
        # agentic round boundary, where `cancel(keep_tokens=True)` frees blocks
        # but leaves their hashes addressable so the next round hits.
        self.kv.reset_hashes(recorder=self._recorder)
        return aborted

    def cancel(
        self,
        reqs: list[Request],
        keep_tokens: bool = True,
        requeue: bool = True,
    ) -> list[Request]:
        """Remove `reqs` from this instance, freeing their KV.

        Args:
            reqs: Requests to remove. Ones this instance does not hold are
                skipped.
            keep_tokens: Whether the generated tokens survive, charging the
                context to `reprefill_tokens` for the eventual resume.
            requeue: Whether a surviving request goes back onto this
                instance's waiting list. False hands ownership to the caller,
                which is what the router does when it reclaims a request to
                place somewhere else. Ignored when `keep_tokens` is False,
                since a request that kept nothing has nothing to resume.
                `instance_id` deliberately survives either way: it records
                where the request last ran, which is what lets a re-placement
                be identified as same-instance or cross-instance. SMG keeps the
                same thing as its `x-base-worker-id` hint.

        Returns:
            The requests actually removed.
        """
        canceled = []
        for r in reqs:
            if r in self.running:
                self.running.remove(r)
            elif r in self.waiting:
                self.waiting.remove(r)
            else:
                continue
            self._release(r, keep_tokens=keep_tokens)
            if keep_tokens:
                r.status = ReqStatus.WAITING
                if requeue:
                    self.waiting.append(r)
            canceled.append(r)
        return canceled
