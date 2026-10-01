"""Single rollout instance: event-driven with closed-form segment advancement.

Models one inference engine instance with continuous batching, KV cache
accounting, and preemption.
"""

from __future__ import annotations

from ....cost.rollout import RolloutCostModel, advance
from .event import EngineEvent, EventKind
from .request import ReqStatus, Request
from .snapshot import InstanceSnapshot
from .state import InstanceLoad, StepStat


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
        reject_if_kv_full: bool = True,
        reject_if_waiting: bool = False,
        reject_if_running_full: bool = False,
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

        self.t_local: float = 0.0
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.step_stats: list[StepStat] = []
        self.n_preemptions: int = 0
        self.completed: list[Request] = []
        self._snapshot: InstanceSnapshot | None = None

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
        )

    def blocks_used(self) -> int:
        return sum(blocks_for(r.prefilled, self.block_size) for r in self.waiting + self.running)

    def free_blocks(self) -> int:
        return self.kv_blocks - self.blocks_used()

    def _save_snapshot(self) -> None:
        """Save current state as the undo point for the next event."""
        self._snapshot = InstanceSnapshot.capture(self)

    def undo_last_event(self) -> None:
        """Restore state from the last saved snapshot, reversing the last event."""
        assert self._snapshot is not None, "No snapshot to undo."
        self._snapshot.restore(self)
        self._snapshot = None

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
            # vLLM's admission gate reserves `request.num_tokens`: the tokens it
            # knows about so far, not the sampling cap. Reserving
            # `prompt + target_len` instead charges every request its worst case up
            # front, which admits materially fewer requests than the real engine.
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

    def _prefill_budget_step(self, version: int):
        n_d = len(self.running)
        n_decode = min(n_d, self.M)
        decoding = self.running[:n_decode]
        budget = self.M - n_decode
        t2 = 0
        prefill_tokens = 0
        n_prefill_reqs = 0
        finished_prefill: list[Request] = []
        n_in_batch = self._n_in_batch()

        free = self.free_blocks()
        for req in sorted(self.waiting, key=lambda r: (r.dispatch_time, r.rid)):
            if budget <= 0:
                break
            if req.prefilled == 0:
                if self.max_concurrency is not None and n_in_batch >= self.max_concurrency:
                    break
                n_in_batch += 1
            need = req.prompt_len + req.generated - req.prefilled
            if need <= 0:
                finished_prefill.append(req)
                continue
            chunk = min(need, budget)
            if self.chunk_cap:
                chunk = min(chunk, self.chunk_cap)
            have = blocks_for(req.prefilled, self.block_size)
            chunk = min(chunk, (have + max(0, free)) * self.block_size - req.prefilled)
            if chunk <= 0:
                break
            free -= blocks_for(req.prefilled + chunk, self.block_size) - have
            req.prefilled += chunk
            prefill_tokens += chunk
            n_prefill_reqs += 1
            t2 += chunk * chunk
            budget -= chunk
            if req.prefilled >= req.prompt_len + req.generated:
                finished_prefill.append(req)

        ctxsum = sum(r.ctx for r in self.running)
        n_tokens = n_decode + prefill_tokens
        n_reqs = n_d + n_prefill_reqs
        dt = self.cost.step_time(n_tokens, t2, ctxsum, n_reqs)
        assert dt > 0, f"Cost model returned non-positive step time: {dt!r}."
        self.step_stats.append(
            StepStat(
                n_tokens,
                n_decode,
                prefill_tokens,
                self.cost.saturated(n_tokens, t2, ctxsum, n_reqs),
                steps=1,
                dt=dt,
                phase="prefill",
                # self.t_local is not yet incremented by the caller at this point.
                t=self.t_local + dt,
            )
        )

        stepped = []
        oldest = min(self.running, key=lambda r: (r.dispatch_time, r.rid)) if self.running else None
        # blocks_for(p + 1) - blocks_for(p) == 1 exactly when p % block_size == 0,
        # which is the `grows` flag below. So track free blocks incrementally
        # instead of recomputing the O(len(queues)) occupancy sum per request.
        free = self.free_blocks()
        for r in decoding:
            grows = (r.prefilled % self.block_size) == 0
            if grows and free <= 0 and r is not oldest:
                continue
            r.add_tokens(version, 1)
            r.prefilled += 1
            if grows:
                free -= 1
            stepped.append(r)
        newly_done = [r for r in stepped if r.remaining <= 0]

        for req in finished_prefill:
            if not self._can_run_more():
                break
            self.waiting.remove(req)
            req.status = ReqStatus.RUNNING
            self.running.append(req)
        return dt, newly_done

    def _decode_batch(self):
        """Return the running batch size and total context, the segment methods' arguments."""
        return len(self.running), sum(r.ctx for r in self.running)

    def _kv_exhaustion_steps(self) -> int | None:
        """
        Steps of decode until this instance would exceed its KV budget.

        Returns the smallest `k >= 0` such that every running request gaining `k`
        tokens overflows `kv_blocks`, or None when even the longest remaining
        request finishes before that happens.
        """
        B = self.block_size
        # blocks_for(p) == p // B + blocks_for(p % B), so occupancy at step k is
        #     blocks(k) = fixed + sum(count * blocks_for(residue + k))
        # over the B residue buckets, independent of how many requests there are.
        # blocks(0) is exactly what `blocks_used()` returns, so `base` below is
        # that value and does not need computing twice.
        fixed = sum(blocks_for(r.prefilled, B) for r in self.waiting)
        residues = [0] * B
        for r in self.running:
            fixed += r.prefilled // B
            residues[r.prefilled % B] += 1
        base = fixed + sum(c * blocks_for(res, B) for res, c in enumerate(residues) if c)
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

        Uses the same allocation logic as `_prefill_budget_step` but tracks
        `prefilled` deltas locally so no request field is modified.
        """
        n_d = len(self.running)
        n_decode = min(n_d, self.M)
        budget = self.M - n_decode
        t2 = 0
        prefill_tokens = 0
        free = self.free_blocks()
        effective: dict[int, int] = {}  # id(req) -> effective prefilled
        n_prefill_reqs = 0
        n_in_batch = self._n_in_batch()

        for req in sorted(self.waiting, key=lambda r: (r.dispatch_time, r.rid)):
            if budget <= 0:
                break
            if req.prefilled == 0:
                if self.max_concurrency is not None and n_in_batch >= self.max_concurrency:
                    break
                n_in_batch += 1
            eff = effective.get(id(req), req.prefilled)
            need = req.prompt_len + req.generated - eff
            if need <= 0:
                continue
            chunk = min(need, budget)
            if self.chunk_cap:
                chunk = min(chunk, self.chunk_cap)
            have = blocks_for(eff, self.block_size)
            chunk = min(chunk, (have + max(0, free)) * self.block_size - eff)
            if chunk <= 0:
                break
            free -= blocks_for(eff + chunk, self.block_size) - have
            effective[id(req)] = eff + chunk
            prefill_tokens += chunk
            n_prefill_reqs += 1
            t2 += chunk * chunk
            budget -= chunk

        ctxsum = sum(r.ctx for r in self.running)
        n_tokens = n_decode + prefill_tokens
        return self.cost.step_time(n_tokens, t2, ctxsum, n_d + n_prefill_reqs)

    def _prefill_one_event(self, version: int) -> EngineEvent:
        """Execute one prefill step and return the resulting event."""
        dt, newly = self._prefill_budget_step(version)
        self.t_local += dt
        if newly:
            done = self._retire(newly)
            return EngineEvent(
                kind=EventKind.REQUEST_COMPLETE,
                instance_id=self.instance_id,
                t=self.t_local,
                completed=done,
            )
        if self._deadlocked():
            self._preempt_for_progress()
            return EngineEvent(
                kind=EventKind.DEADLOCK_PREEMPT,
                instance_id=self.instance_id,
                t=self.t_local,
            )
        return EngineEvent(
            kind=EventKind.PREFILL_STEP,
            instance_id=self.instance_id,
            t=self.t_local,
        )

    def _decode_one_event(self, version: int) -> EngineEvent:
        """Advance one full decode segment (until completion or KV exhaustion) and return the event."""
        n_d, ctxsum = self._decode_batch()
        k_done, _ = advance.next_completion([r.remaining for r in self.running])
        k_kv = self._kv_exhaustion_steps()
        k = k_done if k_kv is None else min(k_done, k_kv)

        if k > 0:
            for r in self.running:
                r.add_tokens(version, k)
                r.prefilled += k
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

        if self.blocks_used() > self.kv_blocks or (k_kv is not None and k >= k_kv):
            self._preempt_one()
            return EngineEvent(
                kind=EventKind.KV_PREEMPT,
                instance_id=self.instance_id,
                t=self.t_local,
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
        """
        self._promote_ready()
        if not self.running and not self.waiting:
            return None
        self._save_snapshot()
        if self.waiting:
            return self._prefill_one_event(version)
        return self._decode_one_event(version)

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
                _, newly = self._prefill_budget_step(version)
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
            k_int = k_done if k_kv is None else min(k_done, k_kv)
            budget = t_limit - self.t_local
            k_budget = self.cost.decode_steps_within(n_d, ctxsum, budget)

            k = min(k_int, k_budget)
            if k > 0:
                for r in self.running:
                    r.add_tokens(version, k)
                    r.prefilled += k
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
                self._preempt_one()
                return done, False

            if k == k_budget and k < k_int:
                self.t_local = t_limit
                return done, True

            newly = [r for r in self.running if r.remaining <= 0]
            if newly:
                done.extend(self._retire(newly))
                return done, False
            if k_kv is not None and k >= k_kv:
                self._preempt_one()
                return done, False
            if k == 0:
                self.t_local = t_limit
                return done, True
        return done, True

    def _deadlocked(self) -> bool:
        if not self.running or self.free_blocks() > 0:
            return False
        decode_blocked = all((r.prefilled % self.block_size) == 0 for r in self.running)
        prefill_blocked = any(
            r.prompt_len + r.generated - r.prefilled > 0 for r in self.waiting
        )
        return decode_blocked and prefill_blocked

    def _preempt_for_progress(self) -> bool:
        if not self.running:
            return False
        oldest = min(self.running + self.waiting, key=lambda r: (r.dispatch_time, r.rid))
        victims = [r for r in self.running if r is not oldest]
        if not victims:
            return False
        victim = max(victims, key=lambda r: (r.dispatch_time, r.rid))
        self._preempt_one(victim)
        return True

    def _retire(self, reqs: list[Request]) -> list[Request]:
        out = []
        for r in reqs:
            if r in self.running:
                self.running.remove(r)
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
        victim.prefilled = 0
        self.waiting.insert(0, victim)
        self.n_preemptions += 1
        return victim

    def _release(self, req: Request, keep_tokens: bool = True) -> None:
        req.num_aborts += 1
        if keep_tokens:
            req.reprefill_tokens += req.ctx
        req.prefilled = 0

    def abort_all(self) -> list[Request]:
        """Interrupt every request for a weight update, keeping its tokens but not its KV.

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
        self.waiting = aborted[:]
        self.running = []
        return aborted

    def cancel(self, reqs: list[Request], keep_tokens: bool = True) -> list[Request]:
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
                self.waiting.append(r)
            canceled.append(r)
        return canceled
