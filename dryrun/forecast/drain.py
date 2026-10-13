"""Forecast of when every request on one rollout instance finishes.

The forecast replays what `NativeInstance` does, without running it. With nothing queued the engine
advances whole decode segments, so the replay prices each one with the same
`RolloutCostModel.decode_segment_time` call and matches the engine to rounding error. With a queue
the engine iterates, so the replay does too: a decode pass over the running batch, then first-come
prefill chunks out of the remaining token budget.

KV is an upper bound. A prompt shared by siblings is charged once per sibling, while its prefill is
still skipped on a prefix hit. Preemption is not modelled: the forecast reports the peak block usage
it would need and flags `kv_overflow` when that exceeds the pool, after which the finish times are
void.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..component.rollout.engine import blocks_for
from .snapshot import Arrival, InstanceState

# Decode-only iterations are priced every this many steps and interpolated between.
_SAMPLE_STRIDE = 16


@dataclass(frozen=True, slots=True)
class DrainResult:
    """What a forecast says about one instance."""

    finish_times: dict[int, float]
    """Absolute simulated completion time of every resident and arrival, `inf` if it never completes."""
    peak_blocks: int
    kv_overflow: bool
    """Whether the pool would have been exceeded, in which case the engine preempts and the times are void."""


class InstanceForecast:
    """The drain of one instance, with arrivals layered on without touching the state."""

    def __init__(self, state: InstanceState, arrivals: Sequence[Arrival] = ()) -> None:
        self._state = state
        self._arrivals = tuple(arrivals)
        self._baseline: DrainResult | None = None

    @property
    def state(self) -> InstanceState:
        return self._state

    @property
    def arrivals(self) -> tuple[Arrival, ...]:
        return self._arrivals

    def with_arrival(self, arrival: Arrival) -> InstanceForecast:
        """Return the forecast in which `arrival` also reaches the instance."""
        return InstanceForecast(self._state, (*self._arrivals, arrival))

    def finish_times(self, arrivals: Sequence[Arrival] = ()) -> DrainResult:
        """
        Forecast every completion, treating `arrivals` as extra requests.

        Args:
            arrivals (Sequence[Arrival]): Requests that reach the instance at their `arrive_time`,
                on top of the ones this forecast already carries. One dated before the
                instance's `now` is queued at `now`.
        """
        if not arrivals:
            if self._baseline is None:
                self._baseline = _replay(self._state, self._arrivals)
            return self._baseline
        return _replay(self._state, (*self._arrivals, *arrivals))


def _replay(state: InstanceState, arrivals: tuple[Arrival, ...]) -> DrainResult:
    keys = [r.key for r in (*state.waiting, *state.running)] + [a.key for a in arrivals]
    if len(set(keys)) != len(keys):
        raise ValueError("Resident and arrival keys must be unique.")
    return _Drain(state, arrivals).run()


class _Queued:
    """A request waiting for prefill, advanced in place as chunks are scheduled."""

    __slots__ = ("key", "ctx", "prefilled", "remaining", "prompt_key", "prompt_len")

    def __init__(self, key: int, ctx: int, prefilled: int, remaining: int, prompt_key: int | None, prompt_len: int):
        self.key = key
        self.ctx = ctx
        self.prefilled = prefilled
        self.remaining = remaining
        self.prompt_key = prompt_key
        self.prompt_len = prompt_len


class _Drain:
    """One replay of the engine over an `InstanceState`, used once."""

    def __init__(self, state: InstanceState, arrivals: tuple[Arrival, ...]) -> None:
        self.state = state
        self.cost = state.cost
        self.block = state.block_size
        self.t = state.now
        by_remaining = sorted(state.running, key=lambda r: r.remaining)
        # The running batch, ordered by remaining tokens so that completions are a prefix.
        self.keys = [r.key for r in by_remaining]
        self.ctx = np.array([r.ctx for r in by_remaining], dtype=np.int64)
        self.rem = np.array([r.remaining for r in by_remaining], dtype=np.int64)
        self.waiting = [
            _Queued(r.key, r.ctx, r.prefilled, r.remaining, r.prompt_key, r.prompt_len) for r in state.waiting
        ]
        self.pending = sorted(arrivals, key=lambda a: a.arrive_time)
        self.n_arrived = 0
        # Shared prompt tokens whose KV is cached, by prompt identity.
        self.cached: dict[int, int] = {}
        for r in state.running:
            self._note_cached(r.prompt_key, r.prompt_len, r.prompt_len)
        for r in state.waiting:
            self._note_cached(r.prompt_key, r.prompt_len, r.prefilled if r.prefilled else r.hit)
        self.used = state.used_blocks
        self.peak = self.used
        self.finish: dict[int, float] = {}
        self.all_keys = [*(r.key for r in state.waiting), *self.keys, *(a.key for a in arrivals)]

    def run(self) -> DrainResult:
        stuck = False
        while True:
            self._admit_due()
            if not self.keys and not self.waiting:
                if self.n_arrived == len(self.pending):
                    break
                self.t = self.pending[self.n_arrived].arrive_time
                continue
            cut = self.pending[self.n_arrived].arrive_time if self.n_arrived < len(self.pending) else math.inf
            if self.waiting:
                if not self._iterate(cut):
                    stuck = True
                    break
            else:
                self._decode_phase(cut)
        finish = {key: self.finish.get(key, math.inf) for key in self.all_keys}
        return DrainResult(finish, self.peak, stuck or self.peak > self.state.kv_blocks)

    def _note_cached(
        self,
        prompt_key: int | None,
        prompt_len: int,
        computed: int,
        into: dict[int, int] | None = None,
    ) -> None:
        """Record that the first `computed` tokens of a prompt are cached, in whole blocks."""
        if prompt_key is None:
            return
        tokens = min(computed, prompt_len) // self.block * self.block
        target = self.cached if into is None else into
        if tokens > target.get(prompt_key, 0):
            target[prompt_key] = tokens

    def _admit_due(self) -> None:
        while self.n_arrived < len(self.pending) and self.pending[self.n_arrived].arrive_time <= self.t:
            a = self.pending[self.n_arrived]
            self.waiting.append(_Queued(a.key, a.ctx, 0, a.remaining, a.prompt_key, a.prompt_len))
            self.n_arrived += 1

    def _join(self, entries: list[_Queued]) -> None:
        keys = self.keys + [e.key for e in entries]
        ctx = np.concatenate((self.ctx, np.array([e.ctx for e in entries], dtype=np.int64)))
        rem = np.concatenate((self.rem, np.array([e.remaining for e in entries], dtype=np.int64)))
        order = np.argsort(rem, kind="stable")
        self.keys = [keys[i] for i in order]
        self.ctx = ctx[order]
        self.rem = rem[order]

    def _blocks(self, tokens: np.ndarray) -> int:
        b = self.block
        return int(((tokens + (b - 1)) // b).sum())

    def _advance_batch(self, k: int) -> None:
        """Give every running request `k` more tokens and charge the blocks they cross into."""
        grown = self.ctx + k
        self.used += self._blocks(grown) - self._blocks(self.ctx)
        self.peak = max(self.peak, self.used)
        self.ctx = grown
        self.rem = self.rem - k

    def _retire_done(self) -> None:
        """Complete the running requests with nothing left to generate, at the current time."""
        m = int(np.searchsorted(self.rem, 0, side="right"))
        if not m:
            return
        for key in self.keys[:m]:
            self.finish[key] = self.t
        self.used -= self._blocks(self.ctx[:m])
        self.keys = self.keys[m:]
        self.ctx = self.ctx[m:]
        self.rem = self.rem[m:]

    def _decode_phase(self, cut: float) -> None:
        """Run closed-form decode segments until the batch drains or `cut` is reached."""
        n = len(self.keys)
        ctx0 = self.ctx
        rems = self.rem.tolist()
        suffix = np.concatenate((np.cumsum(ctx0[::-1])[::-1], [0])).tolist()
        used_other = self.used - self._blocks(ctx0)
        cost = self.cost
        t = self.t
        first_alive: list[int] = []
        advanced: list[int] = []
        i = 0
        adv = 0
        while i < n:
            n_d = n - i
            ctxsum = suffix[i] + n_d * adv
            r = rems[i]
            dt = cost.decode_segment_time(n_d, ctxsum, r - adv)
            budget = cut - t
            if dt > budget:
                adv += cost.decode_steps_within(n_d, ctxsum, budget)
                t = cut
                first_alive.append(i)
                advanced.append(adv)
                break
            t += dt
            adv = r
            first_alive.append(i)
            advanced.append(adv)
            j = i + 1
            while j < n and rems[j] == r:
                j += 1
            for index in range(i, j):
                self.finish[self.keys[index]] = t
            i = j
        # Blocks held at the end of each segment, before its completions leave.
        b = self.block
        tokens = ctx0[None, :] + np.array(advanced, dtype=np.int64)[:, None]
        alive = np.arange(n)[None, :] >= np.array(first_alive, dtype=np.int64)[:, None]
        held = np.where(alive, (tokens + (b - 1)) // b, 0).sum(axis=1)
        self.peak = max(self.peak, used_other + int(held.max()))
        self.ctx = ctx0[i:] + adv
        self.rem = self.rem[i:] - adv
        self.keys = self.keys[i:]
        self.used = used_other + self._blocks(self.ctx)
        self.t = t

    def _iterate(self, cut: float) -> bool:
        """Run one engine iteration while a queue exists. Returns False when the instance cannot progress."""
        st = self.state
        b = self.block
        n_d = len(self.keys)
        if n_d > st.token_budget:
            raise ValueError("The running batch outgrew the token budget, which is not modelled.")
        budget = st.token_budget - n_d
        grow = int(np.count_nonzero(self.ctx % b == 0)) if n_d else 0
        free_after_decode = st.kv_blocks - self.used - grow
        free = free_after_decode
        in_batch = n_d + sum(1 for w in self.waiting if w.prefilled > 0)
        admitted = 0
        # Each entry is `(request, tokens computed once any prefix hit applies, chunk)`. Nothing is
        # written to the replay until the iteration is known to finish before `cut`.
        plan: list[tuple[_Queued, int, int]] = []
        joiners: list[_Queued] = []
        overlay: dict[int, int] = {}
        n_chunks = prefill_ctx = t2 = tokens = 0
        for w in self.waiting:
            if budget <= 0:
                break
            start = w.prefilled
            if start == 0:
                if st.max_concurrency is not None and in_batch >= st.max_concurrency:
                    break
                if n_d or admitted:
                    need_blocks = blocks_for(w.ctx, b)
                    if need_blocks > free - admitted:
                        break
                    admitted += need_blocks
                in_batch += 1
                start = self._prefix_hit(w, overlay)
                free -= blocks_for(start, b)
            have = blocks_for(start, b)
            chunk = min(w.ctx - start, budget)
            if st.chunk_cap:
                chunk = min(chunk, st.chunk_cap)
            chunk = min(chunk, (have + max(0, free)) * b - start)
            if chunk <= 0:
                plan.append((w, start, 0))
                break
            free -= blocks_for(start + chunk, b) - have
            extent = start + chunk
            prefill_ctx += extent
            t2 += extent * extent
            tokens += chunk
            n_chunks += 1
            budget -= chunk
            plan.append((w, start, chunk))
            self._note_cached(w.prompt_key, w.prompt_len, extent, into=overlay)
            if extent >= w.ctx:
                joiners.append(w)
        changed = any(start != w.prefilled or chunk for w, start, chunk in plan)
        if not changed:
            return self._blocked_stretch(cut)
        ctxsum = int(self.ctx.sum()) + n_d
        dt = self.cost.step_time(n_d + tokens, t2, ctxsum, n_d + n_chunks, prefill_ctx)
        if self.t + dt > cut:
            self.t = cut
            return True
        self.t += dt
        if n_d:
            self._advance_batch(1)
        self.used += free_after_decode - free
        self.peak = max(self.peak, self.used)
        for w, start, chunk in plan:
            w.prefilled = start + chunk
        for key, cached in overlay.items():
            if cached > self.cached.get(key, 0):
                self.cached[key] = cached
        self._retire_done()
        if joiners:
            joined = {id(w) for w in joiners}
            self.waiting = [w for w in self.waiting if id(w) not in joined]
            self._join(joiners)
        return True

    def _prefix_hit(self, w: _Queued, overlay: dict[int, int]) -> int:
        """Tokens of `w` the prefix cache covers when it is admitted, whole blocks only."""
        if w.prompt_key is None:
            return 0
        hit = max(self.cached.get(w.prompt_key, 0), overlay.get(w.prompt_key, 0))
        # A full hit hands the last block back, so that one token is still computed.
        if hit and hit >= w.ctx:
            hit -= self.block
        return hit

    def _blocked_stretch(self, cut: float) -> bool:
        """Advance iterations in which the queue gets nothing, up to the next completion or `cut`."""
        n_d = len(self.keys)
        if not n_d:
            return False
        k_done = int(self.rem[0])
        # The engine prices an iteration at the context after its own decode pass.
        first = int(self.ctx.sum()) + n_d
        elapsed = np.cumsum(self._iteration_costs(n_d, first, k_done))
        budget = cut - self.t
        if elapsed[-1] <= budget:
            self.t += float(elapsed[-1])
            self._advance_batch(k_done)
            self._retire_done()
            return True
        self._advance_batch(int(np.searchsorted(elapsed, budget, side="right")))
        self.t = cut
        return True

    def _iteration_costs(self, n_d: int, first: int, k: int) -> np.ndarray:
        """
        Price each of `k` decode-only iterations, sampling every `_SAMPLE_STRIDE` steps and interpolating.

        The engine takes these one step at a time, so a closed-form segment would not match it.
        """
        nodes = list(range(k)) if k <= _SAMPLE_STRIDE else list(range(0, k, _SAMPLE_STRIDE))
        if nodes[-1] != k - 1:
            nodes.append(k - 1)
        step_time = self.cost.step_time
        values = [step_time(n_d, 0, first + n_d * i, n_d, 0) for i in nodes]
        return np.interp(np.arange(k), nodes, values)
