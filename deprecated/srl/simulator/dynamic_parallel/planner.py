"""Length-aware fleet planner: choose the TP shape and the length tiers in one step.

A plan is a set of tiers. Each tier is a group of identical instances that serves
a contiguous range of estimated remaining length, longest range on the highest TP.
Every solver prices a candidate with the same drain model, the completion-time
formula of a replica whose requests finish one by one:

    T = sum_j tau(b_j, ctx_j) * (L_j - L_{j-1})

where `b_j` requests are still decoding during the j-th shortest request's final
stretch and `tau` is the per-token step time of the instance's own cost model.

Three solvers share that model:

- ``dp``: Libra-style dynamic program over (GPUs used, sorted-request prefix).
  Chooses the shape and the partition together.
- ``pp``: enumerate shapes, then solve the ordered contiguous partition exactly
  with a parametric search. The cost of a segment grows with the segment, so a
  greedy fill against a trial makespan is a feasibility test.
- ``twopool``: TailSieve-style isolation. One tail pool on the highest TP and one
  bulk pool on a single lower TP, balanced by bisecting the isolation size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

import numpy as np

from ..._logging import srl_logger
from .plan import FleetPlan

if TYPE_CHECKING:
    from ...cost.rollout.base import RolloutCostModel

_B_GRID = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024], dtype=float)
_C_GRID = np.array(
    [0, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 65536],
    dtype=float,
)
_CAP_GRID = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384)
_CLASS_DROP = 1.25
"""Sorted-length ratio above which the DP starts a new class."""
_ALGORITHMS = ("dp", "pp", "twopool")


class StepCurve:
    """Per-token decode time of one TP degree as a function of batch and context."""

    def __init__(self, cost_model: "RolloutCostModel", tp: int, kv_tokens: int) -> None:
        self.tp = tp
        self.kv_tokens = kv_tokens
        table = np.empty((len(_B_GRID), len(_C_GRID)))
        for i, b in enumerate(_B_GRID):
            for j, c in enumerate(_C_GRID):
                n = int(b)
                table[i, j] = cost_model.step_time(n, 0, int(b * c), n)
        self._table = table
        self._log_b = np.log(_B_GRID)

    def tau(self, b: np.ndarray | float, mean_ctx: np.ndarray | float) -> np.ndarray:
        """Step seconds for `b` decoding requests of mean context `mean_ctx`."""
        b = np.clip(np.asarray(b, dtype=float), 1.0, _B_GRID[-1])
        c = np.clip(np.asarray(mean_ctx, dtype=float), 0.0, _C_GRID[-1])
        lb = np.log(b)
        i = np.clip(np.searchsorted(self._log_b, lb, side="right") - 1, 0, len(_B_GRID) - 2)
        j = np.clip(np.searchsorted(_C_GRID, c, side="right") - 1, 0, len(_C_GRID) - 2)
        fb = (lb - self._log_b[i]) / (self._log_b[i + 1] - self._log_b[i])
        fc = (c - _C_GRID[j]) / (_C_GRID[j + 1] - _C_GRID[j])
        t = self._table
        return (
            t[i, j] * (1 - fb) * (1 - fc)
            + t[i, j + 1] * (1 - fb) * fc
            + t[i + 1, j] * fb * (1 - fc)
            + t[i + 1, j + 1] * fb * fc
        )

    def drain(
        self,
        ctx0: np.ndarray,
        rem: np.ndarray,
        cap: int | None = None,
        weight: np.ndarray | None = None,
        slack: np.ndarray | None = None,
    ) -> float:
        """Seconds for one instance to finish its requests, all admitted together.

        With `slack` the result is ``max_i finish_i / (1 + slack_i)``, so a request
        that may take extra training cycles to finish counts for proportionally less.
        With no slack it is the drain time of the whole instance.

        Args:
            ctx0: Context each request starts with.
            rem: Tokens each request still has to generate.
            cap: Resident cap. When fewer requests fit than are offered, the result
                is lifted to the work-conserving bound at the cap.
            weight: Multiplicity of each row, for a class of identical requests.
            slack: Extra training cycles each request may take.
        """
        n_rows = len(rem)
        if n_rows == 0:
            return 0.0
        w = np.ones(n_rows) if weight is None else np.asarray(weight, dtype=float)
        order = np.argsort(rem, kind="stable")
        s = rem[order]
        c0 = ctx0[order]
        w = w[order]
        active = np.cumsum(w[::-1])[::-1]
        suffix_ctx = np.cumsum((w * c0)[::-1])[::-1]
        prev = np.concatenate(([0.0], s[:-1]))
        mid = 0.5 * (prev + s)
        phase = (s - prev) * self.tau(active, suffix_ctx / active + mid)
        seconds = float(np.sum(phase))
        total = float(active[0])
        limit = self.kv_tokens / max(float(np.sum(w * (c0 + s))) / total, 1.0)
        if cap is not None:
            limit = min(limit, float(cap))
        lift = 1.0
        if total > limit:
            work = float(np.sum(w * s))
            life_ctx = float(np.sum(w * (c0 + 0.5 * s))) / total
            rate = limit / float(self.tau(limit, life_ctx))
            lifted = max(seconds, work / rate)
            lift = lifted / max(seconds, 1e-12)
            seconds = lifted
        if slack is None:
            return seconds
        finish = np.cumsum(phase) * lift
        return float(np.max(finish / (1.0 + np.asarray(slack, dtype=float)[order])))

    def simulate(
        self,
        ctx0: np.ndarray,
        rem: np.ndarray,
        cap: int,
        slack: np.ndarray | None = None,
    ) -> float:
        """Event-driven drain of one instance with a resident cap.

        Requests are admitted in array order, longest first, whenever fewer than
        `cap` are resident and the KV pool holds their expected lifetime context.
        Each event runs the resident batch at its current step time until the next
        request finishes. Returns the makespan, or ``max_i finish_i / (1 + slack_i)``.
        """
        n = len(rem)
        if n == 0:
            return 0.0
        ctx0 = np.asarray(ctx0, dtype=float)
        rem = np.asarray(rem, dtype=float)
        weight = np.ones(n) if slack is None else 1.0 + np.asarray(slack, dtype=float)
        nxt = 0
        active: list[int] = []
        left = np.zeros(n)
        done_tok = np.zeros(n)
        now = 0.0
        worst = 0.0
        kv_limit = 0.98 * self.kv_tokens
        while nxt < n or active:
            reserved = float(np.sum(ctx0[active] + 0.5 * rem[active])) if active else 0.0
            while nxt < n and len(active) < cap:
                need = ctx0[nxt] + 0.5 * rem[nxt]
                if active and reserved + need > kv_limit:
                    break
                active.append(nxt)
                left[nxt] = rem[nxt]
                reserved += need
                nxt += 1
            idx = np.array(active)
            b = len(idx)
            step_tokens = float(np.min(left[idx]))
            mean_ctx = float(np.mean(ctx0[idx] + done_tok[idx] + 0.5 * step_tokens))
            now += step_tokens * float(self.tau(b, mean_ctx))
            left[idx] -= step_tokens
            done_tok[idx] += step_tokens
            finished = idx[left[idx] <= 1e-9]
            if len(finished):
                worst = max(worst, float(np.max(now / weight[finished])))
            active = [i for i in active if left[i] > 1e-9]
        return worst


@dataclass(frozen=True)
class Tier:
    """Instances that serve one length range."""

    tp: int
    instances: tuple[int, ...]
    lo: float
    """Inclusive lower bound on estimated remaining tokens."""
    hi: float
    """Exclusive upper bound on estimated remaining tokens."""
    cap: int
    """Resident requests one instance of the tier may hold."""
    n_items: int
    predicted_s: float


@dataclass(frozen=True)
class TierPlan(FleetPlan):
    """A fleet shape plus the length tier each group of instances serves."""

    tiers: tuple[Tier, ...] = ()
    predicted_makespan_s: float = 0.0
    algorithm: str = ""
    assignment: dict = field(default_factory=dict, compare=False, hash=False, repr=False)
    """Request id to tier index for the requests the planner saw. Ties at a tier
    boundary cannot be split by a length threshold, so the planner's own mapping wins."""

    def describe(self) -> str:
        shape = super().describe()
        if not self.tiers:
            return shape
        spans = ", ".join(
            f"TP{t.tp}x{len(t.instances)}[{t.lo:.0f},{t.hi:.0f}) n={t.n_items} cap={t.cap} t={t.predicted_s:.0f}s"
            for t in self.tiers
        )
        return f"{shape} | {spans}"


def _enumerate_shapes(n_gpus: int, tps: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Every multiset of TP degrees, descending, that spends the budget exactly."""
    ordered = tuple(sorted(tps, reverse=True))
    shapes: list[tuple[int, ...]] = []
    for size in range(1, n_gpus // min(ordered) + 1):
        for combo in combinations_with_replacement(ordered, size):
            if sum(combo) == n_gpus:
                shapes.append(tuple(sorted(combo, reverse=True)))
    return shapes


class TierPlanner:
    """Plans a heterogeneous fleet and its length tiers from estimated work."""

    def __init__(
        self,
        cost_models: dict[int, "RolloutCostModel"],
        kv_tokens_per_tp: dict[int, int],
        n_gpus: int,
        tp_candidates: tuple[int, ...],
        algorithm: str = "pp",
        max_concurrency: int = 1024,
        dp_classes: int = 64,
        switch_margin: float = 0.0,
    ) -> None:
        """
        Args:
            cost_models: TP degree to rollout cost model.
            kv_tokens_per_tp: TP degree to KV capacity of one instance, in tokens.
            n_gpus: GPU budget every plan spends exactly.
            tp_candidates: TP degrees a plan may use.
            algorithm: ``dp``, ``pp`` or ``twopool``.
            max_concurrency: Engine cap on resident requests.
            dp_classes: Upper bound on length classes the DP partitions over.
            switch_margin: A heterogeneous plan must beat the best uniform fleet by
                this fraction of its predicted makespan, or the uniform fleet is
                kept. Guards against chasing gains smaller than the model's error.

        Raises:
            ValueError: If the algorithm is unknown or a model is missing.
        """
        if algorithm not in _ALGORITHMS:
            raise ValueError(f"Unknown planner algorithm {algorithm!r}, expected one of {_ALGORITHMS}.")
        missing = sorted(set(tp_candidates) - set(cost_models))
        if missing:
            raise ValueError(f"No cost model for TP {missing}.")
        self._tps = tuple(sorted({int(t) for t in tp_candidates}, reverse=True))
        self._curves = {tp: StepCurve(cost_models[tp], tp, kv_tokens_per_tp[tp]) for tp in self._tps}
        self._n_gpus = int(n_gpus)
        self._algorithm = algorithm
        self._max_concurrency = int(max_concurrency)
        self._dp_classes = int(dp_classes)
        self._switch_margin = float(switch_margin)
        self._shapes = _enumerate_shapes(self._n_gpus, self._tps)
        self._ids_sorted: np.ndarray | None = None
        self._slack_sorted: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def shapes(self) -> list[tuple[int, ...]]:
        return list(self._shapes)

    def plan(
        self,
        ctx0: np.ndarray,
        rem: np.ndarray,
        algorithm: str | None = None,
        ids: np.ndarray | None = None,
        slack: np.ndarray | None = None,
    ) -> TierPlan:
        """Return the plan that minimizes the predicted makespan of `rem`.

        Args:
            ctx0: Starting context of each request.
            rem: Estimated remaining tokens of each request.
            algorithm: Overrides the planner's default for this call.
            ids: Request id of each row, to return the planner's own assignment.
            slack: Extra training cycles each request may take. A request with slack
                `s` is ordered and priced as if it were `1 + s` times shorter.
        """
        algo = algorithm or self._algorithm
        ctx0 = np.asarray(ctx0, dtype=float)
        rem = np.maximum(np.asarray(rem, dtype=float), 1.0)
        if len(rem) == 0:
            return self._uniform_plan(self._tps[-1], algo)
        slack_arr = None if slack is None else np.asarray(slack, dtype=float)
        if slack_arr is not None and not np.any(slack_arr > 0):
            slack_arr = None
        urgency = rem if slack_arr is None else rem / (1.0 + slack_arr)
        order = np.argsort(-urgency, kind="stable")
        ctx_sorted, rem_sorted = ctx0[order], rem[order]
        self._ids_sorted = None if ids is None else np.asarray(ids)[order]
        self._slack_sorted = None if slack_arr is None else slack_arr[order]
        if algo == "dp":
            chosen = self._plan_dp(ctx_sorted, rem_sorted)
        elif algo == "pp":
            chosen = self._plan_pp(ctx_sorted, rem_sorted)
        else:
            chosen = self._plan_twopool(ctx_sorted, rem_sorted)
        return self._guard_uniform(chosen, ctx_sorted, rem_sorted, algo)

    def _guard_uniform(self, chosen: TierPlan, ctx: np.ndarray, rem: np.ndarray, algo: str) -> TierPlan:
        """Keep the best uniform fleet unless `chosen` beats it by the switch margin."""
        if self._switch_margin <= 0.0 or not chosen.is_heterogeneous:
            return chosen
        best_t, best_seg = float("inf"), None
        for tp in self._tps:
            if self._n_gpus % tp:
                continue
            t, seg = self._pp_for_shape((tp,) * (self._n_gpus // tp), ctx, rem)
            if t < best_t:
                best_t, best_seg = t, seg
        uniform = self._assemble(best_seg, ctx, rem, algo)
        bar = uniform.predicted_makespan_s * (1.0 - self._switch_margin)
        if bar <= chosen.predicted_makespan_s:
            return uniform
        return chosen

    def predicted_time(self, shape: tuple[int, ...], ctx0: np.ndarray, rem: np.ndarray) -> float:
        """Makespan of `shape` under the parametric partition, for comparisons."""
        order = np.argsort(-np.asarray(rem), kind="stable")
        self._slack_sorted = None
        rem_sorted = np.maximum(np.asarray(rem, float)[order], 1.0)
        return self._pp_for_shape(shape, np.asarray(ctx0, float)[order], rem_sorted)[0]

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _uniform_plan(self, tp: int, algo: str) -> TierPlan:
        n = self._n_gpus // tp
        fleet = (tp,) * n
        tier = Tier(tp, tuple(range(n)), 0.0, float("inf"), self._max_concurrency, 0, 0.0)
        return TierPlan(tp_per_instance=fleet, n_gpus=self._n_gpus, tiers=(tier,), algorithm=algo)

    def _tier_time(
        self,
        tp: int,
        k: int,
        ctx: np.ndarray,
        rem: np.ndarray,
        cap: int | None = None,
        slack: np.ndarray | None = None,
    ) -> float:
        """Drain time of a tier of `k` identical instances sharing `ctx`/`rem`.

        Rows arrive longest first and are dealt round-robin, so instance 0 holds the
        largest request and bounds the tier.
        """
        if len(rem) == 0:
            return 0.0
        curve = self._curves[tp]
        cap = self._max_concurrency if cap is None else min(cap, self._max_concurrency)
        worst = 0.0
        for j in range(min(k, 3)):
            sl = None if slack is None else slack[j::k]
            worst = max(worst, curve.drain(ctx[j::k], rem[j::k], cap, slack=sl))
        return worst

    def _tier_sim(
        self, tp: int, k: int, ctx: np.ndarray, rem: np.ndarray, cap: int, slack: np.ndarray | None
    ) -> float:
        """Event-driven drain of a tier of `k` instances, bounded by its worst instance."""
        worst = 0.0
        curve = self._curves[tp]
        for j in range(min(k, 2)):
            sl = None if slack is None else slack[j::k]
            worst = max(worst, curve.simulate(ctx[j::k], rem[j::k], cap, sl))
        return worst

    def _best_cap(
        self, tp: int, k: int, ctx: np.ndarray, rem: np.ndarray, slack: np.ndarray | None = None
    ) -> tuple[int, float]:
        """Resident cap for the tier, and the tier's simulated time at that cap.

        The cap is chosen on the unweighted makespan. Weighting by slack would make
        deferring slack-rich requests behind a small cap look free, which starves
        the instance of throughput the next cycle needs.
        """
        per_instance = max(1, int(np.ceil(len(rem) / k)))
        best_cap = min(self._max_concurrency, max(per_instance, 1))
        best = self._tier_sim(tp, k, ctx, rem, best_cap, None)
        for cap in _CAP_GRID:
            if cap >= per_instance or cap >= self._max_concurrency:
                continue
            t = self._tier_sim(tp, k, ctx, rem, cap, None)
            if t < best * 0.99:
                best_cap, best = cap, t
        return best_cap, self._tier_sim(tp, k, ctx, rem, best_cap, slack)

    def _assemble(
        self,
        shape_segments: list[tuple[int, int, int, int]],
        ctx: np.ndarray,
        rem: np.ndarray,
        algo: str,
    ) -> TierPlan:
        """Build a plan from ``(tp, n_instances, start, stop)`` runs over sorted rows."""
        fleet: list[int] = []
        tiers: list[Tier] = []
        spans = [seg for seg in shape_segments if seg[3] > seg[2]]
        idle = [(tp, k) for tp, k, a, b in shape_segments if b <= a]
        makespan = 0.0
        for index, (tp, k, a, b) in enumerate(spans):
            ids = tuple(range(len(fleet), len(fleet) + k))
            fleet.extend([tp] * k)
            hi = float("inf") if index == 0 else 0.5 * (rem[a - 1] + rem[a])
            lo = 0.0 if index == len(spans) - 1 else 0.5 * (rem[b - 1] + rem[b])
            sl = None if self._slack_sorted is None else self._slack_sorted[a:b]
            cap, t = self._best_cap(tp, k, ctx[a:b], rem[a:b], sl)
            makespan = max(makespan, t)
            tiers.append(Tier(tp, ids, lo, hi, cap, b - a, t))
        for tp, k in idle:
            fleet.extend([tp] * k)
        assignment: dict[int, int] = {}
        if self._ids_sorted is not None:
            for index, (tp, k, a, b) in enumerate(spans):
                for rid in self._ids_sorted[a:b]:
                    assignment[int(rid)] = index
        return TierPlan(
            tp_per_instance=tuple(fleet),
            n_gpus=self._n_gpus,
            tiers=tuple(tiers),
            predicted_makespan_s=makespan,
            algorithm=algo,
            assignment=assignment,
        )

    # ------------------------------------------------------------------
    # pp: enumerate shapes, exact ordered contiguous partition by parametric search
    # ------------------------------------------------------------------

    def _pp_for_shape(
        self, shape: tuple[int, ...], ctx: np.ndarray, rem: np.ndarray
    ) -> tuple[float, list[tuple[int, int, int, int]]]:
        """Partition sorted rows over the tiers of `shape` to minimize the makespan."""
        groups: list[tuple[int, int]] = []
        for tp in sorted(set(shape), reverse=True):
            groups.append((tp, shape.count(tp)))
        n = len(rem)
        slk = self._slack_sorted

        def tt(tp: int, k: int, a: int, b: int) -> float:
            return self._tier_time(tp, k, ctx[a:b], rem[a:b], None, None if slk is None else slk[a:b])

        def fill(target: float) -> list[tuple[int, int, int, int]] | None:
            segments = []
            start = 0
            for index, (tp, k) in enumerate(groups):
                if start >= n:
                    segments.append((tp, k, n, n))
                    continue
                if index == len(groups) - 1:
                    stop = n
                    if tt(tp, k, start, stop) > target:
                        return None
                else:
                    lo, hi = start, n
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if tt(tp, k, start, mid) <= target:
                            lo = mid
                        else:
                            hi = mid - 1
                    stop = lo
                segments.append((tp, k, start, stop))
                start = stop
            return segments if start >= n else None

        high = max(tt(tp, k, 0, n) for tp, k in groups)
        low = 0.0
        best = fill(high)
        if best is None:
            best = [(tp, k, 0, n if i == 0 else n) for i, (tp, k) in enumerate(groups)]
            return float("inf"), best
        for _ in range(28):
            mid = 0.5 * (low + high)
            found = fill(mid)
            if found is None:
                low = mid
            else:
                high, best = mid, found
            if high - low < 1e-3 * max(high, 1e-9):
                break
        return high, best

    def _plan_pp(self, ctx: np.ndarray, rem: np.ndarray) -> TierPlan:
        best_t, best_seg = float("inf"), None
        for shape in self._shapes:
            t, seg = self._pp_for_shape(shape, ctx, rem)
            if t < best_t * 0.999 or best_seg is None:
                best_t, best_seg = t, seg
        return self._assemble(best_seg, ctx, rem, "pp")

    # ------------------------------------------------------------------
    # twopool: TailSieve-style isolation, generalized to a TP per pool
    # ------------------------------------------------------------------

    def _plan_twopool(self, ctx: np.ndarray, rem: np.ndarray) -> TierPlan:
        n = len(rem)
        slk = self._slack_sorted

        def tt(tp: int, k: int, a: int, b: int) -> float:
            return self._tier_time(tp, k, ctx[a:b], rem[a:b], None, None if slk is None else slk[a:b])

        best_t, best_seg = float("inf"), None
        candidates: list[tuple[int, int, int, int]] = []
        for shape in self._shapes:
            levels = sorted(set(shape), reverse=True)
            if len(levels) == 1:
                candidates.append((levels[0], shape.count(levels[0]), levels[0], 0))
            elif len(levels) == 2:
                candidates.append((levels[0], shape.count(levels[0]), levels[1], shape.count(levels[1])))
        for tail_tp, m_tail, bulk_tp, m_bulk in candidates:
            if m_bulk == 0:
                t = tt(tail_tp, m_tail, 0, n)
                seg = [(tail_tp, m_tail, 0, n)]
            else:
                lo, hi = 0, n
                while lo < hi:
                    mid = (lo + hi) // 2
                    t_tail = tt(tail_tp, m_tail, 0, mid)
                    t_bulk = tt(bulk_tp, m_bulk, mid, n)
                    if t_tail < t_bulk:
                        lo = mid + 1
                    else:
                        hi = mid
                t, seg = float("inf"), None
                for q in {max(lo - 1, 0), lo, min(lo + 1, n)}:
                    cost = max(tt(tail_tp, m_tail, 0, q), tt(bulk_tp, m_bulk, q, n))
                    if cost < t:
                        t, seg = cost, [(tail_tp, m_tail, 0, q), (bulk_tp, m_bulk, q, n)]
            if t < best_t * 0.999 or best_seg is None:
                best_t, best_seg = t, seg
        return self._assemble(best_seg, ctx, rem, "twopool")

    # ------------------------------------------------------------------
    # dp: Libra-style, over (GPUs used, class prefix), one instance per transition
    # ------------------------------------------------------------------

    def _classes(self, ctx: np.ndarray, rem: np.ndarray) -> list[tuple[int, int]]:
        """Boundaries of contiguous classes, finer in the tail and capped in size.

        A class is priced at its mean length, so a class that mixes a straggler with
        short requests hides the straggler. Cutting wherever the sorted length drops
        sharply keeps every distinct length level in a class of its own.
        """
        n = len(rem)
        k = self._dp_classes
        cuts = {0, n}
        cuts.update(int(x) for x in np.linspace(0, n, k // 2 + 1))
        work = np.cumsum(rem)
        cuts.update(int(x) for x in np.searchsorted(work, np.linspace(0, work[-1], k // 2 + 1)))
        drops = np.nonzero(rem[:-1] > _CLASS_DROP * rem[1:])[0] + 1
        cuts.update(int(x) for x in drops)
        cuts = sorted(c for c in cuts if 0 <= c <= n)
        max_size = max(1, int(np.ceil(n / max(self._n_gpus * 8, 1))))
        refined = [cuts[0]]
        for a, b in zip(cuts, cuts[1:]):
            while b - a > max_size:
                a += max_size
                refined.append(a)
            refined.append(b)
        refined = sorted(set(refined))
        return list(zip(refined[:-1], refined[1:]))

    def _plan_dp(self, ctx: np.ndarray, rem: np.ndarray) -> TierPlan:
        classes = self._classes(ctx, rem)
        m = len(classes)
        cnt = np.array([b - a for a, b in classes], dtype=float)
        rep_rem = np.array([rem[a:b].mean() for a, b in classes])
        rep_ctx = np.array([ctx[a:b].mean() for a, b in classes])
        rep_slack = (
            None if self._slack_sorted is None else np.array([self._slack_sorted[a:b].mean() for a, b in classes])
        )
        cost = {tp: np.full((m + 1, m + 1), np.inf) for tp in self._tps}
        for tp in self._tps:
            curve = self._curves[tp]
            for a in range(m):
                for b in range(a + 1, m + 1):
                    cost[tp][a, b] = curve.drain(
                        rep_ctx[a:b],
                        rep_rem[a:b],
                        self._max_concurrency,
                        weight=cnt[a:b],
                        slack=None if rep_slack is None else rep_slack[a:b],
                    )
        g_max = self._n_gpus
        dp = np.full((g_max + 1, m + 1), np.inf)
        choice: dict[tuple[int, int], tuple[int, int]] = {}
        dp[0, 0] = 0.0
        for g in range(1, g_max + 1):
            for i in range(1, m + 1):
                for tp in self._tps:
                    if tp > g:
                        continue
                    for x in range(1, i + 1):
                        prev = dp[g - tp, i - x]
                        if not np.isfinite(prev):
                            continue
                        cand = max(prev, cost[tp][i - x, i])
                        if cand < dp[g, i] - 1e-12:
                            dp[g, i] = cand
                            choice[(g, i)] = (tp, x)
        g, i = g_max, m
        runs: list[tuple[int, int, int]] = []
        while i > 0:
            if (g, i) not in choice:
                return self._plan_pp(ctx, rem)
            tp, x = choice[(g, i)]
            runs.append((tp, i - x, i))
            g, i = g - tp, i - x
        runs.reverse()
        merged: list[list[int]] = []
        for tp, a, b in runs:
            row_a, row_b = classes[a][0], classes[b - 1][1]
            if merged and merged[-1][0] == tp and merged[-1][3] == row_a:
                merged[-1][1] += 1
                merged[-1][3] = row_b
            else:
                merged.append([tp, 1, row_a, row_b])
        return self._assemble([tuple(r) for r in merged], ctx, rem, "dp")


class PlannerSolver:
    """Adapter that gives `TierPlanner` the solver interface the simulator expects."""

    wants_pending = True
    """Tells the simulator to include groups it is about to admit, not only in-flight work."""

    def __init__(self, planner: TierPlanner, max_staleness: int = 0) -> None:
        self._planner = planner
        self._max_staleness = int(max_staleness)
        self.last_plan: TierPlan | None = None
        self.history: list[TierPlan] = []

    def observe_training_cycle(self, seconds: float) -> None:
        """Accepted for interface compatibility. The planner prices work directly."""

    def solve(self, snapshot, rollout_version: int) -> TierPlan:
        ctx0 = np.array([r.ctx for r in snapshot], dtype=float)
        rem = np.array([r.remaining for r in snapshot], dtype=float)
        ids = np.array([getattr(r, "rid", -1) for r in snapshot], dtype=np.int64)
        if len(ids) == 0 or np.any(ids < 0):
            ids = None
        age = np.array([rollout_version - r.v_traj for r in snapshot], dtype=float)
        slack = np.clip(self._max_staleness - age, 0.0, None) if self._max_staleness else None
        plan = self._planner.plan(ctx0, rem, ids=ids, slack=slack)
        self.last_plan = plan
        self.history.append(plan)
        srl_logger.info(
            f"Fleet plan from {len(snapshot)} requests: {plan.describe()}, "
            f"predicted makespan {plan.predicted_makespan_s:.0f}s ({plan.algorithm})."
        )
        return plan
