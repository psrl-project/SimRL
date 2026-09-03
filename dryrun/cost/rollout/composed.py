"""
Predict a rollout step by summing measured kernel latencies.

Every other model in this package is fitted to end-to-end server measurements, so it must be
refitted for each new model, GPU, and parallelism degree. This one composes the step from a
kernel perf table instead, which is what makes those refits unnecessary.

Two things make that harder than adding the numbers up.

A composed step costs about 48 us to predict -- five GEMM lookups and one attention lookup
against interpolated surfaces -- against a handful of float operations for a fitted formula. A
1000-step decode segment across 100 instances would be 100k predictions, roughly 5 seconds of
pure cost-model time.

Within one decode segment the batch size is fixed, so the five GEMM utilities are exactly
constant across the segment. When the per-request KV depth (ctxsum / n_d) is deep enough the
generation-attention utility is also nearly flat, and step cost reduces to the closed form
``max(F, alpha + beta * k)`` that ``advance.py`` sums analytically in O(1). At shallow KV
depth the utility surface is steep and the closed form is inaccurate, so a
``LatticeIntegrator`` is used instead. The cutoff is ``saturation_step_threshold``.

**That fast path is off by default, because measuring it end to end showed it does not pay.**
On the real DAPO trace (16 instances, 16 versions), against an all-lattice baseline:

    configuration            sim.run()   predict calls   simulated total   speedup   deviation
    all lattice (default)       6.055s           16310         2828.57s         --          --
    hybrid, threshold 1024      5.056s           15644         2698.03s      1.20x       4.62%
    closed form everywhere      5.161s           15732         2702.60s      1.17x       4.45%

The predict count barely moves (16310 to 15644, 4%) because a real workload spends most of its
predictions on prefill, which segment integration never touches. The saving of a few calls per
decode segment is invisible against that total. Meanwhile the deviation reaches 4.6%, which is
the same magnitude as this model's error against a real vLLM server (decode 4.84%) -- so the
approximation stops being a second-order term and starts polluting the result.

Earlier figures of 300x and above came from a single instance running one artificially long
pure-decode segment. That shape does favour the closed form, and it is not what a real trace
looks like. Set ``saturation_step_threshold`` to a finite value only if a workload is genuinely
dominated by long deep-KV decode runs, and re-measure end to end before trusting the speedup.

The threshold is a per-request KV depth (``ctxsum / n_d``), and it is a deliberately crude
proxy. Measured, the >2% single-segment errors do not spread evenly with depth -- they sit
entirely at ``n_d=16`` with depth 512 or 1024, because the generation-attention utility is
non-monotone there (0.3154 at step 512, 0.2970 at 2048, 0.3363 at 4096) and a linear secant
cannot express that dip. Depth alone therefore does not identify the bad region: at ``n_d=1``
the closed form is under 0.04% everywhere, and ``n_d=64`` is under 0.5%.

A sharper criterion exists and was measured: sample the step cost at the segment's start,
midpoint, and end, and compare the midpoint against the linear interpolation of the other two.
That non-linearity correlates with the real error at 0.838, where a depth threshold correlates
at essentially zero. It costs 3 predict calls to make the routing decision. If the fast path is
ever turned back on, that is the criterion to use rather than a depth cutoff.

Known limitation. A mixed step -- prefill and decode tokens in the same batch -- is
approximated as context attention plus generation attention at the step's batch size, because
only the two pure phases were measured. Against the five buckets the six real mixed iterations
form, it deviates by 0.48%. But those are all near-uniform batches, which is exactly the regime
the mean-prompt-length assumption is built for; a chunked-prefill step is not that shape, and
because attention is quadratic in sequence length, spreading one request's 4096-token chunk
across the batch understates its context term severalfold. So this is the model's largest
uncharacterised approximation despite the small figure. Measuring mixed batches directly is the
fix and is out of scope here.

Tensor parallelism above 1 is validated at `tp=2` above batch 32, and the collective term is
real: `ModelOps.comm_queries` adds two all-reduces per layer, priced from a table measured
through `get_tp_group().all_reduce` -- vLLM's own dispatcher, so the kernel timed is the kernel a
server runs. That distinction was measured, not assumed: on H20 at world sizes 2, 4 and 8,
`torch.distributed.all_reduce` is never what the dispatcher selects, because the custom P2P
kernel covers the entire 8-512 KiB range a decode step lives in.

Against a real `tp=2` vLLM baseline on the same grid as the `tp=1` one, decode scores 3.18% over
the 103 buckets at `requests >= 32` -- better than the 4.84% above -- and prefill 6.95%. The
whole-grid decode mean is 16.39%, which is not a model error at face value: the `requests 5-16`
buckets measure half the batch as slower than twice the batch at twice the KV depth, a
self-contradiction produced by four servers sharing one machine. The `requests 1-4` buckets carry
the same small-batch over-prediction documented at `tp=1`, worsened from 1.1312 to 1.3009.

Two approximations remain. The all-reduce is timed in isolation, so overlap with compute is not
represented -- though it is not the dominant term here, since every phase under-predicts
(median ratio 0.9648-0.9766) where an unmodelled overlap saving worth the collective's 13.1%
share would have shown a ratio near 1.13. And every world size is measured intra-node over
NVLink, so a cross-node `tp` would be priced with the wrong interconnect. `tp=4` and `tp=8` have
collective tables but no end-to-end baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import advance
from .base import RolloutCostModel, RolloutWorkload
from .segment import DEFAULT_STRIDE, LatticeIntegrator

# Both names appear only in annotations, never at runtime, and `from __future__ import
# annotations` keeps those lazy. Importing either eagerly would pull the parquet stack into
# `dryrun.cost.rollout`, which every simulator run imports for its model registry, so a run
# using a fitted model would pay a 1.9 s import and a pyarrow load it never uses.
if TYPE_CHECKING:
    from ..kernel.database import KernelDatabase
    from ..kernel.model_ops import ModelOps

# Disables the closed-form fast path: no per-request KV depth can reach it, so every decode
# segment integrates on the lattice. Measured end to end, the fast path bought 1.2x while
# costing 4.6% deviation -- see the module docstring. Set a finite threshold to opt back in.
CLOSED_FORM_DISABLED = 1 << 62

# Hard ceiling on decode-segment length. No decode segment can reach this under any configured
# output cap, and the lattice is lazy, so a generous ceiling costs nothing.
_SEGMENT_CAP = 1 << 20

# Ceiling on the predict memo, ~11 MB at roughly 172 bytes per entry. Only a stride far below
# the default can reach it, and there the memo degrades to a partial cache rather than failing.
_MAX_CACHE_ENTRIES = 65536


class _SegmentHandle:
    """Common interface returned by ``_segment_route``."""

    def total(self, k: int) -> float:
        """Return the seconds of the first ``k`` steps."""
        raise NotImplementedError

    def steps_within(self, budget: float) -> int:
        """Return the largest step count whose total fits inside ``budget``."""
        raise NotImplementedError


class _ClosedFormHandle(_SegmentHandle):
    """Delegate total/steps_within to ``advance.py``'s closed-form functions."""

    def __init__(self, F: float, alpha: float, beta: float) -> None:
        self._F = F
        self._alpha = alpha
        self._beta = beta

    def total(self, k: int) -> float:
        """Return the seconds of the first ``k`` steps."""
        return advance.elapsed(self._F, self._alpha, self._beta, k)

    def steps_within(self, budget: float) -> int:
        """Return the largest step count whose total fits inside ``budget``."""
        return advance.steps_within(self._F, self._alpha, self._beta, budget)


class ComposedRollout(RolloutCostModel):
    """Sum measured kernel latencies to price a rollout step."""

    model_type = "composed"

    def __init__(
        self,
        database: KernelDatabase,
        ops: ModelOps,
        *,
        tensor_parallel: int = 1,
        stride: int = DEFAULT_STRIDE,
        attention_backend: str | None = None,
        saturation_step_threshold: int = CLOSED_FORM_DISABLED,
    ) -> None:
        """
        Bind a perf table and an op list into a cost model.

        Args:
            database (KernelDatabase): Answers shape queries in seconds.
            ops (ModelOps): The per-step op list, already sharded.
            tensor_parallel (int): Ranks the model is sharded across. Must match the degree
                `ops` was built for, since `ops` is what decides whether a comm term is
                emitted.
            stride (int): Steps between decode-segment lattice nodes.
            attention_backend (str | None): Backend override; `None` uses the backend vLLM's
                own selector picks, which is what a prediction should reflect.
            saturation_step_threshold (int): Minimum per-request KV depth (ctxsum / n_d) above
                which the segment uses the closed-form advance instead of the lattice integrator.
                At this depth utilities are flat enough that step cost is well-approximated by
                ``max(F, alpha + beta * k)``. Below it the lattice is used unchanged.

        Raises:
            ValueError: If `tensor_parallel` disagrees with the degree `ops` was sharded for,
                or `stride` is not positive.
        """
        if tensor_parallel != ops.tensor_parallel:
            # `ops` is the single source of truth: it decides both the sharded GEMM shapes and
            # whether `comm_queries` emits a term. A disagreeing argument would otherwise be
            # silently ignored, and a caller passing 2 against tp=1 ops would get an unsharded
            # prediction with every all-reduce missing and nothing saying so.
            raise ValueError(
                f"ComposedRollout got tensor_parallel={tensor_parallel!r} but its ops were "
                f"sharded for {ops.tensor_parallel!r}. Build ModelOps at the same degree."
            )
        if stride <= 0:
            # LatticeIntegrator would also reject this, but only on the first decode segment.
            # `stride` comes from config (`decode_integration_stride`), so a typo should fail at
            # construction rather than partway into a simulation.
            raise ValueError(f"ComposedRollout stride must be positive, got {stride!r}.")
        self._database = database
        self._ops = ops
        self._stride = stride
        self._saturation_step_threshold = saturation_step_threshold
        self._backend = attention_backend or None
        # Bounded memo on `predict`. Two sources of duplication arise in `advance_to`:
        # (1) `decode_steps_within` and `decode_segment_time` each build a fresh integrator
        #     for the same `(n_d, ctxsum)`, so identical `(n_d, ctxsum, offset)` triples
        #     are evaluated twice -- a 1.75x measured redundancy on a real run_until.
        #     Not memoizing at the integrator level is intentional: each event sees a fresh
        #     ctxsum, so an integrator cache would grow unbounded with near-zero hit rate.
        # (2) `_step_time_estimate` and `_prefill_budget_step` both call `step_time` on
        #     the same batch state, doubling prefill predict calls.
        # Keying on the full frozen `RolloutWorkload` (a hashable dataclass) covers both
        # sites with one mechanism. The cap keeps memory O(1) over an unbounded simulation.
        self._predict_cache: dict[RolloutWorkload, float] = {}
        # Sized so one segment's lattice nodes fit, or arbitrary eviction would thrash and lose
        # the deduplication this exists for. A segment spans at most `_SEGMENT_CAP // stride + 2`
        # nodes: 1026 at the default stride, but 2^20 at `stride=1`, which the docs offer as the
        # exact-sum option. At ~172 bytes per entry an unbounded sizing would reach 180 MB, so
        # it is clamped -- a stride that fine is inherently expensive, and paying it in repeated
        # table lookups is preferable to paying it in resident memory.
        self._predict_cache_cap = min(max(4096, _SEGMENT_CAP // stride + 2), _MAX_CACHE_ENTRIES)

    def step_time(self, n_tokens, t2, ctxsum, n_reqs: int = 0) -> float:
        """
        Adapt the simulator's compact step arguments to `RolloutWorkload`, with caching.

        Overrides the base class to add the bounded memo. `advance_to` calls
        `_step_time_estimate` and `_prefill_budget_step` in sequence on the same batch
        state, so both resolve to an identical `RolloutWorkload`; the second call is free.
        """
        workload = RolloutWorkload.from_step(n_tokens, t2, ctxsum, n_reqs)
        return self._cached_predict(workload)

    def _cached_predict(self, workload: RolloutWorkload) -> float:
        """Check the bounded memo and fall through to `predict` on a miss."""
        cached = self._predict_cache.get(workload)
        if cached is not None:
            return cached
        value = self.predict(workload)
        if len(self._predict_cache) >= self._predict_cache_cap:
            # Evict an arbitrary entry. Intra-event deduplication is the goal, not long-term
            # locality, so LRU would add complexity with no benefit. O(1) in CPython.
            self._predict_cache.pop(next(iter(self._predict_cache)))
        self._predict_cache[workload] = value
        return value

    def predict(self, workload: RolloutWorkload) -> float:
        """
        Predict one engine iteration by summing its kernels.

        Args:
            workload (RolloutWorkload): One iteration's token and context counts.

        Returns:
            float: Strictly positive wall-clock seconds.
        """
        request_count = max(int(workload.request_count), 1)
        token_count = max(int(round(workload.n_tokens)), 1)
        total = 0.0
        for query, multiplier in self._ops.gemm_queries(token_count, request_count):
            total += multiplier * self._database.query_gemm(query)
        for query, multiplier in self._ops.comm_queries(token_count):
            total += multiplier * self._database.query_comm(query)
        total += self._ops.num_layers * self._attention_seconds(workload, request_count)
        return total

    def _attention_seconds(self, workload: RolloutWorkload, request_count: int) -> float:
        """Return one layer's attention cost, adding both phases for a mixed step."""
        seconds = 0.0
        if workload.sum_l2 > 0:
            # `sum_l2` is used only as a boolean gate to detect a context phase; it is not used
            # arithmetically. The mean prompt length `n_tokens / request_count` assumes a
            # near-uniform batch. See the module docstring's known-limitation paragraph for the
            # reasoning and the measured 0.7% deviation on the six available mixed iterations.
            seq_len = max(int(workload.n_tokens // request_count), 1)
            seconds += self._database.query_attention(
                self._ops.attention_query(
                    "context", batch_size=request_count, seq_len=seq_len, backend=self._backend
                )
            )
        if workload.ctxsum > 0:
            step = max(int(workload.ctxsum // request_count), 1)
            seconds += self._database.query_attention(
                self._ops.attention_query(
                    "generation", batch_size=request_count, step=step, backend=self._backend
                )
            )
        if seconds == 0.0:
            # A step with neither prompt nor context tokens still runs one attention call.
            seconds = self._database.query_attention(
                self._ops.attention_query(
                    "generation", batch_size=request_count, step=1, backend=self._backend
                )
            )
        return seconds

    def decode_coeffs(self, n_d: int, ctxsum) -> tuple:
        """
        Return floor, intercept, and slope for the closed-form advance of a decode segment.

        Samples the step cost at offsets 0 and ``stride`` and returns a secant slope. The
        floor is zero because a composed model has no fitted low-load floor.

        Args:
            n_d (int): Requests decoding.
            ctxsum: Total context tokens across the batch.

        Returns:
            tuple: ``(floor, intercept, slope)`` in seconds.
        """
        batch = max(int(n_d), 1)
        first = self._step_seconds(batch, float(ctxsum), 0)
        later = self._step_seconds(batch, float(ctxsum), self._stride)
        slope = max((later - first) / self._stride, 0.0)
        return 0.0, first, slope

    def decode_segment_time(self, n_d: int, ctxsum: float, k: int) -> float:
        """Return the wall-clock seconds of k decode steps over a fixed batch."""
        if k <= 0:
            return 0.0
        return self._segment_route(n_d, ctxsum).total(k)

    def decode_steps_within(self, n_d: int, ctxsum: float, budget: float) -> int:
        """Return the largest step count whose segment time fits inside ``budget``."""
        if budget <= 0:
            return 0
        return self._segment_route(n_d, ctxsum).steps_within(budget)

    def decode_saturated_steps(self, n_d: int, ctxsum: float, k: int) -> int:
        """Return how many of k steps run above the low-load floor."""
        # A composed model has no fitted floor: every step's cost is measured work, so every
        # step is above it by definition.
        return max(int(k), 0)

    def saturated(self, n_tokens, t2, ctxsum, n_reqs=0) -> bool:
        """Return whether the workload is above a fitted low-load floor."""
        # Same reasoning as `decode_saturated_steps`. The base class returns False, which would
        # have this model report a step as unsaturated while simultaneously reporting all of its
        # steps saturated -- telemetry reads both fields, so they must agree.
        return True

    def _use_closed_form(self, n_d: int, ctxsum: float) -> bool:
        """Return True if the KV depth is deep enough for the closed-form path."""
        if n_d <= 0:
            return False
        return (ctxsum / n_d) >= self._saturation_step_threshold

    def _segment_route(self, n_d: int, ctxsum: float) -> _SegmentHandle:
        """Return a handle that delegates total/steps_within to the right integrator."""
        if self._use_closed_form(n_d, ctxsum):
            F, alpha, beta = self.decode_coeffs(n_d, ctxsum)
            return _ClosedFormHandle(F, alpha, beta)
        return self._integrator(n_d, ctxsum)

    def _integrator(self, n_d: int, ctxsum: float) -> LatticeIntegrator:
        """Build a lattice integrator for one decode segment."""
        batch = max(int(n_d), 1)
        base = float(ctxsum)
        return LatticeIntegrator(
            lambda offset: self._step_seconds(batch, base, offset),
            k_max=_SEGMENT_CAP,
            stride=self._stride,
        )

    def _step_seconds(self, n_d: int, ctxsum: float, offset: int) -> float:
        """Return the cost of the step `offset` steps into a decode segment."""
        # Every request gains one token per step, so the batch's context grows by n_d.
        # `_cached_predict` guards the call, so repeated calls from two integrators
        # with the same (n_d, ctxsum, offset) are free after the first evaluation.
        return self._cached_predict(
            RolloutWorkload(
                phase="decode",
                n_tokens=n_d,
                sum_l2=0.0,
                ctxsum=ctxsum + n_d * offset,
                request_count=n_d,
            )
        )
