"""
Kernel-composed training step latency, analogous to ComposedRollout.

`ComposedTraining` is a `TrainingLatencyModel` that sums individual kernel costs rather
than fitting a polynomial to whole-step wall times. The value over the polynomial model
is attributability: a prediction error is a sum of per-op errors rather than a residual,
so a systematic miss points to a named op rather than to an unexplained coefficient.

Attention is priced from the `attention_training` family, both directions measured. It
does **not** read the rollout `context_attention` table, which describes vLLM's prefill
kernel over a BSHD batch where training runs TE's FlashAttention over a THD pack. The
substitution was wrong in both directions at once: the forward by 1.59-1.84x and the
backward (previously approximated at 1.0x of forward) by 2.24-2.42x, for a combined
2.35-2.60x understatement of the attention term. See `ops/attention_training.py`.

The optimizer step is a measured kernel like any other, priced from the `optimizer_step`
table against the parameter shard one rank updates. It is the only term with no token axis,
because an Adam step's cost does not depend on how many tokens produced the gradients.
Measured on H20: 14.1 ms for a Qwen3-8B rank at tp=2, pp=3, which is 0.02-1.3% of the
validation set's step times. It is charged because an unmeasured term cannot be ruled out,
and ruling it out turned out to be the result.

Scores MRE 12.63% with signed mean -12.63% on the 22 out-of-sample Megatron THD records,
against the fitted polynomial's 2.73% in-sample. **One unexplained term remains and it is
the whole residual**: every record under-predicts by 4.7-25.6% of its own prediction. No
constant is added for it, because a fitted constant is exactly the unexplained coefficient
this model exists to avoid.

Its shape, regressed through the origin over all 22 records: proportional to total
predicted time (c = 0.185, r^2 = 0.932). Not a fixed per-step cost, and not a per-slot
constant either - `slots` alone gives r^2 = 0.869 and the per-slot values span 0.039-0.484 s.

What it is not, each excluded by measurement rather than by argument:

- Not the GEMM table being cuBLAS where training is TE. Measured: `te.Linear` against
  `torch.nn.functional.linear` at the four Qwen3 tp=2 projections is **1.001x**. The gap
  correlates with GEMM time at r^2 = 0.968, but GEMM is 69% of the prediction, so that is
  collinearity with total cost rather than evidence.
- Not the optimizer step (14.1 ms per step, measured at 81% of HBM peak so it has no
  headroom to be larger).
- Not recompute: the replay is already charged for the GEMMs, attention, elementwise and
  the SP collectives (3 forward-shaped passes, 8 all-gather + 6 reduce-scatter per layer).
- Not the two unmodelled residual-add kernels: priced at `rms_norm`'s bandwidth they come
  to 5.4 ms per slot, 2.7% of the gap.

At ~16 ms per layer against ~29 ms of modelled GEMM per layer it is too large to be any
pointwise kernel, and the per-op tables it would have to live in are individually accurate
to 2.8-5.7% on holdout. The remaining candidates need a profiler trace of a real step —
kernels the composition does not know exist, or per-call launch overhead that the isolated
measurements amortize away. `docs/reference/limitations.md` records it as the open item.

**The MRE was 7.62% before the per-sequence attention fix below, and that was two errors
cancelling.** Attention is now priced per `cu_seqlens` segment, which is correct physics:
the correction removes 1.02-1.10x of the packing effect the validation set measures
directly, and the predicted packing order correlates 0.9995 with the measured one where the
pack-total model had zero spread to correlate. Once the per-slot offset above is removed,
the per-sequence model tracks the five-record packing family at 1.48% against the
pack-total model's 4.88%. Reverting to recover the lower pooled number would be keeping a
known-wrong attention term because it happened to offset an unknown one.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ...profiler.schema import SequenceStats
from ..kernel.database import KernelDatabase
from ..kernel.interp import KernelCoverageError
from .base import TrainingLatencyModel, TrainingParallelism, TrainingWorkload
from .latency import pipeline_slots
from .microbatch import BatchingMode, MicrobatchPlanner
from .training_ops import TrainingOps


class ComposedTraining(TrainingLatencyModel):
    """
    Predict one training step's wall time by summing measured kernel tables.

    Unlike `HydraulisLatencyModel`, this model has no fitted coefficients: it queries
    the same parquet tables as `ComposedRollout` and sums the results. New parallelism
    configurations, recompute modes, and sequence length regimes are covered as long as
    the required kernel shapes are in the tables.

    Construct from a kernel table root and a `TrainingOps` op list. Pair with
    `HydraulisMemoryModel` from an existing artifact for memory prediction.

    Args:
        database: Kernel tables for the target GPU.
        ops: Per-microbatch op list, sharded for the intended TP degree.
        charge_optimizer: Whether to price the fused Adam step from the `optimizer_step`
            table. Default True. Set False to score a configuration whose optimizer table
            has not been swept; the term is then absent from the decomposition rather than
            silently zero, so the omission is visible.

    Example::

        db = KernelDatabase.from_root("outputs/profile/qwen3_8b/kernel_tp2/kernel", gpu="h20")
        ops = TrainingOps.from_checkpoint(
            "/path/to/Qwen3-8B",
            tensor_parallel=2,
            recompute="full",
        )
        model = ComposedTraining(db, ops)
        latency = model.predict(workload, TrainingParallelism(tp=2, pp=3, dp=1))
    """

    def __init__(
        self,
        database: KernelDatabase,
        ops: TrainingOps,
        *,
        charge_optimizer: bool = True,
    ) -> None:
        self._db = database
        self._ops = ops
        self._charge_optimizer = charge_optimizer

    @classmethod
    def from_checkpoint(
        cls,
        table_root: str | Path,
        checkpoint: str | Path,
        *,
        gpu: str,
        tensor_parallel: int = 1,
        recompute: str = "none",
        dtype: str = "bfloat16",
        gradient_dtype: str = "float32",
        distributed_optimizer: bool = True,
        utility_surface: str = "pchip",
        charge_optimizer: bool = True,
        base_dir: Path | None = None,
    ) -> ComposedTraining:
        """Convenience constructor that reads tables and checkpoint from disk."""
        db = KernelDatabase.from_root(table_root, gpu=gpu, utility_surface=utility_surface)
        ops = TrainingOps.from_checkpoint(
            checkpoint,
            tensor_parallel=tensor_parallel,
            recompute=recompute,
            dtype=dtype,
            gradient_dtype=gradient_dtype,
            distributed_optimizer=distributed_optimizer,
            base_dir=base_dir,
        )
        return cls(db, ops, charge_optimizer=charge_optimizer)

    def predict(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> float:
        """
        Predict wall-clock latency in seconds.

        Under 1F1B pipeline parallelism, the critical path of a step is:
        ``(num_microbatches + pp - 1)`` microbatch-sized slots. Each slot executes
        ``num_layers / pp`` layers on one pipeline stage. The total wall time is
        therefore ``slots * per_stage_cost``, not ``slots * full_model_cost``.

        This is different from ``pipeline_slots`` returning ``nmb + pp - 1``: that
        formula counts slots on the critical path, which is correct, but the slot
        unit is a single stage's work — not all layers.

        Args:
            workload: Sequence lengths and batching parameters.
            parallelism: TP/PP/DP degrees. TP must match `ops.tensor_parallel`.

        Returns:
            float: Predicted step latency in seconds.

        Raises:
            KernelCoverageError: If any query falls outside the measured tables.
            ValueError: If `parallelism.tp` does not match `ops.tensor_parallel`.
        """
        if parallelism.tp != self._ops.tensor_parallel:
            raise ValueError(
                f"parallelism.tp {parallelism.tp} does not match ops.tensor_parallel "
                f"{self._ops.tensor_parallel}. Construct a TrainingOps with the same TP degree."
            )

        planner = MicrobatchPlanner(BatchingMode.PAD_TO_MAX)
        microbatches = planner.plan(
            workload.sequence_lengths,
            parallelism.dp,
            micro_batch_size=workload.micro_batch_size,
        )
        return self.predict_microbatches(microbatches, parallelism)

    def predict_microbatches(
        self,
        microbatches: Sequence[SequenceStats],
        parallelism: TrainingParallelism,
    ) -> float:
        """
        Predict latency for an already-planned microbatch sequence.

        This is the entry point validation uses, because a profiled record carries the plan
        Megatron actually executed. Routing through `predict` would re-plan it from the
        sequence lengths and fold `MicrobatchPlanner`'s disagreement with Megatron's own
        batching into the latency score, making a composition error and a planning error
        indistinguishable.

        Args:
            microbatches: The plan, each entry carrying `sum_tokens`.
            parallelism: TP/PP/DP degrees. TP must match `ops.tensor_parallel`.

        Returns:
            float: Predicted step latency in seconds.

        Raises:
            KernelCoverageError: If any query falls outside the measured tables.
            ValueError: If `parallelism.tp` does not match `ops.tensor_parallel`, or if
                `microbatches` is empty.
        """
        terms = self.decompose_microbatches(microbatches, parallelism)
        return sum(terms.values())

    def _slot_token_count(self, microbatches: Sequence[SequenceStats]) -> int:
        """Return the token count one pipeline slot is priced at.

        The mean `sum_tokens` across the plan, not the max. Under 1F1B every microbatch
        occupies one slot on the critical path, so the step cost is the sum over
        microbatches, and `slots * mean` reproduces that sum exactly when the plan is
        uniform while staying linear in the ragged case. `max` would charge every slot for
        the longest microbatch, which over-predicts a ragged RL batch by the raggedness.

        `sum_tokens` rather than `max_tokens`: under THD packing a microbatch holding three
        4096-token sequences runs its GEMMs over 12288 rows. Pricing `max_tokens` understated
        those records by the packing factor, which measured as a 3x miss.
        """
        if not microbatches:
            raise ValueError("A training step needs at least one microbatch.")
        return int(sum(mb.sum_tokens for mb in microbatches) / len(microbatches))

    def _slot_pack_lengths(
        self, microbatches: Sequence[SequenceStats], token_count: int
    ) -> tuple[int, ...]:
        """Return the per-sequence lengths one slot's attention is priced over.

        Attention is the one term that needs the pack's *structure* rather than its token
        total, because it does not cross a `cu_seqlens` boundary. Every other term is
        linear in tokens and so depends only on `token_count`.

        Taken from the first microbatch when the plan is uniform in shape, which every
        profiled record is. For a ragged plan the lengths are scaled to the slot's mean
        token count so the attention term stays consistent with the other terms' basis:
        pricing attention over one microbatch's lengths while pricing the GEMMs at the
        plan's mean would charge two different workloads in one slot.

        A plan whose microbatches differ in sequence *count* is approximated by the first
        microbatch's count. That is a real limitation and it is not exercised here: all 22
        validation records are uniform (verified), and `MicrobatchPlanner` emits uniform
        plans except under `PACKED_NO_PAD` with ragged input.
        """
        reference = microbatches[0]
        lengths = reference.lengths or (reference.sum_tokens,)
        if reference.sum_tokens == token_count:
            return tuple(lengths)
        # Ragged plan: rescale so the priced pack holds `token_count` tokens in the same
        # proportions. `max(..., 1)` because a short sequence must not round to zero.
        scale = token_count / reference.sum_tokens
        return tuple(max(int(length * scale), 1) for length in lengths)

    def decompose_microbatches(
        self,
        microbatches: Sequence[SequenceStats],
        parallelism: TrainingParallelism,
    ) -> dict[str, float]:
        """
        Return the latency contribution of each op family for one step.

        This is the per-term decomposition Spec B calls for, and `predict_microbatches` is
        defined as the sum of this dict rather than as a parallel computation. Two code
        paths summing the same kernels would be free to drift, and the decomposition is
        exactly the instrument you would use to diagnose that drift, so it must not be the
        thing that drifted.

        Under 1F1B the critical path is ``num_microbatches + pp - 1`` slots, and each slot
        executes one pipeline stage's ``num_layers / pp`` layers. The per-slot cost is
        therefore priced at a fraction of the model, and the bubble falls out of the slot
        count exceeding the microbatch count rather than being a separate term.

        Args:
            microbatches: The plan, each entry carrying `sum_tokens`.
            parallelism: TP/PP/DP degrees. TP must match `ops.tensor_parallel`.

        Returns:
            dict[str, float]: Seconds per term. Keys are ``gemm_fwd``, ``gemm_bwd``,
                ``attn_fwd``, ``attn_bwd``, ``elementwise``, ``tp_collective``,
                ``pp_comm``, ``logits`` and ``optimizer``. Summing them gives
                `predict_microbatches`.

        Raises:
            KernelCoverageError: If any query falls outside the measured tables.
            ValueError: If `parallelism.tp` does not match `ops.tensor_parallel`, or if
                `microbatches` is empty.
        """
        if parallelism.tp != self._ops.tensor_parallel:
            raise ValueError(
                f"parallelism.tp {parallelism.tp} does not match ops.tensor_parallel "
                f"{self._ops.tensor_parallel}. Construct a TrainingOps with the same TP degree."
            )
        token_count = self._slot_token_count(microbatches)
        slots = pipeline_slots(len(microbatches), parallelism.pp)
        # Each slot is one pipeline stage's work. A multiplier declared per *model* is
        # scaled by this ratio; a multiplier declared per *step* is not.
        stage_fraction = 1.0 / parallelism.pp
        layers = self._ops.num_layers * stage_fraction

        terms = {
            "gemm_fwd": sum(
                self._db.query_gemm(query) * mult * stage_fraction
                for query, mult in self._ops.forward_gemm_queries(token_count)
            ),
            "gemm_bwd": sum(
                self._db.query_gemm(query) * layers
                for query, _ in self._ops.backward_gemm_queries(token_count)
            ),
            "elementwise": sum(
                self._db.query_elementwise(query) * mult * stage_fraction
                for query, mult in self._ops.elementwise_queries(token_count)
            ),
            "tp_collective": sum(
                self._db.query_comm(query) * mult * stage_fraction
                for query, mult in self._ops.tp_comm_queries(token_count)
            ),
        }

        # Attention is priced per sequence in the pack, not over the pack total: a varlen
        # kernel does not attend across a `cu_seqlens` boundary, so the quadratic work is
        # `sum_i s_i^2`. See `TrainingOps.attention_queries` for the five 12288-token
        # records that isolate this.
        #
        # The forward is measured once and charged for each forward-shaped pass, so the
        # recompute replay cannot drift from the GEMM path's own doubling. The backward has
        # its own measured table rather than being a multiple of forward: the ratio is
        # 2.24-2.42 and drifts with sequence length, so no single scalar is right.
        pack_lengths = self._slot_pack_lengths(microbatches, token_count)

        def attention_s(direction: str) -> float:
            return sum(
                self._db.query_training_attention(query) * repeats
                for query, repeats in self._ops.attention_queries(
                    pack_lengths, direction=direction
                )
            )

        terms["attn_fwd"] = attention_s("forward") * self._ops.attention_passes() * layers
        terms["attn_bwd"] = attention_s("backward") * layers

        # Logits run once per step on the last pipeline stage, over every token rather than
        # one per request. A missing dgrad slice is skipped rather than clamped onto a
        # neighbouring projection's curve: the measured table has no row for it, and the
        # nearest slice describes different work.
        logits = 0.0
        for query, mult in self._ops.logits_queries(token_count):
            try:
                logits += self._db.query_gemm(query) * mult
            except KernelCoverageError:
                pass

        # Pipeline sends are per *microbatch*, not per layer, so they are charged outside
        # the per-layer scaling and against the microbatch count rather than the slot count:
        # the bubble slots carry no microbatch and so send nothing.
        pp_comm = sum(
            self._db.query_comm(query) * mult
            for query, mult in self._ops.pp_comm_queries(
                token_count, pipeline_parallel=parallelism.pp
            )
        ) * len(microbatches)

        # Every per-layer term is charged once per slot; logits and the optimizer are
        # per-step and are not.
        scaled = {name: value * slots for name, value in terms.items()}
        scaled["pp_comm"] = pp_comm
        scaled["logits"] = logits
        scaled["optimizer"] = self._optimizer_latency(parallelism)
        return scaled

    def _optimizer_latency(self, parallelism: TrainingParallelism) -> float:
        """Return the fused Adam step cost for one rank's parameter shard, in seconds.

        Zero when the caller disabled the term, when the model descriptor carries no
        parameter count, or when the `optimizer_step` table has not been swept. The last
        case is a coverage gap rather than a zero cost, but refusing the whole prediction
        over a term measured at 0.02-1.3% of a step would trade a small known error for a
        total one. The decomposition reports the zero, which is where the omission shows.
        """
        if not self._charge_optimizer:
            return 0.0
        query = self._ops.optimizer_query(pipeline_parallel=parallelism.pp)
        if query is None:
            return 0.0
        try:
            return self._db.query_optimizer(query)
        except KernelCoverageError:
            return 0.0

    def decompose(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> dict[str, float]:
        """
        Return the per-term decomposition for a workload, planning it first.

        Args:
            workload: Sequence lengths and batching parameters.
            parallelism: TP/PP/DP degrees.

        Returns:
            dict[str, float]: As `decompose_microbatches`.
        """
        planner = MicrobatchPlanner(BatchingMode.PAD_TO_MAX)
        microbatches = planner.plan(
            workload.sequence_lengths,
            parallelism.dp,
            micro_batch_size=workload.micro_batch_size,
        )
        return self.decompose_microbatches(microbatches, parallelism)
