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

Scores MRE 8.13% with signed mean -8.13% on the 22 out-of-sample Megatron THD records,
against the fitted polynomial's 2.73% in-sample. The record whose kernels were actually
traced (12288 tokens, nmb=1) predicts to **-0.12%**.

**A kernel trace of a real step validated most of the model and found what was missing.**
Tracing rank 0 at 12288 tokens for nmb 1 and nmb 4 (`DRYRUN_TRACE_STEP=4`, see
`adapters/megatron/measure.py`) gives per-stage costs to compare term by term:

| term | composed | traced | ratio |
|---|---|---|---|
| gemm | 0.8162 s | 0.8299 s | 0.98 |
| attention | 0.3927 s | 0.3936 s | 1.00 |
| tp_collective | 0.0912 s | 0.0824 s | 1.11 |
| elementwise (before) | 0.0190 s | 0.0896 s | 0.21 |

So the GEMM and attention tables are right against real kernels rather than merely
self-consistent, and the SP collective counts derived from TE source are exact: the trace
shows 97 all-gathers and 73 reduce-scatters against the derived 96 and 72, the extra one each
being process-group setup.

**Elementwise was low by 4.6x, and the cause was kernel count, not kernel size.** The trace
issues 158.7 tensor-staging launches per layer per microbatch where the inventory named 7
model kernels, and the population is `aten::copy_`, `aten::cat` (the QKV split), `aten::to`,
`aten::mul` and `aten::add_` — buffer plumbing around TE's fused modules, not norms or
activations. At 4096 tokens **85% of its cost is launch floor** (8.74 us per launch against
2.5 ns per token), which is why scaling the existing seven kernels by 4.6 would have been
wrong: that scales with tokens where most of the real cost does not. Added as a
`tensor_plumbing` elementwise kind with a traced count; closed 4.50pp.

**What remains is per-token work the trace has not named.** The gap goes as total tokens at
r^2 = 0.936, better than the microbatch count (0.860) or total predicted time (0.893), and
is 0.1-18.4% of prediction with every record the same sign. Not fitted.

**The bubble is real and is a named term at 36.1% of the prediction.** It appears on the GPU
as `ncclDevKernel_SendRecv` blocking at the stage boundary: 2.859 s across 2 calls for a
48 MiB handoff, which is 35 MB/s on a 450 GB/s link, so it is a stage waiting rather than
bytes moving. `pp_comm` is the transfer; `bubble` is the wait.

**Per-layer work is charged per microbatch, not per slot**, which the trace settles: every
per-layer kernel count scales exactly 4.00x from nmb=1 to nmb=4 (flash forward 24 -> 96,
all-gather 97 -> 388, reduce-scatter 73 -> 292).

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

        **Per-layer work is charged per microbatch; only the bubble is charged per slot.**
        Verified by tracing rank 0 of a real step at 12288 tokens, nmb 1 against nmb 4:
        every per-layer kernel count scales exactly 4.00x (flash forward 24 -> 96, flash
        backward 12 -> 48, all-gather 97 -> 388, reduce-scatter 73 -> 292), and the device
        time with it (gemm 3.92x, attention 3.98x, elementwise 4.00x).

        Charging per *slot* was a measured over-count: at nmb=1, pp=3 the step has 3 slots
        but one microbatch, so the SP collectives were priced at 288 all-gathers where the
        trace shows 96. The error is `slots / num_microbatches`, which is 3x at nmb=1, 1.5x
        at nmb=4 and 1.03x at nmb=64 — invisible in a pooled MRE fitted over mixed
        microbatch counts, and the reason the residual looked proportional to total cost.

        The bubble is the `pp - 1` slots in which a stage has no microbatch to process. It
        appears on the GPU as `ncclDevKernel_SendRecv` blocking at the stage boundary: the
        trace shows 2.859 s across 2 calls for a 48 MiB handoff, which is 35 MB/s on a
        450 GB/s link, so it is a wait rather than a transfer. Priced as
        `(slots - num_microbatches)` idle slots of one stage's work.

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
        num_microbatches = len(microbatches)
        slots = pipeline_slots(num_microbatches, parallelism.pp)
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

        # The transfer itself, which is per microbatch per boundary crossing. Distinct from
        # `bubble` below: this is the bytes moving, that is the stage waiting.
        pp_comm = sum(
            self._db.query_comm(query) * mult
            for query, mult in self._ops.pp_comm_queries(
                token_count, pipeline_parallel=parallelism.pp
            )
        ) * num_microbatches

        # Per-layer terms are charged once per *microbatch* (traced: every per-layer kernel
        # count scales 4.00x from nmb=1 to nmb=4). The idle slots are charged separately so
        # the two cannot be confused again.
        scaled = {name: value * num_microbatches for name, value in terms.items()}
        scaled["pp_comm"] = pp_comm
        # The 1F1B bubble: `pp - 1` slots in which this stage has no microbatch. Each costs
        # one stage-pass of wall time, which is the per-microbatch cost already computed.
        idle_slots = slots - num_microbatches
        scaled["bubble"] = idle_slots * sum(terms.values())
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
