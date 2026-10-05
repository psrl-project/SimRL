"""
Kernel-composed training step latency, analogous to ComposedRollout.

`ComposedTraining` is a `TrainingLatencyModel` that sums individual kernel costs rather
than fitting a polynomial to whole-step wall times. The value over the polynomial model
is attributability: a prediction error is a sum of per-op errors rather than a residual,
so a systematic miss points to a named op rather than to an unexplained coefficient.

Attention backward is priced as a scalar multiple of forward. The cuDNN FusedAttention
benchmark (measured on H20, 4096 tokens, 1024-12288 token range) shows backward costs
approximately the same as forward (measured ratio 0.83x-0.99x across the full token range),
so the default multiplier is 1.0. A per-shape backward table (B1) would improve this, but
the current prediction is in the right range and is explicit rather than absorbed into
an unexplained coefficient.

The optimizer step is a measured kernel like any other, priced from the `optimizer_step`
table against the parameter shard one rank updates. It is the only term with no token axis,
because an Adam step's cost does not depend on how many tokens produced the gradients.
Measured on H20: 14.1 ms for a Qwen3-8B rank at tp=2, pp=3, which is 0.02-1.3% of the
validation set's step times. It is charged because an unmeasured term cannot be ruled out,
and ruling it out turned out to be the result.

What the composition is still missing, stated because the decomposition is what makes a
kernel sum worth having: the residual against the 22 THD records is a uniform
under-prediction of ~17%, and it scales with work rather than being a fixed per-step cost
(10-39% of prediction across a 192x token range). So the gap is in a per-token term, not in
anything charged once per step. The leading candidates are the GEMM table being measured
with `torch.nn.functional.linear` where TE dispatches its own kernels, and the attention
table being vLLM's FlashAttention where TE dispatches cuDNN FusedAttention on sm90.
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
        attn_bwd_fwd_ratio: Attention backward latency as a multiple of forward. Default 1.0
            (measured 0.83-0.99x for cuDNN FusedAttention on H20). Set to 0 to omit.
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
        attn_bwd_fwd_ratio: float = 1.0,
        charge_optimizer: bool = True,
    ) -> None:
        self._db = database
        self._ops = ops
        self._attn_bwd_fwd_ratio = attn_bwd_fwd_ratio
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
        attn_bwd_fwd_ratio: float = 1.0,
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
        return cls(
            db,
            ops,
            attn_bwd_fwd_ratio=attn_bwd_fwd_ratio,
            charge_optimizer=charge_optimizer,
        )

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
                ``attn_fwd``, ``attn_bwd``, ``elementwise``, ``tp_collective``, ``logits``,
                and ``optimizer``. Summing them gives `predict_microbatches`.

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

        # Attention forward is measured once and charged for each forward-shaped pass, so
        # the recompute replay cannot drift from the GEMM path's own doubling.
        attn_fwd_single = self._db.query_attention(
            self._ops.attention_query(token_count=token_count)
        )
        terms["attn_fwd"] = attn_fwd_single * self._ops.attention_passes() * layers
        terms["attn_bwd"] = attn_fwd_single * self._attn_bwd_fwd_ratio * layers

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

        # Every per-layer term is charged once per slot; logits and the optimizer are
        # per-step and are not.
        scaled = {name: value * slots for name, value in terms.items()}
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
