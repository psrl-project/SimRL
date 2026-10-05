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

Optimizer step cost is treated as a fixed per-step overhead. Its profiled value is a
property of the host-side Adam kernel time and the memory bandwidth over the parameter
shards, both of which are dominated by parameter count rather than sequence length. The
default is zero (no term), which expresses that it is not yet measured; a caller can pass
a measured value, and the per-term decomposition surfaces its magnitude clearly.
"""

from __future__ import annotations

from pathlib import Path

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
        optimizer_overhead_s: Per-step optimizer cost in seconds. Default 0.0 (unmeasured).
            Measured on H20 with Adam + distributed optimizer = approximately 0.15-0.35s for
            an 8.19B parameter model, but this is highly configuration-dependent.

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
        optimizer_overhead_s: float = 0.0,
    ) -> None:
        self._db = database
        self._ops = ops
        self._attn_bwd_fwd_ratio = attn_bwd_fwd_ratio
        self._optimizer_overhead_s = optimizer_overhead_s

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
        optimizer_overhead_s: float = 0.0,
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
        return cls(db, ops, attn_bwd_fwd_ratio=attn_bwd_fwd_ratio,
                   optimizer_overhead_s=optimizer_overhead_s)

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
        num_microbatches = len(microbatches)
        slots = pipeline_slots(num_microbatches, parallelism.pp)

        # Price a single stage's worth of work: num_layers / pp.
        # The slot cost scales the op multipliers to per-stage rather than per-model.
        layers_per_stage = self._ops.num_layers / parallelism.pp

        # Price each microbatch at sum_tokens (THD packing packs multiple sequences into
        # one row, so GEMMs run over sum_tokens not max_tokens; using max_tokens understated
        # packed records by the packing factor).
        avg_sum_tok = sum(mb.sum_tokens for mb in microbatches) / max(len(microbatches), 1)
        tok = int(avg_sum_tok)

        slot_latency = self._price_stage(tok, layers_per_stage)

        return slots * slot_latency + self._optimizer_overhead_s

    def _price_stage(
        self,
        token_count: int,
        layers_per_stage: float,
        *,
        charge_logits: bool = True,
    ) -> float:
        """Price one pipeline stage's worth of work (forward + backward, with recompute).

        Args:
            token_count: Tokens in the microbatch. Under THD packing this is
                ``sum_tokens``, not ``max_tokens``.
            layers_per_stage: ``num_layers / pp``. Fractional values are fine.
            charge_logits: Whether to charge the logits GEMM. It runs once per step on
                the last pipeline stage only, so a caller pricing a middle stage passes
                False. Default True because the single-stage (pp=1) case always pays it.
        """
        total = 0.0
        layers = layers_per_stage  # fractional is fine: the multiplier is linear

        # Forward GEMMs: multiplier already includes recompute doubling (in TrainingOps)
        for query, mult_full in self._ops.forward_gemm_queries(token_count):
            total += self._db.query_gemm(query) * (mult_full * layers / self._ops.num_layers)

        # Backward GEMMs (dgrad + wgrad): each runs once per token per layer
        for query, _ in self._ops.backward_gemm_queries(token_count):
            total += self._db.query_gemm(query) * layers

        # Attention: num_passes (1 or 2 under recompute) for forward
        attn_query = self._ops.attention_query(token_count=token_count)
        attn_fwd_single = self._db.query_attention(attn_query)
        attn_passes = self._ops.attention_passes()
        total += attn_fwd_single * attn_passes * layers

        # Attention backward
        total += attn_fwd_single * self._attn_bwd_fwd_ratio * layers

        # TP collectives: scale per-layer count proportionally
        for query, mult_full in self._ops.tp_comm_queries(token_count):
            mult_stage = mult_full * layers / self._ops.num_layers
            total += self._db.query_comm(query) * mult_stage

        # Logits: once per step on the last stage, over EVERY token (not one per
        # request as in inference). A missing dgrad slice is skipped rather than
        # clamped onto a neighbouring projection's curve.
        if charge_logits:
            for query, mult in self._ops.logits_queries(token_count):
                try:
                    total += self._db.query_gemm(query) * mult
                except KernelCoverageError:
                    pass

        return total

    def _price_microbatch(self, token_count: int) -> float:
        """Price one full model microbatch (forward + backward, all layers). Used for decompose."""
        return self._price_stage(token_count, self._ops.num_layers)

    def decompose(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> dict[str, float]:
        """
        Return the latency contribution of each op family for one step.

        This is the per-term decomposition Spec B calls for: the value of a kernel sum
        is that a miss points to a named op. The dict keys match the trainable terms;
        the sum of values equals ``predict()``.

        Returns:
            dict[str, float]: Keys are ``"gemm_fwd"``, ``"gemm_bwd"``, ``"attn_fwd"``,
                ``"attn_bwd"``, ``"tp_collective"``, ``"optimizer"``, ``"bubble"``.
                All values are in seconds.
        """
        if parallelism.tp != self._ops.tensor_parallel:
            raise ValueError(
                f"parallelism.tp {parallelism.tp} does not match ops.tensor_parallel "
                f"{self._ops.tensor_parallel}."
            )

        planner = MicrobatchPlanner(
            BatchingMode.FIXED_SIZE,
            max_tokens_per_rank=workload.max_tokens_per_microbatch,
        )
        microbatches = planner.plan(
            workload.sequence_lengths,
            parallelism.dp,
            micro_batch_size=workload.micro_batch_size,
        )
        num_microbatches = len(microbatches)
        slots = pipeline_slots(num_microbatches, parallelism.pp)
        avg_token_count = sum(mb.max_tokens for mb in microbatches) / max(len(microbatches), 1)
        tok = int(avg_token_count)

        gemm_fwd = sum(
            self._db.query_gemm(q) * m for q, m in self._ops.forward_gemm_queries(tok)
        )
        gemm_bwd = sum(
            self._db.query_gemm(q) * m for q, m in self._ops.backward_gemm_queries(tok)
        )
        attn_fwd_s = self._db.query_attention(self._ops.attention_query(token_count=tok))
        attn_fwd = attn_fwd_s * self._ops.num_layers
        attn_bwd = attn_fwd * self._attn_bwd_fwd_ratio
        tp_coll = sum(
            self._db.query_comm(q) * m for q, m in self._ops.tp_comm_queries(tok)
        )
        slot_latency = gemm_fwd + gemm_bwd + attn_fwd + attn_bwd + tp_coll
        bubble = (slots - num_microbatches) * slot_latency / max(num_microbatches, 1)

        return {
            "gemm_fwd": gemm_fwd * slots,
            "gemm_bwd": gemm_bwd * slots,
            "attn_fwd": attn_fwd * slots,
            "attn_bwd": attn_bwd * slots,
            "tp_collective": tp_coll * slots,
            "optimizer": self._optimizer_overhead_s,
            "bubble": bubble,
        }
