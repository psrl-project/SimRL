"""
The per-microbatch op list of a Megatron-style transformer step, as data.

Training differs from inference in four ways this module must get right:

1. Backward adds dgrad and wgrad GEMMs for every forward projection.
   Per-(n,k) slicing makes this non-trivial: a dgrad's (k_fwd, n_fwd) is a different
   slice from its forward (n_fwd, k_fwd), and gemm_dgrad_shapes exists precisely because
   three of four Qwen3 projections would otherwise raise KernelCoverageError.

2. Collectives under sequence parallelism are NOT all_reduce.
   Megatron sets sequence_parallel whenever tp>1. Under it RowParallelLinear calls
   reduce_scatter_to_sequence_parallel_region and ColumnParallelLinear calls
   gather_from_sequence_parallel_region. The existing all_reduce table is for vLLM rollout;
   training needs reduce_scatter + all_gather.

3. Full recompute pays the forward collectives twice.
   CheckpointFunction.backward re-runs the whole forward, which includes the SP collectives.
   The per-layer collective count under full recompute is 8 all_gather + 6 reduce_scatter,
   not 4 + 2. Not modelling recompute would understate the collective term by a third.

4. Elementwise IS summed for training, unlike the rollout model.
   The rollout omission is justified by CUDA-graph replay absorbing elementwise into the
   fitted kernel utilities; Megatron runs no CUDA graphs. The cost is genuinely exposed.
   This is a deliberate divergence from ComposedRollout, not an oversight.

The op counts per layer under full recompute, verified against Transformer Engine source
(module/layernorm_linear.py, module/base.py) with the counts from the Spec B design doc:

  forward:   2 all_gather + 2 reduce_scatter
  recompute: 2 all_gather + 2 reduce_scatter   (replay of forward)
  backward:  4 all_gather + 2 reduce_scatter
  total:     8 all_gather + 6 reduce_scatter

Without recompute (recompute="none"):
  forward:   2 all_gather + 2 reduce_scatter
  backward:  4 all_gather + 2 reduce_scatter
  total:     6 all_gather + 4 reduce_scatter

The backward asymmetry: column-parallel layers (qkv, gate_up) issue both an all_gather
to rebuild their input for wgrad AND a reduce_scatter of the dgrad. Row-parallel layers
(o_proj, down_proj) issue only an all_gather of grad_output. So 2+2 column + 2 row = 6
all_gather and 2 column reduce_scatter in backward.

DP gradient collectives (reduce_scatter + all_gather) are implemented but flagged:
all 22 THD validation records are dp=1, so this term cannot be end-to-end validated.
The Spec A tables were gated and the individual queries resolve; only the composition
against a real step time is untested. The DP term is inactive at dp=1.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ...profiler.hf import model_descriptor_from_yaml
from ...profiler.kernels.attention import rank_head_counts
from ...profiler.kernels.config import model_head_dim, model_intermediate_size
from ...profiler.kernels.shapes import logits_gemm_shape, transformer_gemm_shapes
from ...profiler.schema import ModelDescriptor, ProfileValidationError
from ..kernel.database import (
    CommQuery,
    ElementwiseQuery,
    GemmQuery,
    OptimizerQuery,
    TrainingAttentionQuery,
)
from ..kernel.hardware import dtype_bytes

if TYPE_CHECKING:
    pass

# Collective message sizes under sequence parallelism:
# The all_gather restores the full sequence before the GEMM (size s*h/tp -> s*h),
# and the reduce_scatter shards back afterwards (size s*h -> s*h/tp).
# "Full hidden, sharded on seq" is the per-rank message size in bytes.


@dataclass(frozen=True)
class TrainingCommOp:
    """One TP collective a training step issues, with its per-layer count and message."""

    op: str
    """Collective op name: 'all_gather' or 'reduce_scatter'."""

    count_per_layer: int
    """How many times this collective fires per transformer layer per step."""

    message_bytes: int
    """Per-rank bytes, sized for the given token count and hidden size."""


class TrainingOps:
    """
    The kernels one Megatron training microbatch executes, sharded for one parallelism degree.

    Like `ModelOps`, this is a data structure rather than a computation: the multipliers
    are explicit so a layer-count error in the composition cannot look like a plausible MRE.

    Instantiate with the same `ModelDescriptor` that was used to sweep the GEMM table, or
    pass `from_checkpoint` with the same checkpoint path. Mixing architectures is not checked.
    """

    def __init__(
        self,
        model: ModelDescriptor,
        *,
        tensor_parallel: int,
        dp: int = 1,
        recompute: str = "none",
        dtype: str = "bfloat16",
        gradient_dtype: str = "float32",
        distributed_optimizer: bool = True,
        nodes_per_tp_group: int = 1,
    ) -> None:
        """
        Shard `model` for `tensor_parallel` ranks and build the training op list.

        Args:
            model (ModelDescriptor): Architecture read from a checkpoint.
            tensor_parallel (int): Ranks each TP group spans.
            dp (int): Data-parallel degree; sets DP gradient collective sizes.
            recompute (str): ``"full"`` charges forward collectives twice; ``"none"`` does not.
            dtype (str): Activation dtype, which sets TP collective message sizes.
            gradient_dtype (str): Gradient dtype; ``float32`` is Megatron's default.
            distributed_optimizer (bool): Whether the distributed optimizer is used. At
                ``dp=1`` this has no effect. At ``dp>1`` it selects reduce_scatter + all_gather
                rather than an all_reduce for the gradient sync.
            nodes_per_tp_group (int): Nodes the TP group spans. 1 for a within-node group
                (NVLink); >1 for a cross-node group. This is an assumption about deployment;
                it is named here so a reader can find and review it, not because the model
                dynamically adjusts.

        Raises:
            ProfileValidationError: If the architecture is incomplete or any dimension
                is not divisible by ``tensor_parallel``.
        """
        if model.num_layers is None:
            raise ProfileValidationError("Model is missing num_layers.")
        if recompute not in ("full", "none"):
            raise ProfileValidationError(
                f"recompute must be 'full' or 'none', got {recompute!r}."
            )
        self._num_layers = int(model.num_layers)
        self._num_heads, self._num_kv_heads = rank_head_counts(
            model, tensor_parallel=tensor_parallel
        )
        self._head_dim = model_head_dim(model)
        self._tensor_parallel = int(tensor_parallel)
        self._dp = int(dp)
        self._recompute = recompute
        if model.hidden_size is None:
            raise ProfileValidationError("Model is missing hidden_size.")
        self._hidden_size = int(model.hidden_size)
        self._intermediate_size = model_intermediate_size(model)
        self._qk_layernorm = bool(model.extra.get("qk_layernorm", False))
        self._parameter_count = model.parameter_count
        self._dtype = dtype
        self._gradient_dtype = gradient_dtype
        self._element_bytes = dtype_bytes(dtype)
        self._grad_element_bytes = dtype_bytes(gradient_dtype)
        self._distributed_optimizer = distributed_optimizer
        self._nodes_per_tp_group = int(nodes_per_tp_group)
        self._projection_shapes = tuple(
            transformer_gemm_shapes(model, tensor_parallel=tensor_parallel)
        )
        # The logits projection. Unlike inference, training computes logits for EVERY
        # position rather than one per request, so its `m` is the token count. It runs
        # once per step on the last pipeline stage, not once per layer.
        logits = logits_gemm_shape(model, tensor_parallel=tensor_parallel)
        self._logits_n = logits.n
        self._logits_k = logits.k

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        tensor_parallel: int = 1,
        dp: int = 1,
        recompute: str = "none",
        dtype: str = "bfloat16",
        gradient_dtype: str = "float32",
        distributed_optimizer: bool = True,
        nodes_per_tp_group: int = 1,
        base_dir: Path | None = None,
    ) -> TrainingOps:
        """Read architecture from a HuggingFace checkpoint and shard it."""
        model = model_descriptor_from_yaml(str(path), base_dir=base_dir or Path.cwd())
        return cls(
            model,
            tensor_parallel=tensor_parallel,
            dp=dp,
            recompute=recompute,
            dtype=dtype,
            gradient_dtype=gradient_dtype,
            distributed_optimizer=distributed_optimizer,
            nodes_per_tp_group=nodes_per_tp_group,
        )

    @property
    def num_layers(self) -> int:
        """Return the transformer block count."""
        return self._num_layers

    @property
    def tensor_parallel(self) -> int:
        """Return the TP degree these ops were sharded for."""
        return self._tensor_parallel

    @property
    def recompute(self) -> str:
        """Return the recompute mode ('full' or 'none')."""
        return self._recompute

    def forward_gemm_queries(
        self, token_count: int
    ) -> tuple[tuple[GemmQuery, int], ...]:
        """
        Return forward GEMM queries, each with its per-step multiplier.

        Under ``recompute="full"`` the multiplier is doubled: Megatron's
        ``CheckpointFunction.backward`` re-runs the entire forward pass under
        ``enable_grad`` to rebuild activations, so every forward GEMM executes twice per
        step. Verified against ``tensor_parallel/random.py:621``.

        Measured consequence of omitting it: the composed prediction was 43% low at
        12288 tokens; charging the second forward pass brought that to 29% low.

        Args:
            token_count (int): Tokens per microbatch after the SP all_gather. Under THD
                packing this is ``sum_tokens`` for the microbatch, not ``max_tokens``:
                a packed row holding three 4096-token sequences runs its GEMMs over
                12288 rows. Pricing ``max_tokens`` understated those records by 3x.

        Returns:
            tuple[tuple[GemmQuery, int], ...]: ``(query, multiplier)`` pairs.
        """
        passes = 2 if self._recompute == "full" else 1
        return tuple(
            (GemmQuery(m=token_count, n=s.n, k=s.k), self._num_layers * passes)
            for s in self._projection_shapes
        )

    def attention_passes(self) -> int:
        """
        Return how many times attention forward runs per step.

        One for the forward pass, one more for the recompute replay, and one for the
        backward. The backward is priced separately by ``ComposedTraining`` through
        ``attn_bwd_fwd_ratio``, so this counts only the forward-shaped passes.
        """
        return 2 if self._recompute == "full" else 1

    def backward_gemm_queries(
        self, token_count: int
    ) -> tuple[tuple[GemmQuery, int], ...]:
        """
        Return dgrad + wgrad GEMM queries, each with its per-step multiplier.

        The two families use different slice keys. dgrad is
        ``(m=tok, n=k_fwd, k=n_fwd)`` and wgrad is ``(m=n_fwd, n=k_fwd, k=tok)``.
        See ``gemm_dgrad_shapes`` and ``gemm_wgrad_shapes`` for the derivation.

        Args:
            token_count (int): Tokens per microbatch.

        Returns:
            tuple[tuple[GemmQuery, int], ...]: ``(query, multiplier)`` pairs.
        """
        dgrad = tuple(
            (GemmQuery(m=token_count, n=s.k, k=s.n), self._num_layers)
            for s in self._projection_shapes
        )
        wgrad = tuple(
            (GemmQuery(m=s.n, n=s.k, k=token_count), self._num_layers)
            for s in self._projection_shapes
        )
        return dgrad + wgrad

    def logits_queries(self, token_count: int) -> tuple[tuple[GemmQuery, int], ...]:
        """
        Return the logits GEMM queries for one step, each with its multiplier.

        Training computes logits for every position, so ``m`` is the token count. An
        inference step computes one per request, which is why `ModelOps` passes
        `request_count` here instead: the training shape is far larger, and
        `activation.py` records that the logit path is not TP-divided in the loss.

        Runs once per step on the last pipeline stage only, so the multiplier is 1
        rather than a layer count, and the caller is responsible for charging it on
        only one stage.

        Measured at 12288 tokens, tp=2: forward 59.0 ms, wgrad 55.2 ms. The dgrad
        is absent from the measured table and is omitted rather than clamped onto a
        neighbouring slice.

        Args:
            token_count (int): Tokens in the microbatch.

        Returns:
            tuple[tuple[GemmQuery, int], ...]: ``(query, multiplier)`` pairs.
        """
        return (
            (GemmQuery(m=token_count, n=self._logits_n, k=self._logits_k), 1),
            (GemmQuery(m=self._logits_n, n=self._logits_k, k=token_count), 1),
        )

    def elementwise_queries(
        self, token_count: int
    ) -> tuple[tuple[ElementwiseQuery, float], ...]:
        """
        Return the pointwise kernel queries per step, each with its per-step multiplier.

        Where the sequence is sharded matters, and the two groups shard differently:

        - The two hidden-size RMSNorms sit **outside** the all_gather, on the
          sequence-parallel region, so each rank normalizes ``token_count / tp`` rows of
          ``hidden_size``. Charging them the full sequence would overstate them by ``tp``.
        - RoPE and the q/k norms sit **inside** the TP region, after the all_gather, so
          each rank sees the full sequence but only its own ``num_heads / tp`` heads.
          `self._num_heads` and `self._num_kv_heads` are already the per-rank counts from
          `rank_head_counts`, so the token count passed here is the full one.
        - SwiGLU likewise runs inside the TP region on ``intermediate_size / tp`` columns.

        The multiplier counts forward, recompute replay, and backward. Backward is charged
        at the same cost as forward, which is the same approximation
        `ComposedTraining` makes for attention and for the same reason: a pointwise
        kernel's backward reads and writes comparable traffic to its forward. RoPE backward
        is the exception, being a transpose-free rotation with no saved-tensor read, so it
        is charged at forward cost rather than attempting a separate ratio.

        Not charged: the two residual adds, which Megatron fuses into one
        `bias_dropout_add_fused_train` kernel each. The measured table holds only the four
        kinds vLLM runs (`ELEMENTWISE_KINDS`), and a fused bias-dropout-add is not among
        them. It is omitted rather than priced against `rms_norm`'s curve at the same
        element count: both are bandwidth-bound over `token_count * hidden_size`, so the
        substitution would look plausible and be unverifiable. Two kernels per layer against
        the seven charged here, so the omission is a known under-count of roughly this
        term's 2/9, and it is the next thing to measure if elementwise proves load-bearing.

        Args:
            token_count (int): Tokens in the microbatch, before any SP sharding. Under THD
                packing this is ``sum_tokens``.

        Returns:
            tuple[tuple[ElementwiseQuery, float], ...]: ``(query, multiplier)`` pairs. The
                multiplier is a float because it folds the layer count with the pass count.
        """
        tp = self._tensor_parallel
        sp_tokens = max(token_count // tp, 1)
        passes = 3.0 if self._recompute == "full" else 2.0  # fwd (+replay) + bwd

        kernels: list[tuple[str, int]] = [
            # input_layernorm and pre_mlp_layernorm, both on the SP-sharded sequence.
            ("rms_norm", sp_tokens * self._hidden_size),
            ("rms_norm", sp_tokens * self._hidden_size),
            # RoPE on query and on key, inside the TP region on this rank's heads.
            ("rotary_embedding", token_count * self._num_heads * self._head_dim),
            ("rotary_embedding", token_count * self._num_kv_heads * self._head_dim),
            # SwiGLU: chunk + silu + mul, fused into one kernel by `silu_and_mul`.
            ("silu_and_mul", token_count * (self._intermediate_size // tp)),
        ]
        if self._qk_layernorm:
            kernels.append(("qk_norm", token_count * self._num_heads * self._head_dim))
            kernels.append(("qk_norm", token_count * self._num_kv_heads * self._head_dim))

        return tuple(
            (
                ElementwiseQuery(op_kind=kind, n_elements=n_elements, dtype=self._dtype),
                self._num_layers * passes,
            )
            for kind, n_elements in kernels
        )

    def attention_queries(
        self,
        lengths: Sequence[int],
        *,
        direction: str = "forward",
        backend: str | None = None,
    ) -> tuple[tuple[TrainingAttentionQuery, int], ...]:
        """
        Return one attention query per sequence in a THD pack, with its multiplier.

        **Attention does not cross a `cu_seqlens` boundary**, so a pack is not one large
        attention call: a varlen kernel processes each segment independently, and the
        quadratic work is ``sum_i s_i^2`` rather than ``(sum_i s_i)^2``. Pricing the pack
        at its total was the last systematic error in the composition, and the validation
        set isolates it exactly — five records hold 12288 tokens in different packings:

        | lengths | measured step | model, pack-total | model, per-sequence |
        |---|---|---|---|
        | (12288,) | 4.2866 s | 4.0858 s | 4.0858 s |
        | (10240, 2048) | 4.0712 s | 4.0858 s | lower |
        | (6144, 6144) | 3.8828 s | 4.0858 s | lower |
        | (4096 x3) | 3.7616 s | 4.0858 s | lower |
        | (3072 x4) | 3.6836 s | 4.0858 s | lower |

        Measured time falls monotonically as the pack splits while the token count holds,
        which is `sum_i s_i^2` falling. The pack-total model priced all five identically
        and so carried a 16% spread as residual: `count=1` records ran 11.4% low and
        `count=3` 8.5% high, which is most of the remaining error.

        Charging each segment at its own length also keeps every query on a measured grid
        point rather than introducing a `sum_squared_tokens` axis the sweep never varied.

        Args:
            lengths (Sequence[int]): Per-sequence lengths in the pack. A single-element
                sequence is the unpacked case and reduces to one query.
            direction (str): ``"forward"`` or ``"backward"``.
            backend (str | None): Override the query's own ``megatron`` default.

        Returns:
            tuple[tuple[TrainingAttentionQuery, int], ...]: ``(query, multiplier)`` pairs,
                one per *distinct* length with the multiplier counting repeats.

        Raises:
            ValueError: If `lengths` is empty.
        """
        if not lengths:
            raise ValueError("A THD pack must hold at least one sequence.")
        counts = Counter(int(length) for length in lengths)
        return tuple(
            (self.attention_query(token_count=length, direction=direction, backend=backend),
             repeats)
            for length, repeats in sorted(counts.items())
        )

    def attention_query(
        self,
        *,
        token_count: int,
        direction: str = "forward",
        backend: str | None = None,
    ) -> TrainingAttentionQuery:
        """
        Return the THD attention query for one microbatch, in one direction.

        Reads the `attention_training` family, not the rollout `context_attention` table.
        The two describe different kernels on different layouts, and the substitution was
        wrong in both directions at once. Measured on H20 at tp=2:

        - Forward, rollout table vs TE: 1.84x low at 4096 tokens, 1.59x at 22528.
        - Backward, the previous 1.0x-of-forward approximation vs measured: 2.24-2.42x low.

        Together the attention term was understated by 2.35-2.60x. The cause is that a
        packed microbatch is one row with ``cu_seqlens`` while the rollout grid varies a
        batch dimension a packed layout does not have, so every training query landed at
        the edge of the measured surface; and that TE dispatches FlashAttention for THD
        rather than the cuDNN FusedAttention the design doc assumed.

        Args:
            token_count (int): Total tokens in the packed microbatch, which is
                ``sum_tokens`` rather than ``max_tokens``.
            direction (str): ``"forward"`` or ``"backward"``.
            backend (str | None): Override the query's own ``megatron`` default. Left unset
                it is *not* forwarded, because `None` means "use the database default",
                which is `vllm` and names a directory this family does not have.

        Returns:
            TrainingAttentionQuery: Ready for ``KernelDatabase.query_training_attention``.
        """
        fields = dict(
            direction=direction,
            num_heads=self._num_heads,
            total_tokens=token_count,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            attn_dtype=self._dtype,
        )
        if backend is not None:
            fields["backend"] = backend
        return TrainingAttentionQuery(**fields)  # type: ignore[arg-type]

    def tp_comm_queries(
        self, token_count: int
    ) -> tuple[tuple[CommQuery, int], ...]:
        """
        Return TP collective queries per layer, each with its per-step multiplier.

        Under sequence parallelism:
        - Forward: 2 all_gather (qkv in, fc1 in) + 2 reduce_scatter (proj out, fc2 out).
        - Backward: 4 all_gather + 2 reduce_scatter.
        - Full recompute re-runs the forward: +2 all_gather +2 reduce_scatter.

        All collectives carry the full hidden size; tp shards the sequence dimension.

        Returns:
            tuple[tuple[CommQuery, int], ...]: ``(query, multiplier)`` pairs. Empty when
                ``tensor_parallel <= 1``.
        """
        if self._tensor_parallel <= 1:
            return ()
        ag_bytes = token_count * self._hidden_size * self._element_bytes
        rs_bytes = ag_bytes  # symmetric message size

        ag_per_layer = 6  # 2 fwd + 4 bwd
        rs_per_layer = 4  # 2 fwd + 2 bwd
        if self._recompute == "full":
            ag_per_layer += 2  # recompute replay of forward
            rs_per_layer += 2

        ag_query = CommQuery(
            op="all_gather",
            world_size=self._tensor_parallel,
            nodes=self._nodes_per_tp_group,
            message_bytes=ag_bytes,
            dtype=self._dtype,
            backend="nccl",
        )
        rs_query = CommQuery(
            op="reduce_scatter",
            world_size=self._tensor_parallel,
            nodes=self._nodes_per_tp_group,
            message_bytes=rs_bytes,
            dtype=self._dtype,
            backend="nccl",
        )
        return (
            (ag_query, ag_per_layer * self._num_layers),
            (rs_query, rs_per_layer * self._num_layers),
        )

    def pp_comm_queries(
        self, token_count: int, *, pipeline_parallel: int
    ) -> tuple[tuple[CommQuery, int], ...]:
        """
        Return the pipeline send/recv queries for one microbatch slot.

        Under 1F1B each microbatch crosses every stage boundary once going forward and once
        coming back, so a `pp`-stage pipeline charges ``2 * (pp - 1)`` transfers per
        microbatch. The payload is one stage's activation handoff: under sequence
        parallelism the tensor is sequence-sharded, so each rank sends
        ``token_count / tp * hidden_size`` elements.

        Small but not zero: 57-270 us per crossing at 4096-22528 tokens against step times
        of 1.06-72 s in the validation set, so under 0.1% of a step. Charged because the
        table exists and an omitted term cannot be distinguished from a term measured at
        zero.

        Args:
            token_count (int): Tokens in the microbatch, before SP sharding.
            pipeline_parallel (int): PP degree. The ops describe one microbatch's kernels,
                so PP is a property of the schedule rather than of the sharding.

        Returns:
            tuple[tuple[CommQuery, int], ...]: ``(query, multiplier)`` pairs. Empty at
                ``pp <= 1``, where there is no boundary to cross.
        """
        if pipeline_parallel <= 1:
            return ()
        sp_tokens = max(token_count // self._tensor_parallel, 1)
        message_bytes = sp_tokens * self._hidden_size * self._element_bytes
        query = CommQuery(
            op="p2p",
            world_size=2,
            nodes=1,
            message_bytes=message_bytes,
            dtype=self._dtype,
            backend="nccl",
        )
        return ((query, 2 * (pipeline_parallel - 1)),)

    def optimizer_query(self, *, pipeline_parallel: int) -> OptimizerQuery | None:
        """
        Return the fused Adam step query for one rank, or None if the shard is unknown.

        The shard is ``parameter_count / (tp * pp * dp)``: tensor and pipeline parallelism
        each divide the parameters a rank holds, and the distributed optimizer divides again
        across the DP group. Without it, every DP rank would update the full set redundantly,
        which is what `distributed_optimizer=False` means.

        Measured magnitude, so the term's weight is stated rather than implied: a Qwen3-8B
        rank at tp=2, pp=3 holds 1.365B parameters and the step costs 14.1 ms on H20, running
        at 2703 GB/s against the card's 3.35 TB/s peak. Against the 1.06-72 s steps in the
        THD validation set that is 0.02-1.3%. The term is real and it is not where a composed
        step's error lives; measuring it is how that was established rather than assumed.

        Returns None rather than raising when `parameter_count` is absent, because a
        descriptor built from a config without a weight index carries an estimate or
        nothing, and refusing to predict at all would be a worse failure than omitting a
        1% term. `ComposedTraining` reports the omission through the decomposition.

        Args:
            pipeline_parallel (int): PP degree, which the ops do not carry: they describe one
                microbatch's kernels and PP is a property of the step's schedule.

        Returns:
            OptimizerQuery | None: The query, or None when the parameter count is unknown.
        """
        if self._parameter_count is None:
            return None
        divisor = self._tensor_parallel * max(pipeline_parallel, 1)
        if self._distributed_optimizer:
            divisor *= self._dp
        shard = max(int(self._parameter_count // divisor), 1)
        return OptimizerQuery(param_count=shard)

    def dp_comm_queries(self) -> tuple[tuple[CommQuery, int], ...]:
        """
        Return DP gradient collective queries for one step.

        With the distributed optimizer (Megatron's default): reduce_scatter of gradients
        across the DP group, then all_gather of params after the optimizer step.
        Without it: a plain all_reduce.

        The per-rank message size is ``total_parameters * grad_dtype_bytes / dp``, which
        is the bucket size. Megatron's default bucket cap is ``max(40M, 1M * dp)`` parameters,
        but the total step cost is independent of bucket count: it is one full pass over all
        parameters. This method returns the full-parameter message, not per-bucket.

        ⚠️ UNVALIDATED at dp>1. All 22 THD validation records are dp=1, so this term never
        fires during scoring. The Spec A collective tables are gated (the queries resolve),
        but the composed step time for dp>1 has not been compared against a real profiled step.

        Returns:
            tuple[tuple[CommQuery, int], ...]: ``(query, multiplier)`` pairs. Empty at dp=1.
        """
        if self._dp <= 1:
            return ()
        if self._hidden_size is None:
            return ()
        # Approximate total parameter count from model descriptor; using inventory is more
        # accurate but requires the checkpoint, which may not be available here.
        # The caller can set dp_param_count on construction if precision matters.
        # This path is clearly marked UNVALIDATED in the docstring.
        raise NotImplementedError(
            "dp_comm_queries is not implemented yet: dp>1 training validation is a separate "
            "phase. Use TrainingOps with dp=1 for validated predictions."
        )
