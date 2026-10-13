"""
The per-step op list of a transformer, as data rather than as inline arithmetic.

A step's latency is the sum over entries of `query(entry) * multiplier`. Making that a list
is what lets the logits projection be marked once-per-step: written inline, it sits one
indentation level away from being multiplied by the layer count, and the resulting error is a
plausible-looking 36x that no unit test on a single kernel would catch.

Sharding reuses `dryrun.profiler.kernel`' own functions rather than reimplementing the same
divisions. If the composition sharded differently from the sweep, it would query shapes that
were never measured: or worse, query a measured shape corresponding to different work. The
profiler raises rather than floor-dividing an indivisible degree, and that behaviour is
inherited here for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...profiler.hf import model_descriptor_from_yaml
from ...profiler.kernel.grid.attention import rank_head_counts
from ...profiler.kernel.grid.gdn import gdn_head_counts
from ...profiler.kernel.grid.gemm import logits_gemm_shape, transformer_gemm_shapes
from ...profiler.kernel.utils.model_dims import model_head_dim
from ...profiler.schema import ModelDescriptor, ProfileValidationError
from .database import AttentionQuery, CommQuery, GdnQuery, GemmQuery
from .hardware import dtype_bytes


@dataclass(frozen=True)
class GemmOp:
    """One GEMM a step executes, and whether it runs per layer or once."""

    name: str
    n: int
    k: int
    per_layer: bool


class ModelOps:
    """The kernels one inference step executes, sharded for one parallelism degree."""

    def __init__(
        self,
        model: ModelDescriptor,
        *,
        tensor_parallel: int,
        dtype: str = "bfloat16",
    ) -> None:
        """
        Shard `model` for `tensor_parallel` ranks and build the op list.

        Args:
            model (ModelDescriptor): Architecture read from a checkpoint.
            tensor_parallel (int): Ranks the model is sharded across.
            dtype (str): Activation dtype, which sets the all-reduce message size.

        Raises:
            ProfileValidationError: If the architecture is incomplete, or any dimension is
                not divisible by `tensor_parallel`.
        """
        if model.num_layers is None:
            raise ProfileValidationError("Model is missing num_layers.")
        self._num_layers = int(model.num_layers)
        self._num_heads, self._num_kv_heads = rank_head_counts(
            model, tensor_parallel=tensor_parallel
        )
        self._head_dim = model_head_dim(model)
        self._tensor_parallel = int(tensor_parallel)
        if model.hidden_size is None:
            raise ProfileValidationError("Model is missing hidden_size.")
        # The all-reduce carries the full hidden size: TP shards the projections, then reduces
        # across ranks to reassemble the unsharded activation.
        self._hidden_size = int(model.hidden_size)
        self._dtype = dtype
        self._element_bytes = dtype_bytes(dtype)
        self._projection_ops = tuple(
            GemmOp(name=shape.name, n=shape.n, k=shape.k, per_layer=True)
            for shape in transformer_gemm_shapes(model, tensor_parallel=tensor_parallel)
        )
        logits = logits_gemm_shape(model, tensor_parallel=tensor_parallel)
        self._logits_op = GemmOp(name=logits.name, n=logits.n, k=logits.k, per_layer=False)

        # Hybrid attention. A model with `full_attention_interval: N` runs softmax attention
        # on every N-th layer and a gated delta net on the rest, so the two kernels must be
        # multiplied by their own layer counts rather than both by `num_layers`.
        #
        # Getting this wrong is not a small error: Qwen3.5-35B-A3B has 10 softmax layers and
        # 30 GDN ones, so charging all 40 as softmax overstates a 32k-token prefill by about
        # 4x, and charging all 40 as GDN understates it by about as much. `_attention_layers`
        # is `num_layers` for a dense model, which keeps the dense path byte-identical.
        # MoE routing, used by `gemm_queries` to price the expert GEMMs per activated expert
        # rather than once over every token. Zero for a dense model, which keeps that path
        # byte-identical.
        self._num_experts = int(model.extra.get("num_experts") or 0)
        self._experts_per_tok = int(model.extra.get("num_experts_per_tok") or 1)

        interval = model.extra.get("full_attention_interval")
        if interval and int(interval) > 1:
            stride = int(interval)
            self._attention_layers = self._num_layers // stride
            self._gdn_layers = self._num_layers - self._attention_layers
            self._gdn_heads = gdn_head_counts(model, tensor_parallel=tensor_parallel)
        else:
            self._attention_layers = self._num_layers
            self._gdn_layers = 0
            self._gdn_heads = None

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        tensor_parallel: int = 1,
        dtype: str = "bfloat16",
        base_dir: Path | None = None,
    ) -> ModelOps:
        """
        Read architecture from a HuggingFace checkpoint and shard it.

        Args:
            path (str | Path): Checkpoint directory or hub id.
            tensor_parallel (int): Ranks the model is sharded across.
            dtype (str): Activation dtype, which sets the all-reduce message size.
            base_dir (Path | None): Directory a relative `path` resolves against, defaulting
                to the process working directory. A config-driven caller should pass the
                config file's own directory, so a relative checkpoint path means the same
                thing wherever the process was launched from. Matches
                `KernelProfileConfig.from_dict`.

        Returns:
            ModelOps: The op list for one rank.
        """
        model = model_descriptor_from_yaml(str(path), base_dir=base_dir or Path.cwd())
        return cls(model, tensor_parallel=tensor_parallel, dtype=dtype)

    @property
    def num_layers(self) -> int:
        """Return the transformer block count."""
        return self._num_layers

    @property
    def attention_layers(self) -> int:
        """Return how many layers run softmax attention.

        Equal to `num_layers` for a dense model. On a hybrid model it is
        `num_layers // full_attention_interval`, and the remainder run a gated delta net
        instead -- 10 of 40 on Qwen3.5-35B-A3B.
        """
        return self._attention_layers

    @property
    def gdn_layers(self) -> int:
        """Return how many layers run a gated delta net; 0 for a dense model."""
        return self._gdn_layers

    @property
    def num_experts(self) -> int:
        """Return the routed expert count per MoE block, or 0 for a dense model."""
        return self._num_experts

    def gdn_query(
        self,
        phase: str,
        *,
        total_tokens: int = 1,
        batch_size: int = 1,
        backend: str | None = None,
    ) -> GdnQuery:
        """
        Build the query for one gated-delta-net call on this rank.

        Args:
            phase (str): `context` or `decode`.
            total_tokens (int): Packed token total, for the context phase.
            batch_size (int): Concurrent requests, for the decode phase.
            backend (str | None): Table backend, or None for the default.

        Returns:
            GdnQuery: Ready to pass to `KernelDatabase.query_gdn`.

        Raises:
            ProfileValidationError: If this model has no gated-delta-net layers.
        """
        if self._gdn_heads is None:
            raise ProfileValidationError(
                "This model has no gated-delta-net layers, so there is no GDN call to price. "
                "A hybrid model declares `full_attention_interval` in its config."
            )
        key_heads, value_heads, head_dim = self._gdn_heads
        return GdnQuery(
            phase=phase,
            num_key_heads=key_heads,
            num_value_heads=value_heads,
            head_dim=head_dim,
            total_tokens=total_tokens,
            batch_size=batch_size,
            backend=backend,
        )

    @property
    def num_heads(self) -> int:
        """Return the per-rank query head count."""
        return self._num_heads

    @property
    def num_kv_heads(self) -> int:
        """Return the per-rank key/value head count."""
        return self._num_kv_heads

    @property
    def head_dim(self) -> int:
        """Return the per-head dimension."""
        return self._head_dim

    @property
    def tensor_parallel(self) -> int:
        """Return the parallelism degree these ops were sharded for."""
        return self._tensor_parallel

    @property
    def projection_ops(self) -> tuple[GemmOp, ...]:
        """Return the per-layer projection GEMMs."""
        return self._projection_ops

    @property
    def logits_op(self) -> GemmOp:
        """Return the once-per-step vocabulary projection."""
        return self._logits_op

    def gemm_queries(
        self, token_count: int, request_count: int
    ) -> tuple[tuple[GemmQuery, int], ...]:
        """
        Return the GEMM queries of one step, each with its multiplier.

        Args:
            token_count (int): Tokens the projections run over. In decode this equals the
                request count; in prefill it is the total prompt token count.
            request_count (int): Sequences in the batch. The logits projection runs over this
                many tokens, because vLLM computes logits only for each sequence's last
                token.

        Returns:
            tuple[tuple[GemmQuery, int], ...]: `(query, multiplier)` pairs. The multiplier is
                the layer count for a per-layer op and 1 for the logits projection.
        """
        queries: list[tuple[GemmQuery, int]] = []
        for op in self._projection_ops:
            if op.name.startswith("expert_") and self._num_experts:
                # One query per layer, not one per activated expert. The real kernel is a
                # GROUPED gemm: a single launch covers every expert the step routes to, so
                # the launch cost occurs once per layer.
                #
                # Decomposing it per expert was tried and is badly wrong in the other
                # direction: at 64 tokens the step activates 221 experts, each becoming an
                # `m=2` GEMM whose measured cost is almost entirely launch floor (10.45 us),
                # so 40 layers x 221 experts charged 92 ms against a real 29 ms step and
                # pushed decode MRE from 43% to 450%.
                #
                # The genuine term this form still misses is WEIGHT traffic. A grouped GEMM
                # reads every activated expert's weights from HBM -- 1536 MiB per layer at 64
                # tokens against the 48 MiB this shape implies -- which is ~16 ms of the real
                # 29 ms decode step. Fixing that belongs in the GEMM SOL, as a weight-read
                # term scaled by `_activated_experts`, not in the query shape: the shape has
                # to stay one the sweep measured. Recorded as a known limitation rather than
                # papered over; see `docs/composed-rollout.md`.
                queries.append((GemmQuery(m=token_count, n=op.n, k=op.k), self._num_layers))
                continue
            queries.append((GemmQuery(m=token_count, n=op.n, k=op.k), self._num_layers))
        queries.append((GemmQuery(m=request_count, n=self._logits_op.n, k=self._logits_op.k), 1))
        return tuple(queries)

    def expert_weight_bytes(self, token_count: int) -> int:
        """
        Return the expert weight bytes one step reads from HBM, across all layers.

        Zero for a dense model. For a mixture-of-experts block this is the term the fused
        per-token GEMM shape cannot express, and on a small batch it dominates the step: the
        real kernel is a grouped GEMM that reads every ACTIVATED expert's weights whether
        that expert received two tokens or two hundred, while an `m=token_count` GEMM query
        implies only `experts_per_tok` copies.

        Measured consequence on Qwen3.5-35B-A3B: a 64-token decode step activates 221 of 256
        experts, so the real traffic is 1536 MiB per layer against the 48 MiB the query
        implies -- about 16 ms of a 29 ms step, which is most of the gap the composed model
        showed before this term existed.

        Kept out of `gemm_sol` on purpose. That function also normalises MEASURED rows into
        utility, and every measured row is a dense GEMM with one weight copy; adding an
        expert multiplier there would corrupt the interpolation surface rather than only the
        prediction. Kept out of the query shape too, because a shape the sweep never measured
        raises `KernelCoverageError`, and decomposing into one GEMM per expert overcharges by
        3x in the other direction (each tiny GEMM then pays a launch floor that physically
        occurs once per layer).

        Args:
            token_count (int): Tokens in the step.

        Returns:
            int: Bytes of expert weights read, summed over every layer.
        """
        if not self._num_experts:
            return 0
        activated = self._activated_experts(token_count)
        per_expert = 0
        for op in self._projection_ops:
            if not op.name.startswith("expert_"):
                continue
            # The fused shape already carries `experts_per_tok` on one axis, so dividing the
            # product by it recovers ONE expert's weight element count regardless of which
            # axis that was.
            per_expert += op.n * op.k // self._experts_per_tok
        return per_expert * activated * self._element_bytes * self._num_layers

    def expert_weight_bytes_counted(self, token_count: int) -> int:
        """Return the expert weight bytes the GEMM queries already charge.

        One `(k, n)` copy per expert projection per layer, which is what an
        `m=token_count` query's `k * n` memory term carries. Subtracted from
        `expert_weight_bytes` so the grouped-GEMM weight read is added once, not twice.
        """
        if not self._num_experts:
            return 0
        counted = sum(
            op.n * op.k for op in self._projection_ops if op.name.startswith("expert_")
        )
        return counted * self._element_bytes * self._num_layers

    def tokens_per_expert(self, token_count: int) -> float:
        """
        Return the mean token count one activated expert receives.

        `token_count * experts_per_tok / activated_experts`, floored at 1. This is the factor
        a grouped GEMM amortises its weight read over: an expert serving one token must fetch
        its whole matrix for that token, while an expert serving fifty fetches it once for
        all fifty.

        At decode it is ~1 by construction (64 tokens x topk 8 over 221 experts is 2.3), so
        the weight read is charged almost in full. At a 4096-token prefill it is 128, so the
        same read costs a 128th as much per step -- which is what the measured 4.24 ms
        256-token step requires, since the unamortised figure exceeds the device's total
        traffic budget for that step.

        Args:
            token_count (int): Tokens in the step.

        Returns:
            float: Mean tokens per activated expert, at least 1.0.
        """
        if not self._num_experts:
            return 1.0
        activated = self._activated_experts(token_count)
        return max(float(token_count) * self._experts_per_tok / activated, 1.0)

    def _activated_experts(self, token_count: int) -> int:
        """
        Return how many distinct experts a step of `token_count` tokens touches per layer.

        Each token independently picks `experts_per_tok` of `num_experts`, so the expected
        number of distinct experts hit is the classic coupon-collector complement
        `E * (1 - (1 - 1/E)^(t * k))`, saturating at `E`. A balanced router is assumed, which
        is what auxiliary-loss training optimises for.

        This matters because it is the weight-read multiplier. At one token only `k` experts
        are read; by 64 tokens almost every expert is, so the expert weight traffic grows 28x
        over that range while a per-token shape would call it constant.

        Args:
            token_count (int): Tokens in the step.

        Returns:
            int: Distinct experts activated, in `[1, num_experts]`.
        """
        draws = max(int(token_count), 1) * self._experts_per_tok
        expected = self._num_experts * (1.0 - (1.0 - 1.0 / self._num_experts) ** draws)
        return max(1, min(self._num_experts, int(round(expected))))

    def comm_queries(self, token_count: int) -> tuple[tuple[CommQuery, int], ...]:
        """
        Return the all-reduce queries of one step, each with its multiplier.

        vLLM issues one all-reduce after the attention block's output projection and one after
        the MLP's down projection, so a tensor-parallel step pays two per layer. The count is
        returned as data with an explicit multiplier for the same reason `logits_op` carries
        one: written inline it sits one indentation level away from a plausible-looking
        layer-count error.

        Both `op` and `nodes` are passed explicitly rather than defaulted. vLLM keeps its
        tensor-parallel group inside one node, so `nodes=1` is correct here, but it is an
        assumption about deployment rather than a property of the model, and a cross-node
        group read against an intra-node table is optimistic by the NVLink-to-NIC ratio.
        Naming it is what makes it reviewable.

        Args:
            token_count (int): Tokens the all-reduce covers. In decode this equals the request
                count; in prefill it is the total prompt token count.

        Returns:
            tuple[tuple[CommQuery, int], ...]: `(query, multiplier)` pairs, empty at
                `tensor_parallel == 1` because no collective runs.
        """
        if self._tensor_parallel <= 1:
            return ()
        message_bytes = token_count * self._hidden_size * self._element_bytes
        query = CommQuery(
            op="all_reduce",
            world_size=self._tensor_parallel,
            nodes=1,
            message_bytes=message_bytes,
            dtype=self._dtype,
        )
        return ((query, 2 * self._num_layers),)

    def attention_query(
        self,
        phase: str,
        *,
        batch_size: int,
        seq_len: int = 1,
        step: int = 0,
        backend: str | None = None,
    ) -> AttentionQuery:
        """
        Return the attention query of one step, with this rank's head counts.

        Args:
            phase (str): `context` or `generation`.
            batch_size (int): Sequences attending.
            seq_len (int): Context-phase sequence length; must stay 1 for generation.
            step (int): Generation-phase KV depth; must stay 0 for context.
            backend (str | None): Backend override; `None` selects the database default.

        Returns:
            AttentionQuery: Ready to pass to `KernelDatabase.query_attention`.
        """
        return AttentionQuery(
            phase=phase,
            batch_size=batch_size,
            num_heads=self._num_heads,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            seq_len=seq_len,
            step=step,
            backend=backend,
        )
