"""
The per-step op list of a transformer, as data rather than as inline arithmetic.

A step's latency is the sum over entries of `query(entry) * multiplier`. Making that a list
is what lets the logits projection be marked once-per-step: written inline, it sits one
indentation level away from being multiplied by the layer count, and the resulting error is a
plausible-looking 36x that no unit test on a single kernel would catch.

Sharding reuses `dryrun.profiler.kernels`' own functions rather than reimplementing the same
divisions. If the composition sharded differently from the sweep, it would query shapes that
were never measured -- or worse, query a measured shape corresponding to different work. The
profiler raises rather than floor-dividing an indivisible degree, and that behaviour is
inherited here for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...profiler.hf import model_descriptor_from_yaml
from ...profiler.kernels.attention import rank_head_counts
from ...profiler.kernels.config import model_head_dim
from ...profiler.kernels.shapes import logits_gemm_shape, transformer_gemm_shapes
from ...profiler.schema import ModelDescriptor, ProfileValidationError
from .database import AttentionQuery, CommQuery, GemmQuery
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
        queries = [
            (GemmQuery(m=token_count, n=op.n, k=op.k), self._num_layers)
            for op in self._projection_ops
        ]
        queries.append((GemmQuery(m=request_count, n=self._logits_op.n, k=self._logits_op.k), 1))
        return tuple(queries)

    def comm_queries(self, token_count: int) -> tuple[tuple[CommQuery, int], ...]:
        """
        Return the all-reduce queries of one step, each with its multiplier.

        vLLM issues one all-reduce after the attention block's output projection and one after
        the MLP's down projection, so a tensor-parallel step pays two per layer. The count is
        returned as data with an explicit multiplier for the same reason `logits_op` carries
        one: written inline it sits one indentation level away from a plausible-looking
        layer-count error.

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
            world_size=self._tensor_parallel, message_bytes=message_bytes, dtype=self._dtype
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
