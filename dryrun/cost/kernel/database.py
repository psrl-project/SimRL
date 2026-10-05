"""Answer kernel shape queries from measured perf tables.

One query is `latency = SOL(shape, target_spec) / interpolated_utility`. Splitting the
prediction this way is what lets a table measured on one GPU price a shape on another:
utility is a property of the kernel and the silicon it ran on, while SOL carries the
target hardware's peak throughput. Cross-hardware transfer is possible by construction
here but is unvalidated with a single GPU type available, and must not be claimed.

`query.backend` of `None` means the backend vLLM's own selector picks, which is the
default a prediction should use. An explicit backend prices the same shape on an
alternative so backend choices can be compared without moving the default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .floor import load_floors as _load_floors
from .hardware import GpuSpec, canonical_dtype, gpu_slug, load_gpu_spec
from .interp import KernelCoverageError
from .ops.base import op_for
from .sol import FALLBACK_FLOORS_S
from .surface import UtilitySurface
from .table import KernelTable

_DEFAULT_BACKEND = "vllm"
# The collective backend names the inference server, not the communication library, because the
# server is what chooses the library. `CudaCommunicator.all_reduce` reaches
# `torch.distributed.all_reduce` only as a last resort: on H20 (sm_90) at two ranks vLLM's own
# custom P2P kernel covers every message under 8 MiB, and a Qwen3-8B decode step's message is
# 8 KiB to 512 KiB. So a `tp=2` server issues zero NCCL all-reduces, and a table measured with
# `torch.distributed.all_reduce` prices a kernel it never invokes. Pass `backend="nccl"`
# explicitly to read the control table the collective sweep also writes.
_COLLECTIVE_BACKEND = "vllm"
# The optimizer backend names the training framework, not the inference server: the measured
# kernel is TE's `FusedAdam`, which `megatron/core/optimizer/__init__.py` imports in
# preference to apex's and to torch's. vLLM runs no optimizer, so the module-level `vllm`
# default would resolve to a directory that cannot exist.
_OPTIMIZER_BACKEND = "megatron"
# Training attention is measured through Transformer Engine, which is what Megatron calls,
# so it shares the `megatron` backend directory with the optimizer rather than `vllm`.
_TRAINING_BACKEND = "megatron"
_PHASES = ("context", "generation")


@dataclass(frozen=True)
class GemmQuery:
    """One per-rank GEMM shape to price."""

    m: int
    n: int
    k: int
    dtype: str = "bfloat16"
    backend: str | None = None

    def __post_init__(self) -> None:
        for name in ("m", "n", "k"):
            if getattr(self, name) <= 0:
                raise KernelCoverageError(f"GEMM {name} must be positive, got {getattr(self, name)!r}.")

    @property
    def op_name(self) -> str:
        """Return the parquet op name this query reads."""
        return "gemm"

    @property
    def slice_key(self) -> tuple[Any, ...]:
        """Return the categorical key in `GEMM_OP.slice_columns` order."""
        return (canonical_dtype(self.dtype), self.n, self.k)

    @property
    def coords(self) -> tuple[float, ...]:
        """Return interpolation coordinates in `GEMM_OP.axis_columns` order."""
        return (float(self.m),)


@dataclass(frozen=True)
class AttentionQuery:
    """One per-rank attention call to price."""

    phase: str
    batch_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    seq_len: int = 1
    step: int = 0
    attn_dtype: str = "bfloat16"
    kv_cache_dtype: str = "bfloat16"
    block_size: int = 16
    window_size: int = 0
    backend: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise KernelCoverageError(
                f"Attention phase must be one of {', '.join(_PHASES)}, got {self.phase!r}."
            )
        for name in ("batch_size", "num_heads", "num_kv_heads", "head_dim"):
            if getattr(self, name) <= 0:
                raise KernelCoverageError(f"Attention {name} must be positive, got {getattr(self, name)!r}.")
        if self.phase == "context":
            if self.seq_len <= 0:
                raise KernelCoverageError(f"Context attention needs a positive seq_len, got {self.seq_len!r}.")
            if self.step != 0:
                raise KernelCoverageError(f"Context attention requires step == 0, got {self.step!r}.")
        else:
            if self.seq_len != 1:
                raise KernelCoverageError(
                    f"Generation attention requires seq_len == 1, got {self.seq_len!r}."
                )
            if self.step < 0:
                raise KernelCoverageError(f"Generation step must not be negative, got {self.step!r}.")

    @property
    def op_name(self) -> str:
        """Return the parquet op name this query reads."""
        return f"{self.phase}_attention"

    @property
    def slice_key(self) -> tuple[Any, ...]:
        """Return the categorical key in the attention schema's slice order."""
        return (
            self.num_kv_heads,
            self.head_dim,
            canonical_dtype(self.attn_dtype),
            canonical_dtype(self.kv_cache_dtype),
            self.block_size,
            self.window_size,
        )

    @property
    def coords(self) -> tuple[float, ...]:
        """Return interpolation coordinates in the attention schema's axis order."""
        axis = float(self.seq_len) if self.phase == "context" else float(max(self.step, 1))
        return (float(self.num_heads), float(self.batch_size), axis)


@dataclass(frozen=True)
class ElementwiseQuery:
    """One pointwise kernel call to price, identified by kind and element count.

    `op_kind` is a slice column, so each kind gets its own utility surface: `qk_norm`
    normalizes over `head_dim` where `rms_norm` normalizes over `hidden_size`, which puts
    them on different points of the bandwidth curve at the same token count.

    Consumed by `ComposedTraining` only. `ComposedRollout` deliberately omits this family:
    its GEMM and attention utilities were fitted against a CUDA-graph-replayed server whose
    elementwise cost is already inside them, and adding this term double-counts it (measured:
    decode 4.84% -> 11.18% MRE). Megatron runs no CUDA graphs on the training path, so the
    cost is genuinely exposed there. See `ops/elementwise.py` for the full derivation.
    """

    op_kind: str
    n_elements: int
    dtype: str = "bfloat16"
    backend: str | None = None

    def __post_init__(self) -> None:
        from .ops.elementwise import ELEMENTWISE_KINDS

        if self.op_kind not in ELEMENTWISE_KINDS:
            known = ", ".join(ELEMENTWISE_KINDS)
            raise KernelCoverageError(
                f"Unknown elementwise op_kind {self.op_kind!r}. Known kinds: {known}."
            )
        if self.n_elements <= 0:
            raise KernelCoverageError(
                f"Elementwise n_elements must be positive, got {self.n_elements!r}."
            )

    @property
    def op_name(self) -> str:
        """Return the parquet op name this query reads."""
        return "elementwise"

    @property
    def slice_key(self) -> tuple[Any, ...]:
        """Return the categorical key in `ELEMENTWISE_OP.slice_columns` order."""
        return (canonical_dtype(self.dtype), self.op_kind)

    @property
    def coords(self) -> tuple[float, ...]:
        """Return interpolation coordinates in `ELEMENTWISE_OP.axis_columns` order."""
        return (float(self.n_elements),)


@dataclass(frozen=True)
class TrainingAttentionQuery:
    """One THD-packed attention call to price, in one direction.

    Separate from `AttentionQuery` because it describes a different kernel on a different
    layout. A packed training microbatch is one row with `cu_seqlens`, so there is no batch
    dimension to vary, and the measured table is TE's FlashAttention rather than vLLM's
    prefill kernel. Substituting the rollout table understated the forward by 1.59-1.84x.

    `direction` selects the op rather than being a slice column: forward and backward have
    different rooflines (the backward computes three gradients), and a slice would imply
    they interpolate onto one surface.
    """

    direction: str
    num_heads: int
    total_tokens: int
    num_kv_heads: int
    head_dim: int
    attn_dtype: str = "bfloat16"
    attn_mask_type: str = "padding_causal"
    backend: str | None = _TRAINING_BACKEND

    def __post_init__(self) -> None:
        from .ops.attention_training import TRAINING_MASK_TYPES

        if self.direction not in ("forward", "backward"):
            raise KernelCoverageError(
                f"Training attention direction must be 'forward' or 'backward', got "
                f"{self.direction!r}."
            )
        for name in ("num_heads", "total_tokens", "num_kv_heads", "head_dim"):
            if getattr(self, name) <= 0:
                raise KernelCoverageError(
                    f"Training attention {name} must be positive, got {getattr(self, name)!r}."
                )
        if self.attn_mask_type not in TRAINING_MASK_TYPES:
            known = ", ".join(TRAINING_MASK_TYPES)
            raise KernelCoverageError(
                f"Unknown training attention mask type {self.attn_mask_type!r}. Known "
                f"types: {known}. Transformer Engine rejects plain 'causal' for THD."
            )

    @property
    def op_name(self) -> str:
        """Return the parquet op name this query reads."""
        return f"training_attention_{self.direction}"

    @property
    def slice_key(self) -> tuple[Any, ...]:
        """Return the categorical key in the training attention slice order."""
        return (
            self.num_kv_heads,
            self.head_dim,
            canonical_dtype(self.attn_dtype),
            self.attn_mask_type,
        )

    @property
    def coords(self) -> tuple[float, ...]:
        """Return interpolation coordinates in the training attention axis order."""
        return (float(self.num_heads), float(self.total_tokens))


@dataclass(frozen=True)
class OptimizerQuery:
    """One fused optimizer step to price, over one rank's parameter shard.

    The only query in this module with no token or shape axis: an Adam step's cost is
    independent of how many tokens produced the gradients. `param_count` is the shard one
    rank updates, which is `total_parameters / (tp * pp * dp)` under a distributed
    optimizer.

    `backend` defaults to `megatron` rather than to the database's `vllm` default, because
    the measured kernel is TE's `FusedAdam` and vLLM has no optimizer at all. A `vllm`-keyed
    optimizer table would describe nothing.
    """

    param_count: int
    optimizer: str = "adam"
    state_kind: str = "master_fp32"
    backend: str | None = _OPTIMIZER_BACKEND

    def __post_init__(self) -> None:
        from .ops.optimizer import OPTIMIZER_STATE_KINDS

        if self.param_count <= 0:
            raise KernelCoverageError(
                f"Optimizer param_count must be positive, got {self.param_count!r}."
            )
        if self.state_kind not in OPTIMIZER_STATE_KINDS:
            known = ", ".join(OPTIMIZER_STATE_KINDS)
            raise KernelCoverageError(
                f"Unknown optimizer state_kind {self.state_kind!r}. Known kinds: {known}."
            )

    @property
    def op_name(self) -> str:
        """Return the parquet op name this query reads."""
        return "optimizer_step"

    @property
    def slice_key(self) -> tuple[Any, ...]:
        """Return the categorical key in `OPTIMIZER_STEP_OP.slice_columns` order."""
        return (self.optimizer, self.state_kind)

    @property
    def coords(self) -> tuple[float, ...]:
        """Return interpolation coordinates in `OPTIMIZER_STEP_OP.axis_columns` order."""
        return (float(self.param_count),)


@dataclass(frozen=True)
class CommQuery:
    """One collective to price, identified by op and by topology.

    `nodes` has no default. A cross-node collective priced against an intra-node table is
    optimistic by roughly the NVLink-to-NIC ratio (9x on H20 here), and a default would
    make that the silent outcome of forgetting the field. The caller states the topology it
    means, and `model_ops.comm_queries` passes `nodes=1` explicitly for the same reason.
    """

    op: str
    world_size: int
    nodes: int
    message_bytes: int
    dtype: str = "bfloat16"
    backend: str | None = _COLLECTIVE_BACKEND

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise KernelCoverageError(f"Comm world_size must be positive, got {self.world_size!r}.")
        if self.nodes <= 0:
            raise KernelCoverageError(f"Comm nodes must be positive, got {self.nodes!r}.")
        if self.nodes > self.world_size:
            raise KernelCoverageError(
                f"Comm nodes {self.nodes!r} exceeds world_size {self.world_size!r}, which "
                f"describes no rank placement."
            )
        if self.message_bytes <= 0:
            raise KernelCoverageError(
                f"Comm message_bytes must be positive, got {self.message_bytes!r}."
            )
        # Fail here rather than at lookup: a typo'd op name would otherwise surface as a
        # missing-table coverage error, which reads like an unswept topology.
        if op_for(self.op).family != "collective":
            raise KernelCoverageError(
                f"Comm op {self.op!r} is not a collective; it belongs to family "
                f"{op_for(self.op).family!r}."
            )

    @property
    def op_name(self) -> str:
        """Return the parquet op name this query reads."""
        return self.op

    @property
    def slice_key(self) -> tuple[Any, ...]:
        """Return the categorical key in the collective ops' slice order."""
        return (canonical_dtype(self.dtype), self.world_size, self.nodes)

    @property
    def coords(self) -> tuple[float, ...]:
        """Return interpolation coordinates in the all-reduce op's axis order."""
        return (float(self.message_bytes),)


# Union of all concrete query types; used in signatures that must not branch on the type.
_AnyQuery = (
    GemmQuery
    | AttentionQuery
    | TrainingAttentionQuery
    | ElementwiseQuery
    | OptimizerQuery
    | CommQuery
)


class KernelDatabase:
    """Lazily loaded perf tables answering shape queries in seconds."""

    def __init__(
        self,
        *,
        root: Path,
        gpu: str,
        gpu_spec: GpuSpec,
        measurement_spec: GpuSpec,
        default_backend: str = _DEFAULT_BACKEND,
        versions: dict[str, str] | None = None,
        utility_surface: str = "shepard",
    ) -> None:
        self._root = root
        self._gpu = gpu
        self._gpu_spec = gpu_spec
        # Spec for the hardware the measurements were collected on; used to compute
        # utility so grids remain independent of the target GPU.
        self._measurement_spec = measurement_spec
        self._default_backend = default_backend
        self._versions = dict(versions or {})
        self._utility_surface = utility_surface
        self._tables: dict[tuple[str, str], KernelTable] = {}
        # Cache sol_fn closures so KernelTable.grid() memoization can match on identity.
        # Keyed on the table object itself (not id(table)) so the cache holds a strong
        # reference: same pattern as KernelTable.grid() keying on the callable object.
        self._sol_fns: dict[tuple[KernelTable, tuple[Any, ...]], Callable[[tuple[float, ...]], float]] = {}
        # Resolved floors keyed by `(family, backend)`. Keying on the backend rather than the
        # version directory matters: resolving the version scans that directory, so a
        # path-keyed cache still paid a listdir per query. The family is part of the key
        # because each family holds its own floors.json. The fallback warning is emitted once
        # per pair, not once per query.
        self._floors_by_backend: dict[tuple[str, str], dict[str, float]] = {}

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        gpu: str,
        gpu_spec: GpuSpec | None = None,
        default_backend: str = _DEFAULT_BACKEND,
        versions: dict[str, str] | None = None,
        utility_surface: str = "shepard",
    ) -> KernelDatabase:
        """
        Open the perf-table tree under `root` for one GPU.

        Args:
            root (str | Path): Directory holding `{gpu}/{family}/{backend}/{version}/`.
            gpu (str): GPU slug or device name; also selects the bundled spec.
            gpu_spec (GpuSpec | None): Target hardware for SOL. Defaults to the bundled
                spec for `gpu`. Pass explicitly to price on hardware other than the one
                measured.
            default_backend (str): Backend used when a query does not override it.
            versions (dict[str, str] | None): Pin `{backend: version}` when a backend
                directory holds more than one version.
            utility_surface (str): Interpolation strategy name; see `SURFACE_REGISTRY`.

        Returns:
            KernelDatabase: Ready for queries; no parquet is read until one arrives.
        """
        slug = gpu_slug(gpu)
        measurement_spec = load_gpu_spec(slug)
        return cls(
            root=Path(root),
            gpu=slug,
            gpu_spec=gpu_spec or measurement_spec,
            measurement_spec=measurement_spec,
            default_backend=default_backend,
            versions=versions,
            utility_surface=utility_surface,
        )

    @property
    def gpu_spec(self) -> GpuSpec:
        """Return the hardware spec every SOL is evaluated against."""
        return self._gpu_spec

    def backends(self, op_name: str) -> tuple[str, ...]:
        """
        List backend directories holding measurements for one op.

        Args:
            op_name (str): One of `gemm`, `context_attention`, `generation_attention`.

        Returns:
            tuple[str, ...]: Sorted backend names, empty when the family is absent.
        """
        family = op_for(op_name).family
        family_dir = self._root / self._gpu / family
        if not family_dir.is_dir():
            return ()
        return tuple(sorted(item.name for item in family_dir.iterdir() if item.is_dir()))

    def query_gemm(self, query: GemmQuery) -> float:
        """
        Return the predicted latency in seconds of one GEMM.

        Args:
            query (GemmQuery): Per-rank shape. Backward dgrad and wgrad reuse this same
                table with permuted axes.

        Returns:
            float: `SOL(query) / interpolated utility`, in seconds.

        Raises:
            KernelCoverageError: If no table or slice covers the query.
        """
        return self._price(query)

    def query_attention(self, query: AttentionQuery) -> float:
        """
        Return the predicted latency in seconds of one attention call.

        Args:
            query (AttentionQuery): Per-rank shape.

        Returns:
            float: `SOL(query) / interpolated utility`, in seconds.

        Raises:
            KernelCoverageError: If no table or slice covers the query.
        """
        return self._price(query)

    def query_elementwise(self, query: ElementwiseQuery) -> float:
        """
        Return the predicted latency in seconds of one pointwise kernel call.

        Args:
            query (ElementwiseQuery): Kind and element count. The element count is the
                number of elements *read*; the SOL assumes an equal write.

        Returns:
            float: `SOL(query) / interpolated utility`, in seconds.

        Raises:
            KernelCoverageError: If no table or slice covers the query.
        """
        return self._price(query)

    def query_training_attention(self, query: TrainingAttentionQuery) -> float:
        """
        Return the predicted latency in seconds of one THD attention call.

        Args:
            query (TrainingAttentionQuery): Packed shape and direction.

        Returns:
            float: `SOL(query) / interpolated utility`, in seconds.

        Raises:
            KernelCoverageError: If no table or slice covers the query.
        """
        return self._price(query)

    def query_optimizer(self, query: OptimizerQuery) -> float:
        """
        Return the predicted latency in seconds of one fused optimizer step.

        Args:
            query (OptimizerQuery): The parameter shard one rank updates.

        Returns:
            float: `SOL(query) / interpolated utility`, in seconds.

        Raises:
            KernelCoverageError: If no table or slice covers the query.
        """
        return self._price(query)

    def utility(self, query: _AnyQuery) -> float:
        """
        Return the interpolated utility at one query, for diagnostics.

        Args:
            query (GemmQuery | AttentionQuery | ElementwiseQuery | OptimizerQuery | CommQuery):
                Any query this database can price.

        Returns:
            float: Dimensionless efficiency, roughly in (0, 1]. A value above ~1.2 means
                the SOL formula is missing work rather than that the GPU beat its spec.
        """
        return self._grid(query).utility(query.coords)

    def query_comm(self, query: CommQuery) -> float:
        """
        Return the predicted latency in seconds of one collective.

        Args:
            query (CommQuery): Collective op, topology (`world_size`, `nodes`), and
                per-rank message size.

        Returns:
            float: `SOL(query) / interpolated utility`, in seconds.

        Raises:
            KernelCoverageError: If no table or slice covers the query. Both `world_size`
                and `nodes` are slice columns, so an unswept topology raises here rather
                than being interpolated from a different one.
        """
        return self._price(query)

    def _price(self, query: _AnyQuery) -> float:
        """Return `SOL(query) / interpolated utility` in seconds, for any op.

        One implementation rather than one per family, because the backend must bind
        identically on both sides of the division. `_grid` bakes `query.backend or
        self._default_backend` into the closure it derives utility from, so a numerator that
        resolved the backend differently would take its floor from one backend's floors.json
        while the denominator's utility came from another: a wrong latency that still looks
        self-consistent from either side.
        """
        resolved_backend = query.backend or self._default_backend
        grid = self._grid(query)
        return self._sol(self._gpu_spec, query, query.coords, resolved_backend) / grid.utility(query.coords)

    def _sol(
        self,
        spec: GpuSpec,
        query: _AnyQuery,
        coords: tuple[float, ...],
        backend: str | None = None,
    ) -> float:
        """Return the speed-of-light latency of one query, in seconds."""
        # The op owns the formula. Before this, database and holdout each held their own copy
        # of this binding, and drift between them would have changed utility on one side of the
        # holdout round trip: silently, since both sides would still look self-consistent.
        op = op_for(query.op_name)
        floor_s = 0.0
        if op.fits_floor:
            resolved_backend = backend or self._default_backend
            floor_s = self._floors_for_op(op, resolved_backend).get(op.name, 0.0)
        return op.sol(spec, slice_key=query.slice_key, coords=coords, floor_s=floor_s).latency_s

    def _floors_for_op(self, op: Any, backend: str) -> dict[str, float]:
        """Return the per-op floor map for one op's family and backend.

        Cached per `(family, backend)` pair, not per version directory. Resolving the version
        scans the backend directory, so keying the cache on the resolved path still paid a
        listdir on every query: that made one attention query 2337 us where the interpolation
        inside it is 17 us. The family is part of the key because floors.json lives inside the
        family's own version directory, so reading the attention directory for an all-reduce
        would silently apply a zero floor. Falls back to `FALLBACK_FLOORS_S` (with a warning)
        when floors.json is absent.
        """
        cache_key = (op.family, backend)
        cached = self._floors_by_backend.get(cache_key)
        if cached is not None:
            return cached
        resolved = self._resolve_floors(op, backend)
        self._floors_by_backend[cache_key] = resolved
        return resolved

    def _resolve_floors(self, op: Any, backend: str) -> dict[str, float]:
        """Read floors.json for one op's family and backend, falling back to the hardcoded map."""
        fallback = FALLBACK_FLOORS_S.get(backend, {})
        backend_dir = self._root / self._gpu / op.family / backend
        if not backend_dir.is_dir():
            return fallback
        try:
            version = self._resolve_version(backend_dir, backend, op.name)
        except KernelCoverageError:
            return fallback
        loaded = _load_floors(backend_dir / version, backend=backend, warn_missing=True)
        return loaded if loaded is not None else fallback

    def _grid(self, query: _AnyQuery) -> UtilitySurface:
        resolved_backend = query.backend or self._default_backend
        table = self._table(query.op_name, query.backend)
        cache_key = (table, query.slice_key)
        sol_fn = self._sol_fns.get(cache_key)
        if sol_fn is None:
            frozen_query = query
            frozen_spec = self._measurement_spec
            frozen_backend = resolved_backend

            def sol_fn(coords: tuple[float, ...]) -> float:
                return self._sol(frozen_spec, frozen_query, coords, frozen_backend)

            self._sol_fns[cache_key] = sol_fn
        return table.grid(query.slice_key, sol_fn, surface=self._utility_surface)

    def _table(self, op_name: str, backend: str | None) -> KernelTable:
        resolved = backend or self._default_backend
        cache_key = (op_name, resolved)
        cached = self._tables.get(cache_key)
        if cached is not None:
            return cached
        op = op_for(op_name)
        backend_dir = self._root / self._gpu / op.family / resolved
        if not backend_dir.is_dir():
            measured = ", ".join(self.backends(op_name)) or "none"
            raise KernelCoverageError(
                f"No {op_name} perf tables for backend {resolved!r} under {backend_dir}. "
                f"Measured backends: {measured}. "
                f"Run dryrun-profile kernels run --family {op.family} to create them."
            )
        version = self._resolve_version(backend_dir, resolved, op_name)
        table_path = backend_dir / version / f"{op_name}_perf.parquet"
        table = KernelTable.load(table_path, op)
        for device in table.devices():
            device_slug = gpu_slug(device)
            if device_slug != self._gpu:
                raise KernelCoverageError(
                    f"Perf table {table_path!r} contains rows for device {device!r} "
                    f"(slug {device_slug!r}) but was loaded under directory slug {self._gpu!r}."
                )
        self._tables[cache_key] = table
        return table

    def _resolve_version(self, backend_dir: Path, backend: str, op_name: str) -> str:
        pinned = self._versions.get(backend)
        if pinned is not None:
            return pinned
        candidates = sorted(item.name for item in backend_dir.iterdir() if item.is_dir())
        if not candidates:
            raise KernelCoverageError(
                f"Backend directory {backend_dir} holds no version subdirectory. "
                f"Run dryrun-profile kernels run to create one."
            )
        if len(candidates) > 1:
            raise KernelCoverageError(
                f"Backend {backend!r} has {len(candidates)} versions for {op_name}: "
                f"{', '.join(candidates)}. Pass versions={{{backend!r}: ...}} to pick one."
            )
        return candidates[0]
