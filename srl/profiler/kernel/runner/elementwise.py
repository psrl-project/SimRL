"""
Time vLLM's elementwise kernels: rmsnorm, rope, activation, and qk_norm.

Two traps make this module's shape non-obvious, both measured on H20 rather than assumed.

**The raw fused op must be timed, not the module that wraps it.** Calling `RMSNorm(...)` as a
module costs 60.33 us at 16 tokens against 4.57 us for `vllm._custom_ops.rms_norm`: 13.2x.
The gap is Python dispatch through `CustomOp.__call__`, which the server never pays because it
replays these kernels inside a CUDA graph. Timing the module would inflate the table by an
order of magnitude and the utility surface would absorb the error silently.

**A `CustomOp` needs an active `CompilationConfig(custom_ops=["all"])`.** Without it vLLM logs
`Priority not set for op rms_norm, using native implementation` and dispatches its native
PyTorch fallback, which is a different kernel from the fused one the server runs. That is why
the whole loop runs inside `set_current_vllm_config` rather than per point.

`silu_and_mul` is not exposed on `vllm._custom_ops`, so it is reached as `SiluAndMul().op`
after the module is constructed inside the config context; that attribute is what the module's
own `forward_cuda` calls.

Each point's tensors are allocated inside `_time_point`, so they are released when it returns
and the loop can empty the allocator cache knowing nothing still holds them. The allocator
cache must be emptied between points rather than merely dropping references, because a cached
block from one shape is not reusable for a differently-shaped next one.

The element counts come from `grid/elementwise.py`, which keeps them checkable against the
same arithmetic `cost/training/training_ops.py` applies on the query side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..grid.elementwise import (
    ElementwiseDims,
    ElementwisePoint,
    elementwise_dims,
    elementwise_sweep_points,
)
from .base import KernelRunner, WriterFactory

_MAX_POSITION = 40960
_ROPE_THETA = 1.0e6
_RMS_EPS = 1.0e-6


class ElementwiseRunner(KernelRunner):
    """Times every elementwise kind across the token grid, inside one vLLM config context."""

    def run(self, *, writer_factory: WriterFactory) -> tuple[Path, ...]:
        """
        Time every elementwise kind across the configured token grid.

        Args:
            writer_factory (WriterFactory): Builds a writer from
                `(framework, version, kernel_source)`.

        Returns:
            tuple[Path, ...]: The single written parquet path.

        Raises:
            ProfileValidationError: If the architecture is missing `hidden_size` or
                `num_attention_heads`.
            ValueError: If the grid names a kind this runner cannot measure.
        """
        import torch
        import vllm
        from vllm.config import CompilationConfig, ModelConfig, VllmConfig, set_current_vllm_config

        from ..utils.timing import torch_dtype

        dtype_name = self.config.measurement.dtype
        dtype = torch_dtype(dtype_name)
        dims = elementwise_dims(self.model)
        points = elementwise_sweep_points(self.model, self.config.sweep("elementwise"))

        # `custom_ops=["all"]` is load-bearing: without it these modules dispatch vLLM's native
        # PyTorch fallback rather than the fused CUDA kernel the server runs.
        vllm_config = VllmConfig(
            model_config=ModelConfig(
                model=str(self.model.name),
                tokenizer=str(self.model.name),
                trust_remote_code=False,
                dtype=dtype_name,
                seed=0,
                max_model_len=_MAX_POSITION,
            ),
            compilation_config=CompilationConfig(custom_ops=["all"]),
        )

        rows: list[dict[str, object]] = []
        with set_current_vllm_config(vllm_config):
            from vllm.model_executor.layers.activation import SiluAndMul
            from vllm.model_executor.layers.layernorm import RMSNorm
            from vllm.model_executor.layers.rotary_embedding import get_rope

            modules: dict[str, Any] = {
                "hidden_norm": RMSNorm(dims.hidden_size, dtype=dtype).cuda(),
                "head_norm": RMSNorm(dims.head_dim, dtype=dtype).cuda(),
                "silu": SiluAndMul().cuda(),
                "rope": get_rope(
                    head_size=dims.head_dim,
                    max_position=_MAX_POSITION,
                    rope_parameters={"rope_type": "default", "rope_theta": _ROPE_THETA},
                    dtype=dtype,
                ).cuda(),
            }
            for point in points:
                latency_s = self._time_point(point, dims=dims, dtype=dtype, modules=modules)
                # Safe here and not inside `_time_point`: that frame's tensors are gone by the
                # time it returns, so there is nothing left holding the blocks being released.
                torch.cuda.empty_cache()
                rows.append(
                    self.row(
                        "elementwise",
                        dtype=dtype_name,
                        op_kind=point.op_kind,
                        n_elements=point.n_elements,
                        latency_s=latency_s,
                    )
                )

        writer = writer_factory("vllm", str(vllm.__version__), "vllm._custom_ops")
        return (writer.write("elementwise", rows),)

    def _time_point(
        self,
        point: ElementwisePoint,
        *,
        dims: ElementwiseDims,
        dtype: Any,
        modules: dict[str, Any],
    ) -> float:
        """
        Allocate, time, and release one `(op_kind, token_count)` point.

        Tensors are function-locals so they die with this frame; the caller empties the
        allocator cache afterwards.

        Args:
            point (ElementwisePoint): The kind and token count to measure.
            dims (ElementwiseDims): Per-rank dimensions the tensors are shaped from.
            dtype (Any): The `torch.dtype` to allocate at.
            modules (dict[str, Any]): The vLLM modules built inside the config context, whose
                weights and rope cache the raw ops need.

        Returns:
            float: Mean per-call latency in seconds.

        Raises:
            ValueError: If `point.op_kind` has no measurement here. Reachable only by adding a
                kind to `ELEMENTWISE_KINDS` without adding its measurement, which would
                otherwise write a table silently missing that kind.
        """
        import torch
        import vllm._custom_ops as ops

        tokens = point.token_count
        kind = point.op_kind
        # Tensors below are bound into each lambda as default arguments rather than captured, so
        # the callable holds its own reference rather than a cell in this frame.
        if kind == "rms_norm":
            source = torch.randn(tokens, dims.hidden_size, dtype=dtype, device="cuda")
            target = torch.empty_like(source)
            weight = modules["hidden_norm"].weight.data
            return self.time(
                lambda o=target, i=source, w=weight: ops.rms_norm(o, i, w, _RMS_EPS)
            )
        if kind == "tensor_plumbing":
            # Stands for the dtype conversions, contiguity fixes and concatenations a real
            # training step issues around the fused modules. Timed as a bf16 copy of a
            # hidden-size tensor, the modal member of the traced population (`aten::copy_` 414
            # calls, `aten::to` 246, `aten::cat` 193, `aten::mul` 289).
            #
            # One representative rather than five measurements, because the cost is 85% launch
            # floor at these sizes: what separates the members is how many fire, not their
            # bandwidth curve. `TrainingOps` carries the count.
            source = torch.randn(tokens, dims.hidden_size, dtype=dtype, device="cuda")
            target = torch.empty_like(source)
            return self.time(lambda o=target, i=source: o.copy_(i))
        if kind == "rotary_embedding":
            rope = modules["rope"]
            positions = torch.arange(tokens, device="cuda")
            query = torch.randn(
                tokens, dims.num_heads * dims.head_dim, dtype=dtype, device="cuda"
            )
            key = torch.randn(
                tokens, dims.num_kv_heads * dims.head_dim, dtype=dtype, device="cuda"
            )
            return self.time(
                lambda p=positions, q=query, k=key, d=dims.head_dim: ops.rotary_embedding(
                    p, q, k, d, rope.cos_sin_cache, rope.is_neox_style
                )
            )
        if kind == "silu_and_mul":
            silu = modules["silu"]
            gate_up = torch.randn(tokens, 2 * dims.intermediate_size, dtype=dtype, device="cuda")
            gate_out = torch.empty(tokens, dims.intermediate_size, dtype=dtype, device="cuda")
            return self.time(lambda o=gate_out, i=gate_up: silu.op(o, i))
        if kind == "qk_norm":
            # qk_norm normalizes each head independently, so its rows are token*heads long and
            # it normalizes over head_dim rather than hidden_size: a different point on the
            # bandwidth curve at the same token count, which is why it is its own kind.
            weight = modules["head_norm"].weight.data
            source = torch.randn(
                tokens * dims.num_heads, dims.head_dim, dtype=dtype, device="cuda"
            )
            target = torch.empty_like(source)
            return self.time(
                lambda o=target, i=source, w=weight: ops.rms_norm(o, i, w, _RMS_EPS)
            )
        raise ValueError(
            f"No measurement is implemented for elementwise kind {kind!r}. Add one here when "
            f"adding it to ELEMENTWISE_KINDS."
        )
