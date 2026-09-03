"""
Time vLLM's elementwise kernels: rmsnorm, rope, activation, and qk_norm.

Two traps make this module's shape non-obvious, both measured on H20 rather than assumed.

**The raw fused op must be timed, not the module that wraps it.** Calling `RMSNorm(...)` as a
module costs 60.33 us at 16 tokens against 4.57 us for `vllm._custom_ops.rms_norm` -- 13.2x.
The gap is Python dispatch through `CustomOp.__call__`, which the server never pays because it
replays these kernels inside a CUDA graph. Timing the module would inflate the table by an
order of magnitude and the utility surface would absorb the error silently.

**A `CustomOp` needs an active `CompilationConfig(custom_ops=["all"])`.** Without it vLLM logs
`Priority not set for op rms_norm, using native implementation` and dispatches its native
PyTorch fallback, which is a different kernel from the fused one the server runs.

`silu_and_mul` is not exposed on `vllm._custom_ops`, so it is reached as `SiluAndMul().op`
after the module is constructed inside the config context; that attribute is what the module's
own `forward_cuda` calls.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..schema import ModelDescriptor, ProfileValidationError
from ._timing import time_cuda_call, torch_dtype
from .config import KernelProfileConfig, model_head_dim, model_intermediate_size
from .measurements import PerfTableWriter

_MAX_POSITION = 40960
_ROPE_THETA = 1.0e6


def profile_elementwise(
    model: ModelDescriptor,
    config: KernelProfileConfig,
    *,
    writer_factory: Callable[[str, str, str], PerfTableWriter],
) -> tuple[Path, ...]:
    """
    Time every elementwise kind across the configured token grid.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        config (KernelProfileConfig): Sweep grid, dtype, and iteration counts.
        writer_factory (Callable[[str, str, str], PerfTableWriter]): Builds a writer from
            `(framework, version, kernel_source)`.

    Returns:
        tuple[Path, ...]: The single written parquet path.

    Raises:
        ProfileValidationError: If the architecture is missing `hidden_size`.
    """
    import torch
    import vllm
    import vllm._custom_ops as ops
    from vllm.config import CompilationConfig, ModelConfig, VllmConfig, set_current_vllm_config

    if model.hidden_size is None:
        raise ProfileValidationError("Model is missing hidden_size.")
    if model.num_attention_heads is None:
        raise ProfileValidationError("Model is missing num_attention_heads.")

    sweep = config.sweep("elementwise")
    measurement = config.measurement
    dtype_name = measurement.dtype
    dtype = torch_dtype(dtype_name)
    hidden = int(model.hidden_size)
    head_dim = model_head_dim(model)
    intermediate = model_intermediate_size(model)
    heads = int(model.num_attention_heads)
    kv_heads = int(model.num_query_groups or heads)

    # `custom_ops=["all"]` is load-bearing: without it these modules dispatch vLLM's native
    # PyTorch fallback rather than the fused CUDA kernel the server runs.
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(model.name),
            tokenizer=str(model.name),
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

        hidden_norm = RMSNorm(hidden, dtype=dtype).cuda()
        head_norm = RMSNorm(head_dim, dtype=dtype).cuda()
        silu = SiluAndMul().cuda()
        rope = get_rope(
            head_size=head_dim,
            max_position=_MAX_POSITION,
            rope_parameters={"rope_type": "default", "rope_theta": _ROPE_THETA},
            dtype=dtype,
        ).cuda()

        for tokens in sweep.token_counts:
            positions = torch.arange(tokens, device="cuda")

            hidden_in = torch.randn(tokens, hidden, dtype=dtype, device="cuda")
            hidden_out = torch.empty_like(hidden_in)
            rows.append(
                _row(
                    dtype_name,
                    "rms_norm",
                    tokens * hidden,
                    time_cuda_call(
                        # Bound as default arguments rather than captured, so the lambda holds
                        # its own reference and the `del` below can free the tensors.
                        lambda o=hidden_out, i=hidden_in, w=hidden_norm.weight.data: ops.rms_norm(
                            o, i, w, 1e-6
                        ),
                        warmup_iterations=measurement.warmup_iterations,
                        active_iterations=measurement.active_iterations,
                    ),
                )
            )

            query = torch.randn(tokens, heads * head_dim, dtype=dtype, device="cuda")
            key = torch.randn(tokens, kv_heads * head_dim, dtype=dtype, device="cuda")
            rows.append(
                _row(
                    dtype_name,
                    "rotary_embedding",
                    tokens * (heads + kv_heads) * head_dim,
                    time_cuda_call(
                        lambda p=positions, q=query, k=key: ops.rotary_embedding(
                            p, q, k, head_dim, rope.cos_sin_cache, rope.is_neox_style
                        ),
                        warmup_iterations=measurement.warmup_iterations,
                        active_iterations=measurement.active_iterations,
                    ),
                )
            )

            gate_up = torch.randn(tokens, 2 * intermediate, dtype=dtype, device="cuda")
            gate_out = torch.empty(tokens, intermediate, dtype=dtype, device="cuda")
            rows.append(
                _row(
                    dtype_name,
                    "silu_and_mul",
                    tokens * intermediate,
                    time_cuda_call(
                        lambda o=gate_out, i=gate_up: silu.op(o, i),
                        warmup_iterations=measurement.warmup_iterations,
                        active_iterations=measurement.active_iterations,
                    ),
                )
            )

            # qk_norm normalizes each head independently, so its rows are token*heads long.
            head_in = torch.randn(tokens * heads, head_dim, dtype=dtype, device="cuda")
            head_out = torch.empty_like(head_in)
            rows.append(
                _row(
                    dtype_name,
                    "qk_norm",
                    tokens * heads * head_dim,
                    time_cuda_call(
                        lambda o=head_out, i=head_in, w=head_norm.weight.data: ops.rms_norm(
                            o, i, w, 1e-6
                        ),
                        warmup_iterations=measurement.warmup_iterations,
                        active_iterations=measurement.active_iterations,
                    ),
                )
            )

            del positions, hidden_in, hidden_out, query, key, gate_up, gate_out, head_in, head_out
            torch.cuda.empty_cache()

    writer = writer_factory("vllm", str(vllm.__version__), "vllm._custom_ops")
    return (writer.write("elementwise", rows),)


def _row(dtype_name: str, op_kind: str, n_elements: int, latency_s: float) -> dict[str, object]:
    """Build one perf-table row for an elementwise measurement."""
    return {
        "dtype": dtype_name,
        "op_kind": op_kind,
        "n_elements": int(n_elements),
        "latency_s": latency_s,
    }
