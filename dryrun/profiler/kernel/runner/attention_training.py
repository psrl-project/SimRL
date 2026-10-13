"""
Time Transformer Engine's THD-packed attention, forward and backward.

Four traps, all of them measured on H20 rather than assumed, and each one capable of
producing a plausible table describing the wrong kernel.

**A `.post1` flash-attn wheel silently disables the fused path.** TE gates flash attention
on `required <= version <= max` read from package metadata, and PEP 440 orders
`2.8.3.post1` above `2.8.3`, so a wheel matching the upper bound exactly is rejected as too
new. TE then falls back to `UnfusedDotProductAttention` and logs it only at
`NVTE_DEBUG_LEVEL=1`. The first run of this sweep measured that fallback.
`_normalize_flash_attn_version` is the existing fix and must run **before** TE is imported,
which is why it is called at the top of `run` rather than at module scope.

**THD requires `padding_causal`.** `DotProductAttention.forward` asserts
`"padding" in attn_mask_type` when `qkv_format="thd"`, so plain `causal` raises. A backend
probe built with `causal` reports `FlashAttention=False, FusedAttention=False` and looks
like "this hardware has no fused attention", which is how the design doc came to assert
cuDNN. The real answer for THD is FlashAttention 2.8.3.

**The backend is FlashAttention, not cuDNN.** TE disables flash on Hopper+ for its BSHD
path (`utils.py` prefers FusedAttention there), and that is what the design doc recorded.
It does not hold for THD: TE reports `Running with FlashAttention backend (version 2.8.3)`
and disables FusedAttention with `no backend supports the provided input`. The row's
`kernel_source` records which one actually ran, so a future table cannot be read as the
other.

**The backward must be isolated from the forward it needs.** `backward()` requires a
forward to have run, so timing `forward + backward` and subtracting a separately-timed
forward is the only way to get the backward alone. The subtraction is done here rather than
written as a combined row, because a consumer summing a `fwd+bwd` row and a `fwd` row would
double-count the forward.

The grid, including the GQA ratio that derives each point's KV head count, lives in
`grid/attention_training.py` so it can be checked without a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...schema import ProfileValidationError
from ..grid.attention_training import TrainingAttentionPoint, training_attention_points
from .base import KernelRunner, WriterFactory


class TrainingAttentionRunner(KernelRunner):
    """Times THD attention forward and backward against one set of packed tensors per point."""

    def run(self, *, writer_factory: WriterFactory) -> tuple[Path, ...]:
        """
        Time THD attention forward and backward across the configured grid.

        Args:
            writer_factory (WriterFactory): Builds a writer from
                `(framework, version, kernel_source)`.

        Returns:
            tuple[Path, ...]: The forward and backward parquet paths.

        Raises:
            ProfileValidationError: If the architecture is incomplete, if the forward and
                combined measurements are not separable, or if Transformer Engine selects its
                unfused fallback, which would measure a kernel no training run uses.
        """
        # Before importing transformer_engine: the version gate is read at import time.
        from ...adapters.megatron.launch import _normalize_flash_attn_version

        _normalize_flash_attn_version()

        import torch
        import transformer_engine
        from transformer_engine.pytorch.attention import DotProductAttention

        from ..utils.timing import torch_dtype

        dtype_name = self.config.measurement.dtype
        dtype = torch_dtype(dtype_name)
        points = training_attention_points(
            self.model, self.config.sweep("training_attention_forward")
        )

        forward_rows: list[dict[str, object]] = []
        backward_rows: list[dict[str, object]] = []
        kernel_sources: set[str] = set()

        # One module per `(mask_type, num_heads)` pair, reused across every token count:
        # constructing it is the expensive part, and the grid is ordered mask-major then
        # head-major then token-major to make that reuse contiguous.
        attention: Any = None
        module_key: tuple[str, int] | None = None
        for point in points:
            key = (point.mask_type, point.num_heads)
            if key != module_key:
                attention = None
                torch.cuda.empty_cache()
                attention = DotProductAttention(
                    num_attention_heads=point.num_heads,
                    kv_channels=point.head_dim,
                    num_gqa_groups=point.num_kv_heads,
                    attention_dropout=0.0,
                    qkv_format="thd",
                    attn_mask_type=point.mask_type,
                ).cuda()
                module_key = key

            forward_s, backward_s = self._time_point(
                point, attention=attention, dtype=dtype, kernel_sources=kernel_sources
            )
            torch.cuda.empty_cache()
            slice_fields = {
                "num_key_value_heads": point.num_kv_heads,
                "head_dim": point.head_dim,
                "attn_dtype": dtype_name,
                "attn_mask_type": point.mask_type,
                "num_heads": point.num_heads,
                "total_tokens": point.total_tokens,
            }
            forward_rows.append(
                self.row("training_attention_forward", latency_s=forward_s, **slice_fields)
            )
            backward_rows.append(
                self.row("training_attention_backward", latency_s=backward_s, **slice_fields)
            )

        attention = None
        torch.cuda.empty_cache()

        _require_fused_backend(kernel_sources)
        writer = writer_factory(
            "megatron", str(transformer_engine.__version__), sorted(kernel_sources)[0]
        )
        return (
            writer.write("training_attention_forward", forward_rows),
            writer.write("training_attention_backward", backward_rows),
        )

    def _time_point(
        self,
        point: TrainingAttentionPoint,
        *,
        attention: Any,
        dtype: Any,
        kernel_sources: set[str],
    ) -> tuple[float, float]:
        """
        Time one packed point's forward and isolate its backward.

        The q/k/v tensors are function-locals so they die with this frame, before the caller
        empties the allocator cache for the next and possibly larger pack.

        Args:
            point (TrainingAttentionPoint): Head counts and pack length to measure.
            attention (Any): The `DotProductAttention` module for this point's head count.
            dtype (Any): The `torch.dtype` to allocate at.
            kernel_sources (set[str]): Accumulates the backend TE chose, checked after the
                sweep so a grid spanning two backends is refused rather than written as one
                table.

        Returns:
            tuple[float, float]: `(forward_s, backward_s)`, the backward already isolated.

        Raises:
            ProfileValidationError: If the backward comes out non-positive, meaning the two
                measurements are not separable at this iteration count.
        """
        import torch

        total_tokens = point.total_tokens
        cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device="cuda")
        query = torch.randn(
            total_tokens, point.num_heads, point.head_dim,
            dtype=dtype, device="cuda", requires_grad=True,
        )
        key = torch.randn(
            total_tokens, point.num_kv_heads, point.head_dim,
            dtype=dtype, device="cuda", requires_grad=True,
        )
        value = torch.randn(
            total_tokens, point.num_kv_heads, point.head_dim,
            dtype=dtype, device="cuda", requires_grad=True,
        )
        call_kwargs = dict(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=total_tokens,
            max_seqlen_kv=total_tokens,
            attn_mask_type=point.mask_type,
        )
        output = attention(query, key, value, **call_kwargs)
        kernel_sources.add(_selected_backend())
        grad_output = torch.randn_like(output)

        forward_s = self.time(
            lambda a=attention, q=query, k=key, v=value, kw=call_kwargs: a(q, k, v, **kw)
        )

        def forward_and_backward(
            a=attention, q=query, k=key, v=value, kw=call_kwargs, g=grad_output
        ) -> None:
            for tensor in (q, k, v):
                tensor.grad = None
            a(q, k, v, **kw).backward(g)

        combined_s = self.time(forward_and_backward)
        # The backward alone. A combined row would be double-counted by any consumer that also
        # charges the forward, which `ComposedTraining` does twice under full recompute.
        backward_s = combined_s - forward_s
        if backward_s <= 0.0:
            raise ProfileValidationError(
                f"Backward time came out non-positive ({backward_s:.6g}s) at "
                f"{total_tokens} tokens and {point.num_heads} heads, so the forward and "
                f"combined measurements are not separable. Raise "
                f"measurement.active_iterations."
            )
        return forward_s, backward_s


def _selected_backend() -> str:
    """Return the backend Transformer Engine chose for the call just made.

    Read from TE's own module-level `_attention_backends` dict, which is where the
    dispatcher records its decision. Two alternatives were rejected: the `DotProductAttention`
    instance holds `flash_attention`, `fused_attention` and `unfused_attention` submodules
    unconditionally, so their presence proves nothing about which one ran; and re-deriving
    the choice through `get_attention_backend` means reconstructing its parameters by hand,
    which is the error that produced the "cuDNN on Hopper+" claim in the first place (a probe
    built with `causal` rather than `padding_causal` reports every backend unavailable).

    Reading a private dict is the right trade here: it is the single value TE itself acts on,
    so it cannot disagree with the call that was made, and a rename would surface as the
    `unknown` string rather than as a silently wrong label.
    """
    from transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention import (
        _attention_backends,
    )

    if _attention_backends.get("use_flash_attention"):
        return "FlashAttention"
    if _attention_backends.get("use_fused_attention"):
        return "FusedAttention"
    if _attention_backends.get("use_unfused_attention"):
        return "UnfusedDotProductAttention"
    return "unknown"


def _require_fused_backend(kernel_sources: set[str]) -> None:
    """Refuse a table measured on the unfused fallback.

    The fallback materializes the full attention matrix and is several times slower, so a
    table measured on it would be internally consistent and describe a kernel no training
    run executes. It is reached silently when the flash-attn version gate rejects the
    installed wheel, so the check is a gate rather than a warning.
    """
    if not kernel_sources:
        raise ProfileValidationError(
            "No attention call was made, so no backend could be determined. Check that "
            "attention_training.total_tokens and num_heads are non-empty."
        )
    if len(kernel_sources) > 1:
        raise ProfileValidationError(
            f"Transformer Engine selected more than one backend across the sweep "
            f"({sorted(kernel_sources)}), so one table would describe two kernels. Narrow "
            f"the grid so a single backend covers it."
        )
    selected = next(iter(kernel_sources))
    if "Unfused" in selected:
        raise ProfileValidationError(
            "Transformer Engine selected UnfusedDotProductAttention, which materializes "
            "the full attention matrix and is not what a training step runs. The usual "
            "cause is a post-release flash-attn wheel (PEP 440 orders 2.8.3.post1 above "
            "TE's 2.8.3 upper bound, so it is rejected as too new) - verify that "
            "_normalize_flash_attn_version ran before transformer_engine was imported. "
            "Diagnose with NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1."
        )
