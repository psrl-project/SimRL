"""
Routed mixture-of-experts: the grouped gate_up+down GEMM vLLM's `fused_experts()` dispatches.

Replaces pricing `expert_gate_up`/`expert_down` against the generic dense-GEMM table plus an
analytical weight-read correction (`ModelOps.expert_weight_bytes`, now retired). That correction
assumed balanced routing and priced the excess at the device's idealised peak bandwidth; this
table measures the real Triton kernel directly, so compute AND the activated-expert weight-read
traffic are both in the measured latency rather than one of them being an unvalidated bolt-on.

One table, not two like `gdn`. GDN splits into `gdn_context`/`gdn_decode` because prefill and
decode dispatch genuinely different kernel functions. MoE does not: `fused_experts_impl`
(`vllm/model_executor/layers/fused_moe/fused_moe.py`) is the same kernel at every token count on
this hardware (Triton `fused_moe_kernel`, selected via `UnquantizedMoeBackend.TRITON` on
SM90/H20); what changes with `num_tokens` is only an autotuned launch config and a
`naive_block_assignment` flag at very small M, not the kernel identity. So one `num_tokens` sweep
covers both decode- and prefill-shaped regimes.

`router` is not part of this family. `topk_weights`/`topk_ids` are passed INTO `fused_experts()`
already computed, so the router/gating GEMM runs before this kernel and stays on the generic
`gemm` table, unchanged. A model's optional shared expert (every token traverses it
unconditionally, no routing) also stays on the generic `gemm` table: there is no activated-expert
correction to make there in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..hardware import GpuSpec
from ..sol import SolComponents, moe_sol
from .base import register_op

_SLICE_COLUMNS = ("hidden_size", "intermediate_size", "num_experts", "top_k", "moe_dtype")


@dataclass(frozen=True)
class MoeKernelOp:
    """The routed-MoE grouped GEMM, interpolated over `num_tokens` only.

    Frozen because the instance below is a registry singleton shared by every caller.
    """

    name = "moe"
    family = "moe"
    slice_columns = _SLICE_COLUMNS
    axis_columns = ("num_tokens",)
    fits_floor = True
    # `vllm::fused_experts` is not in `CompilationConfig._attention_ops` (the default
    # `splitting_ops` list), unlike GDN's `vllm::qwen_gdn_attention_core`, so it stays inside
    # whatever CUDA graph wraps its forward segment -- both the FULL decode graph and the
    # PIECEWISE prefill/mixed graph's non-split region. There is no eager phase here the way
    # GDN's prefill is. Verified at one low-`num_tokens` point before trusting this for the
    # whole sweep: see `profiler/kernel/runner/moe.py`.
    graph_replayed = True

    def sol(
        self,
        spec: GpuSpec,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """
        Return the roofline of one routed-MoE grouped-GEMM call.

        Args:
            spec (GpuSpec): Hardware the roofline is evaluated against.
            slice_key (tuple[Any, ...]): `(hidden_size, intermediate_size, num_experts,
                top_k, moe_dtype)`.
            coords (tuple[float, ...]): `(num_tokens,)`.
            floor_s (float): Fitted dispatch floor in seconds, applied inside the `max`.

        Returns:
            SolComponents: Compute, memory, and floor components in seconds.
        """
        hidden_size, intermediate_size, num_experts, top_k, dtype = slice_key
        return moe_sol(
            spec,
            num_tokens=int(coords[0]),
            hidden_size=int(hidden_size),
            intermediate_size=int(intermediate_size),
            num_experts=int(num_experts),
            top_k=int(top_k),
            dtype=dtype,
            floor_s=floor_s,
        )

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the `moe:` sweep subtree into a `MoeSweep`."""
        # Imported lazily: the cost path prices kernels from parquet and must not load the
        # profiler package to do it. Four dots because this module is
        # `srl.cost.kernel.ops.moe`. `tests/test_kernel_package_imports.py` checks the depth.
        from ....profiler.kernel.sweep import MoeSweep

        return MoeSweep.from_dict(value)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Time every swept `num_tokens` through vLLM's real `fused_experts()` dispatch."""
        # Imported here so the cost path never loads torch.
        from ....profiler.kernel.runner.moe import MoeRunner

        return MoeRunner(model, config).run(writer_factory=writer_factory)


MOE_OP = MoeKernelOp()
register_op(MOE_OP)
