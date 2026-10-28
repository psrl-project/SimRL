"""Measure vLLM's real fused-MoE grouped GEMM directly, as `GemmRunner` measures linears.

Calls `fused_experts()` (`vllm/model_executor/layers/fused_moe/fused_moe.py:1529`) with hand-
built random tensors matching its real signature -- the same "no VllmConfig, no engine, no
model" pattern `runner/gdn.py` uses, because `fused_experts_impl` touches no `VllmConfig`,
`get_current_vllm_config`, or `torch.distributed`: TP's effect is already fully captured by
`grid/moe.py::moe_dims` sweeping the TP-divided `intermediate_size`, the same convention
`grid/gemm.py` uses.

On H20/SM90, `UnquantizedMoeBackend` backend selection resolves to `TritonExperts`, whose
`.apply()` calls the identical `fused_moe_kernel` Triton kernel `fused_experts_impl` dispatches
directly -- so calling `fused_experts()` standalone exercises the real kernel path a live engine
runs for bf16/unquantized MoE on this hardware.

Unlike `runner/gdn.py`, this is a `KernelRunner` subclass: there is exactly one op and one table
here, so none of the two-tables-from-one-sweep machinery GDN needed applies, and
`KernelRunner.time(call, op_name="moe")` already picks the right timing helper off
`op_for("moe").graph_replayed`.

`vllm::fused_experts` is not in `CompilationConfig._attention_ops` (the default `splitting_ops`
list), unlike GDN's `vllm::qwen_gdn_attention_core`, so it is captured inside both the FULL
decode graph and the PIECEWISE prefill graph's non-split region -- evidence that graph replay is
right everywhere in this sweep, not just at decode shapes. That is a structural argument, not a
measurement: before trusting it for the whole grid, time one small `num_tokens` point (e.g. 1)
both ways (`time_cuda_call` vs `time_cuda_graph_replay`) and compare, the same check `gdn.py`'s
module docstring describes having done for its own decode path. If they disagree sharply,
`MoeKernelOp.graph_replayed` in `cost/kernel/ops/moe.py` is the place to flip.
"""

from __future__ import annotations

from pathlib import Path

from ..grid.moe import moe_dims
from .base import KernelRunner, WriterFactory


class MoeRunner(KernelRunner):
    """Times every swept `num_tokens` through vLLM's real `fused_experts()` dispatch."""

    def run(self, *, writer_factory: WriterFactory) -> tuple[Path, ...]:
        """
        Time every swept token count and write the `moe` table.

        Args:
            writer_factory (WriterFactory): Builds a writer from
                `(framework, version, kernel_source)`.

        Returns:
            tuple[Path, ...]: The single written parquet path.
        """
        import torch
        import vllm
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        from ..utils.timing import torch_dtype

        dtype_name = self.config.measurement.dtype
        dtype = torch_dtype(dtype_name)
        device = torch.device("cuda")
        dims = moe_dims(self.model, tensor_parallel=self.config.tensor_parallel)
        rows: list[dict[str, object]] = []
        for num_tokens in self.config.sweep("moe").num_tokens:
            hidden_states = torch.randn(num_tokens, dims.hidden_size, dtype=dtype, device=device)
            gate_up_w = torch.randn(
                dims.num_experts, 2 * dims.intermediate_size, dims.hidden_size,
                dtype=dtype, device=device,
            )
            down_w = torch.randn(
                dims.num_experts, dims.hidden_size, dims.intermediate_size,
                dtype=dtype, device=device,
            )
            # Random softmax logits, so `topk_ids` lands close to uniform over experts in
            # aggregate -- the same balanced-router assumption `sol.moe_sol` makes, and the one
            # an auxiliary-loss-trained router is optimised toward.
            logits = torch.randn(num_tokens, dims.num_experts, dtype=torch.float32, device=device)
            topk_weights, topk_ids = torch.topk(logits.softmax(dim=-1), dims.top_k, dim=-1)
            topk_ids = topk_ids.to(torch.int32)

            def _call(
                hidden=hidden_states, w1=gate_up_w, w2=down_w,
                tw=topk_weights, ti=topk_ids,
            ) -> None:
                fused_experts(
                    hidden,
                    w1,
                    w2,
                    tw,
                    ti,
                    activation=MoEActivation.SILU,
                    global_num_experts=dims.num_experts,
                )

            latency_s = self.time(_call, op_name="moe")
            rows.append(
                self.row(
                    "moe",
                    hidden_size=dims.hidden_size,
                    intermediate_size=dims.intermediate_size,
                    num_experts=dims.num_experts,
                    top_k=dims.top_k,
                    moe_dtype=dtype_name,
                    num_tokens=int(num_tokens),
                    latency_s=latency_s,
                )
            )
            del hidden_states, gate_up_w, down_w, logits, topk_weights, topk_ids
            torch.cuda.empty_cache()

        writer = writer_factory("vllm", vllm.__version__, "vllm::fused_experts")
        return (writer.write("moe", rows),)
