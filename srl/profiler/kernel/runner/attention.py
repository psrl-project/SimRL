"""Measure both attention phases, through vLLM's selected backend and through FlashInfer.

Both phases come out of one sweep per backend, not one per op: a single KV cache is allocated
and prefill and decode are timed against it, so measuring per op would build the grid twice and
could select a different backend for each phase.

Two backends are measured rather than one so that the choice is a number rather than an
argument. `CommQuery`-style backend selection on the cost side reads whichever table it was
asked for. The row's `kernel_source` records what actually ran.

This runner is a dispatcher: the timing loops live in `_vllm_attn` and `_flashinfer_attn`,
which are kept as separate modules because each imports its own engine at module scope.
"""

from __future__ import annotations

from pathlib import Path

from .base import KernelRunner, WriterFactory


class AttentionRunner(KernelRunner):
    """Times prefill and decode through both vLLM and FlashInfer, usually four tables."""

    def run(self, *, writer_factory: WriterFactory) -> tuple[Path, ...]:
        """
        Measure both attention phases through both backends.

        Args:
            writer_factory (WriterFactory): Builds a writer from
                `(framework, version, kernel_source)`. The kernel source is only known after
                backend selection, so a writer cannot be passed in directly.

        Returns:
            tuple[Path, ...]: The written parquet paths: vLLM context and generation, then
                FlashInfer context and generation. Three paths rather than four when the
                model's GQA group size has no compiled FlashInfer decode kernel, which
                `profile_flashinfer_attention` warns about and skips.
        """
        from ._flashinfer_attn import profile_flashinfer_attention
        from ._vllm_attn import profile_vllm_attention

        written = list(
            profile_vllm_attention(self.model, self.config, writer_factory=writer_factory)
        )
        written.extend(
            profile_flashinfer_attention(self.model, self.config, writer_factory=writer_factory)
        )
        return tuple(written)
