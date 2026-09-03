"""Architecture-derived activation footprint for one training microbatch.

The previous model used `max_tokens * min(num_microbatches, pp)` as an opaque
feature with a single fitted slope. That carried no architecture, so a fit could not
transfer across models, and it attributed a pipeline factor to configurations that do
not have one.

The shape here is derived from what Megatron actually retains under full activation
recompute, then checked against measurement. With `recompute_granularity=full`,
`recompute_method=uniform`, `recompute_num_layers=1`, each transformer layer keeps
only its own input; the intra-layer tensors are rebuilt one layer at a time during
backward. The output projection is not recomputed, so the vocabulary-sized logits
survive the forward pass and dominate for a large vocabulary.

Measured on 8x H20, Qwen3-8B (36 layers, hidden 4096, vocab 151936), micro-batch 1,
sequence lengths 512/1024/2048, across four topologies. Per-token slopes of peak
allocated bytes: 624202 at TP1/PP1, 402054 at TP1/PP4, 398009 at TP4/PP1, 354868 at
TP8/PP1. Fitting all four recovers a layer coefficient of 1.0012 and a logit
coefficient of 1.0580 with at most 1.3% error -- coefficients that also match what the
two-layer and thirty-six-layer slopes give independently, which is why they are treated
as analytic rather than fitted.

Two scaling facts came out of that fit and neither was guessable. The layer term is
divided by both the tensor and pipeline degrees; the logit term is divided by neither.
And sequence length is the only batch axis that matters at PP=1: the number of
microbatches changed the measured peak by 0.0 MiB, because 1F1B runs one microbatch's
forward and backward to completion before starting the next.
"""

from __future__ import annotations

from dataclasses import dataclass

from dryrun.profiler.schema import ModelDescriptor, SequenceStats

from .base import TrainingOutOfDomainError, TrainingParallelism

# Empirical multiplier on the logit term, covering loss-computation temporaries that
# live alongside the logits and are not separately modelled.
#
# Solving the two measured per-token slopes jointly -- 345721 bytes at 2 layers and
# 624403 bytes at 36 layers, same hidden size and vocabulary -- separates the two
# terms exactly: the layer coefficient comes out at 1.0006, i.e. `hidden * 2` bytes
# per layer per token to within 5 bytes, while the logit coefficient comes out at
# 1.0568. So the layer term is analytic and this residual belongs to the loss path.
# It is not vocabulary padding, which accounts for at most 0.4%.
#
# Kept explicit rather than folded into the fitted scale so that the fitted scale
# stays a diagnostic: a value far from one then means the geometry is wrong.
LOGIT_OVERHEAD = 1.0568


@dataclass(frozen=True)
class ActivationGeometry:
    """Architecture values that set the per-token activation footprint."""

    num_layers: int
    hidden_size: int
    vocab_size: int
    activation_dtype_bytes: int = 2

    def __post_init__(self) -> None:
        for name in ("num_layers", "hidden_size", "vocab_size", "activation_dtype_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")

    @classmethod
    def from_model(
        cls,
        model: ModelDescriptor,
        *,
        activation_dtype_bytes: int = 2,
    ) -> ActivationGeometry:
        """Read the geometry from a profiled model descriptor."""
        vocab_size = model.extra.get("vocab_size")
        if model.num_layers is None or model.hidden_size is None or vocab_size is None:
            raise ValueError(
                "The activation model needs num_layers, hidden_size, and vocab_size. "
                f"Model {model.name!r} is missing at least one."
            )
        return cls(
            num_layers=int(model.num_layers),
            hidden_size=int(model.hidden_size),
            vocab_size=int(vocab_size),
            activation_dtype_bytes=activation_dtype_bytes,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "num_layers": float(self.num_layers),
            "hidden_size": float(self.hidden_size),
            "vocab_size": float(self.vocab_size),
            "activation_dtype_bytes": float(self.activation_dtype_bytes),
        }

    @classmethod
    def from_dict(cls, value: dict[str, float]) -> ActivationGeometry:
        return cls(
            num_layers=int(value["num_layers"]),
            hidden_size=int(value["hidden_size"]),
            vocab_size=int(value["vocab_size"]),
            activation_dtype_bytes=int(value.get("activation_dtype_bytes", 2)),
        )

    def checkpoint_bytes_per_token(self, tensor_parallel: int) -> float:
        """
        Return bytes retained per token for layer-input checkpoints.

        One hidden-sized tensor per layer plus the block's own input. Tensor
        parallelism splits the hidden dimension of these tensors.
        """
        per_token = (self.num_layers + 1) * self.hidden_size * self.activation_dtype_bytes
        return per_token / tensor_parallel

    def logit_bytes_per_token(self) -> float:
        """
        Return bytes retained per token for the output projection.

        Not divided by the tensor degree. The output layer is vocabulary-parallel, so
        each rank owns a slice of the logits, but the loss path materializes a
        full-vocabulary-sized footprint per rank anyway. Measured per-token slopes fall
        far more slowly with tensor parallelism than a 1/TP logit term allows: 624202
        bytes at TP=1, 398009 at TP=4, 354868 at TP=8, where dividing both terms would
        predict 156059 and 78029. Fitting the four profiled topologies with the logit
        term left replicated recovers a layer coefficient of 1.0012 and a logit
        coefficient of 1.0580 -- both matching the values obtained independently from
        the two-layer and thirty-six-layer slopes -- with at most 1.3% error. Dividing
        it instead forced the fitted scale to 2.53 at TP=4 and 4.59 at TP=8, which is
        the signature of a wrong term rather than a machine-specific correction.

        `LOGIT_OVERHEAD` covers the loss-computation temporaries; see its definition.
        """
        return self.vocab_size * self.activation_dtype_bytes * LOGIT_OVERHEAD


def in_flight_microbatches(num_microbatches: int, pipeline_parallel: int) -> int:
    """
    Return how many microbatches hold activations at once under 1F1B.

    A pipeline stage keeps up to `pp` microbatches alive through the warmup phase.
    Without pipelining this is always one: each microbatch's backward completes
    before the next forward begins, which is why the measured peak was identical for
    one and four microbatches at PP=1.
    """
    if num_microbatches <= 0 or pipeline_parallel <= 0:
        raise ValueError("num_microbatches and pipeline_parallel must be positive.")
    return min(num_microbatches, pipeline_parallel)


def activation_bytes(
    geometry: ActivationGeometry,
    microbatches: tuple[SequenceStats, ...],
    parallelism: TrainingParallelism,
) -> float:
    """
    Predict activation bytes on the most loaded pipeline stage.

    Under 1F1B the warmup depth and the logit footprint sit on opposite ends of the
    pipeline, so the worst stage is whichever of the two ends is larger:

    - The first stage keeps up to `pp` microbatches in flight before its first
      backward, but holds no logits.
    - The last stage holds the logits, yet its warmup depth is one: it backpropagates
      each microbatch immediately after receiving it.

    That asymmetry is why the measured peak rank at PP=4 was identical for one and four
    microbatches -- 27.81 GiB in both cases. Scaling checkpoints by
    `min(num_microbatches, pp)` on the peak stage would have predicted 0.56 GiB of
    difference. Modelling both ends and taking the larger reproduces it, and it is also
    why the *non*-peak ranks did move slightly with the microbatch count.

    Taking the largest microbatch keeps this an upper bound for ragged batches.
    """
    if not microbatches:
        raise TrainingOutOfDomainError("Activation memory needs at least one microbatch.")
    peak_tokens = max(item.sum_tokens for item in microbatches)
    checkpoints = geometry.checkpoint_bytes_per_token(parallelism.tp) * peak_tokens
    stage_checkpoints = checkpoints / parallelism.pp
    concurrent = in_flight_microbatches(len(microbatches), parallelism.pp)
    first_stage = stage_checkpoints * concurrent
    last_stage = stage_checkpoints + geometry.logit_bytes_per_token() * peak_tokens
    return max(first_stage, last_stage)
