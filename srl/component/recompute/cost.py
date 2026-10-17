"""Log-prob recomputation cost models.

PPO and GRPO need log-probabilities of the consumed batch under the new policy.
That is a forward pass on the training side, and it is not cheap: measured runs put
it at roughly a quarter of the training block. Leaving it at zero understates every
training block by that much.

Three models, in descending order of how much they know:

- `TrainingForwardRecomputeCost` prices it as a fraction of the *same* fitted
  training curve the optimizer step uses. Preferred: the fraction follows from
  FLOP accounting rather than fitting, and it inherits the topology and
  sequence-length behaviour of a real measurement.
- `AnalyticalRecomputeCost` prices it from parameter count and peak FLOPs.
  Needs an achieved-FLOPs number that is really a fitted quantity in disguise.
- `FixedRecomputeCost` charges a constant per token.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...cost.training.base import TrainingCostModel, TrainingParallelism, TrainingWorkload

# A transformer layer costs 2 FLOPs per parameter per token in a forward pass,
# 4 in a backward pass, and 2 more when activations are recomputed instead of
# stored. The forward share of a step therefore depends on the checkpointing
# policy, and nothing else.
_STEP_FLOP_MULTIPLIER = {
    "none": 6.0,
    "selective": 7.0,
    "full": 8.0,
}
_FORWARD_FLOP_MULTIPLIER = 2.0


class RecomputeCostModel(ABC):
    """Log-prob recomputation latency model."""

    @abstractmethod
    def recompute_time(self, workload: TrainingWorkload, parallelism: TrainingParallelism) -> float:
        """Wall-clock seconds to recompute log-probs for one consumed batch.

        Args:
            workload: The batch being recomputed, carrying its sequence lengths
                and checkpointing policy.
            parallelism: Mesh the recompute pass runs on.
        """


class FixedRecomputeCost(RecomputeCostModel):
    """Constant cost per token, divided across the data-parallel ranks."""

    def __init__(self, time_per_token: float):
        self.time_per_token = time_per_token

    def recompute_time(self, workload: TrainingWorkload, parallelism: TrainingParallelism) -> float:
        return self.time_per_token * sum(workload.sequence_lengths) / max(1, parallelism.dp)


class AnalyticalRecomputeCost(RecomputeCostModel):
    """Forward-pass FLOPs divided by an achieved throughput.

    `peak_flops` has to be the *achieved* rate, not the datasheet peak, which
    makes it a fitted coefficient wearing a hardware-constant costume. Prefer
    `TrainingForwardRecomputeCost`, which reads the same information out of a
    measured training curve.
    """

    def __init__(self, model_params: int, peak_flops: float, overhead_factor: float = 1.1):
        self.model_params = model_params
        self.peak_flops = peak_flops
        self.overhead_factor = overhead_factor

    def recompute_time(self, workload: TrainingWorkload, parallelism: TrainingParallelism) -> float:
        total_tokens = sum(workload.sequence_lengths)
        world_size = max(1, parallelism.tp * parallelism.pp * parallelism.cp * parallelism.dp)
        flops = _FORWARD_FLOP_MULTIPLIER * self.model_params * total_tokens
        return flops / (world_size * self.peak_flops) * self.overhead_factor


class TrainingForwardRecomputeCost(RecomputeCostModel):
    """A fraction of the fitted training step, set by FLOP accounting.

    A log-prob pass is the forward half of a training step, so it costs the
    forward share of whatever that step costs. With full activation
    recomputation a step is 8 FLOPs per parameter per token (2 forward, 4
    backward, 2 recomputed) and the forward pass alone is 2, giving one
    quarter. The unembedding runs over the full vocabulary in both, which adds
    a few percent on a model whose vocab is large relative to its width.

    Nothing here is fitted: the ratio comes from the FLOP counts and the absolute
    scale from the training model, so this inherits that model's topology and
    sequence-length behaviour for free. It underestimates slightly against measured
    runs, the residual being backward-pass work that parallelises differently from
    forward (optimizer and gradient reductions have no forward counterpart).

    Args:
        training_cost: The same model pricing the optimizer step.
        hidden_size: Model width, for the unembedding correction.
        vocab_size: Vocabulary size, for the unembedding correction.
        parameter_count: Total parameters, for the unembedding correction.
    """

    def __init__(
        self,
        training_cost: TrainingCostModel,
        hidden_size: int,
        vocab_size: int,
        parameter_count: int,
    ):
        self.training_cost = training_cost
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.parameter_count = parameter_count

    def forward_fraction(self, recompute: str) -> float:
        """Forward share of one training step under a checkpointing policy.

        Raises:
            ValueError: If the policy is not one of none, selective, or full.
        """
        key = recompute.lower()
        try:
            step_multiplier = _STEP_FLOP_MULTIPLIER[key]
        except KeyError as exc:
            raise ValueError(
                f"Unknown activation recompute policy {recompute!r}. "
                f"Known policies: {sorted(_STEP_FLOP_MULTIPLIER)}."
            ) from exc

        # Both passes run the unembedding, so it scales the layer term without
        # changing the forward-to-step ratio of the layers themselves.
        layer_flops = self.parameter_count
        logit_flops = self.hidden_size * self.vocab_size
        correction = (layer_flops + logit_flops) / layer_flops
        return _FORWARD_FLOP_MULTIPLIER / step_multiplier * correction

    def recompute_time(self, workload: TrainingWorkload, parallelism: TrainingParallelism) -> float:
        estimate = self.training_cost.estimate(workload, parallelism)
        return estimate.latency_s * self.forward_fraction(workload.recompute)
