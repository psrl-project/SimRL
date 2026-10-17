"""Read a model's per-layer dimensions off a `ModelDescriptor`, inferring what a checkpoint omits.

These three are shared by the grid layer (which derives shapes from them) and the runner layer
(which builds tensors from them), so they live outside both. Each one refuses rather than
guessing when the descriptor is too incomplete to answer: a zero or a silently-wrong hidden
size produces a measured table describing a model nobody runs.

Kept free of `torch`: `grid/` imports this and the cost path imports `grid/`.
"""

from __future__ import annotations

from ...schema import ModelDescriptor, ProfileValidationError


def model_head_dim(model: ModelDescriptor) -> int:
    """
    Return the per-head dimension, inferring it when the checkpoint omits it.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.

    Returns:
        int: The explicit `head_dim`, or `hidden_size // num_attention_heads`.

    Raises:
        ProfileValidationError: If `head_dim` is absent and either `hidden_size` or
            `num_attention_heads` is missing, leaving nothing to infer from.
    """
    head_dim = model.extra.get("head_dim")
    if head_dim is not None:
        return int(head_dim)
    if model.hidden_size is None or model.num_attention_heads is None:
        raise ProfileValidationError("Model is missing hidden_size or num_attention_heads.")
    return model.hidden_size // model.num_attention_heads


def model_intermediate_size(model: ModelDescriptor) -> int:
    """
    Return the MLP intermediate size, falling back to the common 4x-hidden rule.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.

    Returns:
        int: The explicit `intermediate_size`, or `4 * hidden_size`.

    Raises:
        ProfileValidationError: If `intermediate_size` is absent and `hidden_size` is too.
    """
    intermediate_size = model.extra.get("intermediate_size")
    if intermediate_size is not None:
        return int(intermediate_size)
    if model.hidden_size is None:
        raise ProfileValidationError("Model is missing hidden_size.")
    return 4 * model.hidden_size


def model_vocab_size(model: ModelDescriptor) -> int:
    """
    Return the vocabulary size from the checkpoint's extra fields.

    No fallback, unlike the two above: a vocabulary size cannot be derived from anything
    else in the descriptor, and the logits projection it sizes is the single largest GEMM
    in the model.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.

    Returns:
        int: The declared `vocab_size`.

    Raises:
        ProfileValidationError: If the descriptor carries no `vocab_size`.
    """
    vocab_size = model.extra.get("vocab_size")
    if vocab_size is None:
        raise ProfileValidationError("Model is missing vocab_size in extra fields.")
    return int(vocab_size)
