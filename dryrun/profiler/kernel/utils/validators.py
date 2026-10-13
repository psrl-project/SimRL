"""Parse and validate the integer grids a sweep config declares.

Shared by every sweep rather than written per family: a grid holding a zero or a negative
token count would reach the measurement loop and fail there, by which point the message
names a tensor shape instead of the config key that produced it. The `field` argument is
what keeps the error pointing back at the YAML.
"""

from __future__ import annotations

from typing import Any

from ...schema import ProfileValidationError


def positive_int_tuple(value: Any, *, field: str) -> tuple[int, ...]:
    """
    Parse a YAML list into a tuple of positive integers.

    Args:
        value (Any): The raw YAML value, expected to be a non-empty list or tuple.
        field (str): Dotted config path, quoted back in the error so a bad grid names
            its own key.

    Returns:
        tuple[int, ...]: The parsed entries, in declaration order.

    Raises:
        ProfileValidationError: If `value` is not a non-empty sequence, or any entry is
            not positive.
    """
    if not isinstance(value, (list, tuple)) or not value:
        raise ProfileValidationError(f"{field} must be a non-empty list of positive integers.")
    parsed = tuple(int(item) for item in value)
    if any(item <= 0 for item in parsed):
        raise ProfileValidationError(f"{field} entries must be positive.")
    return parsed


def non_negative_int_tuple(value: Any, *, field: str) -> tuple[int, ...]:
    """
    Parse a YAML list into a tuple of non-negative integers.

    Separate from `positive_int_tuple` because zero is meaningful on one axis:
    `attention.generation_steps` uses it for the prefill-only point, where no KV history
    has accumulated yet.

    Args:
        value (Any): The raw YAML value, expected to be a non-empty list or tuple.
        field (str): Dotted config path, quoted back in the error.

    Returns:
        tuple[int, ...]: The parsed entries, in declaration order.

    Raises:
        ProfileValidationError: If `value` is not a non-empty sequence, or any entry is
            negative.
    """
    if not isinstance(value, (list, tuple)) or not value:
        raise ProfileValidationError(f"{field} must be a non-empty list of non-negative integers.")
    parsed = tuple(int(item) for item in value)
    if any(item < 0 for item in parsed):
        raise ProfileValidationError(f"{field} entries must not be negative.")
    return parsed
