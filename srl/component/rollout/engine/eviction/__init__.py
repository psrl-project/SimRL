"""Built-in eviction strategies.

Importing this package registers every strategy in ``EVICTION_REGISTRY``.
Each lives in its own module so the rules stay independent and comparable.
"""

from . import latest_deadline, longest_ctx, vllm_lru  # noqa: F401
from .strategy import (
    EVICTION_REGISTRY,
    EvictionStrategy,
    build_eviction,
    register_eviction,
)

__all__ = [
    "EVICTION_REGISTRY",
    "EvictionStrategy",
    "build_eviction",
    "register_eviction",
]
