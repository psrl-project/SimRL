"""Built-in routing strategies.

Importing this package registers every strategy in ``STRATEGY_REGISTRY``.
Each strategy lives in its own module so selection rules remain independent.
"""

from . import (  # noqa: F401
    kv_cache_aware,
    length_aware,
    planned_tier_tp,
    prefix_aware,
    random_choice,
    request_num_balance,
    round_robin,
    scg_admission,
    throughput_balance,
    version_aware_tp,
    version_capped_tp,
)

