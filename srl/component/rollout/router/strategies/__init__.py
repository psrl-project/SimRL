"""Built-in routing strategies.

Importing this package registers every strategy in ``STRATEGY_REGISTRY``.
Each strategy lives in its own module so selection rules remain independent.
"""

from . import (  # noqa: F401
    burst_isolation,
    inventory_deadline,
    kv_cache_aware,
    length_aware,
    prefix_aware,
    random_choice,
    request_num_balance,
    round_robin,
    throughput_balance,
)

