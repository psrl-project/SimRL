"""KV block pool: occupancy, prefix reuse, and eviction.

This is a state machine, not a cost model. It fits no coefficient, reads no
artifact, and has no domain gate. `srl/cost/rollout/kv_cache.py` answers
*how many blocks exist*; this package answers *who holds them and what can
hit*. The only interface between the two is the `kv_blocks` integer.

Every mutating method takes a `Recorder` (see `..undo`), because
`advance_one_event` is speculative and the driver rolls back the events it does
not commit.
"""

from .block import KvBlock
from .free_list import FreeBlockList
from .lineage import block_hashes
from .manager import KvCacheManager
from .pool import BlockPool

__all__ = [
    "BlockPool",
    "FreeBlockList",
    "KvBlock",
    "KvCacheManager",
    "block_hashes",
]
