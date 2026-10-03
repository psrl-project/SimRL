# Rollout Prefix Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the rollout engine a real KV block pool with prefix reuse, so a GRPO group pays for its shared prompt once instead of `group_size` times and a post-weight-sync re-prefill hits the blocks the engine kept.

**Architecture:** A new `engine/kv/` package holds a line-by-line port of the vLLM v1 block pool (blocks with `ref_cnt` plus a hash, an LRU free queue, a hash-to-block map, eviction on reuse). Block hashes are *derived* from workload provenance rather than read from a trace column, because we generate the workload and therefore know which tokens came from where. `NativeInstance` keeps its structure: occupancy queries route to the pool, and a cache hit is expressed by pre-setting `prefilled`, so the cost model is untouched.

**Tech Stack:** Python 3.11+, `collections.OrderedDict` for the LRU queue, pytest. No new third-party dependency. Validation uses the existing `dryrun/workload/inference.py` vLLM harness plus a `/metrics` reader.

## Global Constraints

- Read `.claude/coding-style.md` and `.claude/codebase-map.md` before writing code; apply both.
- Activate the environment first: `source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh`.
- No backward compatibility. Remove obsolete paths rather than adding fallbacks, shims, or migrations.
- `block_size` default is 16. Only **full** blocks receive a hash; a partially filled block is never cacheable.
- Prompt-interior blocks key on `source_group_id`, never `group_id`. `group_id` is monotonic and so changes on workload wraparound, which would miss reuse the real engine achieves.
- Fit no coefficients anywhere in this work. Every expected number is arithmetic over `prompt_len`, `block_size`, and the length distribution.
- `_kv_exhaustion_steps()` (`dryrun/component/rollout/engine/instance.py:234-278`) keeps its closed form. Do not replace it with a search.
- Run `pytest tests/ -q` before every commit. A commit that reddens an existing test is a failed task.

---

## File Structure

| Path | Responsibility |
|---|---|
| `dryrun/component/rollout/engine/kv/__init__.py` | Package exports |
| `dryrun/component/rollout/engine/kv/block.py` | `KvBlock`: a block's id, refcount, and hash |
| `dryrun/component/rollout/engine/kv/lineage.py` | Derive a request's block hashes from provenance |
| `dryrun/component/rollout/engine/kv/pool.py` | `BlockPool`: allocation, LRU free queue, hash map, eviction |
| `dryrun/component/rollout/engine/kv/manager.py` | `KvCacheManager`: the four methods `NativeInstance` calls |
| `dryrun/component/rollout/engine/instance.py` | Occupancy routes to the pool; hit pre-sets `prefilled` |
| `dryrun/component/rollout/engine/snapshot.py` | Make pool mutations undoable |
| `dryrun/component/rollout/engine/state.py` | `StepStat` gains hit/computed token counts |
| `dryrun/component/rollout/engine/request.py` | `Request` gains `prefix_hit_tokens` |
| `dryrun/config/schema.py` | `enable_prefix_caching` knob |
| `dryrun/cli/main.py` | Thread the knob into `SimConfig` |
| `dryrun/simulator/config.py` | `SimConfig.enable_prefix_caching` |
| `dryrun/telemetry/writer.py` | Emit the new `StepStat` fields |
| `dryrun/workload/prefix_probe.py` | Read vLLM's prefix-cache counters from `/metrics` |
| `tests/test_kv_pool.py` | Tasks 1-4 |
| `tests/test_kv_instance.py` | Tasks 5-7 |
| `tests/test_kv_config.py` | Task 8 |
| `tests/test_prefix_probe.py` | Task 9 |
| `examples/profile/vllm/prefix_cache_probe.yaml` | Task 10 fixture |
| `docs/simulation/prefix-cache.md` | Task 11 |

**Dependency order:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11. Task 10 needs a GPU; Tasks 1-9 and 11 do not.

---

### Task 1: `KvBlock`

A block is a slot in the pool. It carries a refcount (how many live requests hold it) and optionally a hash (which content it holds, so a later request can find it).

**Files:**
- Create: `dryrun/component/rollout/engine/kv/__init__.py`
- Create: `dryrun/component/rollout/engine/kv/block.py`
- Test: `tests/test_kv_pool.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `KvBlock(block_id: int)` with attributes `block_id: int`, `ref_cnt: int` (starts 0), `block_hash: Hashable | None` (starts `None`); methods `incr_ref() -> None`, `decr_ref() -> None`, `reset_hash() -> None`. Assigning `block_hash` when it is already set raises `ValueError`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kv_pool.py`:

```python
"""Tests for the rollout KV block pool and provenance-derived block hashes."""

from __future__ import annotations

import pytest

from dryrun.component.rollout.engine.kv import KvBlock


class TestKvBlock:
    def test_starts_unheld_and_unhashed(self):
        block = KvBlock(block_id=7)
        assert block.block_id == 7
        assert block.ref_cnt == 0
        assert block.block_hash is None

    def test_refcount_tracks_holders(self):
        block = KvBlock(block_id=0)
        block.incr_ref()
        block.incr_ref()
        assert block.ref_cnt == 2
        block.decr_ref()
        assert block.ref_cnt == 1

    def test_reassigning_a_hash_is_a_bug_not_an_update(self):
        """One physical block holding two logical identities is the hardest bug
        class here, so assigning twice raises instead of silently overwriting."""
        block = KvBlock(block_id=0)
        block.block_hash = ("p", 42, 0)
        with pytest.raises(ValueError, match="already"):
            block.block_hash = ("p", 42, 1)

    def test_reset_hash_allows_reassignment(self):
        block = KvBlock(block_id=0)
        block.block_hash = ("p", 42, 0)
        block.reset_hash()
        assert block.block_hash is None
        block.block_hash = ("r", 9, 3)
        assert block.block_hash == ("r", 9, 3)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/DryRun-RL
pytest tests/test_kv_pool.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'dryrun.component.rollout.engine.kv'`.

- [ ] **Step 3: Write minimal implementation**

Create `dryrun/component/rollout/engine/kv/block.py`:

```python
"""One slot in the KV block pool.

A block is identified by `block_id` and tracks two things: how many live
requests hold it (`ref_cnt`), and which content it holds (`block_hash`).

The two are deliberately independent. A block whose `ref_cnt` has fallen to
zero keeps its hash, so a later request with the same prefix can still find
and reuse it. That state -- free but still addressable -- is what makes reuse
across a weight update or an agentic round boundary possible at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional


@dataclass
class KvBlock:
    """A pool slot, with a refcount and an optional content hash.

    Args:
        block_id: Index of this block in the pool.
    """

    block_id: int
    ref_cnt: int = 0
    _block_hash: Optional[Hashable] = None

    def incr_ref(self) -> None:
        self.ref_cnt += 1

    def decr_ref(self) -> None:
        self.ref_cnt -= 1

    @property
    def block_hash(self) -> Optional[Hashable]:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, value: Hashable) -> None:
        """Assign this block's content hash.

        Raises:
            ValueError: If a hash is already assigned. Overwriting would mean
                one physical block standing for two different prefixes, which
                corrupts every later hit silently rather than loudly.
        """
        if self._block_hash is not None:
            raise ValueError(
                f"Block {self.block_id} already holds hash {self._block_hash!r}; "
                f"call reset_hash() before assigning {value!r}."
            )
        self._block_hash = value

    def reset_hash(self) -> None:
        """Drop this block's content hash, making it assignable again."""
        self._block_hash = None
```

Create `dryrun/component/rollout/engine/kv/__init__.py`:

```python
"""KV block pool: occupancy, prefix reuse, and eviction.

This is a state machine, not a cost model. It fits no coefficient, reads no
artifact, and has no domain gate. `dryrun/cost/rollout/kv_cache.py` answers
*how many blocks exist*; this package answers *who holds them and what can
hit*. The only interface between the two is the `kv_blocks` integer.
"""

from .block import KvBlock

__all__ = ["KvBlock"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_kv_pool.py -q
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add dryrun/component/rollout/engine/kv/ tests/test_kv_pool.py
git commit -m "feat(kv): add KvBlock, the pool slot with refcount and content hash"
```

---

### Task 2: Provenance-derived block hashes

This is the design's pivot. vLLM hashes a block as `hash(parent_hash, block_tokens)`, so a block hash means *every token from position 0 to this block's end*. We hold no tokens, but we hold the provenance that produced them, which answers the same question: two contexts agree on a prefix exactly when they agree on its provenance.

**Files:**
- Create: `dryrun/component/rollout/engine/kv/lineage.py`
- Modify: `dryrun/component/rollout/engine/kv/__init__.py`
- Test: `tests/test_kv_pool.py`

**Interfaces:**
- Consumes: `Request` from `dryrun.component.rollout.engine.request`, using fields `prompt_len: int`, `generated: int`, `rid: int`, `source_group_id: int`.
- Produces: `block_hashes(req: Request, block_size: int) -> list[Hashable]`, one entry per full block of `req.prompt_len + req.generated`, in order. Prompt-interior entries are `("p", source_group_id, j)`; all others are `("r", rid, j)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kv_pool.py`:

```python
from dryrun.component.rollout.engine import Request
from dryrun.component.rollout.engine.kv import block_hashes


def _req(rid: int, prompt_len: int, source_group_id: int, generated: int = 0) -> Request:
    req = Request(
        rid=rid,
        prompt_len=prompt_len,
        target_len=4096,
        v_traj=0,
        source_group_id=source_group_id,
    )
    req.generated = generated
    return req


class TestBlockHashes:
    def test_only_full_blocks_get_a_hash(self):
        """vLLM's own rule: a partially filled block is not cacheable. At
        prompt_len=150 with block_size=16 that is 9 blocks over 144 tokens."""
        ids = block_hashes(_req(rid=1, prompt_len=150, source_group_id=42), block_size=16)
        assert len(ids) == 9

    def test_prompt_interior_blocks_key_on_the_shared_prompt(self):
        ids = block_hashes(_req(rid=1, prompt_len=150, source_group_id=42), block_size=16)
        assert ids[0] == ("p", 42, 0)
        assert ids[8] == ("p", 42, 8)

    def test_siblings_share_every_prompt_block(self):
        """Eight responses to one prompt are what GRPO samples, and the whole
        point of this module is that they pay for the prompt once."""
        a = block_hashes(_req(rid=1, prompt_len=150, source_group_id=42), block_size=16)
        b = block_hashes(_req(rid=2, prompt_len=150, source_group_id=42), block_size=16)
        assert a == b

    def test_different_prompts_share_nothing(self):
        a = block_hashes(_req(rid=1, prompt_len=150, source_group_id=42), block_size=16)
        b = block_hashes(_req(rid=2, prompt_len=150, source_group_id=43), block_size=16)
        assert set(a).isdisjoint(b)

    def test_the_block_straddling_the_prompt_end_is_private(self):
        """prompt_len=150 leaves block 9 covering tokens 144-159, of which only
        144-149 are prompt. Its content differs per response, so it cannot be
        shared even though most of its tokens are."""
        ids = block_hashes(
            _req(rid=7, prompt_len=150, source_group_id=42, generated=20), block_size=16
        )
        assert ids[8] == ("p", 42, 8)
        assert ids[9] == ("r", 7, 9)

    def test_generated_blocks_are_private_to_their_request(self):
        a = block_hashes(
            _req(rid=1, prompt_len=160, source_group_id=42, generated=64), block_size=16
        )
        b = block_hashes(
            _req(rid=2, prompt_len=160, source_group_id=42, generated=64), block_size=16
        )
        assert a[:10] == b[:10]
        assert set(a[10:]).isdisjoint(b[10:])

    def test_a_prompt_that_is_a_whole_number_of_blocks_shares_all_of_them(self):
        ids = block_hashes(_req(rid=1, prompt_len=160, source_group_id=42), block_size=16)
        assert len(ids) == 10
        assert all(h[0] == "p" for h in ids)

    def test_continued_context_reproduces_its_own_earlier_hashes(self):
        """A request resumed after a weight update re-prefills prompt+generated.
        Its generated blocks must hash exactly as they did when first filled,
        or resumption would never hit."""
        first = block_hashes(
            _req(rid=5, prompt_len=160, source_group_id=42, generated=64), block_size=16
        )
        later = block_hashes(
            _req(rid=5, prompt_len=160, source_group_id=42, generated=128), block_size=16
        )
        assert later[: len(first)] == first

    def test_a_context_shorter_than_one_block_has_no_hashes(self):
        assert block_hashes(_req(rid=1, prompt_len=10, source_group_id=1), block_size=16) == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_kv_pool.py -q
```

Expected: FAIL, `ImportError: cannot import name 'block_hashes'`.

- [ ] **Step 3: Write minimal implementation**

Create `dryrun/component/rollout/engine/kv/lineage.py`:

```python
"""Block hashes derived from workload provenance.

vLLM hashes a block as `hash(parent_block_hash, block_token_ids)`
(`third_party/vllm/vllm/v1/core/kv_cache_utils.py:617-623`). Chaining the
parent in is what makes a block hash mean *every token from position 0 to this
block's end*, so that

    A: [hello] [world]
    B: [bye]   [world]

do not collide on block 1: reuse is only valid when the whole prefix agrees.

We hold no token ids. We hold something that answers the same question: two
contexts agree on a prefix exactly when they agree on the provenance that
produced it. So the hash is written rather than computed, from two integers
already on `Request`.

This is where we diverge from Frontier, which must be handed a
`block_hash_ids` column on its trace because it replays real requests and
never sees their tokens. We generate the workload, so the provenance is known
and the trace format stays untouched.

**Known one-directional bias.** Two *different* prompts that happen to share
leading full blocks hit in the real engine and miss here. On RL workloads
distinct problems almost certainly differ at token 0, and every weight update
clears the hash map anyway, so this is second order -- but it only ever
under-reports hits, never over-reports. See `docs/reference/limitations.md`.
"""

from __future__ import annotations

from typing import Hashable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..request import Request


def block_hashes(req: "Request", block_size: int) -> list[Hashable]:
    """Hash ids for every full block of this request's current context.

    Args:
        req: The request. Its context is `prompt_len + generated`, which is the
            token count the engine knows about so far.
        block_size: Tokens per KV block.

    Returns:
        One id per full block, in block order. A block lying wholly inside the
        shared prompt keys on `source_group_id`, so siblings of one prompt
        agree on it. Every other block keys on `rid` and is private.

        `source_group_id` rather than `group_id` is deliberate: `group_id` is
        monotonic by design, so a repeating workload gives the same prompt a
        new id on wraparound and reuse the real engine achieves would be
        missed. `source_group_id` records which workload group the lengths came
        from, which is the prompt's actual identity.

    Raises:
        ValueError: If `block_size` is not positive.
    """
    if block_size <= 0:
        raise ValueError(f"Expected block_size > 0, got {block_size!r}.")

    context = req.prompt_len + req.generated
    # Only full blocks are cacheable, which is vLLM's own rule. The block
    # straddling the end of the prompt is therefore private even though most of
    # its tokens are shared.
    return [
        ("p", req.source_group_id, j)
        if (j + 1) * block_size <= req.prompt_len
        else ("r", req.rid, j)
        for j in range(context // block_size)
    ]
```

Replace `dryrun/component/rollout/engine/kv/__init__.py`:

```python
"""KV block pool: occupancy, prefix reuse, and eviction.

This is a state machine, not a cost model. It fits no coefficient, reads no
artifact, and has no domain gate. `dryrun/cost/rollout/kv_cache.py` answers
*how many blocks exist*; this package answers *who holds them and what can
hit*. The only interface between the two is the `kv_blocks` integer.
"""

from .block import KvBlock
from .lineage import block_hashes

__all__ = ["KvBlock", "block_hashes"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_kv_pool.py -q
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add dryrun/component/rollout/engine/kv/ tests/test_kv_pool.py
git commit -m "feat(kv): derive block hashes from workload provenance

Frontier must be fed a block_hash_ids trace column because it replays real
requests and never sees their tokens. We generate the workload, so
(source_group_id, block_index) answers the same question the chained hash
does, and the trace format stays untouched."
```

---

### Task 3: `BlockPool`

The pool owns allocation, the LRU free queue, the hash map, and eviction. Its central invariant: **a block may be in the free queue and in the hash map at the same time.** That is the "nobody holds it, but it is still addressable" state, and every cross-sync and cross-round hit depends on it.

**Files:**
- Create: `dryrun/component/rollout/engine/kv/pool.py`
- Modify: `dryrun/component/rollout/engine/kv/__init__.py`
- Test: `tests/test_kv_pool.py`

**Interfaces:**
- Consumes: `KvBlock` from Task 1.
- Produces: `BlockPool(num_blocks: int, enable_caching: bool = True)` with properties `num_free_blocks: int`, `num_used_blocks: int`, `usage: float`, `eviction_count: int`; methods `cached_block(block_hash: Hashable) -> KvBlock | None`, `new_blocks(n: int) -> list[KvBlock]`, `touch(blocks: list[KvBlock]) -> None`, `free_blocks(blocks: Iterable[KvBlock]) -> None`, `cache_full_blocks(blocks: list[KvBlock], hashes: list[Hashable], n_cached: int, n_full: int) -> None`, `reset_hashes() -> None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kv_pool.py`:

```python
from dryrun.component.rollout.engine.kv import BlockPool


class TestBlockPoolAllocation:
    def test_starts_entirely_free(self):
        pool = BlockPool(num_blocks=10)
        assert pool.num_free_blocks == 10
        assert pool.num_used_blocks == 0
        assert pool.usage == 0.0

    def test_new_blocks_are_held_and_removed_from_free(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(3)
        assert len(blocks) == 3
        assert all(b.ref_cnt == 1 for b in blocks)
        assert pool.num_free_blocks == 7
        assert pool.usage == 0.3

    def test_overallocation_raises_rather_than_overcommitting(self):
        pool = BlockPool(num_blocks=2)
        with pytest.raises(ValueError, match="free"):
            pool.new_blocks(3)

    def test_freeing_returns_blocks_only_when_the_last_holder_leaves(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(2)
        pool.touch(blocks)  # a second holder
        pool.free_blocks(blocks)
        assert pool.num_free_blocks == 8, "still held by the second holder"
        pool.free_blocks(blocks)
        assert pool.num_free_blocks == 10


class TestBlockPoolHits:
    def test_a_cached_hash_is_findable(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(2)
        pool.cache_full_blocks(blocks, [("p", 1, 0), ("p", 1, 1)], n_cached=0, n_full=2)
        assert pool.cached_block(("p", 1, 0)) is blocks[0]
        assert pool.cached_block(("p", 1, 1)) is blocks[1]

    def test_an_unknown_hash_misses(self):
        assert BlockPool(num_blocks=10).cached_block(("p", 1, 0)) is None

    def test_a_freed_block_keeps_its_hash_and_still_hits(self):
        """The pool's central invariant. Reuse after a weight update and across
        an agentic round boundary both happen on blocks whose ref_cnt is zero
        but whose hash is still mapped."""
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(1)
        pool.cache_full_blocks(blocks, [("p", 1, 0)], n_cached=0, n_full=1)
        pool.free_blocks(blocks)
        assert pool.num_free_blocks == 10
        assert pool.cached_block(("p", 1, 0)) is blocks[0]

    def test_touch_revives_a_free_block_without_double_counting(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(1)
        pool.cache_full_blocks(blocks, [("p", 1, 0)], n_cached=0, n_full=1)
        pool.free_blocks(blocks)
        hit = pool.cached_block(("p", 1, 0))
        pool.touch([hit])
        assert hit.ref_cnt == 1
        assert pool.num_free_blocks == 9

    def test_cache_full_blocks_skips_already_cached_prefix(self):
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(3)
        pool.cache_full_blocks(blocks, [("p", 1, 0), ("p", 1, 1), ("p", 1, 2)], n_cached=2, n_full=3)
        assert pool.cached_block(("p", 1, 0)) is None
        assert pool.cached_block(("p", 1, 2)) is blocks[2]

    def test_caching_is_a_no_op_when_disabled(self):
        pool = BlockPool(num_blocks=10, enable_caching=False)
        blocks = pool.new_blocks(1)
        pool.cache_full_blocks(blocks, [("p", 1, 0)], n_cached=0, n_full=1)
        assert pool.cached_block(("p", 1, 0)) is None

    def test_reset_hashes_clears_the_map_but_not_occupancy(self):
        """A weight update calls reset_prefix_cache, which empties the block
        hash map. Blocks a live request still holds stay held."""
        pool = BlockPool(num_blocks=10)
        blocks = pool.new_blocks(2)
        pool.cache_full_blocks(blocks, [("p", 1, 0), ("p", 1, 1)], n_cached=0, n_full=2)
        pool.reset_hashes()
        assert pool.cached_block(("p", 1, 0)) is None
        assert pool.num_free_blocks == 8
        assert blocks[0].block_hash is None


class TestBlockPoolEviction:
    def test_reuse_evicts_the_oldest_freed_block_first(self):
        pool = BlockPool(num_blocks=2)
        first = pool.new_blocks(1)
        pool.cache_full_blocks(first, [("p", 1, 0)], n_cached=0, n_full=1)
        second = pool.new_blocks(1)
        pool.cache_full_blocks(second, [("p", 2, 0)], n_cached=0, n_full=1)
        pool.free_blocks(first)
        pool.free_blocks(second)
        # Both are free and both still hit. Allocating one must take the
        # least recently freed, which is `first`.
        pool.new_blocks(1)
        assert pool.cached_block(("p", 1, 0)) is None, "oldest freed was evicted"
        assert pool.cached_block(("p", 2, 0)) is second[0], "newer one survived"
        assert pool.eviction_count == 1

    def test_eviction_frees_the_hash_for_reassignment(self):
        pool = BlockPool(num_blocks=1)
        blocks = pool.new_blocks(1)
        pool.cache_full_blocks(blocks, [("p", 1, 0)], n_cached=0, n_full=1)
        pool.free_blocks(blocks)
        reused = pool.new_blocks(1)
        assert reused[0].block_hash is None
        pool.cache_full_blocks(reused, [("p", 9, 0)], n_cached=0, n_full=1)
        assert pool.cached_block(("p", 9, 0)) is reused[0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_kv_pool.py -q
```

Expected: FAIL, `ImportError: cannot import name 'BlockPool'`.

- [ ] **Step 3: Write minimal implementation**

Create `dryrun/component/rollout/engine/kv/pool.py`:

```python
"""The KV block pool: allocation, LRU reuse, and eviction.

Ported from vLLM v1 (`vllm/v1/core/block_pool.py`) rather than approximated,
because what this has to reproduce is an engine's exact admission and reuse
behaviour under pressure, and that behaviour lives in the mechanism.

The central invariant: **a block may sit in the free queue and in the hash map
at the same time.** That is the "nobody holds it, but it is still addressable"
state. Reuse across a weight update and across an agentic round boundary both
land on exactly those blocks, so a pool that dropped hashes on release would
be wrong in the two cases that matter most here.

A hash is destroyed only when its block is handed to a new owner
(`new_blocks` reaching `_maybe_evict`), never on release.

Where vLLM and Frontier both hand-roll a doubly linked free list with
`prev`/`next` pointers on every block, this uses an `OrderedDict` keyed by
block id. `popitem(last=False)` takes the least recently freed and
`pop(block_id)` removes from the middle, both O(1), without two pointers per
block or an invariant to maintain by hand.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Hashable, Optional

from .block import KvBlock


class BlockPool:
    """A fixed set of KV blocks, with prefix-cache bookkeeping.

    Args:
        num_blocks: Pool capacity, from `dryrun/cost/rollout/kv_cache.py`.
        enable_caching: When False, blocks are still allocated and freed but no
            hash is ever recorded, so nothing can hit. This is a data state,
            not a second code path.

    Raises:
        ValueError: If `num_blocks` is negative.
    """

    def __init__(self, num_blocks: int, enable_caching: bool = True) -> None:
        if num_blocks < 0:
            raise ValueError(f"Expected num_blocks >= 0, got {num_blocks!r}.")
        self.num_blocks = int(num_blocks)
        self.enable_caching = bool(enable_caching)
        self.blocks = [KvBlock(block_id=i) for i in range(self.num_blocks)]
        # Insertion order is release order, so the first item is the least
        # recently freed: the LRU victim.
        self._free: OrderedDict[int, KvBlock] = OrderedDict(
            (b.block_id, b) for b in self.blocks
        )
        self._hash_to_block: dict[Hashable, KvBlock] = {}
        self.eviction_count = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def usage(self) -> float:
        """Fraction of the pool currently held, 0.0 for an empty pool."""
        if self.num_blocks <= 0:
            return 0.0
        return self.num_used_blocks / self.num_blocks

    def cached_block(self, block_hash: Hashable) -> Optional[KvBlock]:
        """The block holding `block_hash`, or None on a miss."""
        return self._hash_to_block.get(block_hash)

    def new_blocks(self, n: int) -> list[KvBlock]:
        """Take `n` unheld blocks, evicting their hashes as they change hands.

        Args:
            n: How many blocks to allocate.

        Returns:
            The allocated blocks, each with `ref_cnt == 1`.

        Raises:
            ValueError: If fewer than `n` blocks are free. The caller is
                expected to have checked capacity; overcommitting the pool
                would silently invent memory the engine does not have.
        """
        if n > self.num_free_blocks:
            raise ValueError(
                f"Cannot allocate {n} KV blocks: only {self.num_free_blocks} free."
            )
        taken: list[KvBlock] = []
        for _ in range(n):
            _, block = self._free.popitem(last=False)
            self._maybe_evict(block)
            block.incr_ref()
            taken.append(block)
        return taken

    def touch(self, blocks: list[KvBlock]) -> None:
        """Take a reference to blocks found by hash.

        A hit block may currently be free, in which case it has to leave the
        free queue before anyone can grow into it.
        """
        for block in blocks:
            if block.ref_cnt == 0:
                self._free.pop(block.block_id, None)
            block.incr_ref()

    def free_blocks(self, blocks: Iterable[KvBlock]) -> None:
        """Release a reference, returning blocks whose last holder left.

        The hash is deliberately kept: a released block stays addressable so a
        later request with the same prefix can hit it.
        """
        for block in blocks:
            block.decr_ref()
            if block.ref_cnt == 0:
                self._free[block.block_id] = block

    def cache_full_blocks(
        self,
        blocks: list[KvBlock],
        hashes: list[Hashable],
        n_cached: int,
        n_full: int,
    ) -> None:
        """Record hashes for blocks that have become full.

        Args:
            blocks: This request's blocks, in block order.
            hashes: This request's block hashes, in the same order.
            n_cached: How many leading blocks are already recorded.
            n_full: How many leading blocks are now full.
        """
        if not self.enable_caching or n_cached >= n_full:
            return
        for index in range(n_cached, min(n_full, len(hashes), len(blocks))):
            block = blocks[index]
            if block.block_hash is None:
                block.block_hash = hashes[index]
                self._hash_to_block[block.block_hash] = block

    def reset_hashes(self) -> None:
        """Empty the hash map, leaving occupancy untouched.

        This is vLLM's `reset_prefix_cache`, which a weight update triggers via
        PSRL's `pull_model`. Blocks a live request still holds stay held; they
        simply stop being addressable by content.
        """
        for block in self._hash_to_block.values():
            block.reset_hash()
        self._hash_to_block.clear()

    def _maybe_evict(self, block: KvBlock) -> bool:
        """Drop a block's hash as it is handed to a new owner."""
        if block.block_hash is None:
            return False
        self._hash_to_block.pop(block.block_hash, None)
        block.reset_hash()
        self.eviction_count += 1
        return True
```

Replace `dryrun/component/rollout/engine/kv/__init__.py`:

```python
"""KV block pool: occupancy, prefix reuse, and eviction.

This is a state machine, not a cost model. It fits no coefficient, reads no
artifact, and has no domain gate. `dryrun/cost/rollout/kv_cache.py` answers
*how many blocks exist*; this package answers *who holds them and what can
hit*. The only interface between the two is the `kv_blocks` integer.
"""

from .block import KvBlock
from .lineage import block_hashes
from .pool import BlockPool

__all__ = ["BlockPool", "KvBlock", "block_hashes"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_kv_pool.py -q
```

Expected: 26 passed.

- [ ] **Step 5: Commit**

```bash
git add dryrun/component/rollout/engine/kv/ tests/test_kv_pool.py
git commit -m "feat(kv): add BlockPool with LRU reuse and eviction on rehoming

A freed block keeps its hash, so it stays addressable. Reuse across a weight
update and across an agentic round boundary both land on exactly those blocks.
OrderedDict replaces vLLM's hand-rolled doubly linked free list: same O(1)
semantics, no per-block pointers to keep consistent."
```

---

### Task 4: `KvCacheManager`

The four methods `NativeInstance` calls. This is where the longest-common-prefix hit lookup lives, including vLLM's full-hit edge case.

**Files:**
- Create: `dryrun/component/rollout/engine/kv/manager.py`
- Modify: `dryrun/component/rollout/engine/kv/__init__.py`
- Test: `tests/test_kv_pool.py`

**Interfaces:**
- Consumes: `BlockPool` (Task 3), `block_hashes` (Task 2).
- Produces: `KvCacheManager(num_blocks: int, block_size: int = 16, enable_caching: bool = True)` with attributes `pool: BlockPool`, `queries: int`, `hits: int`; properties `num_used_blocks: int`, `num_free_blocks: int`, `usage: float`; methods `computed_blocks(req) -> tuple[list[KvBlock], int]`, `can_allocate(req, n_tokens, hit_blocks) -> bool`, `allocate(req, n_tokens, hit_blocks) -> list[KvBlock] | None`, `free(req) -> None`, `reset_hashes() -> None`, `blocks_held(req) -> int`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kv_pool.py`:

```python
from dryrun.component.rollout.engine.kv import KvCacheManager


class TestComputedBlocks:
    def test_a_cold_manager_hits_nothing(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        blocks, tokens = mgr.computed_blocks(_req(rid=1, prompt_len=150, source_group_id=42))
        assert blocks == []
        assert tokens == 0

    def test_a_sibling_hits_every_shared_prompt_block(self):
        """prompt_len=150 shares 9 full blocks, so 144 of 150 prompt tokens are
        free for the second response onward. This is the limitation in
        docs/reference/limitations.md that reads 'Prompt prefill charged once
        per response'."""
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        first = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(first, n_tokens=150, hit_blocks=[])

        sibling = _req(rid=2, prompt_len=150, source_group_id=42)
        blocks, tokens = mgr.computed_blocks(sibling)
        assert len(blocks) == 9
        assert tokens == 144

    def test_hits_stop_at_the_first_miss(self):
        """Hashes carry chained semantics, so a miss at block j means content
        has diverged and later blocks must not be used even if they collide."""
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        req = _req(rid=1, prompt_len=160, source_group_id=42, generated=64)
        mgr.allocate(req, n_tokens=224, hit_blocks=[])
        mgr.free(req)
        mgr.pool.reset_hashes()

        # Re-cache only the first two blocks by replaying a short prefix.
        short = _req(rid=1, prompt_len=32, source_group_id=42)
        mgr.allocate(short, n_tokens=32, hit_blocks=[])

        blocks, tokens = mgr.computed_blocks(
            _req(rid=1, prompt_len=160, source_group_id=42, generated=64)
        )
        assert len(blocks) == 2
        assert tokens == 32

    def test_a_full_hit_hands_the_last_block_back(self):
        """vLLM's own edge case
        (vllm_v1_engine_replica_scheduler.py:1393-1399): a full hit leaves zero
        new tokens, but the model must forward at least one token to produce
        the next logits. Fires for real whenever prompt_len is a multiple of
        block_size."""
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        first = _req(rid=1, prompt_len=160, source_group_id=42)
        mgr.allocate(first, n_tokens=160, hit_blocks=[])

        sibling = _req(rid=2, prompt_len=160, source_group_id=42)
        blocks, tokens = mgr.computed_blocks(sibling)
        assert len(blocks) == 9, "ten blocks match, but the last is handed back"
        assert tokens == 144
        assert sibling.prompt_len - tokens == 16

    def test_counters_track_queries_and_hits_in_blocks(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        first = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(first, n_tokens=150, hit_blocks=[])
        mgr.computed_blocks(_req(rid=2, prompt_len=150, source_group_id=42))
        assert mgr.queries == 18, "9 blocks for the cold request, 9 for the sibling"
        assert mgr.hits == 9

    def test_disabled_caching_never_hits(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16, enable_caching=False)
        first = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(first, n_tokens=150, hit_blocks=[])
        blocks, tokens = mgr.computed_blocks(_req(rid=2, prompt_len=150, source_group_id=42))
        assert blocks == []
        assert tokens == 0


class TestAllocation:
    def test_allocation_accounts_occupancy(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        req = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(req, n_tokens=150, hit_blocks=[])
        assert mgr.num_used_blocks == 10, "ceil(150/16)"
        assert mgr.blocks_held(req) == 10

    def test_a_hit_sibling_only_allocates_its_own_tail(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        first = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(first, n_tokens=150, hit_blocks=[])
        used_after_first = mgr.num_used_blocks

        sibling = _req(rid=2, prompt_len=150, source_group_id=42)
        hit_blocks, hit_tokens = mgr.computed_blocks(sibling)
        sibling.prefilled = hit_tokens
        mgr.allocate(sibling, n_tokens=sibling.prompt_len - hit_tokens, hit_blocks=hit_blocks)
        assert mgr.num_used_blocks == used_after_first + 1, "only the private tail block"

    def test_can_allocate_refuses_what_does_not_fit(self):
        mgr = KvCacheManager(num_blocks=4, block_size=16)
        req = _req(rid=1, prompt_len=150, source_group_id=42)
        assert not mgr.can_allocate(req, n_tokens=150, hit_blocks=[])
        assert mgr.allocate(req, n_tokens=150, hit_blocks=[]) is None

    def test_free_releases_everything_the_request_held(self):
        mgr = KvCacheManager(num_blocks=100, block_size=16)
        req = _req(rid=1, prompt_len=150, source_group_id=42)
        mgr.allocate(req, n_tokens=150, hit_blocks=[])
        mgr.free(req)
        assert mgr.num_used_blocks == 0
        assert mgr.blocks_held(req) == 0

    def test_a_resumed_request_hits_its_own_generated_blocks(self):
        """Partial rollout: an abort keeps `generated` and zeroes `prefilled`,
        so the context must be re-prefilled. Its generated blocks hash exactly
        as they did when first filled, so they hit."""
        mgr = KvCacheManager(num_blocks=1000, block_size=16)
        req = _req(rid=5, prompt_len=160, source_group_id=42, generated=320)
        mgr.allocate(req, n_tokens=480, hit_blocks=[])
        mgr.free(req)

        req.prefilled = 0
        blocks, tokens = mgr.computed_blocks(req)
        assert tokens == 464, "30 of 30 full blocks hit, less the handed-back last one"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_kv_pool.py -q
```

Expected: FAIL, `ImportError: cannot import name 'KvCacheManager'`.

- [ ] **Step 3: Write minimal implementation**

Create `dryrun/component/rollout/engine/kv/manager.py`:

```python
"""Per-instance KV cache manager: the engine's view of the block pool.

Mirrors vLLM v1's `KVCacheManager`. `NativeInstance` talks to this and never to
`BlockPool` directly, so occupancy accounting and hit lookup stay in one place.
"""

from __future__ import annotations

from typing import Hashable, TYPE_CHECKING

from .block import KvBlock
from .lineage import block_hashes
from .pool import BlockPool

if TYPE_CHECKING:
    from ..request import Request


def _blocks_for(n_tokens: int, block_size: int) -> int:
    """Blocks needed to hold `n_tokens`, rounding up."""
    return (n_tokens + block_size - 1) // block_size


class KvCacheManager:
    """Allocation and prefix lookup for one engine instance.

    Args:
        num_blocks: Pool capacity.
        block_size: Tokens per block.
        enable_caching: When False no hash is recorded, so nothing hits.

    Attributes:
        queries: Blocks looked up since construction, counted per block to
            match vLLM's `vllm:prefix_cache_queries`.
        hits: Blocks found, matching `vllm:prefix_cache_hits`.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        enable_caching: bool = True,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"Expected block_size > 0, got {block_size!r}.")
        self.block_size = int(block_size)
        self.enable_caching = bool(enable_caching)
        self.pool = BlockPool(num_blocks, enable_caching=enable_caching)
        self._held: dict[int, list[KvBlock]] = {}
        self._n_cached: dict[int, int] = {}
        self.queries = 0
        self.hits = 0

    @property
    def num_used_blocks(self) -> int:
        return self.pool.num_used_blocks

    @property
    def num_free_blocks(self) -> int:
        return self.pool.num_free_blocks

    @property
    def usage(self) -> float:
        return self.pool.usage

    def blocks_held(self, req: "Request") -> int:
        """How many blocks this request currently holds."""
        return len(self._held.get(req.rid, ()))

    def computed_blocks(self, req: "Request") -> tuple[list[KvBlock], int]:
        """The longest cached prefix of this request's context.

        Walks the request's block hashes and stops at the first miss, because a
        hash carries chained semantics: a miss at block `j` means content has
        diverged there, so later blocks must not be reused even if their hashes
        happen to match.

        One edge case is ported verbatim from vLLM
        (`vllm_v1_engine_replica_scheduler.py:1393-1399`): on a *full* hit the
        last block is handed back. A full hit would leave zero new tokens, but
        the model must forward at least one token to produce the next logits.
        This fires for real whenever a group's `prompt_len` is a whole number
        of blocks.

        Args:
            req: The request being admitted or re-prefilled.

        Returns:
            The hit blocks and how many tokens they cover.
        """
        if not self.enable_caching:
            return [], 0

        hashes = block_hashes(req, self.block_size)
        hit: list[KvBlock] = []
        for block_hash in hashes:
            block = self.pool.cached_block(block_hash)
            if block is None:
                break
            hit.append(block)

        self.queries += len(hashes)
        self.hits += len(hit)

        tokens = len(hit) * self.block_size
        # A full hit must still leave one block of work to do.
        if hit and tokens >= req.prompt_len + req.generated:
            hit = hit[:-1]
            tokens -= self.block_size
            self.hits -= 1
        return hit, tokens

    def _new_blocks_needed(
        self,
        req: "Request",
        n_tokens: int,
        hit_blocks: list[KvBlock],
    ) -> int:
        """Blocks to allocate on top of what is held and what hit."""
        computed = req.prefilled + len(hit_blocks) * self.block_size
        required = _blocks_for(computed + n_tokens, self.block_size)
        return required - self.blocks_held(req) - len(hit_blocks)

    def can_allocate(
        self,
        req: "Request",
        n_tokens: int,
        hit_blocks: list[KvBlock],
    ) -> bool:
        """Whether `n_tokens` more of this request fit.

        Hit blocks that are currently free do not consume new capacity, so they
        are discounted from the free count rather than charged twice.
        """
        needed = self._new_blocks_needed(req, n_tokens, hit_blocks)
        revivable = sum(1 for b in hit_blocks if b.ref_cnt == 0)
        return needed <= self.pool.num_free_blocks - revivable

    def allocate(
        self,
        req: "Request",
        n_tokens: int,
        hit_blocks: list[KvBlock],
    ) -> list[KvBlock] | None:
        """Reserve room for `n_tokens` more of this request.

        Args:
            req: The request. `req.prefilled` must already reflect any hit.
            n_tokens: Tokens about to be computed.
            hit_blocks: Blocks from `computed_blocks`, adopted by this request.

        Returns:
            The newly allocated blocks, or None when capacity refuses.

        Raises:
            ValueError: If hit blocks are passed while caching is disabled,
                which would mean a caller invented a hit.
        """
        if hit_blocks and not self.enable_caching:
            raise ValueError("Hit blocks passed while prefix caching is disabled.")
        if not self.can_allocate(req, n_tokens, hit_blocks):
            return None

        self.pool.touch(hit_blocks)
        held = self._held.setdefault(req.rid, [])
        held.extend(hit_blocks)

        needed = self._new_blocks_needed(req, n_tokens, hit_blocks)
        fresh = self.pool.new_blocks(needed) if needed > 0 else []
        held.extend(fresh)

        if self.enable_caching:
            computed = req.prefilled + len(hit_blocks) * self.block_size
            n_full = (computed + n_tokens) // self.block_size
            self.pool.cache_full_blocks(
                held,
                block_hashes(req, self.block_size),
                n_cached=self._n_cached.get(req.rid, len(hit_blocks)),
                n_full=n_full,
            )
            self._n_cached[req.rid] = n_full
        return fresh

    def free(self, req: "Request") -> None:
        """Release every block this request holds, keeping their hashes.

        The hashes survive deliberately: a released block stays addressable, so
        a sibling or a later round can still hit it.
        """
        held = self._held.pop(req.rid, [])
        if held:
            self.pool.free_blocks(reversed(held))
        self._n_cached.pop(req.rid, None)

    def reset_hashes(self) -> None:
        """Empty the hash map. This is vLLM's `reset_prefix_cache`."""
        self.pool.reset_hashes()
        self._n_cached.clear()
```

Replace `dryrun/component/rollout/engine/kv/__init__.py`:

```python
"""KV block pool: occupancy, prefix reuse, and eviction.

This is a state machine, not a cost model. It fits no coefficient, reads no
artifact, and has no domain gate. `dryrun/cost/rollout/kv_cache.py` answers
*how many blocks exist*; this package answers *who holds them and what can
hit*. The only interface between the two is the `kv_blocks` integer.
"""

from .block import KvBlock
from .lineage import block_hashes
from .manager import KvCacheManager
from .pool import BlockPool

__all__ = ["BlockPool", "KvBlock", "KvCacheManager", "block_hashes"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_kv_pool.py -q
```

Expected: 37 passed.

- [ ] **Step 5: Commit**

```bash
git add dryrun/component/rollout/engine/kv/ tests/test_kv_pool.py
git commit -m "feat(kv): add KvCacheManager with longest-prefix hit lookup

Ports vLLM's full-hit edge case verbatim: a full hit hands the last block
back, because the model must forward at least one token to produce the next
logits. Fires whenever a group's prompt_len is a whole number of blocks."

```

---

### Task 5: Wire the pool into `NativeInstance` occupancy

Replace the derived occupancy sum with pool queries. No hit logic yet, so this must be behaviour-preserving: the existing suite is the test.

**Files:**
- Modify: `dryrun/component/rollout/engine/instance.py:20-74`
- Test: `tests/test_kv_instance.py`

**Interfaces:**
- Consumes: `KvCacheManager` (Task 4).
- Produces: `NativeInstance.__init__` gains keyword `enable_prefix_caching: bool = True`; attribute `self.kv: KvCacheManager`. `blocks_used()` and `free_blocks()` keep their signatures.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kv_instance.py`:

```python
"""Tests for prefix caching inside the rollout engine instance."""

from __future__ import annotations

import pytest

from dryrun.component.rollout.engine import NativeInstance, ReqStatus, Request
from dryrun.cost.rollout import PSRLFitted


def _cost() -> PSRLFitted:
    return PSRLFitted()


def _inst(**kwargs) -> NativeInstance:
    defaults = dict(
        cost=_cost(),
        token_budget=8192,
        kv_blocks=1000,
        block_size=16,
    )
    defaults.update(kwargs)
    return NativeInstance(**defaults)


def _req(rid: int, prompt_len: int, source_group_id: int, target_len: int = 64) -> Request:
    return Request(
        rid=rid,
        prompt_len=prompt_len,
        target_len=target_len,
        v_traj=0,
        group_id=source_group_id,
        source_group_id=source_group_id,
    )


class TestOccupancyComesFromThePool:
    def test_an_idle_instance_holds_nothing(self):
        inst = _inst()
        assert inst.blocks_used() == 0
        assert inst.free_blocks() == 1000

    def test_admitting_then_prefilling_charges_blocks(self):
        inst = _inst()
        assert inst.add_request(_req(rid=1, prompt_len=150, source_group_id=1), t=0.0)
        inst.advance_one_event(version=0)
        assert inst.blocks_used() > 0
        assert inst.blocks_used() + inst.free_blocks() == 1000

    def test_the_pool_and_the_instance_agree_after_a_full_run(self):
        inst = _inst()
        for rid in range(4):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=rid), t=0.0)
        for _ in range(200):
            if inst.advance_one_event(version=0) is None:
                break
        assert len(inst.completed) == 4
        assert inst.blocks_used() == 0, "every retired request released its blocks"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_kv_instance.py -q
```

Expected: FAIL. `NativeInstance.__init__` takes `cost` positionally but not as a keyword under that name, or the final assertion fails because `blocks_used()` still derives from `prefilled` and retired requests are not released.

- [ ] **Step 3: Write minimal implementation**

In `dryrun/component/rollout/engine/instance.py`, add the import at the top alongside the existing ones:

```python
from .kv import KvCacheManager
```

In `__init__`, add the parameter after `reject_if_running_full: bool = False`:

```python
        enable_prefix_caching: bool = True,
```

and replace the three KV lines in the body:

```python
        self.kv_blocks = kv_blocks
        self.block_size = block_size
```

with:

```python
        self.kv_blocks = kv_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        # Occupancy, hit lookup, and eviction all live in the pool. Disabling
        # prefix caching leaves the pool doing block accounting and simply
        # records no hashes, so there is no second code path to keep in step.
        self.kv = KvCacheManager(
            num_blocks=kv_blocks,
            block_size=block_size,
            enable_caching=enable_prefix_caching,
        )
```

Replace `blocks_used` and `free_blocks`:

```python
    def blocks_used(self) -> int:
        return self.kv.num_used_blocks

    def free_blocks(self) -> int:
        return self.kv.num_free_blocks
```

Every place that changes a request's `prefilled` must now tell the pool, and **`allocate` must be called before `prefilled` advances**, because it reads `req.prefilled` as the count already computed.

In `_prefill_budget_step`, replace the block

```python
            free -= blocks_for(req.prefilled + chunk, self.block_size) - have
            req.prefilled += chunk
```

with:

```python
            self.kv.allocate(req, n_tokens=chunk, hit_blocks=[])
            free -= blocks_for(req.prefilled + chunk, self.block_size) - have
            req.prefilled += chunk
```

In the decode loop of `_prefill_budget_step`, replace

```python
            r.add_tokens(version, 1)
            r.prefilled += 1
```

with:

```python
            self.kv.allocate(r, n_tokens=1, hit_blocks=[])
            r.add_tokens(version, 1)
            r.prefilled += 1
```

In `_decode_one_event` and in `advance_to`, replace both occurrences of

```python
                for r in self.running:
                    r.add_tokens(version, k)
                    r.prefilled += k
```

with:

```python
                for r in self.running:
                    self.kv.allocate(r, n_tokens=k, hit_blocks=[])
                    r.add_tokens(version, k)
                    r.prefilled += k
```

In `_retire`, release blocks as each request leaves:

```python
    def _retire(self, reqs: list[Request]) -> list[Request]:
        out = []
        for r in reqs:
            if r in self.running:
                self.running.remove(r)
            self.kv.free(r)
            r.status = ReqStatus.DONE
            r.complete_time = self.t_local
            self.completed.append(r)
            out.append(r)
        return out
```

In `_release`, the context is discarded and must be given back:

```python
    def _release(self, req: Request, keep_tokens: bool = True) -> None:
        req.num_aborts += 1
        if keep_tokens:
            req.reprefill_tokens += req.ctx
        self.kv.free(req)
        req.prefilled = 0
```

In `_preempt_one`, likewise:

```python
        self.running.remove(victim)
        victim.status = ReqStatus.PREEMPTED
        victim.num_preemptions += 1
        victim.reprefill_tokens += victim.ctx
        self.kv.free(victim)
        victim.prefilled = 0
        self.waiting.insert(0, victim)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_kv_instance.py -q && pytest tests/ -q
```

Expected: `tests/test_kv_instance.py` 3 passed, and the whole suite green. If `test_e2e.py` or `test_sim_telemetry.py` reddens, the pool and the requests disagree somewhere: find the `prefilled` mutation that skipped `self.kv`.

- [ ] **Step 5: Commit**

```bash
git add dryrun/component/rollout/engine/instance.py tests/test_kv_instance.py
git commit -m "refactor(kv): occupancy comes from the block pool, not from prefilled

Behaviour-preserving: no hit lookup yet. Every prefilled mutation now pairs
with an allocate or a free, which is what lets the next commit introduce hits
without the pool and the requests drifting apart."
```

---

### Task 6: Make pool mutations undoable

**This is a correctness requirement the spec did not state.** `undo_last_event` (`instance.py:80`) is a live path: `simulator.py:358` calls it through `_undo_and_align` for every speculatively-advanced event that does not commit. `InstanceSnapshot` captures `prefilled` per request, which sufficed while occupancy was *derived* from `prefilled`. Now that the pool holds `ref_cnt`, the hash map, and the free queue, an uncommitted prefill step would roll back `prefilled` while leaving pool state mutated, and the two would silently disagree.

**Files:**
- Modify: `dryrun/component/rollout/engine/snapshot.py`
- Test: `tests/test_kv_instance.py`

**Interfaces:**
- Consumes: `KvCacheManager` (Task 4), `NativeInstance.kv` (Task 5).
- Produces: `InstanceSnapshot` gains field `kv: KvCacheManager`, holding a deep copy captured before the event. `restore` reinstates it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kv_instance.py`:

```python
class TestUndoRestoresThePool:
    def test_undoing_an_event_restores_occupancy(self):
        """simulator.py:358 undoes every speculatively advanced event that does
        not commit. A pool left mutated while `prefilled` rolls back would
        disagree with the requests, and nothing downstream would notice."""
        inst = _inst()
        assert inst.add_request(_req(rid=1, prompt_len=150, source_group_id=1), t=0.0)
        inst.advance_one_event(version=0)
        before_used = inst.blocks_used()
        before_free = inst.free_blocks()

        inst.advance_one_event(version=0)
        inst.undo_last_event()

        assert inst.blocks_used() == before_used
        assert inst.free_blocks() == before_free

    def test_undoing_restores_hit_bookkeeping(self):
        inst = _inst()
        assert inst.add_request(_req(rid=1, prompt_len=150, source_group_id=7), t=0.0)
        inst.advance_one_event(version=0)
        queries_before = inst.kv.queries
        hits_before = inst.kv.hits

        inst.advance_one_event(version=0)
        inst.undo_last_event()

        assert inst.kv.queries == queries_before
        assert inst.kv.hits == hits_before

    def test_repeated_advance_and_undo_does_not_leak_blocks(self):
        inst = _inst()
        assert inst.add_request(_req(rid=1, prompt_len=150, source_group_id=1), t=0.0)
        inst.advance_one_event(version=0)
        baseline = inst.blocks_used()
        for _ in range(20):
            inst.advance_one_event(version=0)
            inst.undo_last_event()
        assert inst.blocks_used() == baseline
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_kv_instance.py -q
```

Expected: FAIL on `test_undoing_an_event_restores_occupancy`, with `blocks_used()` higher than `before_used` because the uncommitted step's allocation survived the undo.

- [ ] **Step 3: Write minimal implementation**

In `dryrun/component/rollout/engine/snapshot.py`, add to the imports:

```python
import copy
```

Add the field to `InstanceSnapshot`, after `req_snapshots`:

```python
    # The pool holds ref counts, a hash map, and a free queue, none of which
    # can be reconstructed from `prefilled`. An uncommitted event would
    # otherwise leave the pool and the requests disagreeing, with nothing
    # downstream to notice. `simulator.py:358` undoes events routinely, so this
    # is a live path and not a defensive measure.
    kv: object
```

In `capture`, add to the constructor call:

```python
            kv=copy.deepcopy(inst.kv),
```

In `restore`, add:

```python
        inst.kv = self.kv
```

Update the `InstanceSnapshot` docstring to name the new field:

```python
@dataclass
class InstanceSnapshot:
    """
    Minimal snapshot of `NativeInstance` mutable state, taken before each event.

    Sufficient to fully restore state via `NativeInstance.undo_last_event`.

    `kv` is a deep copy rather than a field list because the pool's state is a
    graph: blocks hold refcounts, the hash map and the free queue both point at
    blocks, and a hit revives a block from the free queue. Reconstructing that
    from scalars would duplicate the pool's own invariants in a second place.
    """
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_kv_instance.py -q && pytest tests/ -q
```

Expected: `tests/test_kv_instance.py` 6 passed, whole suite green.

- [ ] **Step 5: Commit**

```bash
git add dryrun/component/rollout/engine/snapshot.py tests/test_kv_instance.py
git commit -m "fix(kv): make pool state undoable alongside request state

undo_last_event is a live path (simulator.py:358 via _undo_and_align). With
occupancy derived from prefilled this needed nothing; with a pool holding
refcounts, a hash map and a free queue, an uncommitted event would leave the
two silently disagreeing."
```

---

### Task 7: Apply hits, and split `abort_all` into free plus reset

The payload. A hit pre-sets `prefilled`, so the saved tokens leave `t2` and `n_tokens` by themselves and the cost model needs no change.

**Files:**
- Modify: `dryrun/component/rollout/engine/instance.py`
- Modify: `dryrun/component/rollout/engine/state.py`
- Modify: `dryrun/component/rollout/engine/request.py`
- Test: `tests/test_kv_instance.py`

**Interfaces:**
- Consumes: Tasks 4-6.
- Produces: `StepStat` gains `n_prefill_hit: int = 0` and `n_prefill_computed: int = 0`. `Request` gains `prefix_hit_tokens: int = 0`. `NativeInstance` gains `_apply_prefix_hit(req: Request) -> int` returning tokens saved.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kv_instance.py`:

```python
class TestWithinGroupPromptReuse:
    def test_siblings_pay_for_the_shared_prompt_once(self):
        """The limitation in docs/reference/limitations.md: 'When a prompt has n
        sampled responses, the simulator charges prefill n times.' At
        prompt_len=150 with block_size=16 the shared part is 9 full blocks over
        144 tokens, so seven of eight siblings skip 144 tokens each."""
        inst = _inst()
        for rid in range(8):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=99), t=0.0)
        for _ in range(500):
            if inst.advance_one_event(version=0) is None:
                break

        assert len(inst.completed) == 8
        saved = sum(r.prefix_hit_tokens for r in inst.completed)
        assert saved == 7 * 144

    def test_prompt_prefill_drops_to_the_predicted_token_count(self):
        inst = _inst()
        for rid in range(8):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=99), t=0.0)
        for _ in range(500):
            if inst.advance_one_event(version=0) is None:
                break

        prompt_work = sum(s.n_prefill_computed for s in inst.step_stats)
        assert prompt_work == 150 + 7 * 6, "150 cold, then six private tokens each"

    def test_different_prompts_share_nothing(self):
        inst = _inst()
        for rid in range(8):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=rid), t=0.0)
        for _ in range(500):
            if inst.advance_one_event(version=0) is None:
                break
        assert sum(r.prefix_hit_tokens for r in inst.completed) == 0

    def test_disabling_prefix_caching_restores_the_old_cost(self):
        inst = _inst(enable_prefix_caching=False)
        for rid in range(8):
            assert inst.add_request(_req(rid=rid, prompt_len=150, source_group_id=99), t=0.0)
        for _ in range(500):
            if inst.advance_one_event(version=0) is None:
                break
        assert sum(r.prefix_hit_tokens for r in inst.completed) == 0
        assert sum(s.n_prefill_computed for s in inst.step_stats) == 8 * 150


class TestWeightSyncClearsHashes:
    def test_abort_all_frees_blocks_and_clears_the_hash_map(self):
        """PSRL's pause_for_sync aborts every request, then pull_model calls
        reset_prefix_cache. Both steps, in that order."""
        inst = _inst()
        for rid in range(4):
            assert inst.add_request(_req(rid=rid, prompt_len=160, source_group_id=5), t=0.0)
        for _ in range(50):
            inst.advance_one_event(version=0)

        inst.abort_all()
        assert inst.blocks_used() == 0
        assert inst.kv.pool.cached_block(("p", 5, 0)) is None

    def test_a_resumed_group_still_hits_on_the_shared_prompt(self):
        """The post-sync hit rate is not zero: the first sibling repopulates
        the prompt hashes and the rest hit them. This is the mechanism behind
        the 9% measured in outputs/simulate/SMG_LOOPBACK_FIX.md."""
        inst = _inst(kv_blocks=4000)
        for rid in range(4):
            assert inst.add_request(
                _req(rid=rid, prompt_len=160, source_group_id=5, target_len=512), t=0.0
            )
        for _ in range(200):
            inst.advance_one_event(version=0)
        inst.abort_all()
        saved_before = sum(r.prefix_hit_tokens for r in inst.waiting)

        for _ in range(200):
            if inst.advance_one_event(version=1) is None:
                break
        saved_after = sum(
            r.prefix_hit_tokens for r in (*inst.waiting, *inst.running, *inst.completed)
        )
        assert saved_after > saved_before, "siblings hit the repopulated prompt blocks"

    def test_an_agentic_round_boundary_keeps_hashes(self):
        """`cancel(keep_tokens=True)` is the agentic round boundary: free the
        blocks, keep the hashes, so the next round hits. This is the interface
        the agentic spec builds on, and the reason free and reset are separate."""
        inst = _inst()
        req = _req(rid=1, prompt_len=160, source_group_id=5)
        assert inst.add_request(req, t=0.0)
        for _ in range(50):
            inst.advance_one_event(version=0)

        inst.cancel([req], keep_tokens=True)
        assert inst.blocks_used() == 0
        assert inst.kv.pool.cached_block(("p", 5, 0)) is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_kv_instance.py -q
```

Expected: FAIL, `AttributeError: 'Request' object has no attribute 'prefix_hit_tokens'`.

- [ ] **Step 3: Write minimal implementation**

In `dryrun/component/rollout/engine/request.py`, add after `reprefill_tokens: int = 0`:

```python
    prefix_hit_tokens: int = 0
    """Prompt tokens this request skipped because the KV cache already held them.

    Accumulates across resumptions, so a request interrupted by a weight update
    and re-prefilled records what the cache saved each time.
    """
```

In `dryrun/component/rollout/engine/state.py`, add to `StepStat` after `n_prefill: int`:

```python
    n_prefill_hit: int = 0
    n_prefill_computed: int = 0
```

and extend its docstring:

```python
    `n_prefill` is the work this step charged. `n_prefill_hit` is what the
    prefix cache absorbed and `n_prefill_computed` is what was really computed,
    named to match the real engine's own columns (`prefill_tokens` and
    `computed (cache miss)` in `outputs/simulate/SMG_LOOPBACK_FIX.md`) so
    `dryrun-compare` puts like against like.
```

In `dryrun/component/rollout/engine/snapshot.py`, add `prefix_hit_tokens` to `ReqSnapshot`: add the field

```python
    prefix_hit_tokens: int
```

to `capture`

```python
            prefix_hit_tokens=req.prefix_hit_tokens,
```

and to `restore`

```python
        req.prefix_hit_tokens = self.prefix_hit_tokens
```

In `dryrun/component/rollout/engine/instance.py`, add the hit application method after `free_blocks`:

```python
    def _apply_prefix_hit(self, req: Request) -> int:
        """Advance `req.prefilled` over whatever the cache already holds.

        A hit is expressed by moving `prefilled` rather than by a separate
        `num_computed_tokens` field, because both prefill paths already compute
        `prompt_len + generated - prefilled` as the work remaining. The saved
        tokens therefore drop out of `n_tokens` and `t2` on their own, and the
        cost model needs no change at all.

        Args:
            req: A request about to be prefilled, with `prefilled == 0`.

        Returns:
            Tokens the cache covered.
        """
        if not self.enable_prefix_caching or req.prefilled > 0:
            return 0
        hit_blocks, hit_tokens = self.kv.computed_blocks(req)
        if hit_tokens <= 0:
            return 0
        if not self.kv.can_allocate(req, n_tokens=1, hit_blocks=hit_blocks):
            return 0
        self.kv.allocate(req, n_tokens=0, hit_blocks=hit_blocks)
        req.prefilled = hit_tokens
        req.prefix_hit_tokens += hit_tokens
        return hit_tokens
```

In `_prefill_budget_step`, apply the hit before measuring the work. Replace

```python
        free = self.free_blocks()
        for req in sorted(self.waiting, key=lambda r: (r.dispatch_time, r.rid)):
            if budget <= 0:
                break
            if req.prefilled == 0:
                if self.max_concurrency is not None and n_in_batch >= self.max_concurrency:
                    break
                n_in_batch += 1
            need = req.prompt_len + req.generated - req.prefilled
```

with

```python
        free = self.free_blocks()
        prefill_hit = 0
        for req in sorted(self.waiting, key=lambda r: (r.dispatch_time, r.rid)):
            if budget <= 0:
                break
            if req.prefilled == 0:
                if self.max_concurrency is not None and n_in_batch >= self.max_concurrency:
                    break
                n_in_batch += 1
                prefill_hit += self._apply_prefix_hit(req)
                free = self.free_blocks()
            need = req.prompt_len + req.generated - req.prefilled
```

and record it on the `StepStat`. Replace

```python
            StepStat(
                n_tokens,
                n_decode,
                prefill_tokens,
                self.cost.saturated(n_tokens, t2, ctxsum, n_reqs),
```

with

```python
            StepStat(
                n_tokens,
                n_decode,
                prefill_tokens,
                n_prefill_hit=prefill_hit,
                n_prefill_computed=prefill_tokens,
                saturated=self.cost.saturated(n_tokens, t2, ctxsum, n_reqs),
```

Note that this turns the fourth positional argument into a keyword, so check the remaining arguments in that call still line up: `steps=1`, `dt=dt`, `phase="prefill"`, `t=self.t_local + dt` are already keywords.

Apply the same hit in `_step_time_estimate` so the time estimate matches what the step will do. This path predicts a step without performing it, so it must **not** move the counters that `vllm:prefix_cache_queries` is compared against. Add a non-counting lookup to `KvCacheManager` in `dryrun/component/rollout/engine/kv/manager.py`:

```python
    def peek_hit_tokens(self, req: "Request") -> int:
        """Tokens a hit would cover, without recording a query.

        `_step_time_estimate` predicts a step without performing it, so it must
        not move the counters that `vllm:prefix_cache_queries` is compared
        against.
        """
        if not self.enable_caching:
            return 0
        hashes = block_hashes(req, self.block_size)
        n_hit = 0
        for block_hash in hashes:
            if self.pool.cached_block(block_hash) is None:
                break
            n_hit += 1
        tokens = n_hit * self.block_size
        if n_hit and tokens >= req.prompt_len + req.generated:
            tokens -= self.block_size
        return tokens
```

Then in `_step_time_estimate`, replace

```python
            eff = effective.get(id(req), req.prefilled)
            need = req.prompt_len + req.generated - eff
```

with:

```python
            eff = effective.get(id(req), req.prefilled)
            if eff == 0:
                eff = self.kv.peek_hit_tokens(req)
                effective[id(req)] = eff
            need = req.prompt_len + req.generated - eff
```

Finally split `abort_all`. Replace its body (keeping the existing docstring and adding to it):

```python
        aborted = list(self.running) + list(self.waiting)
        for r in aborted:
            self._release(r, keep_tokens=True)
            r.status = ReqStatus.WAITING
        self.waiting = aborted[:]
        self.running = []
        # `_release` already freed each request's blocks. Clearing the hash map
        # is the second, separate step: it is what `pull_model` ->
        # `reset_prefix_cache` does, and it is deliberately *not* done at an
        # agentic round boundary, where `cancel(keep_tokens=True)` frees blocks
        # but leaves their hashes addressable so the next round hits.
        self.kv.reset_hashes()
        return aborted
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_kv_instance.py -q && pytest tests/ -q
```

Expected: `tests/test_kv_instance.py` 13 passed, whole suite green. If `test_e2e.py` shifts timings, that is the intended effect of prefix caching; confirm the direction is *less* prefill work, never more, and update the expected values in that test with a note naming this commit.

- [ ] **Step 5: Commit**

```bash
git add dryrun/component/rollout/engine/ tests/test_kv_instance.py
git commit -m "feat(kv): apply prefix hits; split abort_all into free plus reset

A hit pre-sets prefilled, so saved tokens leave n_tokens and t2 on their own
and the cost model is untouched. A group of eight at prompt_len=150 drops from
1200 prefill tokens to 192, closing the 'prompt prefill charged once per
response' limitation.

free and reset are separate because a weight update clears hashes while an
agentic round boundary must not."
```

---

### Task 8: Config knob and telemetry

**Files:**
- Modify: `dryrun/config/schema.py:236-247`
- Modify: `dryrun/config/defaults/simulate.yaml`
- Modify: `dryrun/simulator/config.py`
- Modify: `dryrun/simulator/simulator.py:68-84`
- Modify: `dryrun/cli/main.py:388-422`
- Modify: `dryrun/telemetry/writer.py:105-120`
- Test: `tests/test_kv_config.py`

**Interfaces:**
- Consumes: Task 7.
- Produces: `RolloutConfig.enable_prefix_caching: bool = True`; `SimConfig.enable_prefix_caching: bool = True`; the `rollout_step` telemetry stream gains `n_prefill_hit` and `n_prefill_computed`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kv_config.py`:

```python
"""Tests that the prefix-caching knob reaches the engine and the telemetry."""

from __future__ import annotations

from dryrun.config.schema import RolloutConfig
from dryrun.simulator.config import SimConfig
from dryrun.workload.grouped import GroupedWorkload


def _workload() -> GroupedWorkload:
    return GroupedWorkload.from_lengths([150] * 8, [64] * 8)


class TestPrefixCachingKnob:
    def test_default_matches_the_simulated_engine_default(self):
        """Real vLLM ships with prefix caching on, so the default should be on:
        a default that does not match the simulated system is a silent bias."""
        assert RolloutConfig().enable_prefix_caching is True
        assert SimConfig(workload=_workload()).enable_prefix_caching is True

    def test_the_knob_reaches_every_instance(self):
        from dryrun.cost.rollout import PSRLFitted
        from dryrun.simulator.simulator import Simulator

        cfg = SimConfig(workload=_workload(), n_instances=2, enable_prefix_caching=False)
        sim = Simulator(cfg, rollout_cost=PSRLFitted())
        assert all(not inst.enable_prefix_caching for inst in sim.instances)
        assert all(not inst.kv.enable_caching for inst in sim.instances)


class TestTelemetryColumns:
    def test_step_stats_carry_hit_and_computed_tokens(self):
        from dryrun.component.rollout.engine import StepStat

        stat = StepStat(
            n_tokens=10,
            n_decode=4,
            n_prefill=6,
            saturated=False,
            n_prefill_hit=144,
            n_prefill_computed=6,
        )
        assert stat.n_prefill_hit == 144
        assert stat.n_prefill_computed == 6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_kv_config.py -q
```

Expected: FAIL, `AttributeError: 'RolloutConfig' object has no attribute 'enable_prefix_caching'`.

- [ ] **Step 3: Write minimal implementation**

In `dryrun/config/schema.py`, add to `RolloutConfig` after `block_size: int = 16`:

```python
    enable_prefix_caching: bool = True
    """Whether the engine reuses KV blocks across requests sharing a prefix.

    Defaults to True because real vLLM does: a default that disagrees with the
    simulated system is a silent bias. Turning it off leaves the pool doing
    block accounting and simply records no hashes, so nothing can hit; it is a
    data state rather than a second code path.
    """
```

In `dryrun/config/defaults/simulate.yaml`, under the `rollout:` block alongside `block_size`, add:

```yaml
  enable_prefix_caching: true
```

In `dryrun/simulator/config.py`, add to `SimConfig` after `block_size: int = 16`:

```python
    enable_prefix_caching: bool = True
```

In `dryrun/simulator/simulator.py`, add to the `NativeInstance(...)` construction after `reject_if_running_full=cfg.reject_if_running_full`:

```python
                    enable_prefix_caching=cfg.enable_prefix_caching,
```

In `dryrun/cli/main.py`, add to the `SimConfig(...)` call after `block_size=rollout.block_size`:

```python
        enable_prefix_caching=rollout.get("enable_prefix_caching", True),
```

In `dryrun/telemetry/writer.py`, add to the dict built around line 110, after `n_prefill=stat.n_prefill`:

```python
                n_prefill_hit=stat.n_prefill_hit,
                n_prefill_computed=stat.n_prefill_computed,
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_kv_config.py -q && pytest tests/ -q
```

Expected: `tests/test_kv_config.py` 3 passed, whole suite green. `tests/test_sim_telemetry.py` and `tests/test_telemetry.py` may assert on the `rollout_step` column set; add the two new columns there if so.

- [ ] **Step 5: Commit**

```bash
git add dryrun/config/ dryrun/simulator/ dryrun/cli/main.py dryrun/telemetry/writer.py tests/test_kv_config.py
git commit -m "feat(kv): add enable_prefix_caching knob, default on, and hit telemetry

Default on because real vLLM is: a default that disagrees with the simulated
system is a silent bias. Telemetry columns are named for the real engine's own
(prefill_tokens / computed) so dryrun-compare puts like against like."
```

---

### Task 9: `/metrics` reader for vLLM's prefix-cache counters

**Files:**
- Create: `dryrun/workload/prefix_probe.py`
- Test: `tests/test_prefix_probe.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `parse_prefix_cache_counters(metrics_text: str) -> PrefixCacheCounters` where `PrefixCacheCounters` is a frozen dataclass with `queries: int`, `hits: int`, and a `hit_rate: float` property; and `read_prefix_cache_counters(base_url: str) -> PrefixCacheCounters`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prefix_probe.py`:

```python
"""Tests for reading vLLM's prefix-cache counters out of /metrics."""

from __future__ import annotations

import pytest

from dryrun.workload.prefix_probe import PrefixCacheCounters, parse_prefix_cache_counters

_SAMPLE = """\
# HELP vllm:prefix_cache_queries GPU prefix cache queries, in terms of number of queried blocks.
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries{engine="0",model_name="Qwen3-8B"} 1440.0
# HELP vllm:prefix_cache_hits GPU prefix cache hits, in terms of number of cached blocks.
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits{engine="0",model_name="Qwen3-8B"} 1008.0
"""


class TestParsing:
    def test_reads_both_counters(self):
        counters = parse_prefix_cache_counters(_SAMPLE)
        assert counters.queries == 1440
        assert counters.hits == 1008

    def test_hit_rate_is_blocks_over_blocks(self):
        """vLLM counts both in blocks, which is the pool's own unit, so no
        conversion is needed anywhere."""
        assert parse_prefix_cache_counters(_SAMPLE).hit_rate == pytest.approx(0.7)

    def test_sums_across_engines(self):
        text = (
            'vllm:prefix_cache_queries{engine="0"} 100.0\n'
            'vllm:prefix_cache_queries{engine="1"} 50.0\n'
            'vllm:prefix_cache_hits{engine="0"} 40.0\n'
            'vllm:prefix_cache_hits{engine="1"} 20.0\n'
        )
        counters = parse_prefix_cache_counters(text)
        assert counters.queries == 150
        assert counters.hits == 60

    def test_ignores_the_external_connector_counters(self):
        """`vllm:external_prefix_cache_*` are a different mechanism and must not
        be folded into the GPU-side numbers."""
        text = (
            'vllm:prefix_cache_queries{engine="0"} 100.0\n'
            'vllm:prefix_cache_hits{engine="0"} 40.0\n'
            'vllm:external_prefix_cache_queries{engine="0"} 999.0\n'
            'vllm:external_prefix_cache_hits{engine="0"} 999.0\n'
        )
        counters = parse_prefix_cache_counters(text)
        assert counters.queries == 100
        assert counters.hits == 40

    def test_absent_counters_raise_rather_than_reporting_zero(self):
        """A server without these counters means prefix caching was never on,
        and a silent 0.0 would read as 'measured no hits'."""
        with pytest.raises(ValueError, match="prefix_cache_queries"):
            parse_prefix_cache_counters("# nothing here\n")

    def test_zero_queries_has_no_hit_rate(self):
        counters = PrefixCacheCounters(queries=0, hits=0)
        assert counters.hit_rate == 0.0

    def test_more_hits_than_queries_is_rejected(self):
        with pytest.raises(ValueError, match="hits"):
            PrefixCacheCounters(queries=5, hits=6)

    def test_counters_subtract_for_a_windowed_reading(self):
        """The counters are cumulative, so a scenario measures a delta."""
        before = PrefixCacheCounters(queries=100, hits=40)
        after = PrefixCacheCounters(queries=250, hits=130)
        delta = after - before
        assert delta.queries == 150
        assert delta.hits == 90
        assert delta.hit_rate == pytest.approx(0.6)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_prefix_probe.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'dryrun.workload.prefix_probe'`.

- [ ] **Step 3: Write minimal implementation**

Create `dryrun/workload/prefix_probe.py`:

```python
"""Read vLLM's prefix-cache counters, for validating the simulated block pool.

`vllm:prefix_cache_queries` and `vllm:prefix_cache_hits`
(`third_party/vllm/vllm/v1/metrics/loggers.py:584-601`) are Prometheus counters
measured in **blocks**, which is the simulated pool's own unit. So a measured
hit rate and a simulated one compare directly, with no token conversion in
between and nothing to calibrate.

Both are cumulative since server start, so a scenario reads them before and
after and takes the difference.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass

_SAMPLE_LINE = re.compile(
    r"^vllm:(?P<name>prefix_cache_queries|prefix_cache_hits)\{[^}]*\}\s+(?P<value>[0-9.eE+-]+)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class PrefixCacheCounters:
    """Blocks queried and blocks found, summed across engines.

    Args:
        queries: Blocks looked up.
        hits: Blocks found.

    Raises:
        ValueError: If either is negative, or if hits exceed queries, which
            would mean the reading is not self-consistent.
    """

    queries: int
    hits: int

    def __post_init__(self) -> None:
        if self.queries < 0 or self.hits < 0:
            raise ValueError(f"Counters must be non-negative, got {self!r}.")
        if self.hits > self.queries:
            raise ValueError(
                f"More hits than queries ({self.hits} > {self.queries}), so this "
                f"reading is not self-consistent."
            )

    @property
    def hit_rate(self) -> float:
        """Fraction of queried blocks that were found, 0.0 when nothing was asked."""
        if self.queries == 0:
            return 0.0
        return self.hits / self.queries

    def __sub__(self, other: "PrefixCacheCounters") -> "PrefixCacheCounters":
        """The delta between two cumulative readings."""
        return PrefixCacheCounters(
            queries=self.queries - other.queries,
            hits=self.hits - other.hits,
        )


def parse_prefix_cache_counters(metrics_text: str) -> PrefixCacheCounters:
    """Extract the GPU prefix-cache counters from a Prometheus exposition.

    Args:
        metrics_text: The body of a `GET /metrics` response.

    Returns:
        The two counters, summed over every engine label.

    Raises:
        ValueError: If `vllm:prefix_cache_queries` is absent. That means the
            server never had prefix caching enabled, and reporting 0.0 would
            read as "measured no hits" rather than "measured nothing".
    """
    totals = {"prefix_cache_queries": 0, "prefix_cache_hits": 0}
    seen = set()
    for match in _SAMPLE_LINE.finditer(metrics_text):
        name = match.group("name")
        totals[name] += int(float(match.group("value")))
        seen.add(name)

    if "prefix_cache_queries" not in seen:
        raise ValueError(
            "No vllm:prefix_cache_queries sample in /metrics, so this server was "
            "not running with prefix caching enabled."
        )
    return PrefixCacheCounters(
        queries=totals["prefix_cache_queries"],
        hits=totals["prefix_cache_hits"],
    )


def read_prefix_cache_counters(base_url: str, timeout_s: float = 10.0) -> PrefixCacheCounters:
    """Fetch and parse `{base_url}/metrics`.

    Args:
        base_url: Server root, e.g. `http://127.0.0.1:8000`.
        timeout_s: Socket timeout.

    Returns:
        The counters at the moment of the call.
    """
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=timeout_s) as response:
        return parse_prefix_cache_counters(response.read().decode("utf-8"))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_prefix_probe.py -q && pytest tests/ -q
```

Expected: `tests/test_prefix_probe.py` 8 passed, whole suite green.

- [ ] **Step 5: Commit**

```bash
git add dryrun/workload/prefix_probe.py tests/test_prefix_probe.py
git commit -m "feat(kv): read vLLM's prefix-cache counters from /metrics

Both counters are measured in blocks, which is the simulated pool's own unit,
so measured and simulated hit rates compare with no conversion and nothing to
calibrate. An absent counter raises rather than reporting 0.0, which would read
as 'measured no hits'."
```

---

### Task 10: The four controlled vLLM scenarios

Requires a GPU. Every expected value is registered **before** measuring; nothing here is fitted. Scenario 4 runs first because LRU eviction is otherwise never executed: KV does not saturate under the current workload, so without it that path ships unexercised.

**Files:**
- Create: `examples/profile/vllm/prefix_cache_probe.yaml`
- Create: `examples/profile/vllm/prefix_cache_probe.sh`
- Test: manual, with results recorded in Task 11

**Interfaces:**
- Consumes: `read_prefix_cache_counters` (Task 9), `VllmServer` from `dryrun/workload/inference.py`.
- Produces: a measured-versus-simulated table consumed by Task 11.

- [ ] **Step 1: Write the fixture**

Create `examples/profile/vllm/prefix_cache_probe.yaml`:

```yaml
# Four controlled prefix-cache scenarios. Each row's expected hit count is
# arithmetic over prompt_len and block_size, registered before measuring.
#
# Scenario 4 is listed first deliberately: KV never saturates under the
# project's main workload, so LRU eviction is the one pool path that no other
# scenario reaches.
model: /path/to/Qwen3-8B
block_size: 16
# Never 1. A single-request point carries ~78 ms of server-side work and
# inflated prefill MRE to 46.98% once before; docs/reference/limitations.md
# records "never quote a figure from a num-prompts: 1 point" as a rule.
n: 8

scenarios:
  - name: lru_eviction
    description: Squeeze the pool until reuse must evict, the only path that reaches _maybe_evict.
    gpu_memory_utilization: 0.30
    prompt_len: 2048
    max_tokens: 256
    n_prompts: 64
    expect: hit_rate_falls_with_pressure

  - name: within_group_shared_prompt
    description: One prompt, eight samples. The shared prefix is paid for once.
    gpu_memory_utilization: 0.90
    prompt_len: 2048
    max_tokens: 64
    n_prompts: 1
    expect_shared_blocks: 128      # 2048 // 16
    expect_hit_rate: 0.875         # 7 of 8 requests hit every shared block

  - name: continued_prefix
    description: Round two's prompt is round one's prompt plus its output.
    gpu_memory_utilization: 0.90
    prompt_len: 2048
    max_tokens: 512
    n_prompts: 1
    rounds: 2
    expect_round2_hit_blocks: 160  # (2048 + 512) // 16

  - name: hit_rate_after_reset
    description: reset_prefix_cache, then resend. Hits fall to zero, then recover.
    gpu_memory_utilization: 0.90
    prompt_len: 2048
    max_tokens: 64
    n_prompts: 1
    reset_between: true
    expect_hit_rate_after_reset: 0.875
```

- [ ] **Step 2: Write the driver script**

Create `examples/profile/vllm/prefix_cache_probe.sh`:

```bash
#!/usr/bin/env bash
# Measure vLLM's prefix-cache hit rate in four controlled scenarios, to check
# the simulated block pool against the engine it models.
#
# Expected values live in prefix_cache_probe.yaml and were registered before
# measuring. Nothing here is fitted: a disagreement means the pool's semantics
# are wrong and the logic must change, not that a coefficient needs tuning.
set -euo pipefail

source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh
cd "$(git rev-parse --show-toplevel)"

MODEL="${MODEL:?set MODEL to a Qwen3-8B checkpoint directory}"
PORT="${PORT:-8000}"
OUT="${OUT:-outputs/profile/prefix_cache_probe}"
mkdir -p "$OUT"

python -m dryrun.workload.prefix_probe_run \
  --config examples/profile/vllm/prefix_cache_probe.yaml \
  --model "$MODEL" \
  --port "$PORT" \
  --output "$OUT/measured.json"

echo "Measured counters written to $OUT/measured.json"
echo "Compare against the simulator with:"
echo "  pytest tests/test_kv_instance.py -q   # same arithmetic, no GPU needed"
```

```bash
chmod +x examples/profile/vllm/prefix_cache_probe.sh
```

- [ ] **Step 3: Run scenario 4 first, then 1 through 3**

```bash
MODEL=/path/to/Qwen3-8B bash examples/profile/vllm/prefix_cache_probe.sh
```

Expected: `outputs/profile/prefix_cache_probe/measured.json` holding four entries, each with `queries`, `hits`, and `hit_rate`. Check each against the `expect_*` field in the yaml.

If a scenario disagrees with its registered expectation, **stop and diagnose the pool**. Do not adjust the expectation to match. Record the disagreement and its diagnosis in Task 11's document.

- [ ] **Step 4: Confirm the simulator reproduces the same arithmetic**

```bash
pytest tests/test_kv_instance.py tests/test_kv_pool.py -q
```

Expected: all pass. These encode the same expected hit counts without needing a GPU, so they are the regression guard once the GPU run has happened.

- [ ] **Step 5: Commit**

```bash
git add examples/profile/vllm/prefix_cache_probe.yaml examples/profile/vllm/prefix_cache_probe.sh
git commit -m "test(kv): add the four controlled vLLM prefix-cache scenarios

Expected hit counts are registered in the fixture before measuring. A
disagreement means the pool's semantics are wrong, not that a coefficient
needs tuning. LRU eviction is listed first because KV never saturates under
the main workload, so no other scenario reaches that path."
```

---

### Task 11: Acceptance gates and documentation

**Files:**
- Create: `docs/simulation/prefix-cache.md`
- Modify: `docs/reference/limitations.md`
- Modify: `docs/simulation/index.md`
- Modify: `outputs/simulate/ALIGNMENT_LOG.md`

**Interfaces:**
- Consumes: measured results from Task 10, the simulator from Tasks 1-8.
- Produces: documentation only.

- [ ] **Step 1: Re-run the aligned baseline and record the three gates**

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh
cd /apdcephfs_zwfy/share_303760348/lhy/DryRun-RL
bash examples/simulate/psrl_baseline_aligned.sh
dryrun-compare outputs/simulate/psrl_baseline_aligned outputs/visualize/qwen3_8b_megatron_baseline
```

Record, from the comparison output:

| Gate | Before | After | Target |
|---|---|---|---|
| mean KV util | -24.1% | *measured* | toward 0 |
| re-prefill per interrupt | 8602 against 6003 (1.43x) | *measured* | toward 1.0 |
| peak KV util | +0.1% | *measured* | unchanged |
| mean running busy | -2.0% | *measured* | unchanged |
| span | +3.6% | *measured* | unchanged |

**If mean KV util has not converged toward 0, do not add a coefficient.** The spec registers this in advance: the correct response is to re-examine pool semantics or look for another mechanism. Write the finding down either way; a directional prediction that failed is a result, not an embarrassment.

- [ ] **Step 2: Write the documentation page**

Create `docs/simulation/prefix-cache.md`:

```markdown
# Prefix caching

The rollout engine reuses KV blocks between requests that share a prefix, which
is what the real vLLM does and what the simulator previously did not model at
all. Three measured consequences motivated building it; see
{doc}`../reference/limitations.md` for where each stood before.

## How a block is identified

vLLM hashes a block as `hash(parent_block_hash, block_token_ids)`, so a block
hash means *every token from position 0 to that block's end*. The chaining is
load-bearing: it is what stops `[hello][world]` and `[bye][world]` colliding on
their second block.

We hold no token ids. We hold the provenance that produced them, which answers
the same question, because two contexts agree on a prefix exactly when they
agree on the provenance of that prefix. So a block id is written rather than
computed:

| block `j` covering `[16j, 16j+16)` | id |
|---|---|
| whole block inside the shared prompt | `("p", source_group_id, j)` |
| otherwise | `("r", rid, j)` |

Only full blocks are cacheable, which is vLLM's own rule. At `prompt_len=150`
that makes blocks 0 through 8 shareable (144 tokens) and block 9 private,
because it straddles the end of the prompt.

This is where we diverge from Frontier, which must be handed a
`block_hash_ids` trace column because it replays real requests and never sees
their tokens. Generating our own workload means the trace format stays
untouched.

## What falls out

- **Within a group.** Eight responses to one prompt share `source_group_id`, so
  seven of them skip the shared prefix. At `prompt_len=150` a group drops from
  `8 x 150 = 1200` prefill tokens to `150 + 7 x 6 = 192`.
- **Partial rollout.** A resumed request's generated blocks hash exactly as they
  did when first filled, so re-prefill hits them.
- **Weight sync.** `abort_all` frees every block *and* clears the hash map, in
  that order, mirroring PSRL's `pause_for_sync` then `pull_model`. The post-sync
  hit rate is therefore not zero: the first sibling repopulates the prompt
  hashes and the rest hit them.

## Measured against the engine

Four controlled scenarios, run with `examples/profile/vllm/prefix_cache_probe.sh`.
Expected hit counts were registered in the fixture before measuring, and nothing
is fitted: `vllm:prefix_cache_queries` and `vllm:prefix_cache_hits` are counted
in blocks, which is the simulated pool's own unit, so the two compare directly.

| Scenario | Expected | Measured | Simulated |
|---|---|---|---|
| LRU eviction under pressure | hit rate falls | *fill from Task 10* | *fill* |
| Within-group shared prompt | 0.875 | *fill* | *fill* |
| Continued prefix | 160 blocks | *fill* | *fill* |
| Hit rate after reset | 0.875 | *fill* | *fill* |

## Turning it off

`rollout.enable_prefix_caching: false`. The pool still does block accounting and
simply records no hashes, so nothing can hit. It is a data state, not a second
code path, which is why the disabled case needs no separate validation.
```

Fill the table's `*fill*` cells from Task 10's `measured.json` and from
`pytest tests/test_kv_instance.py -q`.

- [ ] **Step 3: Update the limitations page**

In `docs/reference/limitations.md`, replace the **Prompt prefill charged once per response** entry under "Operational rough edges" with, under "Model approximations":

```markdown
**Prefix reuse is derived from provenance, so it under-reports hits in one
direction.** Block identity comes from `(source_group_id, block_index)` for
prompt-interior blocks and `(rid, block_index)` otherwise, rather than from a
hash of token ids the simulator never holds. Two *different* prompts that happen
to share leading full blocks hit in the real engine and miss here. On RL
workloads distinct problems almost certainly differ at token 0, and every weight
update clears the hash map anyway, so the effect is second order -- but it is
one-directional: this never over-reports a hit. A workload with a shared system
prompt or shared few-shot prefix would amplify it, and the fix at that point is
an optional explicit prefix-grouping key on the trace, not a change to the hash
rule. See {doc}`../simulation/prefix-cache`.

**Within-group reuse cannot be judged on the aligned baseline.**
`extract_workload_trace` chunks completed requests into groups of `group_size`
in request-id order, because the route trace does not record which prompt each
response belonged to. That reproduces the length distribution and group arity
exactly but not the grouping, so a hit rate measured there reflects fabricated
groups. This is why prefix caching is validated against a purpose-built vLLM
experiment instead; see {doc}`../simulation/prefix-cache`.

**No cross-instance reuse, and the router scatters groups.** The block pool is
per-instance, as in the real engine, while `_select_instance` picks the
least-active one, so a group's siblings land on different instances and
within-group reuse is diluted. Frontier makes this a hard gate: multi-replica
prefix caching requires a sticky scheduler or it raises. We do not port that
gate, because what PSRL actually does has not been verified and copying
Frontier's policy would weld in an unverified assumption. Routing is tracked
separately.
```

- [ ] **Step 4: Update the alignment log and the docs index**

In `outputs/simulate/ALIGNMENT_LOG.md`, under "Open gaps, largest first", replace item 1 with the measured outcome from Step 1, naming the commit. If mean KV util converged, say by how much; if it did not, say so and record the diagnosis.

In `docs/simulation/index.md`, add `prefix-cache` to the toctree.

- [ ] **Step 5: Verify docs build, then commit**

```bash
pytest tests/ -q
cd docs && make html 2>&1 | tail -20 && cd ..
```

Expected: suite green, docs build with no new warnings about the added page.

```bash
git add docs/ outputs/simulate/ALIGNMENT_LOG.md
git commit -m "docs: record prefix-cache design, measurements, and limitations

Three limitations stated plainly: provenance derivation under-reports hits in
one direction, the aligned baseline's groups are fabricated so within-group
reuse cannot be judged there, and the router scatters groups so cross-instance
reuse is diluted."
```

---

## Self-Review

**Spec coverage.** Walked each spec section against the plan.

| Spec item | Task |
|---|---|
| P1 pool under `engine/kv/` | 1, 3, 4 |
| P2 provenance-derived hashes | 2 |
| P3 hit pre-sets `prefilled` | 7 |
| P4 `OrderedDict` free queue | 3 |
| P5 `abort_all` splits into free plus reset | 7 |
| P6 default on, disabling is a data state | 8 |
| P7 validate against a purpose-built experiment | 10 |
| P8 register expectations, fit nothing | 10 |
| §3.2 pool invariant (free and hashed at once) | 3 |
| §3.3 full-hit edge case | 4 |
| §3.4 seams, closed form untouched | 5 |
| §4 four scenarios | 10 |
| §4.1 telemetry fields | 7, 8 |
| §4.2 three gates plus no-regression set | 11 |
| §5 D1, D2, D3, D4 limitations | 11 |

**One addition beyond the spec.** Task 6 (undoable pool state) is not in the
spec, and it is a correctness requirement rather than a nicety:
`undo_last_event` is a live path via `simulator.py:358`, and pool state cannot
be reconstructed from `prefilled`. Without it an uncommitted prefill step leaves
the pool and the requests silently disagreeing. Flagged here because the spec
should be amended to mention it.

**Placeholder scan.** The documentation table in Task 11 carries `*fill*` cells,
which are instructions to transcribe Task 10's measured output, not unspecified
work; Step 2 names the source for each. No other placeholders.

**Type consistency.** Checked the names used across tasks: `block_hashes`
(Tasks 2, 4), `BlockPool.cached_block` / `new_blocks` / `touch` / `free_blocks` /
`cache_full_blocks` / `reset_hashes` (Tasks 3, 4), `KvCacheManager.computed_blocks`
/ `can_allocate` / `allocate` / `free` / `reset_hashes` / `blocks_held` /
`peek_hit_tokens` (Tasks 4, 5, 7), `NativeInstance.kv` and
`enable_prefix_caching` (Tasks 5, 7, 8), `StepStat.n_prefill_hit` /
`n_prefill_computed` (Tasks 7, 8), `Request.prefix_hit_tokens` (Tasks 7, 11),
`PrefixCacheCounters.queries` / `hits` / `hit_rate` (Task 9). `peek_hit_tokens`
is introduced mid-Task-7 rather than in Task 4 where the rest of the manager
lives; that is deliberate, since its motivation (not perturbing counters during
a non-mutating estimate) only becomes visible once `_step_time_estimate` needs
it.
