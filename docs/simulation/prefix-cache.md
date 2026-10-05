# Prefix caching

The rollout engine reuses KV blocks between requests that share a prefix, which
is what the real vLLM does and what the simulator previously did not model at
all. Three measured consequences motivated building it; see
{doc}`../reference/limitations` for where each stood before.

## How a block is identified

vLLM hashes a block as `hash(parent_block_hash, block_token_ids)`
(`third_party/vllm/vllm/v1/core/kv_cache_utils.py:617-623`), so a block hash
means *every token from position 0 to that block's end*. The chaining is
load-bearing: it is what stops `[hello][world]` and `[bye][world]` colliding on
their second block, because a block is only reusable when the whole prefix
agrees.

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

`source_group_id` rather than `group_id` is deliberate. `group_id` is monotonic
by design, so a workload shorter than the run gives the same prompt a new id on
wraparound and reuse the real engine achieves would be missed.
`source_group_id` records which workload group the lengths came from, which is
the prompt's actual identity.

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
- **Agentic round boundaries.** `cancel(keep_tokens=True)` frees blocks but
  leaves their hashes addressable, so the next round hits. Separating `free`
  from `reset` is what makes that possible, and is the interface the agentic
  work builds on.

## The pool

`dryrun/component/rollout/engine/kv/` is a line-by-line port of vLLM v1's block
pool rather than an approximation, because what it has to reproduce is an
engine's admission and reuse behaviour under pressure, and that behaviour lives
in the mechanism.

The central invariant: **a block may sit in the free list and in the hash map at
the same time.** That is the "nobody holds it, but it is still addressable"
state, and reuse across a weight update and across an agentic round boundary
both land on exactly those blocks. A hash is destroyed only when its block is
handed to a new owner, never on release.

The free list is a hand-written doubly linked list, not an `OrderedDict`. A hit
on a currently-free block removes it from the *middle* of the queue, and undoing
that means putting it back exactly where it was; the node's own `prev_free` /
`next_free` are that position. An `OrderedDict` discards it on `pop(key)`, and
re-inserting at the tail would leave occupancy correct while silently rewriting
LRU order -- invisible to `blocks_used()` and decisive for every later eviction.

## The closed-form exhaustion check

`_kv_exhaustion_steps` answers "how many decode steps until the pool is full" in
O(block_size) rather than by searching. Its structure is a residue-bucketed
arithmetic progression: blocks grow by `len(running)` every `block_size` steps,
and within a period the increments are a fixed permutation, so each phase
resolves in one division.

Prefix sharing changes exactly one term in it -- the intercept. The increments
survive untouched, and for a reason worth stating: sharing is fixed at the
prompt, siblings diverge from their first generated token, so **every block a
request grows during decode is private**. `blocks_for(p + k) - blocks_for(p)` is
therefore still the pool's true growth for that request.

The intercept must come from the pool, not from summing `blocks_for(prefilled)`.
That sum over-counts a shared prompt by the group size -- 896 against a real 134
for eight siblings at `prompt_len=2048` -- and the failure mode is not a small
error. The estimate returns 0, the engine preempts, the victim is re-admitted,
and the loop repeats forever with **777 of 1000 blocks free**. Reverting to the
derived sum reproduces it.

The estimate is validated against brute force over randomised shapes, sharing
and not, with zero mismatches.

## Rollback

`advance_one_event` is speculative: the driver advances every instance by one
event, commits the earliest, and discards the rest. So every pool mutation must
reverse, which `dryrun/component/rollout/engine/undo/` provides.

Three candidates were measured against the rule the original snapshot code
already states (`snapshot.py`): capture cost must be proportional to the live
batch, never to the accumulated run.

| Approach | Cost per event | Over the aligned run (4949 steps, 2 instances) | Proportional to |
|---|---|---|---|
| `copy.deepcopy(pool)` | 179 ms | **1771 s** against a 3349 s run | capacity (31279) |
| capture free-list order | 401 us | 3.97 s | capacity (31279) |
| **touched-block old values** | median 2 entries | negligible | **work** |

Only the third obeys the rule, and it is the same shape as the request snapshot
it replaced: store the old value, assign it back. No write-ahead log, which
would introduce a second mental model into a module that has one.

`InstanceSnapshot` is now a holder for one `UndoLog`, so a future subsystem
needing rollback implements the `Undoable` protocol rather than extending a
dataclass that knows every subsystem's fields.

## Measured against the engine

Four controlled scenarios, run with
`examples/profile/vllm/prefix_cache_probe.sh` on one H20 against Qwen3-8B. Every
expectation was registered in the fixture before anything was measured, and
nothing is fitted.

The two sides are compared as **hit rates**, not raw counts. In the vLLM version
vendored here `vllm:prefix_cache_queries` and `vllm:prefix_cache_hits` count
*tokens* (`PrefixCacheStats.record(num_tokens, num_hits, ...)`), while the
simulator's `KvCacheManager` counts *blocks*, because that is the unit its pool
allocates in. A rate is unit-free, so it compares directly; a count would need a
conversion that the partial trailing block makes inexact. Older vLLM releases
counted blocks here, so the unit cannot be assumed from the metric name.

Each scenario exercises a different pool path. Measured on one H20 against
Qwen3-8B at `block_size=16`, prompts of ~2048 tokens:

| Scenario | Pool path | Measured | Simulated | Delta |
|---|---|---|---|---|
| `within_group_shared_prompt` | hit on a block another live request holds | 0.8730 | 0.8682 | -0.0049 |
| `continued_prefix` | hit on a block whose `ref_cnt` reached zero | 0.9978 | 0.9697 | -0.0281 |
| `hit_rate_after_reset` | `reset_prefix_cache`, then repopulate | 0.8730 | 0.8682 | -0.0049 |
| `lru_eviction` | `_maybe_evict` under capacity pressure | 0.0000 | 0.0000 | +0.0000 |

Three of the four land within half a percentage point. `continued_prefix` is the
outlier at 2.8 points, and the direction is informative: the simulator
under-reports, because it hands back the last block of a full hit (one block out
of 128) while the real engine's partial trailing block makes its own ratio
slightly higher. Both effects are block-quantisation artefacts of a single
request's measurement, not a semantic disagreement.

`hit_rate_after_reset` matching `within_group_shared_prompt` exactly, on both
sides, is the substantive result: after a reset the first request repopulates the
hashes and its siblings hit them, so the rate recovers rather than collapsing to
zero. That is the mechanism behind the 9% the real PSRL run measures post-sync.

`lru_eviction` runs first despite looking the most marginal: KV never saturates
under the project's main workload, so it is the one scenario that reaches
eviction at all. Without it that path would ship unexercised, and LRU order is
precisely what decides outcomes in the memory-tight configurations that justify
having a simulator.

## On an aligned PSRL run

Measured on a batch-8 PSRL configuration (since retired in favour of the
batch-32 `examples/simulate/psrl_aligned.sh`, which saturates the KV cache),
caching saved **102,416 prefill tokens**, 3.2% of the uncached prefill volume,
and shortened the simulated span by 4.9 s.

With caching *off* the run is **bit-identical to the commit before this work**
across every rollout aggregate -- step count, total tokens, prefill tokens, and
end time all match to the digit. That is the control that matters: the block pool
replaced a derived occupancy sum without changing any behaviour of its own.

Two cautions about this baseline, both recorded because they bound what it can
show:

- It needs `EXTRAPOLATE=[tokens,activation_bytes]` on the machine this was run on.
  The gate fires at `tokens=20673` against a profiled ceiling of 20480, with
  caching both on and off and at *exactly* the same value, so it is a
  training-domain limit of the baseline rather than anything prefix caching
  caused.
- Its KV figures no longer reproduce the numbers in
  `outputs/simulate/ALIGNMENT_LOG.md` (peak KV util -10.8% here against +0.1%
  recorded there). Re-running the prior commit reproduces the -10.8% exactly, so
  the difference predates this work. It is not diagnosed here.

The 3.2% saving is also far smaller than the within-group arithmetic alone would
suggest, and the reason is {doc}`../reference/limitations`'s third item: the
router scatters a group's eight siblings across instances, so most of the shared
prompt is never co-resident. Fixing that is routing work, not pool work.

## Turning it off

`rollout.enable_prefix_caching: false`. The pool still does block accounting and
simply records no hashes, so nothing can hit. It is a data state, not a second
code path, which is why the disabled case needs no separate validation.
