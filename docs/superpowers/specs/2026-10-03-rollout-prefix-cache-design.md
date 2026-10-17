# Rollout Prefix Cache Design

Date: 2026-10-03
Status: Approved design, pending implementation plan
Scope: Rollout KV cache occupancy and prefix reuse. Agentic multi-turn is a separate spec.

## 1. Context

The simulator accounts for KV cache as a single sum:

```python
def blocks_used(self) -> int:
    return sum(blocks_for(r.prefilled, self.block_size) for r in self.waiting + self.running)
```

Every request owns its blocks outright. There is no block pool, no sharing, no eviction, and
no notion of two requests agreeing on a prefix. The real engine has all four, and two measured
consequences are already recorded in this repository.

**The largest remaining alignment residual names prefix caching as the suspect.**
`outputs/simulate/ALIGNMENT_LOG.md` scores the simulator against the PSRL Qwen3-8B Megatron
baseline:

| metric | aligned | real |
|---|---|---|
| peak KV util | **+0.1%** | 1.000 |
| mean running (busy) | **-2.0%** | 47.1 |
| span (s) | **+3.6%** | 3349.0 |
| mean KV util (busy) | **-24.1%** | 0.563 |

Peak matches and occupancy matches, so the *shape* of the KV curve differs rather than its
scale. The log's own reading: "Prefix caching, which the real engine has and the simulator does
not model, is the obvious suspect: it absorbed 75% of prefill tokens (58,280 -> 14,440
computed) and would flatten early KV growth."

**Re-prefill after a weight update is over-predicted by 1.43x.**
`outputs/simulate/SMG_LOOPBACK_FIX.md`, per interrupt because the two runs cover different
step counts:

| | interrupts | re-prefill tokens | per interrupt |
|---|---|---|---|
| real (fixed gateway, 4 steps) | 122 | 732,380 | **6,003** |
| simulated (10 steps) | 362 | 3,113,981 | **8,602** |

Interrupts per weight update agree (real ~41, simulated ~36), so the gap is in how much each
interrupted request had generated, not in how often interruption happens. That document's own
conclusion: "the simulator has no prefix caching. The real engine still gets a 9% hit rate on
the continued prompt [...] Closing the last 40% means modelling prefix-cache hits on the
re-prefill, which is a genuine next step rather than a calibration tweak."

**A third gap is structural rather than residual.** `docs/reference/limitations.md`, under
operational rough edges: "Prompt prefill charged once per response. When a prompt has `n`
sampled responses, the simulator charges prefill `n` times, once per response, rather than once
per shared prompt." With DAPO's `rollout_n=8` that is 8x the prompt prefill of the real engine.

## 2. What Frontier does, and where we diverge

Frontier (`/apdcephfs_zwfy/share_303760348/lhy/simulator/Frontier`) ships prefix caching, so it
is the reference. Four findings.

**Its block pool is a line-by-line port of vLLM v1, not an analytical approximation.**
`frontier/kv_cache/` is 350 lines: `KVCacheBlock` (ref_cnt plus block_hash), `BlockPool` (LRU
free-block queue, `cached_block_hash_to_block` map, `_maybe_evict_cached_block` on reuse), and
`KVCacheManager` (`get_computed_blocks` / `can_allocate_slots` / `allocate_slots` / `free`). A
hit is the longest common prefix between a request's block hash list and the cache map
(`base_kv_cache_manager.py:64-79`, breaking at the first miss).

**It does not compute hashes. The trace must carry them.**
`_get_request_block_hashes()` is `return request.block_hash_ids`. That field comes from a CSV
column:

```csv
arrived_at,num_prefill_tokens,num_decode_tokens,session_id,block_hash_ids
0.0,32,2,7,11|22
```

and it is a hard gate rather than a silent degradation. With prefix caching enabled,
`base_cluster_scheduler.py:118-156` raises if the request source is not trace replay, if either
column is absent, or if any row leaves one empty. Multi-replica additionally requires a sticky
cluster scheduler (`base_cluster_scheduler.py:104-112`).

**Agentic is a stateful request abstraction with three fields.** `thinking_depth` (round
count), `thinking_round_plans` (per-round prefill and decode token counts), and
`tool_call_latency`. Each intermediate round runs a full prefill-to-decode cycle; on completion
`global_batch_end_event.py:180-195` emits a `ThinkingRoundRequeueEvent` at
`now + tool_call_latency`, which resets ~30 runtime fields, advances the round cursor, and
re-queues with session affinity to the same replica so the next round's prefill hits the prefix
the previous round left behind. Note the intermediate round *frees* its KV
(`_free_request_resources`), so cross-round reuse rests entirely on blocks whose `ref_cnt` has
reached zero while their hash is still mapped. That is also how the real vLLM behaves.

**KV capacity is profiled, not derived.** The paper's Fig. 4 and Table 4 target this directly:
an analytical "total minus weights" model over-reports the admissible budget by 14.10% to
39.73%, while Frontier's single-GPU dummy profile run (weight memory, torch peak increase,
non-torch residency) lands within 1.89%.

### 2.1 The divergence: we hold the lineage, so we derive the hash

Frontier must be fed `block_hash_ids` because it replays real requests and never sees their
tokens. We generate our own workload, so the provenance of every token is known.

vLLM hashes a block as `hash(parent_block_hash, block_token_ids)`
(`third_party/vllm/vllm/v1/core/kv_cache_utils.py:617-623`). The chaining is what makes a block
hash mean *every token from position 0 to this block's end*, so that

```
A: [hello] [world]
B: [bye]   [world]
```

do not collide on block 1. A block is reusable only when the whole prefix agrees, and the chain
is what enforces that.

We hold no token ids, but we hold something that answers the same question: two contexts agree
on a prefix exactly when they agree on its provenance. So the hash is written rather than
computed, from two integers already on `Request`:

| block `j` covering `[Bj, Bj+B)` | hash |
|---|---|
| `B(j+1) <= prompt_len` (whole block inside the shared prompt) | `("p", req.source_group_id, j)` |
| otherwise (straddles the boundary, or sits in generated tokens) | `("r", req.rid, j)` |

Only full blocks get an id, which is vLLM's own rule: a partially filled block is not
cacheable. At `prompt_len=150, B=16` that means blocks 0 through 8 (144 tokens) are shareable
and block 9 is private, because it straddles the end of the prompt.

**`source_group_id`, not `group_id`.** `group_id` is a monotonic key by design: a workload
shorter than the run repeats, so non-monotonic ids would merge two distinct groups. But
monotonicity means the *same prompt* receives a *new* `group_id` on wraparound, and we would
then miss reuse the real engine achieves. `source_group_id` records which workload group the
lengths came from, which is the prompt's actual identity. `request.py:62` currently annotates it
"Provenance only"; this gives it a second use.

**Three scenarios fall out with no further mechanism.**

- *Within a group.* Eight siblings share `source_group_id`, so their prompt-interior hashes
  agree. The first pays, the other seven hit. A group with `prompt_len=150` drops from
  `8 x 150 = 1200` prefill tokens to `150 + 7 x 6 = 192`.
- *Partial rollout resumption.* After an abort, `generated` survives and `prefilled` is zeroed,
  so the re-prefilled context is `prompt + generated`. The generated blocks still hash to
  `("r", rid, j)`, identical to what that request wrote when it first filled them, so they hit.
- *Agentic next round.* Same request id continuing, structurally identical to the previous case.
  This is why the agentic spec can reuse this layer unchanged.

**A falsifiable prediction, registered in advance.** A weight update both frees every block and
clears the hash map (PSRL's `pause_for_sync` reaching an abort, then `pull_model` calling
`reset_prefix_cache`; `instance.py:558-582` already documents the sequence). After the clear,
the first sibling re-prefills and repopulates the map, and the rest hit on the prompt segment
only. So the post-sync hit rate is not zero but a small value set by
`prompt_len / continued_context_length`. Worked through at `prompt_len=150` with ~3000 generated
tokens: `144 / 3000` is about 5%. The real run measures **9%**
(`SMG_LOOPBACK_FIX.md`: 197,097 prefill tokens, 179,193 computed). Same order, and there is no
coefficient to turn: the number is arithmetic over `prompt_len`, `block_size`, and the length
distribution.

## 3. Design decisions

| # | Decision | Rationale |
|---|---|---|
| P1 | Pool lives under `component/rollout/engine/kv/`, not under `cost/` | It is a state machine, not a cost model: it fits no coefficient, reads no artifact, has no domain gate |
| P2 | Derive block hashes from provenance; do not add a trace column | We generate the workload, so the lineage is known. Costs one one-directional bias (D1) and keeps the trace format untouched |
| P3 | Express a hit by pre-setting `prefilled`; add no `num_computed_tokens` field | Both prefill paths already compute `prompt_len + generated - prefilled`, so the saved tokens leave `t2` and `n_tokens` automatically. Zero change to the cost model |
| P4 | Keep vLLM's hand-written doubly linked free list rather than an `OrderedDict` | The `prev`/`next` pointers a mid-list removal preserves *are* the minimum information an undo needs. `OrderedDict.pop(key)` discards position, so a revived hit block could never be put back where it was. See P9 |
| P5 | Split `abort_all` into `free` plus `reset` | A weight update clears hashes; an agentic round boundary must not. Same mechanism, two call sites |
| P6 | Default `enable_prefix_caching=True`; disabling is a data state, not a bypass branch | The default should match the simulated system's default. The pool still does block accounting; `computed_blocks` just returns empty |
| P7 | Validate against a purpose-built vLLM experiment, not against the aligned baseline | That baseline's groups are fabricated (D2), so within-group reuse cannot be judged there |
| P8 | Register every expected hit count before measuring; fit nothing | There is nothing to fit. The experiment decides right or wrong, not a coefficient |
| P9 | Undo pool mutations by storing old values for touched blocks only, never by snapshotting the pool | `copy.deepcopy` of a 31279-block pool measures **179 ms**, which over 4949 steps across 2 instances is **1771 s** against a 3349 s run. Capturing the free-list order per event is 3.97 s, better but still proportional to capacity. Only the touched-block list is proportional to work: median **2** blocks per prefill step |

### 3.1 Module layout

```
srl/component/rollout/engine/kv/
    block.py      KvBlock: block_id, ref_cnt, block_hash, free-list links
    free_list.py  FreeBlockList: the doubly linked LRU queue
    pool.py       BlockPool: allocation, hash -> block map, eviction
    manager.py    KvCacheManager: computed_blocks / can_allocate / allocate / free
    lineage.py    block hash derivation (ours; Frontier is fed this by the trace)
    undo.py       BlockSnapshot: old values for the blocks one event touched
```

`cost/rollout/kv_cache.py` keeps its single job, *how many blocks exist*. The new module owns
*who holds them and what can hit*. The only interface between them is the `kv_blocks` integer.

### 3.2 The pool invariant

Four pieces of state: `blocks` (index is block id), a doubly linked free list of blocks whose
`ref_cnt` is zero (head is least recently freed, so head is the eviction victim),
`hash_to_block`, and `eviction_count`.

The invariant that makes the whole thing work: **a block may be in the free list and in
`hash_to_block` at the same time.** That is the "nobody holds it, but its hash is still
mapped, so it can still hit" state, and every cross-sync and cross-round reuse depends on it.
A hash is destroyed only when its block is handed to someone else (`get_new_blocks` reaching
`_maybe_evict_cached_block`), never on release.

Retain Frontier's setter assertion that raises when a block hash is assigned twice: it catches
one physical block being treated as two logical ones, which is among the hardest classes of bug
to find here.

### 3.3 One vLLM edge case to port verbatim

`vllm_v1_engine_replica_scheduler.py:1393-1399`: when every block hits, hand the last one back.
A full hit leaves `num_new_tokens == 0`, but the model must forward at least one token to
produce the next logits. This branch fires for real in our workload whenever a group's
`prompt_len` is a multiple of `block_size`; without it the engine produces a zero-token step.

### 3.4 Undoing pool mutations

`undo_last_event` (`instance.py:80`) is a live path, not a safety net:
`advance_one_event` is *speculative*. The driver advances every instance by one event, commits
the earliest, and rolls the rest back (`simulator.py:388-396` seeding, `:356-360` undoing). That
is how instances share one global clock. So every pool mutation has to be reversible.

Today `InstanceSnapshot` captures `prefilled` per request, which suffices only because
occupancy is *derived* from it. A pool holding refcounts, a hash map, and a free list breaks
that: an uncommitted prefill step would roll `prefilled` back while leaving pool state mutated,
and the two would disagree with nothing downstream to notice.

The existing snapshot code already establishes how to do this, and states the rule in its own
comment (`snapshot.py:78-83`): **capture cost must be proportional to the live batch, never to
the accumulated run.** Three measured candidates against that rule:

| Approach | Cost per event | Over the aligned run (4949 steps, 2 instances) | Proportional to |
|---|---|---|---|
| `copy.deepcopy(pool)` | 179 ms | **1771 s** against a 3349 s run | capacity (31279) |
| capture free-list order | 401 us | 3.97 s | capacity (31279) |
| **touched-block old values** | median 2 entries | negligible | **work** |

Only the third obeys the rule. It is also the same shape as `ReqSnapshot`: store the old value,
assign it back. No write-ahead log, no inverse operations, no replay — those would introduce a
second mental model into a module that currently has exactly one.

Per event, store one entry per block the event touched:

```python
@dataclass
class BlockSnapshot:
    ref_cnt: int
    block_hash: Hashable | None
    prev_free: KvBlock | None   # position in the free list
    next_free: KvBlock | None
```

`prev_free` / `next_free` are why P4 reverses the `OrderedDict` decision. A hit on a currently
free block removes it from the middle of the list; restoring it means re-linking it exactly
where it was. The linked list carries that position in the node itself, so the "old value" is
already there for the taking. An `OrderedDict` discards it on `pop(key)`, and a block put back
at the tail instead of its original position leaves occupancy *correct* while silently
rewriting LRU order — invisible to `blocks_used()` and fatal to eviction order, which is
exactly what validation scenario 4 measures.

Three further pieces of state, each handled by the existing playbook:

- `hash_to_block` additions: store the keys added, delete them on undo.
- `eviction_count`, `queries`, `hits`: plain scalars, stored and assigned back. These must be
  rolled back too, or an uncommitted event permanently inflates counters that are compared
  against `vllm:prefix_cache_queries` and reported as eviction pressure.
- Lifetime: the touched-block list hangs off the same `_save_snapshot()` / `undo_last_event()`
  pair that already exists. No new lifecycle concept.

**One asymmetry worth stating, because it is real rather than an implementation shortcut.** A
decode segment's pool changes are fully determined by `(rid, prefilled_before, k)`, which
`ReqSnapshot` already stores, because decode grows only private blocks and never hits. A
prefill step is not recomputable that way: it *hits*, and a hit depends on what the hash map
held at that moment, which is global state. The uniform touched-block mechanism covers both, so
this is not exploited — but it is why a "just recompute everything" design cannot work.

### 3.5 Seams in `NativeInstance`

Two methods change body, not signature:

```python
def blocks_used(self)  -> self.kv.num_used_blocks
def free_blocks(self)  -> self.kv.num_free_blocks
```

plus a hit lookup ahead of prefill in `add_request` and `_prefill_budget_step`.

**`_kv_exhaustion_steps()` is not touched.** Its closed form (`instance.py:234-278`) rests on
`blocks(k + B) == blocks(k) + len(running)`. Prefix sharing happens only in the prompt, and
siblings diverge from their first generated token, so every block a request grows during decode
is private. Sharing shifts the `base` constant and leaves the residue and increment structure
intact, so the periodic-linear derivation stays exact. No fallback to an O(cap) search.

## 4. Validation

Four controlled vLLM scenarios on H20. `srl/workload/inference.py` already starts and stops
a server and drives `/v1/completions`; this adds a `/metrics` reader for
`vllm:prefix_cache_queries` and `vllm:prefix_cache_hits`
(`third_party/vllm/vllm/v1/metrics/loggers.py:584-601`). Both counters are **per block**, which
is the pool's own unit, so no conversion is needed.

| # | Scenario | Method | Expected, registered in advance |
|---|---|---|---|
| 1 | Within-group shared prompt | one prompt at `n=8` | shared full blocks `= prompt_len // 16`; first request all miss, other seven all hit, so hits/queries ~ 7/8 |
| 2 | Continued prefix | second round's prompt is round one's prompt plus its output | hits ~ full blocks of round one's context. Exercises **hit on a block whose `ref_cnt` reached zero** |
| 3 | Hit rate after a clear | call `reset_prefix_cache`, resend | hits drop to zero, then recover to ~ `prompt_len / context` as siblings repopulate |
| 4 | LRU eviction | squeeze `gpu_memory_utilization` or raise concurrency until blocks run short | hit rate falls with pressure. The **only** scenario that reaches `_maybe_evict_cached_block` |

**Scenarios 1 and 2 use `n >= 8`, never `num-prompts: 1`.** `limitations.md` records this as a
rule after a single-request point carried ~78 ms of server-side work and inflated prefill MRE to
46.98%, against 3.98% once re-measured at the same shape: "never quote a figure from a
`num-prompts: 1` point."

**Scenario 4 runs first despite looking the most marginal.** KV never saturates under the
current workload, so `_maybe_evict_cached_block` is a path the main line will never execute.
Without scenario 4 it ships unexecuted, and LRU order is precisely what decides outcomes in the
memory-tight configurations (smaller cards, longer agentic sessions) that justify having a
simulator at all.

Deliverables: a fixture beside the existing `examples/profile/vllm/*.yaml`, and a measured
versus simulated table in `docs/`.

### 4.1 Telemetry

`StepStat` gains `n_prefill_hit` and `n_prefill_computed`, named to match the real log's own
columns (`prefill_tokens` / `computed (cache miss)` in `SMG_LOOPBACK_FIX.md`) so
`srl-compare` puts like against like. `Request` gains a cumulative `prefix_hit_tokens`, which
splits the re-prefill residual into "tokens the cache saved" and "what still fails to reconcile".

### 4.2 Acceptance gates

| # | Gate | Current | Expected | Why it is trustworthy |
|---|---|---|---|---|
| 1 | Four controlled scenarios | — | match measured | Expectations registered in advance; mechanism-level |
| 2 | mean KV util | **-24.1%** | converges toward 0 | Per-request, independent of the fabricated grouping |
| 3 | re-prefill per interrupt | 8602 against 6003 (**1.43x**) | converges toward 1.0 | As above |

Must not regress: peak KV util (`+0.1%`), mean running busy (`-2.0%`), span (`+3.6%`).

Gate 1 is the real gate. Gates 2 and 3 are directional predictions measured on
`psrl_baseline_aligned`, whose grouping is fabricated (D2), so within-group reuse is not
credible there.

**A risk owned up front.** Gate 2's direction cannot be proven a priori. Saving prefill flattens
early KV growth, which is the sign the -24.1% needs, but it also advances requests faster, which
can pile occupancy up sooner. The two effects oppose. Our judgement is that the first dominates,
because the real run absorbed 75% of prefill tokens, but that is a judgement and not a
derivation. **If mean KV util does not converge, the correct response is to re-examine the pool
semantics or look for another mechanism, not to introduce a coefficient that pushes the number
into place.** This sentence is in the spec so that there is no retreat available later;
`docs/profiling/methodology.md` names that retreat as a failure mode.

## 5. Limitations

**D1. Provenance derivation under-reports hits, in one direction only.** Two *different* prompts
that happen to share leading full blocks hit in the real engine and miss here. Second-order on
RL workloads, where distinct problems almost certainly differ at token 0 and every weight update
clears the map anyway, but it is one-directional and will be amplified by workloads with a shared
system prompt or shared few-shot prefix. The fix at that point is an optional explicit prefix
grouping key on the trace, not a change to the hash rule.

**D2. The aligned baseline's groups are fabricated, so it cannot judge within-group reuse.**
`psrl_log.py:514-516` chunks completed requests into groups of `group_size` in request-id order
and takes `prompt_len` from the first member: "Grouping is not recorded in the trace [...] That
reproduces the length distribution and the group arity exactly; it does not reproduce which
prompt each response belonged to." This is why P7 exists.

**D3. No cross-instance sharing, and the router scatters groups.** The pool is per-instance, as
in the real engine, while `_select_instance` picks least-active (`simulator.py:175-177`), so a
group's eight siblings land on different instances and within-group reuse is diluted. Frontier
makes this a hard gate (multi-replica prefix caching requires a sticky scheduler, else it
raises; `base_cluster_scheduler.py:104-112`). **We do not port that gate and do not change
routing here**: what PSRL actually does has not been verified, and copying Frontier's sticky
policy would weld in an unverified assumption. This spec delivers a correct pool and makes the
dilution visible in telemetry. Routing is a separate piece of work driven by measurement.

**D4. Capacity remains an analytical estimate, and is out of scope here.**
`cost/rollout/kv_cache.py` returns 32205 against a measured 31279, **+3.0%**. Frontier's paper
criticises analytical KV budgeting at 14.10% to 39.73%, but our residual sits outside that
critique: `non_kv_overhead_bytes` folds in a known per-hardware constant (~2 GiB on H20), and
MoE and MLA are refused at configuration time rather than approximated.

## 6. Out of scope

- **Agentic round chains.** Separate spec. The `free` / `reset` split (P5) is the interface left
  for it.
- **Routing policy** (sticky against least-active). See D3; needs measurement first.
- **Cross-instance and hierarchical KV cache.** Frontier has hierarchical caching; we have not
  yet validated a single tier.
