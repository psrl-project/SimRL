# KV cache and prefix caching

Each rollout instance owns a pool of fixed-size KV blocks. The pool is a port of vLLM v1's
block pool rather than a count of tokens, because what the simulator has to reproduce is an
engine's admission, reuse and eviction behaviour under memory pressure, and that behaviour
lives in the mechanism. This page covers how a block is identified, how the pool behaves,
where capacity comes from, and the closed-form exhaustion check the engine relies on.

The code is `srl/component/rollout/engine/kv/` (`block.py`, `free_list.py`, `pool.py`,
`manager.py`, `lineage.py`), with a line-for-line C++ port in
`srl/component/rollout/engine_cpp/` (see {doc}`backends`).

## How a block is identified

vLLM hashes a block as `hash(parent_block_hash, block_token_ids)`, so a block hash means
every token from position 0 to the end of that block. The chaining matters: a block is
reusable only when the whole prefix agrees.

The simulator holds no token ids. It holds the provenance that produced them, which answers
the same question, because two contexts agree on a prefix exactly when they agree on its
provenance. A block id is therefore written rather than computed (`lineage.block_hashes`):

| Block `j`, covering `[B*j, B*j + B)` for block size `B` | Id |
|---|---|
| lies wholly inside the shared prompt (`(j + 1) * B <= prompt_len`) | `("p", source_group_id, j)` |
| anything else | `("r", rid, j)` |

Only full blocks are cacheable, as in vLLM, so the block that straddles the end of the prompt
is private. The shared id uses `source_group_id`, the workload group the lengths came from,
and not `group_id`. `group_id` is monotonic, so a workload shorter than the run would give
the same prompt a new id on wraparound and reuse the real engine achieves would be missed.

There is one known bias, and it only ever under-reports hits: two different prompts that
share leading full blocks hit in the real engine and miss here. On RL workloads distinct
problems almost always differ at token 0, and a weight update clears the hash map anyway.

## What the pool guarantees

- **Free and addressable at once.** A block may sit in the free list and in the hash map at
  the same time. Releasing a block (`ref_cnt` reaching zero) appends it to the free list and
  keeps its hash. A hash is destroyed only when its block is handed to a new owner
  (`new_blocks` calls `_maybe_evict`), never on release.
- **LRU order.** The free list is a doubly linked list with the least recently freed block at
  the head. A hit on a currently free block removes it from the middle. An
  `OrderedDict` would lose that position and silently rewrite eviction order, so the node's
  own links are kept. `KvCacheManager.free` releases a request's blocks in reverse order.
- **One hash per block.** Assigning a hash to a block that already holds one raises.
- **Capacity.** `allocate` returns `None` when the blocks cannot be found, and hit blocks
  that are currently free are discounted from the free count rather than charged twice.
- **Counters.** `queries` and `hits` count blocks, not tokens. `computed_blocks` walks a
  request's hashes and stops at the first miss, because a miss means content diverged and
  later blocks must not be reused. `peek_hit_tokens` answers the same question without
  touching the counters. Routing strategies and the forecast use it.
- **Full hits.** On a full hit the last block is handed back, so at least one token is still
  computed, because the model must run one token to produce logits. This fires whenever a
  prompt is a whole number of blocks.

A hit is applied by advancing `req.prefilled`. Both prefill paths already compute remaining
work as `prompt_len + generated - prefilled`, so the saved tokens drop out of the step's
`n_tokens` and attention term and the cost model needs no special case. The tokens saved
accumulate in `Request.prefix_hit_tokens` (`request.jsonl`), and each engine step records
`n_prefill_hit` and `n_prefill_computed` (`rollout_step.jsonl`).

## When the cache is hit, kept and cleared

Within a group
: responses to one prompt share `source_group_id`, so after the first sibling prefills, the
  others skip the shared prefix blocks.

Preemption
: the victim's blocks are freed with their hashes intact, so its resumed prefill can hit its
  own blocks unless another request has taken them in the meantime.

Weight update in abort mode
: `abort_all` frees every block and then calls `reset_hashes`, mirroring PSRL's
  `pause_for_sync` followed by `pull_model`. Nothing can hit afterwards except what the first
  re-prefilled sibling repopulates. `in_place` mode keeps blocks and hashes.

`cancel(keep_tokens=True)`
: frees blocks but leaves their hashes addressable. This is the path the router uses to
  reclaim a preempted request, and the interface for round boundaries where the next round
  should hit.

The separation of `free` from `reset_hashes` is what makes the last two cases different.

## Turning it off

`rollout.enable_prefix_caching: false`. The pool still does block accounting and records no
hashes, so nothing can hit. It is a data state rather than a second code path.

## KV capacity

`rollout.kv_blocks` is the pool size in blocks and the denominator of every reported
`kv_cache_usage`. The default of 100000 is a placeholder. Leave it and the simulated KV
curve is not comparable with a measured one. Setting `rollout.kv_cache_sizing.model_path`
replaces it with the capacity a real vLLM engine would allocate (`resolve_kv_blocks` in
`srl/cli/main.py`):

| `probe` | Method | Needs a GPU |
|---|---|---|
| `true` (default) | `probe_kv_capacity` starts a real vLLM engine on a cache miss and reuses the answer, cached in `~/.cache/srl/kv_capacity.json` and keyed on the checkpoint fingerprint, TP, memory budget, `max_model_len`, dtype, GPU name, vLLM version, block size, `max_num_batched_tokens` and `max_num_seqs` | yes, on a miss |
| `false` | arithmetic over `config.json` and the checkpoint's weight size (`kv_blocks_from_memory`). Honest only for a dense model, because vLLM's peak activation and non-torch memory are measured at startup and a hybrid model's page alignment cannot be reconstructed | no |

The engine parameters are read from the run rather than configured twice:
`max_model_len = job.max_prompt_length + job.max_output_length` (32768 when either is
unset), `rollout.block_size`, `rollout.token_budget` as `max_num_batched_tokens`,
`rollout.max_concurrency` as `max_num_seqs`, and `rollout_cost.dtype` (bfloat16 by
default). Sizing keys are `gpu_memory_mib`, `gpu_memory_utilization`, `tp_size`, `gpu_name`
(part of the cache key), `parameter_count` and `non_kv_overhead_bytes` (both only for
`probe: false`).

A request whose `prompt_len + target_len` could never fit the pool even when empty makes
`add_request` raise instead of queueing forever.

## The closed-form exhaustion check

During pure decode the engine advances whole segments in closed form, so it must know how
many steps remain until the pool would be exhausted. `NativeInstance._kv_exhaustion_steps`
answers in O(`block_size`) rather than by searching. The running batch's block growth over
`k` steps is periodic-linear: every `block_size` steps each running request takes one more
block, and within a period the increments are a fixed permutation determined by each
request's `prefilled % block_size`. Each phase's overflow point is the first term of an
arithmetic progression above the budget, which is one division, and the minimum over the
`block_size` phases is exact.

Prefix sharing changes exactly one term, the intercept. Sharing is fixed at the prompt and
siblings diverge from their first generated token, so every block a request grows during
decode is private, and the increments survive untouched. The intercept must come from the
pool (`blocks_used`), not from summing `blocks_for(prefilled)`, which would count a shared
prompt once per sibling, report immediate exhaustion, and make the engine preempt and
re-admit the same request indefinitely.

## Rollback

A speculative event may be undone, so every pool mutation (reference counts, hash
assignment, hash-map entries, free-list links, held-block lists, counters) is registered with
a recorder, and a committed path passes a no-op recorder. Only the old values of what an
event touched are stored, so the cost of an undo follows the work the event did, not the
size of the pool. How events are made speculative and undone is described in
{doc}`backends`.
