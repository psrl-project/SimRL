# Routing

A request is admitted into the **router**, not onto an instance. The router holds it until
some instance can take it, and decides which one. That is where a real gateway puts its
backlog, and it is why the simulated engine's waiting queue now means what the real one's
does: work an instance has accepted, rather than the whole fleet's backlog.

Before this existed, instance selection was three lines of `argmin(n_active)` and a group that
any instance refused was rolled back wholesale. There was nowhere to park a request, so the
backlog went into the engine: the simulated run had a non-empty instance waiting queue in
**99.6%** of samples against the real run's 70.0%.

## Picking a strategy

```bash
dryrun rollout.router.strategy=kv_cache_aware rollout.n_instances=4
```

| `strategy` | Rule |
|---|---|
| `request_num_balance` | Fewest in-flight requests. **The default**, and what the matched PSRL run uses |
| `round_robin` | Cycle through eligible instances |
| `random` | Uniform, seeded via `strategy_args.seed` |
| `kv_cache_aware` | Most KV headroom, counting committed-but-unallocated context |
| `prefix_aware` | Whoever already holds this request's prefix, unless the fleet is skewed |
| `throughput_balance` | Smallest predicted increase in step latency |

`request_num_balance` reproduces the former selection exactly, so the default is
behaviour-preserving: at `n_instances=1` every strategy must produce an identical run, which
`tests/test_router_integration.py` asserts.

Adding a strategy means one class in `component/rollout/router/strategies.py` under
`@register_strategy`, and nothing else. A strategy is pure selection over `InstanceView` and
never queues, checks capacity, or mutates state, so it can be unit tested without an engine.

### Two warnings

**`throughput_balance` has never been matched against a measurement.** It is the only strategy
that consults the cost model, and no run has been validated under it. Offered, not recommended.

**`kv_cache_aware` needs more than free blocks, and getting that wrong fails silently.** A
request queued on an instance has allocated nothing yet, so two instances holding very
different commitments report *identical* free blocks. Reading that alone made the strategy
return the first instance every time and pile an entire batch onto it while the other sat
empty, with the run completing normally and only the figure revealing it. The strategy
subtracts `committed_blocks`, the same quantity `add_request` reserves on, so it and the
admission gate agree about what "full" means.

## Admission and dwell

Two layers, kept separate on purpose:

The **router's gate** decides *eligibility* and mirrors SMG's `admission_reject_reason`:
`max_concurrent_seqs_per_instance` (0 disables it) and `reject_on_waiting`. Both default to
permissive.

The **engine's `add_request`** remains the authority on whether the request actually fits. It
is not duplicated in the router: two places deciding admission is how the two drift apart.

A request whose candidates all refuse stays queued. Dispatch walks the queue in priority order
and **stops at the first request nothing can take** rather than skipping it, which is SMG's
peek-then-pop shape and what stops a long request being starved by a stream of short ones that
keep fitting into the gap it needs.

Queue order is a min-heap on `(version, sort_key, arrival)`. Lower weight versions always go
first, because an older trajectory is closer to expiring. `sort_key` breaks ties:
`short_length` (default), `long_length`, or `small_id`. The key uses the **original** prompt
length even after a loopback, so a resumed request does not drift backwards every time it
returns.

## Loopback

Two triggers, two different rules, because the real system has two.

**A weight update** aborts everything. The requests go back to the *router*, not to the
instance that was serving them, and are placed again. In the matched run **224 of 383** aborted
requests (58.5%) landed on a *different* instance, so pinning them in place is not a neutral
simplification. This is also what puts the spike in the router queue row.

**A preemption** keeps the victim local, at the head of its own instance's queue, exactly as
vLLM does. It only reaches the router when that instance's queue has run deeper than
`rollout.router.max_waiting_after_preempt`, which is vLLM's own
`preemption_notification_threshold`: requeueing into an already-congested instance only makes
it worse. Below the threshold nothing crosses instances, which keeps the common case cheap —
there were 2122 preemptions in the matched run against 404 aborts.

`max_waiting_after_preempt` defaults to `null`, leaving every preemption local.

```{warning}
The preempt-overflow path is **not validated by the aligned baseline**. That run sets the
threshold to 1024 while its instance queues peaked at 218, so the path never fired and all 383
of its loopbacks are weight-update aborts. The path is built against the vLLM source and
covered by its own test, but no measurement confirms it.
```

## When the router runs

A real gateway polls on a timer. A discrete-event simulator has no clock to poll, so the
question has to be answered structurally, and one fact settles it: **a dispatch can only
succeed when KV capacity was just released**, because `add_request`'s binding gate is a block
count and nothing else in the attempt varies with time. A dispatch at any other moment returns
exactly what the previous one did.

Capacity is released in three places, all already event boundaries:

| Release | Event | Count in the matched run |
|---|---|---|
| A finished request's blocks are freed | `REQUEST_COMPLETE` | 1394 |
| A preemption frees its victim | `KV_PREEMPT` / `DEADLOCK_PREEMPT` | 2122 |
| A weight update frees everything | publish | 404 |

So dispatch runs after those commit, plus once on entry to the engine loop. **A decode segment
triggers nothing**: it only grows blocks. That matters more than it sounds, because one
closed-form segment covers a median **122** engine steps, and the run has 432 decode segments
against 108,411 prefill steps. Routing per engine step would be 28x the work for decisions
that cannot differ.

This is not an approximation of a polling loop. A segment ends at the first completion or KV
exhaustion across the batch, which is precisely the first instant capacity can change, so the
trigger set is **exact** given the engine's event granularity. There is deliberately no
`check_interval_ms` analogue: it would be a knob with nothing behind it.

`tests/test_router_integration.py` asserts the claim rather than trusting it, by checking that
free blocks never increase except at those events. A fourth release path added later fails
there instead of silently leaving the router blind.

### Why the router needs no rollback

`advance_one_event` is speculative: the driver advances every instance, commits the earliest,
and undoes the rest. Engine mutations therefore record into an undo log. Router mutations do
not, because the three trigger points all sit *after* an event committed and after every other
instance's pending event was rolled back.

The two requirements are the same requirement: a dispatch needs released capacity to be able to
succeed, and released capacity is only real once the event that released it committed. Routing
after a committed event is both the only useful moment and the only safe one.

`Router.dispatch` asserts no instance holds an open snapshot, so a future call site that
violates this fails loudly at the boundary rather than corrupting occupancy quietly.

## Group stickiness

`rollout.router.enable_group_sticky` pins a prompt group's responses to the instance that
accepted the first one, concentrating a group's KV so its siblings share the prompt prefix.

It defaults to **false**, which is what the matched run does: it scatters all 160 groups across
both instances, 4/4 every time. The pin is *soft* — if the pinned instance cannot take the
request it falls through to the full candidate set, because a hard pin would dwell a request
behind one busy instance while others idle. Pins are released when the group is consumed or
dropped.

## What this changed elsewhere

`job.batch_size` and admission are unaffected: what bounds admission is the policy's own quota,
as it always was. The engine refusal was a secondary gate that produced rollback churn.

Two defects the new queue exposed, both fixed:

- `_discard` matched requests to instances by `instance_id`. A queued request sits on no
  instance, so it was dropped from accounting while still in the queue, and a later dispatch
  would have placed work already written off.
- The engine loop broke out to refill only when the policy granted quota. With a saturated
  policy and a non-empty queue, a completion that freed capacity would not drain it until the
  next training step.

A third was latent and is now closed: an instance's undo snapshot was cleared only on *undo*,
never on commit, so it outlived its event. `commit_last_event` closes it, which both stops the
log growing and makes "is this instance mid-event" an answerable question.
