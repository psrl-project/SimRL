# Routing

A request is first **staged in the router**. It becomes admitted only when the router
successfully places its first group member on an instance. That is where a real gateway puts
its backlog, and it prevents queue dwell from consuming staleness-policy inventory or fixing
the trajectory version too early.

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
| `version_aware_tp` | Soft preference for high TP on stale requests |
| `version_capped_tp` | Stale requests use highest TP under the active cap |
| `scg_admission` | Places only when the request's marginal value beats the dilution it causes |
| `length_aware` | Orders whole prompt groups longest- or shortest-first inside a bounded block, places by predicted remaining load. Reads a `LengthPredictor` from `dryrun/forecast/length.py`, so it runs on the oracle or on a noisy one |

`request_num_balance` reproduces the former selection exactly, so the default is
behaviour-preserving: at `n_instances=1` every strategy must produce an identical run, which
`tests/test_router_integration.py` asserts.

Adding a strategy means one module in `component/rollout/router/strategies/`, with
one class under `@register_strategy`, and an import in that package's `__init__.py`.
Most strategies only select over `InstanceView`. A strategy may also narrow the
candidate set for one queue head and decide whether dispatch may continue to the
next version queue when that head cannot be admitted.

### Strategy hooks beyond `select`

All are optional and duck-typed, so a strategy implements only what it needs.

| Hook | Called | Used by |
|---|---|---|
| `narrow_candidates` | before `select`, per request | `version_capped_tp` |
| `allow_version_bypass` | after a version head cannot be placed | `version_capped_tp` |
| `select_context` | instead of `select`, with the clock | `scg_admission` |
| `rank_queued(window, now)` | per placement, over `rollout.router.queue_window` queued requests of one version | `scg_admission`, `length_aware` |
| `observe_enqueue(reqs)` | when a prompt group is first queued | `length_aware` |

`observe_enqueue` exists because admission is per group while placement is per request. A rule keyed
on the group, such as its slowest member, has no other moment at which every member is known to be
queued together. `queue_window` (default 32) bounds how many requests `rank_queued` sees; a strategy
ordering the whole backlog needs it at least that wide.

### `length_aware` reorders inside a block, on purpose

psrl consumes the first groups to finish, so ordering the queue by length changes which trajectories
are trained on, not only how fast the run goes. Ranking the whole backlog starves whichever end it
ranks last. `window_groups` serves groups in consecutive blocks of that size in arrival order and
reorders only within a block; one training batch of groups is the natural size, since it is the set
sharing a consumption deadline. Any comparison must also report tokens per consumed group, which
should match the baseline's. See `docs/design/psrl-scheduling-headroom.md` for what this buys and where.

### `scg_admission` is the only strategy that refuses

Every other strategy answers "which instance", never "whether". This one prices the
placement and leaves the request in the router when the answer is no:

```
admit  <=>  G_new * rate_after  >  (sum of resident G) * (rate_before - rate_after)
```

`G` is the trajectories bought per extra token of generation budget, obtained by
differencing a conditional length CDF, and both rates come from the same `step_time`
the engine prices its own decode steps with. The result is a residency limit that
moves with context depth and remaining slack, which is what a tuned
`max_concurrent_seqs_per_instance` approximates with a constant. See
`dryrun/value/` for the model and `examples/simulate/scg_admission_sweep.sh` for the
comparison against that constant.

Three things make it safe rather than merely clever. An idle instance is taken
unconditionally, so a fleet whose whole queue prices at zero cannot refuse its way
into the livelock bound. A length model that has not yet seen `min_samples`
trajectories reports itself unready and the strategy falls back to load balancing,
which is counted in `n_fallback` so an untested run is distinguishable from a null
result. And it requires a cost model: the Router raises rather than pricing every
dilution against `None`.

```{warning}
`strategy_args.max_staleness` must match `job.max_staleness`. The deadline is
derived from it, so a mismatch misprices every budget in the same direction, and
nothing checks the two agree.
```

Known limitation: a refusal stops the dispatch pass at that version head, so a
low-value request can block a higher-value one behind it in the same layer. The
queue is ordered by prompt length, which is the admission-selection rule this is
supposed to use, so the ordering is not arbitrary. Reordering within a layer needs
the router to peek deeper than one.

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

Three layers, kept separate on purpose:

The **staleness policy** gates the first placement with `admit_quota`. A successful first
placement atomically admits the whole prompt group and calls `on_admit` once. Its siblings can
remain in the router, but they bypass the admission check because the group already owns its
policy slot. If policy capacity is unavailable, dispatch stops and the request remains staged.

The **router's gate** decides *eligibility* and mirrors SMG's `admission_reject_reason`:
`max_concurrent_seqs_per_instance` (0 disables it), global `reject_on_waiting`, and
`reject_preempted_on_waiting`, which applies the waiting check only to preemption loopbacks.
All three default to permissive.

The **engine's `add_request`** remains the authority on whether the request actually fits. It
is not duplicated in the router: two places deciding admission is how the two drift apart.

A request whose candidates all refuse stays queued. Ordinary strategies stop at the first
version head nothing can take. The router itself does not know which versions are stale or
what TP policy is active. It visits one queue per `v_traj`, oldest first, and asks the strategy
about each head. `version_capped_tp` permits a blocked queue to be bypassed: every old version
is still restricted to the highest TP tier, while the reference version can use the lowest TP
tier after one or more older queues block.

After every successful placement, dispatch restarts at the oldest nonempty version. This keeps
older work first whenever its high-TP tier has capacity without coupling the router to
`version_capped_tp`. Each version layer is a min-heap on `(sort_key, arrival)`. `sort_key`
breaks ties:
`short_length` (default), `long_length`, or `small_id`. The key uses the **original** prompt
length even after a loopback, so a resumed request does not drift backwards every time it
returns.

Dataset ingress follows policy capacity. Staging offers a prompt group only while
`admit_quota` grants room beyond what is already staged and uncommitted, so the policy is the
only admission gate. `Simulator` used to grant ingress from a counter of its own,
`(max_staleness + 1) * batch_size` groups at startup and another `batch_size` per published
version. That counter bounded in-flight work so tightly that no policy's quota ever refused:
measured over a full aligned run, it was consulted 3367 times and refused none, while ingress
credits were exhausted at 1353 of 1359 staging calls. `max_staleness` was therefore the
simulator's guarantee rather than each policy's, which made policies with genuinely different
admission rules indistinguishable -- psrl bounds staleness through its reserve/occupy
inventory and verl does not bound it at all, yet the two produced byte-identical telemetry.
Unadmitted groups that cross a weight update are re-layered at the new rollout version;
admitted or already-generated requests keep their trajectory history.

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
When `reject_preempted_on_waiting=true`, a reclaimed victim skips instances that still have
local waiting work; fresh admissions and weight-update loopbacks remain unaffected.

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
| A preemption frees its victim | `PREEMPT` | 2122 |
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
