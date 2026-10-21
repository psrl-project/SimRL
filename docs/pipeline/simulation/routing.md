# Routing

The simulator does not place work directly on engines. Dataset groups are **staged in the
router**, and a request is placed on an instance only when a routing strategy picks one and
the engine accepts it. A group becomes admitted, in the policy's sense, at the moment its
first response is placed. Queue dwell therefore consumes no policy inventory and does not
fix the trajectory version early, and router queue depth is a real signal rather than a
constant zero.

## The path of a request

```{mermaid}
flowchart TD
    S["_stage: credits become whole groups"] --> Q["RouterQueue: one heap per v_traj"]
    Q -->|"oldest version first"| N["next request: pick_queued, rank_queued or heap head"]
    N --> A{"group admitted?"}
    A -->|no| K{"policy.admit_quota >= group_size"}
    K -->|no| STOP["dispatch stops, request stays staged"]
    K -->|yes| G
    A -->|yes| G["AdmissionGate filters instances"]
    G --> NC["strategy.narrow_candidates"]
    NC --> ST["sticky pin (soft)"]
    ST --> SEL["strategy.select_context or select"]
    SEL --> ADD{"instance.add_request"}
    ADD -->|refused| STAY["stays queued"]
    ADD -->|placed| COMMIT["first member: on_admit, ledger, v_traj. Every member: on_route"]
```

Three components are kept separate on purpose.

- `RouterQueue` (`router/request_queue.py`) orders what is waiting.
- A `RoutingStrategy` (`router/strategy.py`, `router/strategies/`) picks among instances.
- `Router` (`router/router.py`) owns the queue, applies the `AdmissionGate`, and runs
  dispatch.

The staleness policy gates the **first** placement of a group: `_can_place_from_router`
admits it only if `policy.admit_quota(state) >= group_size`, and a successful placement
admits the entire group atomically. Its siblings may stay in the router, but they bypass
the quota because the group already owns its slot. The engine's `add_request` stays the
authority on whether a request actually fits: it raises if the request could never fit even
on an empty instance, and returns `False` when a configured limit refuses it.

## Queue order

Each weight version owns one min-heap, and dispatch tries versions oldest first. The version
key is `v_traj`, the rollout version stamped at staging. Unadmitted staged groups are
re-layered to the new rollout version at every publish that changed it. Admitted work keeps
its history. Inside a version, `rollout.router.sort_key` orders requests:

| `sort_key` | Order |
|---|---|
| `small_id` (default) | arrival, by request id |
| `short_length` | shorter prompt first |
| `long_length` | longer prompt first |

The length keys use `prompt_len`, not the current context, so a request that came back
after an abort does not move further back each time it returns. `small_id` is the default
because a length key makes placement depend on how much of the dataset is staged.

## Picking a strategy

```bash
srl rollout.n_instances=4 rollout.router.strategy=kv_cache_aware
srl rollout.n_instances=4 rollout.router.strategy=length_aware \
    +rollout.router.strategy_args.order=spt
```

`strategy_args` starts empty, so keys are added with `+`.

| `strategy` | Rule | Arguments (defaults) |
|---|---|---|
| `request_num_balance` | fewest in-flight requests, ties to the lowest instance id. The default | none |
| `round_robin` | cycle through the eligible instances | none |
| `random` | uniform with an owned generator | `seed` (0) |
| `kv_cache_aware` | largest `free_blocks - committed_blocks`. Queued requests have allocated nothing yet, so raw free blocks cannot tell a full instance from an empty one | none |
| `prefix_aware` | the instance with the largest prefix hit for this request, unless the hit covers at most `cache_threshold` of its context or the fleet is imbalanced, in which case fewest requests | `cache_threshold` (0.3), `balance_abs_threshold` (64), `balance_rel_threshold` (1.5) |
| `throughput_balance` | smallest predicted step-time increase from `InstanceView.step_time_delta`, which calls the rollout cost model | none |
| `length_aware` | orders whole groups by their longest predicted member and places by predicted load | `order` (default `lpt`, one of `fifo`, `lpt`, `spt`), `balance` (default `load`, one of `count`, `load`), `window_groups` (0), `kv_watermark` (0.0), `length_model` (`oracle`), `length_model_args` |
| `inventory_deadline` | forecasts the training pipeline, see below | see below |

`imbalanced` in `prefix_aware` means the load gap exceeds both the absolute and the relative
threshold.

### `length_aware`

Two independent decisions share one length prediction.

- **Which request next.** A prompt group is trainable only when its slowest member finishes,
  so the group is keyed on its longest predicted member. `lpt` starts the longest group
  first, `spt` the shortest. Members of a group that has already started always go ahead of
  a new group. The ranking sees `rollout.router.queue_window` requests per placement
  (`rank_queued`), so a strategy that orders the whole backlog needs a window at least as
  wide.
- **Where.** `balance=load` picks the instance with the least predicted remaining tokens
  over its residents, `count` the fewest requests.
- **Whether.** `kv_watermark` refuses an instance when its committed blocks plus the
  newcomer's current context would exceed that fraction of the pool. It looks only at what
  residents hold now and needs no length prediction.

Reordering can change which trajectories are trained on, because consumption takes the
first groups to finish. Ranking the whole backlog starves whichever end of the length
distribution it ranks last. `window_groups` serves groups in consecutive blocks of that many
in arrival order and reorders only inside a block. One training batch of groups is the
natural size. A comparison across orderings should also report tokens per consumed group.

The `LengthPredictor` (`srl/forecast/length.py`) is `oracle`, which reads the trace's
`target_len`, or `noisy_oracle(sigma, seed)`, which multiplies it by a deterministic
per-request lognormal error. Both read oracle information, so they bound what a real
predictor could buy.

## Strategy hooks

Only `select(req, candidates) -> index | None` is required. The rest are optional and
duck-typed, so a strategy implements what it needs.

| Hook | Called | Implemented by |
|---|---|---|
| `narrow_candidates(req, candidates, context)` | after the admission gate, returns indices that stay eligible | none built in |
| `allow_version_bypass(req, context)` | when a version's head cannot be placed. True lets dispatch try the next version | none built in (`inventory_deadline` returns false) |
| `select_context(req, candidates, context)` | instead of `select`, with the clock and fleet TP range | `inventory_deadline` |
| `rank_queued(window, now)` | per placement, over the first `queue_window` heap-ordered requests of one version | `length_aware` |
| `pick_queued(window, views)` | per placement, over the whole version layer, instead of `rank_queued` | `inventory_deadline` |
| `observe_enqueue(reqs)` | when a group is first queued, the one moment all members are known to be queued together | `length_aware`, `inventory_deadline` |
| `observe_completion(req)` | when a response completes | none built in |
| `observe_train_cycle(version, t)` | at publish | none built in |
| `bind_runtime(runtime)` | once, when the simulator builds the router | `inventory_deadline` |
| `begin_dispatch(t)` | at the start of each `dispatch()` call | `inventory_deadline` |

`RoutingContext` carries `blocked_versions`, the smallest and largest TP degree in the
fleet, and `now`. `InstanceView` is what a strategy may read about an instance: load,
`free_blocks`, `committed_blocks`, `kv_blocks`, `block_size`, `prefix_hit_tokens(req)`,
`step_time_delta(req)`, `residents()` and the cost model. Strategies receive read-only
views and `TrajView` projections, never `NativeInstance`, so the same code runs on both
engine backends.

Adding a strategy means one module under `srl/component/rollout/router/strategies/`, one
class decorated with `@register_strategy("name")`, and an import in that package's
`__init__.py`.

## Admission gate

`AdmissionGate` decides which instances may be offered a request at all. It decides
eligibility only.

| Key under `rollout.router` | Refuses an instance when |
|---|---|
| `max_concurrent_seqs_per_instance` (0 disables) | it already holds that many requests, waiting plus running |
| `reject_on_waiting` | it has a queue of its own |
| `reject_preempted_on_waiting` | same, but only when placing a preemption loopback |

`rollout.max_concurrency` is a different limit: vLLM's `max_num_seqs` inside the engine. The
`rollout.admission_control.*` flags are engine-side gates inside `add_request`
(`reject_if_kv_full`, `reject_if_waiting`, `reject_if_running_full`). All default to false
because KV capacity is the scheduler's constraint, which the engine already enforces by
leaving a request that does not fit on its own waiting queue.

A request that no candidate can take stays queued. Without `allow_version_bypass`, a blocked
head stops dispatch for all newer versions. After every successful placement dispatch
restarts at the oldest version. A group-sticky pin (`enable_group_sticky`) pins a group's
responses to the instance that took the first one. It is soft, falling through to the full
candidate set when the pinned instance cannot help, and it is released when the group is
consumed or dropped.

## When dispatch runs

A dispatch attempt can only succeed if capacity was released, because `add_request`'s gates
are block counts and nothing else in the attempt varies with time. So `Router.dispatch` is
called only at committed points and never inside a speculative engine event (it asserts
that no instance has an open undo log):

- on entry to the engine loop.
- after a `REQUEST_COMPLETE` commits and frees blocks, when the queue is non-empty or
  ingress credits remain.
- after a cross-instance preemption has been reclaimed.
- after staging new groups, including the batch opened by a publish.

A decode segment only grows blocks, so it triggers nothing. A segment ends at the first
completion or KV exhaustion across the batch, which is the first instant capacity can
change, so the trigger set is exact for the engine's event granularity. There is no polling
interval to configure.

## Loopback

Two triggers, two rules.

**Weight update (abort mode).** Every instance aborts all its requests and hands them back
(`abort_all(requeue=False)`). The simulator enqueues them in the router, so they can land on
a different instance. `num_reroutes` counts re-placements and
`n_cross_instance_reroutes` the ones that changed instance.

**Preemption.** A preempted request stays at the head of its own instance's waiting queue.
It reaches the router only when `rollout.router.max_waiting_after_preempt` is set and that
instance's queue has grown past it, which is vLLM's `preemption_notification_threshold`.
That event is then treated as global: it is committed, other instances' speculative events
are undone and aligned to it, and `reclaim_preempted` takes back up to `overflow` waiting
requests from the tail that were preempted and have no prefill progress, keeping their
tokens. Requests that are mid-prefill are never moved. With the threshold at `null` (the
default) every preemption stays local.

## Eviction

When a decode step cannot get a block, the engine evicts a resident, keeps its generated
tokens, and requeues it locally. Which resident is a separate registry from routing:
`rollout.router.eviction`.

| `eviction` | Victim |
|---|---|
| `vllm_lru` (default) | latest `dispatch_time`, vLLM's reverse arrival order |
| `longest_ctx` | deepest context, freeing the most blocks |
| `latest_deadline` | the largest `deadline_buffer`, then deepest context. Requests with `deadline_buffer == 0` are treated as most urgent |

A strategy returning `None` defers to the engine's default. `deadline_buffer` is stamped
only by `inventory_deadline` at admission, so `latest_deadline` is meaningful only with it,
and the stamp needs the Python engine, see {doc}`/pipeline/simulation/backends`.

## `inventory_deadline`

`inventory_deadline` admits and routes by forecasting what each option does to the training
pipeline. Instead of a configured cap on admitted work, every decision is priced as a whole
`Pipeline`: a predicted training-finish time `E_k` for each pending buffer. An option that
would delay a critical buffer loses to DEFER, and one that would not is free to go wherever
the forecast finds it feasible. The code is `router/strategies/inventory_deadline.py` and
the `srl/forecast/` package.

```bash
srl policy=psrl rollout.n_instances=4 \
    rollout.router.strategy=inventory_deadline \
    +rollout.router.strategy_args.kv_watermark=0.6
```

Requirements, both enforced by the code: the policy must be `psrl`, because
`SchedulingRuntime.inventory()` raises `TypeError` for any other policy, and the Python
engine must be used, because the strategy writes `deadline_buffer` on each request. The
strategy is marked `python_engine_only`, so `auto` picks that engine and `SIMRL_BACKEND=cpp`
is rejected.

### What is forecast

| Module | Role |
|---|---|
| `forecast/snapshot.py` | read-only value types: `Resident`, `Arrival`, `Interrupt`, `InstanceState` (built from a live `NativeInstance` or `CppInstance`, so it works on either backend) |
| `forecast/drain.py` | `InstanceForecast`: replays one instance to get every completion time, peak KV blocks and a `kv_overflow` flag |
| `forecast/fleet.py` | `FleetForecast`: the same for all instances, plus `fold_member`, `placement_order`, `chain_buffers` |
| `forecast/pipeline.py` | `PipelineForecast`, `Pipeline`, `TrainState`, `compare` |
| `forecast/plan.py` | `PlanSession`: one dispatch burst, evaluated incrementally |
| `forecast/length.py` | `LengthPredictor` (`oracle`, `noisy_oracle`) |
| `simulator/runtime.py` | `SchedulingRuntime`: the read-only facade a strategy binds to |

**Drain replay.** With nothing queued the replay prices closed-form decode segments with the
same `decode_segment_time` the engine uses. With a queue it iterates like the engine: a
decode pass over the running batch, then first-come prefill chunks from the remaining token
budget, honouring `max_concurrency`, with a prefix hit by prompt identity and whole blocks.
KV is an upper bound: a prompt shared by siblings is charged once per sibling. Preemption is
not modelled. The replay reports its peak block use and flags `kv_overflow` when that
exceeds the pool, after which its finish times are void and the option is infeasible. An
`Interrupt` models a scheduled publish: every resident is aborted at `time`, the instance
idles for `resume_delay`, and the aborted set is re-prefilled in request-id order.

**Pipeline.** `PipelineForecast.evaluate` feeds predicted group ETAs (the latest member
finish) to psrl's own `forecast_ready_times` to get each buffer's READY time `F_k`, then
chains `publish_k = max(F_k, E_{k-1}) + cycle` and `E_k = publish_k + weight_load +
reconfig`. The chain starts at `now` when the trainer is idle, otherwise at the end of the
step in progress plus its weight load. The cycle
comes from `SchedulingRuntime.cycle_time`, which is the simulator's own `_train_cycle`
(training plus sync plus recompute), priced on the per-response lengths of the `batch_size`
groups predicted to finish first, padded with their mean.

`chain_buffers` projects `horizon_buffers` buffers. Buffer `k`'s READY time depends only on
finishes at or before its own publish, and every publish is later than its buffer's READY
time, so an interrupt scheduled at `publish_k` cannot change what was already resolved:
no fixed point is needed. Hypothetical admissions (`new_groups`) are passed to every
re-evaluation, since each one replays the snapshot from scratch.

A buffer with an empty slot can never fill from the work already admitted, so its exact
`F_k` is infinite. `Pipeline.train_end_floor` keeps such buffers comparable: the same chain
with an infinite READY time replaced by the latest ETA among that buffer's already-reserved
groups (`-inf` for an empty buffer). The floor ignores `new_groups` on purpose, because
counting a candidate's own reservation would charge it for filling a buffer.

**Comparison.** `compare(a, b, eps)` walks the two `train_end_floor` vectors left to right
over their shared prefix and returns the first difference larger than `eps`. Two infinite
values of the same sign tie. A decision that barely moves the nearest buffer is therefore
judged on the next one. The tolerance is not transitive. The strategy uses
`eps = eps_frac * cycle`.

### Decisions

`begin_dispatch` builds one `PlanSession.build(runtime, ...)` per dispatch burst. It
snapshots the psrl inventory, folds the queued members of already-admitted groups onto the
instances where they would finish soonest (`placement_order`: ascending deadline buffer,
then longest predicted remaining), and computes `defer_until`, the earliest future
completion anywhere in the fleet capped at the first publish. Every decision the strategy
commits updates the session in place, so later decisions in the burst see earlier ones
without replaying the fleet.

Two decisions are priced the same way:

- **Routing an admitted group's queued member.** `evaluate_member` prices each instance and
  DEFER, where DEFER means arriving at `defer_until`.
- **Admitting a staged group.** `evaluate_admission` prices the first member on each
  instance, folding the remaining members longest-first onto the best instances, against
  DEFER_GROUP, which is the baseline pipeline (or, with `lookahead_defer`, admitting at
  `defer_until` on the fleet with members predicted to finish before then removed).

Ordering. An admitted group's queued member always goes ahead of a staged group: by deadline
buffer, then longest predicted remaining. Staged groups are ordered by block of
`window_groups` groups in arrival order, then by the group's slowest predicted member
(`window_groups=0` ranks the whole backlog, which starves one end of the length
distribution). A deadline is real urgency and is never window-bounded.

Choice. Among options, the lexicographically best by `compare` wins. Options tied with it
within `eps` are broken toward the least-loaded instance (fewest active requests, then
lowest id). If `throughput_margin_frac > 0` and the winner is an admission, it becomes DEFER
unless adding the first member raises the instance's decode throughput
(`n_d / decode_segment_time`) by at least that fraction. `allow_version_bypass` is always
false: letting a newer version pass a blocked or deferred head would break the deadline
order. The deadline of a group admitted earlier in the same burst, which the snapshot
cannot know, is `rollout_version + horizon`, or the ingress window when the policy has no
horizon. Admission commits stamp `deadline_buffer` on the group's members.

### Knobs

Passed through `rollout.router.strategy_args`.

| Key | Default | Meaning |
|---|---|---|
| `length_model` | `oracle` | name in `LENGTH_PREDICTORS` |
| `length_model_args` | none | constructor keywords for the predictor |
| `horizon_buffers` | 2 | pending buffers each decision projects, clamped to the snapshot's buffers |
| `eps_frac` | 0.01 | tolerance as a fraction of the priced training cycle |
| `defer` | true | false removes DEFER and forces every feasible request onto the best instance |
| `window_groups` | 32 | staged groups per reordering block |
| `kv_watermark` | 0.0 | KV-peak admission watermark, below |
| `lookahead_defer` | false | price DEFER_GROUP as admission at `defer_until` on the reduced fleet |
| `throughput_margin_frac` | 0.0 | smallest decode-throughput gain that justifies an admission. 0 disables, ignored when `defer` is false |

### KV-peak admission watermark

`kv_watermark` in `[0, 1]` is the fraction of an instance's pool that the forecast KV peak
may reach before admitting a staged group there is refused. When pricing an admission
on an instance, the strategy replays that instance with the group's first member added and
rejects the option if `peak_blocks > kv_watermark * kv_blocks`, in addition to the
unconditional `kv_overflow` rejection. Zero disables the check, leaving `kv_overflow` as
the only feasibility bar.

It applies to admission only, never to routing an already-admitted member, which holds a
buffer reservation and must go somewhere. It checks the instance that would take the first
member. The group's remaining members are placed afterwards by soonest finish and are
subject to `kv_overflow` only. This is a different quantity from `length_aware`'s
`kv_watermark`, which compares committed blocks now, with no growth, to the pool.
