# Staleness policies

A staleness policy is the part of an asynchronous RL system that decides what may be
admitted, what may be trained on, and what is thrown away when weights move on. The
simulator owns the clock, the queues and the group bookkeeping. A policy owns only its
counters and answers a fixed set of hooks. There is one typed preset per modelled system,
and each preset is a faithful, small reimplementation of that system's control logic.

This page first describes the contract between the simulator and a policy, then lists
every preset with what it sets, derived from `srl/config/policies/` and `srl/policy/`.

## From preset to policy

```{mermaid}
flowchart LR
    Y["policy=verl (Hydra group)"] --> P["preset_config -> VerlPolicyConfig"]
    P -->|build| S["VerlPolicy"]
    P -->|horizon| H["SimConfig.staleness_horizon"]
    P -->|ingress_window_groups| I["initial ingress credits"]
    O["rollout.max_inflight"] -->|build argument| S
```

`srl/cli/main.py` calls `preset_config(cfg.policy)`, which maps the composed Hydra node
onto the dataclass registered in `PRESETS` (`srl/config/policies/__init__.py`). The preset
then provides three things:

`build(max_inflight)`
: constructs the policy object and stamps `policy.name` with the preset name.
  `rollout.max_inflight` is the argument. Only some presets use it, see the table below.

`horizon`
: the version bound the system declares, or `None` for a system without one. It is copied
  to `SimConfig.staleness_horizon` and from there to `SimState.max_staleness` (0 when
  `None`), to the `StalenessReport.strict` check, and to the `<- policy.horizon` marker in
  the SCG table the CLI prints. Several policies read `st.max_staleness` as their bound, so
  the horizon is not only a label.

`ingress_window_groups(batch_size)`
: how many prompt groups the dataset may stage before the first publish. It is a flow
  limit on dataset ingress, not a staleness bound, and it is deliberately independent of
  the policy's admission quota so that the candidate set the router orders does not change
  with the policy. Each publish then adds `batch_size` credits, and policies that replace
  dropped groups add more.

There is no `job.max_staleness`. Change the bound through the preset's own field, for
example `policy.staleness=1` for `psrl`.

## The policy contract

`StalenessPolicy` (`srl/policy/base.py`) has two abstract methods, `admit_quota` and
`peek_batch`. Everything else has a do-nothing default.

| Hook | Called | Default |
|---|---|---|
| `admit_quota(st)` | when the router offers a group that is not yet admitted. Counted in responses. The simulator requires at least `group_size` | abstract |
| `on_admit(group, st)` | once per group, right after its first response is placed | no-op |
| `on_route(req, instance_id, st)` | after every placement, including re-placements | no-op |
| `on_group_complete(group, st)` | when the last response of a group finishes. `CompleteAction.KEEP` moves it to `ready_groups`, `DROP` discards the whole group | `KEEP` |
| `can_sample(st)` | before each batch request. `False` keeps generating | `True` |
| `peek_batch(st)` / `take_batch(st)` | peek is also used by the engine loop's stop check. Take consumes. A batch is `batch_size` whole groups | abstract / `peek_batch` |
| `on_batch_unavailable(st)` | each time no batch can be formed | no-op |
| `will_sync_after_train(st)` | when pricing a step. `False` skips `sync_time` | `True` |
| `on_publish(new_version, st)` | after `train_version` advances, before weights reach the engines. Returns `Cancel` entries | `[]` |
| `on_train_version_advance(v, st)` | after the cancels are applied | no-op |
| `rollout_version_after_train(st)` | picks the version the engines serve after publish | `st.train_version` |
| `pause_mode(st)` | `"abort"` or `"in_place"`, see below | `"abort"` |
| `on_interrupt(reqs, st)` | after an abort-mode publish re-enqueued the in-flight work | no-op |
| `on_discard(reqs, st)` | when committed requests are permanently removed | no-op |
| `replacement_groups_on_drop(groups, st)` | after a `DROP` from `on_group_complete`. Returns extra ingress credits | `0` |
| `check_invariants(st)` | after every training step | no-op |

`SimState` is the whole visible surface: time, `train_version`, `rollout_version`,
`batch_size` and `group_size`, `max_staleness`, `inflight`, `ready`, `ready_groups`,
`n_admitted`, `n_consumed`, `n_instances` and `groups`.

### The group ledger

The simulator keeps a `GroupLedger` (`srl/policy/ledger.py`) so a policy never scans
request lists. `SimState.groups` is a tuple of immutable `GroupView`s, one per live group:

| Field | Meaning |
|---|---|
| `group_id` | monotonic key, not the workload's own id |
| `admit_version` | rollout version stamped when the first response was placed. Never rewritten |
| `start_version` | earliest weight version among completed members, `admit_version` before any completes |
| `status` | `running`, `partial` (some members done) or `done` |
| `admit_time`, `complete_time` | `complete_time` is the latest member's completion once `done` |
| `members` | request ids in admission order |

The ledger is updated from committed completions only, so it never reflects a segment that
a speculative event may still undo. Groups leave the ledger when consumed or dropped.

### Publish hook and cancels

`on_publish` replaces any notion of "expiry": at the moment a training step publishes, the
policy returns `Cancel(group_id, action, reason=None)` for live groups, running or done,
that the new version makes worthless.

| `CancelAction` | Effect |
|---|---|
| `DROP` | the group is discarded and no replacement is credited |
| `REPLACE` | the group is discarded and one fresh dataset group is credited |
| `RETRY_FROM_SCRATCH` | credited like `REPLACE`. The drop is logged with `retry: true` |

A cancelled group is always removed whole, siblings included, because a group that lost
one response can never be consumed. Drop reasons in `request.jsonl` are `l2_cancel` for a
cancel of a group still generating, `l3_stale` for a finished group (also the default for
`on_group_complete` drops), the cancel's own `reason` when set (`span_cap`), and
`queue_overflow` for verl.

### Pause mode

`pause_mode` belongs to the policy. After a publish that changed the served version:

`abort`
: every instance aborts all its requests. The requests keep their generated tokens but
  lose their KV, and the prefix-cache hash map is cleared. They return to the router, may
  land on a different instance, and re-prefill their whole context. `on_interrupt` fires.

`in_place`
: nothing is interrupted. Requests continue under the new weights with their KV retained.
  Their segment list records which version produced which tokens, so `v_min` and `v_max`
  differ for such requests.

If the served version did not change (slime between pushes) nothing is paused either way.

## The rule library

`srl/policy/rules/` holds the reusable primitives, named after the three control layers.
L1 is admission, L2 is the runtime cancel at publish, L3 is sampling.

| Layer | Primitive | What it does | Used by |
|---|---|---|---|
| L1 | `ProgressQuota(ofp)` | `(ofp + train_version + 1) * batch_size - (accepted + running)` groups | areal, skyrl |
| L1 | `WatermarkGate(over_sampling_ratio)` | 0 while `remaining >= batch_size`, else `max(1, round(ratio * batch_size))` | slime |
| L1 | `LegacyOverage(threshold, require_batches)` | bounded hand-off queue plus a sample-overage counter | verl |
| L2 | `CancelBelowVersion(horizon, action)` | cancel live groups with `admit_version < new_version - horizon` | nemo_rl, prime_rl |
| L2 | `VersionSpanCap(max_preempts, action)` | cancel groups whose members span more than `max_preempts + 1` distinct versions, counting the version being published | dressage |
| L3 | `FIFO` | oldest first by `(dispatch_time, group_id)` of the first member | areal, skyrl, openrlhf, dressage, trl, prime_rl, nemo_rl |
| L3 | `OffPolicySpan(threshold, strategy)` | `span = train_version - admit_version + 1`. `drop` replaces finished groups past the threshold at publish, `wait` blocks sampling while a running group has reached it | verl_v1 |
| L3 | `WarnOnly(threshold)` | counts taken groups past the threshold, never drops | skyrl |
| L3 | `VersionWindow(threshold, refill)` | at publish, finished groups with `new_version - admit_version > threshold` are cancelled, `REPLACE` when `refill` else `DROP` | trl, prime_rl |
| L3 | `DistinctVersionWindow(keep)` | keep the last `keep` distinct observed versions in arrival order. `keep <= 0` is off | dressage |

## Presets at a glance

Defaults are those of the dataclasses in `srl/config/policies/`. `B` is `job.batch_size`.

| Preset | Version bound (`horizon`) | Initial ingress groups | Default pause | Honours `rollout.max_inflight` |
|---|---|---|---|---|
| `psrl` | `staleness` (2) | `(staleness + 1) * B` | abort | no |
| `areal` | `max_head_offpolicyness` (0) | `(2 * ofp + 1) * B` | abort | yes |
| `roll` | `ceil(async_generation_ratio)` (1) | `(ceil(ratio) + 2) * B` | abort | yes |
| `slime` | none | `(ceil(over_sampling_ratio) + 2) * B` | abort | yes |
| `verl` | `ceil(staleness_threshold)` (2) | `(ceil(threshold) + 1) * B` | abort | yes |
| `verl_v1` | `max_off_policy_threshold` (8, or none) | `(max(h, ceil(warmup)) + 2) * B`, `h` = threshold or 1 | abort | no |
| `skyrl` | `max_staleness_steps` (4) | `(steps + 2) * B` | in place | yes |
| `openrlhf` | none | `(async_queue_size + 1) * B` | abort, in place with `partial_rollout_enable` | yes |
| `dressage` | `keep_versions`, or none when 0 | `cap + 2 * B`, `cap` = `max_active_groups` or `B` | in place | yes |
| `nemo_rl` | by sampler: window, lookahead, or none | `max_buffered_rollouts + B` | abort | no |
| `trl` | `max_staleness` (4) | `cap + B`, `cap` = `max_inflight_tasks` or `max(max_staleness, 1) * B` | in place | no |
| `prime_rl` | `max_off_policy_steps` (8) | `min(max_inflight, max((lag + 1) * B, (steps + 2) * B))` | in place, fixed | no (own `max_inflight`) |

Where `rollout.max_inflight` is honoured it caps concurrently generating prompt groups at
that value. Presets that ignore it have their own caps, listed below.

## Per-system presets

Fields are given with their defaults. Every preset also has `name`, which must equal the
group name.

### `psrl`

`staleness: 2`, `proactive_filter: false`, `proactive_threshold: 0`.

Delegates all buffer logic to `psrl.workers.ps.staleness.StalenessInventory`, created
lazily with `num_entries = ready_num_entries = batch_size`, `staleness`, and
`rollout_n = group_size`. An inventory entry is one prompt group.

- **Reserve (`on_admit`).** Every response of the group reserves a slot with
  `max_bid = v_traj + staleness`, pessimistically in the highest buffer it could still fill.
- **Admission.** `admit_quota` is the number of empty entries up to buffer
  `rollout_version + staleness`, times `group_size`.
- **Occupy (`on_group_complete`).** `occupy_data_with_reserve` re-places the finished group
  in the earliest buffer with room, moving reserved entries across buffers to avoid false
  stuck states. A group with no inventory entry, or no buffer, is dropped.
- **Consume.** The batch is the entries of buffer `train_version` once its status is
  `READY` or `READY_WITH_CAPACITY` and `batch_size` of them are in `ready_groups`.
  `take_batch` deletes the buffer.
- **Stuck time.** `on_batch_unavailable` records when a buffer is `STUCK` and
  `take_batch` accumulates `stuck_time`. With `proactive_filter`, reserved entries of a
  stuck buffer are cleared when `0 < reserved <= proactive_threshold` (`filtered` counts
  them).
- **Interrupts keep reservations.** An aborted group keeps the reservation made from its
  original `v_traj`.
- `check_invariants` asserts every placement satisfies `buffer_id - generated_version <= staleness`.

`inventory_snapshot` exposes the inventory read-only. The `inventory_deadline` router
strategy needs it (see {doc}`routing`).

### `areal`

`max_head_offpolicyness: 0`.

A cumulative token bucket. `admit_quota = (ofp + train_version + 1) * B - (accepted +
running)` groups, with `ofp = st.max_staleness`. `running` increments at admission,
and at completion `running` falls and `accepted` rises. Selection is FIFO. Nothing is
dropped and nothing is cancelled, so the bucket bounds total unconsumed work but not the
age of any one sample. `check_invariants` asserts unconsumed responses never exceed
`(ofp + 1) * batch_size * group_size`.

### `roll`

`async_generation_ratio: 1.0`.

Only `ceil(ratio)` matters: the policy reads `st.max_staleness` as its window. The budget
starts at `window * B + B` groups (just `B` when the window is 0) and grows by `B` per
training step. Admission is `budget - (running + completed)` groups.

- Groups are bucketed in cohorts by admission version (`v_traj`).
- Consumption draws oldest cohort first from cohorts in `[train_version - window,
  train_version]`.
- At publish, every cohort with step `<= (new_version - 1) - window` is garbage-collected:
  its finished groups and its running groups are cancelled with `REPLACE`, and `gc_dropped`
  counts them. A late completion from a collected cohort is dropped.
- Dropped groups are replaced one for one.

### `slime`

`update_weights_interval: 1`, `over_sampling_ratio: 1.0`, `pause_mode: abort`.

No staleness parameter. The bound comes from the loop structure. `admit_quota` is 0 while
`remaining >= B`, else `max(1, round(over_sampling_ratio * B))` groups, where `remaining`
counts groups admitted since the last take and resets to 0 on take. Surplus groups beyond a
full batch are dropped whole at completion. The batch is the first `B` ready groups.

`rollout_version_after_train` advances the served version only every
`update_weights_interval` steps, and `will_sync_after_train` is true only at that push, so
sync and weight-load costs are charged only then. `check_invariants` asserts
`train_version - served_version <= interval`. `stall_time` accumulates time spent with
work in flight but no batch.

### `verl`

`staleness_threshold: 2.0`, `require_batches: 1`.

The threshold is a float sample-overage ratio, not a version count (the `horizon` is its
ceiling, used only for reporting and the ingress window). `LegacyOverage` derives
`max_required = int(B * require_batches * (threshold + 1))` and a hand-off queue of the same
length, in groups.

- **Admission.** 0 when the queue is full or `staleness_samples >= max_required`, else
  `max_required - staleness_samples` groups. `staleness_samples` increments at admission and
  is reset at every publish to the in-flight groups plus the queue length.
- **Completion.** A finished group enters the queue. If the queue is full the new group is
  dropped with reason `queue_overflow`.
- **Consumption.** FIFO from the queue. The step starts once the queue holds
  `B * require_batches` groups and consumes exactly that many.

### `verl_v1`

`num_warmup_batches: 1.0`, `max_off_policy_threshold: 8` (`null` disables L3),
`max_off_policy_strategy: drop`, `pause_mode: abort`.

Admission is a step-clocked feed and is not staleness-gated: `floor(warmup * B)` groups
are credited once and `B` more at every publish. The only staleness-aware control is
`OffPolicySpan`, keyed on `admit_version`. With `drop`, finished groups whose span exceeds
the threshold are `REPLACE`d at publish. Running groups are never cancelled. With `wait`,
`can_sample` stays false while any unfinished group has span `>= threshold`, and nothing is
evicted. Selection takes the oldest admission versions first.

### `skyrl`

`max_staleness_steps: 4`, `pause_mode: in_place`.

Admission uses the same bucket as AReaL with `ofp = max_staleness_steps`. Selection is
FIFO. The threshold is warn-only: `take_batch` adds the number of taken groups with
`train_version - admit_version > threshold` to `staleness_violations`, and drops nothing.

### `openrlhf`

`async_queue_size: 1`, `partial_rollout_enable: false`.

A generation permit pool. Each permit is one `B`-group lump, admitted only in full.
Admission is 0 while `async_queue_size` lumps are outstanding, else the remainder of the
current lump. `partial_rollout_enable` selects when a permit is released: at `take_batch`
(the next lump can generate while this one trains), or after the publish that follows
(admission stays closed for the whole training step, so nothing is in flight at a sync).
The pause mode follows: `in_place` with partial rollout, `abort` without. The constructor
rejects `async_queue_size > 1` without `partial_rollout_enable`, because the weight-broadcast
lock that would prevent overlap is not modelled. Selection is FIFO.

### `dressage`

`keep_versions: 0`, `max_active_groups: null`, `output_queue_size: 1000`,
`max_partial_rollout_preempts: null`, `pause_mode: in_place`.

- **L1.** Admit up to `max_active_groups` (default `B`) concurrently active groups, and
  nothing while `len(ready_groups)` has reached 80% of `output_queue_size`.
- **L2.** `VersionSpanCap` with `RETRY_FROM_SCRATCH`: at publish, a group with in-flight
  members that span more than `max_partial_rollout_preempts + 1` distinct versions
  (counting the one being published) is cancelled with reason `span_cap`. `null` disables it.
- **L3.** `DistinctVersionWindow(keep_versions)` keyed on each group's last token version
  (`v_max`): when a group completes, its version joins an arrival-ordered list and the group
  is dropped if its version is no longer among the last `keep_versions` distinct ones.
  `gc_dropped` counts these. Dropped groups are replaced one for one. `0` disables the
  window and also makes `horizon` `None`.

### `nemo_rl`

`sampler: in_order` (`windowed`, `weight_fifo`, `ready_first`), `max_staleness_versions: 1`,
`max_lookahead_versions: 1`, `sample_freshest_first: false`, `max_inflight_prompts: 32`,
`max_buffered_rollouts: 64`, `pause_mode: abort`.

All samplers share one admission gate: fewer than `max_inflight_prompts` active groups,
fewer than `max_buffered_rollouts` admitted-unconsumed groups, and
`dispatch_index < train_version + window`, where the window is `max_lookahead_versions` for
`in_order` and `max_staleness_versions` otherwise. The dispatch index advances by one for
every `B` groups admitted. They differ in selection:

| Sampler | Eligible groups | Order | Cancels at publish |
|---|---|---|---|
| `in_order` | stamped for exactly `train_version` at admission | completion order | none |
| `ready_first` | all | FIFO by dispatch time | none |
| `weight_fifo` | `admit_version >= train_version - max_staleness_versions` | oldest admit version | none. A group below the window is never selected |
| `windowed` | same window | oldest, or newest with `sample_freshest_first` | `CancelBelowVersion(window, DROP)` on running and finished groups |

`horizon` is the staleness window for `windowed` and `weight_fifo`, the lookahead for
`in_order`, and none for `ready_first`.

### `trl`

`max_staleness: 4`, `max_inflight_tasks: -1`, `pause_mode: in_place`.

One bounded queue of admitted-unconsumed groups, capped at `max_inflight_tasks` (`-1`
resolves to `max(max_staleness, 1) * B` when the first quota is asked). Selection is FIFO.
At publish `VersionWindow(max_staleness, refill=True)` replaces finished groups with
`new_version - admit_version > max_staleness`. Real TRL drops single samples after groups
are disaggregated. The simulator keeps groups atomic, so the whole group is dropped.

### `prime_rl`

`max_off_policy_steps: 8`, `max_inflight: 1024`, `target_lag: 1`, `action: drop`.

Admission is independent of the staleness window: fewer than `max_inflight` groups
outstanding, and `dispatch_index < train_version + target_lag` (the dispatch index advances
every `B` groups). At publish two sweeps run over the same horizon: `CancelBelowVersion`
over live groups with the configured `action` (`drop`, `replace` or `retry_from_scratch`)
and `VersionWindow(refill=True)` over finished groups. A finished group matches both, and
the refilling cancel wins. Generation is never interrupted by a publish except for these
cancels, so the pause mode is always `in_place`. `gc_dropped` counts the cancels.

## Writing a policy

1. Subclass `StalenessPolicy` in `srl/policy/<name>.py`, implement `admit_quota` and
   `peek_batch`, and override only the hooks you need. Counters that change when a group is
   admitted, consumed, completed or cancelled have to be updated in the matching hook.
2. Register the class in `POLICY_REGISTRY` (`srl/policy/registry.py`).
3. Add a preset dataclass in `srl/config/policies/<name>.py` with `name`, its fields,
   `build(max_inflight)`, the `horizon` property, and `ingress_window_groups(batch_size)`,
   and add it to `PRESETS`. Registration with Hydra's ConfigStore follows from `PRESETS`.
   Use plain `str` for choice fields: structured configs reject `Literal`.
4. Add `srl/config/defaults/policy/<name>.yaml` documenting the defaults.
5. Test the contract the way `tests/` does for the existing presets, including
   `check_invariants`.

See {doc}`../development/index` for the surrounding workflow, and {doc}`outputs` for the
staleness report a policy's run produces.
