# Metrics

Every quantity below is computed in `srl/diagnosis/`. This file is the definition, the code is the
implementation, and they must agree.

## Phases

A request's lifetime from `admit_time` to `complete_time` is tiled by phases with no gap and no
overlap. Reconstructed from `instance/N` (which records each resident's `generated`, `ctx` and
`prefilled` at every committed change), with gaps labelled from `train`.

| Kind | Definition |
|---|---|
| `prefill` | On an instance, `prefilled < ctx`, first attempt |
| `reprefill` | Same, after a cache discard. `attempt` is 1 or more |
| `decode` | On an instance, `prefilled >= ctx`, `generated` rising |
| `engine_wait` | In an instance's `waiting_len` |
| `router_queue` | Admitted, unfinished, on no instance |
| `weight_load` | An uncovered stretch overlapping a publish's weight-load window, which is `dt - train_time - sync_time - recompute_time` before `end` |

## Decomposition

For one request, with the observed lifetime held fixed:

- `interference = max(0, observed time of prefill and decode phases - their solo price)`
- `queue = router_queue + engine_wait`
- `reprefill`, `weight_load`: the phase totals
- `solo = observed - interference - queue - reprefill - weight_load`

Solo price uses the same cost model the engine used, at a batch of one.

## Critical path

Step `k`'s **binding request** is the member of `k`'s consumed batch with the largest
`complete_time`. `idle = start_k - end_(k-1)`, floored at zero. `wait_cause` is `completion` when
the binding request's `complete_time` equals `start_k`, `no_wait` when idle is zero, and
`unexplained` otherwise. **An unexplained wait is reported, never attributed.**

`sibling_spread` is the gap between the first and last completion within the binding group, since
a group is ready only when its slowest member is.

## Bounds

Both hold the consumed set and every training step duration fixed, and ignore fleet capacity.

| Bound | What changes |
|---|---|
| `span_solo_all` | Every request runs at a batch of one from its admission, chained forward so an earlier step start pulls the next one in. Two variants: re-prefill charged, re-prefill removed |
| `span_tail_only` | Only each step's binding request runs alone. Every other member keeps its observed completion, so the next-slowest becomes binding |

`headroom = observed span / bound span`, so it is at least 1. It caps what a routing, admission or
fleet-shape change could win. It does not say such a change exists.

## Fleet

Per idle window: each instance's `busy_fraction` (overlap of its `rollout_step` segments with the
window, over the window width), the mean router `queue_len` sampled inside it, and from `buffer`
whether the binding group is reserved at `buffer_id - v_traj == staleness_horizon`, which means its
reservation cannot move later.

Whole run, per instance: `busy_seconds`, `tokens`, `last_active`. Wide spread means an unevenly
loaded fleet regardless of the mean concurrency a sweep reported.
