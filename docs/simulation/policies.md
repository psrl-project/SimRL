# Staleness policies

The simulator never moves time forward on its own. The staleness policy decides when a
request may be dispatched and when a training batch may be consumed. Every policy is a
plugin: it receives a `SimState` snapshot and returns a quota or a batch. There are five
built-in policies, each modelling a different real system. Choosing the wrong one, or
misconfiguring `job.max_staleness`, is the most common source of a simulation that
produces plausible-looking numbers for the wrong setup.

## The `max_staleness` confusion

Before reading the individual policies, understand this table. The same configuration key
means a different thing depending on which policy is selected, and one policy ignores it
entirely.

| Policy | What `job.max_staleness` controls |
|--------|-----------------------------------|
| `areal` | Version bound: the token bucket ceiling is `(max_staleness + train_version + 1) * B` |
| `psrl` | Version bound: buffer IDs extend to `rollout_version + max_staleness` |
| `roll` | Version bound: the GC window is `train_version - max_staleness` to `train_version` |
| `verl` | Fallback ratio: used as `staleness_threshold` when `policy.staleness_threshold` is null |
| `slime` | **Ignored.** The staleness bound is `policy.update_weights_interval`, always |

A simulation that switches from `policy=areal` to `policy=slime` with `max_staleness=4`
will not run `slime` with a version bound of four. It will run `slime` with a bound of
one (the default `update_weights_interval`) and silently ignore the four. Set
`policy.update_weights_interval` explicitly when using `slime`.

---

## `psrl`: Reserve / Occupy / Consume (default)

**`policy=psrl` is the default.** It delegates buffer management to
`psrl.workers.ps.staleness_controller.StalenessInventory`, so the simulation
faithfully reproduces the real PSRL system's three-phase lifecycle rather than
approximating it.

### Mechanism

Each request passes through three states inside the inventory:

1. **Reserve** (`on_admit`): a slot is pessimistically claimed in the highest pending
   buffer that the request's trajectory version could still fill, working backwards from
   the last available entry (`get_last_non_reserved`). This is conservative by design:
   it prevents the inventory from over-committing to requests that may complete too late.
2. **Occupy** (`on_complete`): when the rollout engine finishes a request, it is moved
   to the earliest buffer that has room, with cross-buffer entry movement to avoid
   spurious STUCK states. Greedy earliest-first ensures batches are assembled as quickly
   as possible.
3. **Consume** (`peek_batch` / `take_batch`): the policy waits until the buffer for
   `train_version` is in a READY or READY\_WITH\_CAPACITY state. The buffer is deleted
   only when the batch is formally taken, not when it is merely peeked.

The inventory also tracks STUCK states, meaning buffers that cannot reach READY because too
many reserved entries have become stragglers. The optional `proactive_filter` knob lets
the policy evict reserved entries proactively when only `proactive_threshold` or fewer
remain, unblocking a stuck buffer at the cost of dropping those requests.

### Configurable knobs (`srl/config/defaults/policy/psrl.yaml`)

| Key | Default | Meaning |
|-----|---------|---------|
| `proactive_filter` | `false` | Enable proactive eviction of near-stuck buffers |
| `proactive_threshold` | `0` | Evict only when this many or fewer reserved entries remain |

### Known rough edge: external dependency

**`psrl` requires the `psrl` package to be importable.** The module contains a
hard-coded absolute path in `srl/policy/psrl.py` and falls back to inserting it on
`sys.path` when the package is not already importable. On any machine where that path
does not exist, constructing the default policy raises `ImportError` before the
simulation begins.

The fix is to install the package from its source tree:

```bash
pip install --no-deps -e /path/to/psrl
```

After that one-time install, the import succeeds on any machine. If the source is
unavailable, select a different policy with `policy=areal` or `policy=roll`. Do not
add the source directory to `PYTHONPATH` manually unless you understand the version
the path points to, since the path is machine-specific and may not match the version of
PSRL the simulation was validated against.

---

## `areal`: Cumulative token bucket

`areal` reproduces the admission control from the AReaL framework. It imposes a ceiling
on *total unconsumed work* across the whole loop, but it does not bound individual
sample age.

### Mechanism

At every admission decision, the quota is:

```
quota = (max_staleness + train_version + 1) * B - (accepted + running)
```

where `B` is `batch_size`, `accepted` is the cumulative number of completed requests
that have been handed to the training queue, and `running` is the number of requests
currently in flight. The ceiling grows by one batch after each training step
(`train_version` advances), so it does not stall as training makes progress.

Because the bound is on *total unconsumed work*, a slow request that was dispatched
many versions ago is not proactively evicted. It simply occupies part of the ceiling
until it completes. This means a single very slow request can hold the ceiling high and
let many fresh requests in, which is why `areal` does not bound individual staleness.

The optional `max_inflight` cap adds a hard upper bound on simultaneously in-flight
requests, independent of the bucket. It is not exposed in the default YAML and must be
passed as a Python override if needed.

### Configurable knobs

`areal.yaml` exposes no additional knobs. The only parameter is `job.max_staleness`,
which sets `ofp` in the formula above.

---

## `roll`: Token bucket with GC discard

`roll` uses the same admission formula as `areal` but adds two enforcement mechanisms
that `areal` lacks: a consumption floor that prevents batches from being drawn from
too-old groups, and a garbage-collection pass that evicts stragglers before they age out
of the window.

### Mechanism

Requests are grouped by the training version at which they were dispatched. The budget
grows by one `batch_size` after each training step, exactly as in `areal`.

The consumption floor (`min_step = train_version - max_staleness`) prevents batches
from being built out of requests dispatched before the window. At each call to
`peek_batch`, only groups whose step is within `[min_step, train_version]` are eligible.

Enforcement is *proactive* rather than reactive: the `expire` method, called when a
training batch is committed, garbage-collects any group whose step is below
`expected_train_version - max_staleness`. In-flight requests in a GC'd group are
discarded immediately, before training finishes. This keeps the window tight across
training steps and avoids allowing a burst of completions from an old group to
invalidate the staleness contract.

The `gc_dropped` counter records how many requests were evicted by GC, which is useful
for diagnosing a configuration where the staleness window is too tight for the workload.

### Configurable knobs

`roll.yaml` exposes no additional knobs beyond `job.max_staleness`.

---

## `slime`: Loop barrier

`slime` takes a fundamentally different approach: instead of a token bucket or buffer
inventory, it relies on the training loop's barrier structure to bound staleness. The
bound is structural, not parametric.

### Mechanism

The loop structure is: dispatch generation, then (after `update_weights_interval`
training steps) push new weights, then collect the generation results, then dispatch
again. Because generation is dispatched *before* the weight push and consumed *after*,
the maximum staleness any request can have when consumed is exactly
`update_weights_interval`, determined by timing rather than by a threshold parameter.

There is **no staleness parameter** in `slime`. The `job.max_staleness` key is ignored.

At each admission decision, `slime` dispatches up to
`round(over_sampling_ratio * batch_size)` requests at once, then waits for that cohort
to complete before admitting the next wave. Over-sampling (ratio > 1.0) lets the policy
dispatch more requests than one batch requires, so it can select the best `batch_size`
responses from a slightly larger pool, matching the behaviour of real slime deployments
that intentionally generate redundant samples.

When `partial_rollout=False` (the default), any in-flight requests at weight-push time
are evicted rather than resumed. When `partial_rollout=True`, interrupted requests are
retained, and the staleness invariant is then the caller's responsibility.

The `rollout_version_after_train` hook drives weight-push synchronisation: the rollout
engine's visible weight version only advances every `update_weights_interval` steps,
not after every training step.

### Configurable knobs (`srl/config/defaults/policy/slime.yaml`)

| Key | Default | Meaning |
|-----|---------|---------|
| `update_weights_interval` | `1` | Push weights every N training steps. Sets the staleness bound |
| `over_sampling_ratio` | `1.0` | Admit this many times `batch_size` per wave (must be ≥ 1.0) |

---

## `verl`: Window-based admission with bounded queue

`verl` models the fully asynchronous mode of the verl framework. Admission is governed
by a sample-overage ratio rather than a version count, and completed requests are held
in a bounded hand-off queue that drops the oldest entry on overflow.

### Mechanism

The key quantity is `staleness_threshold`, which is a **float sample-overage ratio, not
a version count**. A value of `0.1` means the policy will admit up to
`round(batch_size * (1 + 0.1)) = round(1.1 * B)` requests per version window. The
ceiling is the number of samples that may be "stale" relative to the current weight
version.

At construction, the policy derives:

```
max_required = int(batch_size * require_batches * (staleness_threshold + 1))
max_queue_size = max_required
```

`require_batches` controls how many complete batches must be ready before training
may begin. At its default of 1, a single full batch is sufficient.

After each training step, `staleness_samples` is reset to the number of currently
in-flight and queued requests, so the window tracks the live system state rather than
an absolute counter.

The hand-off queue (`collections.deque`) enforces `maxlen=max_queue_size`. When a
request completes and the queue is full, the oldest entry is evicted (`CompleteAction.DROP`),
and the `dropped` counter increments. This matches verl's behaviour of discarding
overproduced samples rather than blocking the rollout engine.

### Configurable knobs (`srl/config/defaults/policy/verl.yaml`)

| Key | Default | Meaning |
|-----|---------|---------|
| `staleness_threshold` | `null` | Overage ratio. Null falls back to `job.max_staleness` |
| `require_batches` | `1` | Training batches that must be queued before a step starts |

When `staleness_threshold` is null, the simulator uses `job.max_staleness` as the
ratio. Because `job.max_staleness` is typically an integer version count for other
policies, make sure its value makes sense as an overage ratio before relying on the
fallback.

---

## Choosing a policy

The choice of policy should reflect the real system you are evaluating, not just the
staleness bound you want to impose. Each policy is a simulation of a real system's
protocol:

- Use `psrl` to model a PSRL deployment. It requires the `psrl` package, as described above.
- Use `areal` to model AReaL's cumulative token bucket.
- Use `roll` to model ROLL's async admission with GC enforcement.
- Use `slime` to model slime's loop-barrier approach. Set
  `policy.update_weights_interval` to match the real system's push interval.
- Use `verl` to model verl's fully async mode. Set `policy.staleness_threshold`
  explicitly rather than relying on the `job.max_staleness` fallback.

For exploratory runs where the goal is to compare scheduling strategies, `areal` and
`roll` are the most transparent: their admission formulas are short and their knobs are
minimal. `psrl` is the most faithful to a production system but requires the external
dependency.

---

## Writing your own policy

The `StalenessPolicy` ABC in `srl/policy/base.py` defines the full interface. Only
two methods are abstract and must be implemented:

- `admit_quota(st: SimState) -> int`: how many staged responses may commit their
  first placement at this instant. Return 0 to leave them in the router.
- `peek_batch(st: SimState) -> list[Request] | None`: return the next training batch
  without consuming policy state. Return `None` if no batch is ready.

The remaining hooks have do-nothing defaults and can be overridden selectively:

| Method | When called |
|--------|-------------|
| `on_admit` | Once per group, after its first response is successfully placed |
| `on_route` | After every placement, including reroutes of admitted requests |
| `on_group_complete` | After every response in a group finishes. Return `KEEP` or `DROP` |
| `take_batch` | When training actually consumes a batch. Defaults to `peek_batch` |
| `on_batch_unavailable` | Each iteration that no batch could be assembled |
| `on_train_version_advance` | After a training step publishes a new version |
| `will_sync_after_train` | Whether this step ends with a weight push (default: always) |
| `on_interrupt` | When in-flight requests are interrupted by a weight push |
| `on_discard` | When requests are permanently removed |
| `expire` | Called before training commits, to evict requests that would exceed staleness |
| `rollout_version_after_train` | What version the rollout engine should serve after publish |
| `check_invariants` | Assert the policy's own contract (called in debug mode) |

`SimState` is the entire visible surface: it carries the current simulation time, the
current training and rollout weight versions, lists of in-flight and ready requests, and
admission counters. The policy may not mutate `SimState` directly.

A minimal policy that simply caps admission at a fixed count and batches greedily is
roughly thirty lines, following the same pattern as `ArealPolicy`. Register it by adding
a case in the policy factory and a corresponding YAML under
`srl/config/defaults/policy/`. For the development workflow, see
{doc}`../development/index`.
