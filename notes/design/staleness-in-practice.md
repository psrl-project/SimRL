# Staleness control in practice: a survey of twelve systems

## Why this document exists

SimRL simulates a rollout/training loop under a *staleness policy*
(`docs/simulation/policies.md`). Each built-in policy claims to model a real system, so
the policies are only as trustworthy as our reading of those systems. This document
records what the surveyed frameworks actually implement, with line-level citations, and
extracts the structure they share.

This file lives under `docs/design/`, which is excluded from the Sphinx build
(`docs/conf.py:54`). It is a working note, not published documentation.

The short version: **the most widely-cited systems (verl's sync mode, AReaL, SkyRL)
implement their staleness knob as a bound on the volume of unconsumed work, not on the age
of any sample.** This is not universal. Of the twelve systems surveyed, seven do enforce a
genuine per-sample age bound, and one (prime-rl) enforces it against in-flight work before
the generation completes. The design space is better described by two axes than by a
single "how strict" ordering. See “The two axes” below.

All claims below were verified by reading source, not documentation. Where a system's
documentation or a field's help text contradicts its code, both are quoted and the
discrepancy is stated. External repositories were pulled as tarballs from
`codeload.github.com` at the revision cited. See "Reproducing this survey" below for the
exact method.

## Sources surveyed

Seven systems were read end-to-end for this document. Five more were surveyed for their
staleness knob only. verl's two modes ship in one repository but are unrelated
mechanisms, so they are read and cited separately. Unless noted otherwise, revisions are
as of 2026-10-10. verl v1, AReaL, ROLL, SkyRL, slime, Dressage, NeMo-RL, TRL, prime-rl and
OpenRLHF were corrected on 2026-10-16 against the commits cited below (full detail in
`publishing/源码复核记录.md`). ROLL, Dressage, TRL, prime-rl and NeMo-RL's specific
corrections for this document were independently re-fetched and re-checked again on
2026-10-19.

| System | Revision | Depth |
|---|---|---|
| verl (legacy sync mode) | `0.9.0.dev`, commit `e5ca4acb` | full (local checkout, `psrl/third_party/verl`) |
| verl v1 (`trainer.v1.trainer_mode=separate_async`) | `verl-project/verl@5b550b0c` | full (source tarball) |
| AReaL | `areal-project/AReaL@6cad8710` | full (source tarball) |
| ROLL | `alibaba/ROLL@d96d0cce` | full (source tarball) |
| slime | `THUDM/slime@2f4280e3` | full (source tarball) |
| SkyRL | `NovaSky-AI/SkyRL@38146ee9` | full (source tarball) |
| Dressage | `Accio-Lab/Dressage@3e3142fe` | full (source tarball) |
| prime-rl | `PrimeIntellect-ai/prime-rl@d61df569` | full (source tarball) |
| TRL | `huggingface/trl@9d76f2f8` | full (source tarball) |
| NeMo-RL | `NVIDIA-NeMo/RL@726d5eb2` | full (source tarball) |
| OpenRLHF | `OpenRLHF/OpenRLHF@dc2a7ad3` | knob + enforcement path verified |
| Trinity-RFT | `agentscope-ai/Trinity-RFT@main` | knob only, via sub-survey |

The Trinity-RFT row was gathered by a delegated agent and is reported here with that
provenance. It has not been re-read line by line and should be re-verified before being
cited in a paper.

## The common structure: an unconsumed-work lease

The three most widely-cited systems (AReaL, SkyRL and verl) gate admission with the
same inequality, and OpenRLHF implements a degenerate case of it. Writing $A(t)$ for the
cumulative count of admitted-and-completed units, $R(t)$ for units currently generating,
$v(t)$ for the training version, $B$ for the training batch size and $S$ for the
configured knob:

```text
A(t) + R(t)  <=  (S + v(t) + 1) * B
```

Rearranged, with $v(t)\,B$ being exactly the work training has consumed so far:

```text
outstanding(t)  =  A(t) + R(t) - v(t)*B  <=  (S + 1) * B
```

The version term is not an age. It is a *consumption counter* that happens to advance
once per training step. The mechanism is therefore:

> Pre-load $(S{+}1)$ batches of generation work, then release one further batch of
> admission credit each time training consumes a batch.

We call the invariant an **unconsumed-work lease**: the scheduler leases out at most
$(S{+}1)B$ units of work that training has not yet consumed, and each training step
renews the lease by one batch. "Capacity" is avoided as a name because SimRL already
uses it for KV-cache capacity, which is a different and simultaneously-active constraint.

### Why a lease cannot bound staleness

The lease bounds the number of outstanding units, which by Little's law bounds the
*mean* time a unit spends outstanding: $W = L/\lambda$ for arrival rate $\lambda$.
Staleness, however, is a property of the *tail*. A single long-running request occupies
one unit of the lease and keeps its slot indefinitely. The remaining $(S{+}1)B - 1$ units
continue to cycle fresh work through. Nothing in the inequality grows when that request
ages, so nothing evicts it. When it finally completes, it is consumed and trained on at
whatever version it happens to land in.

This is not a subtle failure mode. It is the behaviour the formula specifies, and
SkyRL's own source says so (see below).

## Two sub-families of lease

The lease-based systems split on how the invariant is maintained.

**Credit accumulation** (AReaL, SkyRL). State is a monotone cumulative counter that is
never decremented. If training stalls, the bucket drains and the producer hard-stops.

**Occupancy re-measurement** (verl). The bound is a *constant* $M$ with no version term.
Rather than accumulating credit, the counter is periodically reset to live occupancy at
each weight sync.

These coincide in steady state and diverge under stall. Under a training stall, AReaL's
producer blocks cleanly, whereas verl keeps re-measuring and relies on its bounded
hand-off queue, which evicts the **oldest** entry on overflow, so verl can shed exactly
the work closest to being consumed, precisely when it is most behind.

## Per-system instantiation

### AReaL: credit lease

`areal/infra/staleness_manager.py:109`:

```python
staleness_capacity = (ofp + current_version + 1) * consumer_bs - sample_cnt
```

with `sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running` and
`accepted` incremented, never decremented, in `on_rollout_accepted`. Consumed as an
integer count at `areal/infra/workflow_executor.py:803,811`:

```python
capacity = min(cap_staleness, cap_queue)
...
for _ in range(min(batch_size, capacity)):
```

The knob is `max_head_offpolicyness: int = 0` (`areal/api/cli_args.py:2859`). Its help
text reads *"If the current version is more than this many versions behind, the request
will not be accepted."* **No such check exists in the code.** A repo-wide search for a
comparison between a sample's generation version and the current version returns only a
`deterministic_sampling` warning at `cli_args.py:3036`.

AReaL has no L2 staleness control. A weight sync never interrupts in-flight generation
on a staleness basis. The `running` count above moves to `accepted` only when the
request itself finishes.

AReaL does use per-token versions, but in the **objective**, not in admission.
`areal/trainer/ppo/actor.py:1414-1420` computes
`alpha = (v_proximal - v_behave) / (v_theta - v_behave)` to interpolate proximal
log-probabilities for decoupled PPO. Its `RejectionSamplingConfig` filters on *policy
divergence* (ratio/KL), not age. AReaL's position is therefore coherent: it does not gate
on age because it corrects for age in the loss.

### SkyRL: credit lease, with an explicit admission of the gap

`skyrl/train/fully_async_trainer.py:184`:

```python
consumer_capacity = (self.max_staleness_steps + self._current_global_step) * self.mini_batch_size
producer_staleness_capacity = consumer_capacity - (self._stat.accepted + self._stat.running)
```

Identical to AReaL modulo the `+1`, which differs only because SkyRL's
`_current_global_step` denotes the version being worked on rather than the count of
completed versions. The source names the lineage: *"Modeled after AReal's
StalenessManager … The idea of this controller is from section 5.1 of AReal's paper."*
The knob is `max_staleness_steps: int = 4` (`skyrl/train/config/config.py:1035`).

SkyRL is the most valuable citation in this survey because it states the limitation in
its own class docstring (`fully_async_trainer.py:119-127`):

> This is enforced via a capacity rule, not a hard **per-group** staleness guarantee: we
> bound the **aggregate** number of groups that can be ahead of training so that, in
> **steady state**, staleness remains within the configured budget of
> `max_staleness_steps`.
>
> In pathological cases (e.g., very long-running trajectories), an individual group may
> take more than `max_staleness_steps` of training steps of time to finish generation.
> For such rare cases, we still accept the trajectory and log the staleness metrics with
> a warning.

And it goes further than the others: it *computes* the per-group age and *compares* it,
then declines to act. At `fully_async_trainer.py:1038-1065`:

```python
cur_staleness = self.global_step - cur_generated_output_group.global_step_when_scheduled
...
generator_outputs.append(cur_generated_output_group.generator_output)   # line 1041
...
if cur_staleness > self.max_staleness_steps:                            # line 1061
    # TODO(Charlie): should we drop, drop and resample, or just log?
    logger.warning("Staleness control violated despite using AsyncStalenessManager: ...")
    staleness_violation_count += 1
```

The group is appended to the training batch at line 1041, *before* the check at line
1061. The violation branch contains no `continue` and no removal, only a counter that
feeds the `async/staleness_violation_count` metric. SkyRL's documentation lists dropping
and drop-and-resample as alternatives, both marked "not supported yet".

The design rationale is stated explicitly, and is the cleanest articulation of the
trade-off found in any of the six:

> One could instead over-generate and drop groups that are too stale at training time. We
> prefer proactive admission control to avoid wasted compute and simplify buffer
> management.

### verl (legacy sync mode): occupancy lease

`verl/experimental/fully_async_policy/fully_async_rollouter.py:431`:

```python
self.max_required_samples = int(
    self.required_samples * (self.staleness_threshold + 1)
    * self.config.async_training.trigger_parameter_sync_step
)
```

Gating is two counting tests in `_should_pause_generation` (`:988`, `:996`): queue depth
against `max_queue_size`, and `staleness_samples` against `max_required_samples`. At each
weight sync the counter is re-measured rather than credited (`:475`):

```python
self.staleness_samples = len(self.active_tasks) + await self.message_queue_client.get_queue_size()
```

The hand-off queue is a `deque(maxlen=...)` that `popleft()`s on overflow: FIFO
eviction, which will discard a *fresh* sample if that is what sits at the head.

verl is the most honest of the lease systems about what its knob means.
`staleness_threshold: float = 0.1` and the README defines it as *"the maximum proportion
of stale samples allowed to be used"*: a proportion, not an age. verl also never filters
by age at training time: the per-sample tags `min_global_steps` / `max_global_steps` are
introduced in `verl/trainer/ppo/v1/agent_loop_tq.py:191` with the comment *"These tags
are used for off-policy staleness control, if a trajectory spans too many global steps,
we need to filter it out"*, but their only non-test consumer is
`detach_utils.py:151-159`, which computes `partial_ratio`, `max_partial_span` and
`param_version_diversity` for logging. **No filter exists.**

### verl v1 (`separate_async`): step-clocked admission, a real age filter

Set via `trainer.v1.trainer_mode=separate_async`, this is an unrelated mechanism to the
legacy sync mode above, in the same repository.

**L1.** Admission tracks training progress, not an outstanding-work counter.
`num_warmup_batches` (fractional) sets how many extra batches of prompts are admitted
before training starts. Every subsequent training round admits one more
`train_batch_size` on top of that
(`verl/trainer/ppo/v1/trainer_base.py:624-651,1520-1527`, commit `5b550b0c`). With
`train_batch_size=4, num_warmup_batches=1.5`: 6 prompts admitted at startup, 4 more at the
first round (10 total), then 4 per round after.

**L2.** None.

**L3, the actual enforcement path.** Each admitted prompt's `global_steps` at admission
time is recorded (`prompt_global_steps`, keyed by uid). At training-sample time,
`ReplayBufferAsync` compares the current `global_steps` against it, under one of two
strategies:

`drop` evicts a finished group past the threshold before it can be sampled
(`replay_buffer.py:520-539`):

```python
if global_steps - prompt_global_steps.get(uid, global_steps) + 1 > self.max_off_policy_threshold
```

The `+1` counts the admission round itself. With one update per training round, `drop`'s
staleness bound is `max_off_policy_threshold - 1`. `parameter_sync_step > 1` runs several
sample-and-update cycles inside one `global_steps`, splitting the batch into minibatches
that all share that single version. This does not scale the bound. A conservative upper
bound folding it in is `parameter_sync_step * max_off_policy_threshold - 1`.

`wait` blocks sampling while any *still-generating* group has reached the threshold
(`replay_buffer.py:541-556`), giving it time to finish. Already-finished groups past the
threshold are never evicted. A finished group not drained this round is not lost: the
next round prioritises earlier-admitted groups first, so it is the first one sampled, now
one version staler. `wait`'s own test
(`tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py:676-691`) confirms this: with no
groups still generating, an already-finished group past the threshold is still returned.

**Analysis.** `drop` evicts before sampling and its bound folds cleanly into a per-token
staleness ceiling, so it counts as a strict bound in this survey's terms. `wait` only ever
blocks on in-flight work. It never removes an already-finished over-age group, so nothing
stops one from eventually training at an arbitrarily late version. Setting
`max_off_policy_threshold=None` disables both L3 strategies, leaving L1 as the only bound.

### slime: no knob, staleness is structural

slime has no staleness *parameter* at all, but it does gate admission on a watermark.
The fully-async object-store worker converts a concurrency budget (concurrent responses
divided by responses per prompt) into a group cap, optionally scaled up by a prefetch
factor (`slime/rollout/fully_async_rollout.py:199-223`). Admission checks both the
active-group count against that cap and the completion queue's depth against the same
watermark: once either is full, admission pauses until training drains the queue. With
64 concurrent responses, 8 per group and a prefetch factor of 1, that caps active
generation at 8 groups and pauses admission once the completion queue also holds 8. The
queue itself is unbounded, so an already-running trajectory can finish and push the queue
past the watermark. Only new admission is blocked. This bounds *how much unconsumed work
exists*, the same shape as AReaL/SkyRL/verl's lease, not the age of any one sample.

`Sample.weight_versions` has five read sites in the repository: one metric, two *sort
keys*, one telemetry field, and the write itself. A search for a comparison operator
applied to a version or staleness quantity returns nothing. The documentation states the
consequence twice, e.g. `docs/en/advanced/rollout-scheduling.md`: *"Ordering does not
automatically discard stale samples."*

Beyond the watermark, staleness is emergent from the loop barrier: generation is
dispatched before a weight push and consumed after it, so the bound is set by the
pipeline shape. The knob that genuinely shapes off-policyness here is
`--flush-cache-interval` (`slime/utils/arguments.py:350`, `int`, default 1), which selects
whether in-flight requests are aborted at a weight update or paused in place with their KV
preserved.

Two incidental findings relevant to SimRL's cost models: slime's resume is
budget-correct (`max_new_tokens -= response_length`), so it does not exhibit the
fresh-budget-on-resume behaviour seen in PSRL. With `flush-cache-interval <= 0`, the
KV also survives the weight update in place, so partial rollout does **not** pay a re-prefill
tax.

### ROLL: pipeline depth plus a genuine age filter

ROLL is the first of the two systems that bounds age. Its knob is
`async_generation_ratio: float = 0` (`roll/configs/base_config.py:552`), and it is used
for two different purposes in two different schedulers.

In the agentic path (`roll/distributed/scheduler/rollout_scheduler.py`), it is both a
pipeline depth and an age bound:

```python
for _ in range(self.async_generation_ratio):          # line 277  - depth
    ...
if step - group.create_step > self.async_generation_ratio:   # line 283  - age
    expired_episodes.append(episode_id)
```

Expired groups are popped permanently. The same comparison guards batch retrieval in
`GroupQueueManager.get_batch`, and `put()` discards results from groups already evicted.
`GroupData.create_step` is the group's birth step, so `step - create_step` is a genuine
age in training steps.

In the RLVR path (`roll/distributed/scheduler/generate_scheduler.py`) the first-round
budget is `ceil(B*r) + B` for batch size `B` and ratio `r`, set once in `advance_step`
(`:244-249`): `self.batch_size = math.ceil(batch_size * self.async_generation_ratio)`
followed by `self.batch_size += batch_size`. Every later round only adds one more `B`.
With `B=32, r=2`: a first-round budget of 96, then +32 per round after. The window is
enforced both at consumption (`min_step`, `:358`) and by proactive GC (`max_gc_step`,
`:425`).

**The RLVR path requires `r >= 1.0`, enforced at config time, not merely recommended.**
`roll/pipeline/rlvr/rlvr_config.py:222`: `assert self.async_generation_ratio >= 1.0,
"async_generation_ratio must be >= 1.0"`. `generate_scheduler.py` itself only asserts
`>= 0`, since it is shared with the agentic path below, which has no such floor. RLVR's
own pipeline config is what actually tightens the bound before the scheduler ever sees a
sub-1.0 value.

**ROLL supports fractional values, deliberately, in the RLVR path.** The fractional part
selects what proportion of the batch may be drawn from the oldest eligible cohort
(`:376-379`):

```python
if self.async_generation_ratio % 1 == 0:
    expected = expected_samples - collected_samples
else:
    expected = int(expected_samples * (self.async_generation_ratio % 1))
```

with `math.ceil(self.async_generation_ratio)` setting the window edges. This is a real
fractional staleness semantics: $S = 1.5$ means "one full step back, plus half a batch
from two steps back." Note the asymmetry: the agentic `rollout_scheduler.py` path uses
the same field in `range()` at line 277, which requires an integer and raises `TypeError`
on a fractional value. The field is declared `float` with no `__post_init__` validation
beyond `assert async_generation_ratio == 0 or generate_opt_level == 1`
(`base_config.py:727`), so which behaviour you get depends on which scheduler runs.

### Dressage: age as a first-class feature

Dressage is built on slime and adds what its README calls **"True Staleness Control"**.
That a framework built on slime found it necessary to add this, and to call it *true*, is
itself evidence that the gap documented here is recognised in the field.

`dressage/rollout/staleness.py`:

```python
@dataclass(frozen=True)
class StalenessConfig:
    keep_versions: int | None = None          # line 13

@property
def cutoff_version_index(self) -> int | None:
    return max(0, len(self.versions) - int(self.config.keep_versions))   # line 100

def should_drop_group(self, group: list[Any]) -> bool:                   # line 121
    cutoff = self.cutoff_version_index
    if cutoff is None:
        return False
    return any(self.version_index(info.version) < cutoff
               for info in trajectory_version_infos(group))
```

`trajectory_version_infos` is doing more work than the `any(...)` loop above suggests: for
each distinct trajectory key in the group it first collapses every segment down to the
one with the highest `segment_index`, then reads that segment's own version
(`dressage_end_token_version`). So `should_drop_group` is comparing each trajectory's
**last-trainable-token version** against the cutoff, one comparison per trajectory in the
group, not a raw scan over every observed segment. A trajectory whose first segment is
ancient but whose last segment is recent passes. One whose own last segment has fallen
behind does not, regardless of how recent its earlier segments were.

Three design choices are worth noting:

1. **Versions are tracked in observation order**, appended as encountered by
   `observe_group` (`:111`), never sorted, never subtracted. `keep_versions` counts
   *distinct observed versions*, not steps. This is the only system in the survey where
   an integer type is semantically correct rather than an artefact of `range()`.
2. **Drop granularity is the GRPO group**, deliberately: the documentation explains that
   removing individual trajectories would perturb group-relative advantage
   normalisation.
3. **Dropped groups are topped up, not left to shrink the batch.** The collector's outer
   loop (`generate_rollout_async`, `fully_async_rollout.py`) is a plain
   `while len(data) < target_data_size`: a group dropped by `keep_group` or by a
   re-check in `filter_pending` after the version advances is simply never appended, and
   the loop keeps pulling newly-completed groups until the target size is met regardless
   of how many were dropped along the way. This is the same shape as ROLL's resampling,
   not the batch-shrinking behaviour this document previously attributed to Dressage.

Dressage also bounds *intra-trajectory* version mixing, which no other surveyed system
does. Under partial rollout a single trajectory can be interrupted and resumed across
weight updates. The proxy enforces `--max-partial-rollout-preempts` with
`version_switches = max(0, version_span - 1)`, returning HTTP 502
`partial_rollout_staleness_exceeded` when exceeded.

The filter runs at batch-assembly time, so a stale group has already consumed GPU time
before being dropped. The work is wasted, not prevented: the cost that SkyRL's
admission-control rationale is explicitly trying to avoid.

## Other systems surveyed

prime-rl, TRL and NeMo-RL below were read end-to-end against the commits in the sources
table, same depth as the six systems above. They are kept in this section because each
needed less narrative to cover. OpenRLHF's knob and enforcement path are verified.
Trinity-RFT carries the sub-survey provenance caveat noted above.

**prime-rl** (`PrimeIntellect-ai/prime-rl`) enforces the strictest bound found.
`max_off_policy_steps: int = Field(8, ge=0)` and
`min_fresh_version(step, S) = (step - 1) - S` (`orchestrator/utils.py:42-45`). The `step`
argument is the *training* step about to run, not the version it trains with: training
at step `n` uses the policy published after step `n-1`, i.e. version `v(n-1)`, so the
bound reads as "no episode dispatched before `v(n-1) - S` may train in this batch."
Crucially, `dispatcher.py:427-439` cancels groups that are **still generating**:

```python
stale_groups = [... if group.policy_version_at_start < min_version]
for gid in stale_groups:
    cancelled += await self.drop_group(gid, reason="stale")
```

`drop_group` cancels the in-flight asyncio tasks and releases their permits. prime-rl
*also* gates admission on a capacity limit (`max_inflight`), so it enforces both a work
bound and an age bound simultaneously. It additionally decomposes staleness into
generation time versus queue time (`off_policy/in_flight/*` versus
`off_policy/in_queue/*`), a distinction SimRL can reproduce directly.

**TRL** (`huggingface/trl`, `trl/experimental/async_grpo/`) uses
`max_staleness: int = 4`, help text *"Maximum number of weight update steps a rollout
sample can lag behind the current model version before being discarded."* Enforced at
consume time (`async_grpo_trainer.py:368-372`): `staleness = version - sample.model_version`,
then `continue` to drop. `sample.model_version` is stamped once, when a `RolloutGroup` is
first created (`async_rollout_worker.py`, `pending_groups[group_id] = RolloutGroup(...,
model_version=self.model_version, ...)` guarded by `if group_id not in pending_groups`)
and only ever propagated from the group to its individual samples afterward
(`model_version=group.model_version`), never re-read from the live model version at
generation completion. A group that takes several weight syncs to finish generating is
staled against the version it started under, not the version current when it finally
lands. Notably, its capacity knob is *derived from* the age knob: `max_inflight_tasks`
defaults to `max(max_staleness, 1) * samples_per_step` (the floor keeps `max_staleness=0`
from zeroing out the rollout entirely, per the source comment next to it), on the stated
rationale that generating more would be wasteful since the surplus would be discarded as
stale. This is the inverse of the lease family's reasoning.

**NeMo-RL** (`NVIDIA-NeMo/RL`) exposes staleness as a pluggable policy: a discriminated
union of `windowed`, `ready_first`, `weight_fifo` and `in_order` samplers, each with
`max_staleness_versions: NonNegativeInt = 1`. **The real system's default is `in_order`**
(`single_controller_utils/config.py`: `sampler: SamplerConfig =
Field(default_factory=InOrderSamplerConfig)`), not `windowed`. This is worth stating explicitly,
since `windowed` gets the most narrative attention here for having the most direct L1,
L2 and L3 coverage. `in_order` is pure pipeline depth. `windowed` is a pure age window
with an explicit `should_abort_inflight()`. Backpressure knobs (`max_inflight_prompts`,
`max_buffered_rollouts`) are kept deliberately separate from the staleness knobs. This is
the closest external analogue to SimRL's own pluggable-policy design.

**OpenRLHF** has `--train.async_queue_size: int = 1`, implemented as a literal counting
semaphore. No per-sample version is recorded anywhere, so staleness is bounded only
implicitly by queue depth: the degenerate lease, with $S$ fixed at the queue size.

**Trinity-RFT** (`agentscope-ai/Trinity-RFT`) has `sync_offset: int = 0` (structural) and
`max_inflight_batches: int = 2` (capacity). It tracks `model_version` carefully at the
synchroniser level but attaches no version to individual experiences, so no age
comparison is possible. This is the purest instance of the "tracked but never compared"
pattern.

## Summary

| System | Knob | Type | Fractional | Bounds | Enforcement |
|---|---|---|---|---|---|
| slime | *(none)* | n/a | n/a | pipeline shape | structural, versions are sort keys |
| OpenRLHF | `async_queue_size` | `int` = 1 | no | outstanding work | semaphore, no versions recorded |
| Trinity-RFT | `sync_offset`, `max_inflight_batches` | `int` = 0 / 2 | no | depth + work | structural, versions never compared |
| AReaL | `max_head_offpolicyness` | `int` = 0 | no | outstanding work | credit lease, age corrected in loss, no L2 |
| verl (legacy) | `staleness_threshold` | `float` = 0.1 | **yes** | outstanding work | occupancy lease, FIFO drop on overflow |
| verl v1 | `max_off_policy_threshold` | `int` = 8 | no | age (`drop`) or none (`wait`) | `drop` evicts before sampling. `wait` only blocks in-flight |
| SkyRL | `max_staleness_steps` | `int` = 4 | un-validated | outstanding work | credit lease, violation logged, not acted on |
| ROLL | `async_generation_ratio` | `float` = 0 | **yes** (RLVR path) | depth **and** age | GC + consumption floor, resamples |
| Dressage | `..._keep_versions` | `int` = 0 | no | age (inter + intra) | drop at assembly, no resample |
| TRL | `max_staleness` | `int` = 4 | no | age | drop at consume, capacity derived from it |
| NeMo-RL | `max_staleness_versions` | `int` = 1 | no | age or depth (pluggable) | window evict, optional in-flight abort |
| prime-rl | `max_off_policy_steps` | `int` = 8 | no | work **and** age | **cancels in-flight** + capacity gate |

## The two axes

A one-dimensional "how strict is the bound" ordering does not fit the evidence. Two
orthogonal axes do.

**Where the bound is applied.** *Ex-ante*, by throttling admission, or *ex-post*, by
discarding work that has already been generated.

**What quantity is bounded.** The *volume of outstanding work* (a count), or the *age of
a sample* (a version difference).

```text
                     bounds work volume              bounds sample age
   ex-ante           AReaL, SkyRL, verl (legacy),    prime-rl (in-flight cancel),
                     OpenRLHF, Trinity-RFT           NeMo-RL (windowed sampler)
   ex-post           --                              ROLL, Dressage, TRL, verl v1 (drop)
```

The populated upper-right cell is the important correction to the naive story. Bounding
age *before* the compute is spent is possible, and prime-rl and NeMo-RL's `windowed`
sampler both do it, by cancelling work that has already started but cannot finish in
time. Note what this requires: the decision is made *reactively*, when a new weight
version arrives and a still-running request is observed to have aged out. Neither system
predicts at dispatch time whether a request will survive. They wait until it has
demonstrably failed to.

That distinction is where SimRL has something to say. Reactive cancellation still
pays for everything generated up to the cancellation point: the sunk cost is reduced,
not eliminated. Deciding at *admission* time which requests will still be fresh on
completion requires a predicted completion-time distribution, which is exactly what a
simulator provides and what a live system cannot cheaply obtain.

The two rationales are stated explicitly by the systems themselves, and they conflict.
SkyRL: *"We prefer proactive admission control to avoid wasted compute."* TRL derives its
capacity cap from its age bound on the same reasoning, but in the opposite direction,
sizing the in-flight pool so that few samples *can* age out. Both are trying to avoid the
same waste. They disagree about which quantity to fix.

## Implications for SimRL

Eleven of the twelve systems surveyed above now have a corresponding `StalenessPolicy`
under `srl/policy/` (`POLICY_REGISTRY` in `srl/policy/registry.py` lists eleven, plus
`psrl` for SimRL's own baseline, twelve entries total, not twelve-of-the-survey).
Trinity-RFT is the one exception: it tracks `model_version` at the synchroniser level
but never attaches a version to an individual experience, so there is no per-sample
quantity to key a policy on (see its "structural" classification above). The layered
`Cancel`/`on_publish` mechanism that every runtime-cancelling policy shares replaced an
earlier, narrower `expire()` hook entirely, so there is no longer a per-system "this one
needs a hook that does not exist yet" gap among the eleven that are implemented.

- `ArealPolicy` (`srl/policy/areal.py`) reproduces AReaL's formula exactly, including the
  `accepted + running` term, and declares no L2. The policy documentation's claim that it
  "does not bound individual sample age" is correct. The sharper statement is that no
  per-sample age check exists anywhere in AReaL.
- `RollPolicy`'s consumption floor and its `on_publish`-triggered GC correspond to real
  upstream mechanisms, not modelling liberties, including the `r >= 1.0` floor the RLVR
  pipeline config enforces, which this simulator does not currently assert on its own
  `RollPolicy` construction. This also explains why binding SCG experiments to
  `policy=roll` surfaced wasted tokens that other policies hid: ROLL is one of several
  surveyed systems that discards aged work, and the first one this project implemented.
- `DressagePolicy` (`srl/policy/dressage.py`) already keys its window on each group's
  last trainable token (`last_version = max(req.v_max for req in group)`) and already
  grants a replacement credit on every L3 drop (`replacement_groups_on_drop` returns
  `len(groups)`), both match the corrected real-system behaviour documented above. An
  earlier draft of this document claimed Dressage does not resample after a drop. It
  does, and the policy was already modelling that correctly before this correction.
- A faithful **SkyRL** policy is `areal` without the `+1`, plus a violation counter that
  does not alter behaviour. `SkyRLPolicy` implements exactly this.
- `VerlV1Policy` implements the corrected `drop`/`wait` distinction above: `drop`'s
  threshold folds into a strict per-token bound, `wait` only blocks on still-generating
  work and is reported as not strict.
- slime should not be modelled with an age-based policy at all. `SlimePolicy`'s bound
  comes from the loop-barrier shape plus the admission watermark documented above, not
  from any version comparison.
- `PrimeRLPolicy` combines a flat inflight cap with a dispatch-lag admission gate, plus
  the shared `CancelBelowVersion`/`VersionWindow` primitives for the in-flight cancel and
  ready-pool sweep respectively, the lease-plus-reactive-cancel combination this document
  used to call unimplementable.
- prime-rl's split of staleness into generation time versus queue time is worth adopting
  as a reported metric. It distinguishes "the request was slow" from "the request waited",
  which the current single staleness figure conflates. `StalenessReport` does not yet
  make this split.

## Reproducing this survey

GitHub's code-search API and `git clone` are blocked from the cluster, but `curl` to
`codeload.github.com` is not:

```bash
mkdir -p /tmp/src && curl -sL \
  https://codeload.github.com/<owner>/<repo>/tar.gz/refs/heads/main | tar xz -C /tmp/src
```

Fetching the tarball and grepping locally is both faster and more reliable than
per-file web fetches, which truncate long files and summarise rather than quote.
