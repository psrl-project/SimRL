# Cadence-Aware Lane Scheduling for Async PSRL with Partial Rollout

## Problem

Under async PSRL with partial rollout, the quantity that matters is not one step's
makespan but the total wall time of K training steps. Training and rollout are sized to
run at about the same average speed, so any idle time on either side lengthens the run.
The trainer idles when buffer k is not ready by the time training on buffer k-1 ends. The
fleet idles when every one of the S+1 pending buffers is fully reserved, so staleness
blocks further admission while the trainer is still waiting on a straggler.

Objective (user decision, 2026-10-26): **minimize the end time of the K-th training step**.
Step-interval variance and any oscillation period are diagnostics, not targets.

## What the baseline shows

Setup: psrl default router, S=2, 16 x TP1 Qwen3-8B on H20, fixed 190 s training, 16 steps,
on the real `outputs/workload/qwen3_8b_dapo_38912_tp1` trace (1024 groups x 8, cap 38912).
Run directory: `outputs/simulate/cadence_probe/baseline_s2_n16_tau190`.

- Span 4418 s, against 16 x 190 = 3040 s of training. The trainer is idle 31% of the run.
- Each instance is busy only 84% of the run on average.
- Every buffer is gated by its last group to finish, and that group's slowest member is
  always 26.6k-38.9k tokens long. It runs 400-850 s from admission at 10-23 ms/token,
  because it shares an instance with about 48 others. Alone on TP1 the same request
  decodes at about 7 ms/token.
- The trainer and the fleet are idle at the same time. The problem is the tail
  request's latency, not the fleet's total throughput.
- Idle time per step was `213, 0, 256, 0, 65, 315, 0, 0, 109, 150, 0, 0, 158, 112, 0, 0`.
  This does not show a clean period-2 oscillation. The period hypothesis is tested, not
  assumed.

## Mechanics the design relies on

1. **The window each group has.** A group admitted at version v reserves an entry in the
   furthest pending buffer, v+S. When it completes, it occupies the earliest pending
   buffer in [v, v+S], and an earlier reservation slides back into its slot
   (`psrl/workers/ps/staleness/api.py:162-232`). Reservations only ever move later. Short
   groups therefore fill the early buffers, and a long group's hard deadline is buffer
   v+S. When the trainer never idles, that leaves it a window of about S x tau from
   admission.
2. **Every decision falls at a publish.** psrl opens new reservation capacity only when
   a version is published, so every admission happens at t=0 or at a publish. The
   baseline confirms this: its admit times are exactly the publish times.
3. **Re-placement at a publish costs nothing extra.** `Simulator._publish_weights`
   aborts every in-flight request (`simulator.py:1061-1065`), which discards its KV, and
   hands all of them back to the router. They are re-placed together with the newly
   admitted groups in one dispatch burst, through the strategy's `select_context`, with
   no stickiness. Moving a request to another instance at that moment adds nothing on
   top of the re-prefill it was going to pay anyway.

Together, 2 and 3 mean the entire placement can be re-planned from scratch at every
publish. Between publishes the router only places requests that were held back in its
queue.

## Decision space

What the user allowed:

- **Lanes with different concurrency.** Every instance uses the same TP. An instance can
  act as a fast lane, holding a few requests at low per-token latency, or as a bulk lane,
  holding many requests for high throughput.
- **Re-placement at the version boundary**, by remaining length and deadline.
- **Holding requests back.** A request may stay in the router queue even though KV has
  room.

What is excluded:

- Heterogeneous TP.
- Admitting groups from a later batch early, so admission order stays dataset order.
- Any change to psrl's reserve, occupy, or consume rules.

## Algorithm A: slack lanes (primary)

The planner runs once per dispatch burst that follows a publish (and once at t=0). It
works on every request in flight at that moment: the aborted ones and the newly admitted
ones. Remaining lengths come from the oracle.

1. **Bulk-only plan.** Assign every request to the N instances by LPT on remaining tokens.
   An instance accepts only while its forecast KV peak fits. Requests that do not fit
   stay queued, in priority order.
2. **Price the plan.** For each instance, `InstanceForecast.finish_times` predicts when
   each member finishes. A group's ETA is its slowest member's finish time, with members
   that already finished counted as done. `PipelineForecast.evaluate(group_eta)` turns the
   group ETAs into training-finish times `E_k` for every pending buffer.
   - Objective: `J = sum_k E_k` over the pending horizon. Each `E_k` is the end of a
     training step, so J directly measures objective A within the horizon.
3. **Greedy upgrade.** Find the earliest buffer whose `E_k` exceeds `E_{k-1} + tau_k`,
   which means the trainer would idle before it. Its critical group is the one whose
   completion sets `F_k`. A move takes that group's members still finishing after the
   target and puts them in the fast lane, in one of two ways:
   - add them to an existing fast instance that holds fewer than `c_max` members; the
     forecast also prices the slowdown of the members already there;
   - turn one bulk instance into a new fast instance, then re-run LPT over the remaining
     bulk instances.

   Every candidate move is priced with J. The best move is accepted if it lowers J by
   more than `eps_frac` of one training cycle. If no move helps that buffer, the group is
   marked unfixable and the planner moves on to the next late buffer. It stops when no
   late buffer can be improved, or after `max_moves` moves.

   The number of fast instances and their occupancy are not configured; they come out
   of this search. The only bounds are `c_max` (default 4) and N-1 fast instances.
4. **Follow the plan.**
   - `narrow_candidates`: a fast-lane request can go only to its planned instance. A
     bulk request can go only to bulk instances.
   - `select_context`: choose the planned instance. If that is not possible, choose the
     bulk instance with the fewest remaining tokens.
   - `pick_queued`: requests the plan held back are ordered by group slack, smallest
     first.
   - Once every planned member of a fast instance has finished, that instance becomes
     bulk again and may take queued bulk work.

Two variants isolate where any gain comes from:

- **A-nolane** runs step 1 and step 4 only, with no upgrades. It separates the value of
  the fast lane from the value of plan-based LPT balancing.
- **A+cap** adds the best uniform bulk cap from baseline B1, so A gets the same "hold at
  the knee" lever the strongest baseline has.

## Algorithm C: static length lanes (ablation)

The first k instance ids are permanently fast. At every placement:

- A request with predicted remaining length of at least L goes to the least-loaded fast
  instance with fewer than `c_f` members. If none qualifies, it goes to bulk.
- Every other request goes only to bulk instances, choosing the one with the fewest
  remaining tokens.

The grid is k in {1, 2, 4}, L in {20000, 30000}, and `c_f` in {2, 4}, for 12 arms. Two
numbers are reported:

- the best arm per point, which is an in-sample, optimistic upper bound for C;
- the single arm that is best across all points, which is the honest number.

If C matches A, upgrading by slack adds nothing over a static length threshold.

## Algorithm B: cadence target (conditional)

B is built only if the decision rule below triggers.

At each publish, B binary-searches the smallest period P* in
[mean tau, current mean period]. P* must leave a feasible plan when deadlines are spaced
uniformly: `D_k = E_now + (k - k0 + 1) * P`.
- A plan is feasible if, under the same move set (fast lanes filled earliest deadline
  first, LPT in bulk), every pending buffer meets its `D_k`.
- B adopts that plan.

The difference from A: A improves J greedily from the nearest late buffer outward. B
commits to uniform spacing, so it can give up a little on the nearest buffer to protect
the later ones.

## Bounds and headroom

For every point:

- **LB_roll** is the tokens consumed by K steps divided by the fleet's saturated
  throughput `T_sat`.
- **LB_phys** comes from a contention-free replay, built on psrl's own
  `StalenessInventory`:
  - instances are unlimited, so no request shares one;
  - each request takes its alone-on-TP1 latency from the composed cost model, prefill
    plus decode;
  - training uses the same linear cost as the real run.

  This captures how far the tail physics and the training chain alone hold the run back.
- **LB = max(LB_roll, LB_phys).**
- **Headroom capture** for an arm X is `h = (T_B1 - T_X) / (T_B1 - LB)`.

**Decision rule for B:** build B if the median `h(A)` is below 0.5 over points where
`T_B1 - LB >= 0.05 * T_B1`.

## Experiment

- **Fleet:** 16 x TP1 Qwen3-8B on H20.
  - Composed cost model from `kernel_tp1`.
  - KV from the probe: 31155 blocks x 16 tokens.
  - `token_budget` 16384.
  - `weight_load_time` 0, same as the baseline.
- **Workload:** `qwen3_8b_dapo_38912_tp1`, B=32 groups x G=8, `max_output_length` 38912.
  - Three group-order permutations. Seed 0 is file order. The permuted traces are written
    under `outputs/workload/`.
- **Steps:** K=24. (K+S)*B <= 1024 holds for every S <= 3.
- **Training:** `training_cost=linear` with `time_per_token = rho / T_sat`.
  - `T_sat` is measured once per seed, from a rollout-bound run: tiny training cost, S=2,
    default router. It is the consumed training tokens divided by span.
  - At rho = 1 the trainer consumes tokens exactly as fast as the saturated fleet makes
    them.
- **Regimes:** S in {1, 2, 3} x rho in {0.9, 1.0, 1.1}, giving 9 regimes x 3 seeds = 27
  points.
- **Arms:**
  - Baselines:
    - B0: psrl default (`request_num_balance`, no cap).
    - B1: uniform `max_concurrent_seqs_per_instance` in {16, 24, 32, 48}, best per point.
  - Algorithm A: A-nolane, A, and A+cap.
  - Algorithm C: the 12 arms above.
- **Gates.** A run that fails any of these is dropped and listed, not explained away:
  - all K steps trained;
  - no livelock or crash;
  - consumed groups = K*B;
  - every consumed request has staleness <= S (strict);
  - tokens per consumed group within +-1% of B0 on the same point.

  Preemptions and re-prefill tokens are reported for every run.
- **Metrics:**
  - total time T and speedup against B1-best (primary) and B0;
  - trainer idle and fleet idle;
  - the step-interval series, with lag-1 and lag-2 autocorrelation, which tests the
    period hypothesis;
  - number of fast instances over time;
  - headroom capture h.

## Code layout

- **`srl/forecast/lanes.py`** holds `LanePlan` and `plan_lanes(...)`. It is pure over
  `InstanceState`/`Resident` and `PipelineForecast`, and it is where steps 1-3 of A
  live.
- **`srl/component/rollout/router/strategies/slack_lane.py`** is A, a thin adapter over
  `plan_lanes`. It sets `python_engine_only = True`.
- **`srl/component/rollout/router/strategies/length_lane.py`** is C.
- **`srl/diagnosis/cadence.py`** builds the cadence report from logs (idle split,
  intervals, autocorrelation) and runs the contention-free LB_phys replay.
- **Registry and docs:** both strategies are added to the registry in
  `strategies/__init__.py`, and to the `RouterConfig.strategy` docstring in
  `srl/config/schema.py`.
- **Tests:**
  - the planner, on hand-built fleets where the optimal lane split is known;
  - both strategies' candidate rules;
  - the bound replay checked against a hand-computed two-buffer case.
- **Driver:** `workflow/validate/experiments/cadence_lanes.py`, which is gitignored.
  Results go under `outputs/simulate/cadence_lanes/`.

## Out of scope

- Heterogeneous TP.
- Admitting groups from a later batch early.
- Real KV migration. Re-placement only happens where psrl aborts anyway.
- Length-prediction error. All lengths come from the oracle.
- A warmup knob. psrl fixes the initial admission at (S+1)*B, because reservations cannot
  exceed v+S, so warmup is not swept.
