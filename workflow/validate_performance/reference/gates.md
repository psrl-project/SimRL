# Gates

A result is reportable only if it passes every gate. Gates exist because each one has already
produced a wrong speedup in this repo.

## Hard gates

### G1. Run completed

The run consumed every version it was asked for: `versions_done == n_versions`, and the result is
not livelocked. A run that stops early divides a smaller workload by a smaller span and can score
arbitrarily well.

### G2. Consumed groups

`consumed_groups == batch_size * n_versions`.

Why: consuming fewer groups means the scheduler stalled or the fleet drained, not that it was fast.
A stall shows up here first.

### G3. Tokens per group

`tokens_per_group` within 1% of the same point's control arm.

Why: a scheduler that reorders which groups are consumed can finish the same number of groups on a
different mix of lengths. Shorter mixes finish sooner for reasons unrelated to the improvement under
test. This gate has flagged real regressions.

### G4. Treatment differs from control

The treatment arm's result is not byte-identical to the control's.

Why: an identical result almost always means the flag never reached the code path. Both arms ran
the baseline. Report the wiring bug and fix it before reading anything into the comparison.

### G5. Same simulation binary

Record the md5 of the simulator extension (`srl/**/*.so` when present) before and after the sweep.
All arms in one comparison must share it.

Why: another session rebuilding the extension in place changes the model under a running sweep.

## Baseline discipline

Not gates on a single result, but on the set of results.

- **Controls run in the same session.** Same workload, same model, same TP, same binary.
- **Static caps are swept fresh at every point.** A cap that was best elsewhere is a starting point,
  not a baseline. Use the same grid at every point.
- **Report the deployable baseline and the oracle separately.** The deployable one is a single fixed
  configuration applied to every point. The per-point best is an oracle and exists only to show how
  much headroom a per-point choice would buy.
- **Knobs tuned on the reported points do not count as generalization.** If a value was chosen
  using a point, generalization must be shown on points that played no role in choosing it.

## Failed results

List each failed point and the gate it failed, then leave it out of every table and mean. Do not
annotate a failing row and keep it in the aggregate.
