# Diagnosis report

## 1. What was diagnosed

Experiment, `summary.json` path, rows selected and why. Workload shape in one line.

## 2. Session

Git head. Binary md5. Preflight result. Replay wall time per row.

## 3. Fidelity

Per row, the gated-field comparison. Every row reported here passed. A row that failed is in
section 7 instead, never here.

## 4. Where the span went

Per row: total span, train compute, total idle, and idle as a share of span. Then the idle
breakdown by the binding request's phases, as shares summing to 1.

## 5. Bounds

Per row and per bound: span, headroom, and the assumptions verbatim. State plainly whether the
headroom is large enough for a mechanism to be worth building.

## 6. Fleet state

Per row: the idle windows table (busy fraction per instance, mean queue depth, reserve-at-limit),
and the instance imbalance table.

## 7. Paired diff

For each point with both arms, the component-by-component change from control to treatment. This is
the section that explains a win or a loss.

## 8. Stops and failures

Replays that failed fidelity, checks that failed, timeouts with exact elapsed time. Empty is a
valid answer.

## 9. Ranked hypotheses

Each: the claim, its share of total idle time, the measurement supporting it, and the `validate`
experiment that would falsify it. No speedup claims.

## 10. Not established

What this diagnosis does not show. Single seed, single workload, assumptions of each bound.
