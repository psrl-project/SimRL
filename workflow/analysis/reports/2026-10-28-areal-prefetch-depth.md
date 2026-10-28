# Diagnosis report

## 1. What was diagnosed

Experiment: `examples/simulate/staleness_throughput/areal` (sweep of
`policy.max_head_offpolicyness` 0..5, one run per point via `psrl_aligned.sh` under
`common.sh`). There is no `summary.json`; the rows are the sweep's own run directories
`outputs/simulate/staleness_throughput/areal/p{2,5}`. Rows: `p2` (best consumed throughput of the
sweep) and `p5` (deepest prefetch, 11.7% lower). The question: why does deeper prefetch lose
when it should allow more concurrent generation. `roll` p2 vs p5 checked from its existing logs
only (no replay).

Workload: Qwen3-8B DAPO trace capped at 38912 (`qwen3_8b_dapo_38912_tp1`), 2 x TP1 H20 rollout
instances (500,192 KV tokens each), TP2/PP3 training, batch 8 groups x 8, 10 versions,
`pause_mode=abort` (areal default).

## 2. Session

Git head `cd0a2c8`. Engine `.so` md5 `4c9a6244aeaf`. Replays with full telemetry into
`/tmp/simrl_analysis/areal_ofp/p{2,5}` (about 2 min each). Counterfactuals in
`/tmp/simrl_analysis/areal_cf/`, injected through `/tmp/cf_site/sitecustomize.py` (env-gated
`ArealPolicy.pause_mode` and `SimConfig.kv_blocks` overrides, no repo change).

The shared `models/Qwen3-8B` directory was emptied at 14:40 during this session (not by this
session). Counterfactual runs point `MODEL_PATH` at `/dev/shm/verl-cache/8f1c.../Qwen3-8B`, whose
`checkpoint_fingerprint` (`a9c7ae522e3e25f6`) matches the cached 31262-block KV entry.

## 3. Fidelity

| row | `request.jsonl` | `train.jsonl` |
|---|---|---|
| p2 telemetry replay vs sweep | md5 identical | md5 identical |
| p5 telemetry replay vs sweep | md5 identical | md5 identical |
| p2 counterfactual control (patch inactive, `/dev/shm` checkpoint) vs sweep | identical | identical |
| p5 counterfactual control vs sweep | identical | identical |

Cross-source check: sum of every request's final `generated` (request + instance streams) vs
`rollout_step` `n_decode`: p2 7.643M vs 7.644M, p5 8.234M vs 8.256M (0.26%, unresolved, see 8).

## 4. Where the span went

| row | span | train | trainer idle (waiting on rollout) | consumed | decoded by fleet | unconsumed at end |
|---|---|---|---|---|---|---|
| p2 | 3362 s | 1063 s | 2299 s (68%) | 6.11M | 7.64M | 1.53M (20%) |
| p5 | 3811 s | 1057 s | 2754 s (72%) | 6.12M | 8.26M | 2.12M (26%) |

Both rows are rollout-bound and both fleets are busy over 99% of the span. The consumed set is the
same (mean consumed trajectory 9548 vs 9558 tokens). The binding-request phase decomposition was
not computed: the question is fleet rate, not which request binds a step.

## 5. Bounds

Not computed. `span_solo_all` ignores fleet capacity, and this loss is a fleet-capacity effect.

## 6. Fleet state

| row | time-avg running / waiting per instance | KV util | KV >= 97% | mean decode batch | mean engine step | fleet decode | prefill computed | engine time in prefill-carrying steps |
|---|---|---|---|---|---|---|---|---|
| p2 | 47 / 7 | 0.67 | 6% | 51.5 | 45.2 ms | 2274 tok/s | 6.12M | 18% |
| p5 | 66 / 74 | 0.88 | 76% | 74.9 | 68.7 ms | 2166 tok/s | 14.57M | 92% |

Re-prefill by cause (KV held at each non-final publish vs request `reprefill_tokens`): p2 5.85M,
all publish-abort rebuild (0.40-0.86M discarded per publish). p5 15.26M = 9.07M publish rebuild
(every publish discards a full 1.0M) + 6.19M KV-full preemption. Instance balance is even in both.

## 7. Paired diff (counterfactuals, p5 relative to p2)

| arm | s/step p2 / p5 | consumed tok/step p2 / p5 | fleet decode p2 / p5 | mean batch p2 / p5 |
|---|---|---|---|---|
| base | 336 / 381 (+13.4%) | 621k / 621k | 2274 / 2166 | 52 / 75 |
| no publish abort | 293 / 321 (+9.5%) | 621k / 621k | 2595 / 2587 | 52 / 75 |
| KV x4, abort kept | 333 / 396 (+19.1%) | 621k / 534k | - | - |
| KV x4, no abort | 295 / 293 (-0.7%) | 621k / 528k | 2595 / 2896 | 52 / 129 |
| base, 30 versions | windows +13.4% / +8.9% / +14.0% | full run -12.1% tok/s | - | - |
| no abort, 30 versions | windows +9.5% / +4.8% / +6.4% | full run -7.5% tok/s | per window 2595/2587, 2279/2088, 2143/2174 | - |

No-abort, 30-version ledger per window (decoded - consumed): p2 +1.39M, -0.02M, -0.73M;
p5 +2.08M, -0.12M, -0.11M.

## 8. Stops and failures

- First counterfactual batch failed: `models/Qwen3-8B` vanished, so `resolve_kv_blocks` missed the
  cache and tried to launch vLLM. Rerun against the `/dev/shm` copy (section 2).
- A least-squares fit of segment `dt` on (steps, decode, prefill) gave a negative decode
  coefficient (collinear regressors) and was discarded.
- Decode-token conservation is off by 21.4k (0.26%) on p5 and 0.5k on p2. Not root-caused.

## 9. Ranked hypotheses

1. **Concurrency above about 50 residents per instance buys no decode throughput here.** At mean
   resident context about 7k tokens, the engine step grows in proportion to the batch (no-abort
   arm: 52 -> 75 residents, 40.0 -> 57.7 ms, 2595 vs 2587 tok/s). Even with unlimited KV,
   129 residents give only +11.6%. Falsify: same pair on a short-output trace, where the step
   intercept dominates and the batch should pay.
2. **Publish abort scales with resident KV, and deep prefetch keeps KV full.** It accounts for the
   persistent part of the loss: the base vs no-abort gap is +17 s/step in 10 versions and about
   4.6 points of full-run throughput in 30. With 4x KV and the abort kept, p5 gets worse
   (+19%/step), because each publish then discards more. Falsify: `validate` the areal pair
   with `pause_mode=in_place` over a long horizon.
3. **The extra prefetch is work done ahead that a finite run never consumes.** p5 holds about
   0.7M more generated-but-unconsumed tokens. In the no-abort arm this alone explains +28 s/step
   over 10 versions, at equal fleet rate and an identical consumed set. It is a one-time fill that
   amortizes with horizon length, though it is still about 6% at 30 versions. Falsify: no-abort
   pair at 100+ versions should converge.
4. KV-full preemption (6.19M re-prefill in p5) mostly rides in mixed steps. With no abort, fleet
   decode is equal at 10 versions and 8% lower in window 2 of 30, so its separate cost is small
   and noisy.

## 10. Not established

Single seed, single trace, 2 instances. The figure's metric counts only consumed + discarded
tokens, so it penalizes deep prefetch by construction over a finite horizon. `roll` was diagnosed
from its existing ledger only (same re-prefill and preemption growth, 6.35M -> 14.89M and
302 -> 2880). Its steady consumed rate rises with the ratio, and its bar total falls mainly
because discards fall. No claim here that any configuration is faster in a real system.
