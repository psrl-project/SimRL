# Version-capped TP routing sweep

## Placement-time admission follow-up

The PSRL S=1, training-DP=1 comparison was rerun after moving `on_admit` from
dataset ingress to the first successful router placement. Each arm uses four
rollout GPUs, 20 versions, a 20k generation cap, and the same long-context
trace as the pre-fix run.

| batch | arm | pre-fix SCG(1) | placement-admit SCG(1) | change |
|---:|---|---:|---:|---:|
| 8 | TP1 capped | 1783.4 | 1789.3 | +0.3% |
| 8 | TP2 capped | 2019.4 | 2027.2 | +0.4% |
| 8 | heterogeneous capped | 1853.9 | 1813.5 | -2.2% |
| 8 | TP1 uncapped | 1783.4 | 1789.3 | +0.3% |
| 8 | TP2 uncapped | 2006.7 | 2012.0 | +0.3% |
| 24 | TP1 capped | 2709.4 | 2695.0 | -0.5% |
| 24 | TP2 capped | 2635.2 | 2627.6 | -0.3% |
| 24 | heterogeneous capped | 2535.6 | 2517.5 | -0.7% |
| 24 | TP1 uncapped | 2683.1 | 2675.5 | -0.3% |
| 24 | TP2 uncapped | 2746.0 | 2760.3 | +0.5% |

The heterogeneous ratio falls from 0.918x to 0.895x at batch 8 and from
0.923x to 0.912x at batch 24. The lifecycle fix therefore does not create a
heterogeneous win in this setting.

The new telemetry confirms that placement admission is active. At batch 24,
the capped TP1, TP2, and heterogeneous arms delay 1088, 2672, and 1880
responses respectively before admission; their maximum router dwell is
494 s, 874 s, and 692 s. Both uncapped arms admit immediately. No admitted
request rebases in these runs: PSRL waits for its current buffer before the
next publish, so blocked requests gain placement-time accounting but do not
cross a rollout-version boundary.

All arms still have exactly 5% of consumed requests at staleness 0 and 95% at
staleness 1, with no drops. Consequently SCG(1) remains identical to raw
throughput. Placement-time admission fixes policy inventory and timestamp
semantics, but it cannot improve SCG when every consumed token already counts;
the remaining result is the fixed fleets' throughput ordering.

Machine-readable results:

```text
outputs/simulate/scg_14b_4gpu_psrl_s1_placement_admission/summary.json
outputs/simulate/scg_14b_4gpu_psrl_s1_placement_admission/summary.md
```

## Longer-training PSRL S=1 sweep

The prior PSRL S=1 run used training DP=4. In the heterogeneous arm, measured
training occupied only 14.4% of the total span, or 18.2% including recompute.
This follow-up uses training DP=2 and DP=1 while keeping the rollout fleet,
workload, five capped/uncapped arms, and 20 versions unchanged.

The intended training-share increase was achieved:

| batch | training DP | train / span | (train + recompute) / span |
|---:|---:|---:|---:|
| 8 | 4 | 14.4% | 18.2% |
| 8 | 2 | 22.2% | 28.1% |
| 8 | 1 | 35.9% | 45.3% |
| 24 | 4 | 14.4% | 18.2% |
| 24 | 2 | 25.6% | 32.3% |
| 24 | 1 | 45.3% | 57.3% |

All 20 new arms still have maximum true staleness exactly 1.

### SCG(1) trend

| batch | training DP | best homogeneous | heterogeneous | ratio |
|---:|---:|---:|---:|---:|
| 8 | 4 | 2270.1 | 2137.5 | 0.942x |
| 8 | 2 | 2195.8 | 2025.2 | 0.922x |
| 8 | 1 | 2019.4 | 1853.9 | 0.918x |
| 24 | 4 | 3106.4 | 2820.9 | 0.908x |
| 24 | 2 | 2994.0 | 2733.0 | 0.913x |
| 24 | 1 | 2746.0 | 2535.6 | 0.923x |

Longer training does not create a heterogeneous win. For the large batch it
narrows the gap from 9.2% to 7.7%, because the common training block hides more
of the rollout difference. For the small batch, the ratio becomes slightly
worse.

The selected DP=1, batch-24 curve is:

| S | TP1 capped | TP2 capped | heterogeneous | TP1 uncapped | TP2 uncapped |
|---:|---:|---:|---:|---:|---:|
| 0 | 135.5 | 138.4 | 126.5 | 127.5 | 122.2 |
| 1 | 2709.4 | 2635.2 | 2535.6 | 2683.1 | 2746.0 |

This is convergence from a growing shared cost, not a routing gain. At S=1,
PSRL still makes SCG equal throughput, and the fixed heterogeneous fleet does
not have the highest throughput.

The uncapped TP2 advantage also shrinks as training becomes dominant. At batch
24 it is 10.8% under DP=4 but 4.2% under DP=1. That is another sign that longer
training masks rollout differences rather than fixing them.

![DP1 large-batch heterogeneous run](../../outputs/simulate/scg_14b_4gpu_psrl_s1_train_sweep/dp1_cap20k_b24/half_va/plots/instance_load.png)

![DP1 large-batch uncapped TP2](../../outputs/simulate/scg_14b_4gpu_psrl_s1_train_sweep/dp1_cap20k_b24/all2_uncapped/plots/instance_load.png)

Machine-readable results:

```text
outputs/simulate/scg_14b_4gpu_psrl_s1_train_sweep/summary.json
outputs/simulate/scg_14b_4gpu_psrl_s1_train_sweep/summary.md
```

Reproduce:

```bash
WORK_BASE=outputs/simulate/scg_14b_4gpu_psrl_s1_train_sweep \
POLICY=psrl N_GPUS=4 N_VERSIONS=20 MAX_STALENESS=1 ACTIVE_CAP=40 \
INCLUDE_UNCAPPED_BASELINES=true MAX_PARALLEL_CELLS=4 \
CELLS='dp2_cap20k_b8:2:20000:8:long_ctx dp1_cap20k_b8:1:20000:8:long_ctx
dp2_cap20k_b24:2:20000:24:long_ctx dp1_cap20k_b24:1:20000:24:long_ctx' \
bash examples/simulate/scg_14b_sweep_conditions.sh
```

## PSRL S=1 with uncapped homogeneous baselines

This experiment tightens PSRL to `max_staleness=1` and adds two homogeneous
baselines with the router active cap disabled:

- capped 4xTP1
- capped 2xTP2
- capped heterogeneous 2xTP1 + 1xTP2
- uncapped 4xTP1
- uncapped 2xTP2

The sweep uses 20 versions, batch sizes 8 and 24, and 20k/30k generation caps.
All 20 arms obey the bound: every consumed request has true staleness at most
1, and every arm observes a maximum of exactly 1.

### SCG(1)

| condition | TP1 capped | TP2 capped | heterogeneous | TP1 uncapped | TP2 uncapped | heterogeneous / best homogeneous |
|---|---:|---:|---:|---:|---:|---:|
| cap20k_b8 | 2011.7 | 2263.1 | 2137.5 | 2011.7 | 2270.1 | 0.942x |
| cap30k_b8 | 1504.7 | 1742.3 | 1626.0 | 1504.7 | 1723.3 | 0.933x |
| cap30k_b24 | 2268.7 | 2219.4 | 2202.7 | 2270.1 | 2389.8 | 0.922x |
| cap20k_b24 | 2981.0 | 2803.6 | 2820.9 | 2960.3 | 3106.4 | 0.908x |

As with PSRL S=2, every consumed token is inside the target window, so SCG(1)
equals throughput. Heterogeneous TP does not beat the best homogeneous arm.

The selected small-batch condition has this complete curve:

| S | TP1 capped | TP2 capped | heterogeneous | TP1 uncapped | TP2 uncapped |
|---:|---:|---:|---:|---:|---:|
| 0 | 100.6 | 111.5 | 105.1 | 100.6 | 111.8 |
| 1 | 2011.7 | 2263.1 | 2137.5 | 2011.7 | 2270.1 |

### What the uncapped baselines reveal

At batch 8, capped and uncapped homogeneous results are effectively identical.
The active cap is not binding.

At batch 24, the TP degrees react differently:

| generation cap | TP1 capped | TP1 uncapped | TP2 capped | TP2 uncapped |
|---:|---:|---:|---:|---:|
| 20k | 2981.0 | 2960.3 | 2803.6 | 3106.4 |
| 30k | 2268.7 | 2270.1 | 2219.4 | 2389.8 |

Removing the cap changes TP1 by less than 1%, but improves TP2 by 10.8% at
20k and 7.7% at 30k. In the uncapped 20k run, each TP2 instance averages
approximately 66 running requests and 72 active requests, rather than being
limited to 40.

This is direct evidence that one shared request-count cap is not neutral across
TP degrees in the fitted simulator. The first-order roofline argument may make
the arithmetic-intensity knee TP-independent, but request count is only a proxy
for tokens, context, scheduler overhead, and KV pressure. A per-TP fitted cap is
needed before judging the heterogeneous strategy conclusively.

![PSRL S1 large-batch heterogeneous run](../../outputs/simulate/scg_14b_4gpu_psrl_s1/cap20k_b24/half_va/plots/instance_load.png)

![PSRL S1 uncapped TP2 baseline](../../outputs/simulate/scg_14b_4gpu_psrl_s1/cap20k_b24/all2_uncapped/plots/instance_load.png)

Machine-readable results:

```text
outputs/simulate/scg_14b_4gpu_psrl_s1/summary.json
outputs/simulate/scg_14b_4gpu_psrl_s1/summary.md
```

Reproduce:

```bash
WORK_BASE=outputs/simulate/scg_14b_4gpu_psrl_s1 \
POLICY=psrl N_GPUS=4 N_VERSIONS=20 MAX_STALENESS=1 ACTIVE_CAP=40 \
INCLUDE_UNCAPPED_BASELINES=true MAX_PARALLEL_CELLS=4 \
CELLS='cap20k_b8:4:20000:8:long_ctx cap20k_b24:4:20000:24:long_ctx
cap30k_b8:4:30000:8:long_ctx cap30k_b24:4:30000:24:long_ctx' \
bash examples/simulate/scg_14b_sweep_conditions.sh
```

## PSRL bounded-staleness follow-up

The current experiment uses PSRL instead of verl:

- 4-GPU budget: 4xTP1, 2xTP2, or 2xTP1 + 1xTP2
- `max_staleness=2`
- 20 versions
- batch sizes 4, 8, 16, and 24 groups
- 20k and 30k generation caps
- router active cap 40

All 18 arms obey the bound. Every consumed request has true staleness at most
2, and every arm observes a maximum of exactly 2.

PSRL waits for the current training buffer to become ready rather than
literally waiting for every admitted rollout. Its reserve/occupy controller may
move completed groups between buffers, but the consumed batch is bounded by
the configured staleness window.

### SCG(2)

Because all consumed work is within the window, SCG(2) equals throughput in
every arm:

| condition | 4xTP1 | 2xTP2 | heterogeneous | heterogeneous / best homogeneous |
|---|---:|---:|---:|---:|
| cap30k_b8 | 1818.3 | 1929.6 | 1806.9 | 0.936x |
| cap20k_b24 | 2997.5 | 2815.9 | 2777.9 | 0.927x |
| cap20k_b8 | 2341.9 | 2474.2 | 2284.7 | 0.923x |
| cap20k_b4 | 1649.2 | 1906.2 | 1740.4 | 0.913x |
| cap30k_b24 | 2325.4 | 2256.6 | 2000.3 | 0.860x |
| cap20k_b16 | 2859.0 | 2701.0 | 2420.6 | 0.847x |

The heterogeneous fleet does not beat the best homogeneous fleet at S=2. Its
best result is 0.936x.

### Small-batch curve

At batch 8 and a 30k cap, the fleet is mostly below the active cap:

| S | 4xTP1 | 2xTP2 | heterogeneous |
|---:|---:|---:|---:|
| 0 | 65.6 | 69.7 | 65.2 |
| 1 | 189.7 | 194.6 | 182.3 |
| 2 | 1818.3 | 1929.6 | 1806.9 |

The heterogeneous instances average 16.1 active requests and spend 9.1% of
time at cap.

### Large-batch curve

At batch 24 and a 20k cap:

| S | 4xTP1 | 2xTP2 | heterogeneous |
|---:|---:|---:|---:|
| 0 | 139.5 | 142.2 | 341.0 |
| 1 | 365.3 | 2331.4 | 623.3 |
| 2 | 2997.5 | 2815.9 | 2777.9 |

This offered load exercises the cap. The heterogeneous fleet averages 32.7
active requests per instance and spends 68.3% of instance-time at cap. Its TP2
instance is at cap for 85.7% of the run.

The version split changes *when* useful work arrives:

- Heterogeneous TP improves SCG(0) by 2.40x over the best homogeneous arm.
- Homogeneous TP2 completes much more work by S=1.
- At S=2, every trajectory counts, so the metric reduces to total throughput
  and TP1 wins.

PSRL removes the expired-work pathology seen under verl, but it also removes
the opportunity for routing to rescue SCG(2): there is no work outside the
window to exclude. Under this policy the current strategy is useful for S=0
freshness, not for S=2 goodput.

![PSRL small-batch heterogeneous run](../../outputs/simulate/scg_14b_4gpu_psrl/cap30k_b8/half_va/plots/instance_load.png)

![PSRL large-batch heterogeneous run](../../outputs/simulate/scg_14b_4gpu_psrl/cap20k_b24/half_va/plots/instance_load.png)

Machine-readable results:

```text
outputs/simulate/scg_14b_4gpu_psrl/summary.json
outputs/simulate/scg_14b_4gpu_psrl/summary.md
```

Reproduce:

```bash
WORK_BASE=outputs/simulate/scg_14b_4gpu_psrl \
POLICY=psrl N_GPUS=4 N_VERSIONS=20 MAX_STALENESS=2 \
MAX_PARALLEL_CELLS=3 \
CELLS='cap20k_b4:4:20000:4:long_ctx cap20k_b8:4:20000:8:long_ctx
cap20k_b16:4:20000:16:long_ctx cap20k_b24:4:20000:24:long_ctx
cap30k_b8:4:30000:8:long_ctx cap30k_b24:4:30000:24:long_ctx' \
bash examples/simulate/scg_14b_sweep_conditions.sh
```

## Four-GPU high-load follow-up

The follow-up uses the smaller fleet suggested after the first screen:

- homogeneous TP1: 4xTP1
- homogeneous TP2: 2xTP2
- heterogeneous: 2xTP1 + 1xTP2
- 40 active requests per instance
- batch sizes 16, 24, and 32 groups, or 128, 192, and 256 responses per version
- 20 versions on the same heavy-tail trace

This does exercise the cap. At batch 32, every homogeneous instance stays at
40 active requests for effectively 100% of simulated time. The heterogeneous
TP1 instances are at cap for 96.7% and 96.9% of the run, and its TP2 instance
is at cap for 100%.

It does not improve SCG(2):

| condition | best homogeneous SCG(2) | heterogeneous SCG(2) | ratio |
|---|---:|---:|---:|
| cap20k_b32 | 3100.6 | 1455.8 | 0.470x |
| cap20k_b24 | 2823.3 | 1076.6 | 0.381x |
| cap30k_b24 | 2271.9 | 848.5 | 0.373x |
| cap30k_b16 | 1857.6 | 684.5 | 0.369x |
| cap30k_b32 | 2321.4 | 837.9 | 0.361x |
| cap20k_b16 | 2695.9 | 771.9 | 0.286x |

The least-bad high-load condition is batch 32 with a 20k generation cap. Its
complete SCG curve is:

| S | 4xTP1 | 2xTP2 | 2xTP1 + 1xTP2 |
|---:|---:|---:|---:|
| 0 | 195.1 | 463.4 | 507.7 |
| 1 | 1416.6 | 2861.3 | 1238.2 |
| 2 | 3100.6 | 2895.1 | 1455.8 |
| 3 | 3100.6 | 2895.1 | 1779.9 |
| 4 | 3100.6 | 2895.1 | 2625.4 |
| 5 | 3100.6 | 2895.1 | 2825.5 |

The larger batch removes the earlier utilization objection but exposes a
different problem. Total heterogeneous throughput is only 2.4% below TP2 and
8.9% below TP1, while SCG(2) is 49.7% and 53.0% lower respectively. The loss is
therefore mostly staleness composition, not raw throughput.

The homogeneous fleets consume every selected request by staleness 2. In the
heterogeneous arm, 46.4% of consumed requests have staleness greater than 2.
Its router holds 3.9 version layers on average, compared with 2.1 for TP1 and
2.4 for TP2.

This follows directly from the strict route:

1. One TP2 instance is the only server for every old version.
2. Two TP1 instances serve current work while an old queue is blocked.
3. At the next update, unfinished current work becomes old and joins the
   single-TP2 bottleneck.
4. Under verl, work already older than S=2 is not expired. The oldest queues
   continue to receive high-TP service even though their tokens can no longer
   contribute to SCG(2).

Increasing offered load therefore makes the current rule worse at S=2. It
keeps the cap invariant and still improves S=0 by 9.6% over the best
homogeneous arm, but it spends too much high-TP capacity on trajectories that
are already outside the target window.

![Four-GPU heterogeneous high-load run](../../outputs/simulate/scg_14b_4gpu_highload/cap20k_b32/half_va/plots/instance_load.png)

![Four-GPU homogeneous TP2 high-load run](../../outputs/simulate/scg_14b_4gpu_highload/cap20k_b32/all2/plots/instance_load.png)

Machine-readable results:

```text
outputs/simulate/scg_14b_4gpu_highload/summary.json
outputs/simulate/scg_14b_4gpu_highload/summary.md
```

Reproduce the four-GPU screen:

```bash
WORK_BASE=outputs/simulate/scg_14b_4gpu_highload \
N_GPUS=4 N_VERSIONS=20 MAX_PARALLEL_CELLS=3 \
CELLS='cap20k_b16:4:20000:16:long_ctx cap20k_b24:4:20000:24:long_ctx
cap20k_b32:4:20000:32:long_ctx cap30k_b16:4:30000:16:long_ctx
cap30k_b24:4:30000:24:long_ctx cap30k_b32:4:30000:32:long_ctx' \
bash examples/simulate/scg_14b_sweep_conditions.sh
```

## Earlier eight-GPU screen

The selected workload shape is:

- DeepSeek-R1-Distill-Qwen-14B on 8 H20 GPUs
- `policy=verl`
- 20 versions
- batch size 6 groups, group size 8
- 30,000-token generation cap
- heavy-tail response lengths with p99/p50 approximately 4.6
- router active cap 40 requests per instance

At `S=2`, this heterogeneous fleet did not robustly beat the best homogeneous
fleet. Across seeds 42, 43, and 44:

| seed | 8xTP1 SCG(2) | 4xTP2 SCG(2) | 4xTP1+2xTP2 SCG(2) | heterogeneous / best homogeneous |
|---:|---:|---:|---:|---:|
| 42 | 888.0 | 905.6 | 842.1 | 0.930x |
| 43 | 650.8 | 706.6 | 751.2 | 1.063x |
| 44 | 957.1 | 934.0 | 796.9 | 0.833x |

The mean ratio is 0.942x. One seed wins, but the effect is not robust. Comparing
mean rates gives 796.8 tok/s for heterogeneous TP against 848.7 tok/s for
homogeneous TP2, a 6.1% loss.

The policy does provide a strong freshness shift. Its mean SCG is 2.20x the
best homogeneous mean at S=0 and 1.35x at S=1. The crossover occurs at S=2:

| S | 8xTP1 mean | 4xTP2 mean | heterogeneous mean | heterogeneous / best homogeneous |
|---:|---:|---:|---:|---:|
| 0 | 141.3 | 139.4 | 311.0 | 2.200x |
| 1 | 381.0 | 402.5 | 544.5 | 1.353x |
| 2 | 832.0 | 848.7 | 796.8 | 0.939x |
| 3 | 1129.0 | 1226.6 | 1036.7 | 0.845x |
| 4 | 1449.3 | 1627.7 | 1219.2 | 0.749x |
| 5 | 1810.6 | 2103.5 | 1345.8 | 0.640x |
| 6 | 2430.6 | 2900.7 | 1514.2 | 0.522x |
| 7 | 2811.3 | 2996.6 | 1702.8 | 0.568x |
| 8 | 2822.7 | 2996.6 | 1870.5 | 0.624x |
| 9 | 2822.7 | 2996.6 | 2071.6 | 0.691x |
| 10 | 2822.7 | 2996.6 | 2221.8 | 0.741x |
| 11 | 2822.7 | 2996.6 | 2252.9 | 0.752x |
| 12 | 2822.7 | 2996.6 | 2258.4 | 0.754x |

These are token goodput rates in tokens per simulated second. For S values above
an arm's maximum observed staleness, SCG equals that arm's throughput.

## Nine-condition screen

Every arm used 20 versions and the same 8-GPU budget. Selection used
`heterogeneous SCG(2) / max(TP1, TP2) SCG(2)`.

| condition | batch | generation cap | best homogeneous | heterogeneous | ratio |
|---|---:|---:|---:|---:|---:|
| cap30k_b6 | 6 | 30,000 | 905.6 | 842.1 | 0.930x |
| cap40k_b8 | 8 | 40,000 | 804.3 | 741.5 | 0.922x |
| cap20k_b10 | 10 | 20,000 | 1280.7 | 1176.9 | 0.919x |
| cap30k_b10 | 10 | 30,000 | 1094.7 | 990.1 | 0.904x |
| cap20k_b8 | 8 | 20,000 | 1196.4 | 1027.7 | 0.859x |
| cap30k_b8_flat | 8 | 30,000 | 3613.5 | 3067.0 | 0.849x |
| cap30k_b8 | 8 | 30,000 | 1002.8 | 835.9 | 0.834x |
| cap20k_b6 | 6 | 20,000 | 1173.6 | 924.3 | 0.788x |
| cap40k_b10 | 10 | 40,000 | 969.6 | 742.0 | 0.765x |

The near-uniform `psrl_aligned` trace is a negative control. As expected, it
does not benefit from tail concentration.

## Why heterogeneous TP loses at S=2

The selected seed-42 run shows the structural imbalance:

| fleet/tier | mean active requests per instance | mean running | mean waiting | time at cap |
|---|---:|---:|---:|---:|
| heterogeneous TP1 | 9.4 | 9.3 | 0.1 | 0.0% |
| heterogeneous TP2 | 24.3 | 17.6 | 6.7 | 16.0% |
| homogeneous TP2 | 21.6 | 18.2 | 3.4 | 0.0% |

The router is behaving as designed:

1. Old-version requests are restricted to the two TP2 instances.
2. Once an old-version queue blocks, current-version work is restricted to the
   four TP1 instances.
3. TP2 receives the long tail: consumed requests average 13,485 generated
   tokens on TP2 versus 5,815 on TP1.

However, strict tier separation leaves the four TP1 instances lightly loaded
while the two TP2 instances carry the long-running old-version cohort. The
homogeneous TP2 baseline has four TP2 instances and can spread both short and
long requests across all four. In the selected run:

- Heterogeneous span: 3719.5 s
- Homogeneous TP2 span: 2963.5 s
- Heterogeneous throughput: 2292.8 tok/s
- Homogeneous TP2 throughput: 2933.2 tok/s

The heterogeneous fleet loses 21.8% total throughput. Better freshness recovers
most of that loss by S=2, but not all of it.

This is not only a router-queue-capacity problem. Across the grid, persistent
router queue depth ranged from 0.08 to 25.31 requests, yet every condition lost
at S=2. The more general issue is that the strict version partition cannot lend
idle TP1 capacity to old work.

![Selected heterogeneous instance load](../../outputs/simulate/scg_14b_sweep_v2/cap30k_b6/half_va/plots/instance_load.png)

![Selected homogeneous TP2 instance load](../../outputs/simulate/scg_14b_sweep_v2/cap30k_b6/all2/plots/instance_load.png)

## Interpreting `max_staleness=2` under verl

`VerlPolicy` does not use `job.max_staleness` as a hard version cutoff. Its
admission control uses a sample-overage ratio and a bounded hand-off queue.
`max_staleness=2` selects the SCG target and plot marker, but trajectories may
exceed it. This sweep observed true staleness as high as 15.

## Reproduction

Run the full screen:

```bash
N_VERSIONS=20 MAX_PARALLEL_CELLS=3 \
  bash examples/simulate/scg_14b_sweep_conditions.sh
```

Machine-readable and Markdown summaries are written to:

```text
outputs/simulate/scg_14b_sweep_v2/summary.json
outputs/simulate/scg_14b_sweep_v2/summary.md
```

The seed repeats are under:

```text
outputs/simulate/scg_14b_best_repeats/
```
