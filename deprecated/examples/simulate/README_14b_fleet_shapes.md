# Fleet shape comparison: heterogeneous TP against homogeneous baselines

The current 20-version `version_capped_tp` sweep and repeat analysis are in
[README_14b_version_capped_sweep.md](README_14b_version_capped_sweep.md).
The numbers below are retained as the legacy `version_aware_tp` baseline.

Four `instance_load` figures of the same workload on the same 8 H20 GPUs, differing
only in how those GPUs are split into inference instances and how the router places
work. The subplot titles carry each instance's tensor-parallel degree, coloured and
bolded only when the fleet is actually heterogeneous.

| figure | fleet | router | span | tok/s | SCG(0) | SCG(2) | vs TP1 @S=0 | vs TP1 @S=2 |
|---|---|---|---|---|---|---|---|---|
| `instance_load_homogeneous_tp1.png` | 8xTP1 | balance | 1406s | 6486 | 0.1024 | 0.4667 | 1.000x | 1.000x |
| `instance_load_homogeneous_tp2.png` | 4xTP2 | balance | 1357s | 6821 | 0.1061 | 0.4776 | **1.036x** | **1.023x** |
| `instance_load_heterogeneous_tp_blind_routing.png` | 4xTP1+2xTP2 | balance | 1530s | 5936 | 0.0941 | 0.3869 | 0.919x | 0.829x |
| `instance_load_heterogeneous_tp.png` | 4xTP1+2xTP2 | legacy `version_aware_tp` | 1519s | 6061 | 0.1317 | 0.4372 | **1.285x** | 0.937x |

These numbers predate `version_capped_tp`. The regeneration script now uses a
shared H20 active cap of 40, so rerun it before treating this table as current.

Run: Qwen3.5-35B-A3B, `policy=verl`, fitted `psrl` cost model from measured vLLM
profiles at tp=1 and tp=2, long-context trace (p50 = 6504 output tokens,
p99/p50 = 4.6), 8 versions x batch 16 x group 8. Every arm consumes exactly 1024
trajectories, so SCG differences are span plus staleness-distribution differences.

## What to look for

**`homogeneous_tp1`** — eight single-GPU rows, every one peaking near 45 running
requests. The reference shape: no instance holds more GPUs than another, so the
occupancy curves are directly comparable and nothing is hidden.

**`homogeneous_tp2`** — four two-GPU rows peaking near 97. Twice the GPUs per
instance carrying roughly twice the batch, which is the iso-GPU comparison: the
fleet-level result is only 1.02-1.04x over 8xTP1, not the ~1.5x a per-instance
latency comparison suggests. Shortest span of the four.

**`heterogeneous_tp_blind_routing`** — the failure mode, and the reason the naive
heterogeneous comparison misleads. `request_num_balance` counts requests and ignores
TP, so instances 4-5 (TP2, red) peak at the same ~64 as instances 0-3 (TP1, blue)
while holding twice the hardware. Half the GPU capacity on those rows is idle, and
the arm is the *worst* of the four (0.829x at S=2). `srl` emits a warning for
exactly this configuration.

**`heterogeneous_tp`** — the same fleet with legacy `version_aware_tp`, which sends the
old-version tail to the highest TP available and balances current-version work by
count. The TP2 rows now spike to ~146, a 2.27x peak-load ratio over the TP1 rows.

## The honest reading

Against the **fully homogeneous 8xTP1 baseline**, heterogeneity wins only at S=0
(1.285x) and loses at the S=2 target (0.937x). Homogeneous 4xTP2 beats 8xTP1 at
almost every S while needing no routing cleverness at all.

The large legacy routing gains reported against `half_balance` (1.13-1.40x) are
mostly repair of a self-inflicted wound: `half_balance` wastes half the GPUs on its
TP2 instances. Measured against a fleet that was never broken, the gain mostly
disappears.

So: heterogeneous TP is defensible here only if S=0 is what matters -- near-on-policy
training where only the freshest cohort counts. For S>=1, pick one degree and run it
homogeneously.

⚠️ Single run per arm, no seed replication. The S=0 win (1.285x) and the
blind-routing loss (0.829x) are large enough to trust; ratios in 0.99-1.06 are not.

Regenerate with `bash examples/simulate/scg_14b_fleet_shapes.sh`.
