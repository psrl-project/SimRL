# Profiling and cost model design

## Boundary

DryRun-RL treats versioned profile JSONL as the boundary between execution and modeling:

```text
official benchmark -> untouched raw artifacts -> DryRun adapter -> profile JSONL
profile JSONL -> model-owned fit -> cost artifact -> simulation or search
```

vLLM and Megatron-Bridge source trees are pinned but not vendored. `third_party/pins.json` records the
PyTorch CUDA wheel and exact source commits. `third_party/install.sh` reads that file, installs PyTorch first,
then clones the trees and installs both into the current conda environment. Megatron-Bridge is synchronized from
its checked-in `uv.lock`, including its editable Megatron-LM source. Regression never imports a benchmark process.

Every JSONL record includes:

- Schema version and tagged record type.
- Independent run ID, repetition, status, OOM state, and raw artifact path.
- Model, GPU, topology, and software fingerprints.
- Explicit TP, PP, DP, CP, and world size.
- A rollout or training payload.

Writers reject records unless `world_size = TP * PP * DP * CP`. Training writers also require
`GBS = MBS * DP * num_microbatches`.

## Official profiler adapters

### vLLM

The adapter generates parameter files and calls `vllm bench sweep serve` once for every
workload point. The timing sweep enables the official scheduler iteration log without a
profiler and retains `--save-detailed` client diagnostics. Collection requires
`random-range-ratio=0` so every request in a point has the same prompt length `L`.

Each captured `Iteration(...)` line supplies request/token counts and low-overhead elapsed
time. Attention features are then filled from equal-length formulas rather than profiler
annotations:

- Prefill (and prefill chunks under `max-num-batched-tokens`): `R` context requests and
  `T` context tokens with `T % R == 0` give `n_tokens = T` and `sum_l2 = T^2 / R`.
- Decode: the `k`-th pure-decode step (`k` 0-based after prefill) with `R` generation
  requests gives `n_tokens = R` and `ctxsum = R * (L + 1 + k)`.
- A mixed step emits both terms. Configured phase collection still drops the opposite
  pure phase.

Client TTFT and ITL include queueing, networking, event-loop delay, and possible chunk
coalescing. They remain in the untouched benchmark JSON but are never clustered to infer
scheduler batches. Missing scheduler logs fail collection instead of silently falling back
to client timing.

### Megatron-Bridge

DryRun-RL generates candidates, but execution stays in the pinned official Qwen3 recipe. Candidate constraints
cover GPU divisibility, attention and GQA tensor parallel divisibility, uniform pipeline layer partition, and
global batch divisibility.

Local runs use `scripts/performance/run_recipe.py`. Slurm runs use `setup_experiment.py`. Both use mock data,
training timers, resolved `ConfigContainer`, and PyTorch memory history. Collection prefers a structured
`dryrun_metrics.json` sidecar and otherwise reads TensorBoard timer scalars and generated memory snapshots. OOM
status is preserved as a record.

Each candidate runs in an independent process. Resume skips only successful candidates.

## Rollout models

All rollout models preserve `decode_coeffs()` so the simulator can integrate a pure decode segment without
iterating over every generated token.

### Unified roofline

```text
latency = max(F, W + G*n_tokens + A_p*sum_l2/b + A_d*ctxsum)
```

The fit is a non-negative soft-L1 regression with MAD outlier refitting. Prefill contributes `sum_l2`. Decode
contributes `ctxsum`. A mixed step contributes both.

### PSRL fitted

Prefill and decode use separate attention and non-attention coefficients:

```text
prefill = p_attn_b + p_attn_k*sum_l2
        + max(p_other_floor, p_other_b + p_other_k*n_tokens)

decode  = d_attn_b + d_attn_k*ctxsum
        + max(d_other_floor, d_other_b + d_other_k*n_requests)
```

A mixed step sums both attention terms and takes the maximum of the two non-attention terms. This avoids charging
the shared scheduler floor twice.

### DistServe

The DistServe model maps a fitted effective roofline into architecture constants using hidden size `h`,
intermediate size `m`, and `gemm = 4*h*h + 2*h*m`.

A single model identifies only `C3 + C4*gemm`, not `C3` and `C4` independently. The artifact therefore stores the
minimum-norm canonical solution and an identifiability diagnostic. Cross-model joint calibration is deferred.

## Hydraulis V1 training model

The implementation follows the workload decomposition in Hydraulis Appendix C. It supports CP=1, 1F1B, and
uniform layer partition only.

For local microbatch `j`:

```text
Q_j = sum(sequence_length^2)
T_j = sum(sequence_length)

Q_eff = sum(Q_j) + (PP - 1)*max(Q_j)
T_eff = sum(T_j) + (PP - 1)*max(T_j)

latency = a_q*Q_eff
        + b_t*T_eff
        + c_pipeline*(num_microbatches + PP - 1)
        + update_overhead
```

The coefficients are fitted separately for each `(TP, PP, schedule, activation_recompute)` cell. DP communication
is zero in V1. DP still changes local workload. Prediction balances actual trajectory lengths across DP ranks and
uses the critical rank.

Training memory combines analytical static state with a fitted activation residual:

```text
peak_reserved = parameter_inventory(TP, PP, DP, dtypes, optimizer_sharding)
              + activation_slope*activation_feature
              + residual_constant
```

The inventory accounts for model parameters, gradients, FP32 master parameters when needed, and two Adam moments.
Model parameters are partitioned by TP and PP. Distributed optimizer gradients, master parameters, and moments
are additionally sharded by DP.

The activation feature is the largest local microbatch token count multiplied by the number of in-flight
microbatches, capped at PP. OOM records impose `predicted_peak > device_capacity`. They are not discarded.

Activation checkpoint recompute is a discrete training selector. It is unrelated to DryRun-RL's separate RL
log-prob recompute stage.

## Artifact and domain rules

An artifact stores:

- Artifact version, model type, model and hardware identity, and profile digest.
- Explicit parallelism and selector mappings for each entry.
- Coefficients, feature ranges, sample and outlier counts.
- Training and grouped-run holdout R squared, RMSE, MRE, and maximum error.
- Design matrix rank and model-specific diagnostics.

Training lookup requires an exact profiled TP, PP, CP, schedule, recompute, optimizer, and dtype cell. DP is the
only analytical discrete extension in V1. Numerical features must remain inside fitted ranges unless the caller
explicitly opts into extrapolation. Missing cells and out-of-range features fail fast.

## Minimum useful sweep

The default training YAML uses a cheaper cell than the Hydraulis identifiability recipe:

- One fixed world size. DP is `world_size / (TP * PP)`.
- Micro-batch size 1.
- Explicit local `num_microbatches` covering underfilled PP, full PP, and simulated GBS.
- At least four sequence lengths.
- Full activation recomputation.
- One warmup step and three recorded step times.
- One process per topology. Do not repeat the same config.

A fuller Hydraulis design still benefits from a second micro-batch size, equal-total-token shapes with different
`sum_l2`, and `{PP, 2*PP, 4*PP}` local microbatches. Regression rejects a rank-deficient latency or memory design
by default.

## Deferred stages and acceptance criteria

1. Topology-aware communication and overlap. Calibrate TP, PP, and DP collectives per node topology, model exposed
   overlap explicitly, and pass grouped-topology holdout thresholds before enabling search.
2. Advanced pipeline schedules. Add VPP, interleaved 1F1B, zero-bubble schedules, and measured stage imbalance.
   Validate predicted bubbles against per-stage timers.
3. CP, SP, EP, and MoE. Represent dense and expert meshes separately rather than multiplying dimensions blindly.
   Candidate generation must prove divisibility and rank-mesh constraints before launch.
4. Recompute granularity. Profile full, selective, uniform, and block-level policies. Memory and latency must share
   the same resolved recompute descriptor.
5. Variable-length packing and heterogeneous pipelines. Record actual packed token layouts and stage assignments.
   Validate equal-token workloads with different quadratic attention work.
6. Allocator fragmentation, CPU offload, and activation offload. Predict reserved as well as allocated memory and
   bound OOM false negatives on held-out candidates.
7. Cross-model DistServe calibration. Fit C3 and C4 jointly across architectures and report condition number and
   confidence intervals.
8. Automatic disaggregated GPU allocation. Enable only after candidate-domain checks and held-out error gates are
   part of the search objective. Search must reject unsupported cells instead of silently extrapolating.

## References

- Hydraulis, Appendix C: <https://arxiv.org/abs/2412.07894>
- DistServe: <https://arxiv.org/abs/2401.09670>
- vLLM benchmark sweeps: <https://docs.vllm.ai/en/v0.26.0/benchmarking/sweeps/>
- Megatron-Bridge profiling: <https://docs.nvidia.com/nemo/megatron-bridge/0.4.2/training/profiling.html>
