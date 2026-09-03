# FSDP2/HSDP Training Backend: Profile → Cost Model → Simulate

**Date:** 2026-08-26
**Status:** approved design, not yet implemented

## Motivation

The training side of DryRun-RL only models Megatron. Megatron is powerful but has a
large hyperparameter surface and many parallelism modes, which makes both profiling
sweeps and cost-model cells expensive. FSDP2 (and its hybrid-shard form, HSDP) is the
other backend real RL systems use — psrl drives it through verl with
`actor.strategy=fsdp2` — and it has a far smaller knob count: no pipeline schedule, no
tensor parallelism in the common configuration, and a device mesh described by two
integers.

This adds FSDP as a third backend alongside vLLM (rollout) and Megatron (training),
end to end: profile → cost model → simulate.

## Reference configuration

`psrl/examples/dapo_trainer/qwen2.5_3b_fsdp.sh` is the ground truth for which knobs
matter in practice:

| key | value | meaning |
|---|---|---|
| `actor.strategy` | `fsdp2` | FSDP2 `fully_shard`, not FSDP1 |
| `fsdp_config.param_offload` | `False` | parameters stay resident |
| `fsdp_config.optimizer_offload` | `True` | optimizer state offloaded to CPU |
| `model.enable_gradient_checkpointing` | `True` | activation recompute on |
| `actor.use_dynamic_bsz` | `True` | microbatches formed by token budget |
| `actor.ppo_max_token_len_per_gpu` | `10240` | that budget |
| `ulysses_sequence_parallel_size` | unset (`1`) | no sequence parallelism |

## Environment findings

All measured on this host (TencentOS 3.2, glibc 2.28, 8x H20 sm90,
torch 2.11.0+cu128, transformers 5.5.3, py3.12), not assumed.

1. **verl is not installed and will not be depended on.** `import verl` fails, as does
   `tensordict`. psrl only vendors it under `third_party/verl`. Depending on it would
   drag Ray and the full verl tree into a pure-simulator repository and pin us to a
   verl version. The launcher is built directly on `torch.distributed` + transformers.
2. **FSDP2 primitives work.** HSDP mesh via
   `init_device_mesh("cuda", (repl, shard), ("dp_replicate", "dp_shard"))`, per-layer
   `fully_shard`, `MixedPrecisionPolicy`, and `AdamW` compose correctly. Parameters and
   optimizer state are both `DTensor` with placements
   `(Replicate(), Shard(dim=0))`.
3. **flash-attn had to be built from source** (prebuilt wheels require `GLIBC_2.32`;
   this host has `2.28`). Now installed as step 7 of `third_party/install.sh`, pinned at
   `2.8.3.post1` in `pins.json`. `MAX_JOBS` must stay small: the host reports 384 cores
   but only ~2 GiB free RAM, and an uncapped build is OOM-killed with an empty log.
4. **HF handles packed varlen natively.** With `attn_implementation="flash_attention_2"`,
   passing restarting `position_ids` for concatenated documents runs forward and backward
   correctly. No custom attention wrapper is needed for any batching mode.
5. **`torch.nn.attention.varlen.varlen_attn` exists but had no fused kernel** before
   flash-attn was installed, and ran 2.4-3.4x slower than SDPA. Not used.

### Attention kernel dispatch (affects how packing is judged)

At 4 seqs x 2048 tokens, Qwen3-8B attention shape, bf16 causal, forward+backward:

| kernel | time |
|---|---|
| `F.scaled_dot_product_attention` (auto) -> cuDNN | 4.29 ms |
| SDPA forced to flash backend | 6.71 ms |
| `flash_attn_func` dense | 6.50 ms |
| `flash_attn_varlen_func` packed | 6.52 ms |

SDPA auto-dispatches to cuDNN on H20, so comparing SDPA-default against FA2 varlen
makes packing look 1.5x slower than it is. Within one kernel family varlen equals
dense: **packing itself is free**. Its benefit comes entirely from removing pad tokens.

### Packing benefit on realistic lengths

Bimodal DAPO-like distributions (a truncated mode at 10240, the rest sampled in
800-4000), pad-to-max versus fully packed, forward+backward:

| n_seqs | fill ratio | pad-to-max | packed | speedup |
|---|---|---|---|---|
| 8 | 0.50 | 176.5 ms | 112.8 ms | 1.56x |
| 8 | 0.71 | 26.0 ms | 22.3 ms | 1.17x |
| 16 | 0.53 | 349.5 ms | 229.6 ms | 1.52x |
| 16 | 0.32 | 349.6 ms | 80.5 ms | 4.34x |

A greater-than-4x spread means the batching mode is a first-class input to the cost
model, not a detail.

### FSDP2 static memory is not Megatron's

Measured with a 1.611B-parameter model, world size 4, `MixedPrecisionPolicy`
(bf16 compute / fp32 reduce), AdamW:

| term | Megatron (measured previously) | FSDP2 (measured here) |
|---|---|---|
| parameters | `2p`, **unsharded** | `4p/ws`, sharded |
| gradients | `4p`, **unsharded** (reduce-scatter source) | `4p/ws`, sharded |
| master + Adam moments | `12p/DP`, sharded | `12p/ws`, sharded |
| total | ~`59 GiB` for 8.19B at DP8 | `16p/ws` exactly (6.078 vs 6.000 GiB) |

`dryrun/cost/training/inventory.py` documents that Megatron's fp32 gradient buffer is
*not* DP-sharded, which is why a naive analytic model understates it roughly twofold.
That is **Megatron-specific**: FSDP2 reduce-scatters per layer into already-sharded
gradients, so no full-size buffer exists. Reusing `parameter_inventory` unchanged would
**overstate** FSDP memory by about 2x. FSDP needs its own inventory function.

## The profile/simulate seam

The load-bearing decision: **profiling only ever runs uniform sequence shapes;
packing lives entirely on the simulate side.**

```
profile (uniform shapes)          cost model                 simulate (real lengths)
+----------------------+   +--------------------+   +--------------------------+
| adapters/fsdp/       |   | cost/training/     |   | MicrobatchPlanner        |
|  config  prepare     |-->|   fsdp.py          |<--|  pad_to_max              |
|  launch  execute     |   |  FsdpLatencyModel  |   |  token_budget            |
|  measure collect     |   |  FsdpMemoryModel   |   |  packed_no_pad           |
+----------------------+   +--------------------+   +--------------------------+
   ProfileRecord JSONL       CostArtifact JSON        TrainingCostEstimate
```

Profiling measures `step_time` and `peak_memory` over a clean grid of
`(sequence_length, n_seqs_per_rank, n_microbatches)`. The planner's only job at simulate
time is turning real rollout lengths into a *microbatch plan* — a list of
`SequenceStats` per microbatch, exactly what `latency_features` already consumes — which
the fitted model then prices. `SequenceStats` carries `sum_tokens` and
`sum_squared_tokens`, so the plan stays agnostic about which features a model uses.

Three consequences:

1. **No measured record ever violates the existing schema invariants.** Because profile
   shapes are uniform, `TrainingPayload`'s "every microbatch holds `micro_batch_size`
   sequences" and `global_batch_size == mbs * dp * nmb` both hold. What must change is
   the schedule check, which rejects anything but `"1f1b"` in **four** places — FSDP has
   no pipeline schedule, so its value becomes `"fsdp"`:

   | site | role |
   |---|---|
   | `dryrun/profiler/schema.py:374` | `TrainingPayload`, the collected record |
   | `dryrun/cost/training/base.py:66` | `TrainingWorkload`, the simulate-side query |
   | `dryrun/profiler/search_space.py:44` | `TrainingCandidate` (Megatron's own; FSDP gets `FsdpCandidate`) |
   | `dryrun/cost/training/hydraulis.py:78` | Hydraulis's record filter, which must keep excluding non-1F1B so FSDP records are never fitted by the Megatron model |

   The first two accept `"fsdp"`. The third is unchanged, because FSDP uses a separate
   candidate type. The fourth stays restrictive by design: it is what keeps the two
   backends' records from being silently mixed into one fit. The equal-microbatch and
   `global_batch_size` invariants stay enforced in all cases.
2. **`Parallelism` is untouched.** Under HSDP every rank still receives distinct data,
   so `dp == world_size` and `tp = pp = cp = 1`. Shard and replicate degrees are
   *discrete configuration*, which is what the artifact `selector` mechanism already
   expresses.
3. **The planner has a single consumer.** Since profiling no longer packs, there is no
   risk of the launcher's packing and the simulator's packing diverging.

## Components

### 1. Shared training-adapter layer (new)

`dryrun/profiler/adapters/training/` — genuinely backend-agnostic code, extracted from
the Megatron adapter and consumed by both. Sibling to `megatron/`, `vllm/`, `fsdp/`.

**`run.py`** (~85 lines)
- `OOM_EXIT_CODE`, `oom_marker_paths(run_dir)`
- `run_candidate(config, candidate, prepare, *, dry_run, resume) -> Path` — subprocess
  orchestration, stale-marker cleanup, `run.log`, status classification,
  `status.json`. The single backend-specific step, building the command, is injected as
  the `prepare` callable.
- `read_status(run_dir) -> (status, duration_s, error)` — carries the invariant that
  OOM is decided by marker files and **never** by scanning the log for a phrase. A
  false OOM adds an inequality constraint that can only raise the fitted memory floor,
  so it must be wrong in exactly one place, not two.

**`sidecar.py`** (~50 lines)
- `load_sidecar_metrics(run_dir)`
- `slice_warmup(step_times, warmup_steps)`
- `rank_memory_peaks(rank_memory) -> (peak_allocated, peak_reserved)`
- `iter_candidate_dirs(raw_root, candidates, allow_incomplete)` — the manifest-presence
  loop shared by both `collect_*_records`.

Deliberately **not** extracted, because the mechanism differs even though the purpose
matches:
- resolved-config validation — Megatron reads `ConfigContainer.yaml` through OmegaConf;
  FSDP validates the observed device mesh. One signature would be a parameter bag
  neither backend fills.
- `config.py` — `bridge_dir`, `model_family_name`, `model_recipe_name`, and the pinned
  bridge version are Megatron-only.
- `measure.py` — Megatron subclasses `megatron.bridge.training.callbacks.Callback`;
  FSDP owns its own loop. Only the sidecar payload *shape* is shared, and that is a
  schema concern.

### 2. FSDP profiler adapter (new)

`dryrun/profiler/adapters/fsdp/`, mirroring Megatron's stage boundaries.

**`config.py`** — `FsdpProfileConfig`, frozen, `from_dict` for YAML. Fields: `model`
(via `model_descriptor_from_yaml`), `hardware`, `output_dir`, `compute_dtype`,
`optimizer`, `parameter_dtype`, `gradient_dtype`, `optimizer_state_dtype`,
`max_steps`, `warmup_steps`, `num_layers_override`, `environment`,
`allow_incomplete`, `backend_version` (torch version), plus FSDP knobs:
`reshard_after_forward`, `param_offload`, `optimizer_offload`,
`gradient_checkpointing`, `attn_implementation`.

**`search_space.py`** — `FsdpCandidate` and `enumerate_fsdp_candidates`. A candidate is
`(shard_size, replicate_size, n_seqs_per_rank, sequence_length, gradient_checkpointing,
optimizer_offload, repeat)`, with `world_size == shard_size * replicate_size`. Its
`candidate_id` is a SHA-256 prefix, as `TrainingCandidate` does. `sp` is recorded and
pinned to `1`; a value greater than 1 fails fast with an explicit not-implemented
message.

**`prepare.py`** — writes `launch_spec.json` (schema-versioned) and `manifest.json`,
builds the `torchrun ... -m dryrun.profiler.adapters.fsdp.launch --spec ...` command.

**`launch.py`** — the torchrun child; imports torch only at call time, never from the
package namespace. Responsibilities:
1. build the HSDP mesh from `(replicate_size, shard_size)`;
2. construct the model from the HF config (`num_layers_override` honoured) with
   `attn_implementation="flash_attention_2"`;
3. enable gradient checkpointing when requested;
4. apply `fully_shard` per transformer block, then to the root, with
   `MixedPrecisionPolicy` and optional `CPUOffloadPolicy`;
5. run `max_steps` iterations on synthetic uniform batches;
6. write the sidecar, or per-rank `oom_rank{N}.json` on a caught
   `torch.OutOfMemoryError`.

**`execute.py`** — thin: delegates to `training.run.run_candidate` with its `prepare`.

**`measure.py`** — plain hooks (no framework base class). Records per-step wall time
around a synchronized bracket, per-rank `allocated`/`reserved` peaks gathered with
`all_gather_object`, `static_allocated_bytes_at_train_start`, measured
`parameter_count`, and the **comm/compute decomposition** (see below). No barrier on
the timing path: a barrier would charge every rank for the slowest one.

**`collect.py`** — raw directories into `ProfileRecord` with `TrainingPayload`
(`schedule="fsdp"`). Validates the observed mesh and parameter count against the
candidate, so a silently-wrong mesh fails loudly instead of yielding plausible numbers.

### 3. Cost model (new)

**`dryrun/cost/training/fsdp_inventory.py`** — `fsdp_inventory(...)`, the FSDP static
layout (`16p/ws`, everything sharded), separate from `parameter_inventory` because the
sharding rule genuinely differs. It accounts for `optimizer_offload` moving the fp32
master and Adam moments off device.

**`dryrun/cost/training/microbatch.py`** — `MicrobatchPlanner` and `MicrobatchPlan`.
Three modes:
- `pad_to_max` — every microbatch padded to its longest sequence; costed tokens are
  `n_seqs * max_L`.
- `token_budget` — Karmarkar-Karp balancing on verl's `24576 * L + L^2` workload
  proxy, capped by `max_tokens_per_rank`; still padded within a microbatch.
- `packed_no_pad` — concatenated, zero pad tokens; costed tokens are the true `sum_L`.

`packed_no_pad` requires flash-attn, which is now present; the planner asserts this at
construction rather than silently mispricing.

**`dryrun/cost/training/fsdp.py`**
- `FsdpLatencyModel.predict(workload, parallelism) -> float`, reusing the physically
  structured form in `dryrun/cost/training/latency.py` (added by the Megatron work in
  `8523dd9`):

  ```
  step = per_step_overhead + slots * (token_cost * tokens + microbatch_overhead)
  ```

  For FSDP `pipeline_slots(nmb, pp=1)` collapses to `slots = nmb`, which is exactly the
  degenerate case that module already documents. `latency_features` and
  `robust_step_time` are reused unchanged.
- `FsdpMemoryModel.predict(...) -> int`, `fsdp_inventory` plus a fitted activation
  residual, targeting `peak_allocated_bytes`.
- `FsdpTrainingCost.fit(records, ...) -> CostArtifact` and `from_artifact(...)`.
- Registered as `"fsdp"` in `TRAINING_MODEL_REGISTRY`.

Cell key: `(shard_size, replicate_size, sp=1, recompute, optimizer_offload,
reshard_after_forward, parameter_dtype, gradient_dtype, optimizer_state_dtype,
parameter_count)`.

**Memory target.** Both `peak_allocated_bytes` and `peak_reserved_bytes` are recorded,
both are fitted, and both sets of `r^2`/MRE go into artifact diagnostics. The fit
selects automatically: **the target with the higher validation `r^2` wins**, and the
winner is named in `diagnostics["memory_metric"]` alongside both candidates' scores, so
the choice is auditable rather than implicit. The prior Megatron result — fitting
`reserved` gave `r^2 = 0.196` while `allocated` gave `r^2 = 1.0` — makes `allocated` the
expected winner, but it is recorded as a measurement, not assumed.

Because `allocated` underestimates the true out-of-memory threshold (it excludes
allocator fragmentation), the simulator's OOM check must not compare a fitted
`allocated` prediction directly against device capacity. The artifact therefore also
stores the measured `reserved / allocated` ratio per cell as
`diagnostics["reserved_over_allocated"]`, and `FsdpMemoryModel` applies it when the
selected metric is `allocated`. This keeps the regression on the well-behaved target
while keeping the OOM comparison against the quantity that actually causes OOM.

**Latency identifiability.** `microbatch_overhead` (paid per slot) and
`per_step_overhead` (paid once) are separately identifiable only if `n_microbatches`
varies at fixed sequence length, while `token_cost` needs sequence length to vary at
fixed microbatch count. With micro-batch size one and a uniform shape the design row is
`[1, slots * tokens, slots]`, so sweeping sequence length alone caps the design at rank
two. FSDP requires **rank 3** and the sweep varies both axes, mirroring the rank check
already in `hydraulis.py:245`.

**Do not add a quadratic token term.** The Megatron work measured and rejected it: it
improves in-sample error (6.60% to 4.65%) but not leave-one-out (10.24% to 10.38%), and
the implied coefficient corresponds to roughly 18000% of the H20's bf16 throughput.
Directly measured attention accounts for only 2-3% of a step under full recompute, so a
quadratic term absorbs residual instead of representing attention. FSDP inherits both
the conclusion and the prohibition.

**Measurement before model.** The same work found that one warmup step was insufficient:
the first profiled step was the slowest in 18 of 24 runs, and dropping two steps cut the
median within-candidate spread from 16.4% to 0.5%. FSDP's profile config therefore starts
at `max_steps: 12, warmup_steps: 4`, uses `robust_step_time` (median, not mean, because
step times are right-skewed), and records `step_time_dispersion` per candidate so a noisy
measurement is visible in the artifact rather than silently widening the fit.

**Comm/compute decomposition.** `measure.py` additionally records, for one instrumented
step, the time in all-gather and reduce-scatter versus compute, stored in
`timers_s`. It is not used by the regression; it is ground truth for a later
kernel-summation model (the `ComposedRollout` analogue), and it is cheap to capture now
and expensive to re-profile later. `latency.py` already names feeding training GEMMs
through the rollout kernel tables as the principled next step, and this decomposition is
what would validate it.

### 4. CLI, config, examples

- `dryrun-profile fsdp {generate,run,collect}`, matching the Megatron subcommands.
- `dryrun-regress training --latency-model fsdp --memory-model fsdp`; both flags
  currently hard-code `choices=["hydraulis"]` (`dryrun/cli/regress.py:93-94`) and gain
  `"fsdp"`.
- `dryrun/config/defaults/training_cost/fsdp.yaml`, plus `batching_mode` and
  `max_tokens_per_rank` on `TrainingCostConfig`, and `build_training_cost` wiring.
- `examples/profile/fsdp/qwen3_8b.yaml`, `qwen3_8b_smoke.yaml`
  (`num_layers_override: 2`), and `profile_qwen3_8b.sh`.

## Sweep space

Qwen3-8B on 8x H20. `world_size = 8` throughout.

| dimension | values |
|---|---|
| `(replicate, shard)` | `(1,8)` pure FSDP, `(2,4)`, `(4,2)` HSDP |
| `sequence_length` | 512, 1024, 2048, 4096, 8192 |
| `n_seqs_per_rank` | 1, 2, 4, 8 |
| `gradient_checkpointing` | `True` (psrl's setting) |
| `optimizer_offload` | `True`, `False` |
| `sp` | 1 only |
| `max_steps` / `warmup_steps` | 12 / 4 |

`n_seqs_per_rank` and `sequence_length` vary independently so the design row
`[1, slots * tokens, slots]` reaches rank 3; sweeping sequence length alone would cap it
at rank 2.

## Error handling

- Out of memory: per-rank marker files, caught by exact `torch.OutOfMemoryError` type.
  A driver or NCCL fault stays classified `failed`. OOM records are retained — they
  bound the memory model from below.
- Mesh mismatch: `collect` compares the observed mesh and parameter count against the
  candidate and raises rather than emitting a record attributed to the wrong topology.
- Out-of-domain queries: `TrainingOutOfDomainError` unless `allow_extrapolation`.
- `sp > 1`, FSDP1, and tensor parallelism: fail fast with an explicit message.

## Testing

- `tests/test_fsdp_search_space.py` — enumeration, mesh validity, `candidate_id`
  stability, `sp > 1` rejection.
- `tests/test_fsdp_launch.py` — `build_config`-equivalent spec application without a
  process group; JSON round-trip; OOM sidecar shape. Fixtures under
  `tests/fixtures/fsdp/` mirroring `tests/fixtures/megatron_bridge/`.
- `tests/test_fsdp_collect.py` — sidecar to `TrainingPayload`, warmup slicing, peak
  aggregation, mesh-mismatch rejection, OOM classification.
- `tests/test_microbatch_planner.py` — the three modes; `pad_to_max` costed tokens
  equal `n_seqs * max_L`; `token_budget` respects the cap and reproduces KK balancing
  deterministically; `packed_no_pad` costed tokens equal true `sum_L`.
- `tests/test_fsdp_cost.py` — artifact round-trip, known-coefficient latency,
  rank-deficiency rejection, OOM-as-inequality, `fsdp_inventory` against the measured
  `16p/ws` layout, simulator integration.
- `tests/test_training_shared.py` — the extracted `run.py`/`sidecar.py` behave
  identically for both backends; Megatron's existing tests must still pass unchanged.
- A two-layer smoke profile on 8x H20 is the end-to-end gate before any full sweep.

## Sequencing

1. Extract `adapters/training/`; migrate Megatron onto it; its tests stay green.
2. Accept `schedule="fsdp"` in `TrainingPayload` and `TrainingWorkload`; add
   `FsdpCandidate`. Confirm Hydraulis's filter still excludes FSDP records.
3. `adapters/fsdp/` prepare/launch/measure/execute/collect; smoke config; launch tests.
4. Smoke run on 8 GPUs; confirm sidecar and mesh validation.
5. `microbatch.py` and `fsdp_inventory.py` with unit tests.
6. `cost/training/fsdp.py` fit and evaluate; artifact round-trip.
7. CLI, YAML defaults, example scripts.
8. Full sweep; report both memory targets' `r^2`/MRE; choose and record the metric.

## Out of scope

- Ulysses sequence parallelism (`sp > 1`). Requires an all-to-all attention wrapper
  plus numerical validation against `sp = 1`; the dimension is reserved in the cell key
  and rejected at configuration time.
- FSDP1, tensor parallelism, and pipeline parallelism under FSDP.
- The kernel-summation latency model. This design records the comm/compute
  decomposition that a later one will need.
- `param_offload=True` (psrl pins it `False` because nixl_cpu cannot offload actor
  parameters).
