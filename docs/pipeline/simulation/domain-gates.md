# Domain gates

A cost model refuses a query it cannot justify instead of extrapolating silently. A
plausible number derived from a region where nothing was measured, or where the hardware
cannot even run, is worse than an error, because it passes every downstream check that a
hard failure would have stopped. This page lists the refusals, what each one means, and the
switches that relax them.

## Kinds of refusal

| Gate | Raised as | When |
|---|---|---|
| Feature range | `TrainingOutOfDomainError` | a training feature lies outside the range the artifact was fitted on and is not listed in `extrapolate` |
| Cell lookup | `KeyError` (artifact), `TrainingOutOfDomainError` (model) | the requested mesh or discrete selector matches no fitted cell, or more than one |
| Memory capacity | `RuntimeError` from the simulator | predicted training peak memory exceeds `training_cost.gpu_memory_mib` |
| Kernel coverage | `KernelCoverageError` | the `composed` rollout model is asked for a shape its measured kernel tables do not cover |
| Structural | `ValueError` | an inconsistent configuration, see the end of this page |

## Interpolation and extrapolation

Each fitted artifact entry records, per feature, the range of values it was fitted on
(`feature_ranges`). A query inside the range is **interpolation** and always allowed. A
query outside it is **extrapolation** and raises `TrainingOutOfDomainError` naming the
feature, the value and the range, unless that feature is listed in `training_cost.extrapolate`.
A feature the artifact did not record is not checked. Refitting restores the gate.

```bash
srl training_cost=fsdp training_cost.artifact_path=... \
    'training_cost.extrapolate=[num_microbatches]'
```

`extrapolate` defaults to the empty list. Names are validated at construction against the
union of feature names present in the artifact, so a typo such as `num_microbatch` raises
immediately and lists the valid names, instead of enabling nothing.

A permitted extrapolation is never silent. The prediction proceeds and a note of the form
`extrapolated 'num_microbatches'=8 outside profiled range [1, 4]` is collected into
`TrainingCostEstimate.diagnostics["extrapolation_notes"]` and written to the
`extrapolation_notes` field of that step's `train.jsonl` record.

### Which features can be extrapolated

The names below are the ones the code checks.

| Backend | Latency features | Memory features |
|---|---|---|
| `fsdp` | `tokens`, `num_microbatches`, `sequences_per_microbatch` | `num_microbatches`, `activation_tokens` |
| `hydraulis` | `tokens`, `pipeline_slots`, `sum_squared_tokens`, `micro_batch_size` | `activation_bytes`, and `profiled_dp` when an artifact covers one DP degree and the query uses the analytical DP extension |

Not every feature is equally defensible. The per-feature reasoning is kept next to the
defaults in `srl/config/defaults/training_cost/fsdp.yaml` and `hydraulis.yaml`.

`num_microbatches` and `pipeline_slots`
: The fitted latency form is a sum over microbatches of per-microbatch work, plus a per-step
  overhead and a per-microbatch overhead. Gradient accumulation repeats identical work, so
  the form is linear in the count and extrapolating the count upward is the one extension
  that rests on structure rather than on a trend. The assumption to check is that the
  per-microbatch overhead does not change with the count.

`tokens`, `activation_tokens` and `activation_bytes`
: Do not list them without profiling past the target. Beyond a certain size the hardware
  cannot run the configuration at all, and a linear form with no saturation term still
  returns a smooth number there. The fitted ceiling is in the artifact's feature ranges and
  in the profiling run that produced it ({doc}`/pipeline/profiling/index`).

## Cell lookup

Coefficients are stored per cell, a combination of mesh and discrete selectors that was
profiled independently. A query that matches no cell raises. It is never answered from a
neighbour. A fit averaged across topologies describes none of them: the communication
pattern changes with the mesh, so a fallback would silently calibrate to the wrong one.

`fsdp`
: A cell is one measured mesh. `shard_size` and `replicate_size` pick it, and
  `training_cost.parallelism.dp` must equal `shard_size * replicate_size`. Any other `dp` is
  refused, because the mesh never enters the fitted form and the static memory pool scales
  as `1 / dp`. The schedule must be `fsdp` and `activation_recompute` must equal the cell's.
  Selectors are `shard_size`, `replicate_size`, `optimizer`, `optimizer_offload`,
  `reshard_after_forward` and the three dtypes. When an artifact holds several values of a
  selector and none is given, construction raises.

`hydraulis`
: Latency lookup is exact in TP and CP, and the pipeline degree is derived from the cell's
  per-stage arithmetic, so one TP cell serves every PP. Memory lookup also matches PP, because
  the 1F1B peak depends on which stage is in flight. DP uses the analytical extension. CP other than 1 is refused. The schedule must be `1f1b`. Without
  `max_tokens_per_microbatch`, the global batch must be divisible by `dp * micro_batch_size`.

## The memory capacity check

Set `training_cost.gpu_memory_mib` to the device capacity in MiB. After pricing each step
the simulator compares the predicted peak with it and **raises `RuntimeError`**, ending the
run, if the prediction exceeds capacity. `null` (the default) disables the check.

The FSDP memory model fits `peak_allocated_bytes` and `peak_reserved_bytes`, and picks the
better target on validation. When the allocated target is selected, the prediction is
multiplied by the artifact's measured `reserved_over_allocated` ratio, because an
out-of-memory failure is caused by the allocator's reserved pool and not by live tensors.
The check therefore compares like with like.

## Batching modes

Profiling measures uniform shapes, while a real RL batch has ragged lengths. How real
lengths become microbatches is decided by `MicrobatchPlanner` (`cost/training/microbatch.py`)
for the FSDP model, and by `TrainingWorkload` for `hydraulis`.

For both, the batch is first balanced across the `dp` ranks by a greedy
longest-processing-time pass over verl's workload proxy `24576 * length + length ** 2`, and
the model prices the rank with the largest workload, because a step ends when its slowest
rank ends.

`fsdp` takes `training_cost.batching_mode`:

| Mode | Microbatches on the critical rank |
|---|---|
| `pad_to_max` (default) | consecutive groups of `micro_batch_size` sequences, each padded to its longest member |
| `packed_no_pad` | the same groups, priced on true token counts with no padding |
| `token_budget` | `min(len(sequences), ceil(total_tokens / max_tokens_per_rank))` groups balanced by workload, each padded to its longest member. `micro_batch_size` is ignored, as in verl's dynamic batching. `max_tokens_per_rank` is required |

`hydraulis` has no `batching_mode`. It groups by `micro_batch_size`, or by a token budget
when `training_cost.max_tokens_per_microbatch` is set. That value is the already multiplied
budget: verl uses `ppo_max_token_len_per_gpu * sp_size`, and sequence parallelism makes
`sp_size` equal TP under Megatron.

### When the modes agree

`pad_to_max` and `packed_no_pad` differ only by padding. At `micro_batch_size=1` each
microbatch holds one sequence, so the padded length is its own length and the two are
identical by construction. For larger microbatches they separate only if a microbatch mixes
lengths, and that depends on the order the balanced partitioner leaves sequences in.
`token_budget` equals them only when the budget is small enough that every group holds one
sequence, or when the groups it forms happen to be uniform. Identical results from two modes
are therefore a property of the input shape and not evidence that a mode is unwired. If you
believe padding matters in your system, vary `micro_batch_size` and the length spread
before drawing a conclusion.

## Structural guards

These are not cost-model gates, but they are the other common reasons a run stops at
startup or early.

| Guard | Message gist |
|---|---|
| `job.batch_size < 1`, `group_size < 1` | must be at least 1 |
| `job.group_size` differs from the workload's | group size does not match the workload |
| `SIMRL_UNDOABLE_EVENTS` lacks `request_complete` | completions must be speculative |
| a request needs more KV blocks than an instance owns | it can never be admitted. The pool is too small for `prompt_len + target_len` |
| `rollout_cost=composed` without `kernel_table_root`, `gpu`, `model_path` | each is required |
| `rollout_cost=psrl` without `artifact_path` | required |
| `recompute_cost=training_forward` with no artifact descriptor and no checkpoint path | the geometry cannot be read, and it is never defaulted |
| an unknown policy, strategy, eviction, cost-model or workload name | `ValueError` or `KeyError` listing the valid names |

A run that stops making progress is reported rather than raised: after more than
`rollout.livelock_rounds` rounds without a change in time, in-flight, ready or dropped
counts, `SimResult.livelocked` is set and the CLI prints a warning.
