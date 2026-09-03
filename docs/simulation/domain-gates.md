# Domain gates

A cost model will refuse a query it cannot justify rather than extrapolate silently. This
is the single most common first-run failure, and it is intentional: a plausible-looking
number derived from a region where the hardware cannot even run is worse than no number,
because it passes downstream checks that would have caught a hard error.

This page explains the two categories of refusal — out-of-domain queries and OOM
projections — plus the three batching modes that affect how a real batch's lengths become
a cost query.

---

## Interpolation vs extrapolation

The most common misunderstanding is conflating the two. They are different:

- **Interpolation** means the query lies within the range of the profiled grid. It is
  *always allowed* and requires no configuration. A query at 5905 tokens against a grid
  of `[512, 1024, 2048, 4096, 8192, 16384]` is interpolation — the value falls between
  two profiled points — and the cost model answers it without complaint.
- **Extrapolation** means the query lies outside the profiled range. Going below 512 or
  above 16384 tokens in the example above is extrapolation, and the cost model raises
  `TrainingOutOfDomainError` unless the feature name has been explicitly listed in
  `training_cost.extrapolate`.

The distinction matters because the profiled range usually covers the space where the
hardware actually runs. Outside that space, the model is making a prediction without any
measured ground truth to anchor it.

---

## `training_cost.extrapolate`

`training_cost.extrapolate` is a list of feature names for which out-of-domain queries
are permitted. It defaults to the empty list, meaning any out-of-domain query raises
immediately at predict time.

To permit a specific feature:

```bash
# Hydra override
training_cost.extrapolate=[num_microbatches]

# or via the simulate.sh env-var interface
EXTRAPOLATE=[num_microbatches] bash examples/simulate/simulate.sh
```

**A name not recognised by the artifact raises at model construction time** and lists
the valid names. This catches typos (`num_microbatch` instead of `num_microbatches`)
before any simulation begins. The check uses the feature names recorded in the artifact
itself (`entry.feature_ranges.keys()`), so it stays correct as the feature set evolves.

**A permitted extrapolation is never silent.** When a query exceeds the profiled range
for a listed feature, `check_feature_range` returns a diagnostic string of the form:

```
extrapolated 'num_microbatches'=8 outside profiled range [1, 4]
```

That string is collected into `TrainingCostEstimate.diagnostics["extrapolation_notes"]`
and written verbatim to the `extrapolation_notes` field of the `train.jsonl` record.
Every step that used an extrapolated prediction is therefore auditable after the fact.

---

## Which features are defensible to extrapolate

Not all features are safe to extrapolate, even upward. The FSDP artifact documents the
reasoning per feature in `dryrun/config/defaults/training_cost/fsdp.yaml`.

### `num_microbatches` — yes, upward

Latency under gradient accumulation is exactly linear in the number of microbatches.
Each accumulation step repeats identical work on its own microbatch, and the steps do
not interact. The latency form is:

```
latency = per_step_overhead + num_microbatches * (token_cost * tokens + microbatch_overhead)
```

Because the relationship is a physical identity rather than a statistical correlation,
extrapolation is defensible. Fitting on `num_microbatches` in `{1, 2}` and predicting
the 23 measured `nmb=4` candidates gave a median absolute error of **7.7%**, on par
with the model's own 9–11% in-domain MRE (measured for Qwen3-8B on 8x H20, 2026-08-31,
23 held-out candidates). The risk is not the extrapolation itself but the assumption
that the overhead terms do not change with microbatch count; for the FSDP sweep they
did not.

### `tokens` — no

Beyond 16384 tokens per rank the hardware cannot run. The FSDP sweep found: at 32768
tokens/rank, 0 successes and 18 OOM failures; at 65536 tokens/rank, 0 successes and 9
OOM failures (Qwen3-8B on 8x H20, 2026-08-31). There is no measured latency at those
shapes to extrapolate toward. A prediction there looks numerically plausible because the
linear form has no saturation term, but it describes a configuration that the hardware
physically cannot execute. Do not list `tokens` in `extrapolate` unless you have profiled
beyond 16384 tokens/rank.

The same reasoning applies to `activation_tokens` in the memory model.

---

## Cell lookup

Coefficients are stored per cell, where a cell is an (shard\_size, activation\_recompute,
optimizer, …) combination that was profiled independently. A query that matches no cell
raises rather than falling back to a neighbouring cell.

This is deliberate. A fit averaged over multiple topologies describes none of them: the
shard=2 latency is not the average of shard=4 and shard=8, because the communication
pattern changes with the mesh. A fallback would silently produce an answer calibrated
to the wrong topology. Failing loudly forces the user to either profile the missing cell
or choose one that was profiled.

The cell selectors for the FSDP artifact are `shard_size`, `replicate_size`,
`activation_recompute`, `optimizer`, `optimizer_offload`, and `reshard_after_forward`.
Every selector value must match a profiled cell exactly.

---

## The OOM check

When `training_cost.gpu_memory_mib` is set to the device's capacity in MiB, the
simulator compares the predicted peak memory against that capacity. A step whose
predicted peak exceeds the device is flagged as an OOM in the simulation output rather
than being silently accepted.

This check uses *reserved* bytes (the allocator's high-water mark) rather than
*allocated* bytes, because OOM is triggered by the allocator failing to satisfy a
reservation, not by the instantaneous allocation. The memory model fits `allocated`
bytes (which has $r^2 \approx 1.0$ for the FSDP artifact), then applies a
`reserved/allocated` ratio measured from the profiling sweep to recover the reserved
prediction. The ratio ranged from 1.148 to 1.901 across the three shard cells measured
for Qwen3-8B (8x H20, 2026-08-31).

Leaving `gpu_memory_mib: null` (the default) disables the check and lets all steps
through. Set it to the device's memory when you want the simulation to identify
configurations that would OOM in practice.

---

## The three batching modes

Profiling measures uniform shapes: every sequence in a microbatch has the same length.
A real RL batch has heterogeneous lengths. The `batching_mode` setting controls how
the cost model translates real lengths into a microbatch plan, and the choice matters.

The measured spread between the extremes is **1.17x to 4.34x** in latency for
DAPO-like bimodal length distributions, so the mode is a first-class configuration
choice rather than a detail.

### `pad_to_max` (default)

Every microbatch is padded to its longest sequence. Padding tokens cost full compute
on the critical rank. This is the conservative estimate: it charges for work the
hardware actually does when padding is present.

### `token_budget`

Sequences are grouped to a token budget per rank rather than split into fixed-size
microbatches. The number of microbatches is `ceil(total_tokens / max_tokens_per_rank)`,
capped so no microbatch exceeds `micro_batch_size` sequences. Within each group,
padding still applies to the longest sequence. This reproduces verl's dynamic batching
strategy.

Requires `training_cost.max_tokens_per_rank` to be set, or equivalently the
`MAX_TOKENS_PER_RANK` environment variable in `simulate.sh`. A common value for DAPO
training with Qwen3-8B is 10240.

### `packed_no_pad`

Sequences are concatenated end-to-end with no padding tokens. The cost is charged on
true token counts only. This is the optimistic estimate: it requires the training
backend to support packed sequences (flash-attention with variable-length inputs).

---

## Why a three-way tie is expected, not a wiring bug

At `micro_batch_size=1`, there is exactly one sequence per microbatch. `pad_to_max`
pads to the single sequence's own length — which is itself — so padding adds nothing.
`token_budget` forms microbatches of one sequence each. `packed_no_pad` concatenates
one sequence, which is already contiguous. All three charge the same true token count,
and the simulator correctly reports identical results.

The same collapse occurs at `mbs>1` when `_balanced_partitions` is used. The balanced
partitioner sorts sequences by a workload proxy (verl's
`_VERL_LINEAR_WEIGHT * length + length^2`) and groups them using a greedy
longest-processing-time heuristic. This tends to put long sequences together and short
sequences together, so each microbatch ends up internally uniform. When sequences in a
group are already close in length, the padding penalty in `pad_to_max` is small, the
token-budget grouping produces the same groups, and `packed_no_pad` charges nearly the
same real tokens. The three modes converge again.

To see the modes separate you need `mbs>1` **and** grouping that mixes long and short
sequences in the same microbatch. The **1.17x to 4.34x** figure quoted above was
measured on deliberately ragged microbatches; it does not transfer to a simulation using
the balanced planner. If you believe padding is a significant cost in your real system,
measure the fill ratio in your actual training runs before choosing a mode.

---

## Configuring extrapolation in `simulate.sh`

The example script in `examples/simulate/simulate.sh` exposes the two relevant knobs
as environment variables:

```bash
# Permit num_microbatches to exceed the profiled range
EXTRAPOLATE=[num_microbatches] \
TRAINING_BACKEND=fsdp \
TRAINING_ARTIFACT=outputs/cost/qwen3_8b/training_fsdp.json \
bash examples/simulate/simulate.sh

# verl's token-budget batching
BATCHING_MODE=token_budget \
MAX_TOKENS_PER_RANK=10240 \
bash examples/simulate/simulate.sh
```

The bracket syntax (`[num_microbatches]`) is Hydra's list override format. To permit
multiple features, separate them with commas: `[num_microbatches,sequences_per_microbatch]`.
