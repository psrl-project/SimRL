# Measurement methodology

A cost model can only be as good as the measurements behind it, and the failures this page
guards against all produce *plausible-looking* numbers. A profiling run that crashes tells you
it failed. A run that measured the wrong thing tells you nothing at all. Each section below
states a rule, names the code that enforces it, and says what to check where the code cannot.

Measured accuracy figures are not repeated here. They depend on a GPU, a model and a sweep that
the repository does not carry, so {doc}`../reference/validation` owns the evidence and this page
owns the method.

---

## Discard warmup in the collector, not the launcher

The first steps of a training candidate are not steady state: lazy CUDA context initialisation,
autograd graph construction and allocator growth all land there. Averaging them into a per-token
cost inflates it invisibly.

The launcher records **every** step in `step_times_s`. The collector then drops the leading
`warmup_steps` through `slice_warmup` (`srl/profiler/adapters/base/sidecar.py`). Because the
split happens at `collect` time, the discarded window can change and the fit can be re-derived
without re-running a sweep.

- `max_steps` must exceed `warmup_steps`, or the config is rejected.
- The reference `qwen3_8b.yaml` configs for both backends use `max_steps: 12` and
  `warmup_steps: 4`. The FSDP config class defaults to the same pair. `MegatronBridgeConfig`
  defaults to `4` and `1`, so a Megatron config that omits the `training:` block profiles with a
  much shorter warmup than the example does. The THD example deliberately shortens both.
- `slice_warmup` returns the full series when it holds no more steps than the warmup would
  discard. A candidate is thereby never turned into an unmeasured one by an aggressive
  `warmup_steps`. The contamination shows up as dispersion instead.

---

## Reduce a candidate to one robust number, and read the dispersion

Each candidate contributes **one** design row to a training fit: the median of its retained step
times (`robust_step_time`). Step times are right-skewed, since an interfering process inflates a
step and nothing makes one faster than the hardware allows. The median is unbiased against that
skew where the mean is not, and one row per candidate stops a noisy candidate outvoting a clean
one.

Every fitted latency entry also records `max_step_dispersion` and `median_step_dispersion`, the
relative median absolute deviation of the retained steps (`step_time_dispersion`).

A MAD is robust to outliers by construction, which means it cannot report one. A single
contaminated step among the retained ones leaves it unchanged. When contamination is the concern,
compare the largest retained value in `srl_metrics.json` against the median directly. The code
does not compute that ratio for you.

---

## Out-of-memory results are data, not failures

An OOM candidate bounds the memory model **from below**: whatever the true peak was, it
exceeded the device. Both memory fits turn each OOM record into an inequality and raise the
fitted intercept until the model would predict an OOM there too
(`oom_constraints` and `oom_intercept_increase` in the entry diagnostics). Two consequences:

- A sweep that never OOMs has not found the ceiling. The upper end of a token or microbatch
  axis is established by where candidates start failing, and widening a grid past that point
  measures nothing.
- A fit made only of successes can look good while being unsafe near capacity. A conservative
  intercept that makes the in-sample $r^2$ poor is the mechanism working, not a defect. See
  {doc}`../cost-models/training-fsdp`.

**OOM is never inferred from a log.** `read_status` (`adapters/base/run.py`) classifies a
candidate `oom` only when the launcher wrote an `oom_rank{N}.json` marker after catching the
failure itself. A log that merely contains the phrase stays `failed`, because a false positive
can only *raise* the fitted memory floor, making the model quietly more conservative in a way no
metric reports.

There are exactly two deliberate exceptions, both decided inside the launcher child rather than
from text in `run.log`:

- **Megatron.** A `RuntimeError` naming NCCL and `unhandled cuda error` (or `out of memory`) is
  classified as OOM only if `torch.cuda.mem_get_info()` then shows less than 2% of the device
  free. NCCL allocates communicator buffers outside the PyTorch allocator, so exhaustion
  surfaces there rather than as `torch.OutOfMemoryError`. A fault on a device with room to spare
  re-raises and stays `failed`.
- **FSDP.** A `DistBackendError` whose message contains `failed to cuda calloc` or
  `out of memory` is classified as OOM, and the marker records `oom_source: nccl_alloc`. Any
  other `DistBackendError`, such as a timeout or a rendezvous failure, stays `failed`.

Both write the same marker file, so the collector's rule stays one rule. See {doc}`megatron` and
{doc}`fsdp`.

`run` deletes stale markers before launching a candidate. If a re-run is then killed before it
writes new ones, `status.json` still reads `oom` from the earlier attempt while the collector
sees no markers and reports `failed`. A candidate whose `status.json` and collected record
disagree should be re-run to completion with `--candidate-id`.

---

## Vary every axis the design matrix needs

A regression refuses a rank-deficient design rather than returning coefficients it cannot
separate. The shared latency design row (`LatencyFeatures.design_row`) is

$$
\bigl(1,\ \textstyle\sum_\text{mb} T,\ \text{slots},\ \sum_\text{mb} \sum_i s_i^2\bigr)
$$

and must reach rank `LATENCY_TERM_COUNT = 4`. `check_latency_rank` raises otherwise, naming the
cell and the axes to vary.

- Holding the microbatch count fixed makes the slot column constant, collinear with the
  intercept. The whole sweep then becomes unfittable. Two distinct microbatch counts are the
  minimum, and widely spaced values condition the slot coefficient better than adjacent ones.
- Sweeping sequence length alone moves only the token columns.
- Two sequence lengths leave the linear and quadratic token columns collinear. Use at least
  three.
- FSDP memory fits `[num_microbatches, max_tokens, 1]` and requires rank 3, so it needs both
  axes to vary as well. Hydraulis memory fits `[activation_bytes, 1]` and requires rank 2.

Check the design before committing GPU time. `generate` is free and shows every candidate.

### Identifiable is not the same as well-conditioned

Reaching full rank means the coefficients are separable in principle, not that they are
separately measurable. Under BSHD with a micro-batch size of one, every microbatch has
$\sum_i s_i^2 = (\sum_i s_i)^2$, so the linear and quadratic token columns are strongly
correlated and a non-negative fit can resolve the ambiguity by pushing one coefficient to its
bound of zero. The predictions stay accurate while the individual coefficients stop being
physical per-token costs. A coefficient pinned at zero is a conditioning symptom, not a finding.

THD packing is the remedy: shapes that share a token total but split it differently move
$\sum_i s_i^2$ independently of $\sum_i s_i$. See `sequence_shapes` in {doc}`megatron`.

### The quadratic term is $\sum_i s_i^2$, and its value depends on the domain

Attention does not cross sequence boundaries, so a packed microbatch costs the sum of each
sequence's square. Squaring the packed total overcharges by the sequence count.
`SequenceStats.sum_squared_tokens` records the correct quantity.

Whether a quadratic term earns its place depends on the sequence lengths swept. At short
lengths attention is a small share of a layer and the term mostly absorbs residual, so
leave-one-out can prefer the linear form. At the lengths a real RL batch produces, attention
dominates enough that omitting the term is off by a factor. A leave-one-out result is evidence
about the domain that was measured, not about the model. Record the domain next to any term you
reject, and make the sweep reach the lengths the simulator will ask about, because the fitted
`feature_ranges` become prediction-time gates.

---

## A small held-out split is noise

`srl-regress training` defaults `--validation-fraction` to `0.2`, and the split is by `run_id`
so repeats of one candidate cannot straddle it. On a cell with ten candidates that reserves
two. One unlucky point then dominates the reported held-out error.

Every entry records `validation_metrics.n_samples` and `validation_is_in_sample`. Quote the count
next to any held-out figure, and prefer leave-one-out on grids this small. Rollout fits default
to `--validation-fraction 0.0`, which makes their validation metrics in-sample.

---

## Measure the thing you are modelling

**Allocated versus reserved memory.** Reserved bytes are the caching allocator's segment pool.
They are quantised and sticky, and can move the wrong way when a workload grows, so they make a
poor regression target. Allocated bytes track the footprint. The simulator, however, compares
against device capacity and it is reserved bytes that exhaust a device. Hydraulis therefore fits
allocated and leaves any provisioning margin to the caller. FSDP fits both, selects per cell on
validation $r^2$, and when allocated wins it stores the measured `reserved_over_allocated` ratio
and applies it at predict time.

**Where the profiler sits.** Profiling a step perturbs it and can leave a tail on the next one. The
FSDP launcher therefore measures collective occupancy in a separate step *after* the timed loop,
so that no retained sample is contaminated. Timed and profiled steps must be disjoint.

**Cross-checks must read the system, not the request.** The FSDP launcher records the mesh it
actually built as `observed_mesh`, and `collect` rejects a candidate whose observed mesh differs
from the requested one. Megatron's collector cross-checks the resolved `ConfigContainer.yaml`
the same way. A check that echoed the request back would pass unconditionally.

**Timing mode must match how the engine runs the kernel.** Kernel ops declare
`graph_replayed`. CUDA-event timing of a host-bound kernel measures the Python interpreter
rather than the device. See {doc}`kernels`.

**Run on a node nobody else is using.** Kernel and collective timings are wall-clock sensitive
and nothing in the output reveals that another job shared the node, even when `nvidia-smi` looks
idle. For the same reason, collect any vLLM baseline whose figures will be quoted serially
(`max_parallel: 1`), and treat a larger batch that measures *faster* than a smaller one at the
same depth as the signature of contention rather than of an outlier. See {doc}`vllm`.

---

## Reading a `torchrun` failure

The `SignalException ... got signal: 15` traceback at the top of `run.log` is noise: it is the
launcher tearing down surviving ranks after one rank already died. The real culprit is
**the only rank whose `exitcode` is not `-15`**. Find the exit codes first, identify that rank,
and read its output. Starting from the top-of-log traceback leads into the teardown path every
time.
