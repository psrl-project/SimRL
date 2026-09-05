# Measurement methodology

Everything on this page was learned by getting it wrong first. A cost model can only be as
good as the measurements behind it, and the failures in this list all produced
*plausible-looking* numbers, which is what makes them worth writing down. A profiling run that
crashes tells you it failed. One of these tells you nothing at all.

---

## Discard more warmup than feels necessary

Under a single warmup step, **the first profiled step was the slowest in 18 of 24 Megatron
runs**. Chance alone would give about eight. Twelve steps with four discarded cut the
median within-candidate spread from 16.4% to 0.5%.

Both training configs therefore use `max_steps: 12, warmup_steps: 4`.

The first step is disproportionate for reasons that have nothing to do with steady-state
cost: lazy CUDA context initialisation, autograd graph construction, allocator growth. On
one FSDP candidate the first step took 7.3 s against a 0.49 s steady state, a factor of
fifteen. Averaging that into a per-token cost inflates it invisibly.

The launcher retains all steps and the **collector** slices off the warmup window. That
split matters: it means you can change how many steps are discarded, and re-derive the
fit, without re-running a six-hour sweep.

---

## Report `max(retained) / median`, not just MAD

Median absolute deviation is blind to a single outlier among eight samples, and it missed
profiler contamination **twice** in this project. Both times the MAD looked healthy while
one retained step was several times the others.

Report both. The FSDP smoke gate does, and the run that passed it reported MAD 0.67% *and*
`max/median` 1.025. The second number is the one that would have caught the failures the first
number missed.

The general shape of the trap: a dispersion statistic chosen for robustness against
outliers will, by construction, fail to report outliers. If contamination is the thing you
are worried about, you need a statistic that is sensitive to it.

---

## Out-of-memory results are data, not failures

An OOM candidate bounds the memory model **from below**: whatever the true peak was, it
exceeded the device. That is a real inequality constraint, and the memory fit uses it.

Two consequences follow.

A sweep that never OOMs has not found the ceiling. In the FSDP sweep, 32768 tokens per rank
produced 0 successes against 18 OOMs and 65536 produced 0 against 9, which is precisely how
the token range's upper bound was established. Widening the grid further would have
measured nothing.

**Never infer OOM by scanning a log.** OOM is decided from marker files a launcher writes
on a caught allocation failure. A log that merely contains the phrase must stay `failed`,
because a false positive can only *raise* the fitted memory floor, making the model quietly
more conservative in a way no metric reports. See {doc}`fsdp` for one way this still goes
wrong.

---

## A term can be right or wrong depending only on the domain you measured

The training latency models carry a quadratic token term. An earlier version of this page
said not to add one, and that conclusion was correct for the sweep it was tested against.

Tested on Megatron candidates capped at 2048 tokens:

| Latency form | In-sample MRE | Leave-one-out MRE |
|---|---|---|
| linear | 6.60% | **10.24%** |
| with quadratic term | **4.65%** | 10.38% |

Better in-sample, worse out-of-sample, which is the signature of a term absorbing residual
rather than representing a mechanism. And there was a physical reason to expect no signal:
attention is only about 3.7% of a layer's FLOPs at 2048 tokens under full activation
recomputation.

Extending the sweep to 8192 tokens reverses the verdict. Attention reaches 13.3% of a layer's
FLOPs there, and NNLS leave-one-out on the same cells gives:

| cell | seq ≤ 2048, linear / quadratic | seq ≤ 8192, linear / quadratic |
|---|---|---|
| TP1/PP1 | 14.17% / 16.29% | **60.48% / 5.52%** |
| TP4/PP1 | 20.13% / 18.00% | **72.85% / 26.54%** |

The lesson is not about this particular term. **A leave-one-out result is evidence about the
domain you measured, not about the model.** The earlier conclusion was correctly derived and
still wrong for the domain the simulator needs, because the sweep it rested on did not reach
the sequence lengths a real DAPO batch produces. When a term is rejected, record the domain
alongside the rejection.

Prefer a form you can defend physically. The shared latency model is
`per_step_overhead + slots * (token_cost * tokens + token_quadratic_cost * sum_squared_tokens
+ microbatch_overhead)`, where every term names something real. The recovered `token_cost`
came out at 458.9, 467.0, and 466.7 µs per token across three independently fitted FSDP
meshes, consistent to 2%. Agreement like that across cells that never saw each other's data
is evidence you recovered a physical constant rather than a curve.

**The quadratic feature must be $\sum_i s_i^2$, not $(\sum_i s_i)^2$.** Attention does not
cross sequence boundaries, so a packed microbatch costs the sum of each sequence's square.
Squaring the packed total overcharges by the sequence count, an 8x error for eight packed
sequences.

---

## Vary every axis the design matrix needs

A regression refuses a rank-deficient design rather than returning coefficients it cannot
separate. The usual way to hit this is sweeping the axis you find interesting while holding
constant one the model needs.

The FSDP latency design row is `[1, nmb * tokens, nmb, nmb * sum_squared_tokens]`. Holding
`num_microbatches` constant makes the third column a multiple of the first, capping the design
at rank 3 while the fit demands rank 4, so **the entire sweep becomes unfittable** after six
hours of measurement. Sweeping sequence length alone is not enough, because it moves the token
columns only.

Two `num_microbatches` values are the minimum, and `{1, 4}` beats `{1, 2}` at the same
candidate count because it opens the slot lever wider. Measured on synthetic data at 1%
step-time noise, the worst recovered coefficient was off by 21.5% against 38.3%.

Check the design before committing GPU time, not after.

---

## Identifiable is not the same as well-conditioned

Reaching full rank means the coefficients are separable in principle. It does not mean they
are separately measurable.

The two token columns in the four-term latency design are highly collinear. Over the 40
Megatron candidates, `corr(slots*tokens, slots*sum_squared_tokens) = 0.963` with a condition
number of 2.3e8. The design reaches rank 4, and NNLS still resolves the near-ambiguity by
pushing one coefficient toward zero: the TP8/PP1 cell's `token_cost` collapsed to 1.97e-13,
which is not a claim about that hardware.

Predictions stay accurate because the two terms compensate over the fitted range. The
individual coefficients should not be quoted as measured per-token costs. Report the condition
number alongside the rank, and treat a coefficient pinned at zero as a conditioning symptom
rather than a finding.

---

## A small held-out split is noise, not evidence

`--validation-fraction 0.2` on 10 candidates reserves two points, and one unlucky point then
dominates the reported error. The Megatron TP4/PP1 cell reports 23.28% held-out MRE where its
leave-one-out error over the same cell is 12.46%.

Use leave-one-out on grids this small. A two-point split invites both false alarm and false
confidence, and there is no way to tell which one you got from the number alone.

---

## A minimum useful sweep

Cheaper than a full identifiability recipe, and still sufficient:

- One fixed world size, with DP derived as `world_size / (TP * PP)`.
- Micro-batch size 1.
- Explicit `num_microbatches` values covering an underfilled pipeline, a full pipeline, and
  the simulated global batch size.
- At least four sequence lengths.
- Full activation recomputation.
- Twelve steps, four discarded.
- One process per topology. Do not repeat an identical config.

A fuller design additionally benefits from a second micro-batch size, equal-total-token
shapes with differing $\sum L^2$, and local microbatch counts at `{PP, 2*PP, 4*PP}`.

---

## Measure the thing you are modelling

Two mistakes in this class cost real time.

**Reserved versus allocated memory.** Fitting `reserved` bytes gave $r^2 = 0.196$, and the
value moved *backwards* when sequence length doubled, because reserved bytes are an allocator
caching artefact rather than a footprint. Switching the target to `allocated` gave
$r^2 = 1.0$. Both are still recorded, because the simulator compares against device
capacity and it is reserved bytes that actually cause an OOM, so the ratio is applied at
predict time.

**Where the profiler sits.** An early version profiled a *retained* step, inflating it 5.4
to 13x. A later version moved the profiler off the retained steps but left its tail landing
on the *next* step, inflating that one 2.73x. That was small enough for MAD to miss, which is
how it survived review. Profiled and timed steps must be disjoint.

A third instance of the same class: a topology check that echoed back the requested spec
rather than reading the live device mesh. It passed unconditionally and verified nothing.
Cross-checks must read the system, not the request.

---

## Optimise what actually dominates

Roughly **70% of a training candidate's wall clock is process startup**, not measurement.
Before optimising a sweep, measure where its time goes. The intuitive lever is usually the
wrong one. {doc}`index` has the breakdown, and explains why cutting `max_steps` saves about
1% while costing half the retained samples.

---

## Reading a `torchrun` failure

The `SignalException ... got signal: 15` traceback at the top of `run.log` is noise: it is
the launcher tearing down surviving ranks after one rank already died. The real culprit is
**the only rank whose `exitcode` is not `-15`**.

Find the exit codes first, identify that rank, and read its output. Starting from the
top-of-log traceback leads you into the teardown path every time.
