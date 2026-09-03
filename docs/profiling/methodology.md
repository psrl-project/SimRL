# Measurement methodology

Everything on this page was learned by getting it wrong first. A cost model can only be as
good as the measurements behind it, and the failures in this list all produced
*plausible-looking* numbers — which is what makes them worth writing down. A profiling run
that crashes tells you it failed; one of these tells you nothing at all.

---

## Discard more warmup than feels necessary

Under a single warmup step, **the first profiled step was the slowest in 18 of 24 Megatron
runs**. Chance alone would give about eight. Twelve steps with four discarded cut the
median within-candidate spread from 16.4% to 0.5%.

Both training configs therefore use `max_steps: 12, warmup_steps: 4`.

The first step is disproportionate for reasons that have nothing to do with steady-state
cost: lazy CUDA context initialisation, autograd graph construction, allocator growth. On
one FSDP candidate the first step took 7.3 s against a 0.49 s steady state — a factor of
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
`max/median` 1.025 — the second number is the one that would have caught the failures the
first number missed.

The general shape of the trap: a dispersion statistic chosen for robustness against
outliers will, by construction, fail to report outliers. If contamination is the thing you
are worried about, you need a statistic that is sensitive to it.

---

## Out-of-memory results are data, not failures

An OOM candidate bounds the memory model **from below**: whatever the true peak was, it
exceeded the device. That is a real inequality constraint, and the memory fit uses it.

Two consequences follow.

A sweep that never OOMs has not found the ceiling. In the FSDP sweep, 32768 tokens per rank
produced 0 successes against 18 OOMs and 65536 produced 0 against 9 — which is precisely
how the token range's upper bound was established. Widening the grid further would have
measured nothing.

**Never infer OOM by scanning a log.** OOM is decided from marker files a launcher writes
on a caught allocation failure. A log that merely contains the phrase must stay `failed`,
because a false positive can only *raise* the fitted memory floor, making the model quietly
more conservative in a way no metric reports. See {doc}`fsdp` for one way this still goes
wrong.

---

## Do not add a quadratic token term

It was tried. The result:

| Latency form | In-sample MRE | Leave-one-out MRE |
|---|---|---|
| linear | 6.60% | **10.24%** |
| with quadratic term | **4.65%** | 10.38% |

Better in-sample, worse out-of-sample — the signature of a term absorbing residual rather
than representing a mechanism. The implied coefficient also corresponded to roughly 18000%
of the H20's bf16 throughput, which is not a physical quantity.

There is a reason no quadratic signal exists to find: under full activation recomputation
attention is only 2-3% of a step. The $L^2$ term that dominates a long-context inference
step is nearly absent from a recomputed training step.

Prefer a form you can defend physically. The training latency models take the shape
`per_step_overhead + slots * (token_cost * tokens + microbatch_overhead)`, where every term
names something real. The recovered `token_cost` came out at 583.8 / 600.5 / 601.5 µs per
token across three independently fitted meshes — consistent to 3%. Agreement like that
across cells that never saw each other's data is evidence you recovered a physical constant
rather than a curve.

---

## Vary every axis the design matrix needs

A regression refuses a rank-deficient design rather than returning coefficients it cannot
separate. The usual way to hit this is sweeping the axis you find interesting while holding
constant one the model needs.

The FSDP latency design row is `[1, nmb * tokens, nmb]`. Holding `num_microbatches`
constant caps that design at rank 2 while the fit demands rank 3, so **the entire sweep
becomes unfittable** — after six hours of measurement. Sweeping sequence length alone is
not enough, because it moves the second column only.

Check the design before committing GPU time, not after.

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
- One process per topology — do not repeat an identical config.

A fuller design additionally benefits from a second micro-batch size, equal-total-token
shapes with differing $\sum L^2$, and local microbatch counts at `{PP, 2*PP, 4*PP}`.

---

## Measure the thing you are modelling

Two mistakes in this class cost real time.

**Reserved versus allocated memory.** Fitting `reserved` bytes gave $r^2 = 0.196$, and the
value moved *backwards* when sequence length doubled — because reserved bytes are an
allocator caching artefact, not a footprint. Switching the target to `allocated` gave
$r^2 = 1.0$. Both are still recorded, because the simulator compares against device
capacity and it is reserved bytes that actually cause an OOM, so the ratio is applied at
predict time.

**Where the profiler sits.** An early version profiled a *retained* step, inflating it 5.4
to 13x. A later version moved the profiler off the retained steps but left its tail landing
on the *next* step, inflating that one 2.73x — small enough for MAD to miss, which is how it
survived review. Profiled and timed steps must be disjoint.

A third instance of the same class: a topology check that echoed back the requested spec
rather than reading the live device mesh. It passed unconditionally and verified nothing.
Cross-checks must read the system, not the request.

---

## Optimise what actually dominates

Roughly **70% of a training candidate's wall clock is process startup**, not measurement.
Before optimising a sweep, measure where its time goes — the intuitive lever is usually the
wrong one. {doc}`index` has the breakdown, and explains why cutting `max_steps` saves about
1% while costing half the retained samples.

---

## Reading a `torchrun` failure

The `SignalException ... got signal: 15` traceback at the top of `run.log` is noise: it is
the launcher tearing down surviving ranks after one rank already died. The real culprit is
**the only rank whose `exitcode` is not `-15`**.

Find the exit codes first, identify that rank, and read its output. Starting from the
top-of-log traceback leads you into the teardown path every time.
