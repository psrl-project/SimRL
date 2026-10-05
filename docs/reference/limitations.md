# Limitations and deferred work

This page consolidates every known gap, with the source of each item and what would close it.
Items are grouped by kind. Acceptance criteria from the design document are quoted in full,
because they are the valuable part: they define what "done" means for each deferred item.

## What DryRun-RL does not do at all

Three limits are structural rather than gaps to be closed by more profiling.

- It does not simulate correctness or learning dynamics, only time and memory.
- It does not model network topology. Collective costs come from measured tables at the world
  sizes that were actually swept.
- Mixture-of-experts and multi-head latent attention architectures are **refused at
  configuration time** rather than approximated. A checkpoint carrying `num_experts` or
  `kv_lora_rank` is rejected with a message naming the key and the kernel that would have been
  omitted. See {doc}`../development/model-families`.

## Why the domain checks exist

A regression fitted over sequence lengths of 512 to 8192 tokens tells you nothing about 65536
tokens, and on the hardware this project was developed against, 65536 tokens per rank does not
run at all: 9 of the FSDP attempts at that shape were out-of-memory failures.

So a `CostArtifact` stores `feature_ranges` alongside its coefficients, and a query outside
them raises rather than returning a number. Extrapolation is possible but must be requested per
feature, by name, and every occurrence is recorded in the telemetry. See
{doc}`../simulation/domain-gates`.

That is also why {doc}`validation` exists as its own page: every accuracy figure on this site
carries its sample count and whether it was held out.

---

## Coverage gaps

These are cases where a configuration or hardware combination is valid but was never profiled,
so predictions there are either refused or extrapolated without supporting data.

**Megatron `tokens` sweep range.**
The Megatron-Bridge sweep covers `tokens ∈ [512, 8192]` per rank, widened from an earlier
[512, 2048] specifically because a real batch exceeded it. A DAPO batch's median sequence is
8828 tokens and its longest is 11133, so the tail still sits about 1.36x outside the domain, and
the memory model's `activation_bytes` domain is exceeded too. The gap is now bounded rather
than large, but it is real, and the latency form is linear in tokens while attention is
quadratic, so a stretched tail is understated. Run with
`EXTRAPOLATE=[tokens,activation_bytes]`, which records both exceedances in the telemetry. Fix:
extend `sequence_lengths` to 11264 if the tail needs to be trusted rather than bounded.

**FSDP `num_microbatches` range.**
The FSDP sweep covered `num_microbatches ∈ [1, 4]`. With `batch_size=64, dp=8, mbs=1` the
simulator needs `nmb=8`. The latency form is exactly linear in `nmb`, and fitting on
`nmb ∈ {1, 2}` to predict `nmb=4` measured a median absolute error of 7.7%, above the model's
own in-domain error of 0.65% to 4.24% but still a usable bound. Extrapolation can be enabled
with `training_cost.extrapolate=[num_microbatches]`. See {doc}`../simulation/domain-gates`.
Alternative fix: add `nmb=8` as a sweep point and refit.

**`optimizer_offload=True` never profiled.**
The 108-candidate FSDP sweep pinned `optimizer_offload=False` throughout. A workload that
enables CPU optimizer offload (common for very large models or tight GPU memory budgets) will
match no cell and raise `KeyError` at simulation time. Fix: run a separate sweep with
`optimizer_offload=True` and add the resulting cells to the artifact.

**Kernel sweep: TP=1 `holdout_report.json` is stale.**
The committed `holdout_report.json` under the TP=1 tree was written before the `elementwise`
and `all_reduce` families were swept and holds only five of the current six rows. Re-running
`dryrun-profile kernels holdout` generates six rows, which disagree with the committed file.
See {doc}`../profiling/kernels` for the canonical regenerated table.

---

## Model approximations

These are cases where the model is built and working but its accuracy on certain inputs is
known to be limited.

**`ComposedTraining` is 12.63% against measured steps; a kernel trace located the gap.**
The kernel-composed Megatron latency model scores MRE 12.63%, signed mean -12.63%, on the
22 out-of-sample THD records, against the fitted polynomial's 2.73% in-sample. Every record
under-predicts.

A CUDA kernel trace of rank 0 at 12288 tokens (`DRYRUN_TRACE_STEP=4 dryrun-profile megatron
run --config examples/profile/megatron/qwen3_8b_thd.yaml --candidates ... --index 1`, and
`--index 9` for nmb=4) compares term by term against the composition's per-stage cost:

| term | composed | traced | ratio |
|---|---|---|---|
| gemm | 0.8162 s | 0.8299 s | 0.98 |
| attention | 0.3927 s | 0.3936 s | 1.00 |
| tp_collective | 0.0912 s | 0.0824 s | 1.11 |
| elementwise | 0.0190 s | 0.0896 s | **0.21** |

What this establishes, which scoring against step times alone cannot: the GEMM and
attention tables are correct against the kernels that actually ran, and the sequence-parallel
collective counts derived from TE source are exact (traced 97 all-gathers and 73
reduce-scatters against the derived 96 and 72, the extra one each being process-group setup).

**The elementwise term is low by 4.6x, and that is most of the residual.** Scaled by `pp`
and the microbatch count it accounts for 4.9% of the 4.9% shortfall at nmb=1 and 10.0% of
the 13.6% shortfall at nmb=4. The cause is coverage, not formula: the sweep measures the
four kernels vLLM runs, where a traced Megatron step executes **64 distinct** non-GEMM,
non-attention, non-collective, non-optimizer kernels — generic `at::native` elementwise,
`CatArrayBatchedCopy` for the QKV split, bf16 copies, and TE's own rmsnorm rather than
vLLM's. Fix: sweep that population, keyed per kernel as the existing `op_kind` slice already
allows. Scaling the current term by 4.6 would fit a coefficient instead and is not done.

Two earlier hypotheses are now excluded by measurement rather than argument, and are
recorded because both were plausible: the GEMM table being cuBLAS where training is TE
(`te.Linear` against `torch.nn.functional.linear` is **1.001x**), and a fixed per-step or
per-slot overhead (per-slot residual spans 0.039-0.484 s, a 12x range).

Note also that the trace itself perturbs the step: traced total was +10.7% against the
untraced median at nmb=4. So per-term *ratios* within one trace are usable and traced
*totals* are not comparable to measured step times.

Two further approximations, each attributable:

- **The two residual adds are not charged.** Megatron fuses each into one
  `bias_dropout_add_fused_train` kernel, and the measured elementwise table holds only the
  four kinds vLLM runs. They are omitted rather than priced against `rms_norm`'s curve at
  the same element count, which would be plausible and unverifiable. Two kernels against the
  seven charged, so the elementwise term (1.4% of a step) is under-counted by roughly 2/9.
  Fix: extend the elementwise sweep with the fused bias-dropout-add kernel.
- **A ragged microbatch plan prices attention from the first microbatch's shape.** The
  per-sequence lengths are rescaled to the slot's mean token count, so a plan whose
  microbatches differ in sequence *count* is approximated. Not exercised by any profiled
  record (all 22 are uniform) and `MicrobatchPlanner` emits uniform plans except under
  `PACKED_NO_PAD` with ragged input. Fix: price each microbatch separately rather than
  pricing a mean slot.
- **Compute/communication overlap is not modelled.** The sum is a plain sum, so overlap
  would make it over-predict. It under-predicts everywhere, so overlap cannot be the
  residual above and is if anything masked by it: correcting the residual would make any
  unmodelled overlap visible as over-prediction. Megatron's `overlap_grad_reduce` and
  `overlap_param_gather` make this a real question at `dp>1`; at `dp=1` the only
  overlappable terms are the TP collectives (7.8% of a step) and the pipeline sends. Fix:
  revisit once the residual is explained.

The dispatch floor for both training attention directions is fitted to the candidate grid's
top value (80 us) where the measured flat region implies ~195 us, so shapes entirely below
~3072 tokens are under-predicted. The smallest real training microbatch in the validation
set is 4096 tokens, already past the flat region. Fix: extend `candidates_us` past 80 us,
which re-fits every floor-taking op and so requires re-pinning
`tests/test_collective_baseline_regression.py`.

**`ComposedTraining` at `dp>1` is not implemented.**
`TrainingOps.dp_comm_queries` raises `NotImplementedError` rather than returning a gradient
collective cost. Every THD validation record is `dp=1`, so the term could be written but not
scored, and an implemented-but-unvalidated term is indistinguishable from a validated one at
the call site. Spec A's tables cover the required collectives in `float32`. Fix: profile a
`dp>1` training candidate, then implement the term against it.

**`ComposedRollout` mixed steps.**
A mixed step (prefill and decode tokens in the same batch) is approximated as context
attention plus generation attention at the step's batch size. Only the two pure phases were
profiled, and no mixed batches were measured directly. Against the six real mixed iterations in the
vLLM baseline the error is 0.48%, but six samples is not a validation. A simulator with chunked
prefill produces mixed steps constantly. Fix: add a mixed-batch profiling sweep that measures
the combined attention kernel with a realistic split of prefill and decode tokens.

**`ComposedRollout` TP=2 is validated. Prefill is clean and decode's residual is the known
small-batch gap, not a tensor-parallelism one.** A serial vLLM TP=2 baseline exists
(`outputs/profile/qwen3_8b/vllm_tp2_serial/`, 49,211 iterations, `max_parallel: 1`, same grid as
TP=1) and `kernels validate` scores it:

| Selection | Buckets | MRE |
|---|---|---|
| prefill, whole grid | 29 | **3.60%** |
| decode, `requests >= 5` | 137 | **2.96%** |
| mixed | 7 | 3.64% |
| decode, whole grid | 204 | 9.51% |

The excluded-region figures beat the 4.84% decode / 3.72% prefill published at TP=1, which is the
first end-to-end evidence that kernel composition survives tensor parallelism. At TP=1
`comm_queries` returns empty, so no earlier figure exercised a collective at all.

Prefill needed no model change, only a better baseline. Its six `requests = 1` buckets originally
scored 12.50% because they used `num-prompts: 1`, so each recorded iteration was a single cold
request carrying ~78 ms of server-side work that no steady-state step pays. Fitting
`latency = fixed + per_token * tokens` isolated that offset (+68.2 ms in the serial run against
-1.3 ms in the parallel one, and absent at `requests = 3` on the same server), and it implied a
256-token prefill running 3.8x *slower* on two GPUs than on one, which is physically impossible. Re-measured
with `num-prompts: 8` at the same shape (`examples/profile/vllm/qwen3_8b_tp2_reqs1.yaml`),
`requests = 1` prefill went from 46.98% to 3.98% MRE with the prediction unchanged, and 256 tokens
now measures 18.26 ms against 25.47 ms at TP=1, a sane 1.4x speedup. **Rule: never quote a figure
from a `num-prompts: 1` point.**

Decode's remaining gap is a different problem and a **real model error**: `requests 1-4`
over-predicts (median ratio 1.31, against 1.1312 at TP=1, so it is TP-independent). The mechanism
is now identified, and it is *not* a missing overhead term. The model predicts too much, so adding a term
would make it worse.

Decomposing a `reqs=1` step at depth 5120: GEMM 3.123 ms + attention 2.490 ms + comm 1.268 ms =
6.881 ms predicted, against 5.240 ms measured. GEMM is 3.123 ms at one request and 3.180 ms at
sixteen, so flat, because a decode GEMM is memory-bound and reads the whole weight matrix either way,
which the per-shape numbers confirm. So the prediction carries ~4.4 ms of batch-independent
GEMM+comm while the entire real step is 5.240 ms.

A composed decode step sums **253 kernel latencies** (145 GEMM, 72 all-reduce, 36 attention), each
timed as an isolated call and each carrying its own fitted floor (`all_reduce` 6.0 µs,
`generation_attention` 22.0 µs). The excess is 1.641 ms over 253 launches = **6.5 µs per launch**,
essentially the all-reduce floor. The real server replays the step inside a **CUDA graph**, where
those per-call costs largely disappear. This is the same trap the elementwise table documents from the
other side. At `reqs=64` the same absolute excess is under 3% of a 23 ms step, which is why the
error vanishes with batch.

**Decision (2026-09-03): accepted as-is, not scheduled for a fix.** `requests >= 5` scores 2.96%
and a real RL rollout runs batched, so the affected region is outside the workload this simulator is
built for. The two candidate fixes were both judged not worth their risk. Charging each op's floor
once per step rather than once per kernel would introduce a new fitted graph-replay coefficient,
and a term added to absorb a residual rather than to represent a mechanism is exactly the failure
mode {doc}`../profiling/methodology` warns about. Modelling compute and communication overlap
explicitly is the larger and more principled change, and it remains on the deferred list for when
topology-aware communication is built.

Should it ever be revisited, one experiment distinguishes the two causes: capture a CUDA graph over
one layer's kernel sequence at `reqs=1` and compare its replay against the sum of the same kernels
timed individually. A gap near `n_launches * floor` implicates the per-call floors, while a gap near the
collective's 1.268 ms implicates overlap.

Two approximations remain unquantified at TP>1. The collective is timed in isolation, so any
overlap with compute the server achieves is not represented, though this is *not* the dominant
term at TP=2: every phase under-predicts (signed error -2.31% decode, -2.63% prefill), where an
unmodelled overlap saving worth the collective's 13.2% share would have produced over-prediction
near +13%. And TP=4/8 have per-kernel collective tables but no end-to-end baseline.

**Cross-node collectives are measured, at roughly ±15% rather than ±5%.** Resolved
2026-10-04: `nodes` is a slice column alongside `world_size`, `GpuSpec` carries a separate
`inter_node_bandwidth_bytes_per_s` (5.0e10, one 400 Gb/s bond per GPU, against NVLink's
4.5e11), and the sweep launches over real nodes via ssh instead of `127.0.0.1`. Measured at
`(16, 2)` and `(24, 3)` on 24 H20s.

What remains a limitation is the accuracy, not the coverage. The cross-node tables score
9.9-17.3% held-out MRE where the intra-node ones score 3.2-10.2%, with utility as low as
0.004. Non-monotonic step counts are no worse cross-node than intra-node (2-3 of 17 against
3-4 of 17), which rules out contention and makes this a real property of the surface: the
flat-ring SOL is a poor description of a hierarchical collective. `nodes` being a slice
column is what confines that to the cross-node rows. Price cross-node DP reduction at
roughly ±15% and say so; densifying the grid is the lever if better is needed.

Two further gaps. Cross-node rows carry **no `vllm` column**, because vLLM disables its
custom all-reduce for a group spanning nodes (`custom_all_reduce.py:90`) and falls through
to pynccl, which the control table already measures; the column is left absent rather than
filled with a relabelled NCCL row. And NUMA placement within a node is still unmodelled:
GPUs 0-3 and 4-7 sit on different NUMA domains with different NIC affinity, which `nodes`
does not capture, though all-to-all NVLink makes the GPU-to-GPU path uniform regardless.

**`ComposedRollout` decode over-predicted ~13% at small batch.**
At `requests ∈ [1, 4]` the median decode ratio is 1.1312 against 0.996-1.014 at larger
batches. Three candidate causes were tested and ruled out: CUDA graph capture, the validity of
summing per-kernel latencies, and interpolation error at `m=1`. The remaining candidate is
kernel overlap in the real server, which per-kernel timing cannot observe. No fix is currently
scoped, and recorded as a known limitation.

**`ComposedRollout` no CPU overhead terms.**
A per-request scheduler and sampler overhead was fitted at approximately 16 µs per step and
changed decode MRE by about 0.1 percentage points. It was omitted as negligible for the current
baseline but will matter for mixed batches (where the scheduler runs more frequently) and for
pipeline parallelism. Fix: fit a per-step CPU overhead term from real vLLM traces.

**`ComposedRollout` cross-hardware transfer is unvalidated.**
The SOL/utility factorisation is designed to enable hardware transfer by capturing kernel
efficiency relative to hardware ceiling. Only one GPU type (H20) was available during
development, so this property is unvalidated by construction. Do not claim accuracy on an
unmeasured GPU type. Fix: run the kernel profiling sweep on a second GPU (e.g. A100 or H100)
and compare end-to-end predictions against a vLLM server on that hardware.

**`context_attention/flashinfer` above the 5% holdout gate.**
`context_attention/flashinfer` measures 9.87% MRE at `--fraction 1.0`, against 5.74% for
`context_attention/vllm`. The cause is grid density below `seq_len=1024`, not a floor misfit.
This matters only when simulating a FlashInfer-first server (Blackwell hardware or explicit
`--attention-backend flashinfer`). The vLLM-selected backend on H20 is `FLASH_ATTN` (FA3),
which passes at 5.74%. See {doc}`../profiling/kernels` for the full holdout table. Fix: add
denser `context_seq_lens` grid points below 1024 to the profiling config.

---

## Operational rough edges

These are operational gaps that do not cause wrong predictions but require manual workarounds.

**`policy=psrl` requires a machine-specific path.**
`psrl` is the **default policy**. The implementation (`dryrun/policy/psrl.py`) falls back to
a hard-coded source-tree path when the `psrl` package is not already importable. That path is
machine-specific and will not resolve on a different machine. If the fallback also fails, the
constructor raises `ImportError` with an install hint. Install `psrl` into your environment
with `pip install --no-deps -e /path/to/psrl` to remove the dependency on the hard-coded path.
Link: {doc}`../simulation/policies`.

**FSDP OOM can surface as a NCCL allocation failure.**
When static memory occupies a large fraction of device capacity, CUDA OOM manifests as a NCCL
`calloc` failure rather than a `torch.OutOfMemoryError`. The profiler's OOM classifier reads
a marker file rather than the log, so correctly-written candidates are classified correctly.
A candidate that crashes before writing the marker is classified as `failed` rather than `oom`,
dropping a memory lower bound from the fit. This affects the shard=2 cell in particular, where
static occupancy is highest. See {doc}`../profiling/fsdp`.

**Prefix reuse is derived from provenance, so it under-reports hits in one direction.**
Block identity comes from `(source_group_id, block_index)` for prompt-interior blocks and
`(rid, block_index)` otherwise, rather than from a hash of token ids the simulator never holds.
Two *different* prompts that happen to share leading full blocks hit in the real engine and miss
here. On RL workloads distinct problems almost certainly differ at token 0, and every weight
update clears the hash map anyway, so the effect is second order -- but it is one-directional:
this never over-reports a hit. A workload with a shared system prompt or shared few-shot prefix
would amplify it, and the fix at that point is an optional explicit prefix-grouping key on the
trace, not a change to the hash rule. See {doc}`../simulation/prefix-cache`.

This replaced an earlier and larger inaccuracy, recorded here because the comparison is the
useful part: prefill used to be charged `n` times for a prompt with `n` sampled responses, once
per response rather than once per shared prompt. At DAPO's `rollout_n=8` that was 8x the real
engine's prompt prefill.

**Within-group reuse cannot be judged on the aligned baseline.**
`extract_workload_trace` chunks completed requests into groups of `group_size` in request-id
order, because the route trace does not record which prompt each response belonged to
(`psrl_log.py`, `write_workload_trace`). That reproduces the length distribution and the group
arity exactly but not the grouping, so a hit rate measured there reflects fabricated groups.
This is why prefix caching is validated against a purpose-built vLLM experiment instead; see
{doc}`../simulation/prefix-cache`.

**No cross-instance reuse, and the router scatters groups.**
The block pool is per-instance, as in the real engine, while `_select_instance` picks the
least-active one, so a group's siblings land on different instances and within-group reuse is
diluted. Frontier makes this a hard gate: multi-replica prefix caching requires a sticky
scheduler or it raises. We do not port that gate, because what PSRL actually does has not been
verified and copying Frontier's policy would weld in an unverified assumption. Routing is
tracked separately.

---

## Deferred work and acceptance criteria

The following items are from `docs/design/profiling-and-cost-models.md`, section "Deferred
stages and acceptance criteria". Each item's acceptance criteria are quoted in full because they
define what a completed implementation looks like.

1. **Topology-aware communication and overlap.** Calibrate TP, PP, and DP collectives per
   node topology, model exposed overlap explicitly, and pass grouped-topology holdout
   thresholds before enabling search.

2. **Advanced pipeline schedules.** Add VPP, interleaved 1F1B, zero-bubble schedules, and
   measured stage imbalance. Validate predicted bubbles against per-stage timers.

3. **CP, SP, EP, and MoE.** Represent dense and expert meshes separately rather than
   multiplying dimensions blindly. Candidate generation must prove divisibility and rank-mesh
   constraints before launch. Until this item is complete, MoE and MLA architectures are
   refused by the cost model rather than approximated. See {doc}`../development/model-families`.
   Ulysses sequence parallelism (`sp>1`) is rejected at config time as out of scope.

4. **Recompute granularity.** Profile full, selective, uniform, and block-level policies.
   Memory and latency must share the same resolved recompute descriptor.

5. **Variable-length packing and heterogeneous pipelines.** Record actual packed token
   layouts and stage assignments. Validate equal-token workloads with different quadratic
   attention work.

6. **Allocator fragmentation, CPU offload, and activation offload.** Predict reserved as well
   as allocated memory and bound OOM false negatives on held-out candidates.

7. **Cross-model DistServe calibration.** Fit C3 and C4 jointly across architectures and
   report condition number and confidence intervals.

8. **Automatic disaggregated GPU allocation.** Enable only after candidate-domain checks and
   held-out error gates are part of the search objective. Search must reject unsupported cells
   instead of silently extrapolating.
