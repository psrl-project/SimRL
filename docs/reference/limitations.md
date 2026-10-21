# Limitations

What the code does not do today. Each item below was checked against the code, and the file or
symbol that decides it is named. Items that are accuracy figures live on {doc}`/reference/validation`.

---

## Scope

- SimRL models time and memory. It does not simulate correctness, rewards or learning dynamics.
- A training batch is whole prompt groups. `take_complete_groups` and every preset select complete
  groups, so a policy that trains on part of a group (TRL drops individual stale samples in the real
  system) is approximated by dropping the whole group.
- The rollout fleet is homogeneous. `Simulator` builds `rollout.n_instances` identical engines
  against one rollout cost model. `srl/simulator/dynamic_parallel/` (`FleetPlan`,
  `DynamicParallelFleet`) provides primitives for re-shaping a fleet at a weight update, but
  nothing in `Simulator` constructs them.
- There is no network topology model. Collective cost comes from measured tables at the world
  sizes and node counts that were swept. Weight sync is a configured constant (see below).
- Only H20 hardware is described. `srl/cost/kernel/data/gpu_specs.yaml` holds one entry, and
  cross-hardware transfer of the kernel tables (SOL on the target GPU, utility from the measured
  one) is possible by construction but has never been validated.

## Dependencies and installation

**`psrl` is a hard dependency of the whole simulator.** `srl.simulator`, `srl.forecast`,
`srl.policy.registry` and therefore `srl.cli.main` import `psrl.workers.ps.staleness` at import
time, whichever policy is selected. It is not listed in `pyproject.toml`. Without it, importing
`srl.simulator` raises `ModuleNotFoundError`.

**The C++ engine is a prebuilt binary for one interpreter.** `srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so`
is committed. The default `SIMRL_BACKEND=cpp` raises `ImportError` on another CPython version or
platform until the module is rebuilt from `srl/component/rollout/engine_cpp/` with CMake and
pybind11. Set `SIMRL_BACKEND=auto` to fall back to the Python engine instead, or `native` to use it
outright. `SIMRL_BACKEND` accepts `cpp`, `auto` and `native`. Any other value raises `ValueError`.

**Profiling needs the real stack.** `srl-profile` drives vLLM, Megatron-Bridge or torch and needs
GPUs. Simulation and `srl-regress` do not.

## Configuration

- `ComposedTraining` is not selectable. `build_training_cost` in `srl/cli/main.py` builds `fixed`,
  `analytical`, `linear`, `hydraulis` and `fsdp`. `ComposedTraining` is reachable from Python and
  from `score_composed_training`.
- `recompute_cost.hidden_size`, `vocab_size` and `parameter_count` appear in `simulate.yaml` and are
  never read. The `training_forward` geometry comes from the training artifact or the checkpoint.
- Only the `policy` group is type-checked against its dataclass. A value of the wrong type in
  another group is not rejected at composition.

## Rollout cost models

- `ComposedRollout` handles tensor parallelism only. `pp`, `dp` and `cp` in `rollout_cost.parallelism`
  are not used by it, and `tensor_parallel` must equal the degree the `ModelOps` were sharded for.
- A mixed step (prefill and decode tokens together) is priced as context attention plus
  generation attention at the step's batch size, because only the two pure phases are measured. The
  assumption suits near-uniform batches. A chunked-prefill step that spreads one long chunk across a
  large decode batch understates the quadratic context term.
- The model sums per-kernel latencies and has no per-step CPU scheduling or sampling overhead term.
- Overlap of communication with compute is not modelled. Each collective is timed in isolation.
- Tensor-parallel degrees above 2 have collective tables but no end-to-end baseline against a real
  server in the validation set.
- Collectives are priced by `(world_size, nodes)` slices. Cross-node tables score below the
  intra-node ones (see {doc}`/reference/validation`), because a flat-ring SOL describes a hierarchical
  collective poorly. Cross-node rows exist for NCCL only, since vLLM's custom all-reduce is disabled
  across nodes. NUMA placement inside a node is not represented.
- Hybrid attention models (softmax plus gated delta net layers) and routed mixture-of-experts models
  are priced for the `Qwen3_5Moe*` architectures. Decode is the phase that is priced well. The
  prefill path charges the gated-delta-net dispatch floor on every linear-attention layer and
  over-predicts for these models. The dense path does not have this problem.
- The closed-form decode fast path (`saturation_step_threshold`) is off by default because it
  trades accuracy for little end-to-end speed.
- The FlashInfer decode kernel compiles only some GQA group sizes (1, 2, 3, 4 and 8). A model whose
  query to KV head ratio is outside that set, such as 40 to 8 heads, gets a FlashInfer context table
  and no generation table. Tensor parallelism does not change the ratio. The vLLM backend is
  unaffected.

## Training cost models

- Pipeline schedules other than `1f1b` are not represented (the only values the payload accepts are
  `1f1b` and `fsdp`). Interleaved, zero-bubble and virtual pipeline schedules are not modelled.
- `ComposedTraining` raises `NotImplementedError` for `dp > 1` (`TrainingOps.dp_comm_queries`).
  Its validation set is `dp=1` only, so a data-parallel gradient collective term could be written
  but not scored.
- `ComposedTraining` under-predicts every record of its validation set (see {doc}`/reference/validation`), so it
  is missing at least one additive term. The two residual adds that Megatron fuses into
  `bias_dropout_add_fused_train` are not charged. A ragged microbatch plan is priced from a mean slot.
  The count of tensor-staging launches per layer is traced from one framework version and does not
  transfer to another, so it is a single named constant (`PLUMBING_LAUNCHES_PER_LAYER`).
- The fitted Megatron and FSDP models answer only for cells that were profiled. A discrete selector
  without a matching entry (an `optimizer`, `recompute`, dtype, `optimizer_offload` or
  `reshard_after_forward` that no candidate used) raises `KeyError`. FSDP supports
  `sequence_parallel=1` only.
- A fitted feature range is a hard limit unless the feature is listed in `training_cost.extrapolate`.
  Extrapolation is recorded in the `train` stream and is as accurate as the form is linear in that
  feature. The latency form is linear in token count plus a quadratic term, so a stretched tail is
  understated if the sweep did not reach it.
- Data-parallel scaling is ideal. Both fitted models record `dp_communication: ignored` and
  `dp_latency: ideal-local-workload-scaling` in their artifacts, so a data-parallel degree that was
  not profiled divides the workload per rank and adds no gradient reduction cost. For FSDP the
  profiled mesh cell carries its own sharding communication.
- Activation recompute is a discrete selector (for example `full`, `none`). Selective, uniform and
  block-level granularities are not distinguished.
- Memory predictions are allocated peaks. Allocator fragmentation, CPU offload of optimizer state
  beyond the profiled `optimizer_offload` cell, and activation offload are not predicted.

## Profiling

- A candidate that dies before it writes its OOM marker file is classified `failed`, not `oom`, and
  contributes no lower bound to the memory fit. OOM is read from per-rank marker files and never from
  log text, so an out-of-memory condition that surfaces as another error, such as an NCCL
  allocation failure, is lost.
- `verify-shapes` supports `tensor_parallel=1` only, runs by hand, and compares against the
  HuggingFace module graph with a hard-coded fusion map. It cannot detect that the serving engine
  stopped fusing a projection.
- Each model family must be registered. An architecture with no family, and any mixture-of-experts or
  compressed-KV architecture the dense family does not cover, is refused with a message naming the
  missing kernel. See {doc}`/development/model-families`.
- Multi-node Megatron profiling needs a per-host launch with `--node-rank`. The tool does not start
  the other hosts.

## Simulation fidelity

- **Prefix reuse is derived from provenance.** Block identity is `(source_group_id, block_index)`
  for prompt-interior blocks and `(rid, block_index)` otherwise, because the simulator holds no
  token ids. Two different prompts that share leading full blocks hit in a real engine and miss
  here. The error is one-directional and never reports a hit that would not occur. A workload with
  a shared system prompt would amplify it.
- **No cross-instance cache reuse.** Each instance owns its block pool. A group's responses share
  prefix blocks only on the instance that holds them, so `rollout.router.enable_group_sticky`
  changes the hit rate.
- **A trace extracted from a PSRL run has fabricated grouping.** `extract_workload_trace` chunks
  completed requests into groups of `group_size` in request-id order, because the route trace does
  not record which prompt a response belonged to. The length distribution and group arity are exact.
  Which responses share a prompt is not, so a prefix hit rate measured on that trace is not
  meaningful.
- **`kv_blocks` is a placeholder unless sized.** Without `rollout.kv_cache_sizing.model_path`, the
  literal `100000` is the denominator of every reported `kv_cache_usage`. The `probe: true` path
  needs vLLM and a GPU the first time, and `probe: false` is only reliable for dense models because
  the engine's measured peak activation, non-torch memory and hybrid page alignment cannot be
  derived from `config.json`. A capacity does not transfer between TP degrees or GPU models.
- **Length prediction is an oracle.** `inventory_deadline` and the forecast modules require
  `policy=psrl` and use `oracle` or `noisy_oracle` lengths, which read the trace's `target_len`. No
  predictor that estimates lengths from prompts is built in. The strategy also requires the native
  or `auto` engine backend, because it reads Python request state the C++ engine does not carry.
- **Livelock detection is a guard.** The run returns `livelocked=True` after `rollout.livelock_rounds`
  idle rounds. A configuration that the guard ends early produces a result that is not comparable to a completed run.
- **SCG is a rate.** `SimResult.scg` divides tokens by simulated time, so two runs are comparable
  only if they consumed the same trajectories. Compare the printed `Consumed:` line first.

## Deferred

Not implemented, with the condition that would justify building each:

1. Topology-aware communication with explicit overlap.
2. Ulysses sequence parallelism in the training cost (`sequence_parallel > 1` is refused by the FSDP
   search space and the FSDP fit).
3. Per-sample, non-group-atomic training batches.
4. A second GPU type, to test hardware transfer.
5. Cross-model DistServe calibration of `C3` and `C4`.
6. Automatic GPU allocation between rollout and training that rejects unsupported cells instead of
   extrapolating.
