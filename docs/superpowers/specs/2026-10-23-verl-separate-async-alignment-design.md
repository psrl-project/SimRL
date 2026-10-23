# verl separate_async alignment: design

## Goal

PSRL has a real reference run (`psrl/examples/dapo_trainer/qwen3_8b_megatron_simulate.sh`) and a
matching SimRL simulation (`examples/simulate/psrl_aligned.sh`) that is already validated against
it. This design builds the equivalent pair for verl's own disaggregated async trainer mode
(`trainer.v1.trainer_mode=separate_async`): a real verl run at the same config, a simulation
aligned against it, and the plots that put them side by side. Once both PSRL and verl each have a
validated simulation, the user can compare verl-vs-PSRL performance cheaply through the simulator
instead of running every configuration for real. That comparison itself is out of scope here — this
spec only covers building and validating the verl half.

The work is one dependent pipeline, not independent subsystems: Phase 2 needs to know what Phase 1
actually produces, Phase 3 needs Phase 2's log format, Phase 4 needs Phase 3's artifacts. All four
are designed together here because the user asked for that, but each phase's exact parameters may
still move once the prior phase's real output is in hand — that is called out per phase below.

## Phase 1 — Real verl `separate_async` run matching PSRL's reference config

**Repo.** `/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new`, a clean clone of
`https://github.com/verl-project/verl` (independent of PSRL's own vendored `third_party/verl`
fork). The launch script records the exact commit hash the clone landed on, the same way
`psrl_aligned.sh` documents the PSRL commit it matches.

**Env.** A new conda env built from `verl-new`'s own `requirements.txt`/install scripts, independent
of the existing `verl-exp` env (which pins an older verl). A new `env/verl_new.sh`, alongside the
existing `env/psrl.sh` and `env/verl.sh`. If multi-GPU collectives misbehave on this cluster, port
over the NCCL/UCX tuning from `env/verl.sh` as a fallback — not applied up front.

**Launch script.** A new `verl-new/examples/psrl_compare/qwen3_8b_separate_async_megatron.sh`,
adapted from `verl-new/tests/special_e2e/run_v1_separate_async.sh`'s Megatron branch (confirmed to
already support Megatron actor strategy + `separate_async` together). Config, transcribed from
`psrl/examples/dapo_trainer/qwen3_8b_megatron_simulate.sh`:

- Resource split: `trainer.n_gpus_per_node=6` (train pool, Megatron TP=2 × PP=3) and
  `actor_rollout_ref.rollout.n_gpus_per_node=2` (rollout pool, TP=1, 2 replicas). Both `nnodes=1`.
  One 8-GPU node total, matching PSRL's 6-train/2-gen split exactly.
- `actor_rollout_ref.actor.strategy=megatron`, TP=2, PP=3.
- `data.train_batch_size=32`, `max_prompt_length=1024`, `max_response_length=10240`,
  `actor_rollout_ref.rollout.n=8`, `algorithm.adv_estimator=grpo`,
  `reward.reward_manager.name=dapo` (built in; `recipe/dapo` no longer exists upstream and is not
  needed). Same `dapo-math-17k.parquet` / `aime-2024.parquet` files PSRL uses, unmodified.
- `trainer.v1.trainer_mode=separate_async`,
  `trainer.v1.separate_async.hybrid_rollout.enable_switch=false` (no GPU lending between pools —
  matches PSRL's non-overlapping train/gen GPUs),
  `trainer.v1.separate_async.parameter_sync_step=1` (forced by the trainer's own assertion
  `train_batch_size == parameter_sync_step * ppo_mini_batch_size`, with both sides 32).
- `trainer.v1.sampler.max_off_policy_threshold=1` and
  `trainer.v1.sampler.max_off_policy_strategy=wait`, as the first approximation of PSRL's
  `psrl.staleness=1`. PSRL's own reference run shows interrupted requests resuming
  (`finish_reason="abort"`, `iteration_count=2`) rather than being dropped, so `wait` is the closer
  behavioral match to measure first. **Open parameter**: Phase 4 checks whether this actually
  produces one-step-off-policy behavior comparable to PSRL's run, and adjusts if not.
- `trainer.v1.separate_async.num_warmup_batches` left at verl's own default (`1`) — PSRL's reference
  script has no explicit warmup knob, so this is the closest match available rather than a
  transcribed value. Recorded here because Phase 3's `verl_aligned.sh` reads this same value back
  out of Phase 1's launch script.
- `trainer.total_training_steps=5`, `trainer.val_before_train=False`, `trainer.save_freq=-1`,
  matching PSRL's reference run length so sample counts and run duration are comparable.
- Qwen3-8B HF checkpoint converted to a Megatron dist-checkpoint using verl-new's own converter
  (exact script name to be confirmed during implementation — PSRL's fork uses
  `scripts/convert_hf_to_mcore.py` via MBridge; verl-new ships an analogous tool).

**Cluster.** Single node, 8 free H20 GPUs, confirmed free immediately before launch (`ray status` +
`nvidia-smi`, not assumed from this design doc). A fresh, single-node Ray cluster started there via
`psrl/examples/ray/ray_start.sh <hostfile-with-just-that-node>`, sourcing `env/verl_new.sh` — kept
separate from PSRL's own Ray cluster and from the old `verl-exp` env's cluster.

## Phase 2 — Instrument verl for telemetry

Verl already computes everything needed in memory — vLLM's own scheduler stats, each request's
`stop_reason`/`num_preempted` — but persists none of it to disk. Three small, additive patches,
committed to a dedicated `telemetry-logging` git branch on top of the pinned upstream commit (a
plain branch, not PSRL's `patch/<lib>/<commit>.patch` convention — that convention exists for
dependencies that get re-vendored across upgrades and need their patch reapplied each time;
`verl-new` is a standalone top-level clone, so a branch is the proportionate equivalent and rebases
cleanly onto a newer upstream commit later).

1. **Per-request completion trace** (the one genuinely missing piece). Patch
   `verl/workers/rollout/llm_server.py` (around where `final_output`/`stop_reason`/`num_preempted`
   are finalized, roughly lines 280-360) to append one JSONL line per completed request:
   `{request_id, prompt_tokens, response_tokens, stop_reason, num_preempted, ts}`. Direct analog of
   PSRL's `route_trace.log` `pop` events, already JSON instead of a custom text format.
2. **Per-engine periodic occupancy.** Near `llm_server.py:507`, where vLLM's own stats are already
   gated behind `disable_log_stats`/`prometheus.enable`, add a timer that appends
   `{ts, num_running_reqs, num_waiting_reqs, kv_cache_usage}` per engine replica to a
   `stats_r{r}.jsonl`. Field names match PSRL's own `stats_r*_dp*.jsonl` schema exactly, so the
   SimRL-side converter logic (Phase 3) can mirror `psrl_log.py` almost line for line.
3. **Training-phase timing.** In `verl/trainer/ppo/v1/trainer_separate_async.py` and
   `trainer_base.py`, add explicit begin/end timestamp markers (or a small JSONL writer) around the
   log-prob recompute forward pass, the optimizer step, and the parameter sync / weight push to the
   rollout replicas — mirroring the shape of PSRL's four markers
   (`[SWITCH] switch_to_trainer_mode` / `[Begin Event] TRAIN` / `[End Event] TRAIN` /
   `[End Event] PUSH`), though the literal regexes will differ.

All three are logging only, with no effect on training behavior, gated behind an env var
(`VERL_TELEMETRY_LOG_DIR`) so they are a no-op unless explicitly enabled.

**Open item**: the exact file layout (one file per replica vs. one shared file with a replica-id
field) is decided during implementation, once Phase 1's actual process/replica topology is visible.

## Phase 3 — SimRL-side converter and `verl_aligned.sh`

1. `srl/telemetry/verl_log.py`, structurally mirroring `srl/telemetry/psrl_log.py`:
   - `VerlLogLayout` (file globs and marker patterns for the Phase 2 logs).
   - `convert_verl_logs()`, writing into the same `JsonlStore` streams `SimTelemetry` already
     defines (`instance/{id}`, `train`), so `SimPlotter` needs no verl-specific knowledge. Whether a
     `gateway` stream applies is decided once the real run's logs are in hand — `separate_async`'s
     `LLMServerManager` may not have a router-queue-depth equivalent to PSRL's `smg` gateway.
   - `extract_workload_trace()`, reading the new per-request JSONL — simpler than PSRL's version
     since the source is already structured JSON rather than a custom text log.
2. `srl/cli/verl_plot.py` and `examples/visualize/plot_verl_run.sh`, mirroring the `psrl_plot` pair.
3. `examples/simulate/verl_aligned.sh`, mirroring `psrl_aligned.sh`'s structure (read the real run's
   logs, extract the trace, set every knob to match), with:
   - `POLICY=verl_v1` — not the legacy `verl` policy. (`psrl_aligned.sh` currently defaults to
     `POLICY=verl`, which does not model `separate_async`; `verl_v1` is the policy that does, and
     this default should be corrected there too as part of this change.)
   - `TRAINING_BACKEND=hydraulis`, reusing the existing
     `outputs/cost/qwen3_8b_tp2_pp3_thd/training_hydraulis.json` cost artifact unchanged — the mesh
     is identical to PSRL's, so no new profiling sweep is needed.
   - Phase 1's `num_warmup_batches`, `parameter_sync_step`, `max_off_policy_threshold`,
     `max_off_policy_strategy` passed straight into the `verl_v1` policy's constructor via
     `EXTRA_OVERRIDES`.
4. `examples/visualize/plot_simulate_verl_run.sh`, mirroring `plot_simulate_run.sh`, to render the
   simulated-verl figure.

## Phase 4 — Compare and iterate

1. Run the existing `srl-compare` CLI against the real verl run's converted logs and the
   `verl_aligned` simulation's `output_dir`, producing the same class of agreement metrics
   `psrl_aligned.sh`'s header already reports for PSRL (train/step time, peak and mean KV
   utilization, mean running requests, span).
2. Place both instance-load figures in `examples/visualize/` (e.g.
   `instance_load_verl_separate_async.png` for the real run, `instance_load_verl_simulated.png` for
   the simulation), following the existing naming convention.
3. Document the measured agreement and any open deltas in `verl_aligned.sh`'s own header, in the
   same style `psrl_aligned.sh` already uses for itself.
4. Iterate on whichever `verl_v1` policy parameters do not match (most likely candidate:
   `max_off_policy_threshold`/`max_off_policy_strategy`, flagged as open in Phase 1).

**Explicitly out of scope**: comparing verl's real performance against PSRL's real performance.
That is the next piece of work this unlocks, once both simulations are independently validated
against their own real systems.

## Risks and open items carried into implementation

- Megatron + `separate_async` is confirmed to exist as a supported config path
  (`tests/special_e2e/run_v1_separate_async.sh`), but has not been run end to end here. Phase 1 is
  explicitly the step that finds out whether it works as documented.
- The staleness-parameter mapping (`max_off_policy_threshold`/`strategy` ↔ PSRL's `staleness=1`) is
  an approximation, not a guaranteed exact semantic match. Phase 4 measures and corrects it.
- Building verl-new's env purely from its own requirements (rather than cloning the proven
  `verl-exp` env) risks CUDA/NCCL/driver mismatches this cluster's other envs work around with
  extensive tuning. Addressed reactively in Phase 1 if it surfaces, not pre-emptively.
