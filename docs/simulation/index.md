# Simulation

The simulator replays an asynchronous RL loop on a single global clock. Rollout engines
generate trajectories, a staleness policy decides what may be admitted and consumed, a
training step is priced by a cost model while rollout keeps running, and the new weights
are published at the end of the step. No GPU is used: every duration comes from a cost
model, and every scheduling decision is made by the same Python components that define the
system being studied.

This section covers how a run is wired, the per-system policy presets, routing, the KV
cache, the guards that stop a cost model from answering outside its fitted range, the two
engine backends, and what a run writes.

## The loop

```{mermaid}
flowchart LR
    W[Workload groups] -->|stage| R[Router queue]
    R -->|policy.admit_quota| E[Engine instances]
    E -->|group complete| P{policy.on_group_complete}
    P -->|KEEP| B[ready groups]
    P -->|DROP| X[dropped]
    B -->|policy.take_batch| T[Training step]
    T -->|deadline| U[Publish]
    U -->|policy.on_publish| C[Cancels]
    U -->|pause mode| A[Abort or continue in place]
    U -->|credits| W
```

One iteration of `Simulator.run` does the following.

1. **Stage.** `_stage` turns ingress credits into whole prompt groups and enqueues them in
   the router. Staging does not admit anything.
2. **Admit.** Router dispatch places a group's first response on an instance only if
   `policy.admit_quota(state)` covers a whole group. That placement commits the admission
   of the entire group (`on_admit`, ledger entry, `v_traj` stamped). Siblings may still
   wait in the router.
3. **Generate.** `_run_engine` advances every instance in global time order and collects
   finished responses. A group is released to the policy only when all of its responses
   are done (`on_group_complete` returns `KEEP` or `DROP`).
4. **Consume.** When `policy.can_sample` is open and `take_batch` returns
   `batch_size` complete groups, the step starts.
5. **Train.** The step lasts `train + sync + recompute` seconds. Rollout keeps generating
   until that deadline, then speculative events past the deadline are undone and all
   instances are aligned to it.
6. **Publish.** `train_version` advances, then `policy.on_publish` returns cancels for
   live groups the new version makes worthless, `rollout_version_after_train` picks the
   version the engines serve, and `policy.pause_mode` decides whether in-flight work is
   aborted and re-enqueued (KV lost, re-prefilled) or continues in place. If the served
   version changed, `sync_cost.weight_load_time` is charged. `job.batch_size` new ingress
   credits are granted and the next batch is staged.

The run ends after `job.n_versions` training steps, or earlier with
`SimResult.livelocked` set when nothing changes for more than `rollout.livelock_rounds`
rounds.

## Running one

```bash
source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh
srl policy=verl workload=normal rollout_cost=roofline training_cost=fixed \
    job.n_versions=20 job.batch_size=64 rollout.n_instances=4 output_dir=/tmp/run
```

`srl` is a Hydra application, so every key is an override. `srl --help` prints the config
groups and the resolved defaults, and `srl --cfg job` prints the composed config. The full
key list is in {doc}`../reference/configuration`. The example scripts under
`examples/simulate/` wrap the same entry point: `simulate.sh` is the general launcher and
`psrl_aligned.sh` transcribes one measured PSRL run.

### Config groups

| Group | Options | Defined in |
|---|---|---|
| `policy` | `psrl`, `areal`, `roll`, `slime`, `verl`, `verl_v1`, `skyrl`, `openrlhf`, `dressage`, `nemo_rl`, `trl`, `prime_rl` | typed presets in `srl/config/policies/`, defaults in `srl/config/defaults/policy/` |
| `rollout_cost` | `roofline` (default), `psrl`, `distserve`, `composed` | `srl/config/defaults/rollout_cost/` |
| `training_cost` | `fixed` (default), `linear`, `analytical`, `hydraulis`, `fsdp` | `srl/config/defaults/training_cost/` |
| `workload` | `uniform` (default), `normal`, `lognormal`, `bimodal`, `powerlaw`, `dapo_trace` | `srl/config/defaults/workload/` |

`sync_cost` and `recompute_cost` are plain blocks in `simulate.yaml`, not groups. The
structured schema for everything lives in `srl/config/schema.py`.

The default policy is `psrl`. The cost models are documented in {doc}`../cost-models/index`
and the workload sources in {doc}`../workload/index`.

### Top-level blocks

| Block | Purpose |
|---|---|
| `job` | `n_versions`, `batch_size` (groups per step), `group_size` (null reads it from the workload), `max_prompt_length`, `max_output_length` |
| `rollout` | `n_instances`, `token_budget`, `kv_blocks` or `kv_cache_sizing`, `block_size`, `enable_prefix_caching`, `max_concurrency`, `max_inflight`, `admission_control`, `router`, `livelock_rounds` |
| `sync_cost`, `recompute_cost` | the two small cost models described below |
| `output_dir`, `log_telemetry`, `log_level` | where and whether to write, see {doc}`outputs` |

## Things that most often surprise people

1. **`job.batch_size` counts prompt groups, not responses.** A step consumes
   `batch_size * group_size` responses, and a batch is always whole groups because GRPO
   scores a response against its own group. `group_size` comes from the workload. An
   explicit `job.group_size` that disagrees with the workload raises.
2. **The staleness bound is a property of the policy preset, not of `job`.** There is no
   `job.max_staleness`. Each preset exposes its own knob (`psrl.staleness`,
   `areal.max_head_offpolicyness`, `trl.max_staleness`, ...) with a different meaning, and
   some systems have no version bound at all. See {doc}`policies`.
3. **`psrl` must be importable.** `srl.simulator` and `srl.forecast` import
   `psrl.workers.ps.staleness`, so the separate `psrl` package has to be installed even
   when another policy is selected. The editable install must use strict mode:
   `pip install --no-deps -e <psrl> --config-settings editable_mode=strict`.
4. **Length caps clip the pair, not each axis.** The prompt is clipped to
   `max_prompt_length`. The output is clipped to
   `max_prompt_length + max_output_length - prompt_len`, so a short prompt buys extra
   output room. Training sees per-axis caps, so rollout can generate tokens that training
   never charges. See `clip_total_length` in `srl/simulator/config.py`.
5. **A cost model refuses out-of-domain queries** instead of extrapolating. This is the
   most common first-run failure. See {doc}`domain-gates`.
6. **The C++ engine runs by default (`SIMRL_BACKEND=cpp`).** Some features only work on the
   Python engine. Those are marked `python_engine_only` and raise under `cpp` rather than
   silently switching. See {doc}`backends`.

## Sync and recompute costs

Two small cost models are added to every training step.

`sync_cost`, weight synchronisation to the rollout fleet
: `name: fixed` charges `sync_time` seconds, and only when the policy's
  `will_sync_after_train` is true (slime skips it between pushes). `name: bandwidth`
  exists in `build_sync_cost`, but the simulator calls it as
  `sync_time(0, n_instances, 0)`, so it divides by a zero bandwidth. Use `fixed`.
  `sync_cost.weight_load_time` is separate: the rollout-side pause charged at publish when
  the served weight version changed.

`recompute_cost`, the log-prob forward pass over the consumed batch
: `fixed` charges `time_per_token * tokens / dp`. `analytical` derives it from
  `model_params` and an achieved FLOP rate. `training_forward` prices it as the forward
  share of the fitted training step (2 of 6, 7 or 8 FLOPs per parameter-token depending on
  `training_cost.activation_recompute`, plus an unembedding correction) and reads the
  geometry from the training artifact, or from the checkpoint named by
  `rollout_cost.model_path` or `rollout.kv_cache_sizing.model_path` when the training model
  has no descriptor. The default `fixed` with `time_per_token: 0.0` charges nothing.

## Environment variables

| Variable | Read in | Meaning |
|---|---|---|
| `SIMRL_BACKEND` | `srl/cli/main.py` | `cpp` (default), `auto`, `native`. See {doc}`backends` |
| `SIMRL_UNDOABLE_EVENTS` | `srl/cli/main.py` | comma list of engine events that get speculative undo, default `request_complete,preempt,iteration_done`. See {doc}`backends` |

Hydra's `+log_exclude='instance,rollout_step'` drops telemetry streams at the source, see
{doc}`outputs`.

```{toctree}
:maxdepth: 2

policies
routing
prefix-cache
domain-gates
backends
outputs
```
