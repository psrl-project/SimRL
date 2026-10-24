# verl separate_async alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run verl's `trainer.v1.trainer_mode=separate_async` at PSRL's reference config, instrument
it for telemetry, and build a `verl_aligned.sh` SimRL simulation validated against that real run —
mirroring the existing `psrl_aligned.sh` pair.

**Architecture:** A real run in a freshly cloned `verl-new` (Phase 1), additive telemetry patches on
a dedicated branch gated by an env var so they are a no-op by default (Phase 2), a SimRL-side
converter and simulation mirroring the existing `psrl_log.py`/`psrl_plot.py`/`psrl_aligned.sh` family
but simplified because verl's telemetry is already structured JSON with real timestamps, not a
two-clock text log (Phase 3), and a final empirical comparison pass (Phase 4).

**Tech Stack:** verl (`trainer.v1`, Megatron backend, Ray), SimRL (`srl.telemetry`, `srl.cli`,
`JsonlStore`), bash launch scripts, pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-verl-separate-async-alignment-design.md`

## Global Constraints

- `verl-new` is cloned from `https://github.com/verl-project/verl`, pinned to commit
  `a4b9354968705a441fb7da7367924b15b902e771`. It is already cloned at
  `/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new`.
- The conda env for `verl-new` is created by cloning `psrl_smg_new` (the env `env/psrl.sh`
  activates, already proven on this node's driver/CUDA: `torch 2.11.0+cu128`,
  `vllm 0.22.1.dev0+...cu128`, `megatron-core 0.19.0`), then swapping only the editable `verl`
  install from PSRL's fork (`psrl/third_party/verl`) for the fresh `verl-new` checkout. An earlier
  attempt to build a fully from-scratch env via `verl-new`'s own install script failed: PyPI's
  `vllm==0.24.0` resolves to `torch+cu130`, which this node's driver (535.247.01) cannot run, and
  vLLM's v0.24.0 release ships no `cu128` wheel to substitute. Cloning the already-working env
  sidesteps that CUDA mismatch entirely; the residual risk is verl-new's HEAD-of-main Python code
  expecting slightly newer vllm/megatron APIs than this cloned stack provides — a Python-level
  compatibility question, not a CUDA/build one, and Task 2 is written to surface and resolve that
  kind of gap explicitly rather than force an upgrade.
- Any task that touches the shared Ray cluster or starts GPU work (Tasks 5 and 10 — Task 15 only
  reads Task 10's already-collected logs and runs the CPU-only simulator/plotter/`srl-compare`, so
  it is not a cluster checkpoint) is a **CHECKPOINT**: confirm `ray status` and `nvidia-smi` show the
  target node genuinely free, and get explicit user go-ahead, before running anything. Never touch a
  node the user has flagged as reserved for other work.
- All three telemetry patches in Phase 2 are gated behind the `VERL_TELEMETRY_LOG_DIR` env var —
  no behavior change to training when it is unset.
- `verl_aligned.sh` uses `POLICY=verl_v1`. `psrl_aligned.sh` is never modified by this plan.
- Field names written by the Phase 2 patches must match PSRL's own `stats_r*_dp*.jsonl` schema
  (`num_running_reqs`, `num_waiting_reqs`, `kv_cache_usage`) so Phase 3's converter can mirror
  `psrl_log.py`'s shape.
- **Target node: `28.49.55.85`.** This session's own node (`29.162.247.148`) has GPUs 0-3 held by an
  unrelated job (~92GB each, confirmed via `nvidia-smi`) and is not usable for this plan's GPU work.
  `28.49.55.85` was confirmed fully free (0 MiB / 0% on all 8 GPUs) and reachable over passwordless
  SSH, with the same shared storage (conda envs, the `verl-new` clone) visible there. Every command
  in Tasks 3, 5, and 10 that needs a GPU runs there via
  `ssh 28.49.55.85 "source .../env/verl_new.sh && <command>"`, not on the local node. Re-confirm
  `nvidia-smi` on `28.49.55.85` immediately before each such task, since cluster state can change
  between tasks.

---

## Phase 1 — Real verl `separate_async` run

### Task 1: Pin the base commit and create the conda env

**Files:**
- Create: `/apdcephfs_zwfy10_303541817/share_303541817/lhy/env/verl_new.sh`

**Interfaces:**
- Produces: a conda env named `verl-new` and an env script later tasks `source` before running
  any `verl-new` command.

- [ ] **Step 1: Record the pinned commit**

Run: `git -C /apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new rev-parse HEAD`
Expected: `a4b9354968705a441fb7da7367924b15b902e771` (already the case; this step just re-confirms
nothing has changed the checkout since cloning).

- [ ] **Step 2: Create the conda env by cloning `psrl_smg_new`**

A prior attempt created an empty `verl-new` env with `conda create -n verl-new python=3.12 -y` and
nothing else — remove it first (`conda create --clone` refuses to target a name that already
exists):

```bash
conda env remove -n verl-new -y
```

Then clone the already-working env. This copies a large conda env over fuse-mounted storage and
may take a while — let it run to completion rather than interrupting:

```bash
conda create -n verl-new --clone psrl_smg_new -y
```

Expected: environment created at `/apdcephfs_zwfy10/share_303541817/lhy/anaconda3/envs/verl-new`,
with `torch`, `vllm`, `megatron-core`, `ray` etc. already present (inherited from
`psrl_smg_new`) — verified in Task 2, not here.

- [ ] **Step 3: Write the env file**

Write `/apdcephfs_zwfy10_303541817/share_303541817/lhy/env/verl_new.sh`:

```bash
# conda
source /apdcephfs_zwfy10/share_303541817/lhy/activate
conda activate verl-new
export PATH="/apdcephfs_zwfy10/share_303541817/lhy/anaconda3/envs/verl-new/bin:${PATH}"

# hydra
export HYDRA_FULL_ERROR=1

# ray
export RAY_worker_register_timeout_seconds=30

# vllm
export TOKENIZERS_PARALLELISM="true"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# verl telemetry sink (Phase 2). Empty means disabled; set it to enable.
export VERL_TELEMETRY_LOG_DIR="${VERL_TELEMETRY_LOG_DIR:-}"

echo "env done"
```

- [ ] **Step 4: Verify it sources cleanly**

Run: `bash -c "source /apdcephfs_zwfy10_303541817/share_303541817/lhy/env/verl_new.sh && python3 --version"`
Expected: prints `env done` followed by `Python 3.12.x`, with no errors (no `verl`/`torch` import yet
— that is Task 2).

- [ ] **Step 5: Commit**

```bash
cd /apdcephfs_zwfy10_303541817/share_303541817/lhy
git -C env status --short  # env/ is not its own git repo here; confirm whether it is tracked at all
```

If `env/` is not a git repo (expected — it is a plain directory of shell scripts next to other
untracked env files like `env/verl.sh`), skip committing this file; there is nothing to commit it
to. If it turns out to be tracked, commit only `env/verl_new.sh` with message
`"chore: add env script for the fresh verl-new conda env"`.

---

### Task 2: Swap the editable `verl` install for the fresh `verl-new` checkout

**Files:**
- Modify (install into, not edit): the `verl-new` conda env cloned in Task 1.

**Interfaces:**
- Consumes: `/apdcephfs_zwfy10_303541817/share_303541817/lhy/env/verl_new.sh` (Task 1); the cloned
  env's pre-existing `torch 2.11.0+cu128`, `vllm 0.22.1.dev0+...cu128`, `megatron-core 0.19.0`.
- Produces: an env where `verl` resolves to the `verl-new` checkout (not PSRL's fork) while every
  other package stays exactly what `psrl_smg_new` already had — consumed by every later task that
  runs anything in this env.

- [ ] **Step 1: Uninstall the inherited PSRL-fork `verl` and install `verl-new` in its place**

```bash
source /apdcephfs_zwfy10_303541817/share_303541817/lhy/env/verl_new.sh
pip uninstall -y verl
cd /apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new
pip install --no-deps -e .
```

Expected: the uninstall reports removing `verl 0.9.0.dev0` (PSRL's fork, previously editable from
`psrl/third_party/verl`); the install reports `Successfully installed verl-<version>` pointing at
this checkout. `--no-deps` is deliberate: a normal install would let pip resolve verl-new's
`pyproject.toml` dependencies and could silently upgrade the cloned env's carefully-pinned
`torch`/`vllm`/`megatron-core`.

- [ ] **Step 2: Verify the swap and the inherited GPU stack**

```bash
python3 -c "import verl, vllm, megatron.core, ray; print(verl.__file__, vllm.__version__, torch.__version__)" 2>&1 || true
python3 -c "import torch; print(torch.__version__)"
```

Expected: `verl.__file__` is under `verl-new/verl/__init__.py` (not `psrl/third_party/verl`);
`torch.__version__` is still `2.11.0+cu128` — if it changed, Step 1 upgraded something it should
not have, and that is a stop-and-report condition, not a "proceed anyway" one.

- [ ] **Step 3: Surface and resolve verl-new's genuinely new Python dependencies**

`verl-new`'s v1 trainer imports `transfer_queue` unconditionally at module load — this package did
not exist when PSRL forked verl (0.9.0.dev0, pre-dating the v1 trainer) and is very unlikely to be
in the cloned env. Find out what else is actually missing by trying to import the real module path
this plan's later tasks need, not just the top-level package:

```bash
python3 -c "from verl.trainer.ppo.v1 import trainer_base, trainer_separate_async" 2>&1
```

For each `ModuleNotFoundError` this reports, install just that one package, preferring `--no-deps`
first so it can't drag in an upgrade of anything already pinned:

```bash
pip install --no-deps <package>
python3 -c "import torch, vllm; print(torch.__version__, vllm.__version__)"  # re-check nothing moved
```

If `--no-deps` leaves the newly-installed package unable to import (it genuinely needs a
dependency absent from the env), retry that one package without `--no-deps`, then immediately
re-run the `torch`/`vllm` version check above. If `torch` or `vllm`'s version changed as a result,
stop and report it as a concern rather than continuing — this is exactly the kind of regression
Step 1's `--no-deps` was trying to prevent, and whether to accept the upgrade is a call for whoever
reviews this task, not something to decide mid-install. Repeat until the
`trainer_base, trainer_separate_async` import above succeeds cleanly.

- [ ] **Step 4: Commit**

Nothing to commit in `verl-new` itself (dependency install, not a source change) and nothing in
`SimRL`. No commit for this task.

---

### Task 3: Convert the Qwen3-8B checkpoint to a Megatron dist-checkpoint

**Files:**
- Create (data, not source): `/apdcephfs_zwfy10_303541817/share_303541817/lhy/models/mcore_ckpt_verl_new/Qwen3-8B/`

**Interfaces:**
- Consumes: the `verl-new` env from Task 2; the existing HF checkpoint at
  `/apdcephfs_zwfy10_303541817/share_303541817/lhy/models/Qwen3-8B` (already used by PSRL's own
  reference run).
- Produces: `DIST_CKPT_PATH`, consumed by Task 4's launch script.

- [ ] **Step 1: Confirm the target node is free, then run the converter there**

The converter launches a 6-rank distributed process (TP=2 × PP=3) that needs real GPU allocation
per rank, not just CPU — confirm `28.49.55.85` is still free before running:

```bash
ssh 28.49.55.85 "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"
```

Expected: all 8 GPUs at or near 0 MiB / 0%. If not, stop and report rather than running anyway.

The converter only gets a multi-rank `WORLD_SIZE` from `torchrun`-set env vars
(`verl/models/mcore/bridge_checkpoint.py:70-76`, `initialize_conversion()` defaults `WORLD_SIZE` to
`"1"` via `os.environ.setdefault` otherwise) — a plain `python3` invocation fails with
`WORLD_SIZE=1 must be divisible by TP * PP (6)`. Launch with `torchrun`:

```bash
ssh 28.49.55.85 "source /apdcephfs_zwfy10_303541817/share_303541817/lhy/env/verl_new.sh && \
cd /apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new && \
torchrun --nproc_per_node=6 --nnodes=1 scripts/converter_hf_to_mcore.py \
    --hf_model_path /apdcephfs_zwfy10_303541817/share_303541817/lhy/models/Qwen3-8B \
    --output_path /apdcephfs_zwfy10_303541817/share_303541817/lhy/models/mcore_ckpt_verl_new/Qwen3-8B \
    --tp_size 2 \
    --pp_size 3"
```

Expected: completes without error; expect several minutes for an 8B model reshape across 6 ranks.

- [ ] **Step 2: Verify the output**

The output lands on shared storage, so this can run from either node:
`find /apdcephfs_zwfy10_303541817/share_303541817/lhy/models/mcore_ckpt_verl_new/Qwen3-8B -maxdepth 2`
Expected: non-empty directory tree with per-rank weight shards (6 ranks' worth, matching TP=2×PP=3).

- [ ] **Step 3: Commit**

No commit — this is a generated data artifact, not source.

---

### Task 4: Write the launch script

**Files:**
- Create: `/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/examples/psrl_compare/qwen3_8b_separate_async_megatron.sh`

**Interfaces:**
- Consumes: `DIST_CKPT_PATH` from Task 3; the `verl-new` env from Task 1/2; PSRL's own data files
  (`dapo-math-17k.parquet`, `aime-2024.parquet`, already on disk, used unmodified).
- Produces: the launch script Task 5 runs.

- [ ] **Step 1: Create the `psrl-compare` branch**

All of this plan's verl-new work (this task's launch script and Phase 2's telemetry patches) lands
on one dedicated branch, cut from whatever commit the clone currently sits at, so nothing here ever
touches `verl-new`'s own default branch:

```bash
cd /apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new
git checkout -b psrl-compare
git rev-parse HEAD  # record this; it must equal a4b9354968705a441fb7da7367924b15b902e771
```

Expected: the printed commit equals `a4b9354968705a441fb7da7367924b15b902e771` (Task 1's pinned
commit) — the branch starts exactly there. Task 6 continues on this same branch rather than
creating a new one.

- [ ] **Step 2: Write the script**

Write `/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/examples/psrl_compare/qwen3_8b_separate_async_megatron.sh`:

```bash
#!/usr/bin/env bash
# Real verl trainer.v1.trainer_mode=separate_async run, configured to match
# PSRL's reference run as closely as two different codebases allow:
#   psrl/examples/dapo_trainer/qwen3_8b_megatron_simulate.sh
#
# Adapted from this repo's own tests/special_e2e/run_v1_separate_async.sh
# (Megatron branch). Written against base commit
# a4b9354968705a441fb7da7367924b15b902e771 — rebase the diff against that
# script if this repo moves past that commit and the template changes.
#
# Config choices that depart from the template's own CI defaults, and why:
#   - data.filter_overlong_prompts=False, matching PSRL (template uses True).
#   - Megatron activation recompute (override_transformer_config.recompute_*)
#     instead of the template's generic enable_gradient_checkpointing=True,
#     because PSRL configures recompute exactly this way for the same mesh.
#   - ppo_max_token_len_per_gpu / log_prob_max_token_len_per_gpu = 12800 and
#     rollout.gpu_memory_utilization=0.9, matching PSRL's values for this
#     context length (the template's 4096 / 0.6 were sized for a tiny CI model).
#   - trainer.v1.sampler.max_off_policy_threshold=1,
#     max_off_policy_strategy=wait: a first approximation of PSRL's
#     psrl.staleness=1 (PSRL's own run resumes interrupted requests rather
#     than dropping them). See the design spec's "Risks and open items".
#   - clip_ratio_low/high are left at this repo's own default: PSRL's script
#     does not override them, so neither does this one.
#   - transfer_queue.backend.SimpleStorage.total_storage_size=100000: the
#     default (null) is documented as "unlimited capacity" but actually
#     crashes (simple_storage_bootstrap.py divides by it unconditionally).
#     Not a PSRL-alignment choice -- a workaround for this gap, matching the
#     value two other example scripts in this checkout already use.
set -xeuo pipefail

export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VLLM_USE_V1=1

HF_MODEL_PATH=${HF_MODEL_PATH:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/models/Qwen3-8B}
DIST_CKPT_PATH=${DIST_CKPT_PATH:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/models/mcore_ckpt_verl_new/Qwen3-8B}

TRAIN_FILES=${TRAIN_FILES:-/jizhicfs/lhy/data/dapo/dapo-math-17k.parquet}
VAL_FILES=${VAL_FILES:-/jizhicfs/lhy/data/dapo/aime-2024.parquet}

TRAIN_BATCH_SIZE=32
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=10240
N_RESP_PER_PROMPT=8
PPO_MINI_BATCH_SIZE=32
PARAMETER_SYNC_STEP=$((TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE))
TOTAL_TRAINING_STEPS=5

N_GPUS_TRAINING=6
N_GPUS_ROLLOUT=2
TRAIN_TP=2
TRAIN_PP=$((N_GPUS_TRAINING / TRAIN_TP))
ROLLOUT_TP=1

experiment_name="qwen3_8b_separate_async_megatron"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/examples/psrl_compare/output"
CKPTS_DIR="${OUTPUT_DIR}/ckpts/${experiment_name}"
mkdir -p "${CKPTS_DIR}"

python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step=${PARAMETER_SYNC_STEP} \
    trainer.v1.separate_async.hybrid_rollout.enable_switch=False \
    trainer.v1.sampler.max_off_policy_threshold=1 \
    trainer.v1.sampler.max_off_policy_strategy=wait \
    transfer_queue.enable=True \
    transfer_queue.backend.SimpleStorage.total_storage_size=100000 \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.gen_batch_size=${TRAIN_BATCH_SIZE} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    actor_rollout_ref.model.path="${HF_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12800 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DIST_CKPT_PATH}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=${N_GPUS_ROLLOUT} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12800 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=12800 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    reward.reward_manager.name=dapo \
    trainer.logger='["console"]' \
    trainer.project_name=verl_psrl_compare \
    trainer.experiment_name="${experiment_name}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${N_GPUS_TRAINING} \
    trainer.total_epochs=15 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.default_local_dir="${CKPTS_DIR}" "$@" 2>&1 | tee "${OUTPUT_DIR}/${experiment_name}.log"
```

- [ ] **Step 3: Make it executable**

Run: `chmod +x /apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/examples/psrl_compare/qwen3_8b_separate_async_megatron.sh`

- [ ] **Step 4: Dry-run the config (no GPUs touched)**

```bash
source /apdcephfs_zwfy10_303541817/share_303541817/lhy/env/verl_new.sh
cd /apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new
python3 -m verl.trainer.main_ppo --cfg job \
    model_engine=megatron trainer.use_v1=True trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.parameter_sync_step=1 \
    actor_rollout_ref.model.path=/apdcephfs_zwfy10_303541817/share_303541817/lhy/models/Qwen3-8B \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 data.train_batch_size=32 \
    data.train_files=/jizhicfs/lhy/data/dapo/dapo-math-17k.parquet \
    data.val_files=/jizhicfs/lhy/data/dapo/aime-2024.parquet \
    > /tmp/verl_cfg_dry_run.yaml
```

Expected: exits 0 and writes a fully resolved Hydra config to `/tmp/verl_cfg_dry_run.yaml` with no
`MissingMandatoryValue`/interpolation errors — this validates every override key name resolves
against `ppo_trainer.yaml` before any GPU is touched.

- [ ] **Step 5: Commit (on the `psrl-compare` branch created in Step 1)**

```bash
cd /apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new
git add examples/psrl_compare/qwen3_8b_separate_async_megatron.sh
git commit -m "feat: add launch script matching PSRL's qwen3_8b_megatron_simulate.sh config"
```

---

### Task 5: CHECKPOINT — launch the real run

**Files:** none (execution only).

**Interfaces:**
- Consumes: Task 4's script, Task 1's env, Task 3's checkpoint.
- Produces: console logs under `examples/psrl_compare/output/`, consumed by Phase 2's telemetry
  verification and Phase 4's final comparison.

- [ ] **Step 1: Confirm the target node is actually free — STOP and ask before proceeding**

```bash
ssh 28.49.55.85 "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"
```

Do not proceed past this point without explicit user confirmation that `28.49.55.85` (the Global
Constraints' designated target node — not `29.162.247.148`, which has GPUs 0-3 held by other work)
is still free, and without the user's go-ahead to start a Ray cluster and launch an 8-GPU job
there. Cluster state can change between tasks — re-check even if it was free earlier.

- [ ] **Step 2: Start a fresh, single-node Ray cluster on `28.49.55.85`, under the `verl-new` env**

Do **not** use `psrl/examples/ray/ray_start.sh` — it sources `env/psrl.sh`, and Ray spawns every
worker Python process using whatever interpreter was active when the raylet started
(`--python_worker_command` is pinned to that env at `ray start` time; a driver sourcing a
different env afterwards does not change it). That mismatch is exactly what Task 5 hit the first
time: the raylet's workers imported `verl` from the old PSRL fork regardless of what the driver
script sourced. Since this is a single node, `ray_start.sh`'s `pssh` fan-out isn't needed anyway —
start the cluster directly, under the right env, on the right node:

```bash
ssh 28.49.55.85 "source /apdcephfs_zwfy10_303541817/share_303541817/lhy/env/verl_new.sh && \
ray start --head --port=8887 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-cpus=32"
```

Expected: "Local node IP" / "Ray runtime started" with no error; `ssh 28.49.55.85 "source
/apdcephfs_zwfy10_303541817/share_303541817/lhy/env/verl_new.sh && ray status"` now shows 1 active
node, 8 GPUs. Spot-check the fix worked before launching training:
`ssh 28.49.55.85 "ps aux | grep -o '/[^ ]*python_worker_command[^ ]*' ; ps aux | grep raylet | grep -o 'python_worker_command=[^ ]*'"`
should show the `verl-new` env's path, not `psrl_smg_new`'s.

- [ ] **Step 3: Launch the run on `28.49.55.85`**

The training process itself must run on the node with the GPUs, not locally:

```bash
ssh 28.49.55.85 "source /apdcephfs_zwfy10_303541817/share_303541817/lhy/env/verl_new.sh && \
bash /apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/examples/psrl_compare/qwen3_8b_separate_async_megatron.sh"
```

Expected: completes all 5 training steps without a Python traceback; the final console lines show
`global_steps=5` (or equivalent) with no unhandled exception. This is a real training run — expect
tens of minutes given model size, checkpoint conversion already done.

- [ ] **Step 4: Verify the run actually matched PSRL's shape**

Inspect `examples/psrl_compare/output/qwen3_8b_separate_async_megatron.log` for:
- 2 rollout replicas initialized (matching `N_GPUS_ROLLOUT=2`, `ROLLOUT_TP=1`).
- 6 training GPUs used (no CUDA OOM, no unexpected GPU count mismatch).
- 5 completed training steps.

If anything diverges (OOM, assertion failure, hang), this is the "find out" step the design spec
flagged — stop and report findings rather than silently adjusting config and re-running.

- [ ] **Step 5: Report, do not commit**

Nothing to commit here (an execution log, not source). Report the outcome back before moving to
Phase 2.

---

## Phase 2 — Instrument verl for telemetry

### Task 6: Write the telemetry sink module

**Files:**
- Create: `/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/verl/utils/telemetry_sink.py`
- Test: `/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/tests/utils/test_telemetry_sink_on_cpu.py`

**Interfaces:**
- Produces: `telemetry_dir() -> Path | None`, `write_request_trace(request_id, prompt_tokens,
  response_tokens, stop_reason, num_preempted) -> None`, `build_stats_record(replica_rank,
  num_running_requests, num_waiting_requests, kv_cache_usage, ts) -> dict`, `write_stats(replica_rank,
  snapshot: dict) -> None`, `maybe_start_stats_poller(server_handles, interval_s=2.0) -> ObjectRef |
  None`, `write_step_metrics(step, metrics: dict) -> None`. Consumed by Tasks 7, 8, 9.

- [ ] **Step 1: Confirm the `psrl-compare` branch (created in Task 4, Step 1) is checked out**

```bash
cd /apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new
git branch --show-current
```

Expected: `psrl-compare`. This task continues on that same branch — do not create a new one.

- [ ] **Step 2: Write the failing tests**

Write `tests/utils/test_telemetry_sink_on_cpu.py`:

```python
"""Tests for the optional structured telemetry sink used to compare a real
verl separate_async run against SimRL's simulation. No GPU needed."""

from __future__ import annotations

import json

import pytest

from verl.utils import telemetry_sink


class TestTelemetryDir:
    def test_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("VERL_TELEMETRY_LOG_DIR", raising=False)
        assert telemetry_sink.telemetry_dir() is None

    def test_the_configured_path_when_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VERL_TELEMETRY_LOG_DIR", str(tmp_path))
        assert telemetry_sink.telemetry_dir() == tmp_path


class TestRequestTraceGating:
    def test_nothing_is_written_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VERL_TELEMETRY_LOG_DIR", raising=False)
        telemetry_sink.write_request_trace("r1", 10, 20, "stop", 0)
        assert list(tmp_path.iterdir()) == []

    def test_a_jsonl_line_is_appended_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VERL_TELEMETRY_LOG_DIR", str(tmp_path))
        telemetry_sink.write_request_trace("r1", 10, 20, "stop", 1)
        [path] = list(tmp_path.glob("request_trace_pid*.jsonl"))
        record = json.loads(path.read_text().splitlines()[0])
        assert record["request_id"] == "r1"
        assert record["prompt_tokens"] == 10
        assert record["response_tokens"] == 20
        assert record["stop_reason"] == "stop"
        assert record["num_preempted"] == 1
        assert isinstance(record["ts"], float)


class TestStatsRecord:
    def test_build_stats_record_maps_snapshot_fields(self):
        record = telemetry_sink.build_stats_record(
            replica_rank=2, num_running_requests=5, num_waiting_requests=1, kv_cache_usage=0.5, ts=123.0
        )
        assert record == {
            "ts": 123.0,
            "num_running_reqs": 5,
            "num_waiting_reqs": 1,
            "kv_cache_usage": 0.5,
        }

    def test_write_stats_appends_to_a_per_replica_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VERL_TELEMETRY_LOG_DIR", str(tmp_path))
        telemetry_sink.write_stats(3, {"num_running_requests": 1, "num_waiting_requests": 0, "kv_cache_usage": 0.1})
        [path] = list(tmp_path.glob("stats_r*.jsonl"))
        assert path.name == "stats_r3.jsonl"
        record = json.loads(path.read_text().splitlines()[0])
        assert record["num_running_reqs"] == 1


class TestStepMetrics:
    def test_nothing_is_written_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VERL_TELEMETRY_LOG_DIR", raising=False)
        telemetry_sink.write_step_metrics(1, {"actor/loss": 0.5})
        assert list(tmp_path.iterdir()) == []

    def test_metrics_are_appended_with_step_and_timestamp(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VERL_TELEMETRY_LOG_DIR", str(tmp_path))
        telemetry_sink.write_step_metrics(1, {"actor/loss": 0.5})
        record = json.loads((tmp_path / "train_metrics.jsonl").read_text().splitlines()[0])
        assert record["step"] == 1
        assert record["actor/loss"] == 0.5
        assert isinstance(record["ts"], float)


class TestStatsPollerGating:
    def test_no_poller_is_started_when_disabled(self, monkeypatch):
        monkeypatch.delenv("VERL_TELEMETRY_LOG_DIR", raising=False)
        assert telemetry_sink.maybe_start_stats_poller(server_handles=[]) is None

    def test_a_poller_is_launched_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VERL_TELEMETRY_LOG_DIR", str(tmp_path))
        calls = []
        monkeypatch.setattr(
            telemetry_sink._poll_engine_stats,
            "remote",
            lambda handles, interval_s: calls.append((handles, interval_s)),
        )
        telemetry_sink.maybe_start_stats_poller(server_handles=["h0", "h1"], interval_s=3.0)
        assert calls == [(["h0", "h1"], 3.0)]
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
source /apdcephfs_zwfy10_303541817/share_303541817/lhy/env/verl_new.sh
cd /apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new
pytest tests/utils/test_telemetry_sink_on_cpu.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'verl.utils.telemetry_sink'`.

- [ ] **Step 4: Write the implementation**

Write `verl/utils/telemetry_sink.py`:

```python
"""Optional structured telemetry for comparing a real verl separate_async run
against SimRL's simulation of it.

Every function here is a no-op unless `VERL_TELEMETRY_LOG_DIR` is set, so
these hooks have zero effect on a normal training run. Each record is one
JSON line, appended and flushed immediately so a killed run still leaves
readable output.

Field names match PSRL's own `stats_r*_dp*.jsonl` schema
(`num_running_reqs`, `num_waiting_reqs`, `kv_cache_usage`) so the SimRL-side
converter (`srl.telemetry.verl_log`) can mirror `psrl_log.py`'s shape.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import ray

logger = logging.getLogger(__name__)


def telemetry_dir() -> Path | None:
    """The configured telemetry directory, or None when disabled."""
    raw = os.environ.get("VERL_TELEMETRY_LOG_DIR")
    return Path(raw) if raw else None


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def write_request_trace(
    request_id: str,
    prompt_tokens: int,
    response_tokens: int,
    stop_reason: str | None,
    num_preempted: int | None,
) -> None:
    """Append one completed request to this process's trace file.

    One file per writer process (`request_trace_pid{os.getpid()}.jsonl`)
    rather than one shared file: `FullyAsyncLLMServerClient.generate()` runs
    in whichever AgentLoopWorker process issued the request, and multiple
    processes appending to one file risks interleaved writes.
    """
    log_dir = telemetry_dir()
    if log_dir is None:
        return
    record = {
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "stop_reason": stop_reason,
        "num_preempted": num_preempted,
        "ts": time.time(),
    }
    _append_jsonl(log_dir / f"request_trace_pid{os.getpid()}.jsonl", record)


def build_stats_record(
    replica_rank: int,
    num_running_requests: int,
    num_waiting_requests: int,
    kv_cache_usage: float,
    ts: float,
) -> dict[str, Any]:
    """Map a `vLLMHttpServer.snapshot()` result into PSRL's stats field names."""
    return {
        "ts": ts,
        "num_running_reqs": num_running_requests,
        "num_waiting_reqs": num_waiting_requests,
        "kv_cache_usage": kv_cache_usage,
    }


def write_stats(replica_rank: int, snapshot: dict[str, Any]) -> None:
    """Append one occupancy sample for `replica_rank` to its own stats file."""
    log_dir = telemetry_dir()
    if log_dir is None:
        return
    record = build_stats_record(
        replica_rank,
        snapshot["num_running_requests"],
        snapshot["num_waiting_requests"],
        snapshot["kv_cache_usage"],
        time.time(),
    )
    _append_jsonl(log_dir / f"stats_r{replica_rank}.jsonl", record)


def write_step_metrics(step: int, metrics: dict[str, Any]) -> None:
    """Mirror one step's already-computed metrics dict to a local JSONL file.

    `trainer_base.py`'s `fit()` already builds a `metrics` dict (timing,
    throughput, loss) and logs it to `trainer.logger`'s backends each step.
    This writes the same dict to disk unchanged, so Phase 4 can read PPO
    step/recompute/train timing without inventing a separate marker format.
    """
    log_dir = telemetry_dir()
    if log_dir is None:
        return
    _append_jsonl(log_dir / "train_metrics.jsonl", {"ts": time.time(), "step": step, **metrics})


@ray.remote(num_cpus=0)
def _poll_engine_stats(server_handles: list, interval_s: float) -> None:
    while True:
        for rank, handle in enumerate(server_handles):
            try:
                snapshot = ray.get(handle.snapshot.remote())
                write_stats(rank, snapshot)
            except Exception:
                logger.exception("Stats poll failed for replica %d", rank)
        time.sleep(interval_s)


def maybe_start_stats_poller(server_handles: list, interval_s: float = 2.0):
    """Start a background Ray task polling each replica's `snapshot()`, or
    return None when telemetry is disabled.

    Fire-and-forget: the returned ref is never awaited. A short research run
    does not need an explicit stop; the poller dies with the Ray cluster.
    """
    if telemetry_dir() is None:
        return None
    return _poll_engine_stats.remote(server_handles, interval_s)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/utils/test_telemetry_sink_on_cpu.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add verl/utils/telemetry_sink.py tests/utils/test_telemetry_sink_on_cpu.py
git commit -m "feat: add optional structured telemetry sink gated by VERL_TELEMETRY_LOG_DIR"
```

---

### Task 7: Patch `llm_server.py` for the per-request completion trace

**Files:**
- Modify: `verl/workers/rollout/llm_server.py:365-369`
- Test: `/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/tests/workers/rollout/test_llm_server_telemetry_on_cpu.py`

**Interfaces:**
- Consumes: `telemetry_sink.write_request_trace` (Task 6).

- [ ] **Step 1: Write the failing test**

Write `tests/workers/rollout/test_llm_server_telemetry_on_cpu.py` (create the `tests/workers/rollout/`
directory if it does not already exist):

```python
"""Tests that FullyAsyncLLMServerClient.generate() writes one request-trace
row per completed request, gated by VERL_TELEMETRY_LOG_DIR. No GPU needed —
the underlying server call is mocked out."""

from __future__ import annotations

import asyncio
import json

from omegaconf import OmegaConf

from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerClient
from verl.workers.rollout.replica import TokenOutput


def _config():
    return OmegaConf.create({"actor_rollout_ref": {"rollout": {"name": "vllm"}}})


class TestRequestTraceOnGenerate:
    def test_a_completed_call_appends_one_trace_line(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VERL_TELEMETRY_LOG_DIR", str(tmp_path))

        async def fake_generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
            return TokenOutput(token_ids=[1, 2, 3], stop_reason="stop", num_preempted=0)

        monkeypatch.setattr(LLMServerClient, "generate", fake_generate)

        client = FullyAsyncLLMServerClient(config=_config(), load_balancer_handle=None)
        asyncio.run(client.generate(request_id="req-1", prompt_ids=[9, 9], sampling_params={}))

        [path] = list(tmp_path.glob("request_trace_pid*.jsonl"))
        record = json.loads(path.read_text().splitlines()[0])
        assert record["request_id"] == "req-1"
        assert record["prompt_tokens"] == 2
        assert record["response_tokens"] == 3
        assert record["stop_reason"] == "stop"
        assert record["num_preempted"] == 0

    def test_nothing_is_written_when_telemetry_is_disabled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VERL_TELEMETRY_LOG_DIR", raising=False)

        async def fake_generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
            return TokenOutput(token_ids=[1], stop_reason="stop", num_preempted=0)

        monkeypatch.setattr(LLMServerClient, "generate", fake_generate)

        client = FullyAsyncLLMServerClient(config=_config(), load_balancer_handle=None)
        asyncio.run(client.generate(request_id="req-2", prompt_ids=[9], sampling_params={}))

        assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/workers/rollout/test_llm_server_telemetry_on_cpu.py -v
```

Expected: FAIL — `test_a_completed_call_appends_one_trace_line` fails because no file is written
yet (the patch in Step 3 does not exist).

- [ ] **Step 3: Patch `llm_server.py`**

Add the import near the top of the file (after the existing `from verl.workers.rollout...` imports,
around line 37):

```python
from verl.utils.telemetry_sink import write_request_trace
```

Then change the end of `FullyAsyncLLMServerClient.generate()` (currently lines 365-369):

```python
        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        final_output.extra_fields["num_cached_tokens"] = num_cached_tokens
        return final_output
```

to:

```python
        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        final_output.extra_fields["num_cached_tokens"] = num_cached_tokens
        write_request_trace(
            request_id,
            len(prompt_ids),
            len(final_output.token_ids),
            final_output.stop_reason,
            final_output.num_preempted,
        )
        return final_output
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/workers/rollout/test_llm_server_telemetry_on_cpu.py -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add verl/workers/rollout/llm_server.py tests/workers/rollout/test_llm_server_telemetry_on_cpu.py
git commit -m "feat: write a request-trace row on each completed generate() call"
```

---

### Task 8: Add the per-engine stats poller to the separate_async trainer

**Files:**
- Modify: `verl/trainer/ppo/v1/trainer_separate_async.py:154-169` (`_setup()`)
- Test: `/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/tests/trainer/ppo/v1/test_trainer_separate_async_telemetry_on_cpu.py`

**Interfaces:**
- Consumes: `telemetry_sink.maybe_start_stats_poller` (Task 6).

- [ ] **Step 1: Write the failing test**

Write `tests/trainer/ppo/v1/test_trainer_separate_async_telemetry_on_cpu.py` (create
`tests/trainer/ppo/v1/` if it does not exist):

```python
"""Tests that PPOTrainerSeparateAsync._setup() starts the stats poller against
its standalone rollout pool's server handles. No GPU, no Ray cluster needed —
the trainer's heavy setup machinery is monkeypatched out."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from verl.trainer.ppo.v1 import trainer_separate_async as mod


class TestStatsPollerIsStarted:
    def test_setup_starts_the_poller_with_standalone_server_handles(self, monkeypatch):
        monkeypatch.setenv("VERL_TELEMETRY_LOG_DIR", "/tmp/doesnt-matter-for-this-test")

        calls = []
        monkeypatch.setattr(mod, "maybe_start_stats_poller", lambda server_handles, **kw: calls.append(server_handles))

        trainer = mod.PPOTrainerSeparateAsync.__new__(mod.PPOTrainerSeparateAsync)
        trainer.standalone_server_manager = SimpleNamespace(server_handles=["h0", "h1"])
        trainer._start_stats_poller()

        assert calls == [["h0", "h1"]]

    def test_poller_is_not_started_when_telemetry_is_disabled(self, monkeypatch):
        monkeypatch.delenv("VERL_TELEMETRY_LOG_DIR", raising=False)
        real = mod.maybe_start_stats_poller  # the real gating function, not mocked
        monkeypatch.setattr(mod, "maybe_start_stats_poller", real)

        trainer = mod.PPOTrainerSeparateAsync.__new__(mod.PPOTrainerSeparateAsync)
        trainer.standalone_server_manager = SimpleNamespace(server_handles=["h0"])
        result = trainer._start_stats_poller()

        assert result is None
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/trainer/ppo/v1/test_trainer_separate_async_telemetry_on_cpu.py -v
```

Expected: FAIL — `AttributeError: 'PPOTrainerSeparateAsync' object has no attribute '_start_stats_poller'`.

- [ ] **Step 3: Patch `trainer_separate_async.py`**

Add the import (after the existing `from verl.workers.rollout.llm_server import ...` line, around
line 33):

```python
from verl.utils.telemetry_sink import maybe_start_stats_poller
```

Add a new method anywhere in the class body (e.g. right after `_setup`, i.e. after the current
lines 154-199):

```python
    def _start_stats_poller(self):
        """Start the optional per-engine occupancy poller against the
        standalone rollout pool. A no-op unless VERL_TELEMETRY_LOG_DIR is set.
        """
        return maybe_start_stats_poller(self.standalone_server_manager.server_handles)
```

Then call it at the end of `_setup()` (after `self.standalone_checkpoint_manager = ...` is
assigned, i.e. after the current line 183, before the `if self._enable_hybrid_replicas:` block that
starts at line 186):

```python
        self._start_stats_poller()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/trainer/ppo/v1/test_trainer_separate_async_telemetry_on_cpu.py -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add verl/trainer/ppo/v1/trainer_separate_async.py tests/trainer/ppo/v1/test_trainer_separate_async_telemetry_on_cpu.py
git commit -m "feat: poll standalone rollout replicas' engine stats when telemetry is enabled"
```

---

### Task 9: Mirror per-step metrics to a local JSONL file

**Files:**
- Modify: `verl/trainer/ppo/v1/trainer_base.py:515`

**Interfaces:**
- Consumes: `telemetry_sink.write_step_metrics` (Task 6).

No unit test for this one-line change: `fit()` depends on a fully constructed trainer (real data
loaders, Ray worker groups, resource pools) that would need disproportionate mocking to exercise
meaningfully, and the line itself has no branching logic to test in isolation — `write_step_metrics`
is already unit-tested in Task 6. This is verified empirically in Task 10's real-run checkpoint by
checking `train_metrics.jsonl` has one row per step.

- [ ] **Step 1: Patch `trainer_base.py`**

Add the import (alongside the existing imports at the top of the file, e.g. after `from tqdm import
tqdm` around line 33):

```python
from verl.utils.telemetry_sink import write_step_metrics
```

Then change (currently line 515):

```python
            self.logger.log(data=metrics, step=self.global_steps)
```

to:

```python
            self.logger.log(data=metrics, step=self.global_steps)
            write_step_metrics(self.global_steps, metrics)
```

- [ ] **Step 2: Verify nothing else broke**

```bash
cd /apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new
python3 -c "import ast; ast.parse(open('verl/trainer/ppo/v1/trainer_base.py').read())"
pytest tests/utils/test_telemetry_sink_on_cpu.py tests/workers/rollout/test_llm_server_telemetry_on_cpu.py tests/trainer/ppo/v1/test_trainer_separate_async_telemetry_on_cpu.py -v
```

Expected: the `ast.parse` call prints nothing (valid syntax); all previously-written telemetry tests
still PASS (this change does not affect them).

- [ ] **Step 3: Commit**

```bash
git add verl/trainer/ppo/v1/trainer_base.py
git commit -m "feat: mirror per-step training metrics to a local JSONL file when telemetry is enabled"
```

---

### Task 10: CHECKPOINT — re-run with telemetry enabled and verify the logs

**Files:** none (execution only).

**Interfaces:**
- Consumes: Task 4's launch script, the `psrl-compare` branch (Tasks 6-9).
- Produces: the telemetry directory Phase 3's converter is built against.

- [ ] **Step 1: Confirm the cluster is still free — STOP and ask before proceeding**

Same check as Task 5, Step 1. If the Task 5 Ray cluster from Phase 1 is still up and idle, it may be
reused; otherwise repeat Task 5's Steps 1-2 on a confirmed-free node.

- [ ] **Step 2: Launch with telemetry enabled, on `28.49.55.85`**

```bash
mkdir -p /apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/examples/psrl_compare/output/telemetry
ssh 28.49.55.85 "export VERL_TELEMETRY_LOG_DIR=/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/examples/psrl_compare/output/telemetry && \
source /apdcephfs_zwfy10_303541817/share_303541817/lhy/env/verl_new.sh && \
bash /apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/examples/psrl_compare/qwen3_8b_separate_async_megatron.sh"
```

`mkdir` runs locally first since it's shared storage, visible identically from either node; the
training process itself must run on `28.49.55.85`, the node with the GPUs.

- [ ] **Step 3: Verify the three telemetry streams are non-empty and well-formed**

```bash
cd "${VERL_TELEMETRY_LOG_DIR}"
ls -la
for f in request_trace_pid*.jsonl stats_r*.jsonl train_metrics.jsonl; do
    echo "=== $f ==="; wc -l $f; head -1 $f | python3 -m json.tool
done
```

Expected: at least one `request_trace_pid*.jsonl` file with as many lines as completed requests (on
the order of `train_batch_size * rollout.n` per step, times steps actually admitted — exact count
depends on staleness drops/waits), 2 `stats_r*.jsonl` files (one per rollout replica) each with
roughly `run_duration_s / 2` lines, and `train_metrics.jsonl` with exactly 5 lines (one per training
step). Every `head -1 | python3 -m json.tool` call must print valid, pretty-printed JSON.

- [ ] **Step 4: Review the `psrl-compare` branch's commit log**

```bash
cd /apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new
git log --oneline psrl-compare -8
git push -u origin psrl-compare  # only if the user has a remote to push to and asks for it
```

Report the branch's commit log back; do not push unless explicitly asked.

---

## Phase 3 — SimRL-side converter and `verl_aligned.sh`

### Task 11: Write `srl/telemetry/verl_log.py`

**Files:**
- Create: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/srl/telemetry/verl_log.py`
- Test: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/tests/test_verl_log.py`

**Interfaces:**
- Consumes: `srl.telemetry.store.JsonlStore`, `srl.workload.trace.RequestGroup`,
  `srl.workload.trace.write_group_trace`, `srl.workload.trace.read_group_trace`.
- Produces: `VerlLogLayout`, `ConversionSummary`, `convert_verl_logs(log_dir, store, layout=None) ->
  ConversionSummary`, `extract_workload_trace(log_dir, output_path, group_size, layout=None,
  max_output_tokens=0) -> Path`. Consumed by Task 12.

- [ ] **Step 1: Write the failing tests**

Write `tests/test_verl_log.py`:

```python
"""Tests that a verl separate_async run's telemetry directory converts into
simulator-shaped telemetry streams and a workload trace."""

from __future__ import annotations

import json

import pytest

from srl.telemetry.store import JsonlStore
from srl.telemetry.verl_log import VerlLogLayout, convert_verl_logs, extract_workload_trace
from srl.workload.trace import read_group_trace

_N_SAMPLES = 50


def _write_stats(path, busy: bool) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(_N_SAMPLES):
            f.write(
                json.dumps(
                    {
                        "ts": 1000.0 + i,
                        "num_running_reqs": (i % 16) if busy else 0,
                        "num_waiting_reqs": 0,
                        "kv_cache_usage": (i % 16) / 16.0 if busy else 0.0,
                    }
                )
                + "\n"
            )


def _write_requests(path, start_id: int, n: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "request_id": f"req-{start_id + i:04d}",
                        "prompt_tokens": 100,
                        "response_tokens": 200 + i,
                        "stop_reason": "stop",
                        "num_preempted": 0,
                        "ts": 1000.0 + i,
                    }
                )
                + "\n"
            )


def _build_log_dir(tmp_path):
    log_dir = tmp_path / "verl_telemetry"
    log_dir.mkdir()
    _write_stats(log_dir / "stats_r0.jsonl", busy=True)
    _write_stats(log_dir / "stats_r1.jsonl", busy=True)
    _write_requests(log_dir / "request_trace_pid111.jsonl", start_id=0, n=8)
    _write_requests(log_dir / "request_trace_pid222.jsonl", start_id=8, n=8)
    return log_dir


def _convert(tmp_path, log_dir, **kwargs):
    store = JsonlStore(tmp_path / "logs", overwrite=True)
    summary = convert_verl_logs(log_dir, store, **kwargs)
    store.close()
    return summary, JsonlStore(tmp_path / "logs")


class TestInstanceStreams:
    def test_all_replicas_are_converted(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        summary, _ = _convert(tmp_path, log_dir)
        assert summary.instance_ids == [0, 1]
        assert summary.n_instance_rows == 2 * _N_SAMPLES

    def test_instance_time_starts_at_zero(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        _, store = _convert(tmp_path, log_dir)
        assert next(iter(store.iter("instance/0")))["t"] == pytest.approx(0.0)

    def test_the_field_names_are_the_ones_the_plotter_reads(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        _, store = _convert(tmp_path, log_dir)
        row = next(iter(store.iter("instance/0")))
        assert {"t", "instance_id", "num_running_reqs", "num_waiting_reqs", "kv_cache_usage"} <= set(row)


class TestRequestCounting:
    def test_requests_across_processes_are_summed(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        summary, _ = _convert(tmp_path, log_dir)
        assert summary.n_request_rows == 16


class TestBadInput:
    def test_a_missing_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No verl telemetry directory"):
            _convert(tmp_path, tmp_path / "nope")

    def test_a_directory_without_stats_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No stats files matching"):
            _convert(tmp_path, empty)


class TestWorkloadTraceExtraction:
    def test_groups_are_chunked_by_group_size(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        out = tmp_path / "trace.jsonl"
        extract_workload_trace(log_dir, out, group_size=4)
        groups = read_group_trace(out)
        assert len(groups) == 4
        assert len(groups[0].output_lens) == 4

    def test_a_count_not_divisible_by_group_size_raises(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        with pytest.raises(ValueError, match="not a whole number of groups"):
            extract_workload_trace(log_dir, tmp_path / "trace.jsonl", group_size=5)

    def test_missing_trace_files_raise(self, tmp_path):
        log_dir = tmp_path / "no_trace"
        log_dir.mkdir()
        (log_dir / "stats_r0.jsonl").write_text('{"ts": 0, "num_running_reqs": 0, "num_waiting_reqs": 0, "kv_cache_usage": 0.0}\n')
        with pytest.raises(ValueError, match="No request trace files"):
            extract_workload_trace(log_dir, tmp_path / "trace.jsonl", group_size=4)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
pytest tests/test_verl_log.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'srl.telemetry.verl_log'`.

- [ ] **Step 3: Write the implementation**

Write `srl/telemetry/verl_log.py`:

```python
"""Convert a real verl separate_async run's telemetry directory into the
simulator's own `JsonlStore` streams.

Verl's telemetry (`verl/utils/telemetry_sink.py` in verl-new's
`psrl-compare` branch) already writes structured JSON with proper
wall-clock timestamps, so unlike `psrl_log.py` this module does no text-log
parsing and no UTC-offset inference: every source file already carries a
`ts` field in the same clock.

Nothing here writes to the verl run's telemetry directory: it is opened
read-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .._logging import srl_logger
from ..workload.trace import RequestGroup, write_group_trace
from .store import JsonlStore


@dataclass(frozen=True)
class VerlLogLayout:
    """Where each fact lives inside a verl telemetry directory.

    Args:
        request_trace_glob: Per-process completion trace, one JSON object per
            line. One file per writer process (`telemetry_sink.py` names each
            by its own pid), so every match is read and merged.
        stats_glob: Per-replica engine occupancy, one JSON object per line.
            `{r}` in the filename is the replica rank.
    """

    request_trace_glob: str = "request_trace_pid*.jsonl"
    stats_glob: str = "stats_r*.jsonl"


@dataclass
class ConversionSummary:
    """What `convert_verl_logs` found, for the CLI to report back.

    Args:
        instance_ids: Replica ranks written as `instance/*` streams.
        n_instance_rows: Occupancy samples written across all instances.
        n_request_rows: Completed requests read from the trace files.
        duration_s: Wall-clock span covered by the instance streams.
    """

    instance_ids: list[int] = field(default_factory=list)
    n_instance_rows: int = 0
    n_request_rows: int = 0
    duration_s: float = 0.0


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _instance_id_from_stats_path(path: Path) -> int:
    # "stats_r3.jsonl" -> 3
    return int(path.stem.rsplit("_r", 1)[1])


def convert_verl_logs(
    log_dir: str | Path,
    store: JsonlStore,
    layout: VerlLogLayout | None = None,
) -> ConversionSummary:
    """Rewrite a verl telemetry directory into simulator-shaped streams.

    Writes `instance/{id}` into `store`, using the same field names
    `SimTelemetry` writes, so `SimPlotter` needs no knowledge of where the
    records came from. Times are rebased so that zero is the first engine
    stats sample, matching the simulator's time-since-start axis.

    A `train` stream analogous to PSRL's recompute/train/sync phases is not
    written here: `trainer_base.py`'s per-step metrics mirror
    (`train_metrics.jsonl`) carries different fields than PSRL's three-phase
    breakdown, and reshaping it is deferred to whoever consumes it directly
    once the real run's actual field names are in hand.

    Args:
        log_dir: A verl run's telemetry directory (`VERL_TELEMETRY_LOG_DIR`).
            Opened read-only.
        store: Destination store, typically rooted at `<output_dir>/logs`.
        layout: Which files to read. Defaults to the layout
            `telemetry_sink.py` writes today.

    Returns:
        A summary of what was converted, for the caller to report.

    Raises:
        ValueError: If the directory has no stats files to convert.
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        raise ValueError(f"No verl telemetry directory at {log_dir!s}.")
    layout = layout or VerlLogLayout()

    stats_paths = sorted(log_dir.glob(layout.stats_glob))
    if not stats_paths:
        raise ValueError(f"No stats files matching {layout.stats_glob!r} found under {log_dir!s}.")

    stats_rows = {_instance_id_from_stats_path(p): _read_jsonl(p) for p in stats_paths}
    empty = sorted(iid for iid, rows in stats_rows.items() if not rows)
    if empty:
        raise ValueError(f"Replicas {empty} have empty stats files under {log_dir!s}, so there is nothing to plot.")

    origin = min(rows[0]["ts"] for rows in stats_rows.values())
    last = max(rows[-1]["ts"] for rows in stats_rows.values())
    duration_s = last - origin

    n_instance_rows = 0
    for instance_id, rows in stats_rows.items():
        for row in rows:
            store.write(
                f"instance/{instance_id}",
                t=row["ts"] - origin,
                instance_id=instance_id,
                num_running_reqs=row["num_running_reqs"],
                num_waiting_reqs=row["num_waiting_reqs"],
                kv_cache_usage=row["kv_cache_usage"],
            )
            n_instance_rows += 1

    request_paths = sorted(log_dir.glob(layout.request_trace_glob))
    n_request_rows = sum(len(_read_jsonl(p)) for p in request_paths)

    srl_logger.info(
        "Converted %d instance rows across replicas %s and counted %d completed requests from %s.",
        n_instance_rows,
        sorted(stats_rows),
        n_request_rows,
        str(log_dir),
    )

    return ConversionSummary(
        instance_ids=sorted(stats_rows),
        n_instance_rows=n_instance_rows,
        n_request_rows=n_request_rows,
        duration_s=duration_s,
    )


def extract_workload_trace(
    log_dir: str | Path,
    output_path: str | Path,
    group_size: int,
    layout: VerlLogLayout | None = None,
    max_output_tokens: int = 0,
) -> Path:
    """Write the lengths the engine actually generated as a joint group trace.

    Reads every completed request from the `request_trace_pid*.jsonl` files
    `telemetry_sink.py` writes (one per writer process). Unlike PSRL's
    `route_trace.log`, each line is already the final, cumulative record for
    that request (`FullyAsyncLLMServerClient.generate()`'s own resume loop
    has already finished by the time it writes), so there is no "only read
    the last pop" step to do.

    Grouping is not recorded in the trace, so requests are chunked in
    request_id order into groups of `group_size`, exactly as `psrl_log.py`
    does. That reproduces the length *distribution* and the group *arity*,
    not which prompt each response belonged to.

    Args:
        log_dir: A verl run's telemetry directory. Opened read-only.
        output_path: Destination JSONL trace.
        group_size: Responses per prompt (`actor_rollout_ref.rollout.n`).
        layout: Which files to read.
        max_output_tokens: Generation cap the run used, for truncation stats.
            Defaults to the longest observed response.

    Returns:
        The path written.

    Raises:
        ValueError: If no completed requests are found, or if the count is
            not a whole number of groups.
    """
    log_dir = Path(log_dir)
    layout = layout or VerlLogLayout()
    request_paths = sorted(log_dir.glob(layout.request_trace_glob))
    if not request_paths:
        raise ValueError(
            f"No request trace files matching {layout.request_trace_glob!r} under {log_dir!s}, "
            f"so real generated lengths are unavailable."
        )

    parsed: list[tuple[str, int, int]] = []
    for path in request_paths:
        for row in _read_jsonl(path):
            parsed.append((row["request_id"], row["prompt_tokens"], row["response_tokens"]))

    if not parsed:
        raise ValueError(f"No completed requests in {request_paths!r}, so no generated lengths could be read.")
    if len(parsed) % group_size != 0:
        raise ValueError(
            f"Found {len(parsed)} completed requests, which is not a whole number of groups of "
            f"{group_size}. Pass the group_size this run was launched with."
        )

    parsed.sort(key=lambda row: row[0])
    groups = [
        RequestGroup(
            group_id=index,
            prompt_len=parsed[index * group_size][1],
            output_lens=tuple(row[2] for row in parsed[index * group_size : (index + 1) * group_size]),
        )
        for index in range(len(parsed) // group_size)
    ]
    cap = max_output_tokens or max(row[2] for row in parsed)
    srl_logger.info(
        "Extracted %d trajectories into %d groups of %d from %s.",
        len(parsed),
        len(groups),
        group_size,
        str(log_dir),
    )
    return write_group_trace(
        groups,
        output_path,
        metadata={
            "source": str(log_dir),
            "max_output_tokens": cap,
            "group_size": group_size,
            "provenance": "verl-separate-async-request-trace",
        },
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_verl_log.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add srl/telemetry/verl_log.py tests/test_verl_log.py
git commit -m "feat: add verl_log converter, mirroring psrl_log for JSON-native telemetry"
```

---

### Task 12: Write `srl/cli/verl_plot.py` and `examples/visualize/plot_verl_run.sh`

**Files:**
- Create: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/srl/cli/verl_plot.py`
- Modify: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/pyproject.toml` (add console-script entry)
- Create: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/examples/visualize/plot_verl_run.sh`

**Interfaces:**
- Consumes: `srl.telemetry.verl_log.{VerlLogLayout, convert_verl_logs}` (Task 11),
  `srl.telemetry.store.JsonlStore`, `srl.visualizer.plots.SimPlotter`.

- [ ] **Step 1: Write `srl/cli/verl_plot.py`**

```python
"""Standalone CLI: render the instance-load plot from a real verl
separate_async run's telemetry directory.

This is the verl counterpart to `srl.cli.psrl_plot`: it converts a verl
telemetry directory into the same streams a simulation writes, then renders
the figure with the same `SimPlotter`, so a real run and a simulated one can
be compared without either side having its own drawing code.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from .._logging import setup_logging
from ..telemetry.store import JsonlStore
from ..telemetry.verl_log import VerlLogLayout, convert_verl_logs
from ..visualizer.plots import SimPlotter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="srl-verl-plot",
        description="Render the SimRL instance-load plot from a real verl separate_async run's telemetry.",
    )
    parser.add_argument("log_dir", help="A verl run's telemetry directory (VERL_TELEMETRY_LOG_DIR). Read only.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the converted logs and the plot into.")
    parser.add_argument("--time-label", default="Wall-clock Time (s)", help="X-axis label for the rendered figure.")
    parser.add_argument("--logs-subdir", default="logs", help="Name of the logs subdirectory to write under output_dir.")
    parser.add_argument("--plots-subdir", default="plots", help="Name of the plots subdirectory to write under output_dir.")

    layout_group = parser.add_argument_group("log layout overrides")
    for spec in fields(VerlLogLayout):
        layout_group.add_argument(f"--{spec.name.replace('_', '-')}", default=spec.default, help=f"Default: {spec.default!r}")
    return parser


def main() -> None:
    setup_logging()

    args = _build_parser().parse_args()
    layout = VerlLogLayout(**{spec.name: getattr(args, spec.name) for spec in fields(VerlLogLayout)})

    output_dir = Path(args.output_dir)
    store = JsonlStore(output_dir / args.logs_subdir, overwrite=True)
    try:
        summary = convert_verl_logs(args.log_dir, store, layout=layout)
    finally:
        store.close()

    plotter = SimPlotter(
        JsonlStore(output_dir / args.logs_subdir),
        output_dir / args.plots_subdir,
        time_label=args.time_label,
    )
    path = plotter.plot_instance_load()

    print(f"Converted {args.log_dir}")
    print(f"  instances plotted : {summary.instance_ids}")
    print(f"  occupancy samples : {summary.n_instance_rows}")
    print(f"  completed requests: {summary.n_request_rows}")
    print(f"  time span         : {summary.duration_s:.1f} s")
    print(f"Logs written to: {output_dir / args.logs_subdir}")
    print(f"Plot written to: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register the console-script entry**

In `pyproject.toml`, add a line next to the existing `srl-psrl-plot` entry (around line 30):

```toml
srl-verl-plot = "srl.cli.verl_plot:main"
```

- [ ] **Step 3: Write `examples/visualize/plot_verl_run.sh`**

```bash
#!/usr/bin/env bash
# Render the instance-load plot from a real verl separate_async run's
# telemetry directory, using the same plotting code the simulator uses, so
# the two figures can be compared directly.
#
# The telemetry directory is read only and never modified.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VERL_LOG_DIR="${VERL_LOG_DIR:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/examples/psrl_compare/output/telemetry}"
RUN_NAME="${RUN_NAME:-verl_separate_async}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/visualize/${RUN_NAME}}"
FIGURE="${FIGURE:-${SCRIPT_DIR}/instance_load_${RUN_NAME}.png}"
TIME_LABEL="${TIME_LABEL:-Wall-clock Time (s)}"

if [[ ! -d "${VERL_LOG_DIR}" ]]; then
    echo "Missing verl telemetry directory: ${VERL_LOG_DIR}" >&2
    echo "Point VERL_LOG_DIR at the run's VERL_TELEMETRY_LOG_DIR, for example:" >&2
    echo "  VERL_LOG_DIR=/path/to/telemetry bash examples/visualize/plot_verl_run.sh" >&2
    exit 1
fi

python -m srl.cli.verl_plot "${VERL_LOG_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --time-label "${TIME_LABEL}"

cp "${OUTPUT_DIR}/plots/instance_load.png" "${FIGURE}"
echo "Figure copied to: ${FIGURE}"
```

- [ ] **Step 4: Make it executable and smoke-test with synthetic data**

```bash
chmod +x /apdcephfs_zwfy/share_303760348/lhy/SimRL/examples/visualize/plot_verl_run.sh
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
python3 -c "
import sys
sys.path.insert(0, 'tests')
from test_verl_log import _build_log_dir
import tempfile
with tempfile.TemporaryDirectory() as d:
    from pathlib import Path
    log_dir = _build_log_dir(Path(d))
    print(log_dir)
" 
```

Run the real CLI against that synthetic directory to confirm it executes end to end (adjust the
path printed above):

```bash
python -m srl.cli.verl_plot /tmp/<printed-path> --output-dir /tmp/verl_plot_smoke_test
```

Expected: prints the summary block and `Plot written to: /tmp/verl_plot_smoke_test/plots/instance_load.png`;
that file exists and is non-empty.

- [ ] **Step 5: Commit**

```bash
git add srl/cli/verl_plot.py pyproject.toml examples/visualize/plot_verl_run.sh
git commit -m "feat: add srl-verl-plot CLI and plot_verl_run.sh wrapper"
```

---

### Task 13: Write `examples/simulate/verl_aligned.sh`

**Files:**
- Create: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/examples/simulate/verl_aligned.sh`

**Interfaces:**
- Consumes: `srl.telemetry.verl_log.extract_workload_trace` (Task 11, invoked inline via `python -c`
  the same way `psrl_aligned.sh` invokes `extract_workload_trace` from `psrl_log`); the existing
  `examples/simulate/simulate.sh` and its `conf/*.sh` files; the existing `verl_v1` policy
  (`srl/policy/verl_v1.py`); the existing cost artifact
  `outputs/cost/qwen3_8b_tp2_pp3_thd/training_hydraulis.json`.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Simulate the real verl separate_async run this script is aligned against:
#   verl-new/examples/psrl_compare/qwen3_8b_separate_async_megatron.sh
#
# Every number below is transcribed from that script, mirroring how
# psrl_aligned.sh is written against qwen3_8b_megatron_simulate.sh. The two
# files describe one system, so a knob moved there has to move here too.
#
# Unlike psrl_aligned.sh, the real run's workload trace and telemetry are
# already structured JSON (verl-new's telemetry_sink.py), so this script's
# trace extraction calls srl.telemetry.verl_log instead of srl.telemetry.psrl_log.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VERL_LOG_DIR="${VERL_LOG_DIR:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/examples/psrl_compare/output/telemetry}"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B}"
TRACE="${TRACE:-${ROOT_DIR}/outputs/workload/verl_aligned/joint.jsonl}"
# rollout.n: responses per prompt. Both the trace's group arity and the
# simulator's group_size come from here so they cannot disagree.
export GROUP_SIZE="${GROUP_SIZE:-8}"

if [[ ! -f "${TRACE}" ]]; then
    if [[ ! -d "${VERL_LOG_DIR}" ]]; then
        echo "Missing both the cached trace and the verl telemetry directory." >&2
        echo "  trace:    ${TRACE}" >&2
        echo "  log dir:  ${VERL_LOG_DIR}" >&2
        echo "Point VERL_LOG_DIR at the run's VERL_TELEMETRY_LOG_DIR, or TRACE at a cached extract." >&2
        exit 1
    fi
    echo "Extracting the workload trace from ${VERL_LOG_DIR}..."
    python - "${VERL_LOG_DIR}" "${TRACE}" "${GROUP_SIZE}" <<'PYTRACE'
import sys
from srl.telemetry.verl_log import extract_workload_trace

extract_workload_trace(
    sys.argv[1], sys.argv[2], group_size=int(sys.argv[3]), max_output_tokens=10240
)
PYTRACE
fi

# --- transcribed from qwen3_8b_separate_async_megatron.sh --------------------
export BATCH_SIZE="${BATCH_SIZE:-32}"
export N_INSTANCES="${N_INSTANCES:-2}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-10240}"
export TOKEN_BUDGET="${TOKEN_BUDGET:-12800}"
export N_VERSIONS="${N_VERSIONS:-5}"
export POLICY="${POLICY:-verl_v1}"
export WORKLOAD=dapo_trace

# verl_v1 policy knobs, 1:1 with trainer.v1.separate_async / trainer.v1.sampler
# in the real launch script.
export NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
export MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-1}"
export MAX_OFF_POLICY_STRATEGY="${MAX_OFF_POLICY_STRATEGY:-wait}"

# Training mesh: tensor_model_parallel_size=2, pipeline_model_parallel_size=3,
# context_parallel_size=1, on 6 training GPUs, so DP=1. Identical to PSRL's
# mesh, so the same fitted cost artifact applies unchanged.
export TRAINING_BACKEND=hydraulis
export TRAINING_TP="${TRAINING_TP:-2}"
export TRAINING_PP="${TRAINING_PP:-3}"
export TRAINING_DP="${TRAINING_DP:-1}"
export TRAINING_CP="${TRAINING_CP:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-12800}"
export ACTIVATION_RECOMPUTE="${ACTIVATION_RECOMPUTE:-full}"
export TRAINING_ARTIFACT="${TRAINING_ARTIFACT:-${ROOT_DIR}/outputs/cost/qwen3_8b_tp2_pp3_thd/training_hydraulis.json}"

if [[ ! -f "${TRAINING_ARTIFACT}" ]]; then
    echo "Missing training cost artifact for the TP2/PP3 mesh: ${TRAINING_ARTIFACT}" >&2
    echo "This is the same artifact psrl_aligned.sh uses — build it there first if missing:" >&2
    echo "  CONFIG=examples/profile/megatron/qwen3_8b_thd.yaml CANDIDATE_INDEX=all \\" >&2
    echo "    bash examples/profile/megatron/profile_qwen3_8b.sh run" >&2
    echo "  bash examples/regress/regress_training_hydraulis.sh" >&2
    exit 1
fi

export KV_MODEL_PATH="${KV_MODEL_PATH:-${MODEL_PATH}}"
export KV_GPU_MEMORY_UTILIZATION="${KV_GPU_MEMORY_UTILIZATION:-0.9}"
export RECOMPUTE_MODEL="${RECOMPUTE_MODEL:-training_forward}"
export ROLLOUT_MODEL="${ROLLOUT_MODEL:-composed}"
export KERNEL_ROOT="${KERNEL_ROOT:-${ROOT_DIR}/outputs/profile/qwen3_8b/kernel_tp1/kernel}"
export GPU="${GPU:-h20}"
export MODEL_PATH

export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/simulate/verl_aligned}"
export EXTRAPOLATE="${EXTRAPOLATE:-[tokens,activation_bytes,pipeline_slots,sum_squared_tokens]}"
export EXTRA_OVERRIDES="workload.trace.path=${TRACE} policy.num_warmup_batches=${NUM_WARMUP_BATCHES} policy.max_off_policy_threshold=${MAX_OFF_POLICY_THRESHOLD} policy.max_off_policy_strategy=${MAX_OFF_POLICY_STRATEGY} ${EXTRA_OVERRIDES:-}"

bash "${SCRIPT_DIR}/simulate.sh"
```

- [ ] **Step 2: Confirm the policy override keys (already verified against the source)**

`policy.num_warmup_batches`/`policy.max_off_policy_threshold`/`policy.max_off_policy_strategy` is
the correct Hydra override path — confirmed against
`srl/config/defaults/policy/verl_v1.yaml`, which defines exactly these three fields (plus
`pause_mode`) directly under the `policy` config group:

```yaml
name: verl_v1
num_warmup_batches: 1.0
max_off_policy_threshold: 8
max_off_policy_strategy: drop
pause_mode: abort
```

No further check needed; proceed to Step 3.

- [ ] **Step 3: Dry-run against a synthetic trace (no real verl run needed yet)**

```bash
mkdir -p /tmp/verl_aligned_smoke/telemetry
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, 'tests')
from test_verl_log import _build_log_dir
_build_log_dir(Path('/tmp/verl_aligned_smoke/telemetry').parent)
"
mv /tmp/verl_aligned_smoke/telemetry/verl_telemetry/* /tmp/verl_aligned_smoke/telemetry/
VERL_LOG_DIR=/tmp/verl_aligned_smoke/telemetry GROUP_SIZE=8 N_VERSIONS=1 \
    TRACE=/tmp/verl_aligned_smoke/joint.jsonl \
    bash examples/simulate/verl_aligned.sh
```

Note: the synthetic fixture has 16 requests, not a multiple of `GROUP_SIZE=8` cleanly matching real
data — this step only proves the script's plumbing (file paths, env vars, Hydra override syntax) is
wired correctly, not that the numbers are meaningful. Expect it to either run to completion or fail
with a clear error about the trace/group arity; a Hydra `ConfigCompositionException` on an unknown
`policy.*` key is the one failure mode that means Step 2 needs revisiting.

- [ ] **Step 4: Commit**

```bash
chmod +x /apdcephfs_zwfy/share_303760348/lhy/SimRL/examples/simulate/verl_aligned.sh
git add examples/simulate/verl_aligned.sh
git commit -m "feat: add verl_aligned.sh, mirroring psrl_aligned.sh for the real verl run"
```

---

### Task 14: Write `examples/visualize/plot_simulate_verl_run.sh`

**Files:**
- Create: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/examples/visualize/plot_simulate_verl_run.sh`

**Interfaces:**
- Consumes: `examples/simulate/verl_aligned.sh`'s `output_dir` (Task 13); `srl.cli.plot` (existing,
  unmodified).

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Render the instance-load plot from a finished verl_aligned.sh simulation,
# drawn exactly the way plot_verl_run.sh draws the real run, so the two can
# be compared.
#
# This does not re-simulate. Point it at the output_dir of a prior
# verl_aligned.sh run.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SIM_RUN_DIR="${SIM_RUN_DIR:-${ROOT_DIR}/outputs/simulate/verl_aligned}"
TIME_LABEL="${TIME_LABEL:-Wall-clock Time (s)}"
FIGURE="${FIGURE:-${SCRIPT_DIR}/instance_load_verl_simulated.png}"

if [[ ! -d "${SIM_RUN_DIR}/logs" ]]; then
    echo "Missing simulation logs: ${SIM_RUN_DIR}/logs" >&2
    echo "Simulate first, for example: bash examples/simulate/verl_aligned.sh" >&2
    echo "Then point SIM_RUN_DIR at that run's output_dir." >&2
    exit 1
fi

python -m srl.cli.plot "${SIM_RUN_DIR}" \
    --only instance_load \
    --time-label "${TIME_LABEL}"

cp "${SIM_RUN_DIR}/plots/instance_load.png" "${FIGURE}"
echo "Figure copied to: ${FIGURE}"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x /apdcephfs_zwfy/share_303760348/lhy/SimRL/examples/visualize/plot_simulate_verl_run.sh
```

- [ ] **Step 3: Commit**

```bash
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
git add examples/visualize/plot_simulate_verl_run.sh
git commit -m "feat: add plot_simulate_verl_run.sh, mirroring plot_simulate_run.sh"
```

---

## Phase 4 — Compare and iterate

### Task 15: CHECKPOINT — run the alignment, compare, and document agreement

**Files:**
- Modify: `/apdcephfs_zwfy/share_303760348/lhy/SimRL/examples/simulate/verl_aligned.sh` (header
  comment, with measured results — same pattern `psrl_aligned.sh` already follows for itself).

**Interfaces:**
- Consumes: Task 10's real telemetry directory; Task 13's `verl_aligned.sh`; the existing
  `srl-compare` CLI (`simulated`, `real`, `--logs-subdir` positional/flag args, confirmed from
  `srl/cli/compare.py`).

- [ ] **Step 1: Run the real-run conversion and plot**

```bash
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
VERL_LOG_DIR=/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/examples/psrl_compare/output/telemetry \
    bash examples/visualize/plot_verl_run.sh
```

Expected: `outputs/visualize/verl_separate_async/` populated with `logs/` and `plots/instance_load.png`;
`examples/visualize/instance_load_verl_separate_async.png` created.

- [ ] **Step 2: Run the aligned simulation against the real trace**

```bash
GROUP_SIZE=8 bash examples/simulate/verl_aligned.sh
```

Expected: completes, writing `outputs/simulate/verl_aligned/`.

- [ ] **Step 3: Plot the simulated run**

```bash
bash examples/visualize/plot_simulate_verl_run.sh
```

Expected: `examples/visualize/instance_load_verl_simulated.png` created.

- [ ] **Step 4: Run the quantitative comparison**

```bash
srl-compare outputs/simulate/verl_aligned outputs/visualize/verl_separate_async --logs-subdir logs
```

Expected: prints the same class of agreement metrics `psrl_aligned.sh`'s own header documents for
PSRL (train/step time, peak/mean KV utilization, mean running requests, span).

- [ ] **Step 5: Document the measured agreement**

Edit `examples/simulate/verl_aligned.sh`'s header comment to add a block in the same style
`psrl_aligned.sh` uses for itself, e.g.:

```
# Current agreement (srl-compare against the run's own logs): <fill in the
# actual numbers from Step 4's output here — train/step, peak KV utilisation,
# mean KV utilisation, mean running requests, span>.
```

- [ ] **Step 6: Decide whether the staleness approximation needs adjusting**

If `srl-compare`'s numbers show the simulated run admits noticeably more or fewer in-flight
trajectories than the real run, revisit the `MAX_OFF_POLICY_THRESHOLD`/`MAX_OFF_POLICY_STRATEGY`
choice flagged as open in Task 4 and Task 13 — re-run Task 13's Step 1 (dry run becomes a real run)
with an adjusted value and repeat Steps 2-5 of this task.

- [ ] **Step 7: Commit**

```bash
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
git add examples/simulate/verl_aligned.sh examples/visualize/instance_load_verl_separate_async.png examples/visualize/instance_load_verl_simulated.png
git commit -m "docs: record measured agreement between the real and simulated verl separate_async runs"
```

---

## Self-Review Notes

- **Spec coverage**: Phase 1 (Tasks 1-5) covers the spec's repo/env/launch/cluster requirements.
  Phase 2 (Tasks 6-10) covers all three telemetry patches, using Ray's existing `snapshot()` RPC and
  `fit()`'s existing metrics dict instead of the spec's more speculative "explicit markers" language
  — a concrete simplification discovered by reading the real source, not a deviation from intent.
  Phase 3 (Tasks 11-14) covers the converter, CLI, and both scripts. Phase 4 (Task 15) covers
  `srl-compare` and the documented-agreement requirement. The spec's "gateway stream" open item is
  resolved by omission (Task 11 writes no `gateway` stream — `SimPlotter` already tolerates its
  absence, per `psrl_log.py`'s own `_write_gateway`).
- **Placeholder scan**: no TBD/TODO remain; the two explicitly-flagged uncertainties (checkpoint
  mcore converter CLI args, `verl_v1` policy's Hydra override prefix) were both resolved by reading
  the actual source (`scripts/converter_hf_to_mcore.py`'s argparse, and
  `srl/config/defaults/policy/verl_v1.yaml`) before this plan was finalized.
- **Type/name consistency**: `write_request_trace`, `build_stats_record`, `write_stats`,
  `maybe_start_stats_poller`, `write_step_metrics` are defined once in Task 6 and used with the same
  names and argument order in Tasks 7-9. `VerlLogLayout`, `ConversionSummary`, `convert_verl_logs`,
  `extract_workload_trace` are defined once in Task 11 and used identically in Tasks 12-13.
