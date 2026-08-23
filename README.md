# DryRun-RL

DryRun-RL is a discrete-event simulator for asynchronous, disaggregated RL training. It models rollout,
training, weight synchronization, policy staleness, GPU allocation, and parallelism choices without requiring a
full deployment for every candidate.

## Install

```bash
# Use conda to manage the environment
conda create -n dryrun-rl python=3.12

# Install pinned vLLM and Megatron-Bridge into the current environment
bash third_party/install.sh

# Install DryRun-RL
pip install -e ".[dev]"
```

`third_party/install.sh` reads pins from `third_party/pins.json`, installs the matching PyTorch CUDA
wheel first, then clones vLLM and Megatron-Bridge under `third_party/` and installs them editable into the
active conda environment. The source trees are not committed to this repository.

## Simulate

Hydra controls simulation configuration:

```bash
dryrun policy=psrl job.n_versions=20 rollout.n_instances=4 job.batch_size=8
dryrun job.max_prompt_length=8192 job.max_output_length=8192 workload=normal
dryrun policy=areal workload=bimodal job.n_versions=10
dryrun --multirun policy=psrl,areal,roll rollout.n_instances=1,2,4
```

The principal configuration groups are:

- `rollout_cost=roofline|psrl|distserve`
- `training_cost=fixed|linear|analytical|hydraulis`
- `policy=psrl|areal|roll|slime|verl`
- `workload=uniform|bimodal|lognormal|normal|powerlaw`

Fitted artifacts use explicit topology:

```bash
dryrun \
  rollout_cost=psrl \
  rollout_cost.artifact_path=outputs/cost/qwen3_8b/rollout_psrl.json \
  rollout_cost.parallelism.tp=1 \
  training_cost=hydraulis \
  training_cost.artifact_path=outputs/cost/qwen3_8b/training_hydraulis.json \
  training_cost.parallelism.tp=1 \
  training_cost.parallelism.pp=1 \
  training_cost.parallelism.dp=1 \
  training_cost.micro_batch_size=1
```

View the resolved configuration with `dryrun --cfg job`.

## Profile and regress

Profiling uses YAML workflows and official benchmark entrypoints:

```bash
dryrun-profile vllm generate --config examples/profile/vllm/qwen3_8b_tp1.yaml
dryrun-profile vllm run --config examples/profile/vllm/qwen3_8b_tp1.yaml --dry-run
dryrun-profile vllm collect --config examples/profile/vllm/qwen3_8b_tp1.yaml

dryrun-profile megatron generate --config examples/profile/megatron/qwen3_8b.yaml
dryrun-profile megatron run --config examples/profile/megatron/qwen3_8b.yaml --index 0 --dry-run
dryrun-profile megatron collect --config examples/profile/megatron/qwen3_8b.yaml --index 0
```

Official raw output is preserved. DryRun-RL writes versioned JSONL as the sole regression input.

Each cost model owns its fitting formula:

```bash
dryrun-regress rollout \
  --model psrl \
  --input outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl \
  --output outputs/cost/qwen3_8b/rollout_psrl.json

dryrun-regress training \
  --latency-model hydraulis \
  --memory-model hydraulis \
  --input outputs/profile/qwen3_8b/megatron_bridge/profiles/megatron_bridge.jsonl \
  --output outputs/cost/qwen3_8b/training_hydraulis.json
```

See `examples/profile/`, `examples/regress/`, and
[`docs/design/profiling-and-cost-models.md`](docs/design/profiling-and-cost-models.md) for coverage requirements,
formulas, applicability domains, and staged modeling work.

## Cost models

Rollout models:

- Unified roofline fits a low-load floor plus GEMM, prefill attention, and decode attention terms.
- PSRL fits separate prefill and decode attention and non-attention components.
- DistServe calibrates effective hardware constants from the same profile records and records single-model
  identifiability limits.

Training V1 follows the Hydraulis Appendix C decomposition:

```text
latency = a_q*Q_eff
        + b_t*T_eff
        + c_pipeline*(num_microbatches + PP - 1)
        + update_overhead
```

Peak memory combines parameter inventory with a fitted activation residual. TP and PP partition model state. DP
communication is idealized, while distributed optimizer sharding changes gradient and optimizer memory. CP other
than one, advanced pipeline schedules, and unprofiled discrete cells fail fast.

Artifacts include model and hardware identity, profile digest, explicit parallelism, coefficients, feature ranges,
grouped holdout metrics, outlier counts, and design rank.

## Architecture

The event loop is:

```text
admit -> generate -> complete -> select batch -> recompute -> train -> sync -> publish
```

Pure decode segments use closed-form integration of `tau(k) = max(F, alpha + beta*k)`, so simulation cost does not
grow with every generated token.

Core packages:

- `dryrun/component/rollout/` contains request state, continuous batching, KV accounting, and closed-form advance.
- `dryrun/cost/rollout/` contains roofline, PSRL, and DistServe inference models.
- `dryrun/cost/training/` contains workload, topology, parameter inventory, and Hydraulis latency and memory models.
- `dryrun/profiler/` contains the JSONL contract, constrained candidate generation, and official adapters.
- `dryrun/simulator/` drives the RL loop and emits latency and peak-memory telemetry.
- `dryrun/telemetry/` writes per-run JSONL streams (`instance`, `rollout_step`, `train`, `request`).
- `dryrun/visualizer/` renders offline plots from those JSONL streams.
- `dryrun/policy/` contains PSRL, AReaL, ROLL, Slime, and verl staleness policies.

## Programmatic API

```python
from dryrun.cost.rollout import UnifiedRoofline
from dryrun.cost.training import (
    FixedTrainingLatency,
    FixedTrainingMemory,
    TrainingCostModel,
)
from dryrun.policy.psrl import PSRLPolicy
from dryrun.simulator import SimConfig, Simulator
from dryrun.workload.distributions import bimodal

rollout_cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_d=1e-7)
training_cost = TrainingCostModel(
    FixedTrainingLatency(0.5),
    FixedTrainingMemory(0),
)
lengths = bimodal(200, short_len=50, long_len=500, long_frac=0.25, seed=42)
cfg = SimConfig(batch_size=8, max_staleness=2, n_versions=20, n_instances=4)

sim = Simulator(
    rollout_cost=rollout_cost,
    training_cost=training_cost,
    policy=PSRLPolicy(),
    lengths=lengths,
    cfg=cfg,
)
result = sim.run()
```

## Verify

GPU benchmarks and large editable installs are not unit tests. Local static and synthetic checks are:

```bash
python -m pytest
python -m ruff check .
bash -n third_party/install.sh examples/profile/**/*.sh examples/regress/*.sh
dryrun-profile --help
dryrun-regress --help
```
