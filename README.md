<p align="center">
  <img src="assets/SimRL_logo_merge.png" alt="SimRL" width="560">
</p>

# SimRL

SimRL is a discrete-event simulator for asynchronous, disaggregated RL post-training of large language
models. You profile a real system once, fit cost models from those measurements, then simulate
configurations you never ran — a different parallelism layout, a different staleness policy, a different
rollout-to-training ratio — in seconds rather than GPU-hours.

A simulator is only useful if it knows what it does not know. Every cost model records the input range
it was fitted over and refuses a query outside that range instead of extrapolating silently.

## How it is built

Three concerns, three components, one documented seam between each pair:

| Stage | Component | What it does |
|---|---|---|
| Measure | `srl/profiler/` | Drives vLLM, Megatron-Bridge, FSDP2 or individual kernels and normalises observations into versioned JSONL. Adapters record; they never model. |
| Model | `srl/cost/` | Fits coefficients plus a validity domain from that JSONL: rollout cost (roofline, composed, distserve, psrl), training cost (fixed, analytical, linear, hydraulis, FSDP latency/memory), and a kernel database behind both. Models interpolate; they never measure. |
| Simulate | `srl/simulator/`, `srl/policy/`, `srl/component/` | Drives the rollout/training loop under one of twelve typed system presets, asking the cost models for latency and peak memory as it goes. |

The rollout engine has two interchangeable backends — a Python reference engine and a C++ engine selected
via `SIMRL_BACKEND` — documented in [`docs/simulation/backends.md`](docs/simulation/backends.md).

## Install

```bash
conda create -n simrl python=3.12 -y && conda activate simrl
bash third_party/install.sh
pip install -e ".[dev]"
```

`third_party/install.sh` reads pins from `third_party/pins.json`, installs the matching PyTorch CUDA wheel,
clones vLLM and Megatron-Bridge under `third_party/`, and builds flash-attn from source. The cloned source
trees are not committed. Simulation and cost fitting need no GPU of their own; profiling does.

## Run a simulation

No GPU, no profiling data, no checkpoint:

```bash
srl policy=areal rollout_cost=roofline training_cost=fixed workload=uniform \
    job.n_versions=5 job.batch_size=32 rollout.n_instances=2
```

Telemetry, the run report and plots land in the run directory under `outputs/`. To simulate a real system
preset instead of the toy `areal` configuration above, pick any of the presets in the table below and point
`rollout_cost`/`training_cost` at artifacts you produced with `srl-profile` and `srl-regress`.

## System presets

Each preset encodes one upstream system's scheduling semantics as a typed dataclass under
`srl/config/policies/`; the corresponding Hydra group is `srl/config/defaults/policy/`. Select one with
`policy=<name>`.

| Preset | Scheduling model | Primary knobs |
|---|---|---|
| `psrl` | Reserve / occupy / consume inventory | `staleness`, `proactive_filter` |
| `areal` | Cumulative token bucket | `max_head_offpolicyness` |
| `roll` | Async generation ratio | `async_generation_ratio` |
| `slime` | Loop barrier with over-sampling | `update_weights_interval`, `over_sampling_ratio`, `pause_mode` |
| `verl` | Sample-overage ratio and batch hand-off | `staleness_threshold`, `require_batches` |
| `verl_v1` | Step-clocked admission with an admission-span window | `num_warmup_batches`, `max_off_policy_threshold`, `max_off_policy_strategy`, `pause_mode` |
| `skyrl` | Capacity bound, warn-only staleness | `max_staleness_steps`, `pause_mode` |
| `openrlhf` | Generation permit pool | `async_queue_size`, `partial_rollout_enable` |
| `dressage` | Active/queue watermark, distinct-version window | `keep_versions`, `max_active_groups`, `output_queue_size` |
| `nemo_rl` | Sampler choice behind a shared admission gate | `sampler`, `max_staleness_versions`, `max_buffered_rollouts` |
| `trl` | Bounded task queue, version window checked at dequeue | `max_staleness`, `max_inflight_tasks`, `pause_mode` |
| `prime_rl` | Dispatch-lag admission, publish-time staleness sweep | `max_off_policy_steps`, `target_lag`, `action` |

[`docs/simulation/policies.md`](docs/simulation/policies.md) derives each preset's exact behaviour from its
source file.

## Command-line tools

| Command | Use |
|---|---|
| `srl` | Run a simulation (Hydra configuration). |
| `srl-profile` | Profile a real system: `kv-probe`, `vllm`, `kernels`, `megatron`, `fsdp`. |
| `srl-regress` | Fit rollout or training cost models from profiling JSONL. |
| `srl-workload` | Collect and validate joint grouped workload traces. |
| `srl-plot` | Re-render plots for an existing run without re-simulating. |
| `srl-psrl-plot` | Convert and plot a real PSRL run for comparison. |
| `srl-compare` | Quantify where a simulation disagrees with a real run. |

## Documentation

Full documentation is in [`docs/`](docs/) and builds with Sphinx:

```bash
pip install -r docs/requirements.txt
make -C docs html
python -m http.server -d docs/_build/html 8000
```

Suggested entry points:

| Page | What it gives you |
|---|---|
| `docs/getting-started/installation.md` | Environment and pinned third-party stack |
| `docs/getting-started/concepts.md` | The vocabulary the rest of the docs assume |
| `docs/getting-started/walkthrough/index.md` | One model, one machine, end to end |
| `docs/profiling/index.md` | The profiling pipelines and what each sweep cost |
| `docs/cost-models/index.md` | Every fitted model and its validity domain |
| `docs/simulation/backends.md` | Python vs C++ engine, and when the results differ |
| `docs/reference/limitations.md` | What is known not to work |

## Verify

```bash
pytest                     # tests needing MODEL_PATH are skipped without it
ruff check .
make -C docs strict        # warnings promoted to errors
```
