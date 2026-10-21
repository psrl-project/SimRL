# Installation

## Prerequisites

- Python 3.12. `third_party/install.sh` asserts it, and the C++ rollout engine shipped in the
  repository (`srl/_engine_cpp.cpython-312-*.so`) is built for CPython 3.12.
- A conda-compatible environment manager.
- For profiling only: CUDA 12.8 compatible drivers and the pinned stack from
  `third_party/install.sh`. Simulation itself runs on CPU.
- The `psrl` Python package. The simulator imports `psrl.workers.ps.staleness` when it is
  loaded, so `srl` fails to start without it, whichever `policy=` you select. It is not
  declared in `pyproject.toml`. The install command recorded in `srl/policy/psrl.py` is:

  ```bash
  pip install --no-deps -e /path/to/psrl --config-settings editable_mode=strict
  ```

---

## Environment setup

Create and activate a fresh conda environment before any install step. Everything below
assumes it is active, and every documented command is run from the repository root.

```bash
conda create -n simrl python=3.12
conda activate simrl
```

---

## Step 1: SimRL itself

```bash
pip install -e ".[dev]"
```

This installs the core dependencies (`hydra-core`, `omegaconf`, `numpy`, `scipy`,
`matplotlib`, `tqdm`) plus `pytest`, `pytest-xdist` and `ruff` from the `dev` extra. It
registers seven console scripts:

| Script | Purpose |
|---|---|
| `srl` | Run a simulation (Hydra entry point) |
| `srl-profile` | Profile a backend: `kv-probe`, `vllm`, `kernels`, `megatron`, `fsdp` |
| `srl-regress` | Fit a cost model from profiling JSONL: `rollout`, `training` |
| `srl-plot` | Re-render the plots of a finished run from its telemetry |
| `srl-psrl-plot` | Convert a real PSRL run's logs to telemetry and plot instance load |
| `srl-compare` | Quantify where a simulation disagrees with a converted real run |
| `srl-workload` | Collect and validate joint grouped workload traces |

`pyarrow` (parquet kernel tables) and `aiohttp` (trace collection) are imported by the code
but not declared in `pyproject.toml`. Install them if your environment lacks them.

Verify the install:

```bash
srl --help
srl-profile --help
```

`srl --help` lists the Hydra config groups (`policy`, `rollout_cost`, `training_cost`,
`workload`, plus the schema-only `sync_cost` and `recompute_cost`) and the resolved default
config.

---

## Step 2: pinned third-party stack (profiling only)

Skip this step if you only simulate. `third_party/install.sh` installs the exact versions the
profilers were developed against, read from `third_party/pins.json`:

| Package | Pin in `pins.json` |
|---|---|
| PyTorch / torchvision / torchaudio | 2.11.0 / 0.26.0 / 2.11.0 from the `cu128` index |
| transformers | 5.5.3 |
| flashinfer-python / flashinfer-cubin | 0.6.14 |
| vLLM | tag `v0.26.0`, commit `568afb3a` |
| Megatron-Bridge | tag `v0.4.2`, commit `c8101293`, with its pinned Megatron-LM submodule |
| flash-attn | 2.8.3.post1, built for `sm_90` |

Run it from the repository root with the environment active:

```bash
bash third_party/install.sh
```

The script clones vLLM and Megatron-Bridge into `third_party/` (ignored by git) and proceeds
in numbered steps:

0. Installs `uv` and upgrades `pip` and `wheel`.
1. Installs the pinned PyTorch wheels.
2. Clones vLLM at the pinned commit.
3. Installs vLLM with precompiled extensions against the existing PyTorch.
4. Clones Megatron-Bridge at the pinned commit, with submodules, and checks the Megatron-LM commit.
5. Installs Megatron-Bridge from its `uv.lock`, skipping `transformers` and `flashinfer-python`.
6. Re-pins `transformers` and `flashinfer` to the versions vLLM needs.
7. Builds flash-attn from source.
8. Imports every package to verify the install.

Re-running is safe: a clone already at the pinned commit is reused, and the script refuses to
move a checkout that has local modifications.

### flash-attn is always built from source

The prebuilt flash-attn wheels link against `GLIBC_2.32`. On a host with an older glibc the
package imports but its C extension fails to load, so `install.sh` builds from source
unconditionally and finishes by importing `flash_attn_2_cuda` explicitly. The build is
controlled by two variables:

```bash
FLASH_ATTENTION_MAX_JOBS=4 bash third_party/install.sh
```

`MAX_JOBS` defaults to 8. Each `nvcc` invocation can use about a gigabyte of RAM, so a host
that reports hundreds of cores but little free memory is killed by the OOM reaper with an
empty log. Lower the value if the build stalls without output.

### Hosts with glibc older than 2.39

`nvidia-resiliency-ext` 0.6.0 publishes only `manylinux_2_39` wheels. On an older host the
script detects this, skips the locked wheel, and installs `v0.6.0` from its GitHub tag.

---

## Paths and environment variables

The profiling configs read the checkpoint through OmegaConf interpolation, so exporting one
variable is enough and no config file needs editing:

```bash
export MODEL_PATH=/path/to/Qwen3-8B
```

An unset `MODEL_PATH` falls back to the literal `/path/to/Qwen3-8B` in the config, which will
not resolve. Variables the code reads:

| Variable | Read by | Meaning |
|---|---|---|
| `MODEL_PATH` | `examples/profile/**/*.yaml` | Checkpoint directory for profiling configs |
| `SIMRL_BACKEND` | `srl` | Rollout engine: `cpp` (default), `auto`, or `native` |
| `SIMRL_KV_CACHE` | `srl`, `srl-profile kv-probe` | KV-capacity cache file, default `~/.cache/srl/kv_capacity.json` |
| `SIMRL_LOG_LEVEL` | all CLIs | Level of the `srl` logger, default `INFO` |

The C++ engine runs by default. `SIMRL_BACKEND=auto` uses it when `srl._engine_cpp` imports and
falls back to the Python engine otherwise. `SIMRL_BACKEND=native` forces the Python engine.
{doc}`/pipeline/simulation/backends` explains the two engines.

---

## Step 3: documentation toolchain (optional)

```bash
pip install -r docs/requirements.txt
make -C docs html
```

The requirements are Sphinx, `pydata-sphinx-theme`, `myst-parser`, `sphinx-copybutton`,
`sphinx-design`, and `sphinxcontrib-mermaid`. Output lands in `docs/_build/html/`.
{doc}`/development/docs` describes the stricter `make -C docs strict` target.

---

## Quick verification (no GPU required)

```bash
srl policy=areal rollout_cost=roofline training_cost=fixed workload=uniform \
    job.n_versions=5 job.batch_size=32 rollout.n_instances=2
```

A progress bar, a summary block, and files under `outputs/` mean the install works.
{doc}`/getting-started/walkthrough/00-first-simulation` explains each of them.
