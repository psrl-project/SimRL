# Installation

## Prerequisites

- Python 3.12 (enforced by `install.sh`, though 3.10 and 3.11 work for simulation-only use)
- A conda-compatible environment manager
- CUDA 12.8 compatible drivers, for profiling only. Simulation runs on CPU

---

## Environment setup

Create and activate a fresh conda environment before touching any of the install steps
below. Everything that follows assumes the environment is active.

```bash
conda create -n dryrun-rl python=3.12
conda activate dryrun-rl
```

All documented commands use **repo-relative paths** from the repository root, so begin
every session in that directory.

---

## Path convention used site-wide

This site uses two environment variables throughout its examples. Set them once per shell
session and every documented command will resolve correctly without editing:

```bash
conda activate dryrun-rl
export MODEL_PATH=/path/to/Qwen3-8B
export DATASET_PATH=/path/to/dapo-math-17k.parquet
```

Profile configs read them via OmegaConf's `${oc.env:MODEL_PATH,/path/to/Qwen3-8B}`
interpolation syntax, so exporting the variable is sufficient and no config file editing is
required. When a variable is not exported the default after the comma is used,
which is the `/path/to/…` placeholder shown in the source.

---

## Step 1: pinned third-party dependencies

`third_party/install.sh` installs the exact dependency versions this project was
developed against. It reads `third_party/pins.json`, which pins:

| Package | Pinned version |
|---|---|
| PyTorch | 2.11.0 (`cu128` index) |
| torchvision | 0.26.0 |
| torchaudio | 2.11.0 |
| transformers | 5.5.3 |
| flashinfer-python | 0.6.14 |
| vLLM | v0.26.0 (commit `568afb3`) |
| Megatron-Bridge | v0.4.2 (commit `c810129`) |
| flash-attn | 2.8.3.post1 |

Run the script from the repository root with the environment active:

```bash
bash third_party/install.sh
```

The script proceeds in eight numbered steps:

1. Installs `uv` and upgrades `pip`/`wheel`.
2. Installs the pinned PyTorch wheel from the CUDA 12.8 index.
3. Clones vLLM at the pinned commit and installs it with precompiled extensions
   (`VLLM_USE_PRECOMPILED=1`).
4. Clones Megatron-Bridge at the pinned commit with its `Megatron-LM` submodule.
5. Syncs Megatron-Bridge from its `uv.lock`, skipping packages that conflict with
   vLLM's requirements (`transformers`, `flashinfer-python`).
6. Re-pins `transformers` and `flashinfer` to the vLLM-compatible versions after the
   sync overwrites them.
7. Builds `flash-attn` from source (see note below).
8. Verifies that all eight packages import correctly.

The script is idempotent: if the clone directories already exist at the correct commit,
it skips the clone and proceeds to the install steps.

### A note on flash-attn and the source build

The prebuilt `flash-attn` wheels on PyPI link against `GLIBC_2.32`. Some hosts, including the
one this project was developed on, provide only `GLIBC_2.28`. The
package imports successfully on those hosts but fails to load its C extension at runtime,
which produces a silent correctness failure rather than a clear error.

`install.sh` therefore builds flash-attn from source unconditionally. The source build
is controlled by two environment variables:

```bash
MAX_JOBS="${FLASH_ATTENTION_MAX_JOBS:-8}" \
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTENTION_CUDA_ARCHS}"
```

`MAX_JOBS` is capped at 8 by default (overridable via `FLASH_ATTENTION_MAX_JOBS`). Each
`nvcc` invocation on flash-attn's kernels peaks near a gigabyte of RAM. A host that
reports hundreds of cores but has only a few GiB of free memory will have the build
killed by the OOM reaper, and the error log will be empty, because the process is terminated
before it can write anything. Keeping `MAX_JOBS` small prevents this. If the build
stalls on your machine and produces no output, reduce `MAX_JOBS` further:

```bash
FLASH_ATTENTION_MAX_JOBS=4 bash third_party/install.sh
```

The architecture list is restricted to the target GPU's `sm` number
(`FLASH_ATTENTION_CUDA_ARCHS` from `pins.json`, `90` for H20) rather than the six
architectures in the default sdist, which cuts the build time proportionally.

### Hosts with glibc < 2.39

`nvidia-resiliency-ext` 0.6.0 publishes only `manylinux_2_39` wheels. On a host with
glibc 2.28 the script detects this at install time, skips the locked wheel, and builds
`nvidia-resiliency-ext` from its GitHub tag instead. This is handled automatically.

---

## Step 2: DryRun-RL itself

Install DryRun-RL in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

This registers five console scripts:

| Script | Purpose |
|---|---|
| `dryrun` | Run a simulation |
| `dryrun-profile` | Profile a backend |
| `dryrun-regress` | Fit a cost model |
| `dryrun-plot` | Render offline plots |
| `dryrun-workload` | Workload utilities |

Verify the install:

```bash
dryrun --help
dryrun-profile --help
```

---

## Step 3: documentation toolchain (optional)

The docs toolchain is separate from the simulation dependencies and is only needed if
you want to build the HTML site locally.

Install it into the same environment:

```bash
pip install -r docs/requirements.txt
```

The requirements pin Sphinx, `pydata-sphinx-theme`, `myst-parser`,
`sphinx-copybutton`, `sphinx-design`, and `sphinxcontrib-mermaid`.

Build the HTML:

```bash
make -C docs html
```

The output lands in `docs/_build/html/`. Open `docs/_build/html/index.html` in a
browser to read it locally. The build treats Sphinx warnings as errors (`-W`), so a
broken cross-reference or a malformed MyST directive will abort the build with a clear
message.

---

## Quick verification (no GPU required)

Confirm the simulator itself works without profiling data or a GPU:

```bash
dryrun policy=areal rollout_cost=roofline training_cost=fixed workload=uniform \
    job.n_versions=5 job.batch_size=32 rollout.n_instances=2
```

If this prints a progress bar, a summary table, and writes files under `outputs/`, the
install is working. See {doc}`walkthrough/00-first-simulation` for what the output means and
where to go next.
