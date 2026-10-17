# SimRL

SimRL is a discrete-event simulator for asynchronous, disaggregated RL post-training of large language
models. You profile a real system once, fit cost models from those measurements, then simulate configurations
you never ran (different parallelism, staleness policies, or rollout-to-training ratios) in seconds rather
than GPU-hours.

Its design premise is that a simulator is only useful if it knows what it does not know. Every cost model
records the range of inputs it was fitted over and refuses a query outside that range rather than extrapolating
silently.

## Install

```bash
conda create -n simrl python=3.12 -y && conda activate simrl
bash third_party/install.sh
pip install -e ".[dev]"
```

`third_party/install.sh` reads pins from `third_party/pins.json`, installs the matching PyTorch CUDA wheel
first, then clones vLLM and Megatron-Bridge under `third_party/` and installs them editable into the active
conda environment. It also builds flash-attn from source. The source trees are not committed here.

## Run a simulation

No GPU, no profiling data, and no checkpoint required:

```bash
srl policy=areal rollout_cost=roofline training_cost=fixed workload=uniform \
    job.n_versions=5 job.batch_size=32 rollout.n_instances=2
```

Telemetry and plots land under `outputs/` in the run directory.

## Documentation

Full documentation lives in [`docs/`](docs/) and covers the workload sources, all four profiling pipelines, the
cost models fitted from them, the simulator, and every CLI flag.

```bash
pip install -r docs/requirements.txt
make -C docs html
python -m http.server -d docs/_build/html 8000
```

Suggested entry points:

| Page | What it gives you |
|---|---|
| `docs/getting-started/quickstart.md` | A simulation running in ten minutes |
| `docs/getting-started/concepts.md` | The vocabulary the rest of the docs assume |
| `docs/profiling/index.md` | The four pipelines and what each sweep costs |
| `docs/reference/validation.md` | Every measured accuracy figure, with sample counts and held-out status |
| `docs/reference/limitations.md` | What is known not to work |

## Verify

```bash
pytest                     # 942 pass with MODEL_PATH set; 940 pass and 2 skip without it
ruff check .
make -C docs strict        # docs build with warnings promoted to errors
```
