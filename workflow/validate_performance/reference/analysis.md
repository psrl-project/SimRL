# Reading results

## Where the numbers come from

| Source | What it holds |
|---|---|
| `SimResult` | `sim_time`, `consumed`, `dropped`, `versions_done`, `livelocked`, `train_timestamps`, `throughput`, `scg_curve()` |
| `sim.consumed` | every request that was trained on, each with `generated`, `admit_time`, `complete_time`, `instance_id`, `group_id`, `staleness_true()` |
| `logs/*.jsonl` | per-event telemetry, written when `log_telemetry` is on |
| `run.log` | the printed summary, including the throughput and staleness lines |

`SimResult` is defined in `srl/simulator/config.py`.

## Core metrics

- **speedup** = control `sim_time` / treatment `sim_time`. Only meaningful after the gates pass.
- **tokens_per_group** = `sum(request.generated for request in consumed) / number of distinct
  group_id`. This is the consumption-set gate.
- **mean concurrency** per instance: sweep the `(admit_time, complete_time)` intervals of each
  request on that instance. Two ready-made implementations exist,
  `headroom_probe._concurrency` and the reporting in `inventory_deadline_knobs.py`.
- **peak concurrency**: same sweep, taking the maximum.

## Reusable analysis entry points

Existing scripts under `examples/analysis/` read their own result JSONs and print tables. They are
fine to reuse, but none of them is required.

| Script | Reports |
|---|---|
| `srl-plot <run_dir>` | Regenerate `SimPlotter` plots from an existing run's `logs/` without re-simulating |
| `srl-compare <sim_dir> <real_dir>` | Time-weighted span, train/recompute/sync, running requests, KV util, idle fraction side by side |
| `srl-psrl-plot` | Convert a real PSRL run's logs into the simulator schema and plot it |
| `examples/analysis/sim_digest.py` | Hash every telemetry stream plus the C++ extension; `check --reference` fails on any divergence |
| `examples/analysis/diff_backends.py` | Row-identical comparison of the native and C++ engines over adversarial cases |
| `examples/analysis/headroom_probe.py report <grid>` | Per-point arms against the default and the static baselines |
| `examples/analysis/inventory_deadline_knobs.py report <grid>` | One knob at a time against shipped and best static |
| `examples/analysis/admission_externality.py` | Per-admission decision records, written to `/tmp/admission_externality` |
| `examples/analysis/forecast_accuracy.py` | Forecast error against realized buffer-ready times |

`docs/simulation/outputs.md` is the authoritative description of the telemetry streams and plots.

The scripts above are convenient, not mandatory. Any script that reads real simulator output is
acceptable. What is not acceptable is a performance number that came from somewhere other than a
real run.

## Result layout convention

Sweeps write `outputs/simulate/<experiment>/<arm-or-point>/`. Each run owns its `OUTPUT_DIR`, so an
arm never overwrites another's telemetry. Keep experiment-level summaries in
`outputs/simulate/<experiment>/summary.json`.

`outputs/` is gitignored. Never write results into `workflow/`.
