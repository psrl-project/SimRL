# Running the simulator

Two ways in. Use the in-process harness for parameter sweeps, the shell entry point for a single
production-shaped run. Both drive the same `Simulator`.

## Shell entry point

```bash
export SIMRL_WORKSPACE=<workspace dir holding env/ and SimRL/>   # required, no default
export SIMRL_KV_CACHE=$HOME/.cache/srl/kv_capacity.json           # measured cache, avoids re-probing
source "$SIMRL_WORKSPACE/env/simrl.sh"
cd "$SIMRL_WORKSPACE/SimRL"
OUTPUT_DIR=outputs/simulate/<experiment>/<arm> bash examples/simulate/simulate.sh
```

`simulate.sh` sources five files under `examples/simulate/conf/` and builds one Hydra override list.
Every concern is an env var.

| Env var | Meaning | Default |
|---|---|---|
| `ROLLOUT_MODEL` | `psrl` \| `distserve` \| `roofline` \| `composed` | `psrl` |
| `POLICY` | scheduling policy, must be `psrl` for inventory scheduling | `psrl` |
| `ROLLOUT_TP` | inference tensor-parallel degree | `1` |
| `GPU` | table lookup label, `h20` here | `h20` |
| `MODEL_PATH` / `KERNEL_ROOT` | checkpoint and kernel tables, composed only | 8B TP1 |
| `TRAINING_ARTIFACT` | fitted training artifact, omit for fixed latency | none |
| `TRAINING_LATENCY_S` | fixed training time when no artifact | `50` |
| `WORKLOAD` | `normal` \| `dapo_trace` \| ... | `normal` |
| `DIST_FILE` | `distributions.yaml` for the normal workload | repo default |
| `BATCH_SIZE` | prompt groups per step | `256` |
| `GROUP_SIZE` | responses per prompt, empty reads from the workload | none |
| `MAX_STALENESS` | psrl staleness cap | `2` |
| `MAX_PROMPT_LENGTH` / `MAX_OUTPUT_LENGTH` | per-sequence caps | `2048` / `16384` |
| `N_INSTANCES` | rollout instances | `4` |
| `TOKEN_BUDGET` | per-instance token budget | `65536` |
| `KV_MODEL_PATH` | checkpoint used for KV sizing, omit to use `kv_blocks` literal | none |
| `KV_PROBE` | start vLLM to measure KV capacity | `true` |
| `N_VERSIONS` | training steps | `20` |
| `OUTPUT_DIR` | logs and plots | `outputs/simulate/qwen3_8b_fitted_cost` |
| `EXTRA_OVERRIDES` | extra Hydra overrides, space separated | none |

Note `max_model_len` is not a knob. It is derived as `MAX_PROMPT_LENGTH + MAX_OUTPUT_LENGTH`.

## In-process harness

`examples/analysis/forecast_accuracy.py` owns the in-process path.

- `MediumRegime` declares one workload point.
- `_build_medium(regime, cost_model, mdt)` builds a `Simulator` from it.
- `_composed_cost_model()` loads the rollout cost model. It is hard-coded to Qwen3-8B TP1.
- `_load_dispersed_trace_module()` loads `examples/workload/make_dispersed_trace.py`, a truncated
  lognormal with `mean`, `cv`, `rho`, and a hard cap.

`examples/analysis/headroom_probe.py` builds arms on top of that, and `Point` is the workload
declaration most existing sweeps use. Reuse these rather than rebuilding the workload plumbing.

## Output layout

Every run writes into its `OUTPUT_DIR`:

| Path | Contents |
|---|---|
| `run.log` | stdout and structured log lines |
| `logs/*.jsonl` | telemetry, one stream per event type, written only when `log_telemetry` is on |
| `plots/` | plots produced by `SimPlotter` from the telemetry store |

The in-process path returns the `SimResult` directly. Its fields are in
`reference/analysis.md`.

## Speed and parallelism

An in-process arm takes seconds to a few minutes. A full point with a fresh six-value static cap
sweep plus LPT plus treatments takes tens of minutes. Run long sweeps in the background and use one
process per point.

## Known gaps

- Qwen3-8B TP4 and TP8 kernel trees carry collective tables only and are marked superseded. GEMM
  and attention lookups need their own sweep at that TP.
- The 14B tables exist but the checkpoint directory does not. Only the 8B and 35B MoE checkpoints
  are on disk.
- Qwen3.5-35B-A3B has two KV heads, so TP4 is impossible, and its TP2 tree has no collective
  tables.
- Attention tables cover step up to 32768 and input length up to 16384. Beyond that the utility
  surface holds at the boundary, so long-context results are trends, not accuracy claims.
- The measured KV capacity cache lives at `$HOME/.cache/srl/kv_capacity.json`. Point
  `SIMRL_KV_CACHE` at it or every run re-probes.
