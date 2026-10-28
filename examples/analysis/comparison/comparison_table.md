# SimRL vs Frontier vs aiconfigurator vs LLMServingSim — comparison numbers

Workload: dapo_n64_cap40_kv1_normal (64 requests, Qwen3-8B, TP1, cap=40, H20)

## 1. Discrete event count

| system | events |
|---|---|
| Real vLLM (ground truth) | 36013 |
| SimRL (`run_sim.py`) | 6584 |
| Frontier (FlashInfer) | 36010 |
| Frontier (FA3) | 36010 |
| aiconfigurator | N/A — aiconfigurator has no discrete-event or trace-replay mode -- cli_estimate() is one closed-form steady-state formula evaluation, not a stepped simulation |
| LLMServingSim | 36010 |

## 2. Simulator overhead (wall-clock seconds to run the simulator itself)

| system | seconds |
|---|---|
| SimRL (`run_sim.py --backend native`) | 3.42 |
| SimRL (`run_sim.py --backend cpp`, project CLI default) | 2.56 |
| Frontier (FlashInfer, warm cache) | 86.19 |
| Frontier (FlashInfer, logging disabled) | 75.62 |
| Frontier (FA3, warm cache) | 82.77 |
| aiconfigurator (`cli_estimate()`) | 3.41 |
| LLMServingSim | 627.1 |

## 3. End-to-end accuracy (predicted span vs real span)

| system | span (s) | error vs real |
|---|---|---|
| Real vLLM | 656.98 | — |
| SimRL | 640.41 | -2.5% |
| Frontier (FlashInfer) | 470.51 | -28.4% |
| Frontier (FA3) | 770.27 | +17.2% |
| aiconfigurator (SOL, single-point) | 168.24 | -74.4% |
| LLMServingSim | 692.13 | +5.4% |

aiconfigurator's row is **not** a like-for-like replay prediction: not a replay prediction -- total_output_tokens / cli_estimate's steady-state tokens/s at one representative (isl=mean, osl=mean, batch_size=40) point; see README.
See the README's "How to run aiconfigurator yourself" section for the full derivation. LLMServingSim's
row is a genuine trace replay on the real workload, on a self-authored model config and real H20
profiling under a substituted vllm version, with skew calibration skipped: skipped skew (heterogeneous-decode-batch) calibration for time -- a disclosed fidelity tradeoff, not a structural limitation; see README.

## 4. Instance load

See `instance_load.png` (SimRL, Frontier FlashInfer, Frontier FA3, and real vLLM -- the original
3-system comparison) and `instance_load_4system_kv1.png` / `instance_load_4system_kv0.5.png`
(real vLLM, SimRL, Frontier FA3, LLMServingSim on one chart, kv1 and kv0.5 respectively --
LLMServingSim's curve is reconstructed from its per-request CSV). aiconfigurator has no
running/waiting-over-time series to plot at all, since `cli_estimate()` returns a single
steady-state point, not a trace.
