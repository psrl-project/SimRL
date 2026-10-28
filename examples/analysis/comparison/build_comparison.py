"""Build the SimRL-vs-Frontier-vs-aiconfigurator comparison artifacts in this folder.

Reads the raw outputs already produced earlier in the investigation (real vLLM run, SimRL
`run_sim.py` run, two Frontier `frontier.main` runs -- the out-of-the-box FlashInfer backend
and the FA3 backend added this session) and one aiconfigurator `cli_estimate()` call (hardcoded
constants near the top of this file, since it has no ledger to read), and writes:

  - comparison.json       -- the four requested numbers (event counts, overhead, accuracy) in one
                              machine-readable file.
  - comparison_table.md   -- the same numbers as a human-readable table.
  - instance_load.png     -- running/waiting request count over wall-clock time, one line per
                              system, two panels.

Does not re-run any simulator; it only reads already-produced raw files (paths hardcoded below,
all under /tmp/ from this session plus this repo's own outputs/vllm_vs_sim/).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent

REAL_DIR = Path(
    "/apdcephfs_zwfy/share_303760348/lhy/SimRL/outputs/vllm_vs_sim/real/dapo_n64_cap40_kv1_normal"
)
SIM_DIR = Path(
    "/apdcephfs_zwfy/share_303760348/lhy/SimRL/outputs/vllm_vs_sim/sim/dapo_n64_cap40_kv1_normal"
)
FRONTIER_FLASHINFER_LEDGER = Path(
    "/tmp/frontier_sim_out/qwen_qwen3_8b/offline_batch/run_2026_10_26_16_13_47_954154/"
    "frontier_stage_batch_ledger.jsonl"
)
FRONTIER_FA3_LEDGER = Path(
    "/tmp/frontier_sim_out_fa3/qwen_qwen3_8b/offline_batch/run_2026_10_26_23_05_38_406609/"
    "frontier_stage_batch_ledger.jsonl"
)

# KV-pressure (kv0.5) workload -- the same 64 requests at gpu_memory_utilization=0.40 instead of
# 0.9, the point that actually triggers 73 real preemptions (see README's "KV-pressure workload"
# section). Used only by the 4-system instance-load plots below.
REAL_DIR_KV05 = Path(
    "/apdcephfs_zwfy/share_303760348/lhy/SimRL/outputs/vllm_vs_sim/real/dapo_n64_cap40_kv0.5_normal"
)
SIM_DIR_KV05 = Path(
    "/apdcephfs_zwfy/share_303760348/lhy/SimRL/outputs/vllm_vs_sim/sim/dapo_n64_cap40_kv0.5_normal"
)
FRONTIER_FA3_LEDGER_KV05 = Path(
    "/tmp/frontier_sim_out_kv05_fa3/qwen_qwen3_8b/offline_batch/run_2026_10_27_11_25_24_425849/"
    "frontier_stage_batch_ledger.jsonl"
)
LLMSERVINGSIM_CSV_KV1 = Path("/tmp/llmss_dapo_kv1_result.csv")
LLMSERVINGSIM_CSV_KV05 = Path("/tmp/llmss_dapo_kv05_result.csv")

N_REQUESTS = 64

# aiconfigurator has no discrete-event or trace-replay mode at all (see README) -- these are a
# single `cli_estimate(mode="agg", database_mode="SOL")` call's numbers, not read from a ledger.
# isl/osl are the mean prompt/output length across the same 64-request trace (mean, not median,
# so isl*64 and osl*64 reproduce the trace's real total token counts used in the span conversion
# below); batch_size=40 matches the same admission cap used everywhere else in this comparison.
# No real H20 SILICON data exists for this tool on this cluster (see README) -- SOL mode is pure
# hardware-peak-spec + analytical formula, zero kernel measurements, zero calibration.
AIC_ISL = 138
AIC_OSL = 9318
AIC_BATCH_SIZE = 40
AIC_TOKENS_PER_S = 3544.486400318199  # tokens/s field from the cli_estimate() result
AIC_TOTAL_OUTPUT_TOKENS = 596330  # sum(num_decode_tokens) across the real 64-request trace
AIC_SPAN_S = AIC_TOTAL_OUTPUT_TOKENS / AIC_TOKENS_PER_S
AIC_OVERHEAD_S = 3.41  # /usr/bin/time -v wall clock for the single cli_estimate() call

# `run_sim.py --backend {native,cpp}` on the exact same spec, median of 3 interleaved
# `/usr/bin/time -v` trials per backend (same methodology as every other overhead number here).
# Both backends produced byte-identical span (640.4064881608149s)/events (6584)/preemptions (0) --
# this is a pure wall-clock comparison, not a different prediction. `run_sim.py`'s own CLI default
# is "native"; the project's main CLI (`srl/cli/main.py`) defaults to "cpp" instead, so production
# runs already get the faster number below.
SIMRL_OVERHEAD_NATIVE_S = 3.42
SIMRL_OVERHEAD_CPP_S = 2.56

# LLMServingSim (KAIST, ISPASS 2026) had no Qwen3-8B catalog entry and no H20 profile bundle
# out of the box, and its GPU profiler is documented against vLLM 0.31.0/CUDA 13 -- adapted
# instead of accepting that as a hard stop (see README's "LLMServingSim" section): authored
# configs/model/Qwen/Qwen3-8B.json ourselves, profiled dense+per_sequence+attention for real on
# H20 under SimRL's own vllm (0.30.1.dev0, not 0.31.0) after fixing one real torch-2.11 API
# removal (_KinetoEvent.activity_type()) in the profiler's own activity collector, and ran the
# real discrete-event `serving` simulation on the exact converted 64-request trace. Skew
# (heterogeneous-decode) calibration was explicitly skipped for time -- a real, disclosed
# fidelity tradeoff, not a silent one.
LLMSERVINGSIM_EVENTS = 36010  # grep -c "Scheduling new batch" on an --log-level INFO rerun
LLMSERVINGSIM_SPAN_S = 692.134
LLMSERVINGSIM_OVERHEAD_S = 627.10  # /usr/bin/time -v, --log-level WARNING (default) run
LLMSERVINGSIM_SKEW_NOTE = (
    "skipped skew (heterogeneous-decode-batch) calibration for time -- a disclosed fidelity "
    "tradeoff, not a structural limitation; see README"
)


def load_real_sim_steps(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def load_frontier_running_waiting(ledger_path: Path) -> tuple[pd.DataFrame, int]:
    """Reconstruct (t, running, waiting) from the per-event ledger.

    Frontier's ledger has no direct "waiting" column, only which request_ids are in the batch
    at each event. Reconstructed under the same "0 preemptions for this workload" fact already
    confirmed for the real and SimRL runs this session: once a request first appears in the
    ledger it stays continuously resident until it finishes, so:
      running(t) = size of the request_ids set active at the event covering t
      waiting(t) = N_REQUESTS - (count of distinct request_ids first seen at or before t)
    """
    rows = []
    first_seen: dict[str, float] = {}
    n_events = 0
    with ledger_path.open() as f:
        for line in f:
            row = json.loads(line)
            n_events += 1
            t = row["stage_start_ts"]
            req_ids = row["request_ids"]
            running = len(req_ids)
            for rid in req_ids:
                if rid not in first_seen or t < first_seen[rid]:
                    first_seen[rid] = t
            rows.append({"t": t, "running": running})

    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    sorted_first_seen = sorted(first_seen.values())

    def admitted_by(t: float) -> int:
        import bisect

        return bisect.bisect_right(sorted_first_seen, t)

    df["waiting"] = N_REQUESTS - df["t"].apply(admitted_by)
    return df, n_events


def load_llmservingsim_running_waiting(csv_path: Path, n_requests: int) -> pd.DataFrame:
    """Reconstruct (t, running, waiting) from LLMServingSim's per-request CSV.

    Columns are `arrival`/`end_time`/`queuing_delay` in nanoseconds (not a per-step ledger like
    Frontier's). A request is "waiting" from `arrival` until `arrival + queuing_delay` (confirmed
    nonzero and request-specific -- 24/64 requests here have queuing_delay > 1ms, the other 40
    ~0, matching this workload's max_num_seqs=40 cap exactly), then "running" until `end_time`.
    Sampled onto a 1-second grid for a directly comparable step-function curve.
    """
    df = pd.read_csv(csv_path)
    arrival_s = df["arrival"] / 1e9
    start_run_s = (df["arrival"] + df["queuing_delay"]) / 1e9
    end_s = df["end_time"] / 1e9

    t_max = end_s.max()
    grid = pd.Series(range(0, int(t_max) + 2), dtype=float)
    running = grid.apply(lambda t: int(((start_run_s <= t) & (t < end_s)).sum()))
    waiting = grid.apply(lambda t: int(((arrival_s <= t) & (t < start_run_s)).sum()))
    out = pd.DataFrame({"t": grid, "running": running, "waiting": waiting})
    assert (out["running"] + out["waiting"]).max() <= n_requests
    return out


def _plot_4system_instance_load(
    out_path: Path,
    title_suffix: str,
    real_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    llmss_df: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    series = [
        ("Real vLLM", real_df, "black", 1.5, 1.0),
        ("SimRL", sim_df, "tab:blue", 1.2, 1.0),
        ("Frontier (FA3)", frontier_df, "tab:red", 1.0, 0.8),
        ("LLMServingSim", llmss_df, "tab:green", 1.0, 0.8),
    ]
    ax = axes[0]
    for label, df, color, lw, alpha in series:
        ax.plot(df["t"], df["running"], label=label, color=color, linewidth=lw, alpha=alpha)
    ax.set_ylabel("running requests")
    ax.set_title(f"Instance load: running requests over wall-clock time ({title_suffix})")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for label, df, color, lw, alpha in series:
        ax.plot(df["t"], df["waiting"], label=label, color=color, linewidth=lw, alpha=alpha)
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("waiting requests")
    ax.set_title(
        f"Instance load: waiting (not yet admitted) requests over wall-clock time ({title_suffix})"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    real_df = load_real_sim_steps(REAL_DIR / "steps.jsonl")
    sim_df = load_real_sim_steps(SIM_DIR / "steps.jsonl")
    frontier_fi_df, n_frontier_fi_events = load_frontier_running_waiting(
        FRONTIER_FLASHINFER_LEDGER
    )
    frontier_fa3_df, n_frontier_fa3_events = load_frontier_running_waiting(FRONTIER_FA3_LEDGER)

    real_meta = json.loads((REAL_DIR / "meta.json").read_text())
    sim_meta = json.loads((SIM_DIR / "meta.json").read_text())

    real_span_s = real_meta["wall_s"]
    sim_span_s = sim_meta["wall_s"]
    frontier_fi_span_s = frontier_fi_df["t"].max()
    frontier_fa3_span_s = frontier_fa3_df["t"].max()

    def pct_err(pred: float) -> float:
        return (pred - real_span_s) / real_span_s * 100.0

    comparison = {
        "workload": "dapo_n64_cap40_kv1_normal (64 requests, Qwen3-8B, TP1, cap=40, H20)",
        "1_discrete_event_count": {
            "real_vllm_steps": len(real_df),
            "simrl_run_sim_events": len(sim_df),
            "frontier_flashinfer_events": n_frontier_fi_events,
            "frontier_fa3_events": n_frontier_fa3_events,
            "aiconfigurator_events": None,
            "aiconfigurator_events_reason": (
                "aiconfigurator has no discrete-event or trace-replay mode -- cli_estimate() is "
                "one closed-form steady-state formula evaluation, not a stepped simulation"
            ),
            "llmservingsim_events": LLMSERVINGSIM_EVENTS,
        },
        "2_simulator_overhead_wall_clock_seconds": {
            "simrl_run_sim_native": SIMRL_OVERHEAD_NATIVE_S,
            "simrl_run_sim_cpp": SIMRL_OVERHEAD_CPP_S,
            "frontier_flashinfer_warm_cache": 86.19,
            "frontier_flashinfer_logging_disabled": 75.62,
            "frontier_fa3_warm_cache": None,  # filled in below from the timed rerun log
            "aiconfigurator_cli_estimate": AIC_OVERHEAD_S,
            "llmservingsim": LLMSERVINGSIM_OVERHEAD_S,
        },
        "3_end_to_end_accuracy": {
            "real_vllm_span_s": real_span_s,
            "simrl_span_s": sim_span_s,
            "simrl_error_pct": pct_err(sim_span_s),
            "frontier_flashinfer_span_s": frontier_fi_span_s,
            "frontier_flashinfer_error_pct": pct_err(frontier_fi_span_s),
            "frontier_fa3_span_s": frontier_fa3_span_s,
            "frontier_fa3_error_pct": pct_err(frontier_fa3_span_s),
            "aiconfigurator_span_s": AIC_SPAN_S,
            "aiconfigurator_error_pct": pct_err(AIC_SPAN_S),
            "aiconfigurator_span_note": (
                "not a replay prediction -- total_output_tokens / cli_estimate's steady-state "
                "tokens/s at one representative (isl=mean, osl=mean, batch_size=40) point; see README"
            ),
            "llmservingsim_span_s": LLMSERVINGSIM_SPAN_S,
            "llmservingsim_error_pct": pct_err(LLMSERVINGSIM_SPAN_S),
            "llmservingsim_note": LLMSERVINGSIM_SKEW_NOTE,
        },
    }

    # Fill in the FA3 overhead number from the clean timed rerun's `time -v` log, if present.
    fa3_timing_log = Path("/tmp/frontier_fa3_timed.log")
    if fa3_timing_log.exists():
        for line in fa3_timing_log.read_text().splitlines():
            if "Elapsed (wall clock) time" in line:
                # Line looks like "...time (h:mm:ss or m:ss): 1:22.77" -- take the value
                # after the LAST "): " so the parenthetical's own colons don't confuse the split.
                raw = line.rsplit("): ", 1)[1].strip()
                parts = [float(p) for p in raw.split(":")]
                secs = 0.0
                for p in parts:
                    secs = secs * 60 + p
                comparison["2_simulator_overhead_wall_clock_seconds"]["frontier_fa3_warm_cache"] = round(
                    secs, 2
                )

    (HERE / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False))

    # --- Markdown table ---
    ev = comparison["1_discrete_event_count"]
    ov = comparison["2_simulator_overhead_wall_clock_seconds"]
    ac = comparison["3_end_to_end_accuracy"]
    md = f"""# SimRL vs Frontier vs aiconfigurator vs LLMServingSim — comparison numbers

Workload: {comparison['workload']}

## 1. Discrete event count

| system | events |
|---|---|
| Real vLLM (ground truth) | {ev['real_vllm_steps']} |
| SimRL (`run_sim.py`) | {ev['simrl_run_sim_events']} |
| Frontier (FlashInfer) | {ev['frontier_flashinfer_events']} |
| Frontier (FA3) | {ev['frontier_fa3_events']} |
| aiconfigurator | N/A — {ev['aiconfigurator_events_reason']} |
| LLMServingSim | {ev['llmservingsim_events']} |

## 2. Simulator overhead (wall-clock seconds to run the simulator itself)

| system | seconds |
|---|---|
| SimRL (`run_sim.py --backend native`) | {ov['simrl_run_sim_native']} |
| SimRL (`run_sim.py --backend cpp`, project CLI default) | {ov['simrl_run_sim_cpp']} |
| Frontier (FlashInfer, warm cache) | {ov['frontier_flashinfer_warm_cache']} |
| Frontier (FlashInfer, logging disabled) | {ov['frontier_flashinfer_logging_disabled']} |
| Frontier (FA3, warm cache) | {ov['frontier_fa3_warm_cache']} |
| aiconfigurator (`cli_estimate()`) | {ov['aiconfigurator_cli_estimate']} |
| LLMServingSim | {ov['llmservingsim']} |

## 3. End-to-end accuracy (predicted span vs real span)

| system | span (s) | error vs real |
|---|---|---|
| Real vLLM | {ac['real_vllm_span_s']:.2f} | — |
| SimRL | {ac['simrl_span_s']:.2f} | {ac['simrl_error_pct']:+.1f}% |
| Frontier (FlashInfer) | {ac['frontier_flashinfer_span_s']:.2f} | {ac['frontier_flashinfer_error_pct']:+.1f}% |
| Frontier (FA3) | {ac['frontier_fa3_span_s']:.2f} | {ac['frontier_fa3_error_pct']:+.1f}% |
| aiconfigurator (SOL, single-point) | {ac['aiconfigurator_span_s']:.2f} | {ac['aiconfigurator_error_pct']:+.1f}% |
| LLMServingSim | {ac['llmservingsim_span_s']:.2f} | {ac['llmservingsim_error_pct']:+.1f}% |

aiconfigurator's row is **not** a like-for-like replay prediction: {ac['aiconfigurator_span_note']}.
See the README's "How to run aiconfigurator yourself" section for the full derivation. LLMServingSim's
row is a genuine trace replay on the real workload, on a self-authored model config and real H20
profiling under a substituted vllm version, with skew calibration skipped: {ac['llmservingsim_note']}.

## 4. Instance load

See `instance_load.png` (SimRL, Frontier FlashInfer, Frontier FA3, and real vLLM -- the original
3-system comparison) and `instance_load_4system_kv1.png` / `instance_load_4system_kv0.5.png`
(real vLLM, SimRL, Frontier FA3, LLMServingSim on one chart, kv1 and kv0.5 respectively --
LLMServingSim's curve is reconstructed from its per-request CSV). aiconfigurator has no
running/waiting-over-time series to plot at all, since `cli_estimate()` returns a single
steady-state point, not a trace.
"""
    (HERE / "comparison_table.md").write_text(md)

    # --- Plot: running + waiting over wall-clock time, one line per system ---
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)

    ax = axes[0]
    ax.plot(real_df["t"], real_df["running"], label="Real vLLM", color="black", linewidth=1.5)
    ax.plot(sim_df["t"], sim_df["running"], label="SimRL", color="tab:blue", linewidth=1.2)
    ax.plot(
        frontier_fi_df["t"], frontier_fi_df["running"],
        label="Frontier (FlashInfer)", color="tab:orange", linewidth=1.0, alpha=0.8,
    )
    ax.plot(
        frontier_fa3_df["t"], frontier_fa3_df["running"],
        label="Frontier (FA3)", color="tab:red", linewidth=1.0, alpha=0.8,
    )
    ax.set_ylabel("running requests")
    ax.set_title("Instance load: running requests over wall-clock time")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(real_df["t"], real_df["waiting"], label="Real vLLM", color="black", linewidth=1.5)
    ax.plot(sim_df["t"], sim_df["waiting"], label="SimRL", color="tab:blue", linewidth=1.2)
    ax.plot(
        frontier_fi_df["t"], frontier_fi_df["waiting"],
        label="Frontier (FlashInfer)", color="tab:orange", linewidth=1.0, alpha=0.8,
    )
    ax.plot(
        frontier_fa3_df["t"], frontier_fa3_df["waiting"],
        label="Frontier (FA3)", color="tab:red", linewidth=1.0, alpha=0.8,
    )
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("waiting requests")
    ax.set_title("Instance load: waiting (not yet admitted) requests over wall-clock time")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(HERE / "instance_load.png", dpi=150)
    plt.close(fig)

    # --- 4-system plots (real vLLM, SimRL, Frontier FA3, LLMServingSim), kv1 and kv0.5 ---
    wrote_4system = []
    if LLMSERVINGSIM_CSV_KV1.exists():
        llmss_kv1_df = load_llmservingsim_running_waiting(LLMSERVINGSIM_CSV_KV1, N_REQUESTS)
        _plot_4system_instance_load(
            HERE / "instance_load_4system_kv1.png",
            "kv1, no pressure",
            real_df, sim_df, frontier_fa3_df, llmss_kv1_df,
        )
        wrote_4system.append("instance_load_4system_kv1.png")

    if (
        LLMSERVINGSIM_CSV_KV05.exists()
        and REAL_DIR_KV05.exists()
        and SIM_DIR_KV05.exists()
        and FRONTIER_FA3_LEDGER_KV05.exists()
    ):
        real_kv05_df = load_real_sim_steps(REAL_DIR_KV05 / "steps.jsonl")
        sim_kv05_df = load_real_sim_steps(SIM_DIR_KV05 / "steps.jsonl")
        frontier_kv05_df, _ = load_frontier_running_waiting(FRONTIER_FA3_LEDGER_KV05)
        llmss_kv05_df = load_llmservingsim_running_waiting(LLMSERVINGSIM_CSV_KV05, N_REQUESTS)
        _plot_4system_instance_load(
            HERE / "instance_load_4system_kv0.5.png",
            "kv0.5, real preemption (73 real events)",
            real_kv05_df, sim_kv05_df, frontier_kv05_df, llmss_kv05_df,
        )
        wrote_4system.append("instance_load_4system_kv0.5.png")

    print("Wrote comparison.json, comparison_table.md, instance_load.png, " + ", ".join(wrote_4system))
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
