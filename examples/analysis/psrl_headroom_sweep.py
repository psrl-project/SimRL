#!/usr/bin/env python
"""Where does scheduling have headroom under the psrl policy?

Under `policy=psrl` nothing is dropped and staleness is bounded by construction,
so the only quantity a scheduler can change is how fast the run finishes. This
sweeps the knobs that decide whether it can:

    staleness    how far ahead the fleet may generate
    batch size   prompt groups per step, against what the fleet holds in flight
    training     seconds per step, which sets whether rollout or training binds
    dispersion   spread of response lengths at a fixed mean, and how much of it
                 belongs to the prompt rather than the sample
    fleet size   rollout instances

and compares schedulers that differ only in what they are told about length.
`fifo` is the router as it ships. `lpt` and `spt` order whole prompt groups
longest- or shortest-first using the trace's true lengths, which is the most any
predictor could offer, so a regime where they do not beat `fifo` is a regime where
no length model can either.

Training is a fixed latency rather than a fitted artifact: the point is to move
the ratio of rollout time to training time, and a fitted cost model ties the two
together through the batch.

Usage:
    python examples/analysis/psrl_headroom_sweep.py run pilot
    python examples/analysis/psrl_headroom_sweep.py report pilot
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs" / "simulate" / "psrl_headroom"
SCRATCH_ROOT = Path("/tmp/psrl_headroom_runs")
TRACE_ROOT = ROOT / "outputs" / "workload" / "dispersed"
MODEL_PATH = "/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B"
KERNEL_ROOT = ROOT / "outputs" / "profile" / "qwen3_8b" / "kernel_tp1" / "kernel"


@dataclass(frozen=True)
class Regime:
    """Everything that defines the system, with the scheduler left out."""

    cv: float = 0.5
    rho: float = 0.6
    mean: int = 8000
    cap: int = 32768
    batch: int = 32
    staleness: int = 2
    train_s: float = 345.0
    instances: int = 2
    versions: int = 6
    group_size: int = 8
    trace: str = ""
    """A recorded trace to use instead of a synthetic one; `cv`, `rho` and `mean` are then ignored."""

    def key(self) -> str:
        shape = (
            Path(self.trace).parent.name
            if self.trace
            else f"cv{self.cv:g}_rho{self.rho:g}_m{self.mean}_c{self.cap}"
        )
        return f"{shape}_B{self.batch}_S{self.staleness}_tau{self.train_s:g}_I{self.instances}"


@dataclass(frozen=True)
class Arm:
    """The scheduler: an order, a placement rule and a per-instance cap."""

    order: str = "fifo"
    balance: str = "count"
    active_cap: int = 50
    window_groups: int = 0
    sigma: float = 0.0
    seed: int = 0
    watermark: float = 0.0

    def key(self) -> str:
        base = f"{self.order}-{self.balance}-cap{self.active_cap}-w{self.window_groups}"
        if self.sigma > 0.0:
            base += f"-noise{self.sigma:g}s{self.seed}"
        if self.watermark > 0.0:
            base += f"-wm{self.watermark:g}"
        return base


@dataclass
class Job:
    grid: str
    regime: Regime
    arm: Arm
    tags: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.regime.key()}__{self.arm.key()}"

    @property
    def run_dir(self) -> Path:
        """Telemetry lives on local disk: a run writes tens of MB to 2 GB of it,
        and the shared filesystem the repo sits on does not have room for a sweep."""
        return SCRATCH_ROOT / self.grid / self.name

    @property
    def result_path(self) -> Path:
        return OUT_ROOT / self.grid / f"{self.name}.json"


def ensure_trace(regime: Regime) -> Path:
    if regime.trace:
        return Path(regime.trace)
    n_groups = regime.batch * (regime.versions + regime.staleness + 2)
    path = (
        TRACE_ROOT
        / f"cv{regime.cv:g}_rho{regime.rho:g}_m{regime.mean}_c{regime.cap}_g{n_groups}.jsonl"
    )
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "workload" / "make_dispersed_trace.py"),
            "--out", str(partial),
            "--n-groups", str(n_groups),
            "--group-size", str(regime.group_size),
            "--mean", str(regime.mean),
            "--cv", str(regime.cv),
            "--rho", str(regime.rho),
            "--cap", str(regime.cap),
        ],
        check=True,
        capture_output=True,
    )
    os.replace(partial, path)
    return path


def build_env(job: Job) -> dict[str, str]:
    r, a = job.regime, job.arm
    trace = ensure_trace(r)
    overrides = [
        f"workload.trace.path={trace}",
        "rollout.max_concurrency=1024",
        f"rollout.router.max_concurrent_seqs_per_instance={a.active_cap}",
        "rollout.router.max_waiting_after_preempt=1024",
        "rollout.router.reject_preempted_on_waiting=true",
    ]
    if a.order != "fifo" or a.balance != "count" or a.watermark > 0.0:
        overrides += [
            "rollout.router.strategy=length_aware",
            "rollout.router.queue_window=100000",
            f"+rollout.router.strategy_args.order={a.order}",
            f"+rollout.router.strategy_args.balance={a.balance}",
            f"+rollout.router.strategy_args.window_groups={a.window_groups}",
            f"+rollout.router.strategy_args.kv_watermark={a.watermark}",
        ]
        if a.sigma > 0.0:
            overrides += [
                "+rollout.router.strategy_args.length_model=noisy_oracle",
                f"+rollout.router.strategy_args.length_model_args.sigma={a.sigma}",
                f"+rollout.router.strategy_args.length_model_args.seed={a.seed}",
            ]
    env = dict(os.environ)
    env.update(
        DRYRUN_BACKEND="native",
        POLICY="psrl",
        WORKLOAD="dapo_trace",
        MODEL_PATH=MODEL_PATH,
        KV_MODEL_PATH=MODEL_PATH,
        KV_GPU_MEMORY_UTILIZATION="0.9",
        ROLLOUT_MODEL="composed",
        KERNEL_ROOT=str(KERNEL_ROOT),
        GPU="h20",
        ROLLOUT_TP="1",
        GROUP_SIZE=str(r.group_size),
        BATCH_SIZE=str(r.batch),
        N_INSTANCES=str(r.instances),
        N_VERSIONS=str(r.versions),
        MAX_STALENESS=str(r.staleness),
        MAX_PROMPT_LENGTH="1024",
        MAX_OUTPUT_LENGTH=str(r.cap),
        TOKEN_BUDGET="12800",
        TRAINING_LATENCY_S=str(r.train_s),
        OUTPUT_DIR=str(job.run_dir),
        EXTRA_OVERRIDES=" ".join(overrides),
    )
    env.pop("TRAINING_ARTIFACT", None)
    env.pop("RECOMPUTE_MODEL", None)
    return env


def run_job(job: Job) -> dict:
    metrics_path = job.result_path
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text())
    job.run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    log = job.run_dir / "run.log"
    with log.open("w") as handle:
        proc = subprocess.run(
            ["bash", str(ROOT / "examples" / "simulate" / "simulate.sh")],
            env=build_env(job),
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
    if proc.returncode != 0:
        return {"failed": True, "log": str(log)}
    result = measure(job.run_dir, job.regime.batch, job.arm.active_cap)
    result.update(regime=asdict(job.regime), arm=asdict(job.arm), tags=job.tags)
    metrics_path.write_text(json.dumps(result, indent=1))
    if not os.environ.get("KEEP_TELEMETRY"):
        shutil.rmtree(job.run_dir / "logs" / "instance", ignore_errors=True)
        shutil.rmtree(job.run_dir / "plots", ignore_errors=True)
    return result


def stream(path: Path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def measure(run: Path, batch: int, cap: int) -> dict:
    logs = run / "logs"
    steps = list(stream(logs / "train.jsonl"))
    starts = [s["start"] for s in steps]
    ends = [s["end"] for s in steps]
    n = len(steps)
    span = ends[-1]

    consumed_tokens = 0
    produced = 0
    stale: dict[int, int] = {}
    final: dict[int, dict] = {}
    for rec in stream(logs / "request.jsonl"):
        final[rec["rid"]] = rec
        if rec["event"] == "consume":
            consumed_tokens += rec["generated"]
            s = rec["staleness_true"]
            stale[s] = stale.get(s, 0) + rec["generated"]
    produced = sum(r["generated"] for r in final.values())

    idle = tot = wn = wkv = below = 0.0
    for path in sorted((logs / "instance").glob("*.jsonl")):
        prev = None
        for e in stream(path):
            if prev is not None:
                dt = e["t"] - prev["t"]
                if 0 < dt < 120:
                    tot += dt
                    wn += dt * (prev["num_running_reqs"] + prev["num_waiting_reqs"])
                    wkv += dt * prev["kv_cache_usage"]
                    if prev["num_running_reqs"] == 0:
                        idle += dt
                    if prev["num_running_reqs"] + prev["num_waiting_reqs"] < 0.75 * cap:
                        below += dt
            prev = e

    cycle = (starts[-1] - starts[1]) / (n - 2) if n >= 3 else float("nan")
    stalls = [starts[i] - ends[i - 1] for i in range(1, n)]
    return {
        "span": span,
        "fill": starts[0],
        "train_s": statistics.mean(s["dt"] for s in steps),
        "cycle_ss": cycle,
        "stall_ss": statistics.mean(stalls) if stalls else float("nan"),
        "consumed_tokens": consumed_tokens,
        "tokens_per_group": consumed_tokens / (n * batch),
        "produced_tokens": produced,
        "tput": consumed_tokens / span,
        "tput_ss": consumed_tokens / n / cycle if cycle == cycle else float("nan"),
        "max_staleness": max(stale) if stale else 0,
        "idle_frac": idle / tot if tot else float("nan"),
        "below_sat_frac": below / tot if tot else float("nan"),
        "mean_resident": wn / tot if tot else float("nan"),
        "mean_kv": wkv / tot if tot else float("nan"),
        "n_versions": n,
    }


# --- grids ---------------------------------------------------------------


def grid_pilot() -> list[Job]:
    r = Regime(versions=10)
    arms = [Arm("fifo", "count", 50), Arm("fifo", "load", 50)]
    for order in ("lpt", "spt"):
        for w in (0, 32, 96):
            arms.append(Arm(order, "load", 50, w))
    return [Job("pilot", r, a) for a in arms]


def grid_map1() -> list[Job]:
    """Training time against staleness, FIFO only, with the per-instance cap swept.

    No length knowledge yet: this locates the regimes in which the fleet idles or
    the trainer stalls at all, since a scheduler can only recover time that is
    being lost, and which cap is best is itself a scheduling choice.
    """
    jobs = []
    for train_s in (60, 150, 345, 700, 1400):
        for staleness in (0, 1, 2, 4):
            r = Regime(versions=10, train_s=train_s, staleness=staleness)
            for cap in (32, 64, 128):
                jobs.append(Job("map1", r, Arm("fifo", "count", cap)))
    return jobs


def grid_map2() -> list[Job]:
    """Supply tightness against length dispersion, with length-aware arms.

    A fleet saturates once enough requests are resident, so a scheduler has
    nothing to recover while the staleness window offers more work than the fleet
    can hold. This shrinks the window (batch size, staleness) until it does not,
    and varies the spread of lengths, since a straggler is only costly when
    little other work is left to run beside it. Training time scales with the
    batch so rollout and training stay comparable as the batch changes.
    """
    jobs = []
    for batch in (4, 8, 16, 32):
        for staleness in (0, 1, 2):
            for cv in (0.1, 0.5, 1.5):
                r = Regime(
                    versions=10,
                    batch=batch,
                    staleness=staleness,
                    cv=cv,
                    train_s=round(10.7 * batch, 1),
                )
                for arm in (
                    Arm("fifo", "count", 128),
                    Arm("fifo", "load", 128),
                    Arm("lpt", "load", 128, batch),
                    Arm("spt", "load", 128, batch),
                ):
                    jobs.append(Job("map2", r, arm))
    return jobs


def grid_map3() -> list[Job]:
    """What binds: relax the staleness bound, and find the best static cap.

    Order alone moved throughput by at most a few percent, so before designing
    anything smarter this asks which constraint the scheduler is working against.
    Raising staleness to 8 removes the supply limit (how much work the window
    lets the fleet hold), and the finer cap sweep shows how much concurrency
    alone is worth. The gap to the shipped configuration is the ceiling on what
    scheduling inside those constraints could recover.
    """
    jobs = []
    for cv in (0.5, 1.5):
        for cap in (40, 48, 56, 64, 72, 80, 96):
            r = Regime(versions=12, batch=32, staleness=2, cv=cv, train_s=150.0)
            jobs.append(Job("map3", r, Arm("fifo", "count", cap)))
    for batch in (8, 16, 32):
        for cv in (0.5, 1.5):
            for staleness in (1, 2, 4, 8):
                r = Regime(
                    versions=12,
                    batch=batch,
                    staleness=staleness,
                    cv=cv,
                    train_s=round(10.7 * batch, 1),
                )
                cap = 64 if batch == 32 else 128
                jobs.append(Job("map3", r, Arm("fifo", "count", cap)))
    return jobs


def grid_map4() -> list[Job]:
    """Length-aware ordering on top of the best static cap.

    Holds the cap at the throughput knee and asks what ordering and placement add.
    Reordering is confined to blocks of one or three training batches so the set of
    consumed groups stays comparable to FIFO's, and the tokens-per-group column says
    whether it did.
    """
    jobs = []
    for cv in (0.5, 1.5):
        for train_s in (60.0, 150.0, 345.0):
            for staleness in (1, 2, 4):
                r = Regime(
                    versions=12, batch=32, staleness=staleness, cv=cv, train_s=train_s
                )
                cap = 48
                jobs.append(Job("map4", r, Arm("fifo", "count", cap)))
                jobs.append(Job("map4", r, Arm("fifo", "load", cap)))
                for order in ("lpt", "spt"):
                    for w in (32, 96):
                        jobs.append(Job("map4", r, Arm(order, "load", cap, w)))
    return jobs


def grid_tight1() -> list[Job]:
    """Static caps where the staleness window offers little work.

    The default cap of 128 never binds when the window holds only a few hundred
    requests, so this asks whether holding concurrency down helps there as it does
    when work is plentiful.
    """
    jobs = []
    for batch in (8, 16):
        for staleness in (1, 2):
            for cv in (0.5, 1.5):
                r = Regime(
                    versions=12,
                    batch=batch,
                    staleness=staleness,
                    cv=cv,
                    train_s=round(10.7 * batch, 1),
                )
                for cap in (16, 24, 32, 48, 64):
                    jobs.append(Job("tight1", r, Arm("fifo", "count", cap)))
    return jobs


def grid_map5() -> list[Job]:
    """Where ordering pays most: tight staleness, balanced training, group-shaped tails.

    Sweeps tight staleness against training time and against how much of the length
    variance belongs to the prompt, since a group is only as fast as its slowest
    member.
    """
    jobs = []
    for rho in (0.0, 0.6, 0.9):
        for cv in (0.5, 1.0, 1.5):
            for train_s in (150.0, 345.0, 600.0):
                r = Regime(
                    versions=12,
                    batch=32,
                    staleness=1,
                    cv=cv,
                    rho=rho,
                    train_s=train_s,
                )
                jobs.append(Job("map5", r, Arm("fifo", "count", 48)))
                jobs.append(Job("map5", r, Arm("lpt", "load", 48, 32)))
                jobs.append(Job("map5", r, Arm("lpt", "load", 48, 96)))
    return jobs


def grid_map6() -> list[Job]:
    """Fleet size against batch size: how much supply the fleet is given.

    What matters is the number of requests the window offers per slot the fleet
    could fill, so batch and instance count are swept together. Training time is
    held at a fixed fraction of the rollout time the fleet would need per step, so
    the regime stays balanced as the fleet grows.
    """
    jobs = []
    for instances in (2, 4, 8):
        for batch in (8, 16, 32, 64):
            for staleness in (1, 2):
                r = Regime(
                    versions=12,
                    batch=batch,
                    staleness=staleness,
                    instances=instances,
                    cv=0.5,
                    train_s=round(26.0 * batch / instances, 1),
                )
                jobs.append(Job("map6", r, Arm("fifo", "count", 48)))
                jobs.append(Job("map6", r, Arm("lpt", "load", 48, batch)))
    return jobs


def grid_noise1() -> list[Job]:
    """How wrong may the length prediction be before longest-first stops paying.

    Longest-first with the oracle's answer multiplied by lognormal noise of
    increasing size, in regimes where ordering matters.
    """
    jobs = []
    for cv, train_s in ((0.5, 345.0), (0.5, 600.0), (1.0, 345.0)):
        r = Regime(versions=12, batch=32, staleness=1, cv=cv, rho=0.6, train_s=train_s)
        jobs.append(Job("noise1", r, Arm("fifo", "count", 48)))
        jobs.append(Job("noise1", r, Arm("lpt", "load", 48, 32)))
        for sigma in (0.1, 0.25, 0.5, 1.0, 2.0):
            for seed in (0, 1):
                jobs.append(Job("noise1", r, Arm("lpt", "load", 48, 32, sigma, seed)))
    return jobs


def grid_map5t() -> list[Job]:
    """The map 5 regimes again, recording the share of instance-time below saturation.

    The baseline's drain tail is observable without any length information, so if it
    predicts what longest-first recovers, the headroom can be read off a FIFO run.
    """
    jobs = []
    for job in grid_map5():
        if job.arm.window_groups in (0, 32):
            jobs.append(Job("map5t", job.regime, job.arm))
    return jobs


def grid_lptw() -> list[Job]:
    """How wide the reordering block should be, and how it interacts with the cap.

    A block of one training batch of groups matches the set that shares a
    consumption deadline. Smaller blocks reorder less; larger ones start the next
    batch's long groups before the current batch's short ones are done.
    """
    jobs = []
    for cv, train_s in ((0.5, 345.0), (1.0, 345.0)):
        r = Regime(versions=12, batch=32, staleness=1, cv=cv, rho=0.6, train_s=train_s)
        for w in (8, 16, 32, 48, 64):
            jobs.append(Job("lptw", r, Arm("lpt", "load", 48, w)))
        for cap in (40, 56, 64):
            jobs.append(Job("lptw", r, Arm("lpt", "load", cap, 32)))
            jobs.append(Job("lptw", r, Arm("fifo", "count", cap)))
    return jobs


def grid_taumax() -> list[Job]:
    """Slow training until it, rather than rollout, sets the pace.

    Fewer training GPUs means a longer step. Once it exceeds the rollout time per
    step the cycle is the training time exactly and no rollout schedule can help,
    so this finds where the gain from ordering runs out on the slow side.
    """
    jobs = []
    for staleness in (1, 2):
        for train_s in (60.0, 150.0, 345.0, 600.0, 800.0, 1000.0, 1400.0):
            r = Regime(versions=12, batch=32, staleness=staleness, cv=0.5, rho=0.6, train_s=train_s)
            jobs.append(Job("taumax", r, Arm("fifo", "count", 48)))
            jobs.append(Job("taumax", r, Arm("lpt", "load", 48, 32)))
    return jobs


def grid_real1() -> list[Job]:
    """The strongest synthetic regime again, on the recorded DAPO traces.

    Synthetic traces make dispersion a free variable; this checks the gain survives
    the shape real responses have, including the pile-up at the length cap.
    """
    jobs = []
    for name, cap in (("dapo_32k", 32768), ("psrl_aligned", 10240)):
        trace = str(ROOT / "outputs" / "workload" / name / "joint.jsonl")
        for staleness in (1, 2):
            for train_s in (345.0, 700.0, 1100.0):
                r = Regime(
                    versions=12,
                    batch=32,
                    staleness=staleness,
                    train_s=train_s,
                    cap=cap,
                    trace=trace,
                )
                jobs.append(Job("real1", r, Arm("fifo", "count", 48)))
                jobs.append(Job("real1", r, Arm("lpt", "load", 48, 32)))
                jobs.append(Job("real1", r, Arm("lpt", "load", 48, 32, 0.5, 0)))
    return jobs


def grid_real2() -> list[Job]:
    """Real traces at the training time where rollout and training balance.

    The DAPO 32k trace has a long rollout cycle, so training times chosen for the
    shorter synthetic traces sit far on the rollout-bound side. This moves training
    time to the balance point.
    """
    jobs = []
    trace = str(ROOT / "outputs" / "workload" / "dapo_32k" / "joint.jsonl")
    for staleness in (1, 2):
        for train_s in (1300.0, 1500.0, 1700.0):
            r = Regime(
                versions=12, batch=32, staleness=staleness, train_s=train_s, cap=32768, trace=trace
            )
            jobs.append(Job("real2", r, Arm("fifo", "count", 48)))
            jobs.append(Job("real2", r, Arm("lpt", "load", 48, 32)))
    for train_s in (500.0, 600.0):
        trace2 = str(ROOT / "outputs" / "workload" / "psrl_aligned" / "joint.jsonl")
        r = Regime(versions=12, batch=32, staleness=1, train_s=train_s, cap=10240, trace=trace2)
        jobs.append(Job("real2", r, Arm("fifo", "count", 48)))
        jobs.append(Job("real2", r, Arm("lpt", "load", 48, 32)))
    return jobs


def grid_real3() -> list[Job]:
    """The best cap for the DAPO 32k trace, whose contexts are longer than the synthetic ones."""
    jobs = []
    trace = str(ROOT / "outputs" / "workload" / "dapo_32k" / "joint.jsonl")
    r = Regime(versions=12, batch=32, staleness=2, train_s=345.0, cap=32768, trace=trace)
    for cap in (16, 24, 32, 40, 48, 64):
        jobs.append(Job("real3", r, Arm("fifo", "count", cap)))
    return jobs


def grid_dapolike() -> list[Job]:
    """A synthetic trace with the DAPO 32k trace's mean, spread and group correlation.

    The recorded trace shows almost no ordering gain where the synthetic one shows a
    lot. If a lognormal with matching moments reproduces the real trace's behaviour,
    the difference is scale (longer contexts, KV-bound concurrency); if it still
    shows the gain, the difference is the shape of the real distribution.
    """
    jobs = []
    for staleness in (1, 2):
        for train_s in (1100.0, 1500.0):
            r = Regime(
                versions=12,
                batch=32,
                staleness=staleness,
                train_s=train_s,
                mean=13000,
                cv=0.6,
                rho=0.6,
                cap=32768,
            )
            jobs.append(Job("dapolike", r, Arm("fifo", "count", 48)))
            jobs.append(Job("dapolike", r, Arm("lpt", "load", 48, 32)))
    return jobs


def grid_diag() -> list[Job]:
    """FIFO and longest-first on the recorded DAPO trace and its twin, for mechanism analysis.

    Run with KEEP_TELEMETRY=1 so the per-instance streams survive for inspection.
    """
    trace = str(ROOT / "outputs" / "workload" / "dapo_32k" / "joint.jsonl")
    real = Regime(versions=12, batch=32, staleness=1, train_s=1500.0, cap=32768, trace=trace)
    twin = Regime(
        versions=12, batch=32, staleness=1, train_s=1500.0, mean=13000, cv=0.6, rho=0.6, cap=32768
    )
    return [
        Job("diag", regime, arm)
        for regime in (real, twin)
        for arm in (Arm("fifo", "count", 48), Arm("lpt", "load", 48, 32))
    ]


def grid_tailcmp() -> list[Job]:
    """Drain-tail share on the real DAPO trace against its moment-matched synthetic twin."""
    jobs = []
    trace = str(ROOT / "outputs" / "workload" / "dapo_32k" / "joint.jsonl")
    real = Regime(versions=12, batch=32, staleness=1, train_s=1500.0, cap=32768, trace=trace)
    twin = Regime(
        versions=12, batch=32, staleness=1, train_s=1500.0, mean=13000, cv=0.6, rho=0.6, cap=32768
    )
    for regime in (real, twin):
        jobs.append(Job("tailcmp", regime, Arm("fifo", "count", 48)))
        jobs.append(Job("tailcmp", regime, Arm("lpt", "load", 48, 32)))
    return jobs


GRIDS = {
    "diag": grid_diag,
    "tailcmp": grid_tailcmp,
    "dapolike": grid_dapolike,
    "real3": grid_real3,
    "real2": grid_real2,
    "real1": grid_real1,
    "taumax": grid_taumax,
    "lptw": grid_lptw,
    "map5t": grid_map5t,
    "noise1": grid_noise1,
    "map6": grid_map6,
    "map5": grid_map5,
    "tight1": grid_tight1,
    "pilot": grid_pilot,
    "map1": grid_map1,
    "map2": grid_map2,
    "map3": grid_map3,
    "map4": grid_map4,
}


def collect(grid: str) -> list[dict]:
    rows = []
    for path in sorted((OUT_ROOT / grid).glob("*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def report(grid: str) -> None:
    rows = collect(grid)
    by_regime: dict[str, list[dict]] = {}
    for row in rows:
        r = Regime(**row["regime"])
        by_regime.setdefault(r.key(), []).append(row)
    for key, group in by_regime.items():
        print(f"\n{key}")
        base = next(
            (g for g in group if g["arm"]["order"] == "fifo" and g["arm"]["balance"] == "count"),
            group[0],
        )
        print(
            f"  {'arm':<26}{'span':>7}{'cycle':>7}{'stall':>7}{'tput':>7}{'vs base':>9}"
            f"{'tok/grp':>9}{'idle%':>7}{'N':>6}{'kv':>6}{'maxS':>5}{'waste%':>8}"
        )
        for g in sorted(group, key=lambda g: -g["tput"]):
            a = Arm(**g["arm"])
            waste = 100 * (1 - g["consumed_tokens"] / g["produced_tokens"])
            print(
                f"  {a.key():<26}{g['span']:>7.0f}{g['cycle_ss']:>7.0f}{g['stall_ss']:>7.0f}"
                f"{g['tput']:>7.0f}{g['tput'] / base['tput']:>8.3f}x{g['tokens_per_group']:>9.0f}"
                f"{100 * g['idle_frac']:>7.2f}{g['mean_resident']:>6.1f}{g['mean_kv']:>6.2f}"
                f"{g['max_staleness']:>5}{waste:>8.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("grid", choices=sorted(GRIDS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "run":
        jobs = GRIDS[args.grid]()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for job, result in zip(jobs, pool.map(run_job, jobs)):
                status = "FAILED" if result.get("failed") else f"cycle {result['cycle_ss']:.0f}s"
                print(f"{job.regime.key()}__{job.arm.key()}: {status}", flush=True)
    report(args.grid)


if __name__ == "__main__":
    main()
